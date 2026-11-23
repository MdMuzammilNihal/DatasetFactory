"""Continued pretraining of a released Pythia checkpoint on a packed corpus.

Plain PyTorch loop, so it runs the same locally and on Colab:
- the corpus (``train.bin``) is cut into non-overlapping ``seq_len`` windows;
  one epoch visits each window once, in an order fixed by the seed, so two
  corpora with the same budget get the same number of optimizer steps;
- AdamW, linear warmup (a fraction of the steps) then linear decay to zero,
  gradient clipping, bf16 autocast (fp16 + loss scaling where bf16 is missing);
- loss on ``val.bin`` every ``eval_every`` steps; ``log.jsonl`` per step;
- a resumable ``checkpoint.pt`` every ``save_every`` steps (Colab sessions
  die), the final model in Hugging Face format under ``model/``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .corpus import load_tokens


@dataclass
class TrainConfig:
    corpus: str                          # folder with train.bin / val.bin / meta.json
    out: str                             # run folder
    model: str = "EleutherAI/pythia-160m"
    revision: str = "main"               # released final checkpoint (step 143000)
    seed: int = 0
    seq_len: int = 1024
    global_batch: int = 32               # sequences per optimizer step
    micro_batch: int = 4                 # 8 needs 9.6 GB for Pythia-160M (spills on an 8 GB GPU)
    lr: float = 6e-5                     # 2e-4 was measured worse here
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    precision: str = "auto"              # auto (= fp16 on CUDA) | bf16 | fp16 | fp32
    eval_every: int = 50
    val_windows: int = 64
    save_every: int = 200
    max_steps: int = 0                   # 0 = one full epoch
    stop_at: int = 0                     # save a checkpoint and stop here (time-limited sessions)
    gradient_checkpointing: bool = False
    device: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def accum(self) -> int:
        if self.global_batch % self.micro_batch:
            raise ValueError("global_batch must be a multiple of micro_batch")
        return self.global_batch // self.micro_batch


def windows(n_tokens: int, seq_len: int) -> int:
    return n_tokens // seq_len


def epoch_order(n_windows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n_windows)


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Learning rate for optimizer step ``step`` (0-based)."""
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    if total <= warmup:
        return peak
    return peak * max(0.0, (total - step) / (total - warmup))


def _batch(tokens: np.ndarray, idx: np.ndarray, seq_len: int):
    import torch
    rows = np.stack([np.asarray(tokens[i * seq_len:(i + 1) * seq_len], dtype=np.int64) for i in idx])
    return torch.from_numpy(rows)


def _precision(name: str, device: str):
    import torch
    if device != "cuda" or name == "fp32":
        return None, False
    # Not bf16 by default: Pythia was trained in fp16 and bf16 autocast raises the
    # *untrained* model's loss a lot (measured: 70M 3.74 -> 5.44, 160M 3.32 -> 3.83;
    # fp16 3.77 / 3.33). fp16 needs a GradScaler.
    if name == "bf16":
        return torch.bfloat16, False
    return torch.float16, True


def evaluate_loss(model, tokens: np.ndarray, seq_len: int, n_windows: int, micro_batch: int,
                  device: str, dtype) -> float:
    """Mean next-token loss over the first ``n_windows`` windows (token-weighted)."""
    import torch
    n = min(n_windows, windows(len(tokens), seq_len))
    if n == 0:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for s in range(0, n, micro_batch):
            x = _batch(tokens, np.arange(s, min(n, s + micro_batch)), seq_len).to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                out = model(input_ids=x, labels=x)
            k = x.shape[0] * (seq_len - 1)
            total += float(out.loss) * k
            count += k
    model.train()
    return total / count


def load_model(name: str, revision: str = "main"):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype="float32",
                                                attn_implementation="sdpa")


def train(cfg: TrainConfig, model=None, log: Callable[[str], None] = print,
          resume: bool = False) -> dict[str, Any]:
    import torch

    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(cfg.corpus)
    meta = json.loads((corpus / "meta.json").read_text(encoding="utf-8"))
    train_tokens = load_tokens(corpus / "train.bin")
    val_path = corpus / "val.bin"
    val_tokens = load_tokens(val_path) if val_path.exists() and val_path.stat().st_size else np.empty(0, np.uint16)

    n_win = windows(len(train_tokens), cfg.seq_len)
    total_steps = n_win // cfg.global_batch
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    if total_steps < 1:
        raise ValueError(f"corpus too small: {n_win} windows < one batch of {cfg.global_batch}")
    warmup = int(round(cfg.warmup_frac * total_steps))
    order = epoch_order(n_win, cfg.seed)

    torch.manual_seed(cfg.seed)
    if model is None:
        log(f"loading {cfg.model}@{cfg.revision}")
        model = load_model(cfg.model, cfg.revision)
    model.to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    dtype, use_scaler = _precision(cfg.precision, device)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=cfg.betas, fused=device == "cuda")
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)

    step = 0
    ckpt = out / "checkpoint.pt"
    if resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        log(f"resumed at step {step}")
    elif not resume:
        (out / "log.jsonl").write_text("", encoding="utf-8")

    (out / "train_config.json").write_text(json.dumps(
        {**asdict(cfg), "total_steps": total_steps, "warmup_steps": warmup, "windows": n_win,
         "tokens_per_step": cfg.global_batch * cfg.seq_len, "device": device,
         "dtype": str(dtype), "corpus_meta": {k: meta.get(k) for k in
                                               ("name", "budget", "train_tokens", "sha256", "seed")}},
        indent=2), encoding="utf-8")

    def save_checkpoint() -> None:
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "step": step}, tmp)
        tmp.replace(ckpt)

    logf = open(out / "log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    t_last, tok_last = t_start, step * cfg.global_batch * cfg.seq_len
    rec: dict[str, Any] = {}
    if step == 0:
        v = evaluate_loss(model, val_tokens, cfg.seq_len, cfg.val_windows, cfg.micro_batch, device, dtype)
        logf.write(json.dumps({"step": 0, "val_loss": v}) + "\n")
        log(f"step 0 val_loss {v:.4f}")
    while step < total_steps:
        lr = lr_at(step, total_steps, cfg.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        base = step * cfg.global_batch
        for m in range(cfg.accum):
            idx = order[base + m * cfg.micro_batch: base + (m + 1) * cfg.micro_batch]
            x = _batch(train_tokens, idx, cfg.seq_len).to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                loss = model(input_ids=x, labels=x).loss / cfg.accum
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "loss": round(loss_sum, 5), "lr": lr,
                               "grad_norm": round(gnorm, 4),
                               "tokens": step * cfg.global_batch * cfg.seq_len}
        if not math.isfinite(loss_sum):
            logf.write(json.dumps(rec) + "\n")
            logf.close()
            raise FloatingPointError(f"loss is {loss_sum} at step {step}")
        if step % cfg.eval_every == 0 or step == total_steps:
            rec["val_loss"] = evaluate_loss(model, val_tokens, cfg.seq_len, cfg.val_windows,
                                            cfg.micro_batch, device, dtype)
            now = time.time()
            rec["tok_per_s"] = round((rec["tokens"] - tok_last) / max(now - t_last, 1e-9))
            t_last, tok_last = now, rec["tokens"]
            log(f"step {step}/{total_steps} loss {loss_sum:.4f} val {rec['val_loss']:.4f} "
                f"lr {lr:.2e} {rec['tok_per_s']:,} tok/s")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if cfg.save_every and step % cfg.save_every == 0 and step < total_steps:
            save_checkpoint()
        if cfg.stop_at and step >= cfg.stop_at and step < total_steps:
            save_checkpoint()
            logf.close()
            log(f"stopped at step {step}; continue with resume")
            return {"stopped_at": step, "steps": total_steps}
    logf.close()

    model.save_pretrained(out / "model", safe_serialization=True)
    summary = {"steps": total_steps, "tokens_seen": total_steps * cfg.global_batch * cfg.seq_len,
               "corpus_tokens": int(len(train_tokens)), "seconds": round(time.time() - t_start, 1),
               "final_loss": rec.get("loss"), "final_val_loss": rec.get("val_loss"),
               "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
               if device == "cuda" else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if ckpt.exists():
        ckpt.unlink()          # the final model replaces it
    return summary
