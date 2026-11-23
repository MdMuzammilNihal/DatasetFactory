"""QLoRA fine-tuning on PubMedQA (Module 10d).

Settings: LoRA r 16, alpha 32, dropout 0.05 on ``query_key_value``,
lr 2e-4, 3 epochs, batch 4 x accumulation 4. Other choices:
- 4-bit NF4 base (double quantization, fp16 compute) via bitsandbytes;
  ``quant="none"`` is plain LoRA on an fp32 base with fp16 autocast.
- The training text is exactly the evaluation prompt (``evaluate.pubmedqa_prompt``)
  followed by `` yes`` / `` no`` / `` maybe``; the loss covers only the answer
  tokens, so the adapter learns the answer, not the abstracts.
- Linear warmup (5% of steps) then linear decay to 0, AdamW, clip 1.0,
  fp16 + GradScaler (bf16 measured harmful for Pythia, see 10b).
- Training items: ``pubmedqa_train.jsonl`` (500). The 500 test items are never
  seen; no model selection is done on them (fixed number of epochs).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .evalsets import read_jsonl
from .evaluate import pubmedqa_prompt
from .tok import DEFAULT_TOKENIZER, eos_id, load_tokenizer


@dataclass
class QLoRAConfig:
    model: str                            # HF name or a pretraining run's model folder
    out: str
    train_file: str = "data/eval/pubmedqa_train.jsonl"
    quant: str = "4bit"                   # 4bit | none (plain LoRA)
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query_key_value",)
    lr: float = 2e-4
    epochs: int = 3
    micro_batch: int = 4
    accum: int = 4
    warmup_frac: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_len: int = 2048                   # same as the PubMedQA evaluation
    seed: int = 0
    gradient_checkpointing: bool = False
    device: str = "auto"
    tokenizer: str = DEFAULT_TOKENIZER


def build_examples(rows: list[dict[str, Any]], tok, max_len: int) -> list[tuple[list[int], int]]:
    """(token ids, index of the first answer token) per row, as the evaluation scores them."""
    eos = eos_id(tok)
    out = []
    for row, enc in zip(rows, tok.encode_batch([pubmedqa_prompt(r) for r in rows],
                                               add_special_tokens=False)):
        ans = tok.encode(" " + row["label"], add_special_tokens=False).ids
        room = max_len - len(ans) - 1
        ids = [eos] + enc.ids[-room:]      # keep the end (question) if too long
        out.append((ids + ans, len(ids)))
    return out


def load_base(name: str, quant: str = "4bit", device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    if quant == "none":
        from .pretrain import load_model
        return load_model(name)
    if quant != "4bit":
        raise ValueError(f"unknown quant {quant!r}")
    if device != "cuda":
        raise RuntimeError("4-bit loading needs a CUDA GPU; use quant='none'")
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0},
                                                dtype=torch.float32, attn_implementation="sdpa")


def load_adapted(run_dir: str | Path, device: str = "auto"):
    """Base model (quantized as in training) + the trained adapter, for evaluation."""
    import torch
    from peft import PeftModel
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "qlora_config.json").read_text(encoding="utf-8"))
    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    base = load_base(cfg["model"], cfg["quant"], device)
    return PeftModel.from_pretrained(base, run_dir / "adapter").eval()


def place(model, device: str):
    """Move to ``device`` unless the weights were placed at load time (4-bit)."""
    if getattr(model, "hf_device_map", None) is None:
        model.to(device)
    return model


def _collate(batch: list[tuple[list[int], int]]):
    import torch
    L = max(len(s) for s, _ in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    att = torch.zeros(len(batch), L, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (s, first) in enumerate(batch):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        lab[i, first:len(s)] = ids[i, first:len(s)]
    return ids, att, lab


def train(cfg: QLoRAConfig, model=None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Fine-tune; writes ``adapter/``, ``log.jsonl``, ``qlora_config.json``, ``summary.json``."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from .pretrain import _precision, lr_at

    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(cfg.tokenizer)
    rows = read_jsonl(Path(cfg.train_file))
    examples = build_examples(rows, tok, cfg.max_len)

    torch.manual_seed(cfg.seed)
    if model is None:
        log(f"loading {cfg.model} ({cfg.quant})")
        model = load_base(cfg.model, cfg.quant, device)
    if cfg.quant == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, bias="none",
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"))
    place(model, device).train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    dtype, use_scaler = _precision("auto", device)

    per_step = cfg.micro_batch * cfg.accum
    steps_per_epoch = math.ceil(len(examples) / per_step)
    total = steps_per_epoch * cfg.epochs
    warmup = int(round(cfg.warmup_frac * total))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)

    (out / "qlora_config.json").write_text(json.dumps(
        {**asdict(cfg), "examples": len(examples), "steps": total, "warmup_steps": warmup,
         "trainable_params": trainable, "total_params": total_params, "device": device,
         "dtype": str(dtype),
         "labels": {a: sum(r["label"] == a for r in rows) for a in ("yes", "no", "maybe")}},
        indent=2), encoding="utf-8")
    log(f"{len(examples)} examples, {total} steps, {trainable:,} trainable of {total_params:,}")

    rng = np.random.default_rng(cfg.seed)
    logf = open(out / "log.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    step = 0
    rec: dict[str, Any] = {}
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(examples))
        for s in range(steps_per_epoch):
            group = order[s * per_step:(s + 1) * per_step]
            n_tgt = sum(len(examples[i][0]) - examples[i][1] for i in group)
            lr = lr_at(step, total, cfg.lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = 0.0
            for m in range(0, len(group), cfg.micro_batch):
                ids, att, lab = (t.to(device) for t in _collate(
                    [examples[i] for i in group[m:m + cfg.micro_batch]]))
                with torch.autocast(device_type=device, dtype=dtype, enabled=dtype is not None):
                    logits = model(input_ids=ids, attention_mask=att).logits
                # summed NLL over answer tokens, normalised by the step's answer-token count
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="sum") / n_tgt
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": epoch + 1, "loss": round(loss_sum, 5), "lr": lr,
                   "grad_norm": round(gnorm, 4)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if not math.isfinite(loss_sum):
                logf.close()
                raise FloatingPointError(f"loss is {loss_sum} at step {step}")
        ep = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        ep_loss = float(np.mean([r["loss"] for r in ep if r["epoch"] == epoch + 1]))
        log(f"epoch {epoch + 1}/{cfg.epochs}: mean answer loss {ep_loss:.4f} ({time.time() - t0:.0f}s)")
    logf.close()
    model.save_pretrained(out / "adapter")
    summary = {"steps": total, "seconds": round(time.time() - t0, 1), "final_loss": rec.get("loss"),
               "trainable_params": trainable,
               "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
               if device == "cuda" else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
