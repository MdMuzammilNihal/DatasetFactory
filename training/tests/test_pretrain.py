"""Module 10b: schedule, step accounting and exact resume, on a tiny random model (CPU)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from dftrain.pretrain import TrainConfig, epoch_order, lr_at, train  # noqa: E402


def tiny_model():
    cfg = transformers.GPTNeoXConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                                     num_attention_heads=2, intermediate_size=64,
                                     max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.GPTNeoXForCausalLM(cfg)


def tiny_corpus(tmp, n_tokens=16 * 40 + 7):
    d = tmp / "corpus"
    d.mkdir()
    rng = np.random.default_rng(0)
    # a learnable pattern: token t+1 = (t * 3 + 1) % 128
    seq = [int(rng.integers(128))]
    for _ in range(n_tokens - 1):
        seq.append((seq[-1] * 3 + 1) % 128)
    np.asarray(seq, dtype=np.uint16).tofile(d / "train.bin")
    np.asarray(seq[:200], dtype=np.uint16).tofile(d / "val.bin")
    (d / "meta.json").write_text(json.dumps({"name": "tiny", "train_tokens": n_tokens}))
    return d


def cfg(tmp, corpus, **kw):
    base = dict(corpus=str(corpus), out=str(tmp / "run"), seq_len=16, global_batch=4, micro_batch=2,
                lr=3e-3, eval_every=5, val_windows=4, save_every=0, device="cpu", warmup_frac=0.2)
    base.update(kw)
    return TrainConfig(**base)


def test_lr_schedule():
    assert lr_at(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert lr_at(9, 100, 1.0, 10) == pytest.approx(1.0)
    assert lr_at(10, 100, 1.0, 10) == pytest.approx(1.0)
    assert lr_at(99, 100, 1.0, 10) == pytest.approx(1 / 90)
    assert lr_at(5, 10, 1.0, 0) == pytest.approx(0.5)


def test_epoch_order_is_a_seeded_permutation():
    a = epoch_order(50, 1)
    assert sorted(a) == list(range(50))
    assert np.array_equal(a, epoch_order(50, 1)) and not np.array_equal(a, epoch_order(50, 2))


def test_train_one_epoch_learns_and_writes_outputs(tmp_path):
    corpus = tiny_corpus(tmp_path)
    s = train(cfg(tmp_path, corpus), model=tiny_model(), log=lambda *_: None)
    assert s["steps"] == 40 // 4                      # 40 windows of 16, 4 per step; remainder unused
    log = [json.loads(line) for line in open(tmp_path / "run" / "log.jsonl")]
    vals = [r["val_loss"] for r in log if "val_loss" in r]
    assert vals[-1] < vals[0] - 0.5                    # it learned the pattern
    assert (tmp_path / "run" / "model" / "config.json").exists()
    assert not (tmp_path / "run" / "checkpoint.pt").exists()
    tc = json.loads((tmp_path / "run" / "train_config.json").read_text())
    assert tc["total_steps"] == 10 and tc["warmup_steps"] == 2


def test_stop_and_resume_equals_uninterrupted(tmp_path):
    corpus = tiny_corpus(tmp_path)
    full = tiny_model()
    train(cfg(tmp_path, corpus, out=str(tmp_path / "a")), model=full, log=lambda *_: None)

    part = tiny_model()
    r = train(cfg(tmp_path, corpus, out=str(tmp_path / "b"), stop_at=4), model=part, log=lambda *_: None)
    assert r == {"stopped_at": 4, "steps": 10}
    fresh = tiny_model()                                # a new process would start from the base model
    train(cfg(tmp_path, corpus, out=str(tmp_path / "b")), model=fresh, log=lambda *_: None, resume=True)
    for (n, p1), (_, p2) in zip(full.state_dict().items(), fresh.state_dict().items()):
        assert torch.allclose(p1, p2, atol=1e-6), n
    la = [json.loads(x)["loss"] for x in open(tmp_path / "a" / "log.jsonl") if '"loss"' in x]
    lb = [json.loads(x)["loss"] for x in open(tmp_path / "b" / "log.jsonl") if '"loss"' in x]
    assert lb == pytest.approx(la, abs=1e-5)
