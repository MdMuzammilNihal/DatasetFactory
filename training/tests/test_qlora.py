"""Module 10d: QLoRA data format, training loop and adapter round trip (tiny model, CPU, no 4-bit)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from dftrain import evaluate as ev  # noqa: E402
from dftrain import qlora  # noqa: E402
from dftrain.evalsets import write_jsonl  # noqa: E402
from dftrain.tok import eos_id, load_tokenizer  # noqa: E402


def tiny_dir(tmp):
    cfg = transformers.GPTNeoXConfig(vocab_size=50304, hidden_size=32, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=64,
                                     max_position_embeddings=2048)
    torch.manual_seed(0)
    d = tmp / "base"
    transformers.GPTNeoXForCausalLM(cfg).save_pretrained(d)
    return d


def rows(n=24):
    # learnable: the answer is the last word of the context
    out = []
    for i in range(n):
        a = ("yes", "no", "maybe")[i % 3]
        out.append({"pubid": str(i), "question": f"Question {i}?", "context": f"Text {i}. {a}",
                    "label": a})
    return out


def test_examples_match_evaluation_prompt_and_mask():
    tok = load_tokenizer()
    r = rows(3)
    ex = qlora.build_examples(r, tok, 2048)
    for row, (ids, first) in zip(r, ex):
        assert ids[0] == eos_id(tok)
        assert tok.decode(ids[1:first]) == ev.pubmedqa_prompt(row)
        assert tok.decode(ids[first:]) == " " + row["label"]
    ids, att, lab = qlora._collate(ex)
    assert (lab != -100).sum() == sum(len(s) - f for s, f in ex)
    # long prompts keep their end (the question) and the answer
    long = qlora.build_examples([{**r[0], "context": "word " * 3000}], tok, 64)[0]
    assert len(long[0]) == 64 and tok.decode(long[0][long[1]:]) == " yes"


def test_lora_trains_saves_and_reloads(tmp_path):
    base = tiny_dir(tmp_path)
    write_jsonl(tmp_path / "train.jsonl", rows())
    cfg = qlora.QLoRAConfig(model=str(base), out=str(tmp_path / "run"),
                            train_file=str(tmp_path / "train.jsonl"), quant="none", device="cpu",
                            lr=5e-3, epochs=6, micro_batch=4, accum=2, max_len=128)
    s = qlora.train(cfg, log=lambda *_: None)
    assert s["steps"] == 6 * 3                     # 24 examples / 8 per step
    log = [l for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    losses = [__import__("json").loads(l)["loss"] for l in log]
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
    assert (tmp_path / "run" / "adapter" / "adapter_config.json").exists()
    # the saved adapter changes the base model's answers; reloading gives the same scores
    tok = load_tokenizer()
    m1 = qlora.load_adapted(tmp_path / "run", device="cpu")
    m2 = qlora.load_adapted(tmp_path / "run", device="cpu")
    base_m = transformers.AutoModelForCausalLM.from_pretrained(base).eval()
    prompts = [ev.pubmedqa_prompt(r) for r in rows(6)]
    a = ev._answer_scores(m1, tok, prompts, "cpu", None, 128, 4)
    b = ev._answer_scores(m2, tok, prompts, "cpu", None, 128, 4)
    c = ev._answer_scores(base_m, tok, prompts, "cpu", None, 128, 4)
    assert np.allclose(a, b) and not np.allclose(a, c)
