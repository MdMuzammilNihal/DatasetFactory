"""Module 10c: windowing, NLL bookkeeping, PubMedQA scoring and comparison (tiny model, CPU)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from dftrain import evaluate as ev  # noqa: E402
from dftrain.evalsets import write_jsonl  # noqa: E402
from dftrain.tok import eos_id, load_tokenizer  # noqa: E402


def tiny():
    cfg = transformers.GPTNeoXConfig(vocab_size=50304, hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32,
                                     max_position_embeddings=2048)
    torch.manual_seed(0)
    return transformers.GPTNeoXForCausalLM(cfg).eval()


def test_windows_score_every_token_once():
    for n in (2, 5, 10, 11, 37):
        w = ev.windows_for(n, 8, 3)
        scored = [t for s, e, f in w for t in range(f, e)]
        assert scored == list(range(1, n))
        assert all(e - s <= 8 for s, e, f in w)


def _direct_nll(model, ids):
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    lp = torch.log_softmax(logits[:-1].float(), -1)
    return -float(lp.gather(-1, torch.tensor(ids[1:]).unsqueeze(-1)).sum())


def test_perplexity_matches_direct_computation_and_padding_is_ignored():
    m, tok = tiny(), load_tokenizer()
    texts = ["the cat sat on the mat", "a much longer sentence about clinical trials and outcomes"]
    r = ev.perplexity(m, texts, tok, "cpu", None, max_len=1024, stride=512, batch=2)
    eos = eos_id(tok)
    ids = [[eos] + tok.encode(t, add_special_tokens=False).ids for t in texts]
    direct = [_direct_nll(m, x) for x in ids]
    assert r["doc_nll"] == pytest.approx(direct, rel=1e-4, abs=1e-3)
    assert r["doc_tokens"] == [len(x) - 1 for x in ids]
    assert r["perplexity"] == pytest.approx(math.exp(sum(direct) / sum(len(x) - 1 for x in ids)), rel=1e-4)


def test_sliding_window_equals_full_context_when_window_covers_doc():
    m, tok = tiny(), load_tokenizer()
    text = " ".join(f"word{i}" for i in range(60))
    full = ev.perplexity(m, [text], tok, "cpu", None, max_len=1024, stride=512)
    slid = ev.perplexity(m, [text], tok, "cpu", None, max_len=32, stride=8)
    wide = ev.perplexity(m, [text], tok, "cpu", None, max_len=512, stride=8)
    assert full["doc_tokens"] == slid["doc_tokens"] == wide["doc_tokens"]
    assert wide["doc_nll"] == pytest.approx(full["doc_nll"], rel=1e-5)   # one window: stride unused


def _qa(n):
    labels = ["yes", "no", "maybe"]
    return [{"pubid": str(i), "context": f"Study {i} measured outcome.", "question": f"Does {i} work?",
             "long_answer": "x", "label": labels[i % 3]} for i in range(n)]


def test_pubmedqa_scoring_shapes_and_baseline():
    m, tok = tiny(), load_tokenizer()
    r = ev.pubmedqa(m, tok, _qa(9), _qa(30), "cpu", None, shots=2, seed=0, max_len=256, batch=4)
    assert r["n"] == 9 and r["majority_accuracy"] == pytest.approx(3 / 9)
    for mode in ("raw", "calibrated"):
        assert sum(r[mode]["predicted"].values()) == 9
        assert 0 <= r[mode]["accuracy"] <= 1 and len(r[mode]["correct"]) == 9
    assert ev.few_shot_prefix(_qa(30), 3, 0).count("Answer:") == 3
    assert ev.few_shot_prefix(_qa(30), 3, 0) == ev.few_shot_prefix(_qa(30), 3, 0)


def test_answer_scores_match_direct_loglik():
    m, tok = tiny(), load_tokenizer()
    prompt = "Context: c\nQuestion: q\nAnswer:"
    s = ev._answer_scores(m, tok, [prompt], "cpu", None, 2048, 3)[0]
    eos = eos_id(tok)
    p = [eos] + tok.encode(prompt, add_special_tokens=False).ids
    for k, a in enumerate(ev.ANSWERS):
        aid = tok.encode(" " + a, add_special_tokens=False).ids
        with torch.no_grad():
            lp = torch.log_softmax(m(input_ids=torch.tensor([p + aid])).logits[0].float(), -1)
        direct = sum(float(lp[len(p) - 1 + j, t]) for j, t in enumerate(aid))
        assert s[k] == pytest.approx(direct, abs=1e-3)


def test_macro_f1_and_compare(tmp_path):
    assert ev.macro_f1(["yes", "no", "maybe"], ["yes", "no", "maybe"]) == 1.0
    assert ev.macro_f1(["yes", "no"], ["yes", "yes"]) == pytest.approx((2 / 3) / 3)
    a = {"medical": {"perplexity": 10.0, "doc_nll": [10.0, 20.0], "doc_tokens": [5, 10]},
         "pubmedqa": {"0shot": {"raw": {"correct": [1, 0, 0, 1]}, "calibrated": {"correct": [1, 1, 0, 0]}}}}
    b = {"medical": {"perplexity": 8.0, "doc_nll": [8.0, 16.0], "doc_tokens": [5, 10]},
         "pubmedqa": {"0shot": {"raw": {"correct": [1, 1, 0, 1]}, "calibrated": {"correct": [1, 1, 0, 0]}}}}
    c = ev.compare(a, b)
    assert c["medical"]["ratio_b_over_a"] == pytest.approx(0.8)
    lo, hi = c["medical"]["ratio_ci95"]
    assert lo <= hi < 1
    assert c["pubmedqa_0shot_raw"]["diff"] == pytest.approx(0.25)
    assert c["pubmedqa_0shot_calibrated"]["diff"] == 0


def test_evaluate_driver_writes_results(tmp_path):
    d = tmp_path / "eval"
    d.mkdir()
    write_jsonl(d / "wikitext.jsonl", [{"title": "t", "text": "some wiki text here"}])
    write_jsonl(d / "medical.jsonl", [{"text": "patients were randomized"}])
    write_jsonl(d / "pubmedqa_test.jsonl", _qa(6))
    write_jsonl(d / "pubmedqa_train.jsonl", _qa(12))
    out = tmp_path / "r.json"
    r = ev.evaluate("tiny", d, out, shots=(0,), model=tiny(), device="cpu", log=lambda *_: None)
    assert out.exists() and set(ev.TASKS) <= set(r)
    assert np.isfinite(r["wikitext"]["perplexity"])
    assert r["clinical"]["documents"] == 6
    q = r["pubmedqa"]["0shot"]
    assert len(q["loglik"]) == 6 and len(q["loglik"][0]) == 3 and q["gold"] == [x["label"] for x in _qa(6)]
    assert 0 <= q["yes_no_auroc"]["auroc"] <= 1


def test_yes_no_auroc_matches_sklearn_and_ignores_constant_bias():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    gold = [["yes", "no", "maybe"][i % 3] for i in range(90)]
    ll = rng.normal(size=(90, 3))
    ll[[g == "yes" for g in gold], 0] += 0.7               # some real signal
    r = ev.yes_no_auroc(gold, ll)
    keep = [i for i, g in enumerate(gold) if g != "maybe"]
    want = roc_auc_score([gold[i] == "yes" for i in keep], ll[keep, 0] - ll[keep, 1])
    assert r["auroc"] == pytest.approx(want) and r["n_yes"] == r["n_no"] == 30
    assert r["ci95"][0] < r["auroc"] < r["ci95"][1]
    biased = ll.copy()
    biased[:, 1] += 5.0                                    # always prefers " no"
    assert ev.yes_no_auroc(gold, biased)["auroc"] == pytest.approx(r["auroc"])
    a = {"pubmedqa": {"0shot": {"n": 90, "loglik": ll.tolist(), "gold": gold,
                                "raw": {"correct": [1] * 90}, "calibrated": {"correct": [1] * 90}}}}
    b = {"pubmedqa": {"0shot": {**a["pubmedqa"]["0shot"], "loglik": biased.tolist()}}}
    c = ev.compare(a, b)["pubmedqa_0shot_auroc"]
    assert c["diff"] == pytest.approx(0) and c["diff_ci95"] == pytest.approx([0, 0], abs=1e-12)


def test_code_by_language_matches_pooled_and_leaves_overall_unchanged(tmp_path):
    d = tmp_path / "eval"
    d.mkdir()
    rows = [{"language": l, "text": f"def f{i}(x): return x + {i}"}
            for i, l in enumerate(["python", "go", "python", "go", "java"])]
    write_jsonl(d / "code.jsonl", rows)
    out = tmp_path / "r.json"
    r = ev.evaluate("tiny", d, out, tasks=["code"], model=tiny(), device="cpu", log=lambda *_: None)
    c = r["code"]
    bl = c["by_language"]
    assert set(bl) == {"python", "go", "java"} and bl["python"]["documents"] == 2
    # pooled over languages = the overall perplexity
    tot_nll = sum(math.log(v["perplexity"]) * v["tokens"] for v in bl.values())
    assert math.exp(tot_nll / sum(v["tokens"] for v in bl.values())) == pytest.approx(c["perplexity"])
    # post-hoc path on a result without the key gives the same numbers
    old = {k: v for k, v in c.items() if k != "by_language"}
    out.write_text(__import__("json").dumps({"code": old}), encoding="utf-8")
    again = ev.add_code_languages(out, d)
    assert again["python"]["perplexity"] == pytest.approx(bl["python"]["perplexity"])
