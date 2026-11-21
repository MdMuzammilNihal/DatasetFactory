"""Module 10d: experiment runner end to end on a tiny model (CPU) and its summary."""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from dftrain import experiment as ex  # noqa: E402
from dftrain.evalsets import write_jsonl  # noqa: E402


def setup(tmp):
    cfg = transformers.GPTNeoXConfig(vocab_size=50304, hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32,
                                     max_position_embeddings=2048)
    torch.manual_seed(0)
    transformers.GPTNeoXForCausalLM(cfg).save_pretrained(tmp / "base")
    ev = tmp / "eval"
    ev.mkdir()
    write_jsonl(ev / "wikitext.jsonl", [{"title": "t", "text": "The cat sat on the mat. " * 20}])
    write_jsonl(ev / "medical.jsonl", [{"text": "Patients were treated with aspirin. " * 10}])
    qa = [{"pubid": str(i), "question": f"Q{i}?", "context": f"C{i}.", "long_answer": "L.",
           "label": ("yes", "no", "maybe")[i % 3]} for i in range(9)]
    write_jsonl(ev / "pubmedqa_test.jsonl", qa)
    write_jsonl(ev / "pubmedqa_train.jsonl", qa)
    dev = tmp / "dev"
    dev.mkdir()
    write_jsonl(dev / "wikitext.jsonl", [{"title": "d", "text": "A dog ran in the park. " * 20}])
    write_jsonl(dev / "medical.jsonl", [{"text": "Patients received placebo. " * 10}])
    write_jsonl(dev / "clinical.jsonl", [{"text": "RESULTS: blood pressure fell. " * 10}])
    for name, word in (("a", "alpha"), ("b", "beta")):
        write_jsonl(tmp / f"{name}.jsonl", [{"id": f"{name}{i}", "text": f"{word} {i} " * 30}
                                            for i in range(200)])
    plan = tmp / "plan.toml"
    p = lambda x: str(x).replace("\\", "/")   # noqa: E731
    plan.write_text(f"""
name = "t"
model = "{p(tmp / 'base')}"
budget = "4k"
seeds = [0, 1]
reference = "base"
shots = [0]
qlora = true
eval = "{p(ev)}"
out = "{p(tmp / 'exp')}"
dev_eval = "{p(dev)}"
device = "cpu"
[pretrain]
seq_len = 32
global_batch = 4
micro_batch = 2
eval_every = 100
val_windows = 2
save_every = 0
[qlora_config]
quant = "none"
epochs = 1
max_len = 128
[[arms]]
name = "base"
[[arms]]
name = "a"
parts = ["{p(tmp / 'a.jsonl')}"]
[[arms]]
name = "ab"
parts = ["{p(tmp / 'a.jsonl')}=1", "{p(tmp / 'b.jsonl')}=1"]
""", encoding="utf-8")
    return plan


def test_plan_runs_resumes_and_summarizes(tmp_path):
    plan_path = setup(tmp_path)
    plan = ex.load_plan(plan_path)
    todo = ex.steps(plan)
    kinds = [k for k, *_ in todo]
    # base: 1 evaluation + 2 QLoRA; a, ab: corpus + 2 x (pretrain, evaluate, qlora)
    assert kinds.count("corpus") == 2 and kinds.count("pretrain") == 4
    assert kinds.count("evaluate") == 5 and kinds.count("qlora") == 6
    assert kinds.count("evaluate_dev") == 5
    assert ex.run_plan(plan_path, log=lambda *_: None) == 0
    assert all(out.exists() for *_, out in todo)

    s = json.loads((tmp_path / "exp" / "t" / "summary.json").read_text())
    pre = s["variants"]["pretrained"]
    assert set(pre["arms"]) == {"base", "a", "ab"}
    assert pre["arms"]["a"]["wikitext_ppl"]["n"] == 2
    assert set(pre["vs_reference"]) == {"a", "ab"} and set(pre["vs_reference"]["a"]) == {"0", "1"}
    dev = s["variants"]["dev"]["arms"]
    assert dev["ab"]["clinical_ppl"]["n"] == 2 and "pubmedqa_0shot_acc" not in dev["ab"]
    assert "qlora" in s["variants"] and "pubmedqa_0shot_f1" in s["variants"]["qlora"]["arms"]["ab"]
    md = (tmp_path / "exp" / "t" / "summary.md").read_text(encoding="utf-8")
    assert "| ab | 2 |" in md and "Paired bootstrap" in md

    # a second run finds everything done and only rewrites the summary
    before = (tmp_path / "exp" / "t" / "runs" / "a-s0" / "summary.json").stat().st_mtime
    ex.run_plan(plan_path, log=lambda *_: None)
    assert (tmp_path / "exp" / "t" / "runs" / "a-s0" / "summary.json").stat().st_mtime == before


def test_plan_validation(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text('name = "x"\nbogus = 1\n[[arms]]\nname = "a"\n')
    with pytest.raises(ex.PlanError):
        ex.load_plan(p)
    p.write_text('name = "x"\nreference = "z"\n[[arms]]\nname = "a"\n')
    with pytest.raises(ex.PlanError):
        ex.load_plan(p)
