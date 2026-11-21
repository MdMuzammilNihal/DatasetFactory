"""Code and legal evaluation sets (offline: the samplers are replaced by fixed rows)."""
import json
import sys
import tomllib
from pathlib import Path

import pytest

from dftrain import evalsets

BACKEND = Path(__file__).resolve().parents[2] / "backend"


def words(k, n):
    return " ".join(f"w{k}_{i}" for i in range(n))


def test_code_text_matches_backend_builder():
    if not (BACKEND / "dataset_factory").exists():
        pytest.skip("backend not present")
    sys.path.insert(0, str(BACKEND))
    try:
        from dataset_factory.sources import TEXT_BUILDERS
    finally:
        sys.path.remove(str(BACKEND))
    rows = [{"whole_func_string": "def f(x):\n    return x", "func_documentation_string": "Return x.\nAs is.",
             "language": "python"},
            {"whole_func_string": "func F() int { return 1 }", "func_documentation_string": "F returns one.",
             "language": "go"},
            {"whole_func_string": "def g():\n    \"\"\"Doc here.\"\"\"\n    pass", "func_documentation_string": "Doc here.",
             "language": "python"}]
    for r in rows:
        assert evalsets.code_with_doc(r) == TEXT_BUILDERS["code_with_doc"](r)


def test_legal_test_and_dev_files_are_not_read_by_curation():
    cfg = tomllib.loads((BACKEND / "config" / "default.toml").read_text(encoding="utf-8"))
    read = set(cfg["sources"]["caselaw"]["data_files"])
    held = set(evalsets.LEGAL_FILES["test"]) | set(evalsets.LEGAL_FILES["dev"])
    assert len(held) == 8 and not (held & read) and len(read) == 173 - 8
    assert cfg["sources"]["code_python"].get("split", "train") == "train"


def test_length_rules():
    assert not evalsets.keep_code("x" * 99) and evalsets.keep_code("x" * 100)
    assert not evalsets.keep_code("x" * (evalsets.CODE_MAX_CHARS + 1))
    assert not evalsets.keep_legal(words(0, 99)) and evalsets.keep_legal(words(0, 100))


def test_build_domains_drops_dev_overlap_keeps_test(tmp_path, monkeypatch):
    rows = {("code", "test"): [words(0, 60), words(1, 60)], ("code", "dev"): [words(2, 60), words(3, 60)],
            ("legal", "test"): [words(4, 200)], ("legal", "dev"): [words(5, 200), words(6, 200)]}
    monkeypatch.setattr(evalsets, "_sample_code", lambda split, rng: [{"text": t} for t in rows[("code", split)]])
    monkeypatch.setattr(evalsets, "_sample_legal", lambda split, rng: [{"text": t} for t in rows[("legal", split)]])
    job = tmp_path / "data.jsonl"
    job.write_text("\n".join(json.dumps({"text": t}) for t in
                             ["prefix " + words(1, 60), words(3, 60) + " suffix", words(6, 200), "unrelated"]),
                   encoding="utf-8")
    test, dev = tmp_path / "eval", tmp_path / "dev"
    dev.mkdir()
    (dev / "manifest.json").write_text(json.dumps({"files": {"wikitext": {"documents": 1}}}), encoding="utf-8")
    rep = evalsets.build_domains(test, dev, [job], log=lambda *_: None)
    assert rep["code/test"]["documents"] == 2 and rep["code/test"]["overlap_with_jobs"] == 1
    assert rep["code/dev"]["documents"] == 1 and rep["code/dev"]["dropped_overlap"] == 1
    assert rep["legal/dev"]["documents"] == 1
    assert [r["text"] for r in evalsets.read_jsonl(dev / "legal.jsonl")] == [words(5, 200)]
    man = json.loads((dev / "manifest.json").read_text(encoding="utf-8"))
    assert set(man["files"]) == {"wikitext", "code", "legal"}            # merged, not replaced
    # the corpus builder's decontamination sees the new sets once they exist
    for name in ("wikitext", "medical"):
        (test / f"{name}.jsonl").write_text(json.dumps({"text": "a b"}) + "\n", encoding="utf-8")
    for name in ("pubmedqa_test", "pubmedqa_train"):
        (test / f"{name}.jsonl").write_text(json.dumps({"context": "c", "question": "q", "long_answer": "l"}) + "\n",
                                            encoding="utf-8")
    t = evalsets.eval_texts(test)
    assert t["code"] == rows[("code", "test")] and t["legal"] == rows[("legal", "test")]
