"""Module 10a: n-gram decontamination, eval-set helpers, corpus packing (offline)."""

import json
from pathlib import Path

import numpy as np
import pytest

from dftrain import corpus, evalsets
from dftrain.ngrams import OverlapIndex, ngram_hashes
from dftrain.tok import encode_many, eos_id, load_tokenizer

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
         "omicron pi rho sigma tau upsilon").split()


def sentence(offset: int, n: int = 20) -> str:
    return " ".join(WORDS[(offset + i) % len(WORDS)] + str((offset + i) // len(WORDS))
                    for i in range(n))


def test_ngram_hashes_stable_and_case_insensitive():
    a = ngram_hashes("The Quick brown fox " * 5)
    b = ngram_hashes("the quick, BROWN fox! " * 5)
    assert len(a) == 20 - 13 + 1 and np.array_equal(a, b)
    assert len(ngram_hashes("too short")) == 0


def test_overlap_index_needs_several_shared_13grams():
    idx = OverlapIndex()
    idx.add("eval", [sentence(0, 60)])
    filler = " ".join(f"pad{i}" for i in range(40))
    shared13 = filler + " " + " ".join(sentence(5, 60).split()[:13]) + " " + filler
    shared20 = filler + " " + " ".join(sentence(5, 60).split()[:20]) + " " + filler
    assert idx.shared(shared13) == {"eval": 1}
    assert idx.hits(shared13) == []                      # one stock phrase is not a copy
    assert idx.hits(shared13, min_shared=1) == ["eval"]
    assert idx.hits(shared20) == ["eval"]                # 8 shared 13-grams
    assert idx.hits(" ".join(sentence(5, 60).split()[:14])) == ["eval"]   # short doc, half shared
    assert idx.hits(sentence(200, 40)) == []


def test_split_wikitext_articles():
    lines = [" = A = \n", " text a \n", " = = Sub = = \n", " more a \n", " = B = \n", " text b \n"]
    arts = evalsets.split_wikitext(lines)
    assert [a["title"] for a in arts] == ["A", "B"]
    assert "Sub" in arts[0]["text"] and "text b" in arts[1]["text"]


def test_split_pubmedqa_stratified_and_disjoint():
    rows = [{"pubid": str(i), "label": "yes" if i % 10 < 6 else ("no" if i % 10 < 9 else "maybe")}
            for i in range(1000)]
    test, train = evalsets.split_pubmedqa(rows, 500, seed=0)
    assert len(test) == 500 and len(train) == 500
    assert not {r["pubid"] for r in test} & {r["pubid"] for r in train}
    assert abs(sum(r["label"] == "yes" for r in test) - 300) <= 1
    again, _ = evalsets.split_pubmedqa([dict(r) for r in rows], 500, seed=0)
    assert [r["pubid"] for r in again] == [r["pubid"] for r in test]


def test_parse_count():
    assert corpus.parse_count("20M") == 20_000_000
    assert corpus.parse_count("1.5k") == 1500
    with pytest.raises(corpus.CorpusError):
        corpus.parse_count("lots")


def _job(tmp: Path, name: str, n: int, start: int) -> Path:
    d = tmp / name
    d.mkdir()
    with open(d / "data.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"{name}-{i}", "text": sentence(start + i * 3, 40),
                                "domain": name, "provenance": {"source": f"src-{name}"}}) + "\n")
    return d


def _eval_dir(tmp: Path, contaminated_text: str) -> Path:
    d = tmp / "eval"
    d.mkdir()
    evalsets.write_jsonl(d / "wikitext.jsonl", [{"title": "t", "text": contaminated_text}])
    evalsets.write_jsonl(d / "medical.jsonl", [{"text": "unrelated words only"}])
    for name in ("pubmedqa_test", "pubmedqa_train"):
        evalsets.write_jsonl(d / f"{name}.jsonl", [{"context": "c", "question": "q",
                                                     "long_answer": "a", "label": "yes"}])
    return d


def test_build_corpus_exact_budget_shares_and_decontamination(tmp_path):
    a = _job(tmp_path, "med", 400, 0)
    b = _job(tmp_path, "gen", 400, 7)
    first_med = json.loads((a / "data.jsonl").open(encoding="utf-8").readline())["text"]
    ev = _eval_dir(tmp_path, first_med)
    tok = load_tokenizer()
    budget = 6000
    meta = corpus.build_corpus("mix", [corpus.Part(a, 3), corpus.Part(b, 1)], budget,
                               tmp_path / "out", ev, seed=1, val_every=10, log=lambda *_: None)
    train = corpus.load_tokens(tmp_path / "out" / "train.bin")
    assert len(train) == budget == meta["train_tokens"]
    med, gen = meta["parts"]
    assert med["target_tokens"] == 4500 and gen["target_tokens"] == 1500
    assert med["tokens"] >= 4500 and gen["tokens"] >= 1500
    assert med["decontaminated"].get("wikitext", 0) >= 1      # the copied document is dropped
    assert gen["decontaminated"] == {} or "wikitext" in gen["decontaminated"]
    assert meta["val_documents"] > 0
    # documents are separated by EOS and decode back to the source text
    eos = eos_id(tok)
    assert int((np.asarray(train) == eos).sum()) >= meta["train_documents"] - 1
    first = np.asarray(train)[: int(np.argmax(np.asarray(train) == eos))]
    assert tok.decode(first.tolist()).strip().split()[0][:-1] in WORDS
    # same seed -> identical bytes; other seed -> other order
    meta2 = corpus.build_corpus("mix", [corpus.Part(a, 3), corpus.Part(b, 1)], budget,
                                tmp_path / "out2", ev, seed=1, val_every=10, log=lambda *_: None)
    assert meta2["sha256"] == meta["sha256"]
    meta3 = corpus.build_corpus("mix", [corpus.Part(a, 3), corpus.Part(b, 1)], budget,
                                tmp_path / "out3", ev, seed=2, val_every=10, log=lambda *_: None)
    assert meta3["sha256"]["train.bin"] != meta["sha256"]["train.bin"]


def test_build_corpus_short_part_errors_unless_allowed(tmp_path):
    a = _job(tmp_path, "small", 5, 0)
    with pytest.raises(corpus.CorpusError, match="only"):
        corpus.build_corpus("s", [corpus.Part(a, 1)], 100_000, tmp_path / "o", None,
                            log=lambda *_: None)
    meta = corpus.build_corpus("s", [corpus.Part(a, 1)], 100_000, tmp_path / "o", None,
                               allow_short=True, val_every=0, log=lambda *_: None)
    assert meta["short"] and meta["train_tokens"] == meta["parts"][0]["tokens"]
    tok = load_tokenizer()
    expected = sum(len(ids) + 1 for ids in encode_many(tok, [sentence(i * 3, 40) for i in range(5)]))
    assert meta["train_tokens"] == expected


def test_dev_contamination_check_finds_copies_only():
    from dftrain.evalsets import contaminated
    words = lambda k, n: " ".join(f"w{k}_{i}" for i in range(n))   # noqa: E731
    dev = [words(0, 100), words(1, 100), words(2, 100), "short text only"]
    shared = " ".join(words(1, 100).split()[10:40])               # 30 words copied from dev[1]
    stock = " ".join(words(2, 100).split()[:14])                   # 2 shared 13-grams only
    train = ["unrelated " * 50, "intro " + shared + " outro", stock + " " + words(9, 80), words(0, 100)]
    hit = contaminated(dev, train)
    assert set(hit) == {0, 1} and hit[0] == 88 and hit[1] == 18
