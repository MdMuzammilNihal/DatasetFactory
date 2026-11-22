# Dataset Factory

A reproducible pipeline for curating domain data for LLM pretraining, plus an evaluation framework that shows **when
curation helps and when it does not**.

It is a rebuild, from the project paper, of a college major project (multi-agent data curation for six domains:
general, news, medical, legal, scientific, code). The original repository is not available. The paper is used as a
guide for the design (agents, stages, dashboard, experiments), not as ground truth for numbers; where our own
measurements disagree with it, this README says so.

Framing of the results: this is **not** a claim that "our curation makes better data". Measured on Pythia-160M with
20M-token corpora, curation clearly helps for general web text, helps a little for legal (~1.3%) and code (~0.5%)
text inside a mixed corpus, and the medical filters add nothing measurable. See [Results](#results).

## Architecture

```
backend/    curation pipeline, CLI, FastAPI server       (Python package dataset_factory)
frontend/   Next.js dashboard (live jobs, records, rejections)
training/   corpus building, pretraining, evaluation, QLoRA, experiment runner (package dftrain, standalone)
```

**Per-document pipeline** (inside worker processes; everything tunable lives in `backend/config/default.toml`, the
loader rejects unknown keys, every record carries the config fingerprint):

    licence policy -> global filter (incl. language ID) -> domain validator -> quality score
      -> learned classifier -> PII scrub -> hash + MinHash signature -> record

Then in the main process: deduplication (exact hash, then MinHash at 0.85) and writing.

**Agents** (`agents.py`): `FactoryAgent` streams real Hugging Face corpora, `GeneratorAgent` streams general web text
(FineWeb) as a source of extra candidates, `ScorerAgent` evaluates a document. Candidates are interleaved by
`factory_share`. Sources (13 real ones, plus a clearly labelled synthetic offline fallback) are in
`backend/dataset_factory/sources.py` / `[sources.*]`.

**Orchestrator** (`orchestrator.py`): a `ProcessPoolExecutor`, results merged in chunk order (output identical for any
worker count), checkpoints, exact resume, exact token budget with the Pythia tokenizer. A job folder
`backend/runs/<job_id>/` holds `config.toml`, `data.jsonl`, `rejections.jsonl`, `dedup.bin`, `checkpoint.json`,
`manifest.json` (spec, state, sha256 of the data, config fingerprint).

**Learned classifier:** a fastText-style linear model (hashed word and word-pair features, NumPy inference), a
pass/fail stage after scoring. Every validator check, score feature and pipeline stage can be switched off by name
(`config/ablations/*.toml`, made by `backend/scripts/make_ablation_configs.py`) for ablations.

**API** (`dataset_factory/api`): FastAPI, `GET/POST /api/jobs`, stop/resume, manifest, records, rejections, download,
SSE progress (docs at `/docs`). **Dashboard**: new-job form, job list, live job detail (yield, rejections by label,
by source/agent, licences, score histogram), browse records and rejections.

**`dftrain`**: `evalsets` / `domainsets` (held-out evaluation sets), `corpus` (token-exact mixed corpora with
decontamination), `pretrain` (continued pretraining of Pythia), `evaluate` (perplexity, PubMedQA), `qlora`,
`experiment` (plan runner with seeds, summaries and paired bootstrap), Colab notebook (untested).

## Setup

Tested on Windows 11, Python 3.13, Node 22, RTX 4060 Laptop (8 GB), 24 GB RAM. Keep the project in a short path
(Windows long paths are off).

```powershell
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[api,dev,train]"
# training stack (CUDA 12.6 build of torch):
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e "../training[train,dev]" peft bitsandbytes psutil

cd ../frontend
npm install
```

The Pythia tokenizer (`EleutherAI/pythia-160m`, `tokenizer.json`) is downloaded to the Hugging Face cache on first
use and read from the cache afterwards. Use `datasets` streaming for Hugging Face data (the dataset viewer API
rate-limits bulk pulls).

## How to run

### Curate (CLI, from `backend`)

```powershell
.venv/Scripts/python -m dataset_factory curate legal --tokens 2M      # also --docs, --candidates, --mode random, --synthetic, --input f.jsonl, --sources a,b, --workers, --seed, --out
.venv/Scripts/python -m dataset_factory status runs/<job_id>
.venv/Scripts/python -m dataset_factory resume runs/<job_id>
.venv/Scripts/python -m dataset_factory benchmark --fetch 20000
```

`--mode random` uses the same sources and order with no filters, no dedup, no PII scrub (the same-source baseline).
For a long general job with a RAM guard that stops at a checkpoint and resumes when run again:
`.venv/Scripts/python scripts/scale_general.py --tokens 300M --workers 3 --min-free-gb 4`.

### API and dashboard

```powershell
cd backend;  .venv/Scripts/python -m dataset_factory serve --port 8000
cd frontend; npm run dev          # http://localhost:3000, API URL via NEXT_PUBLIC_API_URL
```

### Train and evaluate (from `training`)

```powershell
../backend/.venv/Scripts/python -m dftrain evalsets                      # held-out test sets (~1.5 min)
../backend/.venv/Scripts/python -m dftrain domainsets                    # code and legal sets
../backend/.venv/Scripts/python -m dftrain corpus NAME --budget 19.8M --part ../backend/runs/<job_id>=0.6 --part ...
../backend/.venv/Scripts/python -m dftrain pretrain --corpus data/corpora/NAME --out data/runs/RUN
../backend/.venv/Scripts/python -m dftrain evaluate data/runs/RUN/model --out data/results/RUN.json
../backend/.venv/Scripts/python -m dftrain compare A.json B.json
../backend/.venv/Scripts/python -m dftrain qlora EleutherAI/pythia-410m --out data/runs/RUN
../backend/.venv/Scripts/python -m dftrain experiment plans/legal-160m.toml [--dry-run] [--summary-only]
bash scripts/queue-night1.sh                                             # the overnight queue of plans used for the results below
```

Re-running `experiment` skips finished steps and resumes a stopped pretraining. Before long GPU runs close other
GPU apps and check `tok_per_s` in the run's `log.jsonl` (~18-19k on the 4060 for 160M); VRAM spilling into shared
memory makes training ~5x slower.

### Tests

```powershell
cd backend;  .venv/Scripts/python -m pytest -q
cd training; ../backend/.venv/Scripts/python -m pytest -q
```

## Verified end to end (2026-11-22)

A fresh run through every layer on real data: dashboard job -> corpus -> pretraining -> evaluation (small, to show
the chain works, not to produce a result).

| step | command / action | result | time |
|---|---|---|---|
| API + dashboard | `serve --port 8000`, `npm run dev` | both up in seconds | ~12 s |
| curate | legal, 2M tokens, real data (CAP caselaw), started from the dashboard form | 813 read, 533 accepted (65.6%), 2,000,068 tokens, 9.6 MB; top rejections `legal:citations` 108, `score:below_threshold` 84 | ~1 min |
| corpus | `dftrain corpus e2e-legal --budget 1.5M --part <job>` | 431 docs, 1,500,000 train tokens (6 docs decontaminated) | 3 s |
| pretrain | `dftrain pretrain ... --max-steps 20` (Pythia-160M) | 655,360 tokens, final val loss 2.528, 6.1 GB, ~18.6k tok/s | 42 s |
| evaluate | `dftrain evaluate ... --tasks wikitext,legal --shots 0` | WikiText ppl 41.03, legal ppl 17.84 | 55 s |

Twenty steps from a re-warmed released model make both perplexities worse than the base model (WikiText 26.01, legal
16.42): too short to recover, as already seen with 2M-token runs. This run checks that the chain works; it is not
evidence about data quality.

## Results

All results: continued pretraining of Pythia-160M on ~19.8M-token corpora (604 steps), perplexity on held-out test sets
that are decontaminated against the training data. "Ratio vs random" is perplexity relative to a corpus built from
the **same sources with no filters** (paired bootstrap; lower is better, below 1 = curation helps). Mean of 3 seeds
(`curated`, `no-validator`, `no-score`, `r1-mix`: 1 seed); dev sets agree in rank everywhere; seed std of ppl is
0.00-0.35.

| arm | medical: WikiText / medical / clinical | legal: WikiText / legal | code: WikiText / code |
|---|---|---|---|
| curated (1 seed legal/code) | 1.513 / 1.019 / **0.975** | 1.405 / 1.003 | 2.720 / **0.987** |
| curated-mix (medical: r1, 1 seed) | 0.915 / 1.014 / 0.992 | **0.910 / 1.024** | **0.955 / 0.998** |
| loose-mix (medical: r2) | 0.907 / 1.000 / 1.000 | 0.914 / 1.031 | 0.962 / 1.003 |
| control-random-mix | **0.904 / 0.999** / 1.000 | 0.914 / 1.037 | 0.955 / 1.003 |
| wiki-mix (raw Wikipedia 40%) | 0.922 / 1.000 / 1.001 | 0.943 / 1.035 | 0.971 / 1.003 |

Arms: *curated* = full pipeline output only; *curated-mix* = 60% curated domain + 40% curated general text;
*loose-mix* = validator and score threshold off; *control-random-mix* = the random job's domain documents (web text
removed) 60% + the same curated general 40%; *wiki-mix* = unfiltered domain part 60% + raw Wikipedia 40%.

Absolute test perplexity (mean +- std): medical random 14.22 +- 0.00, control-mix 14.21 +- 0.01, r2 14.22 +- 0.01;
legal random 15.08, curated-mix 15.44 +- 0.11, control 15.63 +- 0.07, wiki-mix 15.60 +- 0.10; code random 6.30,
curated-mix 6.29 +- 0.01, control 6.32 +- 0.00, wiki-mix 6.32 +- 0.01. Base Pythia-160M: WikiText 26.01, medical
15.46, legal 16.42, code 7.11.

What the tables say:
- **General web text: curation clearly helps.** Curated general text (Wikipedia + filtered FineWeb) in a 40% share
  gives a better WikiText ratio than the random arm's raw FineWeb (medical mixes 0.904-0.915, legal 0.910-0.914,
  code 0.955-0.962, all below 1) and than raw Wikipedia in the same share (wiki-mix 0.922 / 0.943 / 0.971), by
  ~1.5-3%; domain perplexity is the same either way. The gain is not only "it is Wikipedia", though Wikipedia text
  is partly in-domain for WikiText.
- **Legal:** the gain inside a 60/40 mix holds with 3 seeds: curated-mix 1.024 vs control 1.037 (about 1.3%, larger
  than the seed std ~0.7%). Curated legal alone only ties random (1.003, 1 seed).
- **Code:** curated-mix 0.998 vs control 1.003 (~0.5%), small, but in all 3 seeds on test. Curated alone beats random
  (0.987) partly because the random job has 24% web tokens. Per-language test ppl (go / java / javascript / python):
  base 7.86 / 6.73 / 6.36 / 7.81; random 7.10 / 5.84 / 5.67 / 6.94; curated (1 seed) 6.88 / 5.78 / 5.58 / 6.96.
  Curation helps Go most (-3% vs random) and Python not at all (`code:syntax_blocks` removes the most Python).
- **Medical: the filters add nothing.** With 3 seeds, loose-mix = control = random on medical and clinical text.
  Strict curation narrows the data (66% PubMedQA abstracts, 34% PubMed): it is 2.0% worse on general PubMed
  (ratio 1.019, all 3 seeds), 1.51x worse on WikiText, but 2.5% better on clinical-style text (0.975). Removing the
  validator or score threshold made both perplexities slightly better, not worse.
- Continued pretraining on any 20M corpus here makes WikiText worse than the base model unless a mix includes
  general text; only domain data improves domain perplexity.

### PubMedQA and QLoRA

At 160M, PubMedQA 0-shot accuracy is **the yes/no word bias, not medical reasoning**: the paper's 55% equals always
answering "yes" (majority baseline 55.2%). Yes-vs-no AUROC is 0.48-0.52 for every 160M model including the untouched
base, no paired difference excludes 0, and medical text has "no" 28x as often as "yes", which flips some models to
mostly "no". Do not use PubMedQA at 160M to compare corpora. QLoRA (paper Table 4 settings, 500 train items,
3 epochs, 0-shot test of 500 items) on the released models:

| model | acc % (95% CI) | macro-F1 | pred yes/no/maybe |
|---|---|---|---|
| 160M fp16, no tuning | 51.2 (46.8-55.6) | 0.26 | 444/56/0 |
| 160M QLoRA s0 | 55.2 (50.8-59.6) | 0.28 | 473/27/0 |
| 410M 4-bit, no tuning | 47.4 | 0.33 | 252/248/0 |
| 410M QLoRA s0 / s1 / s2 | 65.2 / 68.4 / 68.8 (mean 67.5 +- 2.0) | 0.46 / 0.47 / 0.47 | e.g. 280/213/7 |
| 1B 4-bit, no tuning | 55.4 (50.8-60.0) | 0.35 | 385/115/0 |
| 1B QLoRA s0 | 67.8 (63.6-72.0) | 0.47 | 309/186/5 |

160M learns only the label prior; 410M learns real signal (+10 to +14 points over the majority baseline). A QLoRA
comparison between *corpora* would only be meaningful from 410M up, which needs Colab for pretraining (not run).

### Scaling

`backend/runs/general-20261122-002235-9a07`: general, curated, **300,000,133 tokens**, 306,130 documents from 687,590
candidates (44.5% kept), 31 min of curation time, 1.7 GB folder; checked: line count = accepted, token sum matches,
sha256 = manifest, 0 duplicate ids, 0 exact duplicate texts across 3 resumes. The RAM guard stopped it three times
(free RAM fell to 1.4-3.5 GB while curating next to training) and it was resumed each time. Not used for training.
A 50M-token general run: 94,689 candidates read, 40,873 accepted (43%), 314 s.

## Decisions that differ from the paper (each measured)

- **Learned quality classifier, real language ID, PII scrubbing, licence tracking, exact Pythia token budgets,
  versioned config with seeds and manifest** are additions. `enabled = false` for the classifier gives the paper's
  rule-only pipeline.
- **fp16, not bf16.** bf16 autocast raises the untrained model's loss (Pythia-70M 3.74 fp32 -> 5.44 bf16 vs 3.77
  fp16; 160M 3.32 -> 3.83 vs 3.33). Pythia was trained in fp16 (the paper also lists FP16).
- **Peak lr 6e-5, not 2e-4**, global batch 32 x 1024 tokens (paper 256), 5% warmup (the paper's 2,000 warmup steps
  exceed its own ~763 steps). On 20M real general tokens, val loss: lr 2e-4 ends at 3.200 after a jump to 3.76; lr
  6e-5 ends at 3.131. Re-warming a converged model to 2e-4 spends most of a short run recovering. WikiText ppl with
  lr 2e-4 was 29.66, with 6e-5 25.36.
- **Decontamination** (13-word sequences, >= 8 shared drops a document): 10 Wikipedia articles in the 100M-token
  general run are WikiText test articles, so the paper's WikiText numbers for Wikipedia-containing corpora were
  likely contaminated. A single shared 13-gram (GPT-3's rule) dropped abstracts sharing only a stock phrase.
- **Honest evaluation:** same-source random baseline, per-filter ablation switches, 3 seeds, paired bootstrap,
  dev sets for choosing recipes and test sets reported after, majority baseline and macro-F1 for PubMedQA.
- **Budget 19.8M tokens, not 20M:** a 20M-token curation job counts tokens without EOS; the corpus adds one EOS per
  document and drops decontaminated documents, so some corpora cannot reach exactly 20M. Curate ~21M for 20M.
- **Code `min_chars` 100 (not 200):** Go yield 38% -> 89% (Python 84 -> 89%, Java 92 -> 93%).
- **Medical structure check** also counts finding verbs/phrases, `min_structure_cues` 2 -> 1: PubMed yield 32% -> 41%.
  News thresholds were left unchanged on purpose (lowering them gained +3% on CNN but let 51 of 200 legal texts
  through the news validator, up from 13).

## Weak spots

- Everything is **160M parameters and ~20M tokens**, one GPU, one family (Pythia). The paper's 410M/1B pretraining
  was not run (Colab notebook untested). Gains in legal (~1.3%) and code (~0.5%) are small and rest on 3 seeds;
  `curated` arms for legal/code and medical `r1`/ablations have 1 seed.
- WikiText gains partly reflect that the curated general text is mostly Wikipedia.
- The curated medical corpus is narrow (66% PubMedQA abstracts); the medical validator accepts only ~41% of PubMed.
  The news classifier partly keys on wire-service style; PII scrubbing is pattern matching (no names/addresses);
  three sources declare no licence (CC-News, PubMed, arXiv-summarization).
- Perplexity on held-out text is the only comparison at 160M; no downstream task is informative at this size.
- Resume re-reads each Hugging Face stream to its position (no seeking): slow for deep checkpoints (a 1,800-document
  resume took 491 s, mostly reopening sources). Memory during long curation reached 8.3 GB on a 100M-token run; dedup
  costs ~1.6 KB RAM per accepted document.
- Pipeline speed per worker (~450-680 docs/s) is below the paper's claim (2,477); network reading limits real runs.
- The dashboard has no tests and no auth (local use only); live records lists do not auto-refresh; a job shows
  empty zeros until the first progress callback (10-60 s with real sources).
- The paper's own results should not be treated as targets: its PubMedQA 55% equals always answering "yes", some of
  its tables disagree with each other, and it has no repeated runs.

`CLAUDE.md` holds the full development log (every module, measurements, rejected ideas) for anyone continuing the work.
