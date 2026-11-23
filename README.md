# Dataset Factory

[![CI](https://github.com/MdMuzammilNihal/DatasetFactory/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MdMuzammilNihal/DatasetFactory/actions/workflows/ci.yml) ![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.13-blue) ![tests](https://img.shields.io/badge/tests-289%20passing-brightgreen) ![node](https://img.shields.io/badge/node-22-green)

A reproducible pipeline for curating domain data for LLM pretraining, plus an evaluation framework that shows **when
curation helps and when it does not**. It curates real Hugging Face corpora for six domains (general, news, medical,
legal, scientific, code), with a live dashboard, and measures the effect on Pythia language models.

**Highlights**
- Parallel, resumable curation: language ID, domain validators, quality score, learned classifier, MinHash dedup, PII
  scrubbing, per-document licence tracking, exact token budgets. A stopped-and-resumed job produces the same records in
  the same order as an uninterrupted one, for any worker count.
- Scale: a 300,000,133-token job (306,130 documents from 687,590 candidates) on one laptop.
- FastAPI backend and a Next.js dashboard with live progress (SSE), rejection reasons, score histograms, record browser
  and upload of your own data. One-command Docker demo.
- Evaluation done carefully: same-source unfiltered baselines, per-stage ablations, 13-gram test-set decontamination,
  separate dev and test sets, 3 seeds, paired bootstrap confidence intervals.
- 289 automated tests (backend, training, frontend) and CI on every push.

Measured on Pythia-160M with 20M-token corpora: curation clearly helps for general web text, helps a little for legal
(~1.3%) and code (~0.5%) text inside a mixed corpus, and the medical filters add nothing measurable.
See [Results](#results).

## Screenshots

![dashboard](docs/img/dashboard.png)
![job running](docs/img/job-running.png)

A finished legal job with the record browser: [docs/img/job-completed-legal.png](docs/img/job-completed-legal.png).

## Architecture

```
backend/    curation pipeline, CLI, FastAPI server       (Python package dataset_factory)
frontend/   Next.js dashboard (live jobs, records, rejections)
training/   corpus building, pretraining, evaluation, QLoRA, experiment runner (package dftrain)
docker-compose.yml   API + dashboard as containers
```

Per-document pipeline (worker processes; all settings in `backend/config/default.toml`, unknown keys are rejected,
every record carries the config fingerprint):

    licence policy -> global filter (incl. language ID) -> domain validator -> quality score
      -> learned classifier -> PII scrub -> hash + MinHash signature -> record

Then, in the main process: deduplication (exact hash, then MinHash at 0.85) and writing.

- **Agents:** `FactoryAgent` streams 13 real Hugging Face sources, `GeneratorAgent` adds general web text (FineWeb),
  `ScorerAgent` evaluates documents. A labelled synthetic source is the offline fallback.
- **Orchestrator:** process pool, results merged in chunk order, checkpoints, exact resume. Each job gets a folder
  `backend/runs/<job_id>/` with the data, rejections (no text), dedup index, checkpoint and a manifest (spec, state,
  sha256 of the data, config fingerprint).
- **Learned classifier:** fastText-style linear model, a pass/fail stage after scoring. Every validator check, score
  feature and stage can be switched off by name for ablations.
- **`dftrain`:** held-out eval sets, token-exact mixed corpora with decontamination, continued pretraining of Pythia,
  perplexity and PubMedQA evaluation, QLoRA, and an experiment runner (seeds, summaries, paired bootstrap).

## Quick start

**Docker (one command):**

```bash
docker compose up --build        # then open http://localhost:3000
```

Starts the API (`:8000`) and the dashboard (`:3000`). In the new-job form pick **synthetic data** for an instant demo;
real data streams from Hugging Face and takes 20-110 s per source to start. Job results live in the `runs` volume.
Both ports are published on `127.0.0.1` only: the API has no login, so do not expose it to a network.

**Local setup** (Python 3.13, Node 22):

```powershell
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[api,dev,train]"
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -e "../training[train,dev]" peft bitsandbytes psutil
cd ../frontend; npm install
```

**Curate (from `backend`):**

```powershell
.venv/Scripts/python -m dataset_factory curate legal --tokens 2M   # also --docs, --candidates, --mode random, --synthetic, --input f.jsonl, --sources a,b, --workers, --seed, --out
.venv/Scripts/python -m dataset_factory resume runs/<job_id>
.venv/Scripts/python -m dataset_factory serve --port 8000          # API, docs at /docs
cd ../frontend; npm run dev                                        # dashboard at http://localhost:3000
```

`--mode random` uses the same sources with no filters, dedup or PII scrub (the same-source baseline).
**Your own data:** tick "your data (upload)" on the new-job form (`.jsonl`, `.csv`, `.parquet` or `.txt`, max 200 MB),
or use `POST /api/uploads` then `POST /api/jobs` with `upload_id`, or `curate DOMAIN --input f.jsonl`. Uploads can be
listed and deleted through the API, and unused ones older than 14 days are removed at start-up.

**Train and evaluate (from `training`):**

```powershell
../backend/.venv/Scripts/python -m dftrain evalsets        # held-out test sets; `domainsets` for code/legal
../backend/.venv/Scripts/python -m dftrain corpus NAME --budget 19.8M --part ../backend/runs/<job_id>=0.6 --part ...
../backend/.venv/Scripts/python -m dftrain pretrain --corpus data/corpora/NAME --out data/runs/RUN
../backend/.venv/Scripts/python -m dftrain evaluate data/runs/RUN/model --out data/results/RUN.json
../backend/.venv/Scripts/python -m dftrain experiment plans/legal-160m.toml [--dry-run]
```

**Tests:** `cd backend; pytest -q`, `cd training; pytest -q` and `cd frontend; npm test`.

## Results

Continued pretraining of Pythia-160M on ~19.8M-token corpora (604 steps), measured on decontaminated held-out test sets
(WikiText, held-out PubMed abstracts, PubMedQA contexts, court opinions, CodeSearchNet functions). "Ratio vs random" is
perplexity relative to a corpus built from the **same sources with no filters** (paired bootstrap; below 1 means
curation helps). Mean of 3 seeds unless noted; seed std of perplexity is 0.00-0.35.

![results](docs/img/results.svg)

| arm | medical: WikiText / medical / clinical | legal: WikiText / legal | code: WikiText / code |
|---|---|---|---|
| curated (1 seed legal/code) | 1.513 / 1.019 / **0.975** | 1.405 / 1.003 | 2.720 / **0.987** |
| curated-mix (medical: 1 seed) | 0.915 / 1.014 / 0.992 | **0.910 / 1.024** | **0.955 / 0.998** |
| loose-mix | 0.907 / 1.000 / 1.000 | 0.914 / 1.031 | 0.962 / 1.003 |
| control-random-mix | **0.904 / 0.999** / 1.000 | 0.914 / 1.037 | 0.955 / 1.003 |
| wiki-mix (raw Wikipedia 40%) | 0.922 / 1.000 / 1.001 | 0.943 / 1.035 | 0.971 / 1.003 |

*curated* = pipeline output only; *curated-mix* = 60% curated domain + 40% curated general; *loose-mix* = validator and
score threshold off; *control-random-mix* = the unfiltered job's domain documents (web text removed) 60% + the same
curated general 40%; *wiki-mix* = unfiltered domain part 60% + raw Wikipedia 40%.

Absolute test perplexity: medical random 14.22, control 14.21, loose-mix 14.22; legal random 15.08, curated-mix
15.44 +- 0.11, control 15.63 +- 0.07; code random 6.30, curated-mix 6.29, control 6.32. Base Pythia-160M: WikiText 26.01,
medical 15.46, legal 16.42, code 7.11.

- **General web text: curation clearly helps.** Curated general text in a 40% share beats raw FineWeb and raw Wikipedia
  on WikiText by ~1.5-3%, with the same domain perplexity. Part of this is that Wikipedia is in-domain for WikiText.
- **Legal:** curated-mix 1.024 vs control 1.037 (~1.3%, above the seed std of ~0.7%). Curated legal alone only ties random.
- **Code:** curated-mix 0.998 vs control 1.003 (~0.5%), small but in all 3 seeds. Curation helps Go most (-3% vs
  random) and Python not at all (`code:syntax_blocks` removes the most Python).
- **Medical: the filters add nothing.** loose-mix = control = random. Strict curation narrows the data (66% PubMedQA
  abstracts): about 2% worse on general PubMed, 1.51x worse on WikiText, 2.5% better on clinical-style text.
- Continued pretraining on any 20M corpus here makes WikiText worse than base unless a mix includes general text.

### PubMedQA and QLoRA

At 160M, PubMedQA 0-shot accuracy is **the yes/no word bias, not medical reasoning**: 55% accuracy equals always
answering "yes" (majority baseline 55.2%), and yes-vs-no AUROC is 0.48-0.52 for every 160M model, base included. QLoRA
(500 train items, 3 epochs, 0-shot test of 500) on the released models:

| model | acc % (95% CI) | macro-F1 | pred yes/no/maybe |
|---|---|---|---|
| 160M fp16, no tuning | 51.2 (46.8-55.6) | 0.26 | 444/56/0 |
| 160M QLoRA s0 | 55.2 (50.8-59.6) | 0.28 | 473/27/0 |
| 410M 4-bit, no tuning | 47.4 | 0.33 | 252/248/0 |
| 410M QLoRA s0 / s1 / s2 | 65.2 / 68.4 / 68.8 (mean 67.5 +- 2.0) | 0.46 / 0.47 / 0.47 | e.g. 280/213/7 |
| 1B 4-bit, no tuning | 55.4 (50.8-60.0) | 0.35 | 385/115/0 |
| 1B QLoRA s0 | 67.8 (63.6-72.0) | 0.47 | 309/186/5 |

160M learns only the label prior; 410M learns real signal (+10 to +14 points over majority).

### Scaling

A general curated job reached **300,000,133 tokens** (306,130 documents from 687,590 candidates, 44.5% kept, 31 min,
1.7 GB). Checked: line count = accepted, token sum matches, sha256 = manifest, no duplicate ids or texts across 3
resumes. A 50M-token run took 314 s.
