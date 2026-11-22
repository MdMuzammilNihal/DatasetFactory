# Resume and GitHub copy (every number is measured in this repository; see README and CLAUDE.md)

## GitHub "About" (one line) and topics
Reproducible multi-domain data-curation pipeline for LLM pretraining, with an evaluation framework that measures when curation actually helps (FastAPI, Next.js, PyTorch).
Topics: `llm`, `data-curation`, `pretraining`, `nlp`, `fastapi`, `nextjs`, `pytorch`, `qlora`, `data-quality`, `deduplication`

## Resume entry (compact)
**Dataset Factory** - LLM pretraining data curation and evaluation (Python, FastAPI, Next.js/TypeScript, PyTorch) | github.com/<user>/<repo>
- Built a resumable, parallel curation pipeline for 6 domains (language ID, quality filters, domain validators, learned classifier, MinHash dedup, PII scrubbing, per-document licence tracking); curated a 300M-token dataset (306k documents) from streamed Hugging Face corpora; a stopped-and-resumed job produces the same records in the same order as an uninterrupted one.
- Shipped a FastAPI backend and a live Next.js dashboard (SSE progress, rejection reasons, score histograms, record browser, stop/resume).
- Designed the evaluation: same-source unfiltered baselines, per-stage ablations, 13-gram test-set decontamination, separate dev/test sets, 3 seeds, paired bootstrap CIs; ran 57 Pythia-160M continued-pretraining runs plus 4-bit QLoRA on 160M-1B models.
- Found that domain filters help little: legal text about 1.3% and code about 0.5% lower perplexity inside a mixed corpus, medical none; curated general text beat raw FineWeb/Wikipedia on WikiText. Also found 10 WikiText test articles in the Wikipedia training data and that the source paper's 55% PubMedQA accuracy equals always answering "yes".
- 245 automated tests (215 backend, 30 training), GitHub Actions CI.

## Two-line version
Built and evaluated an LLM-pretraining data-curation system (300M-token run, resumable, dashboard) and a rigorous evaluation framework (baselines, ablations, decontamination, 3 seeds) that showed most domain filters add under 1.3% - and corrected claims from the source paper.

## Talking points (interview)
- Why the controls matter: the curated medical corpus looked worse than random until the control (random medical text with the same general text mixed in) showed the filters add nothing and the gain came from the mix.
- Engineering: deterministic chunk-ordered merging makes output independent of worker count; memory guard for long jobs; decisions measured on real data (fp16 vs bf16 loss 3.74 -> 5.44 on the untrained model, lr 6e-5 vs 2e-4).
- Limits, stated openly: 160M models and 20M-token corpora; no downstream benchmark for legal/code; PubMedQA is only meaningful from 410M up.
