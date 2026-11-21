#!/bin/bash
# Night 1 queue (prepared 2026-11-21): extra seeds for the close comparisons + raw-Wikipedia mix arms,
# then per-language code perplexity, then summaries, then data/logs/night1.done.
# Run from anywhere:  bash training/scripts/queue-night1.sh   (about 29 pretraining runs, ~19 min each)
# Re-running continues: finished steps are skipped and a stopped pretraining resumes from its checkpoint.
cd "$(dirname "$0")/.." || exit 1
PY=../backend/.venv/Scripts/python
export PYTHONUNBUFFERED=1
mkdir -p data/logs
run_plan() {   # plan name; one retry if the runner exits non-zero (e.g. a CUDA hiccup)
  for attempt in 1 2; do
    $PY -m dftrain experiment plans/$1.toml >> data/logs/night1-$1.log 2>&1 && return 0
    echo "plan $1 attempt $attempt failed $(date)" >> data/logs/night1.status
  done
  return 1
}
echo "start $(date)" > data/logs/night1.status
for p in medical-160m-recipes legal-160m code-160m; do
  run_plan $p; echo "$p exit=$? $(date)" >> data/logs/night1.status
done
# per-language code perplexity for every code result (existing ones are filled in from per-document NLL)
$PY -m dftrain langppl data/experiments/code-160m/results/*.json --eval data/eval > data/logs/night1-langppl.log 2>&1
$PY -m dftrain langppl data/experiments/code-160m/results-dev/*.json --eval data/eval-dev >> data/logs/night1-langppl.log 2>&1
for p in medical-160m-recipes legal-160m code-160m; do
  $PY -m dftrain experiment plans/$p.toml --summary-only >> data/logs/night1-summary.log 2>&1
done
echo "end $(date)" >> data/logs/night1.status
echo done > data/logs/night1.done
