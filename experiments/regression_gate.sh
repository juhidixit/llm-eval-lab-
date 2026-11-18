#!/usr/bin/env bash
# Experiment 2: does the gate catch realistic regressions in a RAG app?
#   ./experiments/regression_gate.sh record   run baseline + candidates with local models (CPU, ~2 h)
#   ./experiments/regression_gate.sh replay   recompute every run and gate decision from cassettes (seconds)
set -euo pipefail
MODE="${1:-replay}"
SUITE=suites/rag_qa.yaml
mkdir -p results/runs results/gates

evallab run "$SUITE" --cache "$MODE" --out results/runs/baseline.json
evallab run "$SUITE" --cache "$MODE" --set system.prompt=prompts/rag_qa_v2_friendly.txt \
  --out results/runs/v2_friendly_prompt.json
evallab run "$SUITE" --cache "$MODE" --set system.max_contexts=1 \
  --out results/runs/v3_top1_context.json

for name in v2_friendly_prompt v3_top1_context; do
  evallab gate --suite "$SUITE" --baseline results/runs/baseline.json --candidate "results/runs/$name.json" \
    --markdown-out "results/gates/$name.md" || true
done
