#!/usr/bin/env bash
# Rerank percentile sensitivity sweep on the 15-case short subset.
# Reproduces the sensitivity argument around Proposition 1.
#
# Usage:
#   bash scripts/run_percentile_sweep.sh
#
# Writes percentile_sweep_results.txt in the repo root.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT=percentile_sweep_results.txt
: > "$OUT"
for pct in 0.10 0.20 0.30 0.40 0.50; do
  find src evaluation -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  echo "=== PERCENTILE=$pct ===" | tee -a "$OUT"
  RERANK_PCT=$pct python -u evaluation/quick_eval.py 2>&1 \
    | tee "/tmp/_eval_p${pct}.log" \
    | grep -E "SHORT_SCORE|MEAN_HPWL|MEAN_V_REL|FEASIBLE|MEAN_AREA_GAP" \
    | tee -a "$OUT"
done
echo "=== DONE ===" | tee -a "$OUT"
