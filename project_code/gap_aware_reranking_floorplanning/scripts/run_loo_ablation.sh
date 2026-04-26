#!/usr/bin/env bash
# Leave-one-out candidate-generator ablation on the 15-case short subset.
# Disables each of the seven candidate generators in turn and reports
# SHORT_SCORE, mean HPWL gap, v_rel, and feasibility count.
#
# Usage:
#   bash scripts/run_loo_ablation.sh
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
OUT=loo_ablation_results.txt
: > "$OUT"

for gen in "dit-direct" "dit-abacus" "dit-cluster" "shelf" "skyline-clust" "sp-sa" "nn-shelf"; do
  find src evaluation -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  echo "=== DISABLE=$gen ===" | tee -a "$OUT"
  ABLATE_DISABLE="$gen" python -u evaluation/quick_eval.py 2>&1 \
    | grep -E "SHORT_SCORE|MEAN_HPWL|MEAN_V_REL|FEASIBLE|EVAL_WALL" \
    | tee -a "$OUT"
done

echo "=== DISABLE=none (baseline) ===" | tee -a "$OUT"
find src evaluation -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
python -u evaluation/quick_eval.py 2>&1 \
  | grep -E "SHORT_SCORE|MEAN_HPWL|MEAN_V_REL|FEASIBLE|EVAL_WALL" \
  | tee -a "$OUT"
echo "=== DONE ===" | tee -a "$OUT"
