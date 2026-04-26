# Gap-Aware Candidate Reranking for Diffusion-Guided Multi-Strategy Floorplanning

This directory contains the core project code used by the Code Ocean
capsule.

The top-level capsule entry points live in:

- `../../code/main.py` for the default inference-only run
- `../../code/train.py` for optional retraining from scratch

## Checkpoint layout

Inside the full capsule, released checkpoints live in:

`../../data/checkpoints/`

The solver searches for checkpoints in this order:

1. `CHECKPOINT_DIR`
2. package-local `checkpoints/`
3. top-level `../../data/checkpoints/`
4. top-level `../../models/` for backward compatibility
5. package-local `models/`
6. `./models/`

The released inference run uses:

- `dit_base_ckpt.pt`
- `nn_hint_ckpt.pt`

## Core contents

- `src/solver.py`
  Multi-strategy floorplanner with the gap-aware reranker
- `evaluation/quick_eval.py`
  FloorSet-Lite evaluation harness
- `evaluation/quick_eval_prime.py`
  FloorSet-Prime evaluation harness
- `src/train_dit_base.py`
  Optional DiT-base training
- `src/train_nn_hint.py`
  Optional NN-hint training
- `src/train_dit_large_hpwl.py`
  Optional negative-result training variant
- `results/*.csv`
  Released experiment tables used to regenerate manuscript artifacts

## Training note

Training is optional and not part of the main Code Ocean run. The
training scripts now use fixed seeds and save checkpoints into
`../../data/checkpoints/` by default when run inside the full capsule.

Approximate single-GPU training times:

- `NN-hint`: about 6 hours
- `DiT-base`: about 11 hours
- `DiT-large-hpwl`: longer; included for the negative-result analysis

## Headline metrics

Released full-set metrics from the paper:

- FloorSet-Lite: `S_100 = 1.40228`, `100/100` feasible
- FloorSet-Prime: `S_100 = 1.54862`, `91/100` feasible

The top-level capsule run does not rerun the full 100-case benchmark by
default because that is too slow for one-click verification. Instead, it
runs a short inference-only sanity check and regenerates paper tables
and figures from the released CSV traces.
