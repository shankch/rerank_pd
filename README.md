# Code Ocean Capsule Layout

This folder is prepared as a Code Ocean style reproducibility capsule
for the floorplanning paper.

The default run is inference-only. It loads the released checkpoints
from `data/checkpoints/`, runs a short verification evaluation, and
recreates paper-facing tables and figures in `results/codeocean_run/`.
It does not retrain models by default.

## Main run

From the capsule root:

```bash
pip install -r requirements.txt
python code/run_capsule.py
```

What the main run does:

- loads the released `DiT-base` and `NN-hint` checkpoints from
  `data/checkpoints/`
- runs a short inference-only verification pass on FloorSet-Lite and
  FloorSet-Prime
- copies the released paper CSV tables into `results/codeocean_run/`
- regenerates lightweight figures from the saved CSV traces
- writes a summary report in `results/codeocean_run/run_summary.md`

The main run is intended to finish in minutes, not hours.

## Optional training

Training is included for full reproduction from scratch, but it is not
part of the default Code Ocean run:

```bash
python code/train.py --target nn-hint
python code/train.py --target dit-base
```

Approximate training times on a single RTX 3090:

- `NN-hint`: about 6 hours
- `DiT-base`: about 11 hours
- `DiT-large-hpwl`: longer and included only for the negative-result
  analysis

The training scripts use fixed random seeds and save checkpoints into
`data/checkpoints/` by default. The released final-run checkpoints used
by the main inference-only run are already provided there.

## Capsule layout

- `code/`
  Main capsule entry points for evaluation and optional training
- `data/checkpoints/`
  Released checkpoints used by the default inference-only run
- `project_code/gap_aware_reranking_floorplanning/`
  Core solver, evaluation harnesses, training scripts, and released CSV
  results
- `supplementary/`
  Figure exports and raw manuscript support files

## Dataset note

The solver expects the public FloorSet repository to be available either
at `../FloorSet`, inside the project tree, or via the `FLOORSET_ROOT`
environment variable. The helper scripts preserve that behavior.

## Checkpoints

The default capsule run expects:

- `data/checkpoints/dit_base_ckpt.pt`
- `data/checkpoints/nn_hint_ckpt.pt`

Checksums are listed in `data/checkpoints/SHA256SUMS.txt`.

## Outputs

Generated outputs are written to:

`results/codeocean_run/`

This directory is ignored by git so local reruns do not dirty the
artifact.
