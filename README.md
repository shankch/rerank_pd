# Final Submission Package

This folder contains the final paper bundle plus the single cleaned
code path needed to reproduce the reported results.

Older draft code, archive snapshots, and parallel legacy repos were
removed on purpose so that a researcher can immediately find the
relevant artifact.

## What To Use

Use only:

`project_code/gap_aware_reranking_floorplanning/`

That package contains:

- the final solver source
- evaluation scripts for FloorSet-Lite and FloorSet-Prime
- the result CSVs used in the paper
- the ablation and sensitivity scripts
- the detailed reproduction README

The code-only repository layout expects model weights to be provided
separately via:

`models/`

In the local full artifact bundle, this folder can contain the released
checkpoints directly. In a public code-only GitHub mirror, it is kept as
a placeholder with instructions, checksums, and filenames so the repo
stays lightweight.

## Package Layout

- `paper.tex`, `paper.pdf`, `ieeeaccess.cls`, `IEEEtran.cls`, and the
  logo assets are the final journal-submission paper files.
- `supplementary/` contains the raw CSV tables and figure exports used
  to support the manuscript.
- `models/` contains the checkpoint manifest and placement
  instructions. The actual `.pt` files may be hosted separately.
- `project_code/gap_aware_reranking_floorplanning/` is the only code
  package that should be used for reproduction.

## Quick Reproduction Path

1. Place the released checkpoint files into `models/` if you are using
   the full local artifact, or download them separately if you are
   using a code-only GitHub mirror.
2. Change into
   `project_code/gap_aware_reranking_floorplanning/`.
3. Follow the detailed instructions in
   `project_code/gap_aware_reranking_floorplanning/README.md`.
4. Clone the public FloorSet dataset repository and set
   `FLOORSET_ROOT` if your directory layout differs from the default.
5. Run:

```bash
python evaluation/quick_eval.py --full
python evaluation/quick_eval_prime.py --full
```

Expected headline metrics:

- FloorSet-Lite: `S_100 = 1.40228`, `100/100` feasible
- FloorSet-Prime: `S_100 = 1.54862`, `91/100` feasible

## Notes

- The solver searches for checkpoints in this order:
  `CHECKPOINT_DIR`, package-local `checkpoints/`, top-level `models/`,
  package-local `models/`, then `./models/`.
- The top-level paper files are kept because this folder is both the
  final submission bundle and the accompanying reproducibility package.
- Public mirrors should describe this as an artifact accompanying a
  manuscript under review, not as an IEEE Access publication unless the
  paper has already been accepted.
