# Gap-Aware Candidate Reranking for Diffusion-Guided Multi-Strategy Floorplanning

Reference implementation and reproduction artifact for the manuscript:

> **Gap-Aware Candidate Reranking for Diffusion-Guided Multi-Strategy Floorplanning**
> Shashank, Independent Researcher, 2026

This package reproduces the headline results:

| Benchmark | `S_100` | Feasible | Mean HPWL gap | Mean `V_rel` |
| --- | --- | --- | --- | --- |
| FloorSet-Lite (rectangular blocks, 100 cases) | **1.40228** | 100/100 | `+0.91` | `0.15` |
| FloorSet-Prime (polygonal blocks, 100 cases) | **1.54862** | 91/100 | `+0.73` | `0.11` |

The same solver and hyperparameters are used on both benchmarks, with
no benchmark-specific retuning.

## Release layout

In the cleaned `submission_final/` artifact, checkpoints are expected
to live outside the code package:

```text
submission_final/
|-- models/
|   |-- dit_base_ckpt.pt
|   `-- nn_hint_ckpt.pt
`-- project_code/
    `-- gap_aware_reranking_floorplanning/
```

The solver auto-detects that layout. It searches for checkpoints in
this order:

1. `CHECKPOINT_DIR`
2. package-local `checkpoints/`
3. top-level `../../models/`
4. package-local `models/`
5. `./models/`

So you can either use the full `submission_final/` bundle with local
weights present, or use a code-only mirror and place checkpoints
locally after download.

## Public model names

This release uses descriptive public names rather than older internal
experiment labels:

- `DiT-base` = released 29M-parameter diffusion prior
- `DiT-large-hpwl` = larger 57.8M-parameter negative-result variant
  trained with an HPWL auxiliary loss
- `NN-hint` = regression network that provides centroid hints for the
  classical candidate generators

The corresponding training scripts and released checkpoint names are:

- `src/train_dit_base.py` -> `dit_base_ckpt.pt`
- `src/train_dit_large_hpwl.py` -> `dit_large_hpwl_ckpt.pt`
- `src/train_nn_hint.py` -> `nn_hint_ckpt.pt`

For convenience, the solver also accepts the older local names
`dit_ckpt.pt` and `nn_ckpt.pt`.

## Repository layout

```text
gap_aware_reranking_floorplanning/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitattributes
|-- .gitignore
|-- paper/
|   |-- paper.pdf
|   |-- paper.tex
|   |-- ieeeaccess.cls
|   `-- IEEEtran.cls
|-- src/
|   |-- solver.py
|   |-- dreamplace.py
|   |-- sequence_pair.py
|   |-- train_dit_base.py
|   |-- train_dit_large_hpwl.py
|   `-- train_nn_hint.py
|-- evaluation/
|   |-- quick_eval.py
|   `-- quick_eval_prime.py
|-- checkpoints/
|   `-- README.md
|-- results/
|   |-- experiment_summary.csv
|   |-- per_case_metrics.csv
|   |-- runtime_evolution.csv
|   `-- hpwl_vrel_scatter_data.csv
`-- scripts/
    |-- run_percentile_sweep.sh
    `-- run_loo_ablation.sh
```

## Quick start

### 1. Requirements

- Python 3.9 or 3.10
- CUDA-capable GPU, tested on NVIDIA RTX 3090 24 GB
- About 15 GB disk space for checkpoints and the FloorSet dataset
- Linux or Windows with a bash-compatible shell

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Clone the FloorSet dataset repository

This repo does not ship the FloorSet dataset. Obtain it separately from
Intel Labs:

```bash
git clone https://github.com/IntelLabs/FloorSet.git
cd FloorSet
pip install -r requirements.txt
```

Suggested directory layout:

```text
your_workspace/
|-- FloorSet/
`-- submission_final/
    |-- models/
    `-- project_code/gap_aware_reranking_floorplanning/
```

If your layout differs, set `FLOORSET_ROOT` before running any script:

```bash
export FLOORSET_ROOT=/path/to/FloorSet
```

### 3. Checkpoints

Two released checkpoints are required for direct reproduction:

- `../../models/dit_base_ckpt.pt` or an equivalent local override
- `../../models/nn_hint_ckpt.pt` or an equivalent local override

The optional negative-result checkpoint
`dit_large_hpwl_ckpt.pt` is not required for the headline results.

In the lightweight GitHub mirror, the `.pt` files are intentionally not
committed. Keep the filenames and checksums from `../../models/`, then
download or archive the binaries separately.

If you choose to store the checkpoints in GitHub anyway, enable Git LFS
first because `dit_base_ckpt.pt` exceeds GitHub's normal 100 MB
per-file limit:

```bash
git lfs install
git clone <repo-url>
```

If you use Zenodo or another archival host instead, upload the files
with the same names listed in `../../models/SHA256SUMS.txt`.

### 4. Run the headline evaluation

FloorSet-Lite:

```bash
cd project_code/gap_aware_reranking_floorplanning
python evaluation/quick_eval.py --full
```

Expected headline result:

```text
FULL_SCORE:    1.40228
FEASIBLE:      100/100
MEAN_RT:       41.1
MEAN_HPWL_GAP: +0.910
MEAN_AREA_GAP: +0.446
MEAN_V_REL:    0.152
```

FloorSet-Prime:

```bash
python evaluation/quick_eval_prime.py --full
```

Expected headline result:

```text
FULL_SCORE:    1.54862
FEASIBLE:      91/100
MEAN_HPWL_GAP: +0.730
MEAN_V_REL:    0.108
```

For a faster sanity check on the 15-case subset:

```bash
python evaluation/quick_eval.py
python evaluation/quick_eval_prime.py
```

## Ablation and sensitivity studies

### Rerank percentile sweep

This reranks candidates using several percentile baselines to confirm
that `p = 0.30` is the best operating point on the short subset:

```bash
bash scripts/run_percentile_sweep.sh
```

You can also set the percentile directly:

```bash
RERANK_PCT=0.25 python evaluation/quick_eval.py
```

### Leave-one-out candidate-generator ablation

This disables each candidate generator in turn:

```bash
bash scripts/run_loo_ablation.sh
```

Or disable generators directly:

```bash
ABLATE_DISABLE="dit-direct,dit-abacus" python evaluation/quick_eval.py
```

## Training

Training is not required to reproduce the headline results. The
released checkpoints are sufficient.

The training scripts currently write checkpoints into `src/`; after
training, move the resulting files either into a local `checkpoints/`
folder, a local `models/` folder, or the top-level
`submission_final/models/` directory.

### Train the NN-hint model

```bash
python src/train_nn_hint.py
```

### Train the DiT-base prior

```bash
python src/train_dit_base.py
```

### Train the DiT-large-hpwl negative-result variant

```bash
python src/train_dit_large_hpwl.py
```

## Rerank summary

The solver enumerates up to seven heterogeneous candidate layouts:
shelf, skyline, SP+SA, DREAMPlace-lite, DiT-direct, DiT-abacus, and
DiT-cluster. For each feasible candidate it records raw HPWL, raw
bounding-box area, and the exact soft-violation count used by the
contest evaluator. The final selector uses percentile-anchored
normalization plus the exponential violation penalty that matches the
benchmark metric.

The core implementation lives in `src/solver.py`.

## Reproducibility fingerprint

All paper results were produced with:

- PyTorch 2.0.x on CUDA 11.x
- NVIDIA RTX 3090, 24 GB VRAM
- single-GPU execution
- fixed rerank percentile `p = 0.30`
- tiered DiT sampling law `K(n) = 3, 2, 1, 0` for
  `n <= 40, 40 < n <= 60, 60 < n <= 90, n > 90`
- cosine diffusion schedule with `T = 100` timesteps

## Citation

If you use this code or the released checkpoints before journal
acceptance, cite the manuscript as an unpublished work rather than as a
published IEEE Access article:

```bibtex
@unpublished{DiffGuideFL2026,
  title  = {Gap-Aware Candidate Reranking for Diffusion-Guided Multi-Strategy Floorplanning},
  author = {Shashank},
  note   = {Manuscript under review},
  year   = {2026}
}
```

Please also cite the FloorSet benchmark:

```bibtex
@techreport{floorset2024,
  title       = {FloorSet: A VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs},
  author      = {Garg, Uday and Kundu, Shreyasi and Markov, Igor L. and others},
  institution = {Intel Labs},
  year        = {2024},
  note        = {\url{https://huggingface.co/datasets/IntelLabs/FloorSet}}
}
```

## License

MIT. See `LICENSE`.

Third-party components retain their original licenses:

- FloorSet dataset and loaders: see the upstream
  `IntelLabs/FloorSet` license
- IEEE class files under `paper/`: included only for reproducibility of
  the manuscript sources

## Contact

Author: Shashank `<sshashan@alumni.usc.edu>`

Please open a GitHub issue for reproduction problems; use email for
research discussion or collaboration.
