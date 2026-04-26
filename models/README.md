# Released Model Checkpoints

This folder is the checkpoint handoff point for the cleaned
`submission_final/` reproduction package.

## Files

- `SHA256SUMS.txt` - checksum manifest for archive verification
- `README.md` - placement and release instructions

The actual `.pt` files may be present in a private/local artifact bundle
or hosted separately in Zenodo, Hugging Face, or a GitHub release. They
are intentionally omitted from the lightweight code-only repository.

## Usage

The solver in
`project_code/gap_aware_reranking_floorplanning/src/solver.py`
automatically searches this folder when run from the full
`submission_final/` bundle.

If you copy only the code package elsewhere, you can instead:

- set `CHECKPOINT_DIR` to a custom folder
- place the checkpoints in a local `checkpoints/` directory
- place the checkpoints in a local `models/` directory

## Public release note

If you upload these files before the paper is accepted, describe them
as a reproducibility artifact accompanying a manuscript under review.
Do not describe them as an IEEE Access publication unless the article
has actually been accepted and published.

## Hosting note

- GitHub requires Git LFS for `dit_base_ckpt.pt` if you decide to store
  the binary there
- Zenodo can host the files directly and is a good option if you want a
  citable archived snapshot

## SHA-256

- `dit_base_ckpt.pt`:
  `6834f811df1e60167441cca47e8b7678d448ae8a32b94f231008c7f0ec5c00c5`
- `nn_hint_ckpt.pt`:
  `e46bddf06ef46164b2458dd0238571c481baf2b038351e61e67167dccdb9036b`
