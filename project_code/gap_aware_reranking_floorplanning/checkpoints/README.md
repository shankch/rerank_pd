# Optional Local Checkpoint Folder

The cleaned `submission_final/` artifact stores the released model
weights in the top-level `../../data/checkpoints/` directory.

This folder is kept only as an optional local override location. If you
copy the code package out of the full artifact, you may place
`dit_base_ckpt.pt` and `nn_hint_ckpt.pt` here, or set
`CHECKPOINT_DIR` to another directory.
