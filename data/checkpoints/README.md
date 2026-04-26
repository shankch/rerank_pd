# Released Checkpoints

These checkpoints are the released weights used by the default
inference-only capsule run.

They are the final-run checkpoints that back the manuscript numbers used
by the default verification path.

Required files:

- `dit_base_ckpt.pt`
- `nn_hint_ckpt.pt`

Optional training scripts write back into this same directory by
default when run from the full capsule.

The main Code Ocean run loads these checkpoints and does evaluation only;
it does not retrain them.
