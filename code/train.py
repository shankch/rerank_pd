#!/usr/bin/env python3
"""
Optional training wrapper for full reproduction from scratch.

This is not part of the default Code Ocean run.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def _resolve_code_root() -> Path:
    env_root = os.environ.get("CODE_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        THIS_DIR.parent,  # local repo layout
        THIS_DIR,         # Code Ocean layout
        Path.cwd(),
        Path.cwd().parent,
    ])
    rel = Path("project_code") / "gap_aware_reranking_floorplanning"
    for candidate in candidates:
        if (candidate / rel).exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project_code/gap_aware_reranking_floorplanning relative "
        f"to {THIS_DIR} or the current working directory."
    )


ROOT = _resolve_code_root()
PROJECT_ROOT = ROOT / "project_code" / "gap_aware_reranking_floorplanning"
SRC_DIR = PROJECT_ROOT / "src"


def _default_checkpoint_dir() -> Path:
    codeocean_results = Path("/results/checkpoints")
    if os.name != "nt" and codeocean_results.parent.exists():
        return codeocean_results
    return ROOT / "data" / "checkpoints"


DEFAULT_CHECKPOINT_DIR = _default_checkpoint_dir()

TRAIN_TARGETS = {
    "nn-hint": {
        "script": SRC_DIR / "train_nn_hint.py",
        "eta": "~6h on a single RTX 3090",
    },
    "dit-base": {
        "script": SRC_DIR / "train_dit_base.py",
        "eta": "~11h on a single RTX 3090",
    },
    "dit-large-hpwl": {
        "script": SRC_DIR / "train_dit_large_hpwl.py",
        "eta": "longer GPU run; negative-result variant only",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional training wrapper")
    parser.add_argument(
        "--target",
        required=True,
        choices=sorted(TRAIN_TARGETS.keys()),
        help="Training target to reproduce from scratch",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Where trained checkpoints should be written",
    )
    args = parser.parse_args()

    target = TRAIN_TARGETS[args.target]
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Optional full reproduction run")
    print(f"Target: {args.target}")
    print(f"Expected runtime: {target['eta']}")
    print(f"Checkpoint output dir: {ckpt_dir}")
    print("This training path is not run by default in the Code Ocean capsule.")
    print("The released final-run checkpoints used by the default run are already present in data/checkpoints/.")

    env = os.environ.copy()
    env["CHECKPOINT_DIR"] = str(ckpt_dir)

    subprocess.run(
        [sys.executable, str(target["script"])],
        cwd=str(PROJECT_ROOT),
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
