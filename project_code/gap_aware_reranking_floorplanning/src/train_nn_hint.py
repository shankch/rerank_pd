#!/usr/bin/env python3
"""
Train the NN-hint model to predict block centroids from (features,
connectivity, pins). Outputs positions used as target hints for the legalizer.

Architecture:
  - Per-block features: [area, fixed, preplaced, mib_id, cluster_id, bound_code,
    agg_b2b_weight, agg_p2b_weight, pin_target_x, pin_target_y]
  - Encoder: MLP → tokens, then self-attention across blocks (permutation inv).
  - Decoder: MLP → (cx, cy) normalized to [0, 1].

Training:
  - Loss: L1 on normalized centroids (position error) + small aspect loss.
  - Data from iccad2026_evaluate.get_training_dataloader.
"""
from __future__ import annotations

import math
import random
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).parent.resolve()
PACKAGE_ROOT = THIS_DIR.parent
SUBMISSION_ROOT = PACKAGE_ROOT.parent.parent
import os as _os
_env = _os.environ.get("FLOORSET_ROOT")
_candidates = ([Path(_env)] if _env else []) + [
    THIS_DIR.parent.parent / "FloorSet",
    THIS_DIR.parent / "FloorSet",
    Path.cwd() / "FloorSet",
]
FLOORSET_ROOT = next((p for p in _candidates if p.exists()), _candidates[0])
CONTEST_DIR = FLOORSET_ROOT / "iccad2026contest"
sys.path.insert(0, str(FLOORSET_ROOT))
sys.path.insert(0, str(CONTEST_DIR))

# Auto-accept data download.
import lite_dataset as _lds
_lds.decide_download = lambda url: True

from iccad2026_evaluate import get_training_dataloader  # noqa: E402

MAX_BLOCKS = 120
F_DIM = 12  # per-block feature count
HIDDEN = 256
TRAIN_SEED = 42


def _set_global_seed(seed: int = TRAIN_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _default_checkpoint_dir() -> Path:
    ckpt_env = _os.environ.get("CHECKPOINT_DIR")
    if ckpt_env:
        path = Path(ckpt_env)
    else:
        capsule_dir = SUBMISSION_ROOT / "data" / "checkpoints"
        path = capsule_dir if (SUBMISSION_ROOT / "code").exists() else PACKAGE_ROOT / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


class BlockEncoder(nn.Module):
    def __init__(self, hidden: int = HIDDEN, nhead: int = 4, nlayers: int = 3,
                 out_dim: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(F_DIM, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=4 * hidden,
            dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feats: (B, N, F), mask: (B, N) where True = padding.
        x = self.input_proj(feats)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.out_proj(x)  # (B, N, 2)


def build_features(batch):
    """Convert a training batch into per-block feature tensor + targets."""
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree, fp_sol, metrics = batch
    # Squeeze the batch dim if it's 1.
    B = area_target.shape[0]
    N_max = area_target.shape[1]

    feats = torch.zeros(B, N_max, F_DIM)
    mask = torch.zeros(B, N_max, dtype=torch.bool)  # True = padding
    tgt_cx = torch.zeros(B, N_max)
    tgt_cy = torch.zeros(B, N_max)

    for b in range(B):
        n = int((area_target[b] != -1).sum().item())
        if n == 0:
            mask[b] = True
            continue
        # Block count.
        mask[b, n:] = True
        # Per-block base features.
        feats[b, :n, 0] = area_target[b, :n].float()
        feats[b, :n, 1] = constraints[b, :n, 0].float()  # fixed
        feats[b, :n, 2] = constraints[b, :n, 1].float()  # preplaced
        feats[b, :n, 3] = constraints[b, :n, 2].float()  # mib_id
        feats[b, :n, 4] = constraints[b, :n, 3].float()  # cluster_id
        feats[b, :n, 5] = constraints[b, :n, 4].float()  # boundary code

        # Aggregate p2b: for each block, sum weights & weighted pin position.
        p2b = p2b_conn[b]
        p2b_valid = p2b[p2b[:, 0] >= 0]
        if p2b_valid.numel() > 0:
            pin = p2b_valid[:, 0].long(); blk = p2b_valid[:, 1].long(); w = p2b_valid[:, 2].float()
            ok = (blk >= 0) & (blk < n) & (pin >= 0) & (pin < pins_pos.shape[1])
            pin, blk, w = pin[ok], blk[ok], w[ok]
            if w.numel() > 0:
                px = pins_pos[b, pin, 0].float(); py = pins_pos[b, pin, 1].float()
                agg_w = torch.zeros(n); agg_px = torch.zeros(n); agg_py = torch.zeros(n)
                agg_w.scatter_add_(0, blk, w)
                agg_px.scatter_add_(0, blk, w * px)
                agg_py.scatter_add_(0, blk, w * py)
                feats[b, :n, 6] = agg_w
                feats[b, :n, 7] = agg_px / (agg_w + 1e-6)
                feats[b, :n, 8] = agg_py / (agg_w + 1e-6)

        # Aggregate b2b: for each block, sum connection weight to other blocks.
        b2b = b2b_conn[b]
        b2b_valid = b2b[b2b[:, 0] >= 0]
        if b2b_valid.numel() > 0:
            i = b2b_valid[:, 0].long(); j = b2b_valid[:, 1].long(); w = b2b_valid[:, 2].float()
            ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            i, j, w = i[ok], j[ok], w[ok]
            if w.numel() > 0:
                agg_w = torch.zeros(n)
                agg_w.scatter_add_(0, i, w)
                agg_w.scatter_add_(0, j, w)
                feats[b, :n, 9] = agg_w

        # Block count feature (normalized).
        feats[b, :n, 10] = n / MAX_BLOCKS
        feats[b, :n, 11] = math.sqrt(n) / math.sqrt(MAX_BLOCKS)

        # Targets: centroid from ground-truth floorplan (w, h, x, y in fp_sol).
        fp = fp_sol[b]
        for i in range(n):
            w_i = float(fp[i, 0]); h_i = float(fp[i, 1])
            x_i = float(fp[i, 2]); y_i = float(fp[i, 3])
            tgt_cx[b, i] = x_i + w_i / 2
            tgt_cy[b, i] = y_i + h_i / 2

    # Normalize features and targets by per-sample bbox.
    # Use sqrt of sum(areas) as length scale.
    with torch.no_grad():
        areas_pos = feats[:, :, 0].clone()
        areas_pos[mask] = 0.0
        total_area = areas_pos.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = torch.sqrt(total_area).unsqueeze(-1)
        # Normalize area feature.
        feats[:, :, 0] = feats[:, :, 0] / (scale.squeeze(-1) + 1e-6)
        feats[:, :, 7] = feats[:, :, 7] / (scale.squeeze(-1) + 1e-6)
        feats[:, :, 8] = feats[:, :, 8] / (scale.squeeze(-1) + 1e-6)
        tgt_cx = tgt_cx / (scale.squeeze(-1) + 1e-6)
        tgt_cy = tgt_cy / (scale.squeeze(-1) + 1e-6)

    return feats, mask, tgt_cx, tgt_cy, scale.squeeze()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _set_global_seed(TRAIN_SEED)
    print(f"Device: {device}", flush=True)
    print(f"Fixed training seed: {TRAIN_SEED}", flush=True)
    print("Loading training dataloader (may download ~6 GB if first time)...", flush=True)
    num_train = 1000000
    batch_size = 32
    dl = get_training_dataloader(batch_size=batch_size, num_samples=num_train)
    print(f"Training on {num_train} samples, batch size {batch_size}", flush=True)

    model = BlockEncoder(hidden=HIDDEN, nhead=8, nlayers=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100000, eta_min=1e-5)
    n_epochs = 3

    step = 0
    t0 = time.time()
    ckpt_path = _default_checkpoint_dir() / "nn_hint_ckpt.pt"
    try:
        for epoch in range(n_epochs):
            for batch in dl:
                try:
                    feats, mask, tgt_cx, tgt_cy, scale = build_features(batch)
                except Exception as e:
                    print(f"batch build error: {e}", flush=True)
                    continue
                feats = feats.to(device); mask = mask.to(device)
                tgt_cx = tgt_cx.to(device); tgt_cy = tgt_cy.to(device)

                opt.zero_grad()
                pred = model(feats, mask)
                pred_cx = pred[:, :, 0]; pred_cy = pred[:, :, 1]
                loss_cx = (torch.abs(pred_cx - tgt_cx) * (~mask).float()).sum() / (~mask).sum().clamp_min(1)
                loss_cy = (torch.abs(pred_cy - tgt_cy) * (~mask).float()).sum() / (~mask).sum().clamp_min(1)
                loss = loss_cx + loss_cy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

                if step % 100 == 0:
                    dt = time.time() - t0
                    print(f"epoch={epoch} step={step} loss={loss.item():.4f} "
                          f"cx={loss_cx.item():.4f} cy={loss_cy.item():.4f} t={dt:.1f}s", flush=True)
                if step > 0 and step % 500 == 0:
                    torch.save({"model_state": model.state_dict(),
                                "config": {"hidden": HIDDEN, "f_dim": F_DIM,
                                           "nlayers": 6, "nhead": 8}},
                               ckpt_path)
                step += 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Training interrupted at step {step}: {e}", flush=True)

    torch.save({"model_state": model.state_dict(),
                "config": {"hidden": HIDDEN, "f_dim": F_DIM, "nlayers": 6, "nhead": 8}},
               ckpt_path)
    print(f"Saved checkpoint to {ckpt_path} at step {step}", flush=True)


if __name__ == "__main__":
    main()
