#!/usr/bin/env python3
"""
DiT-large-hpwl: larger-capacity DiT with an HPWL-aware auxiliary loss.

Loss = L_eps + lambda_hpwl * L_hpwl_x0
  where L_hpwl_x0 is computed on the x0 prediction (decoded from predicted
  epsilon and the noisy x_t), and penalizes wire length relative to ground
  truth. This pushes the model toward HPWL-optimal placements, not just
  mean-field position estimates.

Saves to dit_large_hpwl_ckpt.pt while keeping the released
DiT-base checkpoint intact.
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).parent.resolve()
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
# Put THIS_DIR FIRST so our local DiT-base definition wins over contest code.
sys.path.insert(0, str(THIS_DIR))
# Evict any stale train_dit_base import.
for _m in list(sys.modules):
    if _m == "train_dit_base":
        del sys.modules[_m]

import lite_dataset as _lds
_lds.decide_download = lambda url: True
from iccad2026_evaluate import get_training_dataloader  # noqa: E402
from train_dit_base import (  # noqa: E402
    DiT, make_schedule, build_features, timestep_embedding,
    MAX_BLOCKS, F_DIM,
)

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 12
N_TIMESTEPS = 100


def smooth_hpwl_from_centroids(cx, cy, b2b_i, b2b_j, b2b_w,
                               p2b_pin_blk, p2b_w, p2b_px, p2b_py, mask):
    """
    Simple HPWL estimator from per-block centroids. Uses weighted L1 distance
    between connected block pairs + pin-to-block. Not the exact contest HPWL
    (which uses bbox per net), but tightly correlated.
    Returns scalar mean HPWL per non-padded block, for gradient flow.
    """
    loss = torch.zeros((), device=cx.device)
    if b2b_i.numel() > 0:
        dx = cx[b2b_i] - cx[b2b_j]
        dy = cy[b2b_i] - cy[b2b_j]
        loss = loss + (b2b_w * (dx.abs() + dy.abs())).sum()
    if p2b_pin_blk.numel() > 0:
        dx = cx[p2b_pin_blk] - p2b_px
        dy = cy[p2b_pin_blk] - p2b_py
        loss = loss + (p2b_w * (dx.abs() + dy.abs())).sum()
    n_valid = (~mask).sum().clamp_min(1).float()
    return loss / n_valid


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    num_train = 1000000
    batch_size = 16  # smaller batch for larger model memory
    dl = get_training_dataloader(batch_size=batch_size, num_samples=num_train)
    print(f"Training DiT-large-hpwl on {num_train} samples, batch {batch_size}", flush=True)
    print(f"Config: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}", flush=True)

    model = DiT(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                feature_dim=F_DIM, out_dim=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    ema_model = DiT(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                    feature_dim=F_DIM, out_dim=4).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_decay = 0.9995

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    n_epochs = 6
    total_steps = (num_train // batch_size) * n_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-5)

    alphas, betas, alpha_bar = make_schedule(N_TIMESTEPS, device)
    ckpt_path = THIS_DIR / "dit_large_hpwl_ckpt.pt"

    # HPWL loss ramped up slowly so the epsilon loss establishes first.
    lambda_hpwl_base = 0.1

    # Optionally resume from a known-good checkpoint.
    resume_step = 0
    resume_candidates = [
        ckpt_path,
        THIS_DIR / "dit_ckpt_v4b.pt",
        THIS_DIR / "dit_ckpt_v4.pt",
    ]
    resume_path = next((p for p in resume_candidates if p.exists()), None)
    if resume_path is not None:
        try:
            prev = torch.load(resume_path, map_location=device, weights_only=False)
            ema_model.load_state_dict(prev["model_state"])
            model.load_state_dict(prev["model_state"])
            resume_step = int(prev.get("step", 0))
            print(f"Resumed DiT-large-hpwl from checkpoint at step {resume_step}", flush=True)
        except Exception as e:
            print(f"Resume failed ({e}), starting fresh.", flush=True)

    step = resume_step
    t0 = time.time()
    for epoch in range(n_epochs):
        for batch in dl:
            try:
                feats, mask, targets, scale = build_features(batch, include_shape_targets=True)
            except Exception as e:
                print(f"feature error: {e}", flush=True)
                continue
            B, N, _ = feats.shape
            feats = feats.to(device)
            mask = mask.to(device)
            x0 = targets.to(device)

            # Sample random timesteps. Bias toward LOW t so x0_hat is stable
            # (x0_hat = (x_t - sqrt(1-ab)*eps)/sqrt(ab) blows up when ab ~ 0,
            # i.e. at high t). Using t < N_TIMESTEPS // 3 for HPWL loss only.
            t = torch.randint(0, N_TIMESTEPS, (B,), device=device)
            a_bar = alpha_bar[t].view(B, 1, 1)
            noise = torch.randn_like(x0)
            x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise

            opt.zero_grad()
            pred = model(feats, x_t, t, key_padding_mask=mask)
            valid = (~mask).float().unsqueeze(-1)
            # Epsilon loss (standard diffusion).
            loss_eps = ((pred - noise) ** 2 * valid).sum() / \
                       (valid.sum().clamp_min(1))

            # HPWL aux loss only on low-noise steps (ab > 0.5) where x0_hat
            # is numerically stable. For high-noise steps skip HPWL.
            loss_hpwl = torch.zeros((), device=device)
            hpwl_count = 0
            low_noise_mask = (a_bar.squeeze(-1).squeeze(-1) > 0.5)  # (B,)
            if low_noise_mask.any():
                # x0 reconstruction from predicted epsilon.
                x0_hat = (x_t - (1 - a_bar).sqrt() * pred) / a_bar.sqrt().clamp_min(0.1)
                # Clamp to sane range so occasional spikes can't poison loss.
                x0_hat = x0_hat.clamp(-5.0, 5.0)
                area_target, b2b_conn, p2b_conn, pins_pos, constraints, \
                    tree, fp_sol, metrics = batch
                for b_idx in range(B):
                    if not bool(low_noise_mask[b_idx].item()):
                        continue
                    nb = int((~mask[b_idx]).sum().item())
                    if nb <= 1:
                        continue
                    cx = x0_hat[b_idx, :nb, 0]
                    cy = x0_hat[b_idx, :nb, 1]
                    tx = x0[b_idx, :nb, 0]
                    ty = x0[b_idx, :nb, 1]
                    b2b = b2b_conn[b_idx]
                    b2b_valid = b2b[b2b[:, 0] >= 0]
                    if b2b_valid.numel() > 0:
                        i = b2b_valid[:, 0].long().to(device)
                        j = b2b_valid[:, 1].long().to(device)
                        w_b = b2b_valid[:, 2].float().to(device)
                        ok = (i >= 0) & (i < nb) & (j >= 0) & (j < nb)
                        i, j, w_b = i[ok], j[ok], w_b[ok]
                        if w_b.numel() > 0:
                            pred_wl = (w_b * ((cx[i] - cx[j]).abs()
                                              + (cy[i] - cy[j]).abs())).sum()
                            gt_wl = (w_b * ((tx[i] - tx[j]).abs()
                                            + (ty[i] - ty[j]).abs())).sum()
                            loss_hpwl = loss_hpwl + (pred_wl - gt_wl).abs()
                            hpwl_count += 1
                if hpwl_count > 0:
                    loss_hpwl = loss_hpwl / hpwl_count
            # Guard against NaN/Inf explosion: skip if abnormal.
            if not torch.isfinite(loss_hpwl):
                loss_hpwl = torch.zeros((), device=device)

            # Ramp HPWL weight linearly from 0 to base over first 10k steps.
            lambda_hpwl = lambda_hpwl_base * min(step / 10000.0, 1.0)
            loss = loss_eps + lambda_hpwl * loss_hpwl

            # Final guard on total loss.
            if not torch.isfinite(loss):
                # Skip this batch — something exploded.
                opt.zero_grad()
                step += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for p, ep in zip(model.parameters(), ema_model.parameters()):
                    ep.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if step % 200 == 0:
                dt = time.time() - t0
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} "
                      f"eps={loss_eps.item():.4f} hpwl={loss_hpwl.item():.4f} "
                      f"lam_hpwl={lambda_hpwl:.3f} t={dt:.1f}s", flush=True)
            if step > 0 and step % 2000 == 0:
                torch.save({
                    "model_state": ema_model.state_dict(),
                    "step": step,
                    "config": {"d_model": D_MODEL, "n_heads": N_HEADS,
                               "n_layers": N_LAYERS, "n_timesteps": N_TIMESTEPS,
                               "f_dim": F_DIM, "out_dim": 4}},
                    ckpt_path)
            step += 1

    torch.save({
        "model_state": ema_model.state_dict(),
        "step": step,
        "config": {"d_model": D_MODEL, "n_heads": N_HEADS,
                   "n_layers": N_LAYERS, "n_timesteps": N_TIMESTEPS,
                   "f_dim": F_DIM, "out_dim": 4}},
        ckpt_path)
    print(f"Saved DiT-large-hpwl checkpoint to {ckpt_path} at step {step}", flush=True)


if __name__ == "__main__":
    main()
