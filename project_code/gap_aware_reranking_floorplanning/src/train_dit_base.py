#!/usr/bin/env python3
"""
DiT-base prior for block placement.

Input: noisy normalized (cx, cy) positions + timestep + conditioning.
  Conditioning = per-block features (area, constraints) + aggregated
  connectivity stats + pin-target estimate.

Output: predicted noise epsilon for each block's (cx, cy).

Training: forward-diffuse ground-truth positions, learn to predict noise.
Inference: iterative denoising K steps from N(0, I).

Architecture:
  * Token = [noisy_cx, noisy_cy, time_emb(t), block_features] → linear → d_model
  * Stack of pre-norm transformer blocks with adaLN modulation from time emb
  * Output head → (eps_cx, eps_cy) per block

Based on DiT (Peebles & Xie 2022) adapted for variable-length set of blocks.
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

import lite_dataset as _lds
_lds.decide_download = lambda url: True
from iccad2026_evaluate import get_training_dataloader  # noqa: E402

MAX_BLOCKS = 120
F_DIM = 12  # per-block raw features
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 10
N_TIMESTEPS = 100  # diffusion steps (both training and inference)
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


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings. t shape (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero modulation from timestep/condition."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model))
        self.adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))
        # Zero-init the modulation output so the block starts as identity.
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, N, d_model), c: (B, d_model), mask: (B, N) True=pad.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaln(c).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        y = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(y)
        return x


class DiT(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, feature_dim: int = F_DIM,
                 out_dim: int = 4):
        super().__init__()
        self.out_dim = out_dim  # 2 = (cx,cy); 4 = (cx,cy,log_aspect_w,log_aspect_h)
        # Project block features + noisy positions into the latent space.
        self.feat_proj = nn.Linear(feature_dim, d_model)
        self.pos_proj = nn.Linear(out_dim, d_model)  # noisy targets
        # Condition from timestep (shared across blocks).
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        self.head = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor, noisy_pos: torch.Tensor,
                t: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # feats: (B, N, F_DIM), noisy_pos: (B, N, 2), t: (B,), mask: (B, N).
        d_model = self.feat_proj.out_features
        t_emb = timestep_embedding(t, d_model)
        c = self.time_embed(t_emb)  # (B, d_model)
        x = self.feat_proj(feats) + self.pos_proj(noisy_pos)
        for blk in self.blocks:
            x = blk(x, c, key_padding_mask=key_padding_mask)
        shift, scale = self.final_adaln(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(x)  # (B, N, 2)


# --- Diffusion schedule (linear beta) ------------------------------------

def make_schedule(n_steps: int = N_TIMESTEPS, device: str = "cpu"):
    """Simple cosine schedule."""
    s = 0.008
    t = torch.linspace(0, n_steps, n_steps + 1, device=device, dtype=torch.float64)
    f = torch.cos((t / n_steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 0.999)
    alphas = 1 - betas
    return alphas.float(), betas.float(), alpha_bar[1:].float()


# --- Feature extraction (same as train_nn.py) ----------------------------

def build_features(batch, include_shape_targets: bool = True):
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree, fp_sol, metrics = batch
    B = area_target.shape[0]
    N_max = area_target.shape[1]
    feats = torch.zeros(B, N_max, F_DIM)
    mask = torch.zeros(B, N_max, dtype=torch.bool)  # True = padding
    tgt_cx = torch.zeros(B, N_max)
    tgt_cy = torch.zeros(B, N_max)
    tgt_logw = torch.zeros(B, N_max)  # log(w / sqrt(area)) — aspect signal
    tgt_logh = torch.zeros(B, N_max)  # log(h / sqrt(area))

    for b in range(B):
        n = int((area_target[b] != -1).sum().item())
        if n == 0:
            mask[b] = True
            continue
        mask[b, n:] = True
        feats[b, :n, 0] = area_target[b, :n].float()
        feats[b, :n, 1] = constraints[b, :n, 0].float()
        feats[b, :n, 2] = constraints[b, :n, 1].float()
        feats[b, :n, 3] = constraints[b, :n, 2].float()
        feats[b, :n, 4] = constraints[b, :n, 3].float()
        feats[b, :n, 5] = constraints[b, :n, 4].float()

        p2b = p2b_conn[b]
        valid = p2b[p2b[:, 0] >= 0]
        if valid.numel() > 0:
            pin = valid[:, 0].long(); blk = valid[:, 1].long(); w = valid[:, 2].float()
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

        b2b = b2b_conn[b]
        valid = b2b[b2b[:, 0] >= 0]
        if valid.numel() > 0:
            i = valid[:, 0].long(); j = valid[:, 1].long(); w = valid[:, 2].float()
            ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            i, j, w = i[ok], j[ok], w[ok]
            if w.numel() > 0:
                agg_w = torch.zeros(n)
                agg_w.scatter_add_(0, i, w)
                agg_w.scatter_add_(0, j, w)
                feats[b, :n, 9] = agg_w

        feats[b, :n, 10] = n / MAX_BLOCKS
        feats[b, :n, 11] = math.sqrt(n) / math.sqrt(MAX_BLOCKS)

        fp = fp_sol[b]
        for i in range(n):
            w_i = float(fp[i, 0]); h_i = float(fp[i, 1])
            x_i = float(fp[i, 2]); y_i = float(fp[i, 3])
            tgt_cx[b, i] = x_i + w_i / 2
            tgt_cy[b, i] = y_i + h_i / 2
            a_i = max(float(area_target[b, i]), 1.0)
            tgt_logw[b, i] = math.log(max(w_i, 0.1) / math.sqrt(a_i))
            tgt_logh[b, i] = math.log(max(h_i, 0.1) / math.sqrt(a_i))

    with torch.no_grad():
        areas_pos = feats[:, :, 0].clone()
        areas_pos[mask] = 0.0
        total_area = areas_pos.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = torch.sqrt(total_area).unsqueeze(-1)
        feats[:, :, 0] = feats[:, :, 0] / (scale.squeeze(-1) + 1e-6)
        feats[:, :, 7] = feats[:, :, 7] / (scale.squeeze(-1) + 1e-6)
        feats[:, :, 8] = feats[:, :, 8] / (scale.squeeze(-1) + 1e-6)
        tgt_cx = tgt_cx / (scale.squeeze(-1) + 1e-6)
        tgt_cy = tgt_cy / (scale.squeeze(-1) + 1e-6)

    if include_shape_targets:
        # Stack all 4 targets: (cx, cy, log_aspect_w, log_aspect_h).
        targets = torch.stack([tgt_cx, tgt_cy, tgt_logw, tgt_logh], dim=-1)  # (B, N, 4)
        return feats, mask, targets, scale.squeeze()
    return feats, mask, tgt_cx, tgt_cy, scale.squeeze()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _set_global_seed(TRAIN_SEED)
    print(f"Device: {device}", flush=True)
    print(f"Fixed training seed: {TRAIN_SEED}", flush=True)
    num_train = 1000000
    batch_size = 20
    dl = get_training_dataloader(batch_size=batch_size, num_samples=num_train)
    print(f"Training DiT on {num_train} samples, batch {batch_size}", flush=True)

    model = DiT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    # EMA for smoother predictions.
    ema_model = DiT().to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_decay = 0.9995

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    n_epochs = 6
    total_steps = (num_train // batch_size) * n_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-5)

    alphas, betas, alpha_bar = make_schedule(N_TIMESTEPS, device)
    ckpt_path = _default_checkpoint_dir() / "dit_base_ckpt.pt"

    step = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        for batch in dl:
            try:
                feats, mask, targets, scale = build_features(batch, include_shape_targets=True)
            except Exception as e:
                print(f"feature error: {e}", flush=True)
                continue
            B, N, _ = feats.shape
            feats = feats.to(device); mask = mask.to(device)
            x0 = targets.to(device)  # (B, N, 4): cx, cy, log_aspect_w, log_aspect_h

            # Sample random timesteps.
            t = torch.randint(0, N_TIMESTEPS, (B,), device=device)
            a_bar = alpha_bar[t].view(B, 1, 1)
            noise = torch.randn_like(x0)
            x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise

            # Predict epsilon.
            opt.zero_grad()
            pred = model(feats, x_t, t, key_padding_mask=mask)
            # Masked MSE over non-padding.
            loss = ((pred - noise) ** 2 * (~mask).float().unsqueeze(-1)).sum() / \
                   ((~mask).sum().clamp_min(1) * 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            # EMA update.
            with torch.no_grad():
                for p, ep in zip(model.parameters(), ema_model.parameters()):
                    ep.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if step % 200 == 0:
                dt = time.time() - t0
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} t={dt:.1f}s", flush=True)
            if step > 0 and step % 2000 == 0:
                torch.save({
                    "model_state": ema_model.state_dict(),
                    "config": {"d_model": D_MODEL, "n_heads": N_HEADS,
                               "n_layers": N_LAYERS, "n_timesteps": N_TIMESTEPS,
                               "f_dim": F_DIM, "out_dim": 4}},
                    ckpt_path)
            step += 1

    torch.save({
        "model_state": ema_model.state_dict(),
        "config": {"d_model": D_MODEL, "n_heads": N_HEADS,
                   "n_layers": N_LAYERS, "n_timesteps": N_TIMESTEPS,
                   "f_dim": F_DIM, "out_dim": 4}},
        ckpt_path)
    print(f"Saved DiT checkpoint to {ckpt_path} at step {step}", flush=True)


if __name__ == "__main__":
    main()
