"""
DREAMPlace-lite analytical placement + abacus-style cell legalization.

This module exposes `place_blocks_analytical(...)` which:
  1. Starts from initial block centers (e.g. from DiT/NN/shelf pack).
  2. Runs Nesterov-momentum gradient descent on block centers with:
       - Weighted half-perimeter wirelength (smooth, log-sum-exp approx)
       - Gaussian-density overlap penalty (ePlace-style bell-shaped)
       - Anchor loss for preplaced / fixed-position blocks
  3. Legalizes with an abacus-style row packer that minimizes displacement
     from the analytical positions.

All tensors are torch; uses CUDA if available. The legalizer returns
(x, y) per block with no overlaps and every preplaced block at its exact
target position.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Analytical placement (Nesterov-DREAMPlace-lite)
# ---------------------------------------------------------------------------

def smooth_hpwl(
    cx: torch.Tensor, cy: torch.Tensor,
    b2b_i: torch.Tensor, b2b_j: torch.Tensor, b2b_w: torch.Tensor,
    p2b_blk: torch.Tensor, p2b_w: torch.Tensor, p2b_px: torch.Tensor, p2b_py: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    """
    Weighted half-perimeter wirelength using log-sum-exp smoothing.

    For each net k with blocks/pins {x_a}, HPWL_k ≈ gamma*(log sum exp(x_a/gamma) +
    log sum exp(-x_a/gamma)), which → (max x_a - min x_a) as gamma → 0.
    Since our "nets" here are pairs (b2b) or (pin, block) (p2b), this reduces
    to the sum of weighted |Δx| + |Δy|, smoothed by log(e^a + e^b) ≈ max.
    """
    loss = torch.tensor(0.0, device=cx.device, dtype=cx.dtype)
    if b2b_i.numel() > 0:
        dx = cx[b2b_i] - cx[b2b_j]
        dy = cy[b2b_i] - cy[b2b_j]
        # log(exp(a/gamma) + exp(-a/gamma)) * gamma ≈ |a| for small gamma
        loss = loss + (b2b_w * gamma * (torch.logaddexp(dx / gamma, -dx / gamma)
                                         + torch.logaddexp(dy / gamma, -dy / gamma))).sum()
    if p2b_blk.numel() > 0:
        dx = cx[p2b_blk] - p2b_px
        dy = cy[p2b_blk] - p2b_py
        loss = loss + (p2b_w * gamma * (torch.logaddexp(dx / gamma, -dx / gamma)
                                         + torch.logaddexp(dy / gamma, -dy / gamma))).sum()
    return loss


def gaussian_density_penalty(
    cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, h: torch.Tensor,
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """
    Pair-wise Gaussian repulsion (ePlace-style). Each pair (i, j) contributes
    exp(-(Δx^2 + Δy^2) / σ_ij^2) * area_i * area_j — penalizes blocks too close
    together. σ_ij = (w_i + w_j + h_i + h_j) / 4 * sigma_scale.
    """
    n = cx.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=cx.device, dtype=cx.dtype)
    # Pairwise differences via broadcasting.
    dx = cx.unsqueeze(1) - cx.unsqueeze(0)  # (n, n)
    dy = cy.unsqueeze(1) - cy.unsqueeze(0)
    # σ per pair ≈ average block half-size.
    sigma = ((w.unsqueeze(1) + w.unsqueeze(0)) / 2
             + (h.unsqueeze(1) + h.unsqueeze(0)) / 2) * 0.5 * sigma_scale
    sigma = sigma.clamp_min(1.0)
    r2 = (dx * dx + dy * dy) / (sigma * sigma)
    # Mass per block ~ w*h. Zero the diagonal.
    mass = (w * h).clamp_min(1.0)
    pair_mass = mass.unsqueeze(1) * mass.unsqueeze(0)
    pen = torch.exp(-r2) * pair_mass
    # Remove self-pairs.
    pen = pen - torch.diag(torch.diag(pen))
    return pen.sum() * 0.5  # each pair counted twice


def overlap_penalty(
    cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, h: torch.Tensor,
) -> torch.Tensor:
    """
    Hard AABB overlap area, differentiable via ReLU clamping.
    Penalizes any actual overlap strongly.
    """
    n = cx.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=cx.device, dtype=cx.dtype)
    x1 = cx - w / 2; x2 = cx + w / 2
    y1 = cy - h / 2; y2 = cy + h / 2
    ii, jj = torch.triu_indices(n, n, offset=1, device=cx.device)
    ox = torch.relu(torch.min(x2[ii], x2[jj]) - torch.max(x1[ii], x1[jj]))
    oy = torch.relu(torch.min(y2[ii], y2[jj]) - torch.max(y1[ii], y1[jj]))
    return (ox * oy).sum()


def analytical_place(
    init_positions: List[Tuple[float, float, float, float]],
    block_count: int,
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor],
    n_steps: int = 300,
    lr: float = 0.8,
    gamma_schedule: Tuple[float, float] = (8.0, 0.3),
    density_weight_schedule: Tuple[float, float] = (0.5, 50.0),
    overlap_weight: float = 1e4,
    device: str = None,
) -> List[Tuple[float, float, float, float]]:
    """
    Run gradient descent on block centers (Nesterov-accelerated Adam).

    Returns the raw analytical positions (possibly overlapping); caller
    should legalize with `abacus_legalize`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = block_count
    init_cx = torch.tensor([p[0] + p[2] / 2 for p in init_positions], dtype=torch.float32, device=device)
    init_cy = torch.tensor([p[1] + p[3] / 2 for p in init_positions], dtype=torch.float32, device=device)
    w = torch.tensor([p[2] for p in init_positions], dtype=torch.float32, device=device)
    h = torch.tensor([p[3] for p in init_positions], dtype=torch.float32, device=device)

    is_pinned = torch.zeros(n, dtype=torch.bool, device=device)
    if constraints is not None and constraints.dim() > 1 and constraints.shape[1] >= 2:
        # Preplaced hard-pinned at exact centroid.
        is_pinned = (constraints[:n, 1] != 0).to(device)
    # Also treat boundary-code blocks as pinned — their positions are driven
    # by the legalizer's bbox-edge enforcement, not the analytical objective.
    if constraints is not None and constraints.dim() > 1 and constraints.shape[1] >= 5:
        has_bnd = (constraints[:n, 4] != 0).to(device)
        is_pinned = is_pinned | has_bnd

    # Flatten connectivity to tensor arrays on device.
    def _filter_edges(conn, is_p2b):
        if conn is None or conn.numel() == 0:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.float32, device=device))
        valid = conn[:, 0] >= 0
        c = conn[valid]
        a = c[:, 0].long()
        b = c[:, 1].long()
        ww = c[:, 2].float()
        if is_p2b:
            ok = (b < n) & (a < pins_pos.shape[0]) & (a >= 0) & (b >= 0)
        else:
            ok = (a < n) & (b < n) & (a >= 0) & (b >= 0)
        return a[ok].to(device), b[ok].to(device), ww[ok].to(device)

    b2b_i, b2b_j, b2b_w = _filter_edges(b2b_conn, is_p2b=False)
    p2b_pin, p2b_blk, p2b_w = _filter_edges(p2b_conn, is_p2b=True)
    if p2b_pin.numel() > 0:
        pins = pins_pos.float().to(device)
        p2b_px = pins[p2b_pin, 0]
        p2b_py = pins[p2b_pin, 1]
    else:
        p2b_px = p2b_py = torch.empty(0, device=device)

    cx = init_cx.clone().requires_grad_(True)
    cy = init_cy.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([cx, cy], lr=lr, betas=(0.9, 0.999))

    gamma0, gamma1 = gamma_schedule
    d0, d1 = density_weight_schedule

    for step in range(n_steps):
        t = step / max(n_steps - 1, 1)
        gamma = gamma0 * (gamma1 / gamma0) ** t  # exponential anneal
        density_weight = d0 + (d1 - d0) * t

        optimizer.zero_grad()
        hpwl = smooth_hpwl(cx, cy, b2b_i, b2b_j, b2b_w,
                           p2b_blk, p2b_w, p2b_px, p2b_py, gamma=gamma)
        density = gaussian_density_penalty(cx, cy, w, h)
        overlap = overlap_penalty(cx, cy, w, h)
        loss = hpwl + density_weight * density + overlap_weight * overlap
        loss.backward()
        # Mask gradients on pinned blocks.
        with torch.no_grad():
            if is_pinned.any():
                cx.grad[is_pinned] = 0.0
                cy.grad[is_pinned] = 0.0
        torch.nn.utils.clip_grad_norm_([cx, cy], max_norm=10.0)
        optimizer.step()
        # Re-pin preplaced hard.
        with torch.no_grad():
            if is_pinned.any():
                cx.data[is_pinned] = init_cx[is_pinned]
                cy.data[is_pinned] = init_cy[is_pinned]

    cx_f = cx.detach().cpu().numpy()
    cy_f = cy.detach().cpu().numpy()
    w_f = w.cpu().numpy(); h_f = h.cpu().numpy()
    return [(float(cx_f[i] - w_f[i] / 2), float(cy_f[i] - h_f[i] / 2),
             float(w_f[i]), float(h_f[i])) for i in range(n)]


# ---------------------------------------------------------------------------
# Abacus-style row legalization (displacement-minimizing)
# ---------------------------------------------------------------------------

class _AbacusCluster:
    """A horizontal cluster inside a row — used by abacus legalization."""
    __slots__ = ("xc", "wc", "qc", "ec", "members")

    def __init__(self):
        self.xc = 0.0  # left-x of cluster
        self.wc = 0.0  # total width
        self.qc = 0.0  # running q
        self.ec = 0.0  # effective weight (number of members)
        self.members: List[Tuple[int, float, float]] = []  # (block_id, target_x, width)

    def add_block(self, block_id: int, target_x: float, width: float, weight: float = 1.0):
        self.members.append((block_id, target_x, width))
        self.ec += weight
        self.qc += weight * (target_x - self.wc)
        self.wc += width
        self.xc = self.qc / self.ec


def abacus_row(
    targets: List[Tuple[int, float, float]],  # (block_id, target_x, width)
    x_min: float,
    x_max: float,
) -> Dict[int, float]:
    """
    Classic Abacus 1D legalization for a single row.
    Given blocks with (target_x, width), place them left-aligned in the
    row [x_min, x_max] with no overlap, minimizing sum |x_i - target_x_i|.
    Returns {block_id: legal_x}.
    Standard textbook algorithm (Spindler, Schlichtmann 2008).
    """
    # Sort by target x.
    sorted_bs = sorted(targets, key=lambda t: t[1])
    clusters: List[_AbacusCluster] = []

    for (bid, tx, bw) in sorted_bs:
        last = clusters[-1] if clusters else None
        if last is None or last.xc + last.wc <= tx:
            c = _AbacusCluster()
            c.xc = max(tx, x_min)
            c.add_block(bid, tx, bw)
            clusters.append(c)
        else:
            last.add_block(bid, tx, bw)
            # Collapse with previous clusters if they overlap.
            while (len(clusters) >= 2
                   and clusters[-2].xc + clusters[-2].wc > clusters[-1].xc):
                prev = clusters[-2]
                cur = clusters[-1]
                for (bid2, tx2, bw2) in cur.members:
                    prev.add_block(bid2, tx2, bw2)
                clusters.pop()
        # Respect left boundary.
        if clusters[-1].xc < x_min:
            clusters[-1].xc = x_min
        # Respect right boundary: shift cluster left if it overflows x_max.
        if clusters[-1].xc + clusters[-1].wc > x_max:
            clusters[-1].xc = x_max - clusters[-1].wc

    # Final cleanup: re-shift right-overflowing clusters leftward (chain).
    for c in reversed(clusters):
        if c.xc + c.wc > x_max:
            c.xc = x_max - c.wc
    for i in range(1, len(clusters)):
        if clusters[i].xc < clusters[i - 1].xc + clusters[i - 1].wc:
            clusters[i].xc = clusters[i - 1].xc + clusters[i - 1].wc
    # Enforce left boundary.
    for c in clusters:
        if c.xc < x_min:
            c.xc = x_min

    result: Dict[int, float] = {}
    for c in clusters:
        x_cur = c.xc
        for (bid, tx, bw) in c.members:
            result[bid] = x_cur
            x_cur += bw
    return result


def force_relax_legalize(
    analytical_positions: List[Tuple[float, float, float, float]],
    block_count: int,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor],
    max_iters: int = 2000,
    step_size: float = 0.5,
    step_decay: float = 0.9995,
) -> Optional[List[Tuple[float, float, float, float]]]:
    """
    Force-directed relaxation: compute per-block net force from all overlapping
    neighbors, apply simultaneously (Jacobi iteration) with decaying step size.

    This handles dense overlapping starts (e.g. raw diffusion predictions)
    much better than pair-wise nudging since it lets all blocks move
    together, not one-pair-at-a-time.
    """
    import numpy as np
    n = block_count
    eps = 1e-6
    pos = np.array(analytical_positions, dtype=np.float64)

    is_pinned = np.zeros(n, dtype=bool)
    if constraints is not None and target_positions is not None \
       and constraints.dim() > 1 and constraints.shape[1] >= 2:
        for i in range(n):
            if constraints[i, 1] != 0:
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    pos[i] = [tx, ty, tw, th]
                    is_pinned[i] = True

    ii, jj = np.triu_indices(n, k=1)
    # Precompute pairs' sum-widths and sum-heights (fixed for this call).
    wi_arr = pos[:, 2]
    hi_arr = pos[:, 3]
    # For each pair (i, j), the min non-overlap distance in x is (wi+wj)/2
    # (center-to-center). We'll use pair-centroid forces.
    sigma_x = (wi_arr[ii] + wi_arr[jj]) / 2.0  # min separation along x
    sigma_y = (hi_arr[ii] + hi_arr[jj]) / 2.0

    s = step_size
    for it in range(max_iters):
        # Centroids.
        cx = pos[:, 0] + wi_arr / 2
        cy = pos[:, 1] + hi_arr / 2
        dx = cx[ii] - cx[jj]
        dy = cy[ii] - cy[jj]
        # Overlap magnitudes (positive = overlapping).
        ox = np.maximum(sigma_x - np.abs(dx), 0.0)
        oy = np.maximum(sigma_y - np.abs(dy), 0.0)
        # Only count as overlap if BOTH axes overlap.
        overlap = (ox > eps) & (oy > eps)
        if not overlap.any():
            break
        # Resolve along the smaller overlap axis, per-pair.
        resolve_x = ox <= oy
        # Force magnitude = overlap on chosen axis; direction = sign(dx or dy).
        fx = np.where(resolve_x & overlap,
                      np.sign(dx) * ox, 0.0)
        fy = np.where(~resolve_x & overlap,
                      np.sign(dy) * oy, 0.0)
        # Apply to block i (positive) and j (negative); Jacobi accumulation.
        sum_fx = np.zeros(n, dtype=np.float64)
        sum_fy = np.zeros(n, dtype=np.float64)
        np.add.at(sum_fx, ii, fx)
        np.add.at(sum_fx, jj, -fx)
        np.add.at(sum_fy, ii, fy)
        np.add.at(sum_fy, jj, -fy)
        # Don't move pinned blocks.
        sum_fx[is_pinned] = 0.0
        sum_fy[is_pinned] = 0.0
        # Scale and apply.
        pos[:, 0] += s * sum_fx
        pos[:, 1] += s * sum_fy
        # Clamp to non-negative.
        pos[~is_pinned, 0] = np.maximum(pos[~is_pinned, 0], 0.0)
        pos[~is_pinned, 1] = np.maximum(pos[~is_pinned, 1], 0.0)
        s *= step_decay

    # Residual overlap check.
    cx = pos[:, 0] + wi_arr / 2
    cy = pos[:, 1] + hi_arr / 2
    dx = cx[ii] - cx[jj]
    dy = cy[ii] - cy[jj]
    ox = np.maximum(sigma_x - np.abs(dx), 0.0)
    oy = np.maximum(sigma_y - np.abs(dy), 0.0)
    if ((ox > eps) & (oy > eps)).any():
        # Fall back to nudge to clean up residual.
        return None
    return [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in pos]


def robust_legalize(
    analytical_positions: List[Tuple[float, float, float, float]],
    block_count: int,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor],
) -> Optional[List[Tuple[float, float, float, float]]]:
    """
    Three-stage legalizer for dense/overlapping starts:
      1. Pre-expand: scale centroids outward from COM so overlaps shrink.
      2. Force-relax: Jacobi iterations to remove residual overlaps.
      3. Nudge: final pair-wise cleanup.
    Tries multiple expansion factors; returns first feasible.
    """
    import numpy as np
    pos0 = np.array(analytical_positions, dtype=np.float64)
    n = block_count
    if n <= 1:
        return analytical_positions

    # Compute bbox and center of mass of centroids.
    cx0 = pos0[:, 0] + pos0[:, 2] / 2
    cy0 = pos0[:, 1] + pos0[:, 3] / 2
    com_x = cx0.mean(); com_y = cy0.mean()

    # Preserved preplaced positions.
    is_pinned = np.zeros(n, dtype=bool)
    if constraints is not None and target_positions is not None \
       and constraints.dim() > 1 and constraints.shape[1] >= 2:
        for i in range(n):
            if constraints[i, 1] != 0:
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                if tx >= 0 and ty >= 0:
                    is_pinned[i] = True

    # Iteration budget scales down for large n (O(n^2) per iter).
    if n > 100:
        fr_configs = [(0.5, 800), (0.3, 1500)]
        nd_iter = 1000
        expand_factors = (1.0, 1.5, 2.2)
    elif n > 60:
        fr_configs = [(0.5, 1500), (0.3, 3000)]
        nd_iter = 1500
        expand_factors = (1.0, 1.3, 1.7, 2.2)
    else:
        fr_configs = [(0.5, 3000), (0.3, 6000)]
        nd_iter = 2000
        expand_factors = (1.0, 1.3, 1.7, 2.2)

    for expand_factor in expand_factors:
        # Expand non-pinned centroids outward from COM.
        pos = pos0.copy()
        if expand_factor > 1.0:
            for i in range(n):
                if is_pinned[i]:
                    continue
                new_cx = com_x + (cx0[i] - com_x) * expand_factor
                new_cy = com_y + (cy0[i] - com_y) * expand_factor
                pos[i, 0] = new_cx - pos[i, 2] / 2
                pos[i, 1] = new_cy - pos[i, 3] / 2
            pos[~is_pinned, 0] = np.maximum(pos[~is_pinned, 0], 0.0)
            pos[~is_pinned, 1] = np.maximum(pos[~is_pinned, 1], 0.0)
        expanded = [(float(p[0]), float(p[1]), float(p[2]), float(p[3]))
                    for p in pos]
        for step_size, max_iters in fr_configs:
            result = force_relax_legalize(
                expanded, block_count, constraints, target_positions,
                max_iters=max_iters, step_size=step_size)
            if result is not None:
                return result
        nudged = nudge_legalize(expanded, block_count, constraints,
                                target_positions, max_iters=nd_iter)
        if nudged is not None:
            return nudged
    return None


def nudge_legalize(
    analytical_positions: List[Tuple[float, float, float, float]],
    block_count: int,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor],
    max_iters: int = 500,
) -> Optional[List[Tuple[float, float, float, float]]]:
    """
    Vectorized overlap-resolution legalizer: iteratively push overlapping
    pairs apart along the shorter overlap axis. Preplaced are pinned.
    """
    import numpy as np
    n = block_count
    eps = 1e-6
    pos = np.array(analytical_positions, dtype=np.float64)  # (n, 4)

    is_pinned = np.zeros(n, dtype=bool)
    if constraints is not None and target_positions is not None \
       and constraints.dim() > 1 and constraints.shape[1] >= 2:
        for i in range(n):
            if constraints[i, 1] != 0:
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    pos[i] = [tx, ty, tw, th]
                    is_pinned[i] = True

    # Precompute upper-triangle pair indices once.
    ii, jj = np.triu_indices(n, k=1)

    for _ in range(max_iters):
        xi = pos[ii, 0]; yi = pos[ii, 1]; wi = pos[ii, 2]; hi = pos[ii, 3]
        xj = pos[jj, 0]; yj = pos[jj, 1]; wj = pos[jj, 2]; hj = pos[jj, 3]
        ox = np.minimum(xi + wi, xj + wj) - np.maximum(xi, xj)
        oy = np.minimum(yi + hi, yj + hj) - np.maximum(yi, yj)
        overlap_mask = (ox > eps) & (oy > eps)
        if not overlap_mask.any():
            break
        # Direction: positive shift for i (if i to left of j, shift i left).
        ci_x = xi + wi / 2
        cj_x = xj + wj / 2
        ci_y = yi + hi / 2
        cj_y = yj + hj / 2
        i_left = ci_x < cj_x
        i_below = ci_y < cj_y
        # For each overlapping pair, pick axis with smaller overlap.
        shorter_x = ox <= oy  # if True, resolve along x
        # Per-pair shift vectors.
        shift_x_i = np.where(i_left, -ox / 2, ox / 2)
        shift_x_j = -shift_x_i
        shift_y_i = np.where(i_below, -oy / 2, oy / 2)
        shift_y_j = -shift_y_i
        # Zero out the non-chosen axis.
        shift_x_i = np.where(overlap_mask & shorter_x, shift_x_i, 0.0)
        shift_x_j = np.where(overlap_mask & shorter_x, shift_x_j, 0.0)
        shift_y_i = np.where(overlap_mask & ~shorter_x, shift_y_i, 0.0)
        shift_y_j = np.where(overlap_mask & ~shorter_x, shift_y_j, 0.0)

        # If one side is pinned, transfer both sides' shift to the mobile one.
        pin_i = is_pinned[ii]
        pin_j = is_pinned[jj]
        # Both pinned: zero out.
        both_pin = pin_i & pin_j
        shift_x_i = np.where(both_pin, 0.0, shift_x_i)
        shift_x_j = np.where(both_pin, 0.0, shift_x_j)
        shift_y_i = np.where(both_pin, 0.0, shift_y_i)
        shift_y_j = np.where(both_pin, 0.0, shift_y_j)
        # i pinned → double-shift j.
        shift_x_j = np.where(pin_i & ~pin_j, shift_x_j - shift_x_i, shift_x_j)
        shift_y_j = np.where(pin_i & ~pin_j, shift_y_j - shift_y_i, shift_y_j)
        shift_x_i = np.where(pin_i & ~pin_j, 0.0, shift_x_i)
        shift_y_i = np.where(pin_i & ~pin_j, 0.0, shift_y_i)
        # j pinned → double-shift i.
        shift_x_i = np.where(pin_j & ~pin_i, shift_x_i - shift_x_j, shift_x_i)
        shift_y_i = np.where(pin_j & ~pin_i, shift_y_i - shift_y_j, shift_y_i)
        shift_x_j = np.where(pin_j & ~pin_i, 0.0, shift_x_j)
        shift_y_j = np.where(pin_j & ~pin_i, 0.0, shift_y_j)

        # Accumulate per-block shifts (scatter-add).
        dx = np.zeros(n, dtype=np.float64)
        dy = np.zeros(n, dtype=np.float64)
        np.add.at(dx, ii, shift_x_i)
        np.add.at(dx, jj, shift_x_j)
        np.add.at(dy, ii, shift_y_i)
        np.add.at(dy, jj, shift_y_j)
        # Don't move pinned blocks (safety).
        dx[is_pinned] = 0.0
        dy[is_pinned] = 0.0
        pos[:, 0] += dx
        pos[:, 1] += dy
        # Clamp non-pinned to non-negative.
        pos[~is_pinned, 0] = np.maximum(pos[~is_pinned, 0], 0.0)
        pos[~is_pinned, 1] = np.maximum(pos[~is_pinned, 1], 0.0)

    # Final feasibility check.
    xi = pos[ii, 0]; yi = pos[ii, 1]; wi = pos[ii, 2]; hi = pos[ii, 3]
    xj = pos[jj, 0]; yj = pos[jj, 1]; wj = pos[jj, 2]; hj = pos[jj, 3]
    ox = np.minimum(xi + wi, xj + wj) - np.maximum(xi, xj)
    oy = np.minimum(yi + hi, yj + hj) - np.maximum(yi, yj)
    if ((ox > eps) & (oy > eps)).any():
        return None  # could not resolve overlaps
    return [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in pos]


def abacus_legalize(
    analytical_positions: List[Tuple[float, float, float, float]],
    block_count: int,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor],
    bbox_width: float,
) -> Optional[List[Tuple[float, float, float, float]]]:
    """
    Full 2D legalization:
      1. Assign each block to a row whose y is closest to its analytical y.
      2. Use per-row abacus to legalize x-positions.
      3. Sort rows by y; stack them to eliminate y-overlap (rows may have
         varying heights since macros are not standard cells).

    Preplaced blocks are pinned at their target (x, y); other blocks in the
    same row are placed around them.

    Row assignment strategy (approximate — variable-height rows are hard):
      - Sort mobile blocks by analytical y.
      - Greedily form rows filling to ~bbox_width; each row's height is the
        max block height in it.
      - Preplaced blocks create "floating" y-positions; if a row's y-range
        crosses a preplaced y-range AND x-overlaps, we split the row.

    For simplicity we first legalize assuming no preplaced, then push
    overlapping mobile blocks away from preplaced.
    """
    n = block_count
    preplaced: Dict[int, Tuple[float, float, float, float]] = {}
    if constraints is not None and target_positions is not None:
        pp_col = constraints[:n, 1] if constraints.dim() > 1 and constraints.shape[1] >= 2 \
                 else torch.zeros(n)
        for i in range(n):
            if pp_col[i] == 0:
                continue
            tx = float(target_positions[i, 0])
            ty = float(target_positions[i, 1])
            tw = float(target_positions[i, 2])
            th = float(target_positions[i, 3])
            if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                preplaced[i] = (tx, ty, tw, th)
    pp_ids = set(preplaced.keys())

    mobile = [(i, analytical_positions[i]) for i in range(n) if i not in pp_ids]
    mobile.sort(key=lambda t: t[1][1])  # by analytical y

    # Fit mobile blocks into greedy rows of width bbox_width.
    rows: List[List[Tuple[int, float, float, float, float]]] = []  # (bid, x, y, w, h)
    cur_row: List[Tuple[int, float, float, float, float]] = []
    cur_row_w = 0.0
    for (i, (x, y, w, h)) in mobile:
        if cur_row and cur_row_w + w > bbox_width + 1e-6:
            rows.append(cur_row)
            cur_row = []
            cur_row_w = 0.0
        cur_row.append((i, x, y, w, h))
        cur_row_w += w
    if cur_row:
        rows.append(cur_row)

    # Place preplaced blocks first; their y remains fixed.
    positions: List[Optional[Tuple[float, float, float, float]]] = [None] * n
    for i, xywh in preplaced.items():
        positions[i] = xywh

    # Stack rows vertically, bumping past preplaced that overlap.
    y_cursor = 0.0
    for row in rows:
        row_h = max(r[4] for r in row) if row else 0.0
        # Abacus legalization of x-positions inside the row.
        row_targets = [(r[0], r[1], r[3]) for r in row]
        legal_x = abacus_row(row_targets, x_min=0.0, x_max=bbox_width)
        # Bump y_cursor past any preplaced that overlaps in y AND x.
        while True:
            bumped = False
            for (px, py, pw, ph) in preplaced.values():
                # x-overlap: rows span [0, bbox_width], so overlap iff pp is inside.
                if py + ph <= y_cursor + 1e-9 or py >= y_cursor + row_h - 1e-9:
                    continue
                # y-range overlaps; need x-check.
                # Find the block whose x-range overlaps pp.
                overlap_x = False
                for (bid, tx, bw) in row_targets:
                    bx = legal_x[bid]
                    if bx < px + pw and bx + bw > px:
                        overlap_x = True
                        break
                if overlap_x:
                    new_y = py + ph
                    if new_y > y_cursor + 1e-9:
                        y_cursor = new_y
                        bumped = True
                        break
            if not bumped:
                break
        # Assign positions.
        for (bid, tx, bw) in row_targets:
            _, _, _, _, bh = next((r for r in row if r[0] == bid), (None, 0, 0, 0, 0))
            positions[bid] = (legal_x[bid], y_cursor, bw, bh)
        y_cursor += row_h

    # Sanity: all blocks placed.
    for i in range(n):
        if positions[i] is None:
            return None
    return positions  # type: ignore[return-value]
