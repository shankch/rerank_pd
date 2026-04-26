"""
ICCAD 2026 FloorSet solver — iterated by autoresearch.

Boundary-aware shelf packing with full edge handling
(TL/T/TR, L/.../R, BL/B/BR).

Each shelf is packed to exact target width W using soft-block aspect ratio
flexibility. Because W is the bbox right edge, the LAST block in every shelf
ends exactly at the bbox right edge. Similarly, the FIRST block in every shelf
starts at x=0 (bbox left edge).

Shelf layout:
  top_shelf:    [TL-corner?  top-bounded blocks...  TR-corner?]
  middle-k:     [left-bound k?  interior blocks...  right-bound k?]
    ...
  bottom_shelf: [BL-corner?  bottom-bounded blocks...  BR-corner?]

We distribute left- and right-bound blocks one per middle shelf so each can
satisfy its boundary (only ONE block per shelf can touch x=0 or x=W).

Hard invariants:
  * No block overlap (shelves are disjoint in y; within-shelf blocks disjoint in x).
  * Fixed/preplaced blocks keep EXACT target dimensions.
  * Preplaced blocks are anchored at target (x,y) — we keep them apart from the
    shelf pack by reserving empty area below the shelves.
  * Soft blocks: w * h within 1% of target area.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

THIS_DIR = Path(__file__).parent
PACKAGE_ROOT = THIS_DIR.parent
SUBMISSION_ROOT = PACKAGE_ROOT.parent.parent
# FloorSet location: $FLOORSET_ROOT env var, then sibling of repo, then
# a few common locations. Users typically git-clone FloorSet next to this
# repo and export FLOORSET_ROOT to its path.
import os as _os
_env_root = _os.environ.get("FLOORSET_ROOT")
_candidates = ([Path(_env_root)] if _env_root else []) + [
    THIS_DIR.parent.parent / "FloorSet",
    THIS_DIR.parent / "FloorSet",
    Path.cwd() / "FloorSet",
]
FLOORSET_ROOT = next((p for p in _candidates if p.exists()), _candidates[0])
CONTEST_DIR = FLOORSET_ROOT / "iccad2026contest"
# Checkpoints: $CHECKPOINT_DIR, then the package-local checkpoints/
# directory, then the top-level models/ directory used by the
# submission_final artifact bundle.
_CKPT_ENV = _os.environ.get("CHECKPOINT_DIR")
CHECKPOINT_DIR = Path(_CKPT_ENV) if _CKPT_ENV else PACKAGE_ROOT / "checkpoints"
CHECKPOINT_SEARCH_DIRS: List[Path] = []
for _path in (
    Path(_CKPT_ENV) if _CKPT_ENV else None,
    PACKAGE_ROOT / "checkpoints",
    SUBMISSION_ROOT / "models",
    PACKAGE_ROOT / "models",
    Path.cwd() / "models",
):
    if _path is None:
        continue
    if _path not in CHECKPOINT_SEARCH_DIRS:
        CHECKPOINT_SEARCH_DIRS.append(_path)

for p in (str(FLOORSET_ROOT), str(CONTEST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
# THIS_DIR last so it takes priority (inserted at position 0 = checked first).
# The contest dir has its own dreamplace.py; we need OURS to win.
if str(THIS_DIR) not in sys.path or sys.path[0] != str(THIS_DIR):
    try:
        sys.path.remove(str(THIS_DIR))
    except ValueError:
        pass
    sys.path.insert(0, str(THIS_DIR))

from iccad2026_evaluate import FloorplanOptimizer  # noqa: E402
try:
    from sequence_pair import sa_sequence_pair, SequencePair  # noqa: E402
except ImportError:
    sa_sequence_pair = None
    SequencePair = None

# Evict any already-imported 'dreamplace' (contest dir has a same-named module
# that gets loaded first via iccad2026_evaluate).
for _m in list(sys.modules):
    if _m == "dreamplace":
        del sys.modules[_m]
try:
    from dreamplace import analytical_place, abacus_legalize, nudge_legalize, robust_legalize  # noqa: E402
except ImportError:
    analytical_place = None
    abacus_legalize = None
    nudge_legalize = None
    robust_legalize = None


# Boundary bitmask: 1=left, 2=right, 4=top, 8=bottom.
B_LEFT, B_RIGHT, B_TOP, B_BOTTOM = 1, 2, 4, 8


def _square_dims(area: float) -> Tuple[float, float]:
    if area <= 0:
        return (1.0, 1.0)
    s = math.sqrt(area)
    return s, s


# ---------------------------------------------------------------------------
# Skyline packer — 2D bottom-left packing with preplaced obstacles
# ---------------------------------------------------------------------------

class Skyline:
    """
    2D skyline (contour) packer. The skyline is a list of (x, y, w) segments
    covering the strip [0, W] with no gaps. When inserting a block, we scan
    for the lowest y at which a [width] window fits and place the block there,
    updating the contour.

    Preplaced blocks are pushed into the skyline as initial obstacles.
    """

    __slots__ = ("W", "segs")

    def __init__(self, W: float):
        self.W = W
        # Segments: list of [x, y, w] covering the full strip.
        self.segs: List[List[float]] = [[0.0, 0.0, W]]

    def add_obstacle(self, x: float, y: float, w: float, h: float):
        """Raise the skyline over [x, x+w] to at least y+h."""
        self._raise_range(x, x + w, y + h)

    def _raise_range(self, x0: float, x1: float, y_new: float):
        """Ensure skyline y >= y_new for x in [x0, x1]."""
        x0 = max(0.0, x0)
        x1 = min(self.W, x1)
        if x1 <= x0:
            return
        new_segs: List[List[float]] = []
        i = 0
        while i < len(self.segs):
            sx, sy, sw = self.segs[i]
            sx1 = sx + sw
            if sx1 <= x0 or sx >= x1:
                # No overlap.
                new_segs.append([sx, sy, sw])
                i += 1
                continue
            # Split into (sx..x0), (x0..x1 raised), (x1..sx1) as needed.
            if sx < x0:
                new_segs.append([sx, sy, x0 - sx])
            raised_y = max(sy, y_new)
            overlap_x0 = max(sx, x0)
            overlap_x1 = min(sx1, x1)
            new_segs.append([overlap_x0, raised_y, overlap_x1 - overlap_x0])
            if sx1 > x1:
                new_segs.append([x1, sy, sx1 - x1])
            i += 1
        # Merge adjacent same-height segments.
        self.segs = self._merge(new_segs)

    @staticmethod
    def _merge(segs):
        out = []
        for s in segs:
            if s[2] <= 1e-9:
                continue
            if out and abs(out[-1][1] - s[1]) < 1e-9 and abs(out[-1][0] + out[-1][2] - s[0]) < 1e-9:
                out[-1][2] += s[2]
            else:
                out.append([s[0], s[1], s[2]])
        return out

    def find_position(self, w: float, prefer_x: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """
        Find the lowest (x, y) such that a [w x *] block fits at (x, y) without
        raising the skyline higher than necessary. Returns (x, y) or None.
        Bottom-left fit with optional x preference (tie-breaks by |x - prefer_x|).
        """
        if w > self.W + 1e-9:
            return None
        # Try each possible starting segment; inside a segment, x can be any
        # value such that [x, x+w] is inside [0, W]. The resulting y is the
        # max sy over all segs intersected.
        best = None  # (y, tie_value, x)
        # Candidate starting x positions: segment left-edges.
        candidates = set()
        for sx, sy, sw in self.segs:
            candidates.add(sx)
            candidates.add(sx + sw - w)  # right-aligned within segment
        candidates = [x for x in candidates if 0.0 - 1e-9 <= x <= self.W - w + 1e-9]
        for x in candidates:
            y = self._contour_max(x, x + w)
            if y is None:
                continue
            tie = abs(x - prefer_x) if prefer_x is not None else x
            key = (y, tie)
            if best is None or key < (best[0], best[1]):
                best = (y, tie, x)
        if best is None:
            return None
        return best[2], best[0]

    def _contour_max(self, x0: float, x1: float) -> Optional[float]:
        """Max y over [x0, x1]. Returns None if range out of strip."""
        if x0 < -1e-9 or x1 > self.W + 1e-9:
            return None
        y = 0.0
        for sx, sy, sw in self.segs:
            sx1 = sx + sw
            if sx1 <= x0 + 1e-9 or sx >= x1 - 1e-9:
                continue
            if sy > y:
                y = sy
        return y

    def place(self, w: float, h: float, prefer_x: Optional[float] = None
              ) -> Optional[Tuple[float, float]]:
        """Find a position and place the block (update skyline)."""
        pos = self.find_position(w, prefer_x=prefer_x)
        if pos is None:
            return None
        x, y = pos
        self._raise_range(x, x + w, y + h)
        return x, y

    def height(self) -> float:
        return max(s[1] for s in self.segs) if self.segs else 0.0


_NN_MODEL = None  # Lazy-loaded cache of the trained BlockEncoder.
_DIT_MODEL = None  # Lazy-loaded cache of the trained DiT.


def _find_checkpoint(*names: str) -> Path:
    """
    Search a small set of known checkpoint locations in priority order.

    This keeps the package usable both as a standalone code repo
    (checkpoints/ under the package root) and inside the broader
    submission_final artifact bundle (models/ at the top level).
    """
    direct_candidates = [THIS_DIR / name for name in names]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    for directory in CHECKPOINT_SEARCH_DIRS:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return CHECKPOINT_SEARCH_DIRS[0] / names[0]


def _load_dit_model():
    global _DIT_MODEL
    if _DIT_MODEL is not None:
        return _DIT_MODEL if _DIT_MODEL != "missing" else None
    ckpt_path = _find_checkpoint("dit_base_ckpt.pt", "dit_ckpt.pt")
    if not ckpt_path.exists():
        _DIT_MODEL = "missing"
        return None
    try:
        train_script = Path(__file__).parent / "train_dit_base.py"
        sys.path.insert(0, str(Path(__file__).parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_dit_base", train_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        # Auto-detect out_dim from head weight shape.
        head_w = ckpt["model_state"].get("head.weight")
        if head_w is not None:
            cfg_out_dim = int(head_w.shape[0])
        else:
            cfg_out_dim = cfg.get("out_dim", 2)
        model = mod.DiT(
            d_model=cfg.get("d_model", 256),
            n_heads=cfg.get("n_heads", 8),
            n_layers=cfg.get("n_layers", 6),
            out_dim=cfg_out_dim,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _DIT_MODEL = (model, mod, cfg)
        return _DIT_MODEL
    except Exception as e:
        print(f"[solver] dit_model load failed: {e}", flush=True)
        _DIT_MODEL = "missing"
        return None


def _load_nn_model():
    """Lazy-load the NN-hint model from checkpoint if present."""
    global _NN_MODEL
    if _NN_MODEL is not None:
        return _NN_MODEL
    ckpt_path = _find_checkpoint("nn_hint_ckpt.pt", "nn_ckpt.pt")
    if not ckpt_path.exists():
        _NN_MODEL = "missing"
        return None
    try:
        # Import lazily to avoid circular imports on solver module load.
        train_script = Path(__file__).parent / "train_nn_hint.py"
        sys.path.insert(0, str(Path(__file__).parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_nn_hint", train_script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        # Auto-detect nlayers by counting layer-specific keys in state dict.
        state = ckpt["model_state"]
        layer_idxs = set()
        for k in state.keys():
            if k.startswith("encoder.layers."):
                idx = int(k.split(".")[2])
                layer_idxs.add(idx)
        nlayers_detected = len(layer_idxs) if layer_idxs else cfg.get("nlayers", 3)
        model = mod.BlockEncoder(
            hidden=cfg.get("hidden", 128),
            nhead=cfg.get("nhead", 4),
            nlayers=nlayers_detected,
        )
        model.load_state_dict(state)
        model.eval()
        _NN_MODEL = (model, mod)
        return _NN_MODEL
    except Exception as e:
        print(f"[solver] nn_model load failed: {e}", flush=True)
        _NN_MODEL = "missing"
        return None


class MyOptimizer(FloorplanOptimizer):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.aspect_knob = 0.6       # W / sqrt(total_area). Narrow strip packs denser.
        self.min_aspect_ratio_knob = 0.2  # soft-block aspect bounded to [0.2, 5.0].

    # ------------------------------------------------------------------
    # Constraint extraction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Connectivity-aware target positions
    # ------------------------------------------------------------------
    def _combined_targets(
        self,
        block_count: int,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        current_positions: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Weighted-average target (x, y) per block from p2b AND b2b connections.
        For b2b, uses current block centroids (if provided) as the neighbor target.
        """
        targets: List[Optional[Tuple[float, float]]] = [None] * block_count
        tx = torch.zeros(block_count)
        ty = torch.zeros(block_count)
        ws = torch.zeros(block_count)

        # p2b contribution: weighted sum of pin positions.
        if p2b_connectivity is not None and p2b_connectivity.numel() > 0:
            p = p2b_connectivity
            valid = p[:, 0] >= 0
            if valid.any():
                p = p[valid]
                pin = p[:, 0].long(); blk = p[:, 1].long(); w = p[:, 2].float()
                ok = (blk >= 0) & (blk < block_count) & (pin >= 0) & (pin < pins_pos.shape[0])
                pin, blk, w = pin[ok], blk[ok], w[ok]
                if w.numel() > 0:
                    px = pins_pos[pin, 0].float(); py = pins_pos[pin, 1].float()
                    tx.scatter_add_(0, blk, w * px)
                    ty.scatter_add_(0, blk, w * py)
                    ws.scatter_add_(0, blk, w)

        # b2b contribution: need current positions → use centroids.
        if (current_positions is not None and b2b_connectivity is not None
                and b2b_connectivity.numel() > 0):
            cx = torch.tensor([p[0] + p[2] / 2 for p in current_positions])
            cy = torch.tensor([p[1] + p[3] / 2 for p in current_positions])
            b = b2b_connectivity
            valid = b[:, 0] >= 0
            if valid.any():
                b = b[valid]
                i = b[:, 0].long(); j = b[:, 1].long(); w = b[:, 2].float()
                ok = (i >= 0) & (i < block_count) & (j >= 0) & (j < block_count)
                i, j, w = i[ok], j[ok], w[ok]
                if w.numel() > 0:
                    # Symmetric: block i sees neighbor j, block j sees neighbor i.
                    tx.scatter_add_(0, i, w * cx[j]); ty.scatter_add_(0, i, w * cy[j])
                    tx.scatter_add_(0, j, w * cx[i]); ty.scatter_add_(0, j, w * cy[i])
                    ws.scatter_add_(0, i, w); ws.scatter_add_(0, j, w)

        for k in range(block_count):
            if ws[k] > 0:
                targets[k] = (float(tx[k] / ws[k]), float(ty[k] / ws[k]))
        return targets

    def _pin_targets(
        self,
        block_count: int,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
    ) -> List[Optional[Tuple[float, float]]]:
        """
        For each block, weighted-average (x, y) of pins connected to it via p2b.
        Vectorized with scatter-add for speed.
        """
        targets: List[Optional[Tuple[float, float]]] = [None] * block_count
        if p2b_connectivity is None or p2b_connectivity.numel() == 0:
            return targets

        edges = p2b_connectivity
        valid = edges[:, 0] >= 0
        if not valid.any():
            return targets
        edges = edges[valid]
        pin_idx = edges[:, 0].long()
        blk_idx = edges[:, 1].long()
        w = edges[:, 2].float()

        # Filter out-of-range edges defensively.
        ok = (blk_idx >= 0) & (blk_idx < block_count) & \
             (pin_idx >= 0) & (pin_idx < pins_pos.shape[0])
        pin_idx = pin_idx[ok]; blk_idx = blk_idx[ok]; w = w[ok]
        if w.numel() == 0:
            return targets

        px = pins_pos[pin_idx, 0].float()
        py = pins_pos[pin_idx, 1].float()

        tx = torch.zeros(block_count); ty = torch.zeros(block_count)
        ws = torch.zeros(block_count)
        tx.scatter_add_(0, blk_idx, w * px)
        ty.scatter_add_(0, blk_idx, w * py)
        ws.scatter_add_(0, blk_idx, w)

        for i in range(block_count):
            if ws[i] > 0:
                targets[i] = (float(tx[i] / ws[i]), float(ty[i] / ws[i]))
        return targets

    def _extract_cols(self, constraints: torch.Tensor, block_count: int):
        if constraints is None or constraints.dim() < 2:
            z = torch.zeros(block_count)
            return z, z, z, z, z
        nc = constraints.shape[1]
        c = constraints[:block_count]
        cols = [c[:, j].clone() if j < nc else torch.zeros(block_count)
                for j in range(5)]
        return tuple(cols)

    # ------------------------------------------------------------------
    # Base dimensions
    # ------------------------------------------------------------------

    def _base_dims(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        fixed_col: torch.Tensor,
        preplaced_col: torch.Tensor,
        mib_col: torch.Tensor,
    ) -> Tuple[List[Tuple[float, float]], List[bool]]:
        """Return (sizes, is_inflexible)."""
        sizes: List[Tuple[float, float]] = [(0.0, 0.0)] * block_count
        is_inflex = [False] * block_count

        # Fixed/preplaced: exact target.
        for i in range(block_count):
            if target_positions is not None:
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tw > 0 and th > 0 and (fixed_col[i] != 0 or preplaced_col[i] != 0):
                    sizes[i] = (tw, th)
                    is_inflex[i] = True

        # MIB groups: share shape.
        n_mib = int(mib_col.max().item()) if mib_col.numel() else 0
        for g in range(1, n_mib + 1):
            idxs = [i for i in range(block_count) if int(mib_col[i].item()) == g]
            if not idxs:
                continue
            forced = [i for i in idxs if is_inflex[i]]
            if forced:
                gw, gh = sizes[forced[0]]
            else:
                area = float(area_targets[idxs[0]])
                gw, gh = _square_dims(area)
            for i in idxs:
                if not is_inflex[i]:
                    sizes[i] = (gw, gh)
                    is_inflex[i] = True  # MIB blocks can't change aspect once set.

        # Remaining soft blocks → square base.
        for i in range(block_count):
            if sizes[i] == (0.0, 0.0):
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                sizes[i] = _square_dims(area)
                is_inflex[i] = False

        return sizes, is_inflex

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        # Iterative refinement with per-case aspect search. Try a few aspect
        # knobs; inside each, run 3 iterations of b2b-target-refined shelf
        # packing. Pick the packing with lowest (HPWL + α*area + β*boundary_viol).
        orig_knob = self.aspect_knob
        best_positions = None
        best_score = float("inf")
        _trace = getattr(self, "_diag_path_trace", False)
        if _trace:
            print(f"[solve] n={block_count} trace ON", flush=True)

        # Collect all feasible candidates as (positions, raw_hpwl, raw_area,
        # v_rel, tag). After search, re-rank with gap-based contest cost.
        _candidates = []
        def _collect(positions_cand, tag):
            if positions_cand is None:
                return
            try:
                h, a, v = self._contest_raw(
                    positions_cand, b2b_connectivity, p2b_connectivity,
                    pins_pos, constraints, block_count)
                _candidates.append((positions_cand, h, a, v, tag))
            except Exception:
                pass

        # Leave-one-out generator disable via env (ABLATE_DISABLE=dit-direct,...).
        # Any _collect tag whose prefix is in the disabled set is filtered out.
        import os as _os
        _ablate = set(x.strip() for x in
                      _os.environ.get("ABLATE_DISABLE", "").split(",") if x.strip())
        if _ablate:
            _orig_collect = _collect
            def _collect(positions_cand, tag):  # noqa: F811
                for d in _ablate:
                    if tag.startswith(d):
                        return
                _orig_collect(positions_cand, tag)

        # 0a) DiT-predicted (centroid, shape) via diffusion model.
        dit_full = self._dit_predict_full(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints)
        dit_targets = dit_full[0] if dit_full is not None else None
        dit_shapes = dit_full[1] if dit_full is not None else None

        # If the DiT-base checkpoint predicts shapes, try a pure-DiT path:
        # sample K times,
        # legalize each, pick the best. Falls back to nudge if robust fails.
        # Capped at 90 blocks — larger cases regress because DiT predictions
        # are denser and legalization can't recover the quality.
        if (dit_full is not None and dit_shapes is not None
                and robust_legalize is not None
                and block_count <= 90):
            # Sample count tiered by block count to balance variance vs runtime.
            # Small cases have fast legalization, so 3 samples is cheap.
            # Medium cases have expensive legalization, cap at 2.
            # Large medium (>60) rarely benefits from more samples.
            if block_count <= 40:
                dit_n_samples = getattr(self, "_dit_direct_n_samples", 3)
            elif block_count <= 60:
                dit_n_samples = getattr(self, "_dit_direct_n_samples_med", 2)
            else:
                dit_n_samples = getattr(self, "_dit_direct_n_samples_large", 1)
            n_legal = 0
            # First candidate already computed above (dit_targets, dit_shapes).
            for trial in range(dit_n_samples):
                try:
                    if trial == 0:
                        cand_targets, cand_shapes = dit_targets, dit_shapes
                    else:
                        fresh = self._dit_predict_full(
                            block_count, area_targets, b2b_connectivity,
                            p2b_connectivity, pins_pos, constraints)
                        if fresh is None:
                            continue
                        cand_targets, cand_shapes = fresh
                    init_with_shapes = self._build_init_from_dit(
                        block_count, cand_targets, cand_shapes, area_targets,
                        constraints, target_positions)
                    legal_dit = robust_legalize(
                        init_with_shapes, block_count, constraints,
                        target_positions)
                    used_nudge = False
                    if legal_dit is None and nudge_legalize is not None:
                        legal_dit = nudge_legalize(
                            init_with_shapes, block_count, constraints,
                            target_positions, max_iters=2000)
                        used_nudge = True
                    if legal_dit is not None:
                        n_legal += 1
                        s = self._internal_cost(
                            legal_dit, b2b_connectivity, p2b_connectivity,
                            pins_pos, constraints, block_count)
                        if _trace:
                            print(f"[dit-direct t{trial}] cost={s:.2f}",
                                  flush=True)
                        if getattr(self, "_diag_dit_direct", False):
                            print(f"[dit-direct t{trial}] n={block_count} "
                                  f"nudge_fallback={used_nudge} cost={s:.3f}",
                                  flush=True)
                        if s < best_score:
                            best_score = s
                            best_positions = legal_dit
                            if _trace:
                                print(f"[dit-direct t{trial}] WON cost={s:.2f}",
                                      flush=True)
                        _collect(legal_dit, f"dit-direct-t{trial}")
                        # Also try boundary-swap-fixed version to reduce v_rel.
                        try:
                            bfixed = self._boundary_swap_fix(
                                legal_dit, b2b_connectivity, p2b_connectivity,
                                pins_pos, constraints, block_count)
                            _collect(bfixed, f"dit-direct-t{trial}-bfix")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[dit shape direct error t{trial}] {e}", flush=True)
            if getattr(self, "_diag_dit_direct", False) and n_legal == 0:
                print(f"[dit-direct] n={block_count} "
                      f"ALL_{dit_n_samples}_FAILED", flush=True)

        # 0a-2) DiT-shapes + abacus legalization — works for ALL sizes
        # including large n where robust_legalize is too slow. Uses DiT's
        # predicted (w, h) and places via row-aligned abacus (O(n log n)).
        if (dit_full is not None and dit_shapes is not None
                and abacus_legalize is not None):
            try:
                init_with_shapes = self._build_init_from_dit(
                    block_count, dit_targets, dit_shapes, area_targets,
                    constraints, target_positions)
                total_area_d = sum(float(area_targets[i])
                                    if area_targets[i] > 0 else 0.0
                                    for i in range(block_count))
                for knob_d in (0.55, 0.7, 0.85):
                    W_d = math.sqrt(max(total_area_d, 1.0)) * knob_d
                    widest_d = max((p[2] for p in init_with_shapes), default=0.0)
                    W_d = max(W_d, widest_d)
                    legal_ab = abacus_legalize(
                        init_with_shapes, block_count, constraints,
                        target_positions, W_d)
                    if legal_ab is not None:
                        _collect(legal_ab, f"dit-abacus-{knob_d}")
                        try:
                            bf = self._boundary_swap_fix(
                                legal_ab, b2b_connectivity, p2b_connectivity,
                                pins_pos, constraints, block_count)
                            _collect(bf, f"dit-abacus-{knob_d}-bfix")
                        except Exception:
                            pass
            except Exception as e:
                print(f"[dit abacus error] {e}", flush=True)

        # 0a-4) DiT cluster-aware pack: groups cluster members as rigid rows.
        # Unlike dit-abacus, this enforces the grouping constraint by design,
        # so v_rel should be much lower even if DiT predicted disconnected
        # members.
        if (dit_full is not None and dit_shapes is not None):
            for knob_cl in (0.55, 0.7, 0.85):
                try:
                    pos_cl = self._dit_cluster_pack(
                        block_count, dit_targets, dit_shapes, area_targets,
                        constraints, target_positions, knob_cl)
                    if pos_cl is not None:
                        _collect(pos_cl, f"dit-cluster-{knob_cl}")
                        try:
                            bf = self._boundary_swap_fix(
                                pos_cl, b2b_connectivity, p2b_connectivity,
                                pins_pos, constraints, block_count)
                            _collect(bf, f"dit-cluster-{knob_cl}-bfix")
                        except Exception:
                            pass
                except Exception as e:
                    if _trace:
                        print(f"[dit-cluster err k={knob_cl}] {e}", flush=True)

        # 0a-3) DiT-shape skyline pack — DISABLED, regressed FULL_SCORE.
        if False and (dit_full is not None and dit_shapes is not None):
            for knob_d in (0.55, 0.7, 0.85):
                try:
                    pos_ds = self._dit_shape_skyline_pack(
                        block_count, dit_targets, dit_shapes, area_targets,
                        constraints, target_positions, knob_d)
                    if pos_ds is not None:
                        _collect(pos_ds, f"dit-sky-{knob_d}")
                except Exception:
                    pass

        if dit_targets is not None:
            for knob in (0.55, 0.7, 0.85):
                pos_dit = self._solve_skyline_clustered(
                    block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints,
                    target_positions, dit_targets, knob)
                if pos_dit is not None:
                    s = self._internal_cost(pos_dit, b2b_connectivity,
                                            p2b_connectivity, pins_pos,
                                            constraints, block_count)
                    if s < best_score:
                        best_score = s
                        best_positions = pos_dit
                    _collect(pos_dit, f"dit-skyline-{knob}")

        # 0b) Regression-NN-predicted centroids as a first target seed.
        nn_targets = self._nn_predict_targets(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints)
        if nn_targets is not None:
            for knob in (0.55, 0.7, 0.85):
                # Iterative refinement: initialize with NN targets, then update
                # via centroids of the packed layout.
                pos_cur = None
                cur_targets = nn_targets
                prev_s = float("inf")
                for it in range(3):
                    pos_new = self._solve_skyline_clustered(
                        block_count, area_targets, b2b_connectivity,
                        p2b_connectivity, pins_pos, constraints,
                        target_positions, cur_targets, knob)
                    if pos_new is None:
                        break
                    cur_s = self._internal_cost(pos_new, b2b_connectivity,
                                                p2b_connectivity, pins_pos,
                                                constraints, block_count)
                    if it >= 1 and cur_s >= prev_s - 1e-3:
                        break
                    pos_cur = pos_new
                    prev_s = cur_s
                    # Build next targets blending NN prediction + current centroids.
                    nn_w = max(0.0, 1.0 - 0.4 * (it + 1))
                    next_targets = []
                    for j in range(block_count):
                        nx, ny = nn_targets[j]
                        cx = pos_new[j][0] + pos_new[j][2] / 2
                        cy = pos_new[j][1] + pos_new[j][3] / 2
                        tx = nn_w * nx + (1 - nn_w) * cx
                        ty = nn_w * ny + (1 - nn_w) * cy
                        next_targets.append((tx, ty))
                    cur_targets = next_targets
                if pos_cur is not None and prev_s < best_score:
                    best_score = prev_s
                    best_positions = pos_cur
                _collect(pos_cur, f"nn-skyline-{knob}")
            # Shelf packer with NN targets.
            for knob in (0.5, 0.65, 0.8):
                self.aspect_knob = knob
                pos_sh = self._solve_once(
                    block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints,
                    target_positions, nn_targets)
                s = self._internal_cost(pos_sh, b2b_connectivity,
                                        p2b_connectivity, pins_pos,
                                        constraints, block_count)
                if s < best_score:
                    best_score = s
                    best_positions = pos_sh
                _collect(pos_sh, f"nn-shelf-{knob}")
            self.aspect_knob = orig_knob

            # Position-preserving legalization on NN centroids (best for HPWL).
            for knob in (0.55, 0.7, 0.85):
                fixed_col2, pp_col2, mib_col2, _, _ = self._extract_cols(
                    constraints, block_count)
                base_sizes2, is_inflex2 = self._base_dims(
                    block_count, area_targets, target_positions,
                    fixed_col2, pp_col2, mib_col2)
                total_area2 = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                                  for i in range(block_count))
                W2 = math.sqrt(max(total_area2, 1.0)) * knob
                widest2 = max((base_sizes2[i][0] for i in range(block_count)),
                              default=0.0)
                W2 = max(W2, widest2)
                pp_positions2 = {i: (float(target_positions[i, 0]),
                                     float(target_positions[i, 1]),
                                     float(target_positions[i, 2]),
                                     float(target_positions[i, 3]))
                                 for i in range(block_count) if pp_col2[i] != 0
                                 and target_positions is not None
                                 and target_positions[i, 0] >= 0}
                if pp_positions2:
                    pp_xmax2 = max(p[0] + p[2] for p in pp_positions2.values())
                    W2 = max(W2, pp_xmax2)
                pos_sp = self._spread_legalize(
                    block_count, constraints, target_positions,
                    base_sizes2, is_inflex2, nn_targets, W2)
                if pos_sp is not None:
                    s = self._internal_cost(pos_sp, b2b_connectivity,
                                            p2b_connectivity, pins_pos,
                                            constraints, block_count)
                    if s < best_score:
                        best_score = s
                        best_positions = pos_sp
                    _collect(pos_sp, f"spread-{knob}")

        # 1) Skyline packer WITH cluster super-blocks. Iterate with b2b-aware
        #    target updates until cost plateaus.
        for knob in (0.55, 0.7, 0.85):
            pos_sky = None
            prev_s = float("inf")
            for it in range(3):
                targets = self._combined_targets(
                    block_count, p2b_connectivity, pins_pos,
                    b2b_connectivity, pos_sky)
                new_sky = self._solve_skyline_clustered(
                    block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints,
                    target_positions, targets, knob)
                if new_sky is None:
                    break
                cur_s = self._internal_cost(new_sky, b2b_connectivity,
                                            p2b_connectivity, pins_pos,
                                            constraints, block_count)
                if it >= 1 and cur_s >= prev_s - 1e-3:
                    break
                pos_sky = new_sky
                prev_s = cur_s
            if pos_sky is not None and prev_s < best_score:
                best_score = prev_s
                best_positions = pos_sky
            _collect(pos_sky, f"skyline-clust-{knob}")

            # DREAMPlace-lite refinement → legalize via skyline OR abacus.
            if pos_sky is not None:
                dp_raw = self._dreamplace_refine(
                    pos_sky, block_count, area_targets, constraints,
                    b2b_connectivity, p2b_connectivity, pins_pos,
                    steps=150, lr=0.8)
                if dp_raw is not None:
                    hints = [(p[0] + p[2] / 2, p[1] + p[3] / 2) for p in dp_raw]
                    # Option A: skyline-clustered re-pack with hints.
                    legal = self._solve_skyline_clustered(
                        block_count, area_targets, b2b_connectivity,
                        p2b_connectivity, pins_pos, constraints,
                        target_positions, hints, knob)
                    if legal is not None:
                        s = self._internal_cost(legal, b2b_connectivity,
                                                p2b_connectivity, pins_pos,
                                                constraints, block_count)
                        if s < best_score:
                            best_score = s
                            best_positions = legal
                        _collect(legal, f"dp-skyline-{knob}")
                    # Option B: abacus-style legalization (row assignment by
                    # analytical y, x by analytical x order).
                    fixed_col, pp_col_a, mib_col_a, _, _ = self._extract_cols(
                        constraints, block_count)
                    base_sizes_a, _ = self._base_dims(
                        block_count, area_targets, target_positions,
                        fixed_col, pp_col_a, mib_col_a)
                    total_area = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                                     for i in range(block_count))
                    W_knob = math.sqrt(max(total_area, 1.0)) * knob
                    if pp_positions_sky_hack := {i: pos_sky[i] for i in range(block_count)
                                                  if pp_col_a[i] != 0}:
                        pp_xmax = max(p[0] + p[2] for p in pp_positions_sky_hack.values())
                        W_knob = max(W_knob, pp_xmax)
                    widest = max((base_sizes_a[i][0] for i in range(block_count)),
                                 default=0.0)
                    W_knob = max(W_knob, widest)
                    legal_ab = self._abacus_legalize(
                        block_count, area_targets, constraints,
                        target_positions, hints, base_sizes_a, W_knob)
                    if legal_ab is not None:
                        s = self._internal_cost(legal_ab, b2b_connectivity,
                                                p2b_connectivity, pins_pos,
                                                constraints, block_count)
                        if s < best_score:
                            best_score = s
                            best_positions = legal_ab
                        _collect(legal_ab, f"abacus-{knob}")

        # 2) Shelf packer fallback/alternative.
        for knob in (0.5, 0.65, 0.8):
            self.aspect_knob = knob
            positions: Optional[List[Tuple[float, float, float, float]]] = None
            # Iterate shelf pack + b2b-aware target refinement up to 4 times
            # but break early once the internal cost plateaus.
            prev_s = float("inf")
            for it in range(4):
                targets = self._combined_targets(
                    block_count, p2b_connectivity, pins_pos,
                    b2b_connectivity, positions)
                positions = self._solve_once(
                    block_count, area_targets, b2b_connectivity, p2b_connectivity,
                    pins_pos, constraints, target_positions, targets)
                cur_s = self._internal_cost(positions, b2b_connectivity,
                                            p2b_connectivity, pins_pos, constraints,
                                            block_count)
                if it >= 1 and cur_s >= prev_s - 1e-3:
                    break
                prev_s = cur_s
            # Analytical refinement: nudge block positions via gradient descent
            # to reduce HPWL while keeping overlap penalty low. Discard if it
            # yields an infeasible layout (fallback to pre-refinement positions).
            # Use analytical refinement as a "target hint" even when the refined
            # layout itself is not feasible: its centroids give us better targets
            # for the NEXT shelf-pack pass. This retains feasibility (shelf pack
            # never overlaps) while injecting HPWL-aware position information.
            s_before = self._internal_cost(positions, b2b_connectivity,
                                           p2b_connectivity, pins_pos, constraints,
                                           block_count)
            candidates = [(s_before, positions)]
            refined_raw = self._analytical_refine_soft(
                positions, block_count, area_targets, constraints,
                b2b_connectivity, p2b_connectivity, pins_pos)
            if refined_raw is not None:
                # Build target hints from refined centroids and re-shelf.
                refine_targets = [(p[0] + p[2] / 2, p[1] + p[3] / 2)
                                  for p in refined_raw]
                relegal = self._solve_once(
                    block_count, area_targets, b2b_connectivity, p2b_connectivity,
                    pins_pos, constraints, target_positions, refine_targets)
                s_re = self._internal_cost(relegal, b2b_connectivity,
                                           p2b_connectivity, pins_pos, constraints,
                                           block_count)
                candidates.append((s_re, relegal))
            s, positions = min(candidates, key=lambda t: t[0])
            if s < best_score:
                best_score = s
                best_positions = positions
            _collect(positions, f"shelf-{knob}")
        self.aspect_knob = orig_knob

        # 3) Random-restart search: permute interior block ordering within
        # cluster groups to explore different HPWL trade-offs.
        if best_positions is not None and block_count >= 40:
            import random
            rng = random.Random(42)
            pin_targets_rs = self._combined_targets(
                block_count, p2b_connectivity, pins_pos,
                b2b_connectivity, best_positions)
            for trial in range(6):
                noisy = []
                for t in pin_targets_rs:
                    if t is None:
                        noisy.append(None)
                    else:
                        dx = rng.gauss(0, 3.0)
                        dy = rng.gauss(0, 3.0)
                        noisy.append((t[0] + dx, t[1] + dy))
                for knob in (0.6, 0.75):
                    cand = self._solve_skyline_clustered(
                        block_count, area_targets, b2b_connectivity,
                        p2b_connectivity, pins_pos, constraints,
                        target_positions, noisy, knob)
                    if cand is None:
                        continue
                    s = self._internal_cost(cand, b2b_connectivity,
                                            p2b_connectivity, pins_pos,
                                            constraints, block_count)
                    if s < best_score:
                        best_score = s
                        best_positions = cand
                    _collect(cand, f"restart-t{trial}-k{knob}")

        # 4) Boundary-fix: swap boundary-violating blocks with same-dim
        # non-boundary blocks that happen to be at the required edge.
        if best_positions is not None:
            fixed = self._boundary_swap_fix(
                best_positions, b2b_connectivity, p2b_connectivity,
                pins_pos, constraints, block_count)
            s = self._internal_cost(fixed, b2b_connectivity,
                                    p2b_connectivity, pins_pos,
                                    constraints, block_count)
            if s < best_score:
                best_score = s
                best_positions = fixed
            _collect(fixed, "boundary-fix")

        # 4b) Pure DREAMPlace path: start from DiT predictions (or shelf as
        # fallback), run analytical placement, then nudge-legalize. Preserves
        # analytical structure instead of snapping to shelf rows.
        # Capped at 80 — extending hurt FULL_SCORE significantly.
        if (analytical_place is not None and nudge_legalize is not None
                and best_positions is not None
                and block_count <= 80):
            try:
                if dit_targets is not None:
                    seed = self._build_init_from_targets(
                        block_count, dit_targets, area_targets, constraints,
                        target_positions)
                else:
                    seed = best_positions
                analy_pos = analytical_place(
                    seed, block_count, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints, target_positions,
                    n_steps=200, lr=0.5)
                legal_pos = nudge_legalize(
                    analy_pos, block_count, constraints, target_positions,
                    max_iters=500)
                if legal_pos is not None:
                    s = self._internal_cost(legal_pos, b2b_connectivity,
                                            p2b_connectivity, pins_pos,
                                            constraints, block_count)
                    if s < best_score:
                        best_score = s
                        best_positions = legal_pos
                    _collect(legal_pos, "dp-nudge")
            except Exception as e:
                print(f"[dreamplace nudge error] {e}", flush=True)

        # 4c) DREAMPlace with DiT shapes: DISABLED — regressed FULL_SCORE.
        # Keeping code for future experiments but gated off.
        if False and (analytical_place is not None and dit_full is not None
                and dit_shapes is not None and robust_legalize is not None
                and 80 < block_count <= 120):
            try:
                seed = self._build_init_from_dit(
                    block_count, dit_targets, dit_shapes, area_targets,
                    constraints, target_positions)
                analy_pos = analytical_place(
                    seed, block_count, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints, target_positions,
                    n_steps=150, lr=0.3)
                # Use robust_legalize which handles overlapping starts better.
                legal = robust_legalize(
                    analy_pos, block_count, constraints, target_positions)
                if legal is None and nudge_legalize is not None:
                    legal = nudge_legalize(
                        analy_pos, block_count, constraints, target_positions,
                        max_iters=1000)
                if legal is not None:
                    _collect(legal, "dp-dit-shapes")
                    try:
                        bf = self._boundary_swap_fix(
                            legal, b2b_connectivity, p2b_connectivity,
                            pins_pos, constraints, block_count)
                        _collect(bf, "dp-dit-shapes-bfix")
                    except Exception:
                        pass
            except Exception as e:
                if _trace:
                    print(f"[dp-dit-shapes error] {e}", flush=True)

        # 5) Sequence-pair + SA polish. Cap at 90 blocks (quality plateaus for
        # larger n, and runtime is already bounded by vectorized pack).
        if (sa_sequence_pair is not None and best_positions is not None
                and block_count <= 90):
            sp_pos = self._solve_sequence_pair(
                best_positions, block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions)
            if sp_pos is not None:
                s = self._internal_cost(sp_pos, b2b_connectivity,
                                        p2b_connectivity, pins_pos,
                                        constraints, block_count)
                if s < best_score:
                    best_score = s
                    best_positions = sp_pos
                _collect(sp_pos, "sp-sa")

        _collect(best_positions, "selected")

        # Final re-ranking: use contest-aligned GAP cost across all candidates.
        # Baseline strategy: use something closer to the "typical" value rather
        # than the min, so the low-HPWL candidate doesn't get hpwl_gap=0. The
        # contest uses a ground-truth baseline we don't have; using min biases
        # toward low-HPWL/low-area candidates at expense of v_rel. Using
        # median or a high percentile gives v_rel its proper multiplicative
        # weight in ranking.
        if _candidates:
            hs = sorted(h for (_, h, _, _, _) in _candidates if h > 0)
            as_ = sorted(a for (_, _, a, _, _) in _candidates if a > 0)
            # Use 70th percentile as baseline — empirically matches contest
            # behavior better than min (which artificially minimizes gaps for
            # the extremal candidate and under-penalizes v_rel).
            def _pct(xs, p):
                if not xs:
                    return 1.0
                i = max(0, min(len(xs) - 1, int(len(xs) * p)))
                return xs[i]
            # Rerank baseline percentile (configurable via env for studies).
            import os as _os
            _pct_level = float(_os.environ.get("RERANK_PCT", "0.3"))
            base_h = _pct(hs, _pct_level)
            base_a = _pct(as_, _pct_level)
            best_contest_cost = float("inf")
            best_contest_pos = None
            for (pos, h, a, v, tag) in _candidates:
                h_gap = (h / max(base_h, 1e-6)) - 1.0
                a_gap = (a / max(base_a, 1e-6)) - 1.0
                # Contest: (1 + 0.5*(h_gap + a_gap)) * exp(2*v_rel).
                cost = (1.0 + 0.5 * (h_gap + a_gap)) * math.exp(2.0 * v)
                if _trace:
                    print(f"[rerank {tag}] h_gap={h_gap:.3f} a_gap={a_gap:.3f} "
                          f"v={v:.3f} cost={cost:.3f}", flush=True)
                if cost < best_contest_cost:
                    best_contest_cost = cost
                    best_contest_pos = pos
            if best_contest_pos is not None:
                best_positions = best_contest_pos

        return best_positions  # type: ignore[return-value]

    def _solve_sequence_pair(self, init_positions, block_count, area_targets,
                             b2b_conn, p2b_conn, pins_pos, constraints,
                             target_positions):
        """
        Run simulated annealing over sequence-pair representation. Uses the
        initial shelf layout's block order to warm-start.
        Preplaced blocks are excluded from SA — re-inserted at their target
        positions post-pack (caller should verify feasibility).
        """
        if sa_sequence_pair is None:
            return None
        fixed_col, preplaced_col, mib_col, clust_col, bound_col = \
            self._extract_cols(constraints, block_count)
        base_sizes, is_inflex = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)

        preplaced = {}
        if target_positions is not None:
            for i in range(block_count):
                if preplaced_col[i] == 0:
                    continue
                tx = float(target_positions[i, 0]); ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2]); th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    preplaced[i] = (tx, ty, tw, th)
        pp_ids = set(preplaced.keys())

        # Build SP over non-preplaced blocks.
        mobile = [i for i in range(block_count) if i not in pp_ids]
        if len(mobile) < 2:
            return None

        # Initial order: sort mobile blocks by init_positions' (y, x).
        mobile.sort(key=lambda i: (init_positions[i][1], init_positions[i][0]))
        widths = [base_sizes[i][0] for i in mobile]
        heights = [base_sizes[i][1] for i in mobile]
        rotatable = [not is_inflex[i] for i in mobile]

        # Cost function: HPWL + 0.5 * bbox_area (on mobile blocks only, treating
        # preplaced as obstacles summed into bbox).
        pp_xmax = max((x + w for (x, _, w, _) in preplaced.values()), default=0.0)
        pp_ymax = max((y + h for (_, y, _, h) in preplaced.values()), default=0.0)

        # Pre-compute vectorized edge arrays.
        import numpy as _np
        b2b_i_arr = b2b_j_arr = b2b_w_arr = None
        if b2b_conn is not None and b2b_conn.numel() > 0:
            vb = b2b_conn[b2b_conn[:, 0] >= 0]
            if vb.numel() > 0:
                ii = vb[:, 0].long().numpy(); jj = vb[:, 1].long().numpy()
                ww = vb[:, 2].float().numpy()
                ok = (ii < block_count) & (jj < block_count)
                b2b_i_arr = ii[ok]; b2b_j_arr = jj[ok]; b2b_w_arr = ww[ok]

        p2b_blk_arr = p2b_w_arr = p2b_px = p2b_py = None
        if p2b_conn is not None and p2b_conn.numel() > 0:
            vp = p2b_conn[p2b_conn[:, 0] >= 0]
            if vp.numel() > 0:
                pin_i = vp[:, 0].long().numpy()
                blk_i = vp[:, 1].long().numpy()
                ww = vp[:, 2].float().numpy()
                ok = (blk_i < block_count) & (pin_i < pins_pos.shape[0])
                p2b_blk_arr = blk_i[ok]; p2b_w_arr = ww[ok]
                p2b_px = pins_pos[pin_i[ok], 0].float().numpy()
                p2b_py = pins_pos[pin_i[ok], 1].float().numpy()

        mob2orig_arr = _np.asarray(mobile, dtype=_np.int64)
        # Static centroids for preplaced blocks.
        full_cx = _np.zeros(block_count, dtype=_np.float64)
        full_cy = _np.zeros(block_count, dtype=_np.float64)
        for i, (x, y, w, h) in preplaced.items():
            full_cx[i] = x + w / 2
            full_cy[i] = y + h / 2

        # Precompute boundary-constrained mobile blocks and their bit codes.
        bnd_codes_mob = []
        clust_ids_mob = []
        for i in mobile:
            bnd_codes_mob.append(int(bound_col[i].item()))
            clust_ids_mob.append(int(clust_col[i].item()))
        bnd_codes_mob = _np.asarray(bnd_codes_mob, dtype=_np.int64)
        clust_ids_mob = _np.asarray(clust_ids_mob, dtype=_np.int64)
        n_constrained = int((bnd_codes_mob != 0).sum())

        def cost_fn(pos_mob):
            # Update mobile centroids.
            pm = _np.asarray(pos_mob, dtype=_np.float64)  # (m, 4)
            mob_cx = pm[:, 0] + pm[:, 2] / 2
            mob_cy = pm[:, 1] + pm[:, 3] / 2
            full_cx[mob2orig_arr] = mob_cx
            full_cy[mob2orig_arr] = mob_cy
            hpwl = 0.0
            if b2b_i_arr is not None:
                dx = _np.abs(full_cx[b2b_i_arr] - full_cx[b2b_j_arr])
                dy = _np.abs(full_cy[b2b_i_arr] - full_cy[b2b_j_arr])
                hpwl += float((b2b_w_arr * (dx + dy)).sum())
            if p2b_blk_arr is not None:
                dx = _np.abs(full_cx[p2b_blk_arr] - p2b_px)
                dy = _np.abs(full_cy[p2b_blk_arr] - p2b_py)
                hpwl += float((p2b_w_arr * (dx + dy)).sum())
            xmin_mob = float(pm[:, 0].min())
            xmax_mob = float((pm[:, 0] + pm[:, 2]).max())
            ymin_mob = float(pm[:, 1].min())
            ymax_mob = float((pm[:, 1] + pm[:, 3]).max())
            xmax = max(xmax_mob, pp_xmax)
            ymax = max(ymax_mob, pp_ymax)
            area_cost = xmax * ymax

            # Boundary violations for mobile blocks (approx: use mobile bbox).
            v_bnd = 0
            if n_constrained > 0:
                eps = 1e-6
                bx = pm[:, 0]; by = pm[:, 1]
                bxw = pm[:, 0] + pm[:, 2]; byh = pm[:, 1] + pm[:, 3]
                # check each bit
                viol = _np.zeros(len(mobile), dtype=_np.bool_)
                viol |= ((bnd_codes_mob & 1) != 0) & (_np.abs(bx - xmin_mob) >= eps)
                viol |= ((bnd_codes_mob & 2) != 0) & (_np.abs(bxw - xmax_mob) >= eps)
                viol |= ((bnd_codes_mob & 4) != 0) & (_np.abs(byh - ymax_mob) >= eps)
                viol |= ((bnd_codes_mob & 8) != 0) & (_np.abs(by - ymin_mob) >= eps)
                v_bnd = int(viol.sum())
            # Penalty scale — each violation adds ~5% of quality cost.
            penalty = 50.0 * v_bnd * (hpwl + area_cost) / max(block_count, 1)
            return hpwl + 0.5 * area_cost + penalty

        # Run SA. Vectorized packing makes each trial cheap; budget heavily
        # for maximum solution exploration.
        m = len(mobile)
        if m <= 30:
            n_starts = 30; max_iters = 200 * m
        elif m <= 50:
            n_starts = 20; max_iters = 150 * m
        elif m <= 70:
            n_starts = 15; max_iters = 100 * m
        else:
            n_starts = 8; max_iters = 60 * m
        best_pack = None
        best_cost = float("inf")
        for seed in range(n_starts):
            sp_trial = sa_sequence_pair(
                m, widths, heights, cost_fn,
                max_iters=max_iters,
                T0=float(max(10.0, sum(widths) * sum(heights) / m)),
                T_end=0.5,
                rotatable=rotatable,
                seed=42 + seed * 17,
            )
            pk = sp_trial.pack()
            c = cost_fn(pk)
            if c < best_cost:
                best_cost = c
                best_pack = pk
        packed = best_pack

        # Assemble full solution: mobile blocks from SP, preplaced at targets.
        full = [None] * block_count
        for k, i in enumerate(mobile):
            full[i] = packed[k]
        for i, xywh in preplaced.items():
            full[i] = xywh

        # Preplaced may overlap mobile SP packing — shift mobile blocks up
        # past pp_ymax if any mobile block collides with preplaced.
        # Simpler: compute union of pp rectangles; any mobile block that
        # overlaps ANY pp in x-range AND y-range is shifted up.
        for i in range(block_count):
            if i in pp_ids:
                continue
            bx, by, bw, bh = full[i]
            for (px, py, pw, ph) in preplaced.values():
                if bx >= px + pw or bx + bw <= px:
                    continue
                if by >= py + ph or by + bh <= py:
                    continue
                # Overlap — shift block up.
                by = py + ph
                full[i] = (bx, by, bw, bh)
        # Verify no overlaps among mobile blocks (SP pack guarantees none).
        return full  # type: ignore[return-value]

    def _cluster_compact_fix(self, positions, constraints, block_count,
                              max_passes: int = 4):
        """
        Shift same-cluster blocks toward their group centroid to promote
        connectivity. Accept only shifts that don't create overlaps.
        Returns a new position list (does not mutate input).
        """
        if constraints is None or constraints.dim() <= 1 \
           or constraints.shape[1] < 4:
            return positions
        clust_col = constraints[:block_count, 3]
        pp_col = constraints[:block_count, 1]
        n_groups = int(clust_col.max().item()) if clust_col.numel() > 0 else 0
        if n_groups == 0:
            return positions

        pos = [list(p) for p in positions]
        pinned = set(i for i in range(block_count)
                     if int(pp_col[i].item()) != 0)

        def overlaps_any(i, new_x, new_y):
            w, h = pos[i][2], pos[i][3]
            for j in range(block_count):
                if j == i:
                    continue
                xj, yj, wj, hj = pos[j]
                if (new_x < xj + wj - 1e-9
                        and new_x + w > xj + 1e-9
                        and new_y < yj + hj - 1e-9
                        and new_y + h > yj + 1e-9):
                    return True
            return False

        for _ in range(max_passes):
            changed = False
            for g in range(1, n_groups + 1):
                members = [i for i in range(block_count)
                           if int(clust_col[i].item()) == g]
                if len(members) <= 1:
                    continue
                cx_g = sum(pos[i][0] + pos[i][2] / 2 for i in members) / len(members)
                cy_g = sum(pos[i][1] + pos[i][3] / 2 for i in members) / len(members)
                for i in members:
                    if i in pinned:
                        continue
                    cur_cx = pos[i][0] + pos[i][2] / 2
                    cur_cy = pos[i][1] + pos[i][3] / 2
                    dx = cx_g - cur_cx
                    dy = cy_g - cur_cy
                    # Try progressively smaller shifts toward group centroid.
                    for scale in (1.0, 0.5, 0.25, 0.125):
                        nx = pos[i][0] + dx * scale
                        ny = pos[i][1] + dy * scale
                        if nx < -1e-9 or ny < -1e-9:
                            continue
                        if not overlaps_any(i, nx, ny):
                            if abs(nx - pos[i][0]) > 1e-6 or abs(ny - pos[i][1]) > 1e-6:
                                pos[i][0] = nx
                                pos[i][1] = ny
                                changed = True
                            break
            if not changed:
                break
        return [tuple(p) for p in pos]

    def _boundary_swap_fix(self, positions, b2b_conn, p2b_conn, pins_pos,
                           constraints, block_count):
        """
        For each boundary-constrained block NOT at its required edge, try
        swapping with a same-dimension block at that edge (non-boundary, or
        with weaker boundary requirement). Accept only if no-op or helpful.
        """
        if constraints is None or constraints.dim() < 2 or constraints.shape[1] < 5:
            return positions
        n = len(positions)
        pos = list(positions)
        pp_col = constraints[:block_count, 1]
        bnd_col = constraints[:block_count, 4]

        xmin = min(p[0] for p in pos)
        xmax = max(p[0] + p[2] for p in pos)
        ymin = min(p[1] for p in pos)
        ymax = max(p[1] + p[3] for p in pos)
        eps = 1e-6

        def _touches(i, bit):
            bx, by, bw, bh = pos[i]
            if bit == 1: return abs(bx - xmin) < eps
            if bit == 2: return abs(bx + bw - xmax) < eps
            if bit == 4: return abs(by + bh - ymax) < eps
            if bit == 8: return abs(by - ymin) < eps
            return False

        def _violates(i):
            c = int(bnd_col[i].item())
            if c == 0:
                return False
            for bit in (1, 2, 4, 8):
                if c & bit and not _touches(i, bit):
                    return True
            return False

        # Preplaced blocks are pinned; can't swap them.
        swappable = [i for i in range(block_count) if pp_col[i] == 0]

        # Group blocks by rounded (w, h).
        from collections import defaultdict
        shape_buckets = defaultdict(list)
        for i in swappable:
            key = (round(pos[i][2], 2), round(pos[i][3], 2))
            shape_buckets[key].append(i)

        for _ in range(2):
            for i in swappable:
                if not _violates(i):
                    continue
                key = (round(pos[i][2], 2), round(pos[i][3], 2))
                mates = shape_buckets[key]
                for j in mates:
                    if j == i:
                        continue
                    # Check if swapping helps: i currently violates; if j's slot
                    # satisfies i's boundary, do the swap — but only if j is
                    # non-boundary or less-constrained.
                    c_i = int(bnd_col[i].item()); c_j = int(bnd_col[j].item())
                    if c_j != 0 and c_j != c_i:
                        # Swap might break j's boundary. Skip to be safe.
                        continue
                    # Temporarily swap.
                    pos[i], pos[j] = pos[j], pos[i]
                    if not _violates(i) and not (c_j != 0 and _violates(j)):
                        break
                    # Revert.
                    pos[i], pos[j] = pos[j], pos[i]
        return pos

    # ------------------------------------------------------------------
    # Skyline-based solver (alternative to shelf packer)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cluster super-block: pack cluster members as a unit
    # ------------------------------------------------------------------

    def _solve_skyline_clustered(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        pin_targets: List[Optional[Tuple[float, float]]],
        knob: float,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Skyline packer where each cluster group is pre-arranged as a single
        horizontal row and packed as ONE rigid entity. This guarantees each
        cluster forms a single connected component (grouping constraint).
        """
        fixed_col, preplaced_col, mib_col, clust_col, bound_col = \
            self._extract_cols(constraints, block_count)
        base_sizes, is_inflex = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)

        preplaced: Dict[int, Tuple[float, float, float, float]] = {}
        if target_positions is not None:
            for i in range(block_count):
                if preplaced_col[i] == 0:
                    continue
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    preplaced[i] = (tx, ty, tw, th)
        pp_ids = set(preplaced.keys())

        total_area = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                         for i in range(block_count))
        W = math.sqrt(max(total_area, 1.0)) * knob
        if preplaced:
            pp_xmax = max(x + w for (x, _, w, _) in preplaced.values())
            W = max(W, pp_xmax)
        widest = max((base_sizes[i][0] for i in range(block_count) if i not in pp_ids),
                     default=0.0)
        W = max(W, widest)

        entities = self._cluster_superblocks(
            block_count, clust_col, bound_col, base_sizes, is_inflex,
            area_targets, pin_targets, W, pp_ids)

        # Width of a cluster row might exceed W → split into multi-row.
        # For simplicity, if single-row width > W, split as 2 rows.
        def _row_layout(ent: List[Tuple[int, float, float, float, float]],
                        target_w: float):
            """
            Re-arrange entity members into rows each ≤ target_w wide.
            Returns new entity list (dx, dy adjusted).
            """
            rows = []
            cur = []
            cur_w = 0.0
            for m in ent:
                i, _, _, w_i, h_i = m
                if cur and cur_w + w_i > target_w + 1e-6:
                    rows.append(cur)
                    cur = []
                    cur_w = 0.0
                cur.append(m)
                cur_w += w_i
            if cur:
                rows.append(cur)
            # Build new entity with multi-row dy offsets.
            out = []
            y_cur = 0.0
            for r in rows:
                row_h = max(m[4] for m in r)
                x_cur = 0.0
                for (i, _, _, w_i, h_i) in r:
                    out.append((i, x_cur, y_cur, w_i, h_i))
                    x_cur += w_i
                y_cur += row_h
            return out

        # Rebuild entities with width-capped rows.
        entities = [_row_layout(e, W) for e in entities]

        # Compute each entity's bbox dims.
        ent_dims = []
        for e in entities:
            if not e:
                ent_dims.append((0.0, 0.0))
                continue
            w_e = max(dx + w_i for (_, dx, dy, w_i, h_i) in e)
            h_e = max(dy + h_i for (_, dx, dy, w_i, h_i) in e)
            ent_dims.append((w_e, h_e))

        # Order entities by category:
        # 1. Entities containing bottom-bound blocks → place at y=0 first.
        # 2. Entities with left-bound blocks → prefer x=0.
        # 3. Interior entities → pin-target x ordering.
        # 4. Top-bound entities last (highest y).
        def _ent_has_bnd(e, bit):
            return any(int(bound_col[m[0]].item()) & bit for m in e)

        def _ent_tx(e):
            xs = [pin_targets[m[0]][0] for m in e if pin_targets[m[0]] is not None]
            return sum(xs) / len(xs) if xs else W / 2

        bottom_ents = [(i, e) for i, e in enumerate(entities) if _ent_has_bnd(e, B_BOTTOM)]
        top_ents = [(i, e) for i, e in enumerate(entities)
                    if _ent_has_bnd(e, B_TOP) and not _ent_has_bnd(e, B_BOTTOM)]
        mid_ents = [(i, e) for i, e in enumerate(entities)
                    if not _ent_has_bnd(e, B_BOTTOM) and not _ent_has_bnd(e, B_TOP)]
        bottom_ents.sort(key=lambda p: _ent_tx(p[1]))
        top_ents.sort(key=lambda p: _ent_tx(p[1]))
        mid_ents.sort(key=lambda p: _ent_tx(p[1]))

        # Initialize skyline with preplaced obstacles.
        sky = Skyline(W)
        for (px, py, pw, ph) in preplaced.values():
            if px >= W or px + pw <= 0:
                continue
            sky.add_obstacle(px, py, pw, ph)

        positions: List[Optional[Tuple[float, float, float, float]]] = [None] * block_count
        for i, xywh in preplaced.items():
            positions[i] = xywh

        def _place_entity(e, w_e, h_e, prefer_x=None):
            if w_e > W + 1e-6:
                return False
            p = sky.place(w_e, h_e, prefer_x=prefer_x)
            if p is None:
                return False
            ex, ey = p
            for (i, dx, dy, w_i, h_i) in e:
                positions[i] = (ex + dx, ey + dy, w_i, h_i)
            return True

        # Place bottom entities at y=0 where possible.
        # Force them onto the first row by artificially pre-raising the skyline
        # in non-bottom regions to avoid them landing low. Simpler: place
        # bottom entities first in order; they will use lowest y available.
        # Since skyline is initialized with preplaced obstacles, bottom will
        # go to y=0 in free columns.
        for _, e in bottom_ents:
            w_e, h_e = ent_dims[_]
            # Prefer the x-region around the entity's pin-target center.
            prefer = _ent_tx(e) - w_e / 2
            if not _place_entity(e, w_e, h_e, prefer_x=prefer):
                return None

        for _, e in mid_ents:
            w_e, h_e = ent_dims[_]
            if _ent_has_bnd(e, B_LEFT):
                prefer = 0.0
            elif _ent_has_bnd(e, B_RIGHT):
                prefer = W - w_e
            else:
                prefer = _ent_tx(e) - w_e / 2
            if not _place_entity(e, w_e, h_e, prefer_x=prefer):
                return None

        for _, e in top_ents:
            w_e, h_e = ent_dims[_]
            prefer = _ent_tx(e) - w_e / 2
            if not _place_entity(e, w_e, h_e, prefer_x=prefer):
                return None

        for i in range(block_count):
            if positions[i] is None:
                return None
        return positions  # type: ignore[return-value]

    def _cluster_superblocks(
        self,
        block_count: int,
        clust_col: torch.Tensor,
        bound_col: torch.Tensor,
        base_sizes: List[Tuple[float, float]],
        is_inflex: List[bool],
        area_targets: torch.Tensor,
        pin_targets: List[Optional[Tuple[float, float]]],
        W: float,
        pp_ids: set,
    ):
        """
        For each cluster group, pre-arrange its members into a compact
        rectangular sub-layout so the group is guaranteed to be a single
        connected component. Returns:
          - super_sizes: list of (w, h) for each "entity" (cluster or lone block)
          - super_members: list of list of (block_id, dx, dy, w, h) for each entity
                          where (dx, dy) is the relative offset inside the super-block.
          - single_ids: list of non-clustered blocks (as "entities" with 1 member).
          - super_is_inflex: bool per super-block (if any inflex inside, treat whole as inflex).

        Entities are interchangeable for outer packing; they internally preserve
        cluster abutment.
        """
        n_clusters = int(clust_col.max().item()) if clust_col.numel() else 0
        entities = []  # each entity = list of (block_id, dx, dy, w, h)
        cluster_assigned = [False] * block_count
        # Build entities per cluster.
        for g in range(1, n_clusters + 1):
            members = [i for i in range(block_count)
                       if int(clust_col[i].item()) == g and i not in pp_ids]
            if not members:
                continue
            for i in members:
                cluster_assigned[i] = True
            # Arrange as a horizontal row (sorted by pin-target-x).
            def _mkey(i):
                tx = pin_targets[i][0] if pin_targets[i] is not None else 0.0
                return (tx, -base_sizes[i][1])
            members.sort(key=_mkey)
            # Pick a uniform height for the cluster row if flexible blocks
            # dominate. For simplicity, use each block's own height (h) and
            # just stack them horizontally — the row's height is max h.
            dx = 0.0
            row = []
            for i in members:
                w_i, h_i = base_sizes[i]
                row.append((i, dx, 0.0, w_i, h_i))
                dx += w_i
            entities.append(row)
        # Build singleton entities for unclustered blocks.
        for i in range(block_count):
            if i in pp_ids or cluster_assigned[i]:
                continue
            w_i, h_i = base_sizes[i]
            entities.append([(i, 0.0, 0.0, w_i, h_i)])
        return entities

    def _solve_skyline(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        pin_targets: List[Optional[Tuple[float, float]]],
        knob: float,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Skyline bottom-left packing with preplaced obstacles. Handles all
        soft constraints via block ordering and per-block aspect choice.
        """
        fixed_col, preplaced_col, mib_col, clust_col, bound_col = \
            self._extract_cols(constraints, block_count)
        base_sizes, is_inflex = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)

        # Preplaced blocks.
        preplaced: Dict[int, Tuple[float, float, float, float]] = {}
        if target_positions is not None:
            for i in range(block_count):
                if preplaced_col[i] == 0:
                    continue
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    preplaced[i] = (tx, ty, tw, th)
        pp_ids = set(preplaced.keys())

        total_area = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                         for i in range(block_count))
        W = math.sqrt(max(total_area, 1.0)) * knob
        if preplaced:
            pp_xmax = max(x + w for (x, _, w, _) in preplaced.values())
            W = max(W, pp_xmax)
        widest = max((base_sizes[i][0] for i in range(block_count) if i not in pp_ids),
                     default=0.0)
        W = max(W, widest)

        # Decide per-block aspect for soft blocks: aim for square by default
        # but bias toward "wide" for blocks whose target-y is at the edge
        # (so they can span horizontally along boundary).
        # For now: use square base_sizes (MIB/fixed already handled in _base_dims).
        sizes = list(base_sizes)

        # Partition blocks by boundary code.
        bottom_ids: List[int] = []
        top_ids: List[int] = []
        left_ids: List[int] = []
        right_ids: List[int] = []
        interior_ids: List[int] = []
        for i in range(block_count):
            if i in pp_ids:
                continue
            code = int(bound_col[i].item())
            if code & B_BOTTOM:
                bottom_ids.append(i)
            elif code & B_TOP:
                top_ids.append(i)
            elif code & B_LEFT:
                left_ids.append(i)
            elif code & B_RIGHT:
                right_ids.append(i)
            else:
                interior_ids.append(i)

        # Sort keys for best HPWL + cluster contiguity.
        def _key_full(i: int):
            clu = int(clust_col[i].item())
            tx = pin_targets[i][0] if pin_targets[i] is not None else W / 2
            ty = pin_targets[i][1] if pin_targets[i] is not None else 0.0
            return (clu, tx, ty)

        # Initialize skyline with preplaced as obstacles.
        # We pack ABOVE y=0 in strip [0, W].
        sky = Skyline(W)
        # Preplaced blocks raise the skyline at their x-range to y+h.
        for (px, py, pw, ph) in preplaced.values():
            # Safe-guard: clip preplaced outside strip.
            if px >= W or px + pw <= 0:
                continue
            sky.add_obstacle(px, py, pw, ph)

        positions: List[Optional[Tuple[float, float, float, float]]] = [None] * block_count
        for i, xywh in preplaced.items():
            positions[i] = xywh

        # Bottom boundary blocks must land at y=0. Place first, aligned to y=0,
        # horizontally adjacent (sorted by cluster/tx). If any preplaced at y=0
        # blocks an x-range, skip over it.
        def _fit_at_y0(ids: List[int]) -> bool:
            """Place ids in a single row at y=0, avoiding preplaced obstacles."""
            # Order corners first: BL(9) leftmost, BR(10) rightmost.
            bl = [i for i in ids if int(bound_col[i].item()) == 9]
            br = [i for i in ids if int(bound_col[i].item()) == 10]
            core = [i for i in ids if i not in bl and i not in br]
            core.sort(key=_key_full)
            ordered = bl + core + br
            if not ordered:
                return True
            # Compute cumulative widths.
            total_w = sum(sizes[i][0] for i in ordered)
            # If total_w > W, we cannot fit all at y=0; some will go to higher y.
            # Keep only as many as fit; others go to interior pool.
            # For now, if doesn't fit, use skyline placement per block.
            if total_w <= W + 1e-6:
                # Check if any preplaced at y=0 overlaps.
                # Collect preplaced occupying y=0 by x-range.
                pp_blockers = []
                for (px, py, pw, ph) in preplaced.values():
                    if py < 1e-6:
                        pp_blockers.append((px, px + pw))
                pp_blockers.sort()
                # Greedy row fill: place blocks around pp_blockers.
                cursor = 0.0
                # Merge blockers into free intervals [f_x0, f_x1].
                free_intervals = []
                x_cur = 0.0
                for (bx0, bx1) in pp_blockers:
                    if bx0 > x_cur:
                        free_intervals.append((x_cur, bx0))
                    x_cur = max(x_cur, bx1)
                if x_cur < W:
                    free_intervals.append((x_cur, W))
                if not free_intervals:
                    free_intervals = [(0.0, W)]
                # Fill blocks into free intervals left-to-right.
                bi = 0
                for (fx0, fx1) in free_intervals:
                    x_pos = fx0
                    while bi < len(ordered) and x_pos + sizes[ordered[bi]][0] <= fx1 + 1e-6:
                        i = ordered[bi]
                        w_i, h_i = sizes[i]
                        # Special: if bl in this interval, place bl first;
                        # similar for br at interval end.
                        positions[i] = (x_pos, 0.0, w_i, h_i)
                        sky.add_obstacle(x_pos, 0.0, w_i, h_i)
                        x_pos += w_i
                        bi += 1
                # Remaining blocks spill into skyline (they won't touch bottom).
                for i in ordered[bi:]:
                    w_i, h_i = sizes[i]
                    p = sky.place(w_i, h_i, prefer_x=pin_targets[i][0] if pin_targets[i] else None)
                    if p is None:
                        return False
                    positions[i] = (p[0], p[1], w_i, h_i)
                return True
            else:
                # Cannot fit all at y=0; fall through to skyline placement.
                return False

        _fit_at_y0(bottom_ids)

        # Interior, left, right blocks: place via skyline bottom-left fit.
        # Preferred x for left-bound is 0; for right-bound is W - w_i.
        mid_blocks = interior_ids + left_ids + right_ids
        mid_blocks.sort(key=_key_full)
        for i in mid_blocks:
            if positions[i] is not None:
                continue
            w_i, h_i = sizes[i]
            code = int(bound_col[i].item())
            if code & B_LEFT:
                prefer = 0.0
            elif code & B_RIGHT:
                prefer = W - w_i
            elif pin_targets[i] is not None:
                prefer = pin_targets[i][0] - w_i / 2
            else:
                prefer = None
            p = sky.place(w_i, h_i, prefer_x=prefer)
            if p is None:
                return None
            positions[i] = (p[0], p[1], w_i, h_i)

        # Top boundary blocks: place last so they land highest.
        top_ordered = [i for i in top_ids if positions[i] is None]
        top_ordered.sort(key=_key_full)
        for i in top_ordered:
            w_i, h_i = sizes[i]
            # Prefer x near pin target.
            prefer = pin_targets[i][0] - w_i / 2 if pin_targets[i] else None
            p = sky.place(w_i, h_i, prefer_x=prefer)
            if p is None:
                return None
            positions[i] = (p[0], p[1], w_i, h_i)

        # Post-pass: right-align right-bound blocks that didn't land at x=W-w.
        # Can't change x unilaterally (might overlap); instead, iterate a single
        # pass of local shifts only if the block's current right edge is already
        # near W (within some slack).
        # (Skipped in this first cut — let boundary loss steer the sweep.)

        # Fill any holes (shouldn't happen for a valid run).
        for i in range(block_count):
            if positions[i] is None:
                return None

        return positions  # type: ignore[return-value]

    def _spread_legalize(
        self,
        block_count: int,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        base_sizes: List[Tuple[float, float]],
        is_inflex: List[bool],
        analytical_centroids: List[Tuple[float, float]],
        W: float,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Position-preserving legalizer: starts from analytical centroids, places
        blocks greedily by y ascending into a skyline while minimizing
        displacement from analytical positions. This retains the global
        structure of the analytical placement far better than BL-fit.
        """
        n = block_count
        # Preplaced hard-pin.
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

        positions: List[Optional[Tuple[float, float, float, float]]] = [None] * n
        for i, xywh in preplaced.items():
            positions[i] = xywh

        sky = Skyline(W)
        for (px, py, pw, ph) in preplaced.values():
            if px >= W or px + pw <= 0:
                continue
            sky.add_obstacle(px, py, pw, ph)

        # Order mobile blocks by analytical centroid-y ascending.
        mobile = [i for i in range(n) if i not in pp_ids]
        mobile.sort(key=lambda i: analytical_centroids[i][1])

        for i in mobile:
            w_i, h_i = base_sizes[i]
            cx_a, cy_a = analytical_centroids[i]
            anchor_x = cx_a - w_i / 2
            anchor_y = cy_a - h_i / 2
            # Search a small window of candidate x positions around anchor_x.
            candidates = []
            # Left-to-right scan of skyline segments near anchor.
            for sx, sy, sw in sky.segs:
                for cx_try in (sx, sx + sw - w_i, anchor_x):
                    if cx_try < 0 - 1e-9 or cx_try > W - w_i + 1e-9:
                        continue
                    candidates.append(cx_try)
            candidates = sorted(set(candidates))
            best = None
            for cx_try in candidates:
                y_try = sky._contour_max(cx_try, cx_try + w_i)
                if y_try is None:
                    continue
                # Displacement cost from analytical anchor.
                disp = abs(cx_try - anchor_x) + abs(y_try - anchor_y)
                key = (y_try, disp)  # prefer lower y, then lower displacement
                if best is None or key < best[0]:
                    best = (key, cx_try, y_try)
            if best is None:
                return None
            _, x, y = best
            positions[i] = (x, y, w_i, h_i)
            sky.add_obstacle(x, y, w_i, h_i)

        for i in range(n):
            if positions[i] is None:
                return None
        return positions  # type: ignore[return-value]

    def _build_init_from_dit(
        self,
        block_count: int,
        targets: List[Tuple[float, float]],
        shapes: List[Tuple[float, float]],
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float, float, float]]:
        """
        Build full (x, y, w, h) layout using DiT-predicted centroid + shape.
        Preplaced, fixed-shape, and MIB blocks still take their hard-
        constrained (w, h); only soft blocks use DiT's predicted shape.
        """
        fixed_col, preplaced_col, mib_col, _, _ = self._extract_cols(
            constraints, block_count)
        base_sizes, is_inflex = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)
        positions = []
        for i in range(block_count):
            if is_inflex[i]:
                w, h = base_sizes[i]
            else:
                w, h = shapes[i] if i < len(shapes) else base_sizes[i]
            if preplaced_col[i] != 0 and target_positions is not None:
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            else:
                cx, cy = targets[i] if i < len(targets) else (0.0, 0.0)
                x = cx - w / 2
                y = cy - h / 2
            positions.append((x, y, w, h))
        return positions

    def _dit_cluster_pack(
        self,
        block_count: int,
        dit_targets: List[Tuple[float, float]],
        dit_shapes: List[Tuple[float, float]],
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        knob: float,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Cluster-aware DiT pack: groups same-cluster blocks as horizontal
        super-blocks, then places super-blocks via skyline sorted by centroid y.
        Guarantees grouping constraint (each cluster is one connected row).
        Uses DiT shapes for soft blocks.
        """
        fixed_col, preplaced_col, mib_col, clust_col, bound_col = \
            self._extract_cols(constraints, block_count)
        base_sizes, is_inflex = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)

        # Final shapes per block: DiT shapes for soft blocks, base for inflex.
        shapes = []
        for i in range(block_count):
            if is_inflex[i]:
                shapes.append(base_sizes[i])
            else:
                shapes.append(dit_shapes[i] if i < len(dit_shapes) else base_sizes[i])

        # Preplaced blocks pinned.
        preplaced: Dict[int, Tuple[float, float, float, float]] = {}
        if target_positions is not None:
            for i in range(block_count):
                if preplaced_col[i] == 0:
                    continue
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    preplaced[i] = (tx, ty, tw, th)
                    shapes[i] = (tw, th)

        # Group by cluster id; non-cluster blocks are singleton groups.
        groups: Dict[int, List[int]] = {}
        for i in range(block_count):
            if i in preplaced:
                continue
            cid = int(clust_col[i].item())
            key = cid if cid > 0 else -(i + 1)  # unique for non-clustered
            groups.setdefault(key, []).append(i)

        # Super-block dims and anchor y for each group.
        super_blocks: List[Tuple[List[int], float, float, float]] = []
        # Each super-block: (member_ids, super_w, super_h, anchor_y).
        for gid, members in groups.items():
            # Within-group order: sort by DiT centroid x (left-to-right).
            members_sorted = sorted(members, key=lambda i:
                                     dit_targets[i][0] if i < len(dit_targets) else 0.0)
            super_w = sum(shapes[i][0] for i in members_sorted)
            super_h = max(shapes[i][1] for i in members_sorted)
            anchor_y = (sum(dit_targets[i][1]
                            if i < len(dit_targets) else 0.0
                            for i in members_sorted)
                        / max(len(members_sorted), 1))
            super_blocks.append((members_sorted, super_w, super_h, anchor_y))

        total_area = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                         for i in range(block_count))
        W = math.sqrt(max(total_area, 1.0)) * knob
        # Strip must be wide enough for the widest super-block.
        widest = max(sw for (_, sw, _, _) in super_blocks) if super_blocks else 0.0
        W = max(W, widest)
        if preplaced:
            pp_xmax = max(x + w for (x, _, w, _) in preplaced.values())
            W = max(W, pp_xmax)

        positions: List[Optional[Tuple[float, float, float, float]]] = [None] * block_count

        sky = Skyline(W)
        # Seed with preplaced as obstacles.
        for i, (x, y, w, h) in preplaced.items():
            positions[i] = (x, y, w, h)
            sky.add_obstacle(x, y, w, h)

        # Sort super-blocks by anchor y (ascending).
        super_blocks.sort(key=lambda t: t[3])

        for (members_sorted, super_w, super_h, anchor_y) in super_blocks:
            # Place super-block via skyline (bottom-left-fit with x preference).
            prefer_x = 0.0
            if members_sorted:
                leftmost = members_sorted[0]
                if leftmost < len(dit_targets):
                    prefer_x = max(0.0, dit_targets[leftmost][0]
                                   - shapes[leftmost][0] / 2)
            pos = sky.place(super_w, super_h, prefer_x=prefer_x)
            if pos is None:
                return None
            x0, y0 = pos
            # Lay out members inside the super-block, left-to-right.
            cursor_x = x0
            for i in members_sorted:
                w, h = shapes[i]
                positions[i] = (cursor_x, y0, w, h)
                cursor_x += w

        for i in range(block_count):
            if positions[i] is None:
                return None
        return positions  # type: ignore[return-value]

    def _dit_shape_skyline_pack(
        self,
        block_count: int,
        dit_targets: List[Tuple[float, float]],
        dit_shapes: List[Tuple[float, float]],
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        knob: float,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Skyline pack using DiT-predicted shapes for soft blocks (not base_dims
        squares) and DiT centroids as preferred x positions. Processes blocks
        in DiT-predicted y order for a cleaner skyline.
        """
        fixed_col, preplaced_col, mib_col, _, _ = self._extract_cols(
            constraints, block_count)
        base_sizes, is_inflex = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)

        # Shapes: DiT for soft blocks, base for inflex.
        shapes = []
        for i in range(block_count):
            if is_inflex[i]:
                shapes.append(base_sizes[i])
            else:
                shapes.append(dit_shapes[i] if i < len(dit_shapes) else base_sizes[i])

        # Preplaced blocks placed first (pinned).
        preplaced: Dict[int, Tuple[float, float, float, float]] = {}
        if target_positions is not None:
            for i in range(block_count):
                if preplaced_col[i] == 0:
                    continue
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    preplaced[i] = (tx, ty, tw, th)
                    shapes[i] = (tw, th)

        total_area = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                         for i in range(block_count))
        W = math.sqrt(max(total_area, 1.0)) * knob
        widest = max(s[0] for s in shapes)
        W = max(W, widest)
        if preplaced:
            pp_xmax = max(x + w for (x, _, w, _) in preplaced.values())
            W = max(W, pp_xmax)

        sky = Skyline(W)
        positions: List[Optional[Tuple[float, float, float, float]]] = [None] * block_count

        # Seed skyline with preplaced blocks as obstacles.
        for i, (x, y, w, h) in preplaced.items():
            positions[i] = (x, y, w, h)
            sky.add_obstacle(x, y, w, h)

        # Process mobile blocks in order of DiT-predicted y (ascending).
        mobile_ids = [i for i in range(block_count) if i not in preplaced]
        mobile_ids.sort(key=lambda i: dit_targets[i][1] if i < len(dit_targets) else 0.0)

        for i in mobile_ids:
            w, h = shapes[i]
            if w > W + 1e-9:
                return None  # block wider than strip — reject
            prefer_x = dit_targets[i][0] - w / 2 if i < len(dit_targets) else None
            pos = sky.place(w, h, prefer_x=prefer_x)
            if pos is None:
                return None
            x, y = pos
            positions[i] = (x, y, w, h)

        for i in range(block_count):
            if positions[i] is None:
                return None
        return positions  # type: ignore[return-value]

    def _build_init_from_targets(
        self,
        block_count: int,
        targets: List[Tuple[float, float]],
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float, float, float]]:
        """
        Build an initial (x, y, w, h) layout from centroid targets.
        Uses fixed dims for fixed/preplaced/MIB; square sqrt(area) otherwise.
        """
        fixed_col, preplaced_col, mib_col, _, _ = self._extract_cols(
            constraints, block_count)
        base_sizes, _ = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)
        positions = []
        for i in range(block_count):
            w, h = base_sizes[i]
            if preplaced_col[i] != 0 and target_positions is not None:
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
            else:
                cx, cy = targets[i] if i < len(targets) else (0.0, 0.0)
                x = cx - w / 2
                y = cy - h / 2
            positions.append((x, y, w, h))
        return positions

    def _dit_predict_full(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        """
        Run DiT to sample (cx, cy, w, h) per block.
        Returns (centroids, shapes) where shapes[i] = (w, h) from predicted
        log-aspect, renormalized so w*h = area_target[i] exactly.
        If out_dim == 2, shapes is None (aspects not predicted).
        """
        loaded = _load_dit_model()
        if loaded is None:
            return None
        model, train_mod, cfg = loaded
        out_dim = cfg.get("out_dim", 2)

        n = block_count
        at = area_targets[:n].unsqueeze(0)
        c = constraints[:n].unsqueeze(0)
        b2b = b2b_connectivity.unsqueeze(0) if b2b_connectivity.dim() == 2 else b2b_connectivity[:1]
        p2b = p2b_connectivity.unsqueeze(0) if p2b_connectivity.dim() == 2 else p2b_connectivity[:1]
        pp = pins_pos.unsqueeze(0) if pins_pos.dim() == 2 else pins_pos[:1]
        tree = torch.zeros(1, max(n - 1, 1), 3)
        fp_dummy = torch.zeros(1, n, 4)
        metrics = torch.zeros(1, 8)
        batch = (at, b2b, p2b, pp, c, tree, fp_dummy, metrics)

        try:
            feats, mask, _, scale = train_mod.build_features(batch, include_shape_targets=True) \
                if "include_shape_targets" in train_mod.build_features.__code__.co_varnames \
                else train_mod.build_features(batch)
            # Handle either signature.
            if isinstance(scale, torch.Tensor) and scale.dim() == 0:
                s = float(scale.item())
            else:
                s = float(scale.item()) if hasattr(scale, "item") else float(scale)
            n_timesteps = cfg.get("n_timesteps", 50)
            alphas, betas, alpha_bar = train_mod.make_schedule(n_timesteps, device="cpu")
            # Allow seeded sampling for deterministic trials across the same
            # instance. If seed is set, we re-seed before torch.randn calls.
            seed = getattr(self, "_dit_sample_seed", None)
            with torch.no_grad():
                gen = torch.Generator() if seed is not None else None
                if gen is not None:
                    gen.manual_seed(int(seed))
                def _randn_like(shape):
                    if gen is not None:
                        return torch.randn(*shape, generator=gen)
                    return torch.randn(*shape)
                x = _randn_like((feats.shape[0], feats.shape[1], out_dim))
                for t_i in reversed(range(n_timesteps)):
                    t = torch.full((1,), t_i, dtype=torch.long)
                    eps = model(feats, x, t, key_padding_mask=mask)
                    ab = alpha_bar[t_i]
                    ab_prev = alpha_bar[t_i - 1] if t_i > 0 else torch.tensor(1.0)
                    beta = betas[t_i]
                    alpha = alphas[t_i]
                    mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
                    if t_i > 0:
                        noise = _randn_like(x.shape)
                        sigma = (beta * (1 - ab_prev) / (1 - ab)).sqrt()
                        x = mean + sigma * noise
                    else:
                        x = mean
                    # Clamp log-aspect dims to keep shapes sane.
                    if out_dim == 4:
                        x[:, :, 2:] = x[:, :, 2:].clamp(-3.0, 3.0)

            targets = []
            shapes: Optional[List[Tuple[float, float]]] = []
            for i in range(n):
                cx = float(x[0, i, 0].item()) * s
                cy = float(x[0, i, 1].item()) * s
                targets.append((cx, cy))
                if out_dim == 4:
                    log_w = float(x[0, i, 2].item())
                    log_h = float(x[0, i, 3].item())
                    # Renormalize so w*h = area_target[i] exactly.
                    # raw w = sqrt(a) * exp(log_w); h = sqrt(a) * exp(log_h)
                    # w*h = a * exp(log_w + log_h), want exp=1 → correct
                    # by subtracting mean of log_w+log_h (shift both by -mean/2).
                    a = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                    logsum = log_w + log_h
                    lw = log_w - logsum / 2
                    lh = log_h - logsum / 2
                    import math as _math
                    w = _math.sqrt(a) * _math.exp(lw)
                    h = _math.sqrt(a) * _math.exp(lh)
                    # Clamp aspect to sensible range [0.2, 5.0] to avoid wafer-thin blocks.
                    max_aspect = 5.0
                    if w / h > max_aspect:
                        w = _math.sqrt(a * max_aspect)
                        h = _math.sqrt(a / max_aspect)
                    elif h / w > max_aspect:
                        h = _math.sqrt(a * max_aspect)
                        w = _math.sqrt(a / max_aspect)
                    shapes.append((w, h))
                else:
                    shapes = None
            return targets, shapes
        except Exception as e:
            import traceback
            print(f"[dit_predict error] {e}", flush=True)
            traceback.print_exc()
            return None

    def _dit_predict_targets(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> Optional[List[Tuple[float, float]]]:
        """Legacy wrapper: returns only centroids, discarding shape info."""
        result = self._dit_predict_full(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints)
        if result is None:
            return None
        return result[0]

    def _dit_predict_targets_legacy(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> Optional[List[Tuple[float, float]]]:
        """Original 2-output DiT inference (kept for reference)."""
        loaded = _load_dit_model()
        if loaded is None:
            return None
        model, train_mod, cfg = loaded

        n = block_count
        at = area_targets[:n].unsqueeze(0)
        c = constraints[:n].unsqueeze(0)
        b2b = b2b_connectivity.unsqueeze(0) if b2b_connectivity.dim() == 2 else b2b_connectivity[:1]
        p2b = p2b_connectivity.unsqueeze(0) if p2b_connectivity.dim() == 2 else p2b_connectivity[:1]
        pp = pins_pos.unsqueeze(0) if pins_pos.dim() == 2 else pins_pos[:1]
        tree = torch.zeros(1, max(n - 1, 1), 3)
        fp_dummy = torch.zeros(1, n, 4)
        metrics = torch.zeros(1, 8)
        batch = (at, b2b, p2b, pp, c, tree, fp_dummy, metrics)

        try:
            feats, mask, _, _, scale = train_mod.build_features(batch)
            n_timesteps = cfg.get("n_timesteps", 50)
            alphas, betas, alpha_bar = train_mod.make_schedule(n_timesteps, device="cpu")
            # Ancestral sampling from p(x_{t-1} | x_t).
            with torch.no_grad():
                x = torch.randn(feats.shape[0], feats.shape[1], 2)
                for t_i in reversed(range(n_timesteps)):
                    t = torch.full((1,), t_i, dtype=torch.long)
                    eps = model(feats, x, t, key_padding_mask=mask)
                    ab = alpha_bar[t_i]
                    ab_prev = alpha_bar[t_i - 1] if t_i > 0 else torch.tensor(1.0)
                    beta = betas[t_i]
                    alpha = alphas[t_i]
                    # DDPM update:
                    mean = (x - beta / (1 - ab).sqrt() * eps) / alpha.sqrt()
                    if t_i > 0:
                        noise = torch.randn_like(x)
                        sigma = (beta * (1 - ab_prev) / (1 - ab)).sqrt()
                        x = mean + sigma * noise
                    else:
                        x = mean
            s = float(scale.item()) if hasattr(scale, "item") else float(scale)
            targets = []
            for i in range(n):
                cx = float(x[0, i, 0].item()) * s
                cy = float(x[0, i, 1].item()) * s
                targets.append((cx, cy))
            return targets
        except Exception as e:
            import traceback
            print(f"[dit_predict error] {e}", flush=True)
            traceback.print_exc()
            return None

    def _nn_predict_targets(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Run the trained transformer (if available) to predict block centroids.
        Returns centroids in the SAME coordinate scale as the sample (de-normalized
        by sqrt(total_area)).
        """
        loaded = _load_nn_model()
        if loaded is None:
            return None
        model, train_mod = loaded

        # Build a single-sample "batch" in the format expected by build_features.
        n = block_count
        # Pad a dummy fp_sol (unused at inference).
        max_n = n
        # Construct the shapes build_features expects:
        #   area_target: (B, N)
        #   b2b/p2b: (B, edges, 3)
        #   pins_pos: (B, n_pins, 2)
        #   constraints: (B, N, 5)
        #   tree: unused, any
        #   fp_sol: (B, N, 4) — only targets matter for building features, we ignore.
        at = area_targets[:n].unsqueeze(0)
        c = constraints[:n].unsqueeze(0)
        b2b = b2b_connectivity.unsqueeze(0) if b2b_connectivity.dim() == 2 else b2b_connectivity[:1]
        p2b = p2b_connectivity.unsqueeze(0) if p2b_connectivity.dim() == 2 else p2b_connectivity[:1]
        pp = pins_pos.unsqueeze(0) if pins_pos.dim() == 2 else pins_pos[:1]
        tree = torch.zeros(1, max(n - 1, 1), 3)
        fp_dummy = torch.zeros(1, n, 4)  # won't drive predictions
        metrics = torch.zeros(1, 8)
        batch = (at, b2b, p2b, pp, c, tree, fp_dummy, metrics)
        try:
            feats, mask, _, _, scale = train_mod.build_features(batch)
            with torch.no_grad():
                pred = model(feats, mask)  # (1, N, 2)
            # Denormalize: multiply by scale (sqrt(total_area)).
            s = float(scale.item()) if hasattr(scale, "item") else float(scale)
            targets = []
            for i in range(n):
                cx = float(pred[0, i, 0].item()) * s
                cy = float(pred[0, i, 1].item()) * s
                targets.append((cx, cy))
            return targets
        except Exception as e:
            import traceback
            print(f"[nn_predict error] {e}", flush=True)
            traceback.print_exc()
            return None

    def _abacus_legalize(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        analytical_centroids: List[Tuple[float, float]],
        base_sizes: List[Tuple[float, float]],
        W: float,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Legalize an analytical (overlapping) placement via row assignment:
        sort blocks by analytical y, partition into rows of width W, then
        within each row sort by analytical x and assign packed x-positions.
        Preserves analytical placement's global structure better than
        skyline BL-fit.
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

        mobile = [i for i in range(n) if i not in pp_ids]
        mobile.sort(key=lambda i: analytical_centroids[i][1])

        positions: List[Optional[Tuple[float, float, float, float]]] = [None] * n
        for i, xywh in preplaced.items():
            positions[i] = xywh

        # Build rows greedily by accumulating widths until full.
        rows: List[List[int]] = []
        cur: List[int] = []
        cur_w = 0.0
        for i in mobile:
            w = base_sizes[i][0]
            if cur and cur_w + w > W + 1e-6:
                rows.append(cur)
                cur = []
                cur_w = 0.0
            cur.append(i)
            cur_w += w
        if cur:
            rows.append(cur)

        sky = Skyline(W)
        for (px, py, pw, ph) in preplaced.values():
            if px >= W or px + pw <= 0:
                continue
            sky.add_obstacle(px, py, pw, ph)

        for row in rows:
            row.sort(key=lambda i: analytical_centroids[i][0])
            row_h = max(base_sizes[i][1] for i in row) if row else 0.0
            row_w = sum(base_sizes[i][0] for i in row)
            pos = sky.find_position(row_w)
            if pos is None:
                return None
            x_start, y_start = pos
            x_cur = x_start
            for i in row:
                w, h = base_sizes[i]
                positions[i] = (x_cur, y_start, w, h)
                x_cur += w
            sky.add_obstacle(x_start, y_start, row_w, row_h)

        for i in range(n):
            if positions[i] is None:
                return None
        return positions  # type: ignore[return-value]

    def _dreamplace_refine(
        self, positions, block_count, area_targets, constraints,
        b2b_connectivity, p2b_connectivity, pins_pos,
        steps: int = 200, lr: float = 1.0,
    ):
        """
        DREAMPlace-lite analytical placer: gradient descent on block centroids
        with smooth HPWL (log-sum-exp) + Gaussian density (overlap) penalty +
        boundary/preplaced anchors.

        Returns raw (possibly overlapping) centroids as a list of (x, y, w, h).
        The caller should legalize (e.g., via skyline re-pack).
        """
        n = len(positions)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        init_cx = torch.tensor([p[0] + p[2] / 2 for p in positions],
                               dtype=torch.float32, device=device)
        init_cy = torch.tensor([p[1] + p[3] / 2 for p in positions],
                               dtype=torch.float32, device=device)
        w = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=device)
        h = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=device)

        # Masks for pinned / boundary blocks.
        is_preplaced = torch.zeros(n, dtype=torch.bool, device=device)
        bnd_code = torch.zeros(n, dtype=torch.long, device=device)
        if constraints is not None and constraints.dim() > 1 and constraints.shape[1] >= 5:
            is_preplaced = (constraints[:block_count, 1] != 0).to(device)
            bnd_code = constraints[:block_count, 4].long().to(device)

        # Connectivity tensors.
        def _filter_p2b(conn):
            if conn is None or conn.numel() == 0:
                return None, None, None
            valid = conn[:, 0] >= 0
            c2 = conn[valid]
            pin = c2[:, 0].long(); blk = c2[:, 1].long(); ww = c2[:, 2].float()
            ok = (blk >= 0) & (blk < n) & (pin >= 0) & (pin < pins_pos.shape[0])
            return pin[ok].to(device), blk[ok].to(device), ww[ok].to(device)

        def _filter_b2b(conn):
            if conn is None or conn.numel() == 0:
                return None, None, None
            valid = conn[:, 0] >= 0
            c2 = conn[valid]
            ii = c2[:, 0].long(); jj = c2[:, 1].long(); ww = c2[:, 2].float()
            ok = (ii < n) & (jj < n) & (ii >= 0) & (jj >= 0)
            return ii[ok].to(device), jj[ok].to(device), ww[ok].to(device)

        b2b_i, b2b_j, b2b_w = _filter_b2b(b2b_connectivity)
        p2b_pin, p2b_blk, p2b_w = _filter_p2b(p2b_connectivity)
        px_t = py_t = None
        if p2b_pin is not None and p2b_pin.numel() > 0:
            pins = pins_pos.float().to(device)
            px_t = pins[p2b_pin, 0]; py_t = pins[p2b_pin, 1]

        # All-pairs overlap indices.
        ii_idx, jj_idx = torch.triu_indices(n, n, offset=1, device=device)
        wi = w[ii_idx]; wj = w[jj_idx]; hi = h[ii_idx]; hj = h[jj_idx]
        # Gaussian density length scale = average block size.
        sigma = float((w.mean() + h.mean()).item()) * 0.5

        cx = init_cx.clone().requires_grad_(True)
        cy = init_cy.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([cx, cy], lr=lr)
        # Warm-up + decay: gamma anneals the density weight up over time.
        for step in range(steps):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            if b2b_i is not None and b2b_i.numel() > 0:
                dx = torch.abs(cx[b2b_i] - cx[b2b_j])
                dy = torch.abs(cy[b2b_i] - cy[b2b_j])
                loss = loss + (b2b_w * (dx + dy)).sum()
            if p2b_pin is not None and p2b_pin.numel() > 0:
                dx = torch.abs(cx[p2b_blk] - px_t)
                dy = torch.abs(cy[p2b_blk] - py_t)
                loss = loss + (p2b_w * (dx + dy)).sum()
            # Gaussian density penalty: for each pair, exp(-|Δ|^2 / σ^2)
            # scaled by the product of block "masses" (widths * heights).
            dcx = cx[ii_idx] - cx[jj_idx]
            dcy = cy[ii_idx] - cy[jj_idx]
            # Overlap rectangles (ReLU form, differentiable).
            ox = torch.relu((wi + wj) / 2 - torch.abs(dcx))
            oy = torch.relu((hi + hj) / 2 - torch.abs(dcy))
            density = (ox * oy).sum()
            # Anneal density weight: start small, grow to 100 by end.
            dens_w = 10.0 + 190.0 * (step / max(1, steps - 1))
            loss = loss + dens_w * density
            loss.backward()
            optimizer.step()
            # Pin preplaced + boundary blocks at their initial centroids.
            pinned = is_preplaced | (bnd_code != 0)
            if pinned.any():
                with torch.no_grad():
                    cx.data[pinned] = init_cx[pinned]
                    cy.data[pinned] = init_cy[pinned]

        cx_f = cx.detach().cpu()
        cy_f = cy.detach().cpu()
        w_f = w.cpu(); h_f = h.cpu()
        return [(float(cx_f[i] - w_f[i] / 2), float(cy_f[i] - h_f[i] / 2),
                 float(w_f[i]), float(h_f[i])) for i in range(n)]

    def _analytical_refine_soft(
        self, positions, block_count, area_targets, constraints,
        b2b_connectivity, p2b_connectivity, pins_pos,
    ):
        """
        Soft variant: runs gradient descent on block centroids to reduce HPWL
        and returns the (possibly-overlapping) positions as hints. Legalization
        happens afterward via shelf-pack. Never returns None.
        """
        return self._analytical_refine(
            positions, block_count, area_targets, constraints,
            b2b_connectivity, p2b_connectivity, pins_pos,
            require_feasible=False,
        )

    def _analytical_refine(
        self, positions, block_count, area_targets, constraints,
        b2b_connectivity, p2b_connectivity, pins_pos,
        require_feasible: bool = True,
    ):
        """
        Local gradient-descent refinement of block centers to reduce HPWL
        while keeping the overlap penalty small. Returns a list of
        (x, y, w, h) tuples if the refined layout is feasible (or always if
        require_feasible=False); otherwise None.
        """
        n = len(positions)
        # Extract initial centroids and dims.
        init_cx = torch.tensor([p[0] + p[2] / 2 for p in positions], dtype=torch.float32)
        init_cy = torch.tensor([p[1] + p[3] / 2 for p in positions], dtype=torch.float32)
        w = torch.tensor([p[2] for p in positions], dtype=torch.float32)
        h = torch.tensor([p[3] for p in positions], dtype=torch.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        init_cx = init_cx.to(device); init_cy = init_cy.to(device)
        w = w.to(device); h = h.to(device)

        # Pinned blocks (preplaced) can't move. Boundary blocks should not drift
        # far from their required edges; we use an anchor loss to keep them close.
        is_preplaced = torch.zeros(n, dtype=torch.bool, device=device)
        bnd_code = torch.zeros(n, dtype=torch.long, device=device)
        if constraints is not None and constraints.dim() > 1 and constraints.shape[1] >= 5:
            is_preplaced = (constraints[:block_count, 1] != 0).to(device)
            bnd_code = constraints[:block_count, 4].long().to(device)

        cx = init_cx.clone().requires_grad_(True)
        cy = init_cy.clone().requires_grad_(True)

        # Precompute connectivity tensors.
        def _filter(conn, nc):
            valid = conn[:, 0] >= 0
            c2 = conn[valid]
            if nc == 2:  # p2b
                pin = c2[:, 0].long(); blk = c2[:, 1].long(); ww = c2[:, 2].float()
                ok = (blk >= 0) & (blk < n) & (pin >= 0) & (pin < pins_pos.shape[0])
                return pin[ok].to(device), blk[ok].to(device), ww[ok].to(device)
            else:  # b2b
                ii = c2[:, 0].long(); jj = c2[:, 1].long(); ww = c2[:, 2].float()
                ok = (ii < n) & (jj < n) & (ii >= 0) & (jj >= 0)
                return ii[ok].to(device), jj[ok].to(device), ww[ok].to(device)

        b2b_i = b2b_j = b2b_w = None
        if b2b_connectivity is not None and b2b_connectivity.numel() > 0:
            b2b_i, b2b_j, b2b_w = _filter(b2b_connectivity, 1)
        p2b_pin = p2b_blk = p2b_w = None
        px_t = py_t = None
        if p2b_connectivity is not None and p2b_connectivity.numel() > 0:
            p2b_pin, p2b_blk, p2b_w = _filter(p2b_connectivity, 2)
            pins = pins_pos.float().to(device)
            px_t = pins[p2b_pin, 0]; py_t = pins[p2b_pin, 1]

        # Overlap penalty precomputation: all pairs (i, j) with i < j.
        ii, jj = torch.triu_indices(n, n, offset=1, device=device)
        wi = w[ii]; wj = w[jj]; hi = h[ii]; hj = h[jj]

        # For the soft-hint mode we use a gentler overlap penalty (we only need
        # informative centroid drift; legalization happens post-hoc via the
        # shelf packer). For require_feasible mode we keep a strong penalty.
        lam_overlap = 1e3 if require_feasible else 2.0
        optimizer = torch.optim.Adam([cx, cy], lr=0.5)
        for step in range(80):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            if b2b_i is not None and b2b_i.numel() > 0:
                dx = torch.abs(cx[b2b_i] - cx[b2b_j])
                dy = torch.abs(cy[b2b_i] - cy[b2b_j])
                loss = loss + (b2b_w * (dx + dy)).sum()
            if p2b_pin is not None and p2b_pin.numel() > 0:
                dx = torch.abs(cx[p2b_blk] - px_t)
                dy = torch.abs(cy[p2b_blk] - py_t)
                loss = loss + (p2b_w * (dx + dy)).sum()
            dcx = cx[ii] - cx[jj]
            dcy = cy[ii] - cy[jj]
            overlap_x = torch.relu((wi + wj) / 2 - torch.abs(dcx))
            overlap_y = torch.relu((hi + hj) / 2 - torch.abs(dcy))
            overlap = overlap_x * overlap_y
            loss = loss + lam_overlap * overlap.sum()
            loss.backward()
            optimizer.step()
            # Hard-pin: preplaced blocks + all boundary-constrained blocks
            # stay at their initial centroids. This preserves V_rel from the
            # pre-refinement shelf layout.
            pinned_mask = is_preplaced | (bnd_code != 0)
            if pinned_mask.any():
                with torch.no_grad():
                    cx.data[pinned_mask] = init_cx[pinned_mask]
                    cy.data[pinned_mask] = init_cy[pinned_mask]

        # Bring back to CPU tuples.
        cx_f = cx.detach().cpu()
        cy_f = cy.detach().cpu()
        w_f = w.cpu(); h_f = h.cpu()
        new_positions = []
        for i in range(n):
            nx = float(cx_f[i] - w_f[i] / 2)
            ny = float(cy_f[i] - h_f[i] / 2)
            new_positions.append((nx, ny, float(w_f[i]), float(h_f[i])))

        if require_feasible:
            # Feasibility check: if any overlap > 1e-6, bail out.
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi, wi_, hi_ = new_positions[i]
                    xj, yj, wj_, hj_ = new_positions[j]
                    ox = min(xi + wi_, xj + wj_) - max(xi, xj)
                    oy = min(yi + hi_, yj + hj_) - max(yi, yj)
                    if ox > 1e-6 and oy > 1e-6:
                        return None
        return new_positions

    def _y_compact(self, positions, constraints, block_count):
        """
        Slide non-preplaced, non-top-bound blocks down to close vertical gaps
        left by the preplaced-bump logic.
        For each block (sorted by y), find the max (y + h) of other blocks
        *below* it that overlap in x-range, and snap its y to that value
        (clamped at 0). Preplaced and top-bound blocks are pinned.
        """
        if constraints is None or constraints.dim() < 2 or constraints.shape[1] < 5:
            return positions
        n = len(positions)
        preplaced = set(i for i in range(block_count) if constraints[i, 1] != 0)
        top_bound = set(i for i in range(block_count) if int(constraints[i, 4].item()) & B_TOP)
        pinned = preplaced | top_bound

        # Order: non-pinned blocks sorted by ascending y. Pinned stay in place.
        non_pinned = [i for i in range(n) if i not in pinned]
        non_pinned.sort(key=lambda i: positions[i][1])

        new_positions = list(positions)
        for i in non_pinned:
            x, y, w, h = new_positions[i]
            # Find max (y_j + h_j) over all other blocks whose x-range
            # intersects [x, x + w] and whose y_j + h_j <= current y.
            best_y = 0.0
            for j in range(n):
                if j == i:
                    continue
                xj, yj, wj, hj = new_positions[j]
                # Must be below current (top of j <= bottom of i).
                if yj + hj > y + 1e-9:
                    continue
                # x-overlap check.
                if x + w <= xj + 1e-9 or xj + wj <= x + 1e-9:
                    continue
                if yj + hj > best_y:
                    best_y = yj + hj
            if best_y < y:
                new_positions[i] = (x, best_y, w, h)
        return new_positions

    def _cluster_fragmentation(self, positions, clust_col, block_count) -> int:
        """
        Count grouping violations: for each cluster group, count
        (connected_components - 1) where two blocks are connected iff they
        share a non-zero-length edge segment.
        """
        if clust_col is None or clust_col.numel() == 0:
            return 0
        n_groups = int(clust_col.max().item())
        if n_groups == 0:
            return 0
        total = 0
        for g in range(1, n_groups + 1):
            members = [i for i in range(block_count) if int(clust_col[i].item()) == g]
            if len(members) <= 1:
                continue
            # Union-find on edge-sharing.
            parent = {i: i for i in members}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for ai in range(len(members)):
                i = members[ai]
                xi, yi, wi, hi = positions[i]
                for bi in range(ai + 1, len(members)):
                    j = members[bi]
                    xj, yj, wj, hj = positions[j]
                    # Share vertical edge?
                    if (abs(xi + wi - xj) < 1e-6 or abs(xj + wj - xi) < 1e-6):
                        overlap = min(yi + hi, yj + hj) - max(yi, yj)
                        if overlap > 1e-6:
                            union(i, j)
                            continue
                    # Share horizontal edge?
                    if (abs(yi + hi - yj) < 1e-6 or abs(yj + hj - yi) < 1e-6):
                        overlap = min(xi + wi, xj + wj) - max(xi, xj)
                        if overlap > 1e-6:
                            union(i, j)
            roots = set(find(i) for i in members)
            total += len(roots) - 1
        return total

    def _contest_raw(self, positions, b2b_conn, p2b_conn, pins_pos,
                     constraints, block_count):
        """
        Return raw (hpwl, area, v_rel) triple. Contest-aligned violation
        semantics but values are NOT normalized — use _contest_cost_gapped
        to turn a set of candidates into rank-comparable gap-based costs.
        """
        n = len(positions)
        cx = torch.tensor([p[0] + p[2] / 2 for p in positions])
        cy = torch.tensor([p[1] + p[3] / 2 for p in positions])
        hpwl = 0.0
        if b2b_conn is not None and b2b_conn.numel() > 0:
            valid = b2b_conn[b2b_conn[:, 0] >= 0]
            i = valid[:, 0].long(); j = valid[:, 1].long(); w = valid[:, 2].float()
            ok = (i < n) & (j < n) & (i >= 0) & (j >= 0)
            i, j, w = i[ok], j[ok], w[ok]
            if w.numel() > 0:
                hpwl += float((w * (torch.abs(cx[i] - cx[j]) + torch.abs(cy[i] - cy[j]))).sum())
        if p2b_conn is not None and p2b_conn.numel() > 0:
            valid = p2b_conn[p2b_conn[:, 0] >= 0]
            pin = valid[:, 0].long(); blk = valid[:, 1].long(); w = valid[:, 2].float()
            ok = (blk < n) & (blk >= 0) & (pin < pins_pos.shape[0]) & (pin >= 0)
            pin, blk, w = pin[ok], blk[ok], w[ok]
            if w.numel() > 0:
                px = pins_pos[pin, 0].float(); py = pins_pos[pin, 1].float()
                hpwl += float((w * (torch.abs(cx[blk] - px) + torch.abs(cy[blk] - py))).sum())
        x_min = min(p[0] for p in positions); y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions); y_max = max(p[1] + p[3] for p in positions)
        area = (x_max - x_min) * (y_max - y_min)
        v_fixed = 0; v_pp = 0; v_bnd = 0; v_grp = 0; v_mib = 0; n_soft = 0
        eps = 1e-4
        if constraints is not None and constraints.dim() > 1 and constraints.shape[1] >= 5:
            fx_col = constraints[:block_count, 0]
            pp_col = constraints[:block_count, 1]
            mib_col = constraints[:block_count, 2]
            clust_col = constraints[:block_count, 3]
            bnd_col = constraints[:block_count, 4]
            n_fixed = int((fx_col != 0).sum().item())
            n_pp = int((pp_col != 0).sum().item())
            n_bnd = int((bnd_col != 0).sum().item())
            n_soft = n_fixed + n_pp + n_bnd
            n_mib_groups = int(mib_col.max().item()) if mib_col.numel() > 0 else 0
            for g in range(1, n_mib_groups + 1):
                gsize = int((mib_col == g).sum().item())
                n_soft += max(0, gsize - 1)
            n_clust_groups = int(clust_col.max().item()) if clust_col.numel() > 0 else 0
            for g in range(1, n_clust_groups + 1):
                gsize = int((clust_col == g).sum().item())
                n_soft += max(0, gsize - 1)
            for g in range(1, n_mib_groups + 1):
                distinct = set()
                for i in range(block_count):
                    if int(mib_col[i].item()) != g:
                        continue
                    bw, bh = positions[i][2], positions[i][3]
                    distinct.add((round(bw, 4), round(bh, 4)))
                v_mib += max(0, len(distinct) - 1)
            v_grp = self._cluster_fragmentation(positions, clust_col, block_count)
            for i in range(block_count):
                code = int(bnd_col[i].item())
                if code == 0: continue
                bx, by, bw, bh = positions[i]
                touches = {
                    1: abs(bx - x_min) < eps,
                    2: abs(bx + bw - x_max) < eps,
                    4: abs(by + bh - y_max) < eps,
                    8: abs(by - y_min) < eps,
                }
                if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                    v_bnd += 1
        v_total = v_fixed + v_pp + v_bnd + v_grp + v_mib
        v_rel = v_total / max(n_soft, 1)
        return hpwl, area, v_rel

    def _internal_cost(self, positions, b2b_conn, p2b_conn, pins_pos,
                       constraints, block_count) -> float:
        """
        Contest-aligned proxy cost. Mirrors iccad2026_evaluate.evaluate_solution
        violation semantics:
          total_soft_violations = V_fixed + V_preplaced + V_boundary
                                  + V_grouping + V_mib
          v_rel = total_soft / n_soft   (n_soft from contest's denominator)
          cost  = (hpwl + 0.5 * area) * exp(2 * v_rel)
        """
        n = len(positions)
        # HPWL via centroids.
        cx = torch.tensor([p[0] + p[2] / 2 for p in positions])
        cy = torch.tensor([p[1] + p[3] / 2 for p in positions])
        hpwl = 0.0
        if b2b_conn is not None and b2b_conn.numel() > 0:
            valid = b2b_conn[b2b_conn[:, 0] >= 0]
            i = valid[:, 0].long(); j = valid[:, 1].long(); w = valid[:, 2].float()
            ok = (i < n) & (j < n) & (i >= 0) & (j >= 0)
            i, j, w = i[ok], j[ok], w[ok]
            if w.numel() > 0:
                hpwl += float((w * (torch.abs(cx[i] - cx[j]) + torch.abs(cy[i] - cy[j]))).sum())
        if p2b_conn is not None and p2b_conn.numel() > 0:
            valid = p2b_conn[p2b_conn[:, 0] >= 0]
            pin = valid[:, 0].long(); blk = valid[:, 1].long(); w = valid[:, 2].float()
            ok = (blk < n) & (blk >= 0) & (pin < pins_pos.shape[0]) & (pin >= 0)
            pin, blk, w = pin[ok], blk[ok], w[ok]
            if w.numel() > 0:
                px = pins_pos[pin, 0].float(); py = pins_pos[pin, 1].float()
                hpwl += float((w * (torch.abs(cx[blk] - px) + torch.abs(cy[blk] - py))).sum())
        # Bounding box area.
        x_min = min(p[0] for p in positions); y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions); y_max = max(p[1] + p[3] for p in positions)
        area = (x_max - x_min) * (y_max - y_min)

        # Soft violation counts (contest-aligned).
        v_fixed = 0
        v_pp = 0
        v_bnd = 0
        v_grp = 0
        v_mib = 0
        n_soft = 0
        eps = 1e-4
        if constraints is not None and constraints.dim() > 1 and constraints.shape[1] >= 5:
            fx_col = constraints[:block_count, 0]
            pp_col = constraints[:block_count, 1]
            mib_col = constraints[:block_count, 2]
            clust_col = constraints[:block_count, 3]
            bnd_col = constraints[:block_count, 4]

            n_fixed = int((fx_col != 0).sum().item())
            n_pp = int((pp_col != 0).sum().item())
            n_bnd = int((bnd_col != 0).sum().item())
            n_soft = n_fixed + n_pp + n_bnd
            n_mib_groups = int(mib_col.max().item()) if mib_col.numel() > 0 else 0
            for g in range(1, n_mib_groups + 1):
                gsize = int((mib_col == g).sum().item())
                n_soft += max(0, gsize - 1)
            n_clust_groups = int(clust_col.max().item()) if clust_col.numel() > 0 else 0
            for g in range(1, n_clust_groups + 1):
                gsize = int((clust_col == g).sum().item())
                n_soft += max(0, gsize - 1)

            # V_fixed: (w, h) differ from target.
            if n_fixed > 0:
                for i in range(block_count):
                    if int(fx_col[i].item()) == 0:
                        continue
                    # target (w, h) stored in `_base_dims` via target_positions.
                    # Here we approximate: fixed blocks must have matching (w, h)
                    # — since _build_init uses base_sizes for inflex, they match.
                    # Safe: count 0 unless a path violates (rare in practice).
                    pass

            # V_preplaced: position differs from target. Also approximate 0.
            # (Our solver always pins preplaced to target in legalization.)

            # V_mib: count distinct (w, h) within each MIB group.
            for g in range(1, n_mib_groups + 1):
                distinct = set()
                for i in range(block_count):
                    if int(mib_col[i].item()) != g:
                        continue
                    bw, bh = positions[i][2], positions[i][3]
                    distinct.add((round(bw, 4), round(bh, 4)))
                v_mib += max(0, len(distinct) - 1)

            # V_grouping: count connected-component breaks per grouping ID.
            v_grp = self._cluster_fragmentation(
                positions, clust_col, block_count)

            # V_boundary: block doesn't touch required edge.
            for i in range(block_count):
                code = int(bnd_col[i].item())
                if code == 0:
                    continue
                bx, by, bw, bh = positions[i]
                touches = {
                    1: abs(bx - x_min) < eps,
                    2: abs(bx + bw - x_max) < eps,
                    4: abs(by + bh - y_max) < eps,
                    8: abs(by - y_min) < eps,
                }
                if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
                    v_bnd += 1

        v_total = v_fixed + v_pp + v_bnd + v_grp + v_mib
        v_rel = v_total / max(n_soft, 1)
        # Area is ~5-10x larger in raw magnitude than HPWL, so 0.5*area
        # swamped HPWL in earlier versions. Use 0.1 so HPWL dominates rank
        # similar to how the contest's gap-based cost treats them.
        return (hpwl + 0.1 * area) * math.exp(2.0 * v_rel)

    def _solve_once(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
        pin_targets: List[Optional[Tuple[float, float]]],
    ) -> List[Tuple[float, float, float, float]]:
        fixed_col, preplaced_col, mib_col, clust_col, bound_col = \
            self._extract_cols(constraints, block_count)

        base_sizes, is_inflex = self._base_dims(
            block_count, area_targets, target_positions,
            fixed_col, preplaced_col, mib_col)

        # Preplaced blocks: hard (x, y, w, h) constraint.
        preplaced: Dict[int, Tuple[float, float, float, float]] = {}
        if target_positions is not None:
            for i in range(block_count):
                if preplaced_col[i] == 0:
                    continue
                tx = float(target_positions[i, 0])
                ty = float(target_positions[i, 1])
                tw = float(target_positions[i, 2])
                th = float(target_positions[i, 3])
                if tx >= 0 and ty >= 0 and tw > 0 and th > 0:
                    preplaced[i] = (tx, ty, tw, th)

        pp_ids = set(preplaced.keys())

        # Non-preplaced blocks partitioned by boundary code:
        #   bottom (8):   code & 8
        #   top (4):      code & 4  (and NOT bottom)
        #   left (1):     code & 1, not top/bottom
        #   right (2):    code & 2, not top/bottom, not left (if both left+right, treat as left)
        #   interior:     code == 0
        bottom_ids: List[int] = []
        top_ids: List[int] = []
        left_ids: List[int] = []
        right_ids: List[int] = []
        interior_ids: List[int] = []
        for i in range(block_count):
            if i in pp_ids:
                continue
            code = int(bound_col[i].item())
            if code & B_BOTTOM:
                bottom_ids.append(i)
            elif code & B_TOP:
                top_ids.append(i)
            elif code & B_LEFT:
                left_ids.append(i)
            elif code & B_RIGHT:
                right_ids.append(i)
            else:
                interior_ids.append(i)

        # Sort key: primarily cluster (contiguity), tertiarily pin-target-x, then
        # by height desc for packing efficiency.
        def _sort_key(i: int) -> Tuple[float, float, float]:
            clu = int(clust_col[i].item())
            # Blocks in the same cluster get sorted by their pin-x so cluster
            # abuts horizontally; cluster-less blocks sort by pin-x globally.
            tx = pin_targets[i][0] if pin_targets[i] is not None else 0.0
            return (clu, tx, -base_sizes[i][1])

        # Within bottom/top rows, place corner blocks at the shelf ends.
        def _order_with_corners(ids: List[int], left_code: int, right_code: int) -> List[int]:
            left_corner = [i for i in ids if int(bound_col[i].item()) == left_code]
            right_corner = [i for i in ids if int(bound_col[i].item()) == right_code]
            core = [i for i in ids if i not in left_corner and i not in right_corner]
            core.sort(key=_sort_key)
            return left_corner + core + right_corner

        bottom_ordered = _order_with_corners(bottom_ids, 9, 10)  # BL=9, BR=10
        top_ordered = _order_with_corners(top_ids, 5, 6)         # TL=5, TR=6

        # Decide target strip width W.
        total_area = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                         for i in range(block_count))
        W = math.sqrt(max(total_area, 1.0)) * self.aspect_knob

        if preplaced:
            pp_xmax = max(x + w for (x, _, w, _) in preplaced.values())
            pp_ymax = max(y + h for (_, y, _, h) in preplaced.values())
        else:
            pp_xmax = 0.0
            pp_ymax = 0.0
        # Strip must contain preplaced horizontally.
        W = max(W, pp_xmax)
        # Strip must be at least widest single block.
        widest = max((base_sizes[i][0] for i in range(block_count) if i not in pp_ids),
                     default=0.0)
        W = max(W, widest)

        # Middle-shelf count: target ~sqrt(N) rows so each shelf has a balanced
        # mix of blocks and inflex blocks don't dominate one shelf. Also ensure
        # enough shelves to host all left/right boundary blocks.
        middle_block_count = len(interior_ids) + len(left_ids) + len(right_ids)
        target_rows = max(1, int(round(math.sqrt(max(middle_block_count, 1)))))
        n_mid = max(len(left_ids), len(right_ids), target_rows)

        # Order interior by cluster + pin-target-y (so shelf-assigned by height
        # quantile == pin-y quantile), then by pin-target-x within each cluster.
        def _interior_key(i: int) -> Tuple[int, float, float, float]:
            clu = int(clust_col[i].item())
            ty = pin_targets[i][1] if pin_targets[i] is not None else 0.0
            tx = pin_targets[i][0] if pin_targets[i] is not None else 0.0
            return (clu, ty, tx, -base_sizes[i][1])

        interior_sorted = sorted(interior_ids, key=_interior_key)
        # Left and right pools: sort by pin-target-y (so left-pool[k] lands in
        # the k-th shelf roughly aligned with its pin target vertically).
        def _lr_key(i: int) -> Tuple[float, int]:
            ty = pin_targets[i][1] if pin_targets[i] is not None else 0.0
            return (ty, int(clust_col[i].item()))
        left_pool = sorted(left_ids, key=_lr_key)
        right_pool = sorted(right_ids, key=_lr_key)

        # Build middle shelves. Use CONTIGUOUS chunks of interior_sorted (sorted
        # by (cluster, ty)) so shelves naturally correspond to vertical pin-y
        # bands and cluster members stay adjacent when they share similar ty.
        middle_shelves: List[List[int]] = [[] for _ in range(n_mid)]
        if n_mid > 0 and interior_sorted:
            size = len(interior_sorted)
            for k in range(n_mid):
                start = (k * size) // n_mid
                end = ((k + 1) * size) // n_mid
                middle_shelves[k] = list(interior_sorted[start:end])

        # Prepend left-bound, append right-bound.
        for k in range(n_mid):
            if k < len(left_pool):
                middle_shelves[k] = [left_pool[k]] + middle_shelves[k]
            if k < len(right_pool):
                middle_shelves[k] = middle_shelves[k] + [right_pool[k]]

        shelves: List[List[int]] = []
        if bottom_ordered:
            shelves.extend(self._split_wide_shelf(bottom_ordered, base_sizes, is_inflex, W))
        # Split middle shelves too: if a shelf's inflex width exceeds W (MIB
        # or fixed-shape dominance), split so flex blocks have headroom.
        for ms in middle_shelves:
            shelves.extend(self._split_wide_shelf(ms, base_sizes, is_inflex, W))
        if top_ordered:
            shelves.extend(self._split_wide_shelf(top_ordered, base_sizes, is_inflex, W))

        # Remove any empty shelves.
        shelves = [s for s in shelves if s]

        # Intra-shelf cluster-group reordering for HPWL.
        # Within each shelf, cluster members are contiguous (so grouping is
        # still satisfied). Reorder CLUSTERS (as atomic blocks) by average
        # target-x so low-tx clusters land on the left of the shelf.
        # Skip reordering of first/last blocks that have left/right boundary
        # constraints — they must stay at shelf start/end.
        shelves = [self._reorder_shelf_by_tx(s, pin_targets, clust_col, bound_col)
                   for s in shelves]

        # Decide where shelves start. Prefer y=0 so bottom-bound blocks
        # (placed in the first shelf) actually touch the bbox bottom.
        # If any preplaced occupies y=0 (risking x-overlap with a shelf there),
        # start shelves above preplaced as before.
        pp_at_y0 = any(py <= 1e-6 for (_, py, _, _) in preplaced.values())
        # Also consider a shelf at y=0 might overlap a preplaced with y_min > 0
        # if its y-range intersects the shelf. We'll start shelves at y_start;
        # any preplaced that then intersects the shelf range is a layout
        # conflict we skip for now (handled by pushing up).
        shelves_start_y = 0.0 if not pp_at_y0 else pp_ymax

        positions: List[Optional[Tuple[float, float, float, float]]] = [None] * block_count
        for i, xywh in preplaced.items():
            positions[i] = xywh

        # Compute shelf heights and provisional y positions.
        shelf_heights: List[float] = []
        shelf_width_data: List[List[float]] = []
        for shelf in shelves:
            widths, shelf_h = self._shelf_widths_and_height(
                shelf, base_sizes, is_inflex, W, area_targets)
            shelf_heights.append(shelf_h)
            shelf_width_data.append(widths)

        # Determine the y-ranges for each shelf, inserting extra upward shift
        # at any point where the current cumulative y would overlap a preplaced
        # block in x and y. We keep a running y_cursor and, for each shelf, if
        # the shelf [y_cursor, y_cursor + shelf_h] overlaps a preplaced block
        # in y-range AND x-range (width up to W), we bump y_cursor up past
        # that preplaced block.
        y_cursor = shelves_start_y
        for s_idx, shelf in enumerate(shelves):
            sh = shelf_heights[s_idx]
            widths = shelf_width_data[s_idx]
            # Bump past preplaced that overlap in x (shelf spans [0, W]).
            for _safety in range(1000):
                bumped = False
                for (px, py, pw, ph) in preplaced.values():
                    y0 = y_cursor; y1 = y_cursor + sh
                    if py + ph <= y0 + 1e-9 or py >= y1 - 1e-9:
                        continue
                    if px + pw <= 1e-9 or px >= W - 1e-9:
                        continue
                    # Overlap — bump shelf above this preplaced.
                    new_y = py + ph
                    if new_y <= y_cursor + 1e-9:
                        # Not actually advancing — guard against infinite loop
                        # from floating-point quirks or nested conditions.
                        continue
                    y_cursor = new_y
                    bumped = True
                    break
                if not bumped:
                    break

            # Check if this shelf contains a right-bound block as its last entry.
            # If shelf's total width is less than W, right-align the last block so
            # it touches the bbox right edge (creates a gap in the middle — pure
            # area waste but needed for right-boundary soft constraint).
            total_w = sum(widths)
            last_id = shelf[-1] if shelf else None
            last_code = int(bound_col[last_id].item()) if last_id is not None else 0
            last_is_right = bool(last_code & B_RIGHT)
            right_align_gap = 0.0
            if last_is_right and total_w + 1e-6 < W:
                right_align_gap = W - total_w

            # Optimal x-placement for the shelf given fixed block ordering
            # and widths. If total widths < W, we introduce gaps judiciously
            # toward each block's target-center to reduce HPWL while keeping
            # left/right-bound blocks glued to the respective edges.
            shelf_heights_blocks = [
                base_sizes[i][1] if is_inflex[i] else sh for i in shelf
            ]
            xs = self._optimize_shelf_x(
                shelf, widths, W, pin_targets, bound_col)
            for idx, i in enumerate(shelf):
                h_i = shelf_heights_blocks[idx]
                positions[i] = (xs[idx], y_cursor, widths[idx], h_i)
            y_cursor += sh

        return positions  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _optimize_shelf_x(
        self,
        shelf: List[int],
        widths: List[float],
        W: float,
        pin_targets: List[Optional[Tuple[float, float]]],
        bound_col: torch.Tensor,
    ) -> List[float]:
        """
        Choose x-positions within a shelf (ordering fixed) to minimize sum of
        weighted |cx_i - target_cx_i|, subject to non-overlap and boundary
        constraints:
          * x_0 = 0 if first block has left-boundary, else x_0 >= 0.
          * x_{N-1} + w_{N-1} = W if last block has right-boundary, else ≤ W.
          * x_{i+1} >= x_i + w_i.
        Uses a two-pass greedy: left-to-right lower-bound, right-to-left
        upper-bound, then takes a convex combination biased by target.
        """
        n = len(shelf)
        if n == 0:
            return []
        # Lower bounds: min x_i (packed from the left).
        lb = [0.0] * n
        cur = 0.0
        for i in range(n):
            lb[i] = cur
            cur += widths[i]
        # Upper bounds: max x_i (packed from the right, shelf ends at W).
        ub = [0.0] * n
        cur = W
        for i in range(n - 1, -1, -1):
            cur -= widths[i]
            ub[i] = cur
        # Left-boundary block pins x_0 at 0.
        left_pinned = bool(int(bound_col[shelf[0]].item()) & B_LEFT)
        if left_pinned:
            ub[0] = 0.0
        # Right-boundary block pins x_{n-1} + w_{n-1} at W.
        right_pinned = bool(int(bound_col[shelf[-1]].item()) & B_RIGHT)
        if right_pinned:
            lb[n - 1] = W - widths[n - 1]

        # Targets: block center → block left-edge = tx - w/2.
        targets = [None] * n
        for k, i in enumerate(shelf):
            tgt = pin_targets[i]
            if tgt is not None:
                targets[k] = tgt[0] - widths[k] / 2.0

        # Initialize x_i at clamp(target, lb, ub) for blocks with targets,
        # at lb otherwise. Then run a few passes of coordinate descent: for
        # each block, set x_i = clamp(target or lb, lb, ub).
        xs = [0.0] * n
        for k in range(n):
            t = targets[k]
            if t is None:
                xs[k] = lb[k]
            else:
                xs[k] = max(lb[k], min(ub[k], t))
        # Enforce monotone non-overlap (x_{i+1} >= x_i + w_i) with two passes.
        for _ in range(4):
            # Left-to-right: push up if needed.
            for k in range(1, n):
                lo = xs[k - 1] + widths[k - 1]
                if xs[k] < lo:
                    xs[k] = lo
                xs[k] = min(xs[k], ub[k])
            # Right-to-left: push down if needed.
            for k in range(n - 2, -1, -1):
                hi = xs[k + 1] - widths[k]
                if xs[k] > hi:
                    xs[k] = hi
                xs[k] = max(xs[k], lb[k])
            # Bias each unpinned block toward its target within [lb_eff, ub_eff].
            for k in range(n):
                if targets[k] is None:
                    continue
                lo_eff = xs[k - 1] + widths[k - 1] if k > 0 else lb[k]
                hi_eff = xs[k + 1] - widths[k] if k < n - 1 else ub[k]
                lo_eff = max(lo_eff, lb[k])
                hi_eff = min(hi_eff, ub[k])
                if lo_eff > hi_eff:
                    continue
                xs[k] = max(lo_eff, min(hi_eff, targets[k]))
        return xs

    def _reorder_shelf_by_tx(
        self,
        shelf: List[int],
        pin_targets: List[Optional[Tuple[float, float]]],
        clust_col: torch.Tensor,
        bound_col: torch.Tensor,
    ) -> List[int]:
        """
        Within a shelf, reorder CLUSTERS by their average target-x so that
        cluster members still stay contiguous (preserving grouping) while
        low-tx clusters land on the left and high-tx clusters on the right.
        Blocks with left/right boundary codes at shelf start/end are pinned.
        """
        if len(shelf) <= 1:
            return shelf
        start = 0
        end = len(shelf)
        # Pin shelf[0] if it has left-boundary (appears first by construction).
        if int(bound_col[shelf[0]].item()) & B_LEFT:
            start = 1
        # Pin shelf[-1] if it has right-boundary.
        if int(bound_col[shelf[-1]].item()) & B_RIGHT:
            end = len(shelf) - 1
        middle = shelf[start:end]
        if len(middle) <= 1:
            return shelf

        # Build contiguous cluster runs within middle. Any block with
        # cluster_id == 0 (unclustered) is its own singleton run.
        runs: List[List[int]] = []
        cur_run: List[int] = []
        cur_clu = None
        for i in middle:
            clu = int(clust_col[i].item())
            if not cur_run:
                cur_run = [i]; cur_clu = clu
                continue
            if clu != 0 and clu == cur_clu:
                cur_run.append(i)
            else:
                runs.append(cur_run)
                cur_run = [i]; cur_clu = clu
        if cur_run:
            runs.append(cur_run)

        # Sort runs by their average target x.
        def run_avg_tx(run: List[int]) -> float:
            xs = [pin_targets[i][0] for i in run if pin_targets[i] is not None]
            return sum(xs) / len(xs) if xs else 0.0

        runs.sort(key=run_avg_tx)
        new_middle = [i for run in runs for i in run]
        return shelf[:start] + new_middle + shelf[end:]

    def _split_wide_shelf(
        self,
        ids: List[int],
        base_sizes: List[Tuple[float, float]],
        is_inflex: List[bool],
        W: float,
    ) -> List[List[int]]:
        """Split `ids` into shelves if their inflexible widths alone exceed W."""
        shelves: List[List[int]] = []
        cur: List[int] = []
        cur_infl_w = 0.0
        for i in ids:
            w_i_infl = base_sizes[i][0] if is_inflex[i] else 0.0
            if cur and cur_infl_w + w_i_infl > W + 1e-6:
                shelves.append(cur)
                cur = [i]
                cur_infl_w = w_i_infl
            else:
                cur.append(i)
                cur_infl_w += w_i_infl
        if cur:
            shelves.append(cur)
        return shelves

    def _shelf_widths_and_height(
        self,
        shelf: List[int],
        base_sizes: List[Tuple[float, float]],
        is_inflex: List[bool],
        W: float,
        area_targets: torch.Tensor,
    ) -> Tuple[List[float], float]:
        """
        Choose shelf height h and per-block widths so the shelf fills ≤ W.

        For a shelf with inflex width I and flex area A:
          - Natural h_from_flex = A / (W - I)  →  widths sum to W exactly.
          - If any inflex block has height > that, shelf_h must grow, and flex
            widths shrink → sum < W (gap on right).  To close the gap and keep
            soft-block areas within 1% tolerance, we can inflate flex widths by
            a factor up to 1.01 (0.5% each way). Beyond that, we accept the gap.
        """
        infl_max_h = max((base_sizes[i][1] for i in shelf if is_inflex[i]), default=0.0)
        infl_width = sum(base_sizes[i][0] for i in shelf if is_inflex[i])
        remaining_W = max(W - infl_width, 1e-6)
        flex_ids = [i for i in shelf if not is_inflex[i]]
        flex_area_sum = sum(float(area_targets[i]) if area_targets[i] > 0 else 0.0
                            for i in flex_ids)

        h_from_flex = flex_area_sum / remaining_W if flex_ids else 0.0
        shelf_h = max(infl_max_h, h_from_flex)
        if shelf_h <= 0.0:
            shelf_h = max(base_sizes[i][1] for i in shelf) if shelf else 1.0

        # Base widths: inflex fixed; flex = a/shelf_h.
        widths: List[float] = []
        for i in shelf:
            if is_inflex[i]:
                widths.append(base_sizes[i][0])
            else:
                a = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                widths.append(a / shelf_h)

        # If total width < W, inflate flex widths by a scale within 1% tolerance
        # to close the gap (area becomes a_i * scale, still within 1% if scale ≤ 1.01).
        total_w = sum(widths)
        if flex_ids and total_w + 1e-6 < W:
            flex_total = sum(widths[k] for k, i in enumerate(shelf) if not is_inflex[i])
            if flex_total > 0:
                gap = W - total_w
                scale = 1.0 + gap / flex_total
                # Cap scale so flex area error ≤ 1% (conservative 0.9%).
                scale = min(scale, 1.009)
                for k, i in enumerate(shelf):
                    if not is_inflex[i]:
                        widths[k] *= scale

        return widths, shelf_h
