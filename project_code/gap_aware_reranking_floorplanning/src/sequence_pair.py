"""
Sequence-pair representation for floorplanning.

A sequence-pair (Γ+, Γ-) of n blocks encodes a compact floorplan topology:
  * If block i precedes j in BOTH Γ+ and Γ-: i is LEFT of j.
  * If i precedes j in Γ+ but AFTER in Γ-:   i is BELOW j.
  (The other two cases are symmetric.)

Packing: for each block, x_i = longest weighted path ending at i in the
horizontal constraint graph (edges from "left-of" relation, weight = w_src).
Similarly y_i from the vertical graph.

This implementation handles fixed-shape / preplaced / MIB blocks by using
their pinned (w, h); preplaced positions cannot be enforced exactly in pure
sequence-pair, but we pin them via a post-legalization step.
"""
from __future__ import annotations

import random
from typing import List, Tuple


class SequencePair:
    __slots__ = ("n", "gp", "gm", "widths", "heights")

    def __init__(self, n: int, gp: List[int], gm: List[int],
                 widths: List[float], heights: List[float]):
        self.n = n
        self.gp = list(gp)
        self.gm = list(gm)
        self.widths = list(widths)
        self.heights = list(heights)

    @classmethod
    def from_order(cls, order: List[int], widths: List[float],
                   heights: List[float]) -> "SequencePair":
        """Initial sequence-pair from a given order (both perms = order)."""
        return cls(len(order), order, order, widths, heights)

    def copy(self) -> "SequencePair":
        return SequencePair(self.n, self.gp, self.gm,
                            self.widths, self.heights)

    def pack(self) -> List[Tuple[float, float, float, float]]:
        """Compute (x, y, w, h) for each block from the sequence-pair."""
        import numpy as np
        n = self.n
        gp = np.asarray(self.gp, dtype=np.int64)
        gm = np.asarray(self.gm, dtype=np.int64)
        pos_p = np.empty(n, dtype=np.int64); pos_p[gp] = np.arange(n)
        pos_m = np.empty(n, dtype=np.int64); pos_m[gm] = np.arange(n)
        widths = np.asarray(self.widths, dtype=np.float64)
        heights = np.asarray(self.heights, dtype=np.float64)

        # X-coords: process blocks in Γ+ order. For each j, predecessors are
        # earlier-in-Γ+ blocks with pos_m < pos_m[j]. Vectorize the max over
        # those predecessors by keeping a running array of x[i]+widths[i]
        # keyed by pos_m[i].
        x = np.zeros(n, dtype=np.float64)
        # Running array sized by pos_m slot: best[k] = max of x[i]+widths[i]
        # among processed blocks with pos_m[i] == k. We take cumulative max
        # up to pos_m[j]-1 to get the predecessor maximum.
        best_x = np.zeros(n, dtype=np.float64)
        for idx_j in range(n):
            j = int(gp[idx_j])
            mj = int(pos_m[j])
            # max of best_x[0..mj-1]
            x[j] = best_x[:mj].max() if mj > 0 else 0.0
            best_x[mj] = x[j] + widths[j]

        # Y-coords: process blocks in Γ- order. Predecessors have pos_p > pos_p[j]
        # AND pos_m < pos_m[j]. Vectorize by keeping best_y[pos_p] = max y+h.
        y = np.zeros(n, dtype=np.float64)
        best_y = np.zeros(n, dtype=np.float64)
        for idx_j in range(n):
            j = int(gm[idx_j])
            pj = int(pos_p[j])
            # max of best_y[pj+1..n-1]
            y[j] = best_y[pj + 1:].max() if pj + 1 < n else 0.0
            best_y[pj] = y[j] + heights[j]

        return [(float(x[i]), float(y[i]), float(widths[i]), float(heights[i]))
                for i in range(n)]

    # -------------------------
    # SA moves (mutate in place)
    # -------------------------
    def swap_gp(self, a: int, b: int):
        """Swap two elements in Γ+."""
        self.gp[a], self.gp[b] = self.gp[b], self.gp[a]

    def swap_gm(self, a: int, b: int):
        """Swap two elements in Γ-."""
        self.gm[a], self.gm[b] = self.gm[b], self.gm[a]

    def swap_both(self, i: int, j: int):
        """Swap ids i and j in both permutations (maintains orientation
        between them but changes their relationship to others)."""
        pi_p = self.gp.index(i); pj_p = self.gp.index(j)
        self.gp[pi_p], self.gp[pj_p] = self.gp[pj_p], self.gp[pi_p]
        pi_m = self.gm.index(i); pj_m = self.gm.index(j)
        self.gm[pi_m], self.gm[pj_m] = self.gm[pj_m], self.gm[pi_m]

    def rotate(self, i: int):
        """Swap width/height of block i (only valid if block is soft)."""
        self.widths[i], self.heights[i] = self.heights[i], self.widths[i]


def sa_sequence_pair(
    n: int,
    widths: List[float],
    heights: List[float],
    cost_fn,
    max_iters: int = 2000,
    T0: float = 100.0,
    T_end: float = 0.1,
    initial_gp: List[int] = None,
    initial_gm: List[int] = None,
    rotatable: List[bool] = None,
    seed: int = 42,
) -> SequencePair:
    """
    Run simulated annealing over sequence-pairs.
    cost_fn(positions) → float. Lower is better.
    rotatable[i] == True iff block i can have its (w, h) swapped.
    Returns the best SequencePair encountered.
    """
    rng = random.Random(seed)
    if initial_gp is None:
        initial_gp = list(range(n))
    if initial_gm is None:
        initial_gm = list(range(n))
    sp = SequencePair(n, initial_gp, initial_gm, widths, heights)
    best_sp = sp.copy()
    cur_cost = cost_fn(sp.pack())
    best_cost = cur_cost
    T = T0
    alpha = (T_end / T0) ** (1.0 / max(max_iters, 1))

    for it in range(max_iters):
        # Propose a move.
        move_type = rng.randint(0, 3)
        if move_type == 0:
            a, b = rng.sample(range(n), 2)
            sp.swap_gp(a, b)
            undo = ("gp", a, b)
        elif move_type == 1:
            a, b = rng.sample(range(n), 2)
            sp.swap_gm(a, b)
            undo = ("gm", a, b)
        elif move_type == 2:
            a, b = rng.sample(range(n), 2)
            # swap_both needs block ids, not positions.
            sp.swap_both(sp.gp[a], sp.gp[b])
            undo = ("both", sp.gp[a], sp.gp[b])
        else:
            # Rotate a random rotatable block.
            if rotatable is None:
                rot_idxs = list(range(n))
            else:
                rot_idxs = [i for i in range(n) if rotatable[i]]
            if not rot_idxs:
                continue
            i = rng.choice(rot_idxs)
            sp.rotate(i)
            undo = ("rot", i, None)

        positions = sp.pack()
        new_cost = cost_fn(positions)
        delta = new_cost - cur_cost
        accept = delta < 0 or rng.random() < (2.71828 ** (-delta / max(T, 1e-6)))

        if accept:
            cur_cost = new_cost
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_sp = sp.copy()
        else:
            # Undo the move.
            kind = undo[0]
            if kind == "gp":
                sp.swap_gp(undo[1], undo[2])
            elif kind == "gm":
                sp.swap_gm(undo[1], undo[2])
            elif kind == "both":
                sp.swap_both(undo[1], undo[2])
            elif kind == "rot":
                sp.rotate(undo[1])
        T *= alpha
    return best_sp
