#!/usr/bin/env python3
"""
Quick evaluation harness for FloorSet solver experiments.

Runs `solver.py::MyOptimizer` on a representative subset of the validation set
(or full 100 cases with --full) and prints key metrics on stdout.

Key outputs (grep these):
  SHORT_SCORE: <float>   exponentially-weighted cost on short subset (primary metric)
  FULL_SCORE:  <float>   same, on full 100-case validation (only with --full)
  FEASIBLE:    <n/total> feasible count
  MEAN_RT:     <seconds>
  MEAN_HPWL_GAP: <float>
  MEAN_AREA_GAP: <float>
  MEAN_V_REL:  <float>

DO NOT MODIFY THIS FILE DURING EXPERIMENTATION.
"""
import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent
# FloorSet location: env var, then various fallbacks.
import os as _os
_env = _os.environ.get("FLOORSET_ROOT")
_candidates = ([Path(_env)] if _env else []) + [
    REPO_ROOT.parent / "FloorSet",
    REPO_ROOT / "FloorSet",
    Path.cwd() / "FloorSet",
]
FLOORSET_ROOT = next((p for p in _candidates if p.exists()), _candidates[0])
CONTEST_DIR = FLOORSET_ROOT / "iccad2026contest"
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(FLOORSET_ROOT))
sys.path.insert(0, str(CONTEST_DIR))
sys.path.insert(0, str(SRC_DIR))

import importlib.util
import math
import warnings
warnings.filterwarnings("ignore")

import torch

from iccad2026_evaluate import (
    FloorplanOptimizer,
    evaluate_solution,
    compute_total_score,
    compute_cost,
    M_PENALTY,
)
from lite_dataset_test import FloorplanDatasetLiteTest

# Short subset: 15 representative cases spread across block-count range.
# Picked to bias toward larger instances (which dominate the exponential weighting).
SHORT_IDS = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 99]


def load_optimizer(solver_path: Path) -> FloorplanOptimizer:
    spec = importlib.util.spec_from_file_location("user_solver", solver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Find a subclass of FloorplanOptimizer that isn't the base.
    for name in dir(mod):
        obj = getattr(mod, name)
        if (isinstance(obj, type)
                and issubclass(obj, FloorplanOptimizer)
                and obj.__name__ != "FloorplanOptimizer"):
            return obj(verbose=False)
    if hasattr(mod, "MyOptimizer"):
        return mod.MyOptimizer(verbose=False)
    raise RuntimeError("No MyOptimizer(FloorplanOptimizer) class in solver.py")


def extract_baseline(inputs, labels, block_count):
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    polygons, metrics = labels

    positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append((float(x_min), float(y_min),
                              float(x_max - x_min), float(y_max - y_min)))
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))

    # Prefer stored metrics (matches generate_baselines logic in evaluator).
    area = None
    hpwl_b2b_base = None
    hpwl_p2b_base = None
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0:
            hpwl_b2b_base = float(metrics[-2])
        if metrics[-1] >= 0:
            hpwl_p2b_base = float(metrics[-1])

    if area is None:
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        area = (x_max - x_min) * (y_max - y_min)
    if hpwl_b2b_base is None or hpwl_p2b_base is None:
        # Recompute from ground-truth polygons if metrics missing.
        from iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b
        hpwl_b2b_base = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b_base = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)

    return {
        "hpwl_baseline": hpwl_b2b_base + hpwl_p2b_base,
        "area_baseline": area,
    }, positions


def build_target_positions(target_pos, constraints, block_count):
    opt_target_pos = torch.full((block_count, 4), -1.0)
    if target_pos is None or constraints is None:
        return opt_target_pos
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_preplaced = nc > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = target_pos[i]
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th
    return opt_target_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default=str(SRC_DIR / "solver.py"))
    parser.add_argument("--full", action="store_true",
                        help="Run on full 100-case validation set")
    parser.add_argument("--ids", type=str, default="",
                        help="Comma-separated case IDs (overrides short/full)")
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="Hard per-case timeout (unused, advisory)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    solver_path = Path(args.solver).resolve()
    print(f"Loading solver: {solver_path}", flush=True)
    optimizer = load_optimizer(solver_path)

    print("Loading validation dataset...", flush=True)
    dataset = FloorplanDatasetLiteTest(str(FLOORSET_ROOT) + "/")

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip() != ""]
    elif args.full:
        ids = list(range(len(dataset)))
    else:
        ids = SHORT_IDS

    print(f"Running on {len(ids)} cases: {ids}", flush=True)

    results = []
    runtimes = []
    t_total = time.time()
    for idx in ids:
        sample = dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())

        baseline, target_pos = extract_baseline(inputs, labels, block_count)
        opt_target_pos = build_target_positions(target_pos, constraints, block_count)

        try:
            t0 = time.time()
            positions = optimizer.solve(
                block_count, area_target, b2b_conn, p2b_conn,
                pins_pos, constraints, opt_target_pos
            )
            rt = time.time() - t0
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(dict(
                test_id=idx, block_count=block_count, cost=M_PENALTY,
                is_feasible=False, hpwl_gap=0.0, area_gap=0.0,
                v_rel=1.0, runtime=0.0, error=str(e),
            ))
            continue

        m = evaluate_solution(
            {"positions": positions, "runtime": rt},
            baseline, constraints, b2b_conn, p2b_conn, pins_pos,
            area_target, target_pos, median_runtime=1.0,
        )
        runtimes.append(rt)
        results.append(dict(
            test_id=idx, block_count=block_count, cost=m.cost,
            is_feasible=m.is_feasible, hpwl_gap=m.hpwl_gap,
            area_gap=m.area_gap, v_rel=m.violations_relative,
            runtime=rt, overlaps=m.overlap_violations,
            area_viol=m.area_violations, dim_viol=m.dimension_violations,
        ))

        if args.verbose:
            flag = "OK" if m.is_feasible else f"INF(ov={m.overlap_violations},av={m.area_violations},dim={m.dimension_violations})"
            print(f"  case {idx:3d} n={block_count:3d} {flag:20s} "
                  f"hpwl_gap={m.hpwl_gap:+.3f} area_gap={m.area_gap:+.3f} "
                  f"v_rel={m.violations_relative:.3f} rt={rt:6.2f}s cost={m.cost:.3f}",
                  flush=True)

    # Recompute costs using median runtime (as contest does).
    if runtimes:
        median_rt = sorted(runtimes)[len(runtimes) // 2]
        for r in results:
            if r.get("error") is None and r["is_feasible"]:
                rf = r["runtime"] / max(median_rt, 0.01)
                r["cost"] = compute_cost(r["hpwl_gap"], r["area_gap"],
                                         r["v_rel"], rf, True)

    costs = [r["cost"] for r in results]
    blocks = [r["block_count"] for r in results]
    total = compute_total_score(costs, blocks)
    feasible = sum(1 for r in results if r["is_feasible"])
    mean_rt = sum(r["runtime"] for r in results) / max(len(results), 1)
    mean_hpwl = sum(r["hpwl_gap"] for r in results if r["is_feasible"]) / max(feasible, 1)
    mean_area = sum(r["area_gap"] for r in results if r["is_feasible"]) / max(feasible, 1)
    mean_vrel = sum(r["v_rel"] for r in results if r["is_feasible"]) / max(feasible, 1)

    score_tag = "FULL_SCORE" if args.full else "SHORT_SCORE"
    total_t = time.time() - t_total
    print("=" * 60, flush=True)
    print(f"{score_tag}:    {total:.5f}", flush=True)
    print(f"FEASIBLE:      {feasible}/{len(results)}", flush=True)
    print(f"MEAN_RT:       {mean_rt:.3f}", flush=True)
    print(f"MEAN_HPWL_GAP: {mean_hpwl:+.4f}", flush=True)
    print(f"MEAN_AREA_GAP: {mean_area:+.4f}", flush=True)
    print(f"MEAN_V_REL:    {mean_vrel:.4f}", flush=True)
    print(f"EVAL_WALL:     {total_t:.2f}s", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
