#!/usr/bin/env python3
"""
Inference-only Code Ocean entry point.

This script:
1. verifies that the released checkpoints are readable
2. runs short evaluation-only sanity checks on Lite and Prime
3. copies released paper tables
4. regenerates lightweight figures from the released CSV traces

It intentionally does not retrain models.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

THIS_DIR = Path(__file__).resolve().parent


def _resolve_code_root() -> Path:
    env_root = os.environ.get("CODE_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        THIS_DIR.parent,  # local repo layout: submission_final/code/main.py
        THIS_DIR,         # Code Ocean layout: /code/main.py
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


def _resolve_checkpoint_dir() -> Path:
    env_ckpt = os.environ.get("CHECKPOINT_DIR")
    if env_ckpt:
        return Path(env_ckpt)
    codeocean_data = Path("/data/checkpoints")
    if os.name != "nt" and codeocean_data.exists():
        return codeocean_data
    return ROOT / "data" / "checkpoints"


CHECKPOINT_DIR = _resolve_checkpoint_dir()


def _resolve_results_root() -> Path:
    for env_name in ("RESULTS_DIR", "CODEOCEAN_RESULTS_DIR"):
        raw = os.environ.get(env_name)
        if raw:
            return Path(raw)
    codeocean_results = Path("/results")
    if os.name != "nt" and codeocean_results.exists():
        return codeocean_results
    return ROOT / "results"


OUTPUT_ROOT = _resolve_results_root() / "codeocean_run"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"

VERIFICATION_IDS = "0,20,40,70,99"
PAPER_HEADLINE_METRICS = {
    "FloorSet-Lite": {
        "score_name": "S_100",
        "score": 1.40228,
        "feasible": "100/100",
    },
    "FloorSet-Prime": {
        "score_name": "S_100",
        "score": 1.54862,
        "feasible": "91/100",
    },
}


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def verify_checkpoint(path: Path) -> dict:
    prefix = path.read_bytes()[:128]
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} looks like a Git LFS pointer file, not the real checkpoint binary. "
            "Upload the actual .pt file as a Code Ocean data asset under /data/checkpoints/"
        )
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "model_state" not in obj:
        raise RuntimeError(f"Unexpected checkpoint format: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "top_level_keys": sorted(obj.keys()),
    }


def parse_eval_metrics(stdout: str) -> dict:
    metrics = {}
    patterns = {
        "score_name": r"^(SHORT_SCORE|FULL_SCORE):\s+([0-9.]+)$",
        "feasible": r"^FEASIBLE:\s+(.+)$",
        "mean_rt": r"^MEAN_RT:\s+([0-9.]+)$",
        "mean_hpwl_gap": r"^MEAN_HPWL_GAP:\s+([+\-0-9.]+)$",
        "mean_area_gap": r"^MEAN_AREA_GAP:\s+([+\-0-9.]+)$",
        "mean_v_rel": r"^MEAN_V_REL:\s+([0-9.]+)$",
        "eval_wall": r"^EVAL_WALL:\s+([0-9.]+)s$",
    }
    for line in stdout.splitlines():
        for key, pattern in patterns.items():
            m = re.match(pattern, line.strip())
            if not m:
                continue
            if key == "score_name":
                metrics["score_name"] = m.group(1)
                metrics["score"] = float(m.group(2))
            elif key in {"mean_rt", "mean_v_rel", "eval_wall"}:
                metrics[key] = float(m.group(1))
            elif key in {"mean_hpwl_gap", "mean_area_gap"}:
                metrics[key] = float(m.group(1))
            else:
                metrics[key] = m.group(1)
    return metrics


def run_eval(script_name: str, benchmark: str, skip_inference: bool = False) -> dict:
    if skip_inference:
        return {
            "benchmark": benchmark,
            "status": "skipped",
            "reason": "skip-inference flag enabled",
        }
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / script_name),
        "--ids",
        VERIFICATION_IDS,
    ]
    env = os.environ.copy()
    env.setdefault("CHECKPOINT_DIR", str(CHECKPOINT_DIR))
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path = OUTPUT_ROOT / f"{benchmark.lower().replace('-', '_')}_eval.log"
    log_path.write_text(proc.stdout + "\n\nSTDERR\n------\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{benchmark} evaluation failed with code {proc.returncode}. "
            f"See {log_path}"
        )
    metrics = parse_eval_metrics(proc.stdout)
    metrics["benchmark"] = benchmark
    metrics["verification_ids"] = VERIFICATION_IDS
    return metrics


def copy_paper_tables() -> list[Path]:
    copied = []
    src_dir = PROJECT_ROOT / "results"
    for csv_path in sorted(src_dir.glob("*.csv")):
        dest = TABLES_DIR / csv_path.name
        shutil.copy2(csv_path, dest)
        copied.append(dest)
    headline_path = TABLES_DIR / "paper_headline_metrics.csv"
    with headline_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "score_name", "score", "feasible"])
        for benchmark, row in PAPER_HEADLINE_METRICS.items():
            writer.writerow([benchmark, row["score_name"], row["score"], row["feasible"]])
    copied.append(headline_path)
    return copied


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_runtime_evolution() -> Path:
    rows = read_csv_rows(PROJECT_ROOT / "results" / "runtime_evolution.csv")
    labels = [row["configuration"] for row in rows]
    values = [float(row["per_case_s"]) for row in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(len(values)), values, marker="o", linewidth=2)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Per-case runtime (s)")
    plt.title("Runtime evolution across kept solver configurations")
    plt.tight_layout()
    out = FIGURES_DIR / "runtime_evolution.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_hpwl_vrel_scatter() -> Path:
    rows = read_csv_rows(PROJECT_ROOT / "results" / "hpwl_vrel_scatter_data.csv")
    grouped = {}
    for row in rows:
        grouped.setdefault(row["candidate_family"], []).append(row)

    plt.figure(figsize=(6, 4.5))
    for family, points in grouped.items():
        xs = [float(p["v_rel"]) for p in points]
        ys = [float(p["hpwl_gap_relative_to_min"]) for p in points]
        plt.scatter(xs, ys, s=40, label=family)
    plt.xlabel("Relative violation count (V_rel)")
    plt.ylabel("HPWL gap relative to best raw candidate")
    plt.title("Candidate regimes in released rerank traces")
    plt.legend(frameon=False)
    plt.tight_layout()
    out = FIGURES_DIR / "hpwl_vrel_scatter.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_short_subset_costs() -> Path:
    rows = read_csv_rows(PROJECT_ROOT / "results" / "per_case_metrics.csv")
    case_ids = [row["case_id"] for row in rows]
    costs = [float(row["cost_contest"]) for row in rows]

    plt.figure(figsize=(8, 4.5))
    plt.bar(case_ids, costs, color="#4C78A8")
    plt.xlabel("Case ID")
    plt.ylabel("Contest cost")
    plt.title("Released short-subset per-case contest costs")
    plt.tight_layout()
    out = FIGURES_DIR / "short_subset_costs.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def write_summary(checkpoints: dict, evals: list[dict], tables: list[Path], figures: list[Path]) -> None:
    summary = {
        "checkpoints": checkpoints,
        "verification_runs": evals,
        "paper_headline_metrics": PAPER_HEADLINE_METRICS,
        "tables": [str(p) for p in tables],
        "figures": [str(p) for p in figures],
    }
    json_path = OUTPUT_ROOT / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Code Ocean Run Summary",
        "",
        "## Checkpoints",
    ]
    for name, meta in checkpoints.items():
        lines.append(f"- `{name}` loaded successfully from `{meta['path']}`")
        lines.append(f"  size: {meta['size_bytes']} bytes")
        lines.append(f"  keys: {', '.join(meta['top_level_keys'])}")
    lines.extend([
        "",
        "## Verification inference runs",
    ])
    for item in evals:
        if item.get("status") == "skipped":
            lines.append(f"- {item['benchmark']}: skipped ({item['reason']})")
            continue
        lines.append(
            f"- {item['benchmark']}: {item['score_name']}={item['score']:.5f}, "
            f"feasible={item['feasible']}, mean_rt={item['mean_rt']:.3f}s, "
            f"ids={item['verification_ids']}"
        )
    lines.extend([
        "",
        "## Released full-set paper metrics",
    ])
    for benchmark, row in PAPER_HEADLINE_METRICS.items():
        lines.append(
            f"- {benchmark}: {row['score_name']}={row['score']:.5f}, feasible={row['feasible']}"
        )
    lines.extend([
        "",
        "## Notes",
        "- The default capsule run is inference-only and does not retrain models.",
        "- Full 100-case benchmark numbers are reproduced from the released paper traces,",
        "  while the default run performs a shorter checkpoint-backed sanity check.",
    ])
    (OUTPUT_ROOT / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    skip_inference = "--skip-inference" in sys.argv
    ensure_dirs()

    required = {
        "dit_base_ckpt.pt": CHECKPOINT_DIR / "dit_base_ckpt.pt",
        "nn_hint_ckpt.pt": CHECKPOINT_DIR / "nn_hint_ckpt.pt",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required checkpoint files in data/checkpoints/: "
            + ", ".join(missing)
        )

    checkpoint_meta = {name: verify_checkpoint(path) for name, path in required.items()}
    lite_metrics = run_eval("quick_eval.py", "FloorSet-Lite", skip_inference=skip_inference)
    prime_metrics = run_eval("quick_eval_prime.py", "FloorSet-Prime", skip_inference=skip_inference)
    tables = copy_paper_tables()
    figures = [
        plot_runtime_evolution(),
        plot_hpwl_vrel_scatter(),
        plot_short_subset_costs(),
    ]
    write_summary(checkpoint_meta, [lite_metrics, prime_metrics], tables, figures)
    print(f"Wrote Code Ocean outputs to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
