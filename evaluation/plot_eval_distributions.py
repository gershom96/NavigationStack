#!/usr/bin/env python3
"""
Create histogram distributions for each bag inside eval log files and combined distributions.

Input layout (existing):
    outputs/evals/<eval_name>/*.log       # JSONL where each line has keys: bag, data, ...

Outputs:
    outputs/visualization/distribution/<eval_name>/<model_slug>/<bag_slug>.png
    outputs/visualization/distribution/<eval_name>/<model_slug>/collective.png   (per log file)
    outputs/visualization/distribution/<eval_name>/collective.png               (across all logs in eval)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

# Use non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_EVAL_DIR = Path("outputs/evals")
OUTPUT_BASE = Path("outputs/visualization/distribution")


def slugify(text: str, default: str = "bag") -> str:
    """Turn arbitrary text into a filename-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return slug[:200] if slug else default


def plot_histogram(
    data: Iterable[float],
    title: str,
    out_path: Path,
    xlim: Tuple[float, float] | None = None,
    bins: np.ndarray | int | None = None,
) -> None:
    """Plot a histogram with mean/median markers and save it."""
    values = np.fromiter((float(x) for x in data), dtype=float)
    if values.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="#3b82f6", edgecolor="white", alpha=0.85)
    plt.axvline(values.mean(), color="#ef4444", linestyle="--", linewidth=1.5, label=f"mean={values.mean():.3f}")
    plt.axvline(np.median(values), color="#10b981", linestyle=":", linewidth=1.5, label=f"median={np.median(values):.3f}")
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def finetune_status(model_slug: str) -> str:
    """Return 'finetuned' if the slug contains that substring, else 'baseline'."""
    return "finetuned" if "finetuned" in model_slug.lower() else "baseline"


def load_log(log_path: Path) -> Tuple[List[float], List[Tuple[str, List[float]]]]:
    """Return (all_values, [(bag_name, bag_values), ...]) for a log file."""
    all_values: List[float] = []
    bags: List[Tuple[str, List[float]]] = []

    with log_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            bag_name = record.get("bag", "unknown_bag")
            data = record.get("data") or []
            if not data:
                continue

            bags.append((bag_name, data))
            all_values.extend(data)

    return all_values, bags


def process_eval(eval_dir: Path) -> None:
    eval_out_dir = OUTPUT_BASE / eval_dir.name
    eval_collective: List[float] = []
    logs_data: List[Tuple[str, List[float], List[Tuple[str, List[float]]]]] = []

    for log_file in sorted(eval_dir.glob("*.log")):
        model_slug = log_file.stem
        collective, bags = load_log(log_file)
        logs_data.append((model_slug, collective, bags))
        eval_collective.extend(collective)

    # Determine shared x-limits & bins per metric (eval), clipping outliers so shapes stay visible.
    xlim = None
    bins: np.ndarray | int | None = None
    if eval_collective:
        values = np.array(eval_collective, dtype=float)
        # Clip to percentile window to avoid huge empty stretches from rare outliers.
        low, high = np.percentile(values, [1, 95])  # tighter window than before
        if high <= low:  # degenerate
            low, high = float(values.min()), float(values.max())
        # Add a small padding so bars at the edges are visible.
        span = max(high - low, 1e-6)
        pad = span * 0.0
        xlim = (float(low - pad), float(high + pad))

        # Use Freedman–Diaconis rule on the combined values within the clipped range.
        bins = np.histogram_bin_edges(values, bins="fd", range=(low, high))
        # Guard against too few edges
        if bins.size < 2:
            bins = 30

    for model_slug, collective, bags in logs_data:
        log_out_dir = eval_out_dir / model_slug

        # Per-bag plots (shared xlim/bins per metric for comparability across models)
        for bag_name, data in bags:
            bag_slug = slugify(bag_name)
            title = f"{eval_dir.name} | {model_slug} | {bag_name}"
            plot_histogram(data, title, log_out_dir / f"{bag_slug}.png", xlim=xlim, bins=bins)

        # Per-log collective
        if collective:
            title = f"{eval_dir.name} | {model_slug} | collective"
            status = finetune_status(model_slug)
            fname = f"collective_{eval_dir.name}_{model_slug}_{status}.png"
            plot_histogram(collective, title, log_out_dir / fname, xlim=xlim, bins=bins)

    # Per-eval collective across logs
    if eval_collective:
        title = f"{eval_dir.name} | collective across logs"
        plot_histogram(eval_collective, title, eval_out_dir / f"collective_{eval_dir.name}.png", xlim=xlim, bins=bins)


def main() -> None:
    if not BASE_EVAL_DIR.exists():
        raise SystemExit(f"Eval directory not found: {BASE_EVAL_DIR}")

    for eval_dir in sorted(p for p in BASE_EVAL_DIR.iterdir() if p.is_dir()):
        process_eval(eval_dir)

    print(f"Distributions saved under {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
