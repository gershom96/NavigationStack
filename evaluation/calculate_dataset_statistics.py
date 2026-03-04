import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


def _strip_data_array(line: str) -> str:
    """
    Replace the potentially huge data array with an empty list so JSON parsing stays fast.
    """
    data_idx = line.find('"data"')
    mean_idx = line.rfind('"mean"')
    if data_idx == -1 or mean_idx == -1 or data_idx > mean_idx:
        return line

    start = line.find("[", data_idx, mean_idx)
    end = line.rfind("]", data_idx, mean_idx)
    if start == -1 or end == -1 or start >= end:
        return line

    return line[: start + 1] + "]" + line[end + 1 :]


def _parse_record(line: str) -> Optional[Tuple[int, float, float]]:
    cleaned = _strip_data_array(line.strip())
    if not cleaned:
        return None

    try:
        record = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    frame_count = record.get("frame_count")
    mean = record.get("mean")
    std = record.get("std")

    if frame_count in (None, 0) or mean is None or std is None:
        data = record.get("data")
        if data:
            frame_count = len(data)
        else:
            return None


    return int(frame_count), float(mean), float(std)

def _summarize_log(log_path: str, is_near_miss: bool) -> Optional[Dict[str, float]]:
    totals: Dict[str, float] = {
        "bags": 0,
        "frames": 0,
    }

    if is_near_miss:
        totals["total"] = 0.0
    else:
        totals["weighted_mean_sum"] = 0.0
        totals["weighted_sq_sum"] = 0.0

    with open(log_path, "r") as f:
        for raw_line in f:
            parsed = _parse_record(raw_line)
            if not parsed:
                continue

            count, mean, std = parsed
            totals["bags"] += 1
            totals["frames"] += count
            if is_near_miss:
                totals["total"] += mean * count
            else:
                totals["weighted_mean_sum"] += mean * count
                totals["weighted_sq_sum"] += (std * std + mean * mean) * count

    if totals["frames"] == 0:
        return None

    if is_near_miss:
        return {
            "bags": totals["bags"],
            "frames": totals["frames"],
            "total": totals["total"],
        }

    dataset_mean = totals["weighted_mean_sum"] / totals["frames"]
    dataset_var = totals["weighted_sq_sum"] / totals["frames"] - dataset_mean * dataset_mean
    dataset_std = math.sqrt(dataset_var) if dataset_var > 0.0 else 0.0

    return {
        "bags": totals["bags"],
        "frames": totals["frames"],
        "mean": dataset_mean,
        "std": dataset_std,
    }


def _describe_log(log_path: str) -> Tuple[str, str, str]:
    """
    Returns (eval_name, model, mode)
    """
    path = Path(log_path)
    eval_name = path.parent.name
    fname = path.name

    if fname.endswith("_results_finetuned.log"):
        suffix = "_results_finetuned.log"
        mode = "finetuned"
    else:
        suffix = "_results.log"
        mode = "pretrained"

    stem = fname[: -len(suffix)]
    if "_" in stem:
        model, _ = stem.split("_", 1)
    else:
        model = stem

    return eval_name, model, mode


def _collect_log_files(eval_root: str, model_filter: Optional[str]) -> list[str]:
    pattern = os.path.join(eval_root, "*", "*_results*.log")
    files = [
        path
        for path in glob.glob(pattern)
        if os.path.isfile(path)
        and (model_filter is None or Path(path).name.startswith(f"{model_filter}_"))
    ]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Compute dataset-level statistics for evaluation logs.")
    parser.add_argument(
        "--eval-root",
        default="outputs/evals",
        help="Root directory containing evaluation result logs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model prefix to filter logs (e.g., omnivla, vint, gnm).",
    )
    args = parser.parse_args()

    log_files = _collect_log_files(args.eval_root, args.model)
    if not log_files:
        print(f"[WARN] No log files found under {args.eval_root}")
        return

    for log_path in log_files:
        eval_name, model, mode = _describe_log(log_path)
        is_near_miss = eval_name.startswith("near_miss")
        summary = _summarize_log(log_path, is_near_miss)

        if not summary:
            print(f"[WARN] Skipping {log_path}: no valid records.")
            continue

        if is_near_miss:
            print(
                f"{eval_name} - {model} ({mode}): "
                f"bags={summary['bags']} total_near_misses={int(summary['total'])}"
            )
        else:
            print(
                f"{eval_name} - {model} ({mode}): "
                f"bags={summary['bags']} frames={summary['frames']} "
                f"mean={summary['mean']:.4f} std={summary['std']:.4f}"
            )


if __name__ == "__main__":
    main()
