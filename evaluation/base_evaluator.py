import os
import json

import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

class BaseEvaluator():
    def __init__(self, output_path: str, model: str):
        self.output_path = output_path
        self._eval_name = "base_eval"
        self.log_file_path = None
        self.log_file_path_ft = None
        self.bag_name = None
        self.frames = None
        self.model = model
        self.all_evals_from_data = False

    def _open_output_files(self):
        output_dir = os.path.join(self.output_path, self._eval_name)
        os.makedirs(output_dir, exist_ok=True)

        self.log_file_path = os.path.join(output_dir, f"{self.model}_{self._eval_name}_results.log")
        self.log_file_path_ft = os.path.join(output_dir, f"{self.model}_{self._eval_name}_results_finetuned.log")

    def analyze_bag(self):
        # Placeholder for analysis logic
        return 

    def log_metrics(self, evaluations, finetuned: bool = True):

        if finetuned:
            log_file_path = self.log_file_path_ft
        else:
            log_file_path = self.log_file_path

        mean, std = self.calculate_statistics(evaluations)
        if mean is None or std is None:
            return

        bag_stem = Path(self.bag_name).stem
        evaluations = [float(val) for val in evaluations] if evaluations else []

        with open(log_file_path, "a") as f:
            json_record = {
                "bag": bag_stem,
                "frame_count": len(evaluations),
                "data": evaluations,
                "mean": mean,
                "std": std,
            }
            f.write(json.dumps(json_record) + "\n")  # one object per line

            mode = "finetuned" if finetuned else "pretrained"
            print(f"[INFO] Evaluation results for {self.bag_name} {self._eval_name} {mode}: {mean:.4f} ± {std:.4f}\n")

    def calculate_statistics(self, evaluations: list):
        if not evaluations:
            print("[WARNING] No evaluations to calculate statistics.")
            return None, None

        mean = float(np.mean(evaluations))
        std = float(np.std(evaluations))
        return mean, std

    def run(self, bag_name: str, frames: list):
        self.bag_name = bag_name
        self.frames = frames
        eval_finetuned = self.analyze_bag(finetuned=True)
        eval_pretrained = self.analyze_bag(finetuned=False)
        self.log_metrics(eval_pretrained, finetuned=False)
        self.log_metrics(eval_finetuned, finetuned=True)

        print(f"\n[DONE] Annotations written to {self.output_path}")
