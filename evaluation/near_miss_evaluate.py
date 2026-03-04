import json
import os
from pathlib import Path

from evaluation.base_evaluator import BaseEvaluator

class NearMissEvaluator(BaseEvaluator):
    """
    Evaluates how many near misses occur within each bag file
    Near miss : when the predicted path comes within a certain threshold distance of an obstacle
    """

    def __init__(self, output_path: str, model: str, dataset: str = "scand"):
        super().__init__(output_path, model)

        self._eval_name = f"near_miss_{dataset}"

        self.all_evals_from_data = True
        self.dependent_eval = "proximity_scand" if dataset == "scand" else "proximity"

        dependent_output_dir = os.path.join(self.output_path, self.dependent_eval)
        self.dependent_log_file_path_ft = os.path.join(dependent_output_dir, f"{self.model}_{self.dependent_eval}_results_finetuned.log")
        self.dependent_log_file_path = os.path.join(dependent_output_dir, f"{self.model}_{self.dependent_eval}_results.log")
        self._open_output_files()

        self.width_dict = {
            "jackal": 0.43,
            "spot": 0.5,
        }
    
    def analyze_bag(self, finetuned: bool = True):
        print(f"[INFO] Analyzing near misses for {self.bag_name}")
        near_miss_counts = []

        if finetuned:
            log_file_path = self.dependent_log_file_path_ft
        else:
            log_file_path = self.dependent_log_file_path

        if not os.path.exists(log_file_path):
            return near_miss_counts

        if "Spot" in self.bag_name:
            threshold = self.width_dict["spot"]
        elif "Jackal" in self.bag_name:
            threshold = self.width_dict["jackal"]
        else:
            threshold = 0.5

        with open(log_file_path, "rb") as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if json_record.get("bag") != self.bag_name:
                    continue

                data = json_record.get("data", [])
                near_miss_counts.append(sum(1 for val in data if val < threshold))
                break

        return near_miss_counts
