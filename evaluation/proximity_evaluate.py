from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from evaluation.metrics.obs_proximity import min_clearance_to_obstacles_ls
from evaluation.base_evaluator import BaseEvaluator

class ProximityEvaluator(BaseEvaluator):
    def __init__(self, output_path: str, model: str, dataset: str = "scand"):
        super().__init__(output_path, model)
        
        self._eval_name = f"proximity_{dataset}"
        self._open_output_files()

    def analyze_bag(self, finetuned: bool = True):
        print(f"[INFO] Analyzing obstacle proximity for {self.bag_name}")
        min_clearances = []

        for frame in tqdm(self.frames, desc=f"Eval {self._eval_name} {self.bag_name}", unit="frame"):
            if frame.goal_idx == -1:
                continue

            goal_frame = self.frames[frame.goal_idx]
            if finetuned:
                path_xy = frame.path_ft
            else:
                path_xy = frame.path_bl
            

            angle_max = frame.angle_min + (len(frame.laserscan) - 1) * frame.angle_increment
            min_clearance = min_clearance_to_obstacles_ls(
                path_xy=path_xy,
                laserscan=frame.laserscan,
                angle_increment=frame.angle_increment,
                angle_min=frame.angle_min,
                angle_max=angle_max,
                range_min=frame.range_min,
                range_max=frame.range_max,
            )

            if min_clearance != np.inf:
                min_clearances.append(min_clearance)
        return min_clearances
