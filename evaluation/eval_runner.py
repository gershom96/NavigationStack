"""
Shared evaluation runner that processes each bag once, runs inference once per
checkpoint (finetuned / baseline), and caches the resulting trajectories on a
frame list. Metric-specific evaluators can then consume the cached paths
instead of re-running the model.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pathlib import Path

import argparse
import yaml
import json
import os
import sys
import importlib
import glob
from collections import defaultdict
from typing import Optional, Dict, List
from tqdm import tqdm
from dataclasses import dataclass

import rosbag
from cv_bridge import CvBridge

from datasets.preprocess_scand_a_chop import _resample_path

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_sources.visualnav_transformer.deployment.src.utils import load_model as deployment_load_model
from policy_sources.omnivla.inference.run_omnivla_modified import Inference
from policy_sources.visualnav_transformer.deployment.src.utils import transform_images
from policy_sources.visualnav_transformer.train.vint_train.training.train_utils import get_action
from PIL import Image as PILImage

from evaluation.proximity_evaluate import ProximityEvaluator
from evaluation.goal_distance_evaluate import GoalDistanceEvaluator
from evaluation.alignment_evaluate import AlignmentEvaluator
from evaluation.near_miss_evaluate import NearMissEvaluator

class InferenceConfigOriginal:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000
    vla_path: str = "./weights/omnivla-finetuned-cast"   
    resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

class InferenceConfigFinetuned:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000    
    vla_path: str = "./weights/omnivla-finetuned-chop"   
    resume_step: Optional[int] = 222500
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    
@dataclass
class FrameItem:
    frame_idx: int
    image: np.ndarray
    image_path : str
    timestamp: int
    path_ft: Optional[np.ndarray] = None
    path_bl: Optional[np.ndarray] = None
    path_gt: Optional[np.ndarray] = None
    laserscan: Optional[np.ndarray] = None
    angle_min: Optional[float] = None
    angle_increment: Optional[float] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    pos: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    cum_distance: float = 0.0
    goal_idx: int = -1

class EvalRunner:
    def __init__(
        self,
        bag_dir: str,
        output_path: str,
        inference_out: str,
        test_train_split_path: str,
        pref_annotations_path: str,
        model: str,
        fov_angle: float = 90.0,
        num_points: int = 8,
        sample_goals: bool = True,
        max_distance: float = 20.0,
        min_distance: float = 2.0,
        evaluators: List = [],
    ):
        self.bag_dir = bag_dir
        self.bag_name = None
        self.pref_annotations_path = pref_annotations_path
        self.pref_annotations = None
        self.test_train_split_path = test_train_split_path
        self.model_name = model
        self.num_points = num_points
        self.sample_goals = sample_goals
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.inference_out = Path(inference_out)

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.fov_angle = fov_angle
        self.bridge = CvBridge()

        self.evaluators = evaluators
        self.vla_config = InferenceConfigOriginal()
        self.vla_config_finetuned = InferenceConfigFinetuned()
        self.frames : list[FrameItem] = []

        parser = argparse.ArgumentParser(description="Config to model")
        parser.add_argument(
            "--config",
            "-c",
            default=f"configs/chop_{self.model_name}_vnt.yaml",
            type=str,
            help="Path to the config file in config folder",
        )
        args = parser.parse_args()

        if self.model_name in {"vint", "gnm", "nomad"}:
            with open("configs/chop_default_vnt.yaml", "r") as f:
                default_config = yaml.safe_load(f)

            config = default_config

            with open(args.config, "r") as f:
                user_config = yaml.safe_load(f)

            config.update(user_config)
        else:
            with open("configs/chop_omnivla.yaml", "r") as f:
                config = yaml.safe_load(f)

        self.config = config
        self.context_frames = self.config.get("context_size", 0)
        self.image_root = Path("/media/beast-gamma/Media/Datasets/SCAND/images/")

        self.all_evals_from_data = True
        for eval in self.evaluators:
            if not eval.all_evals_from_data:
                self.all_evals_from_data = False
                break

    def _get_timestamps_from_expert_annotations(self):
        self.pref_file = os.path.join(self.pref_annotations_path, f"{Path(self.bag_name).stem}.json")
        with open(self.pref_file, "r") as f:
            self.pref_annotations = json.load(f)
        timestamps = []
        for key in self.pref_annotations.get("annotations_by_stamp", {}).keys():
            timestamps.append(int(key))
        return timestamps

    def _save_paths_json(self, bag_name: str):
        stem = Path(bag_name).stem
        output_file = self.inference_out / self.model_name / f"{stem}_paths.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bag": stem,
            "frames": {}
        }
        for frame in self.frames:
            frame_dict = {
                "frame_idx": frame.frame_idx,
                "img_path": frame.image_path,
                "path_ft": frame.path_ft.tolist() if frame.path_ft is not None else None,
                "path_bl": frame.path_bl.tolist() if frame.path_bl is not None else None,
                "path_gt": frame.path_gt.tolist() if frame.path_gt is not None else None,
            }
            data["frames"][frame.timestamp] = frame_dict

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

    def load_model(self, finetuned: bool = True, ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = None
        noise_scheduler = None
        
        if self.config.get("model_type") in {"vint", "gnm", "nomad"}:
            ckpt_path = self.config["chop_finetuned_path"] if finetuned else self.config["pretrained_model_path"]
            sys.modules["vint_train"] = importlib.import_module("policy_sources.visualnav_transformer.train.vint_train")
            sys.modules["vint_train.models"] = importlib.import_module("policy_sources.visualnav_transformer.train.vint_train.models")
            sys.modules["vint_train.models.vint"] = importlib.import_module("policy_sources.visualnav_transformer.train.vint_train.models.vint")

            model = deployment_load_model(str(ckpt_path), self.config, device)

            if self.model_name == "nomad":
                noise_scheduler = DDPMScheduler(
                    num_train_timesteps=self.config["num_diffusion_iters"],
                    beta_schedule='squaredcos_cap_v2',
                    clip_sample=True,
                    prediction_type='epsilon'
                )

        elif self.model_name == "omnivla":
            if finetuned:
                vla_config = self.vla_config_finetuned
            else:
                vla_config = self.vla_config

            model = Inference(save_dir="./inference",
                            ego_frame_mode=True,
                            save_images=False, 
                            radians=True,
                            vla_config=vla_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        if model is None:
            raise RuntimeError("Model failed to initialize.")

        # Some wrappers (e.g., OmnivLA Inference) are not nn.Modules; guard attribute usage.
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        return model, noise_scheduler
    
    def run_inference(self, model, frame: FrameItem, goal_frame: FrameItem, noise_scheduler=None):
        if hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start_idx = max(0, frame.frame_idx - self.context_frames)
        context_imgs = [f.image[:, :, ::-1] for f in self.frames[start_idx:frame.frame_idx + 1]]  # BGR -> RGB
        context_pil = [PILImage.fromarray(img) for img in context_imgs]
        goal_pil = PILImage.fromarray(goal_frame.image[:, :, ::-1])

        if self.model_name in {"vint", "gnm"}:
            obs_tensor = transform_images(context_pil, self.config["image_size"])
            goal_tensor = transform_images(goal_pil, self.config["image_size"])
            obs_tensor = obs_tensor.to(device)
            goal_tensor = goal_tensor.to(device)
            with torch.no_grad():
                _, action_pred = model(obs_tensor, goal_tensor)
            path_xy = action_pred[0, :, :2].detach().cpu().numpy()
        elif self.model_name == "nomad":
            if noise_scheduler is None:
                raise RuntimeError("Noise scheduler required for NoMaD inference.")
            obs_images = transform_images(context_pil, self.config["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(device)
            goal_tensor = transform_images(goal_pil, self.config["image_size"], center_crop=False).to(device)
            mask = torch.zeros(1, device=device).long()

            obsgoal_cond = model('vision_encoder', obs_img=obs_images, goal_img=goal_tensor, input_goal_mask=mask)
            obs_cond = obsgoal_cond

            num_diffusion_iters = self.config["num_diffusion_iters"]
            noise_scheduler.set_timesteps(num_diffusion_iters)
            with torch.no_grad():
                noisy_action = torch.randn((1, self.config["len_traj_pred"], 2), device=device)
                naction = noisy_action
                for k in noise_scheduler.timesteps:
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            naction = get_action(naction)
            path_xy = naction[0, :, :2].detach().cpu().numpy()
            path_xy = path_xy * 0.38 # Scale to meter To.do: make this configurable
        elif self.model_name == "omnivla":
            cur_img = PILImage.fromarray(frame.image[:, :, ::-1]) #BGR to RGB
            cur_pos = frame.pos
            cur_yaw = frame.yaw

            goal_img = PILImage.fromarray(goal_frame.image[:, :, ::-1])
            goal_pos = goal_frame.pos
            goal_yaw = goal_frame.yaw

            model.update_current_state(cur_img, cur_pos, cur_yaw)
            model.update_goal(goal_image_PIL=goal_img, 
                                    goal_utm=goal_pos,
                                    goal_compass=goal_yaw, 
                                    lan_inst_prompt=None)
            model.run()
            waypoints = model.waypoints.reshape(-1, model.waypoints.shape[-1])
            path_xy = waypoints[:, :2] * model.metric_waypoint_spacing  # Convert to meters
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        # Normalize to shape (N, 2) for downstream metrics
        path_xy = np.asarray(path_xy)
        if path_xy.ndim == 3:
            path_xy = path_xy.reshape(-1, path_xy.shape[-1])
        if path_xy.shape[-1] > 2:
            path_xy = path_xy[:, :2]

        return path_xy
    
    def sample_goal_indices(self):
        goal_dist = np.random.uniform(self.min_distance, self.max_distance, size=len(self.frames))
        for i in range(self.context_frames, len(self.frames)-1):
            cur_dist = self.frames[i].cum_distance
            for j in range(i + 1, len(self.frames)):
                next_dist = self.frames[j].cum_distance
                if next_dist - cur_dist >= goal_dist[i]:
                    self.frames[i].goal_idx = j
                    break
            else:
                self.frames[i].goal_idx = j

    def process_image(self, msg):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return cv_img
    
    def process_laserscan_msg(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angle_min = float(msg.angle_min)
        angle_increment = float(msg.angle_increment)
        range_min = float(msg.range_min)
        range_max = min(float(msg.range_max), self.max_distance)

        # Replace NaNs with inf
        ranges = np.nan_to_num(ranges, nan=np.inf)
        # Clip very large values
        ranges[ranges > range_max] = np.inf

        # Apply FOV mask (centered at 0 yaw)
        if self.fov_angle < 360.0:
            angles = angle_min + np.arange(len(ranges)) * angle_increment
            half_fov = np.deg2rad(self.fov_angle) / 2.0
            mask = (angles >= -half_fov) & (angles <= half_fov)
            ranges = np.where(mask, ranges, np.inf)

        return ranges, angle_min, angle_increment, range_min, range_max
    
    def process_odom(self, msg):
        quaternion = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        return pos, yaw

    def prefill_paths(self, finetuned: bool = True):
        model, noise_scheduler = self.load_model(finetuned=finetuned)

        for i, frame in enumerate(tqdm(self.frames, desc=f"Prefill {self.bag_name}", unit="frame")):
            if frame.goal_idx == -1:
                continue
            
            goal_frame = self.frames[frame.goal_idx]
            path_xy = self.run_inference(model, frame, goal_frame, noise_scheduler=noise_scheduler)
            if path_xy is None:
                raise RuntimeError("Inference failed to produce a path.")
            elif path_xy.size == 0:
                raise RuntimeError("Inference produced an empty path.")
            if finetuned:
                self.frames[i].path_ft = path_xy
            else:
                self.frames[i].path_bl = path_xy

    def preprocess_bag(self, bag_path: str):
        self.bag_name = Path(bag_path).name
        stem = Path(self.bag_name).stem
        self.frames = []

        self.timestamps = self._get_timestamps_from_expert_annotations()

        if len(self.timestamps) == 0:
            print(f"[WARN] No timestamps found in expert annotations for {self.bag_name}, skipping bag.")
            return

        print(f"\n=== Processing {self.bag_name} ===")

        if "Jackal" in self.bag_name:
            self.image_topic = "/camera/rgb/image_raw/compressed"
            self.laserscan_topic = "/velodyne_2dscan"
            self.odom_topic = "/jackal_velocity_controller/odom"
        elif "Spot" in self.bag_name:
            self.image_topic = "/image_raw/compressed"
            self.laserscan_topic = "/scan"
            self.odom_topic = "/odom"

        # print(self.timestamps[:10])
        skip_count = 0
        timestamp_counter = 0

        # Filling self.frames with rosbag data
        with rosbag.Bag(bag_path, "r") as bag:

            count = 0
            scan_data = None
            last_pos = None
            pos = None
            yaw = None
            cum_distance = 0.0
            for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[self.image_topic, self.laserscan_topic, self.odom_topic])):
                # print(int(str(t)), int(self.timestamps[timestamp_counter]), timestamp_counter, len(self.timestamps))
                if timestamp_counter >= len(self.timestamps):
                    # print(len(self.timestamps), timestamp_counter)
                    break
                if(int(str(t)) > int(self.timestamps[timestamp_counter]) and pos is None):
                    while (int(str(t)) > int(self.timestamps[timestamp_counter]) and timestamp_counter < len(self.timestamps)):
                        timestamp_counter += 1
                        skip_count += 1
                if topic == self.odom_topic:
                    pos, yaw = self.process_odom(msg)
                elif topic == self.image_topic:
                    cv_img = self.process_image(msg)
                elif topic == self.laserscan_topic:
                    scan_data = self.process_laserscan_msg(msg)
                if str(t) == str(self.timestamps[timestamp_counter]):
                    if cv_img is not None and scan_data is not None and pos is not None and yaw is not None:
                        if last_pos is None:
                            cum_distance = 0.0
                        else:
                            cum_distance += np.linalg.norm(pos - last_pos)
                        last_pos = pos
                        ranges, angle_min, angle_increment, range_min, range_max = scan_data
                        gt_path = self.pref_annotations.get("annotations_by_stamp", {}).get(str(t), {}).get("paths", {}).get("3", {}).get("points", None)
                        if gt_path is not None:
                            gt_path = np.array(gt_path, dtype=np.float32)
                            gt_path = _resample_path(gt_path, self.num_points + 1)[1:]  # Resample and drop origin
                        
                        image_filename = f"img_{t}.png"
                        image_path = f"{stem}/img_{str(t)}.png"

                        if (self.image_root / stem / image_filename).is_file():
                                
                            self.frames.append(
                                FrameItem(
                                    frame_idx=count,
                                    image=cv_img,
                                    image_path=image_path,
                                    timestamp=int(str(t)),
                                    laserscan=ranges,
                                    angle_min=angle_min,
                                    angle_increment=angle_increment,
                                    range_min=range_min,
                                    range_max=range_max,
                                    path_gt=gt_path,
                                    pos=pos,
                                    yaw=yaw,
                                    cum_distance=cum_distance,
                                )
                            )
                            count += 1
                        timestamp_counter += 1
                        
            if self.sample_goals:
                self.sample_goal_indices()
    
        print(f"[INFO] Loaded {len(self.frames)} frames from bag after skipping {skip_count} frames.")
        if not self.frames:
            print("[WARN] No frames.")
            # raise Exception
            return
        
        # Filling paths from models. 
        self.prefill_paths(finetuned=True)
        self.prefill_paths(finetuned=False)

    def run(self, processed: list):
        bag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))
        pref_files = sorted(glob.glob(os.path.join(self.pref_annotations_path, "*.json")))
        pref_dict = defaultdict(bool)
        for pf in pref_files:
            pref_dict[Path(os.path.basename(pf)).stem] = True

        if not bag_files:
            print(f"[ERROR] No .bag files found in {self.bag_dir}")
            return
        else:
            print(f"[INFO] Found {len(bag_files)} .bag files in {self.bag_dir}")
        with open(self.test_train_split_path, 'r') as f:
            test_train_bags = json.load(f)

        for bp in bag_files:
            self.bag_name = Path(os.path.basename(bp)).stem
            if test_train_bags.get(self.bag_name, "train") == "train" or not pref_dict.get(self.bag_name, False) or self.bag_name in processed:
                print(f"[INFO] Skipping training bag: {self.bag_name}")
                continue

            if not self.all_evals_from_data:
                self.preprocess_bag(bp)

            for evaluator in self.evaluators:
                evaluator.run(self.bag_name, self.frames)

            # Persist cached paths for downstream metrics.
            if not self.all_evals_from_data:
                self._save_paths_json(self.bag_name)

if __name__ == "__main__":

    output_paths = "./outputs/evals/"
    inference_out = "./outputs/trajectories/"
    model_name = "gnm"
    dataset_split = "./data/annotations/test-train-split.json"
    pref_annotations_path = "./data/annotations/preferences"
    bag_dir = "/media/beast-gamma/Media/Datasets/SCAND/rosbags/"

    proximity_evaluator = ProximityEvaluator(output_path=output_paths, scand=True, model=model_name)
    goal_distance_evaluator = GoalDistanceEvaluator(output_path=output_paths, scand=True, model=model_name)
    alignment_evaluator = AlignmentEvaluator(output_path=output_paths, scand=True, model=model_name)
    near_miss_evaluator = NearMissEvaluator(output_path=output_paths, scand=True, model=model_name)

    # evaluators = [proximity_evaluator, goal_distance_evaluator, alignment_evaluator]
    evaluators = [proximity_evaluator]

    # processed = ["A_Jackal_Fountain_Library_Fri_Oct_29_9", "A_Jackal_REC_Abandon_Sat_Nov_13_92", "A_Spot_AHG_AHG_Mon_Nov_8_27"]
    # processed = ["A_Jackal_Fountain_Library_Fri_Oct_29_9", "A_Jackal_REC_Abandon_Sat_Nov_13_92", 
    #              "A_Spot_AHG_AHG_Mon_Nov_8_27", "A_Spot_AHG_Library_Fri_Nov_5_21", 
    #              "A_Spot_Bass_Rec_Fri_Nov_26_126", "A_Spot_Dobie_Dobie_Thu_Nov_11_73",
    #              "A_Spot_EER_OsCafe_Tue_Nov_9_39", "A_Spot_GDC_AHG_Thu_Nov_18_124", 
    #              "A_Spot_Jester_Jester_Wed_Nov_10_63", "A_Spot_Library_AHG_Mon_Nov_8_28",
    #              "A_Spot_Library_Fountain_Fri_Nov_12_82", "A_Spot_Library_MLK_Thu_Nov_18_123",
    #              "A_Spot_Library_MLK_Thu_Nov_18_123", "B_Spot_AHG_Library_Tue_Nov_9_34", 
    #              "B_Spot_Fountain_Union_Tue_Nov_9_36", "B_Spot_JCL_JCL_Mon_Nov_15_108", 
    #              "B_Spot_Library_Jester_Thu_Nov_11_75", "A_Spot_Security_NHB_Wed_Nov_10_54" ]
    processed = []

    runner = EvalRunner(
        bag_dir=bag_dir,
        output_path=output_paths,
        inference_out=inference_out,
        test_train_split_path=dataset_split,
        pref_annotations_path=pref_annotations_path,
        model=model_name,
        fov_angle=90.0,
        num_points=8,
        sample_goals=True,
        min_distance=2.0,
        max_distance=20.0,
        evaluators=evaluators
    )
    runner.run(processed)
