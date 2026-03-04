#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import threading
from typing import Optional, List
import os, sys, importlib
import yaml 

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
from cv_bridge import CvBridge

# Default topomap location (relative to this file)
# TOPOMAP_IMAGES_ROOT = os.path.join(os.path.dirname(__file__), "topomaps", "images")
TOPOMAP_IMAGES_ROOT = "/workspace/chop/policy_sources/visualnav_transformer/deployment/topomaps/images"  # updated path

@dataclass
class FrameItem:
    image: np.ndarray
    pos: Optional[np.ndarray] = None
    yaw: Optional[float] = None

@dataclass
class ContextFrame:
    image: np.ndarray

class ModelNode(Node):
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        finetuned: bool = False,
        topomap_name: str = "topomap",
        goal_node: int = -1,
        radius: int = 4,
        close_threshold: int = 3,
        waypoint_index: int = 2,
        odom_topic: str = "/odom",
        image_topic: str = "/camera/image_raw/compressed",
    ):
        super().__init__("model_node")

        self.qos_profile  = QoSProfile(
                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST,  
                        depth=15  
                    )
        
        self.qos_profile_r  = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,  
                depth=15  
            )
        # ---------- Params ----------
        self.declare_parameter("path_frame_id", "base_link")  # semantic frame name
        self.declare_parameter("waypoint_spacing", 0.38)                 # spacing

        self.path_frame_id = self.get_parameter("path_frame_id").value
        self.waypoint_spacing = float(self.get_parameter("waypoint_spacing").value)
        self.config_path = config_path
        self.model_name = model_name
        self.finetuned = finetuned
        self.topomap_name = topomap_name
        self.goal_node_arg = goal_node
        self.radius = radius
        self.close_threshold = close_threshold
        self.waypoint_index = waypoint_index

        self.odom_topic = odom_topic
        self.image_topic = image_topic

        # ---------- State ----------
        self._lock = threading.Lock()
        self._started_sent = False

        # latest observation cache (set by callbacks)
        self._have_cur_img = False
        self._have_goal_img = False
        self._have_cur_pose = False
        self._have_goal_pose = False
        self._have_context = False
        self.topomap: Optional[List[PILImage.Image]] = None
        self.closest_node: int = 0
        self.goal_node_idx: Optional[int] = None

        self.config = self._load_config()
        self.context_update_period = self.config.get("context_update_period", 0.3)
        self._cv = threading.Condition(self._lock)

        self._dirty = False          # something changed since last inference
        self._shutdown = False
        self._inference_running = False  # acts as "busy"
         
        # ---------- ROS I/O ----------
        self.pub_started = self.create_publisher(Empty, "/started", 10)
        self.pub_path = self.create_publisher(Path, "/path", 10)

        self.bridge = CvBridge()
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, qos_profile=self.qos_profile)
        self.sub_goal_img = self.create_subscription(CompressedImage, "/goal/image/compressed", self.on_goal_image, 10)
        self.sub_goal_pose = self.create_subscription(PoseStamped, "/goal/pose", self.on_goal_pose, 10)
        self.sub_nav = self.create_subscription(Empty, "/nav_cmd", self.on_nav_cmd, 10)
        self.context_timer = self.create_timer(self.context_update_period, self.update_context_from_current)
        self._worker = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker.start()
        self.get_logger().info(f"worker alive={self._worker.is_alive()}")
        

        self.create_subscription(CompressedImage, self.image_topic, self.on_image, 10)

        self.get_logger().info(
            f"step_m={self.waypoint_spacing}, frame_id={self.path_frame_id}, config_path={self.config_path}"
        )

        self.model, self.noise_scheduler = self._load_model(finetuned=self.finetuned)

        if self.model_name in {"vint", "gnm", "nomad"}:
            self._load_topomap()

        self.cur_frame = FrameItem(
            image=None,
            pos=None,
            yaw=None
        )

        self.goal_frame = FrameItem(
            image=None,
            pos=None,
            yaw=None
        )

        self.context_frames = [ContextFrame(image=None) for _ in range(self.config.get("context_size", 0) + 1)]
        self.goal_img_needed = self.config.get("need_goal_img", True)

        print(f"ModelNode initialized with model {self.model_name}.")

    def _to_path_msg(self, path_xy: np.ndarray) -> Path:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id  # semantic: "start frame"

        for x, y in path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        return msg

    def _load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if self.model_name not in config:
            raise ValueError(f"Model {self.model_name} not found in config.")
        
        with open(config[self.model_name]["config_path"], "r") as f:
            config_model = yaml.safe_load(f) 

        return config_model

    def _load_model(self, finetuned: bool = True, ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = None
        noise_scheduler = None
        
        # Implement model loading logic based on self.model_name and finetuned flag

        return model, noise_scheduler

    def _load_topomap(self):
        """Load ordered topomap images from the given directory."""
        topomap_dir = os.path.join(TOPOMAP_IMAGES_ROOT, self.topomap_name)
        if not os.path.isdir(topomap_dir):
            raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")
        filenames = sorted(
            [f for f in os.listdir(topomap_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        if not filenames:
            raise RuntimeError(f"No images found in topomap directory: {topomap_dir}")
        self.topomap = [PILImage.open(os.path.join(topomap_dir, f)) for f in filenames]
        if self.goal_node_arg == -1:
            self.goal_node_idx = len(self.topomap) - 1
        else:
            if not (0 <= self.goal_node_arg < len(self.topomap)):
                raise ValueError(f"goal_node {self.goal_node_arg} out of range for topomap of size {len(self.topomap)}")
            self.goal_node_idx = self.goal_node_arg
        self.closest_node = 0
        self.get_logger().info(
            f"Loaded topomap '{self.topomap_name}' with {len(self.topomap)} nodes. Goal node: {self.goal_node_idx}"
        )

    def _inference_worker(self):
        while True:
            # print(f"Waiting for inference conditions: shutdown={self._shutdown}, dirty={self._dirty}, ready={self._ready_to_infer_locked()}, running={self._inference_running}")

            with self._cv:
                # Wait until: shutdown OR (dirty and ready and not busy)
                self._cv.wait_for(lambda: self._shutdown or
                                        (self._dirty and self._ready_to_infer_locked() and not self._inference_running))

                if self._shutdown:
                    return

                # Claim work
                self._inference_running = True
                self._dirty = False

                # Snapshot inputs (shallow copies are fine; you can deep-copy later if needed)
                cur = FrameItem(
                    image=None if self.cur_frame.image is None else self.cur_frame.image.copy(),
                    pos=None if self.cur_frame.pos is None else self.cur_frame.pos.copy(),
                    yaw=self.cur_frame.yaw
                )
                goal = FrameItem(
                    image=None if self.goal_frame.image is None else self.goal_frame.image.copy(),
                    pos=None if self.goal_frame.pos is None else self.goal_frame.pos.copy(),
                    yaw=self.goal_frame.yaw
                )

                # Snapshot context images (optional; safe)
                ctx_imgs = [ContextFrame(image=cf.image.copy()) for cf in self.context_frames if cf.image is not None]

                model = self.model
                noise_scheduler = self.noise_scheduler

                # Publish /started once, when we actually start inferencing
                if not self._started_sent:
                    self._started_sent = True
                    self._have_cur_img = False
                    self._have_cur_pose = False
                    self.pub_started.publish(Empty())
                    self.get_logger().info("Published /started (once).")

            # ---- Run inference outside lock ----
            try:
                path_xy = self.run_inference(model=model, cur_frame=cur, goal_frame=goal, context_frames=ctx_imgs, noise_scheduler=noise_scheduler)
            except Exception as e:
                self.get_logger().error(f"Inference failed: {repr(e)}")
                path_xy = None
            # path_xy = self.run_inference(model=model, cur_frame=cur, goal_frame=goal, context_frames=ctx_imgs, noise_scheduler=noise_scheduler)
            if path_xy is not None:
                self._started_sent = False
                try:
                    self.pub_path.publish(self._to_path_msg(path_xy))
                except Exception as e:
                    self.get_logger().error(f"Publishing /path failed: {repr(e)}")

            with self._cv:
                self._inference_running = False
                # If something became dirty while we were inferencing, loop will run again immediately

    def _trigger_inference(self):
        with self._cv:
            self._dirty = True
            self._cv.notify()
    
    def _ready_to_infer_locked(self) -> bool:

        if self.model_name in {"vint", "gnm", "nomad"}:
            # only need current image and context for topomap-based visual nav
            if not (self._have_cur_img and self._have_context):
                return False
        elif self.model_name == "omnivla":
            if not (self._have_cur_img and (self._have_goal_img or self._have_goal_pose) and self._have_cur_pose):
                return False
        # Add additional model-specific conditions here if needed
        return True
    # ---------------- callbacks ----------------
    def on_goal_image(self, msg: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self.goal_frame.image = img
            self._have_goal_img = True
        self._trigger_inference()

    def on_image(self, msg: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._have_cur_img = True
            self.cur_frame.image = img
        self._trigger_inference()

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        with self._lock:
            self.cur_frame.pos = np.array([p.x, p.y])
            self.cur_frame.yaw = yaw
            self._have_cur_pose = True
        self._trigger_inference()

    def on_goal_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        with self._lock:
            self.goal_frame.pos = np.array([p.x, p.y])
            self.goal_frame.yaw = yaw
            self._have_goal_pose = True
        self._trigger_inference()

    def on_nav_cmd(self, _msg: Empty):
        # Placeholder for navigation trigger; currently no-op
        return

    def update_context_from_current(self):
        with self._lock:
            if not self.context_frames:
                return
            if self.cur_frame.image is None:
                return
            self.context_frames.pop(0)
            self.context_frames.append(ContextFrame(image=self.cur_frame.image.copy()))
            self._have_context = all(cf.image is not None for cf in self.context_frames)

        self._trigger_inference()

    def run_inference(self, model, cur_frame, goal_frame, context_frames, noise_scheduler=None):
        if hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.model_name in {"vint", "gnm", "nomad"}:
            context_imgs = [f.image[:, :, ::-1] for f in context_frames]  # BGR -> RGB
            context_pil = [PILImage.fromarray(img) for img in context_imgs]

            start = max(self.closest_node - self.radius, 0)
            end = min(self.closest_node + self.radius + 1, self.goal_node_idx)
            
            if self.model_name in {"vint", "gnm"}:                # ViNT/GNM
                batch_obs_imgs = []
                batch_goal_data = []
                for sg_img in self.topomap[start : end + 1]:
                    transf_obs_img = transform_images(context_pil, self.config["image_size"])
                    goal_data = transform_images(sg_img, self.config["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, paths = model(batch_obs_imgs, batch_goal_data)
                distances = distances.detach().cpu().numpy()
                paths = paths.detach().cpu().numpy()
                min_dist_idx = int(np.argmin(distances))
                if distances[min_dist_idx] > self.close_threshold:
                    chosen_path = paths[min_dist_idx]
                    self.closest_node = start + min_dist_idx
                else:
                    chosen_path = paths[min(min_dist_idx + 1, len(paths) - 1)]
                    self.closest_node = min(start + min_dist_idx + 1, self.goal_node_idx)
                # convert to path_xy
                path_xy = np.array(chosen_path[:, :2]).reshape(chosen_path.shape[0], 2)
            
            elif self.model_name == "nomad":
                if noise_scheduler is None:
                    raise RuntimeError("Noise scheduler required for NoMaD inference.")
                obs_images = transform_images(context_pil, self.config["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1).to(device)
                mask = torch.zeros(1, device=device).long()

                goal_image = [
                    transform_images(g_img, self.config["image_size"], center_crop=False).to(device)
                    for g_img in self.topomap[start : end + 1]
                ]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model(
                    'vision_encoder',
                    obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                    goal_img=goal_image,
                    input_goal_mask=mask.repeat(len(goal_image)),
                )
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = dists.detach().cpu().numpy().flatten()
                min_idx = int(np.argmin(dists))
                self.closest_node = min_idx + start
                sg_idx = min(min_idx + int(dists[min_idx] < self.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # sample actions via diffusion
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
                path_xy = path_xy * self.waypoint_spacing  # meters
        elif self.model_name == "omnivla":
            cur_img = PILImage.fromarray(cur_frame.image[:, :, ::-1]) #BGR to RGB
            cur_pos = cur_frame.pos
            cur_yaw = cur_frame.yaw

            if goal_frame.image is None:
                base_img = np.zeros((self.config["image_size"][0], self.config["image_size"][1], 3), dtype=np.uint8)
                goal_img = PILImage.fromarray(base_img[:, :, ::-1])
            else:
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
            path_xy = waypoints[:, :2] * self.waypoint_spacing  # Convert to meters
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        # Normalize to shape (N, 2) for downstream metrics
        path_xy = np.asarray(path_xy)
        if path_xy.ndim == 3:
            path_xy = path_xy.reshape(-1, path_xy.shape[-1])
        if path_xy.shape[-1] > 2:
            path_xy = path_xy[:, :2]

        return path_xy
    
    def destroy_node(self):
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="Run the model node")
    parser.add_argument("-c", "--config", type=str, help="Path to config file", default="./configs/chop_inference_run.yaml")
    parser.add_argument("-m", "--model", type=str, help="Model name", default="omnivla")
    parser.add_argument("--finetuned", action="store_true", help="Use finetuned weights")
    parser.add_argument("--topomap", type=str, default="topomap", help="Topomap directory name under topomaps/images")
    parser.add_argument("--goal-node", type=int, default=-1, help="Goal node index (-1 uses last node)")
    parser.add_argument("--radius", type=int, default=4, help="Temporal radius of nodes to consider for localization")
    parser.add_argument("--close-threshold", type=int, default=3, help="Distance threshold to advance to next node")
    parser.add_argument("--waypoint-index", type=int, default=2, help="Index of waypoint to use from model outputs")
    parser.add_argument("--odom", type=str, default="/odom_lidar", help="Odom topic name")
    parser.add_argument("--image", type=str, default="/camera/camera/color/image_raw/compressed", help="Image topic name")

    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ModelNode(
        config_path=args.config,
        model_name=args.model,
        finetuned=args.finetuned,
        topomap_name=args.topomap,
        goal_node=args.goal_node,
        radius=args.radius,
        close_threshold=args.close_threshold,
        waypoint_index=args.waypoint_index,
        odom_topic=args.odom,
        image_topic=args.image,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
