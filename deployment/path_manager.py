#!/usr/bin/env python3
import argparse
import math
import threading
from typing import List, Optional, Tuple
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R

from deployment.utils.transformations import start_to_current
from deployment.utils.visualization import load_calibration, overlay_path

class PathManagerNode(Node):
    def __init__(self, camera_config_file: str, visualize: bool, odom_topic: str = "/odom_lidar", image_topic: str = "/camera/camera/color/image_raw/compressed"):
        super().__init__("path_manager")

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
        # ---- Params ----
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("world_frame", "odom")
        self.declare_parameter("behind_margin", 0.09)    # pop if x < -margin
        self.declare_parameter("reach_radius", 0.1)     # pop if dist < radius (extra robustness)
        self.declare_parameter("overlay_enabled", visualize)
        self.declare_parameter("overlay_topic", "/path_overlay")

        self.base_frame = self.get_parameter("base_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.behind_margin = float(self.get_parameter("behind_margin").value)
        self.reach_radius = float(self.get_parameter("reach_radius").value)
        self.overlay_enabled = bool(self.get_parameter("overlay_enabled").value)
        self.overlay_topic = self.get_parameter("overlay_topic").value

        self.cam_matrix, self.dist_coeffs, self.T_base_from_cam = load_calibration(camera_config_file)
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)

        self.odom_topic = odom_topic
        self.image_topic = image_topic

        # ---- State ----
        self._lock = threading.Lock()

        # latest homogeneous from base to world
        self._current_T_w : Optional[np.ndarray] = None
        # snapshot homogeneous from base to world
        self._start_T_w: Optional[np.ndarray] = None

        # path points stored in START robot frame (what model outputs)
        self._path_start_xy: np.ndarray = np.empty((0, 2))
        self._pts_w: np.ndarray = np.empty((0, 2))
        self._image: Optional[np.ndarray] = None

        # ---- ROS I/O ----
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, self.qos_profile)
        self.create_subscription(Empty, "/started", self.on_started, self.qos_profile)
        self.create_subscription(Path, "/path", self.on_path, self.qos_profile)
        self.create_subscription(Empty, "/req_goal", self.on_req_goal, self.qos_profile)

        if self.overlay_enabled:
            self.bridge = CvBridge()
            self.create_subscription(CompressedImage, self.image_topic, self.on_image, 10)
            self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 10)

        self.pub_next_goal = self.create_publisher(PoseStamped, "/next_goal", 10)
        self.pub_active_path = self.create_publisher(Path, "/active_path", 10)

        self.get_logger().info(
            f"PathManagerNode running. base_frame={self.base_frame}, "
            f"behind_margin={self.behind_margin}, reach_radius={self.reach_radius}"
        )

    # ---------------- callbacks ----------------

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        c, s = np.cos(yaw), np.sin(yaw)
        with self._lock:
            self._current_T_w = np.eye(3)
            self._current_T_w[:2, :2] = [[c, -s], [s, c]]
            self._current_T_w[:2, 2]  = [p.x, p.y]

    def on_started(self, _msg: Empty):
        with self._lock:
            if self._current_T_w is None:
                self.get_logger().warn("Got /started but no odom yet; cannot snapshot start pose.")
                return
            self._start_T_w = self._current_T_w.copy()
        self.get_logger().info("Saved start odom pose from /started.")

    def on_path(self, msg: Path):
        if not msg.poses:
            self.get_logger().warn("Received empty /path.")
            with self._lock:
                self._path_start_xy = np.empty((0, 2))
            return

        with self._lock:
            if self._start_T_w is None:
                self.get_logger().warn("Received /path but no /started pose saved; ignoring.")
                return
            # store points in start frame (model frame)
            self._path_start_xy = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in msg.poses])

        # Rebase to current frame + drop behind + publish
        self._drop_behind_and_publish()

    def on_req_goal(self, _msg: Empty):
        # Planner says it reached the current goal -> pop front and publish next
        with self._lock:
            if self._path_start_xy.size != 0:
                self._path_start_xy = self._path_start_xy[1:]
        self._drop_behind_and_publish()

    def on_image(self, msg: CompressedImage):
        if not self.overlay_enabled:
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._image = img.copy()
            if self._pts_w is not None and self._current_T_w is not None:
                _pts_w = self._pts_w.copy()
                T_c = self._current_T_w.copy()
            else:
                T_c = None
                _pts_w = None
        if _pts_w is not None and T_c is not None:
            pts_w_h = np.vstack([_pts_w.T, np.ones(_pts_w.shape[0])])  # (3,N)
            pts_cur = (np.linalg.inv(T_c) @ pts_w_h)[:2, :].T  # (N,2)
            overlay_img = overlay_path(pts_cur, img, self.cam_matrix, self.T_cam_from_base)
            if overlay_img is not None:
                out_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
                out_msg.header.stamp = msg.header.stamp
                out_msg.header.frame_id = self.world_frame
                self.pub_overlay.publish(out_msg)
            else:
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out_msg.header.stamp = msg.header.stamp
                out_msg.header.frame_id = self.world_frame
                self.pub_overlay.publish(out_msg)

    
    # ---------------- core logic ----------------

    def _drop_behind_and_publish(self):
        with self._lock:
            if self._path_start_xy.size == 0:
                return
            T_s = self._start_T_w
            T_c = self._current_T_w
            pts_start = self._path_start_xy.copy()

        if T_s is None or T_c is None:
            return
        
        pts_cur = start_to_current(T_s, T_c, pts_start)  # (N,2)

        dists = np.linalg.norm(pts_cur, axis=1)
        behind = pts_cur[:, 0] < -self.behind_margin
        reached = dists < self.reach_radius
        keep = ~(behind | reached)

        with self._lock:
            self._path_start_xy = self._path_start_xy[keep, :]
        pts_cur = pts_cur[keep, :]

        # current frame -> world frame
        pts_w = []
        T_w_c = T_c
        pts_c_h = np.vstack([pts_cur.T, np.ones(pts_cur.shape[0])])  # (3,N)
        pts_w   = (T_w_c @ pts_c_h)[:2, :].T  # (N,2)

        self._pts_w = pts_w

        self._publish_if_available(pts_w)
        if self.overlay_enabled and pts_w.size != 0 and self._image is not None:
            overlay_img = overlay_path(pts_cur, self._image, self.cam_matrix, self.T_cam_from_base)
            if overlay_img is not None:
                out_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding="bgr8")
                out_msg.header.stamp = self.get_clock().now().to_msg()
                out_msg.header.frame_id = self.world_frame
                self.pub_overlay.publish(out_msg)

    def _publish_if_available(self, pts_world: np.ndarray):

        if pts_world.size == 0:
            return
        gx, gy = pts_world[-1]

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.world_frame
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        self.pub_next_goal.publish(goal)

        path_msg = Path()
        path_msg.header = goal.header
        for x_w, y_w in pts_world:
            ps = PoseStamped()
            ps.header = goal.header
            ps.pose.position.x = float(x_w)
            ps.pose.position.y = float(y_w)
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_active_path.publish(path_msg)

def main():

    parser = argparse.ArgumentParser(description="Run the Path Manager")
    parser.add_argument("-c", "--config", type=str, help="Path to Camera config file", default="./deployment/camera_matrix.json")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results", default=True)
    parser.add_argument("--odom", type=str, default="/odom_lidar", help="Odom topic name")
    parser.add_argument("--image", type=str, default="/camera/camera/color/image_raw/compressed", help="Image topic name")
    args, ros_args = parser.parse_known_args()
    
    rclpy.init()
    node = PathManagerNode(camera_config_file=args.config, visualize=args.visualize, odom_topic=args.odom, image_topic=args.image)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
