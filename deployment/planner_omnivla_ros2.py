#!/usr/bin/env python3
import math
import threading
import argparse

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from scipy.spatial.transform import Rotation as R

def clip_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PlannerOmniVLANode(Node):
    def __init__(self, cmd_vel: str = '/cmd_vel'):
        super().__init__("planner_omnivla")

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=15,
        )

        # Tunables mirroring the vanilla OmnivLA controller
        self.declare_parameter("dt", 1.0 / 3.0)
        self.declare_parameter("linear_clip", 0.5)
        self.declare_parameter("angular_clip", 1.0)
        self.declare_parameter("max_v", 0.3)
        self.declare_parameter("max_w", 0.3)
        self.declare_parameter("goal_tolerance", 0.1)
        self.declare_parameter("yaw_tolerance", 0.35)
        self.declare_parameter("control_rate_hz", 20.0)

        self.dt = float(self.get_parameter("dt").value)
        self.lin_clip = float(self.get_parameter("linear_clip").value)
        self.ang_clip = float(self.get_parameter("angular_clip").value)
        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.goal_tol = float(self.get_parameter("goal_tolerance").value)
        self.yaw_tol = float(self.get_parameter("yaw_tolerance").value)
        self.control_dt = 1.0 / float(self.get_parameter("control_rate_hz").value)

        # State
        self._lock = threading.Lock()
        self._pose = None  # (x, y, yaw) in odom/world
        self._goal = None  # (x, y) in odom/world
        self._goal_done = False
        self.cmd_vel = cmd_vel

        # ROS I/O

        choice = input("Publish? 1 or 0: ")
        
        if(int(choice) == 1):
            self.pub_cmd = self.create_publisher(Twist, self.cmd_vel, 10)
            print("Publishing to cmd_vel")
        else:
            self.pub_cmd = self.create_publisher(Twist, "/dont_publish", 1)
            print("Not publishing!")

        self.req_goal_pub = self.create_publisher(Empty, "/req_goal", 10)
        self.pub_req_goal = self.create_publisher(Empty, "/req_goal", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, self.qos_profile)
        self.create_subscription(PoseStamped, "/next_goal", self.on_goal, self.qos_profile)
# 
        # self.create_timer(self.control_dt, self._control_step)

        self.get_logger().info(
            f"planner_omnivla ready (dt={self.dt}, lin_clip={self.lin_clip}, ang_clip={self.ang_clip}, "
            f"max_v={self.max_v}, max_w={self.max_w}, goal_tol={self.goal_tol}, yaw_tol={self.yaw_tol})"
        )

    # Callbacks
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        rot_q = msg.pose.pose.orientation
        roll,pitch,yaw = R.from_quat([rot_q.x, rot_q.y, rot_q.z, rot_q.w]).as_euler('xyz')

        with self._lock:
            self._pose = (p.x, p.y, yaw)

    def on_goal(self, msg: PoseStamped):
        with self._lock:
            self._goal = (msg.pose.position.x, msg.pose.position.y)
            self._goal_done = False

    def atGoal(self, dist, heading_err):
        if self._pose is None or self._goal is None:
            return False
        elif dist <= self.goal_tol and abs(heading_err) <= self.yaw_tol:
            return True
        return False

    # Control
    def _control_step(self):
        with self._lock:
            if self._pose is None or self._goal is None:
                return
            x, y, yaw = self._pose
            gx, gy = self._goal
            goal_done = self._goal_done

        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)
        heading_err = clip_angle(math.atan2(dy, dx) - yaw)

        cmd = Twist()
        if self.atGoal(dist, heading_err):
            self.pub_req_goal.publish(Empty())
            with self._lock:
                self._goal_done = True
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        else:
            linear_vel_value, angular_vel_value = self._compute_cmd(dx, dy, heading_err)
            cmd.linear.x = linear_vel_value
            cmd.angular.z = angular_vel_value
        print(cmd)
        self.pub_cmd.publish(cmd)

    def _comp_compute_cmdute_cmd(self, dx: float, dy: float, heading_err: float):
        EPS = 1e-8
        dt = self.dt

        if abs(dx) < EPS and abs(dy) < EPS:
            linear_vel_value = 0.0
            angular_vel_value = clip_angle(heading_err) / dt
        elif abs(dx) < EPS:
            linear_vel_value = 0.0
            angular_vel_value = math.copysign(math.pi / (2 * dt), dy)
        else:
            linear_vel_value = dx / dt
            angular_vel_value = math.atan(dy / dx) / dt

        linear_vel_value = float(np.clip(linear_vel_value, 0.0, self.lin_clip))
        angular_vel_value = float(np.clip(angular_vel_value, -self.ang_clip, self.ang_clip))

        maxv, maxw = self.max_v, self.max_w
        if abs(linear_vel_value) <= maxv:
            if abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * math.copysign(abs(rd), linear_vel_value)
                angular_vel_value_limit = maxw * math.copysign(1.0, angular_vel_value)
        else:
            if abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * math.copysign(1.0, linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * math.copysign(1.0, linear_vel_value)
                    angular_vel_value_limit = maxv * math.copysign(1.0, angular_vel_value) / abs(rd)
                else:
                    linear_vel_value_limit = maxw * math.copysign(abs(rd), linear_vel_value)
                    angular_vel_value_limit = maxw * math.copysign(1.0, angular_vel_value)

        return linear_vel_value_limit, angular_vel_value_limit
    
    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)  # Process incoming messages
            self._control_step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Path Manager")
    parser.add_argument("--cmd", type=str, default='/cmd_vel', help="Command topic name")
    args, ros_args = parser.parse_known_args()
    rclpy.init()
    node = PlannerOmniVLANode(cmd_vel=args.cmd)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
