#!/usr/bin/env python3
import argparse
import math
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PlannerPIDNode(Node):
    def __init__(self, cmd_vel):
        super().__init__("planner_pid")

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=15,
        )

        # Tunable params
        self.declare_parameter("k_linear_p", 1.0)
        self.declare_parameter("k_linear_i", 0.0)
        self.declare_parameter("k_linear_d", 0.0)
        self.declare_parameter("k_angular_p", 2.0)
        self.declare_parameter("k_angular_i", 0.0)
        self.declare_parameter("k_angular_d", 0.0)
        self.declare_parameter("max_linear", 0.35)
        self.declare_parameter("max_angular", 1.0)
        self.declare_parameter("goal_tolerance", 0.1)
        self.declare_parameter("yaw_tolerance", 0.35)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("integral_limit", 1.0)

        self.k_lp = float(self.get_parameter("k_linear_p").value)
        self.k_li = float(self.get_parameter("k_linear_i").value)
        self.k_ld = float(self.get_parameter("k_linear_d").value)
        self.k_ap = float(self.get_parameter("k_angular_p").value)
        self.k_ai = float(self.get_parameter("k_angular_i").value)
        self.k_ad = float(self.get_parameter("k_angular_d").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.goal_tol = float(self.get_parameter("goal_tolerance").value)
        self.yaw_tol = float(self.get_parameter("yaw_tolerance").value)
        self.control_dt = 1.0 / float(self.get_parameter("control_rate_hz").value)
        self.integral_limit = float(self.get_parameter("integral_limit").value)

        # State
        self._lock = threading.Lock()
        self._pose: Optional[tuple[float, float, float]] = None  # x, y, yaw (odom frame)
        self._goal: Optional[tuple[float, float]] = None         # x, y (world frame)
        self._goal_done = False
        self._lin_i = 0.0
        self._ang_i = 0.0
        self._prev_lin_err: Optional[float] = None
        self._prev_ang_err: Optional[float] = None
        self.cmd_vel = cmd_vel

        # ROS I/O
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel, 10)
        self.pub_req_goal = self.create_publisher(Empty, "/req_goal", 10)

        self.create_subscription(Odometry, "/odom", self.on_odom, self.qos_profile)
        self.create_subscription(PoseStamped, "/next_goal", self.on_goal, self.qos_profile)

        # self.create_timer(self.control_dt, self._control_step)

        self.get_logger().info(
            f"planner_pid ready (k_lin={self.k_lp}, k_ang={self.k_ap}, "
            f"max_v={self.max_linear}, max_w={self.max_angular}, "
            f"goal_tol={self.goal_tol}, yaw_tol={self.yaw_tol})"
        )

    # Callbacks
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._lock:
            self._pose = (p.x, p.y, yaw)

    def on_goal(self, msg: PoseStamped):
        with self._lock:
            self._goal = (msg.pose.position.x, msg.pose.position.y)
            self._goal_done = False
            self._lin_i = 0.0
            self._ang_i = 0.0
            self._prev_lin_err = None
            self._prev_ang_err = None

    # Control
    def _control_step(self):
        with self._lock:
            if self._pose is None or self._goal is None:
                return
            x, y, yaw = self._pose
            gx, gy = self._goal
            dx = gx - x
            dy = gy - y
            dist = math.hypot(dx, dy)
            heading_err = wrap_angle(math.atan2(dy, dx) - yaw)
            goal_done = self._goal_done

        if dist < self.goal_tol and abs(heading_err) < self.yaw_tol:
            if not goal_done:
                self.pub_req_goal.publish(Empty())
                with self._lock:
                    self._goal_done = True
                    self._lin_i = 0.0
                    self._ang_i = 0.0
                    self._prev_lin_err = None
                    self._prev_ang_err = None
            cmd = Twist()
            self.pub_cmd.publish(cmd)
            return

        # Scale linear down when misaligned to avoid lateral drift
        alignment = max(0.0, math.cos(heading_err))

        # PID terms
        lin_err = dist
        ang_err = heading_err

        lin_der = 0.0 if self._prev_lin_err is None else (lin_err - self._prev_lin_err) / self.control_dt
        ang_der = 0.0 if self._prev_ang_err is None else (ang_err - self._prev_ang_err) / self.control_dt

        self._lin_i = max(-self.integral_limit, min(self.integral_limit, self._lin_i + lin_err * self.control_dt))
        self._ang_i = max(-self.integral_limit, min(self.integral_limit, self._ang_i + ang_err * self.control_dt))

        v_cmd = alignment * (self.k_lp * lin_err + self.k_li * self._lin_i + self.k_ld * lin_der)
        w_cmd = self.k_ap * ang_err + self.k_ai * self._ang_i + self.k_ad * ang_der

        self._prev_lin_err = lin_err
        self._prev_ang_err = ang_err

        v_cmd = max(-self.max_linear, min(self.max_linear, v_cmd))
        w_cmd = max(-self.max_angular, min(self.max_angular, w_cmd))

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.pub_cmd.publish(cmd)
   
    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)  # Process incoming messages
            self._control_step()

def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description="Run the Path Manager")
    parser.add_argument("--cmd", type=str, default='/cmd_vel', help="Command topic name")
    args, ros_args = parser.parse_known_args()
    node = PlannerPIDNode(cmd_vel=args.cmd)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
