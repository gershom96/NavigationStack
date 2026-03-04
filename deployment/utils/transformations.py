import numpy as np
import math

def start_to_current(T_w_start: np.ndarray, T_w_cur: np.ndarray, points_start: np.ndarray) -> np.ndarray:
    """
    points_start: (N,2) in START frame
    returns:      (N,2) in CURRENT frame
    p^c = (T^w_c)^-1 * T^w_s * p^s
    """
    pts = np.asarray(points_start, dtype=np.float64)
    N = pts.shape[0]
    pts_h = np.ones((3, N), dtype=np.float64)
    pts_h[0, :] = pts[:, 0]
    pts_h[1, :] = pts[:, 1]

    T_c_s = np.linalg.inv(T_w_cur) @ T_w_start   # 3x3
    pts_c = (T_c_s @ pts_h)[:2, :].T             # (N,2)

    return pts_c


def odom_to_robot(config, x_odom, y_odom):
    
    # print(x_odom.shape[0])
    x_rob_odom_list = np.asarray([config.x for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([config.y for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob

# def transform_traj_to_costmap(config, traj):
#     # Get trajectory points wrt robot
#     traj_odom = traj[:,0:2]
#     traj_rob_x = (traj_odom[:, 0] - config.x)*math.cos(config.th) + (traj_odom[:, 1] - config.y)*math.sin(config.th)
#     traj_rob_y = -(traj_odom[:, 0] - config.x)*math.sin(config.th) + (traj_odom[:, 1] - config.y)*math.cos(config.th)
#     traj_norm_x = (traj_rob_x/config.costmap_resolution).astype(int)
#     traj_norm_y = (traj_rob_y/config.costmap_resolution).astype(int)

#     # Get traj wrt costmap
#     traj_cm_col = config.costmap_shape[0]/2 - traj_norm_y
#     traj_cm_row = config.costmap_shape[0]/2 - traj_norm_x

#     return traj_cm_col, traj_cm_row
