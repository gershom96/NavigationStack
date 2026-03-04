from scipy.spatial import cKDTree
import numpy as np
from evaluation.utils.pointcloud_utils import pointcloud_to_laserscan, median_filter_laserscan, laserscan_to_obstacle_points

def interpolate_path(path_xy, step=0.05):
    interp = [path_xy[0]]
    for p0, p1 in zip(path_xy[:-1], path_xy[1:]):
        d = np.linalg.norm(p1 - p0)
        n = max(1, int(d / step))
        for i in range(1, n + 1):
            interp.append(p0 + (p1 - p0) * (i / n))
    return np.array(interp)

def min_clearance_to_obstacles_pcld(path_xy, cloud_msg, z_min, z_max, angle_min, angle_max, angle_increment, range_min, range_max):
    
    laserscan = pointcloud_to_laserscan(
        cloud_msg=cloud_msg,  # Placeholder, as we already have cloud_xyz
        z_min=z_min,
        z_max=z_max,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=angle_increment,
        range_min=range_min,
        range_max=range_max
    )
    laserscan = median_filter_laserscan(laserscan, kernel_size=5)
    obs_xy = laserscan_to_obstacle_points(laserscan, angle_min, angle_increment, range_max)

    if obs_xy.size == 0:
        return np.inf

    path_xy = interpolate_path(path_xy, step=0.1)
    tree = cKDTree(obs_xy)
    dists, _ = tree.query(path_xy)
    return dists.min()

def min_clearance_to_obstacles_ls(path_xy, laserscan, angle_increment, angle_min, angle_max, range_min, range_max):

    laserscan = median_filter_laserscan(laserscan, kernel_size=5)
    obs_xy = laserscan_to_obstacle_points(laserscan, angle_min, angle_increment, range_max)

    if obs_xy.size == 0:
        return np.inf

    # print(path_xy)
    path_xy = interpolate_path(path_xy, step=0.1)
    tree = cKDTree(obs_xy)
    dists, _ = tree.query(path_xy)
    return dists.min()
