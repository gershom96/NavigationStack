import numpy as np
import sensor_msgs.point_cloud2 as pc2
from math import atan2, sqrt, pi

def pointcloud_to_laserscan(
    cloud_msg,
    z_min=-0.1,
    z_max=0.5,
    angle_min=-pi,
    angle_max=pi,
    angle_increment=0.005,
    range_min=0.2,
    range_max=20.0,
):
    num_bins = int((angle_max - angle_min) / angle_increment)
    ranges = np.full(num_bins, np.inf)

    for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
        x, y, z = p

        # height filter
        if z < z_min or z > z_max:
            continue

        r = sqrt(x*x + y*y)
        if r < range_min or r > range_max:
            continue

        theta = atan2(y, x)
        if theta < angle_min or theta > angle_max:
            continue

        idx = int((theta - angle_min) / angle_increment)
        ranges[idx] = min(ranges[idx], r)

    return ranges

def median_filter_laserscan(ranges, kernel_size=5):
    filtered_ranges = np.copy(ranges)
    half_k = kernel_size // 2
    for i in range(len(ranges)):
        start_idx = max(0, i - half_k)
        end_idx = min(len(ranges), i + half_k + 1)
        window = ranges[start_idx:end_idx]
        filtered_ranges[i] = np.median(window[np.isfinite(window)])
    return filtered_ranges

def laserscan_to_obstacle_points(
    ranges,
    angle_min=-pi,
    angle_increment=0.005,
    range_max=20.0
):
    points = []
    for i, r in enumerate(ranges):
        if np.isinf(r) or np.isnan(r) or r == range_max:
            continue
        theta = angle_min + i * angle_increment
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append((x, y))
    return np.array(points)