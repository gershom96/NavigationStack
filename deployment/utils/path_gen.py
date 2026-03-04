import numpy as np
from typing import Tuple

def make_corridor_polygon(traj_b: np.ndarray,
                          width_m: float = 0.5, 
                          bridge_pts: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given centerline (N,3) and heading samples (N,), create left/right offsets and a closed polygon.
    width_m: robot width; offsets at ±width_m/2.
    Returns:
      left_b, right_b: (N,3) in base_link
      poly_b: (2N,3) polygon points (left forward then right backward)
    """
    d = width_m * 0.5
    x = traj_b[:, 0]
    y = traj_b[:, 1]
    # normal to heading (x-forward, y-left):

    thetas = create_yaws_from_path(traj_b)  # (N,)
    n_x = -np.sin(thetas)
    n_y =  np.cos(thetas)

    xL = x + d * n_x
    yL = y + d * n_y
    xR = x - d * n_x
    yR = y - d * n_y

    z = np.zeros_like(x)
    left_b  = np.stack([xL, yL, z], axis=1)
    right_b = np.stack([xR, yR, z], axis=1)

    left_b = left_b[left_b[:, 0]>0]
    right_b = right_b[right_b[:, 0]>0]

    if bridge_pts > 0:
        bx = np.linspace(xL[-1], xR[-1], bridge_pts)
        by = np.linspace(yL[-1], yR[-1], bridge_pts)
        bridge_end = np.stack([bx, by, np.zeros_like(bx)], axis=1)
    # Build polygon: left (0→N-1) + right (N-1→0)
    poly_b = np.vstack([left_b, bridge_end,right_b[::-1]])
    return left_b, right_b, poly_b

def create_yaws_from_path(path_b: np.ndarray) -> np.ndarray:
    """Create yaw angles (radians) from a base_link path."""
    deltas = np.diff(path_b[:, :2], axis=0)  # (N-1, 2)
    yaws = np.arctan2(deltas[:, 1], deltas[:, 0])  # (N-1,)
    # Append last yaw to maintain same length
    if len(yaws) > 0:
        yaws = np.concatenate([yaws, yaws[-1:]], axis=0)
    else:
        yaws = np.array([0.0])
    return yaws