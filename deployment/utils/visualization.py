import json
import math
from typing import Optional
import numpy as np
import cv2
from deployment.utils.path_gen import make_corridor_polygon

from scipy.interpolate import CubicSpline

COLOR_FILL = (0, 0, 255)  # Red
COLOR_EDGE = (0, 0 , 200) # Darker Red

def load_calibration(json_path: str, spot: Optional[bool]=False, jackal: Optional[bool]=False):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if data is None or ("H_cam_bl" not in data and "spot" not in data and "jackal" not in data):
        raise ValueError(f"Missing H_cam_bl in {json_path}")

    if spot:
        h = data["spot"]["H_cam_bl"]
    elif jackal:
        h = data["jackal"]["H_cam_bl"]
    else:
        h = data["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    xt, yt, zt = float(h["x"]), float(h["y"]), float(h["z"])

    # Rotation about +y (camera pitched down is positive pitch if y up/right-handed)
    Ry = np.array([
        [ 0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll),  math.sin(roll)]
    ], dtype=np.float64)

    T_base_from_cam = np.eye(4, dtype=np.float64)
    T_base_from_cam[:3, :3] = Ry
    T_base_from_cam[:3, 3]  = np.array([xt, yt, zt], dtype=np.float64)

    fx = data["Intrinsics"]["fx"]
    fy = data["Intrinsics"]["fy"]
    cx = data["Intrinsics"]["cx"]
    cy = data["Intrinsics"]["cy"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = None  # explicitly no distortion
    return K, dist, T_base_from_cam


def overlay_path(pts_cur: np.ndarray, img: Optional[np.ndarray] = None, cam_matrix: Optional[np.ndarray] = None,
                 T_cam_from_base: Optional[np.ndarray] = None, 
                 fill_color = COLOR_FILL, edge_color = COLOR_EDGE) -> Optional[np.ndarray]:
    if pts_cur.size == 0:
        return
    if cam_matrix is None or T_cam_from_base is None:
        return
    if img is None:
        return
    
    dense_pts = densify_path(pts_cur, num_points=1000)
    if dense_pts is None or dense_pts.size == 0:
        return

    # Lift 2D (x,y) into 3D (z=0) before transforms
    dense_pts3 = np.hstack([dense_pts, np.zeros((dense_pts.shape[0], 1))])
    left_b, right_b, poly_b = make_corridor_polygon(dense_pts3, width_m=0.5)

    traj_c = transform_points(T_cam_from_base, dense_pts3)
    left_c = transform_points(T_cam_from_base, left_b)
    right_c= transform_points(T_cam_from_base, right_b)
    poly_c = transform_points(T_cam_from_base, poly_b)
    
    ctr_2d  = project_points_cam(cam_matrix, None, traj_c)
    left_2d = project_points_cam(cam_matrix, None, left_c)
    right_2d= project_points_cam(cam_matrix, None, right_c)
    poly_2d = project_points_cam(cam_matrix, None, poly_c)

    draw_corridor(img, poly_2d, left_2d, right_2d,
                        fill_alpha=0.35, fill_color=fill_color, edge_color=edge_color, edge_thickness=2)
    draw_polyline(img, ctr_2d, 2, fill_color)

    return img

def densify_path(pts_cur: np.ndarray, num_points: int = 1000) -> np.ndarray:
    if pts_cur.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    x, y = pts_cur[:,0], pts_cur[:,1]
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts_cur, axis=0), axis=1))]
    if t[-1] == 0:
        return np.empty((0, 2), dtype=np.float64)  # all points identical
    t /= t[-1]
    
    sx, sy = CubicSpline(t, x), CubicSpline(t, y)
    tq = np.linspace(0, 1, num_points)
    Px, Py = sx(tq), sy(tq)

    dense_points = np.stack([Px, Py], 1)

    return dense_points

def project_points_cam(K: np.ndarray, dist, P_cam: np.ndarray) -> np.ndarray:
    """Project Nx3 camera-frame points to pixels. No distortion if dist is None."""
    if P_cam is None:
        return np.empty((0, 2), dtype=np.float32)
    P_cam = np.asarray(P_cam, dtype=np.float64).reshape(-1, 3)
    if P_cam.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Drop non-finite rows to avoid cv2 errors
    finite_mask = np.all(np.isfinite(P_cam), axis=1)
    P_cam = P_cam[finite_mask]
    if P_cam.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(P_cam.astype(np.float64), rvec, tvec, K, None)
    return pts2d.reshape(-1, 2)

def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points; returns Nx3."""
    assert T.shape == (4, 4)
    N = P.shape[0]
    Ph = np.hstack([P, np.ones((N, 1))])
    Qh = (T @ Ph.T).T
    return Qh[:, :3]

def draw_corridor(img: np.ndarray, poly_2d: np.ndarray, left_2d: np.ndarray, right_2d: np.ndarray,
                  fill_alpha: float = 0.35,
                  fill_color = (0,0,255),   # BGR
                  edge_color = (0,0,200),
                  edge_thickness: int = 2,):
    H, W = img.shape[:2]
    # Clip to image bounds
    def clip_pts(uv, polygon=False):
        pts = []
        for (u,v) in uv:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                pts.append([ui, vi])
        if polygon and len(pts) >= 3:
            # Ensure polygon is closed
            if pts[0] != pts[-1]:
                pts.append(pts[0])

        return np.array(pts, dtype=np.int32)

    poly = clip_pts(poly_2d, polygon=True)
    L = clip_pts(left_2d)
    R = clip_pts(right_2d)

    if len(poly) >= 3:
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], fill_color)
        img[:] = cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0)

    if len(L) >= 2:
        cv2.polylines(img, [L], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)
    if len(R) >= 2:
        cv2.polylines(img, [R], isClosed=False, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)

def draw_polyline(img: np.ndarray, pts2d: np.ndarray, thickness: int, color):
    H, W = img.shape[:2]
    poly = []
    for (uu, vv) in pts2d:
        ui, vi = int(round(uu)), int(round(vv))
        if 0 <= ui < W and 0 <= vi < H:
            poly.append((ui, vi))
    for i in range(len(poly) - 1):
        cv2.line(img, poly[i], poly[i + 1], color, thickness, lineType=cv2.LINE_AA)
