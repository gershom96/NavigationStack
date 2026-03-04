"""
Lightweight viewer to overlay baseline vs finetuned (and GT) trajectories on frame images.

Usage:
    python evaluation/visualize_trajs.py --model vint --bag A_Spot_AHG_Library_Fri_Nov_5_21

Defaults look for trajectories under outputs/trajectories/<model>/<bag>_paths.json
and images under /media/beast-gamma/Media/Datasets/SCAND/images/<bag>/img_<stamp>.png

Controls:
    Left/Right arrows : previous/next frame
    s key or "Save" button : save current overlay to outputs/visualization/<bag>/<model>/<frame>.png
    q key or close window   : exit
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from deployment.utils.visualization import (
    project_points_cam,
    transform_points,
    draw_corridor,
    draw_polyline,
    load_calibration,
    overlay_path
)
from deployment.utils.path_gen import make_corridor_polygon

# Colors in BGR for OpenCV drawing
COLOR_BL = (255, 165, 0)  # orange baseline
COLOR_FT = (0, 0, 255)    # red finetuned
CORRIDOR_FILL = {
    "bl": (0, 140, 255),  # lighter orange
    "ft": (0, 0, 255),
}

def load_frames(traj_path: Path) -> List[Tuple[int, Dict]]:
    with open(traj_path, "r") as f:
        data = json.load(f)
    frames = []
    for ts, frame in data["frames"].items():
        frames.append((int(ts), frame))
    frames.sort(key=lambda x: x[0])
    return frames


class TrajViewer:
    def __init__(self, args: argparse.Namespace, bag: Optional[str] = None):
        self.model = args.model

        if bag:
            self.bag = bag
        else:
            self.bag = args.bag

        self.image_root = Path(args.image_root).expanduser()
        self.traj_path = Path(args.traj_root).expanduser() / self.model / f"{self.bag}_paths.json"
        self.save_root = Path(args.save_root).expanduser() / self.bag / self.model
        self.save_root.mkdir(parents=True, exist_ok=True)

        if "Spot" in self.bag:
            spot = True
            jackal = False
        elif "Jackal" in self.bag:
            spot = False
            jackal = True
        else:
            spot = False
            jackal = False
        self.K, self.dist, self.T_base_from_cam = load_calibration(args.camera_config, spot=spot, jackal=jackal)
        self.T_cam_from_base = np.linalg.inv(self.T_base_from_cam)
        self.corridor_width = args.corridor_width
        self.sync_saved = not getattr(args, "no_sync", False)

        if not self.traj_path.is_file():
            raise FileNotFoundError(f"Traj file not found: {self.traj_path}")

        self.frames = load_frames(self.traj_path)
        if not self.frames:
            raise RuntimeError("No frames in trajectory JSON.")

        self.idx = 2200
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.18)
        self.img_artist = None

        # Save button
        ax_save = plt.axes([0.8, 0.05, 0.12, 0.07])
        self.btn_save = Button(ax_save, "Save", color="#4CAF50", hovercolor="#66BB6A")
        self.btn_save.on_clicked(lambda event: self.save_current())

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        if self.sync_saved:
            self._sync_saved_frames()
        self.update()

    def load_image(self, rel_path: str) -> np.ndarray:
        img_path = self.image_root / rel_path
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        return img

    def project_and_draw(self, img: np.ndarray, path: Optional[List[List[float]]], color, fill_color) -> np.ndarray:
        if path is None:
            return img
        pts = np.asarray(path, dtype=np.float64)
        if pts.size == 0:
            return img
        # densify and build corridor for better visibility
        pts3 = np.hstack([pts[:, :2], np.zeros((pts.shape[0], 1))])  # z=0
        left_b, right_b, poly_b = make_corridor_polygon(pts3, width_m=self.corridor_width)

        traj_c = transform_points(self.T_cam_from_base, pts3)
        left_c = transform_points(self.T_cam_from_base, left_b)
        right_c= transform_points(self.T_cam_from_base, right_b)
        poly_c = transform_points(self.T_cam_from_base, poly_b)

        ctr_2d  = project_points_cam(self.K, self.dist, traj_c)
        left_2d = project_points_cam(self.K, self.dist, left_c)
        right_2d= project_points_cam(self.K, self.dist, right_c)
        poly_2d = project_points_cam(self.K, self.dist, poly_c)

        draw_corridor(img, poly_2d, left_2d, right_2d,
                      fill_alpha=0.30, fill_color=fill_color, edge_color=color, edge_thickness=2)
        draw_polyline(img, ctr_2d, 2, color)
        return img

    def render_frame(self):
        ts, frame = self.frames[self.idx]
        img = self.load_image(frame["img_path"])
        path_bl = np.array(frame.get("path_bl", []), dtype=np.float64)
        path_ft = np.array(frame.get("path_ft", []), dtype=np.float64)
        img = overlay_path(path_bl, img, self.K, self.T_cam_from_base, COLOR_BL, CORRIDOR_FILL["bl"])
        img = overlay_path(path_ft, img, self.K, self.T_cam_from_base, COLOR_FT, CORRIDOR_FILL["ft"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ts, img_rgb

    def update(self):
        ts, img_rgb = self.render_frame()
        self.ax.clear()
        self.ax.imshow(img_rgb)
        self.ax.set_title(f"{self.bag} | {self.model} | frame {self.idx+1}/{len(self.frames)} | ts {ts}")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ["right", "d"]:
            self.idx = (self.idx + 1) % len(self.frames)
            self.update()
        elif event.key in ["left", "a"]:
            self.idx = (self.idx - 1) % len(self.frames)
            self.update()
        elif event.key == "s":
            self.save_current()
        elif event.key in ["q", "escape"]:
            plt.close(self.fig)

    def save_current(self):
        ts, img_rgb = self.render_frame()
        out_path = self.save_root / f"{self.idx:04d}_{ts}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved {out_path}")

    def _parse_saved_indices(self, root: Path) -> List[int]:
        indices = []
        for png in root.glob("*.png"):
            stem = png.stem
            if "_" not in stem:
                continue
            prefix = stem.split("_", 1)[0]
            if prefix.isdigit():
                indices.append(int(prefix))
        return indices

    def _sync_saved_frames(self):
        """
        If other models have already saved overlays for this bag, auto-save
        the corresponding frames for the current model to ease comparison.
        """
        bag_dir = self.save_root.parent  # .../outputs/visualization/<bag>
        if not bag_dir.exists():
            return
        wanted_indices = []
        for model_dir in bag_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == self.model:
                continue
            wanted_indices.extend(self._parse_saved_indices(model_dir))

        if not wanted_indices:
            return

        wanted_indices = sorted(set(i for i in wanted_indices if 0 <= i < len(self.frames)))
        print(f"[INFO] Syncing {len(wanted_indices)} frames from other models for {self.bag}...")
        current_saved = set(self._parse_saved_indices(self.save_root))

        for idx in wanted_indices:
            if idx in current_saved:
                continue
            self.idx = idx
            self.save_current()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Trajectory viewer")
    parser.add_argument("--model", required=True, help="Model name (subfolder under traj_root).")
    parser.add_argument("--bag", required=False, default="B_Spot_JCL_JCL_Mon_Nov_15_108", help="Bag stem without .bag")
    parser.add_argument("--traj-root", default="outputs/trajectories", help="Root containing <model>/<bag>_paths.json")
    parser.add_argument("--image-root", default="/media/beast-gamma/Media/Datasets/SCAND/images", help="Root of extracted images")
    parser.add_argument("--save-root", default="outputs/visualization", help="Where to save overlays when requested")
    parser.add_argument("--camera-config", default="evaluation/scand_cameras.json", help="Camera intrinsics + per-robot extrinsics JSON")
    parser.add_argument("--corridor-width", type=float, default=0.5, help="Robot/corridor width in meters")
    parser.add_argument("--no-sync", action="store_true", help="Disable auto-saving frames already saved by other models for this bag.")
    return parser.parse_args()


def main():
    args = parse_args()
    bag = "A_Spot_AHG_Library_Fri_Nov_5_21"

    if bag:
        viewer = TrajViewer(args, bag=bag)
    else:
        viewer = TrajViewer(args)

    plt.show()


if __name__ == "__main__":
    main()

# python evaluation/visualize_trajs.py \
#   --model vint \
#   --bag A_Spot_AHG_Library_Fri_Nov_5_21 \
#   --traj-root outputs/trajectories \
#   --image-root /media/beast-gamma/Media/Datasets/SCAND/images \
#   --save-root outputs/visualization \
#   --corridor-width 0.5
