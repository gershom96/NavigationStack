# NavigationStack

Reusable navigation stack originally built during CHOP, now structured as a general template for deploying vision-navigation policies and evaluating them offline on robotics datasets.

## What this repo contains

This repository has two main components:

- `deployment/`: ROS2 runtime nodes for model inference, path management, and control.
- `evaluation/`: offline evaluation pipeline for running models on bag files and scoring trajectory quality/safety.

The stack currently supports model families used in CHOP:

- `vint`
- `gnm`
- `nomad`
- `omnivla`

## Repository structure

```text
.
├── deployment/
│   ├── model_run.py              # Inference node: camera/goal/odom -> /path
│   ├── path_manager.py           # Reframes path, publishes /next_goal + /active_path
│   ├── planner_pid_ros2.py       # PID controller for /next_goal -> /cmd_vel
│   ├── planner_dwa_ros2.py       # DWA controller variant
│   ├── planner_omnivla_ros2.py   # OmnivLA-style controller variant
│   ├── camera_matrix.json        # Camera intrinsics/extrinsics example
│   └── utils/
└── evaluation/
    ├── eval_runner.py            # Main offline runner (inference + metrics + trajectory cache)
    ├── proximity_evaluate.py     # Obstacle-clearance metric
    ├── goal_distance_evaluate.py # Goal-reaching metric
    ├── alignment_evaluate.py     # GT alignment metric
    ├── near_miss_evaluate.py     # Near-miss count based on proximity logs
    ├── calculate_dataset_statistics.py
    ├── plot_eval_distributions.py
    ├── visualize_trajs.py
    └── utils/
```

## Deployment dataflow (ROS2)

`deployment/model_run.py` and `deployment/path_manager.py` form the core deployment loop:

1. `model_run.py` subscribes to robot observations and publishes predicted path on `/path`.
2. `path_manager.py` snapshots the start frame (`/started`), converts the predicted path to current/world frame, and publishes:
   - `/next_goal` (for controller)
   - `/active_path` (for monitoring)
   - `/path_overlay` (optional image overlay)
3. A planner node (`planner_pid_ros2.py`, `planner_dwa_ros2.py`, or `planner_omnivla_ros2.py`) tracks `/next_goal` and publishes velocity commands.

Core topic contracts:

- Inputs to model node: odometry + compressed camera image, and optional goal image/pose.
- Model output: `/path` (`nav_msgs/Path`) in start robot frame.
- Path manager output: `/next_goal` (`geometry_msgs/PoseStamped`) in world frame.

## Quick start (deployment)

Run each node in a separate terminal inside a ROS2 environment:

```bash
# 1) Model inference node
python deployment/model_run.py \
  --model omnivla \
  --config ./configs/chop_inference_run.yaml \
  --odom /odom_lidar \
  --image /camera/camera/color/image_raw/compressed

# 2) Path manager
python deployment/path_manager.py \
  --config ./deployment/camera_matrix.json \
  --odom /odom_lidar \
  --image /camera/camera/color/image_raw/compressed

# 3) Controller (choose one)
python deployment/planner_pid_ros2.py --cmd /cmd_vel
# or
python deployment/planner_dwa_ros2.py --cmd /cmd_vel
# or
python deployment/planner_omnivla_ros2.py --cmd /cmd_vel
```

## Evaluation pipeline

`evaluation/eval_runner.py` does the heavy lifting:

- Loads bag files + preference annotations.
- Samples goals by traveled-distance window.
- Runs both checkpoints (finetuned and baseline/pretrained).
- Saves generated trajectories to JSON cache:
  - `outputs/trajectories/<model>/<bag>_paths.json`
- Runs selected evaluators and writes JSONL logs under:
  - `outputs/evals/<metric_name>/`

Run:

```bash
python evaluation/eval_runner.py
```

Then summarize metrics over all bags:

```bash
python evaluation/calculate_dataset_statistics.py --eval-root outputs/evals
```

Plot distributions:

```bash
python evaluation/plot_eval_distributions.py
```

Visualize predicted trajectories on images:

```bash
python evaluation/visualize_trajs.py \
  --model omnivla \
  --bag A_Spot_AHG_Library_Fri_Nov_5_21 \
  --traj-root outputs/trajectories \
  --image-root /media/beast-gamma/Media/Datasets/SCAND/images \
  --save-root outputs/visualization \
  --camera-config evaluation/scand_cameras.json
```

## Expected external dependencies

This code assumes external CHOP modules are available on `PYTHONPATH`, notably:

- `policy_sources.visualnav_transformer`
- `policy_sources.omnivla`
- `datasets.preprocess_scand_a_chop`

It also expects a ROS + CV + ML stack (examples):

- `rclpy`, ROS message packages (`geometry_msgs`, `nav_msgs`, `sensor_msgs`, `std_msgs`)
- `rosbag`, `cv_bridge`
- `numpy`, `scipy`, `opencv-python`, `Pillow`
- `torch`, `diffusers`, `matplotlib`, `tqdm`, `PyYAML`

## Reusing this stack in new projects

This repo is best used as a skeleton:

1. Keep the ROS interfaces (`/path`, `/started`, `/next_goal`, `/req_goal`) stable.
2. Adapt model loading/inference behavior in `deployment/model_run.py` and `evaluation/utils/loaders.py`.
3. Point dataset paths/configs to your environment (bags, annotations, images, checkpoints).
4. Keep evaluators modular: add new metrics by subclassing `evaluation/base_evaluator.py`.

## Notes

- `pyproject.toml` is currently a placeholder and does not yet declare package metadata/dependencies.
- Several default paths in scripts are CHOP/local-machine specific; update them before use in a new environment.
