# SEGMVVS360 — Semantic Equirectangular Gaussian Mixture Virtual Visual Servoing

**Author:** Hussein Loubani — `Hussein.loubani@utbm.fr`
**Institution:** CIAD-UTBM

SEGMVVS360 is a 360° visual servoing pipeline that drives a robot to a target
pose by minimizing the difference between the current and desired
**equirectangular** views. Building facades are used as landmarks: each frame
is segmented into a binary building mask, the mask is convolved into a smooth
**Gaussian-Mixture image** `G`, and the control law minimizes `e = G - Gd`
through a Gauss–Newton iteration on a 7-DOF interaction matrix
(6 pose DOFs + 1 adaptive Gaussian-width DOF `λ`).

## 1. Architecture

The launch file starts three ROS nodes plus one GPU service:

| Node / Service | Type | Role |
|---|---|---|
| `real_image_publisher` | Python | Publishes the equirectangular RGB sequence on `/real_image` and exposes `/total_targets` |
| `building_segmentation_service` | Python (GPU) | OneFormer-based service that returns the building mask, the semantic visualization, and the dynamic-occlusion mask |
| `gaussian_mixture_server` | Python (GPU, PyTorch) | Computes `G`, `dG/du`, `dG/dv`, `dG/dλ` from a binary mask via FFT-based circular convolution on the equirectangular plane |
| `segmvvs360_node` | C++ | Main VS loop: synchronizes image+pose, calls the segmentation and GMM services, builds the interaction matrix, runs the control law, publishes the next pose on `/skyline_controller/set_pose` |

The visual-servoing schedule has two stages:
- **Stage 1** — `λ = lambda_step1` (broad Gaussians, fast convergence)
- **Stage 2** — `λ = lambda_step2` (tight Gaussians, refinement)

`λ` is also part of the optimization (7th DOF), so the schedule is just an
initialization for each stage.

## 2. Dependencies

- ROS Noetic
- ViSP 3.5.0 + `visp_bridge`
- Boost (`filesystem`)
- CUDA 12.x + PyTorch ≥ 2.1 (for the GMM and segmentation services)
- HuggingFace Transformers (`shi-labs/oneformer_cityscapes_swin_large`)
- The companion package [`gaussian_mixture`](../gaussian_mixture) (in this repo)

A Docker image with the full environment is available at `docker/Dockerfile`
in the parent project (ROS Noetic + ViSP 3.5.0 + CUDA 12.2 + PyTorch 2.1.2).

## 3. Build

```bash
cd ~/catkin_ws/src
git clone https://github.com/husein-loubani/SEGMVVS-360
cd ..
catkin build SEGMVVS360 gaussian_mixture
source devel/setup.bash
```

## 4. Launch parameters

Edit [`launch/SEGMVVS360.launch`](launch/SEGMVVS360.launch) before running.

### Image and control

| Param | Default | Meaning |
|---|---|---|
| `im_width` / `im_height` | 320 / 160 | Equirectangular working resolution |
| `im_size_factor` | 1.0 | Multiplicative factor on top of width/height |
| `gain_step1` / `gain_step2` | 0.5 / 0.5 | Control-law gain for each stage |
| `lambda_step1` / `lambda_step2` | 10.0 / 1.0 | Initial Gaussian width for each stage |
| `iterations_step1` / `iterations_step2` | 15 / 15 | Iterations spent in each stage |

### Topics

| Param | Default | Subscribed by |
|---|---|---|
| `real_image_topic` | `/real_image` | publisher → segmentation service |
| `mask_topic` | `/skyline/mask` | binary building mask |
| `color_topic` | `/skyline/color` | colored semantic visualization |
| `depth_topic` | `/skyline/depth` | per-pixel scene depth |
| `get_pose_topic` | `/skyline_controller/pose` | current robot pose (input) |
| `set_pose_topic` | `/skyline_controller/set_pose` | next robot pose (output) |

### Services

| Param | Default |
|---|---|
| `pgm_service_name` | `/compute_equirectangular_gaussian_mixture_lambda` |
| `seg_service_name` | `/building_segmentation` |

### Segmentation

| Param | Default | Meaning |
|---|---|---|
| `target_class_ids` | `[2]` | Cityscapes classes treated as buildings |
| `dynamic_class_ids` | `[11..18]` | Vehicles + pedestrians masked out |
| `use_dynamic_mask` | `true` | Whether to remove dynamic-class pixels from the mask |

### Publisher

| Param | Default | Meaning |
|---|---|---|
| `image_dir` | `share/frames/` | Folder containing the equirectangular RGB sequence |
| `skip_stride` | `20` | Publish 1 frame out of every N |
| `max_frames` | `-1` | Optional cap on the number of frames (negative = unlimited) |

### Output

| Param | Default | Meaning |
|---|---|---|
| `results_root` | `share/Results` | Folder where the run's outputs are written |
| `save_results` | `true` | Master toggle: write per-frame initial/final RGB, masks, GMMs, depth visualizations |
| `save_all_iterations` | `false` | If `true`, also save every Gauss–Newton iteration of every frame (much larger output) |

The pose log (`pose_log.txt`) is always written: it goes inside `results_root`
when set, otherwise to the current working directory.

## 5. Usage

1. Make sure your camera driver and robot controller are publishing the topics
   listed above.
2. Launch:
   ```bash
   roslaunch SEGMVVS360 SEGMVVS360.launch
   ```
3. Wait for the GMM and segmentation services to come up (the node blocks
   until both are ready).
4. Follow the on-screen instructions to select the **initial pose** by clicking
   on the real-image window. The pose at the moment of the click becomes the
   desired pose.
5. The visual-servoing loop then runs through the published frames and writes
   outputs into `results_root`.

## 6. Outputs

Inside `results_root` (when `save_results=true`):

| Folder | Contents |
|---|---|
| `real_rgb_images/` | Equirectangular RGB at each frame |
| `real_binary_masks/` | Building mask used as desired image at each frame |
| `real_semantic_segmentation_images/` | Colored semantic visualization |
| `initial_real_gaussian_mixtures/` / `final_real_gaussian_mixtures/` | `Gd` and `G` per frame |
| `initial_synth_albedo/` / `final_synth_albedo/` | Synthetic image at the initial / converged pose |
| `initial_synth_depth_images/` / `final_synth_depth_images/` | Synthetic depth visualizations |
| `all_iterations/` | Per-iteration dumps (only if `save_all_iterations=true`) |
| `pose_log.txt` | One line per frame: `tx ty tz θux θuy θuz` |
| `vel_log.txt` | One line per VS iteration: `vx vy vz wx wy wz λ` |
| `time_log.txt` | Wall-clock timestamp per frame |

## 7. Citation

If you use this code, please cite the corresponding paper (ICRA 2026, in
submission). BibTeX entry will be added once the paper is accepted.
