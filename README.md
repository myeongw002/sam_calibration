# SAM Calibration Pipeline

This repository contains a camera-LiDAR calibration workflow built around:
- SAM2 automatic image mask generation
- 3D point cloud preprocessing and projected cluster masks
- mask-to-cluster matching and iterative extrinsic pose refinement

The main pipeline is implemented in `scripts/calibration.py`.

## Project Layout

- `scripts/automatic_mask_generator.py`: Generate SAM2 masks (`raw_masks.npz`, metadata, overlays)
- `scripts/calibration.py`: End-to-end calibration (preprocess, matching, optimization, outputs)
- `scripts/cluster_projection.py`: Standalone cluster projection/debug script (contains hardcoded paths)
- `scripts/find_mask_cluster_groups.py`: Standalone mask-group matching script
- `sam2/`: Local SAM2 source and checkpoints
- `data/kitti/`: KITTI-style sample data and outputs
- `data/nuscenes/`: NuScenes-style data folder

## Requirements

Recommended environment:
- Linux
- Python 3.10+
- CUDA GPU (optional but strongly recommended for SAM2)

Python packages used by scripts:
- `numpy`
- `opencv-python`
- `scipy`
- `open3d`
- `torch`
- `Pillow`

SAM2 is vendored in `sam2/`, and the mask generator defaults to:
- SAM2 repo path: `/workspace/sam_calibration/sam2`
- checkpoint: `/workspace/sam_calibration/sam2/checkpoints/sam2.1_hiera_large.pt`

## Quick Start (Docker)

If you already built an image named `sam_calib_img`, run:

```bash
docker run -it --rm \
  --name sam_calib \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XDG_RUNTIME_DIR=/tmp/runtime-docker \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/sam_calib_ws:/workspace \
  sam_calib_img bash
```

Inside container:

```bash
cd /workspace/sam_calibration
```

## End-to-End Pipeline

### 1) Generate SAM2 masks

```bash
python scripts/automatic_mask_generator.py \
  --image /workspace/sam_calibration/data/kitti/images/000002.png \
  --output-dir /workspace/sam_calibration/data/kitti/output/images
```

Main output files:
- `/workspace/sam_calibration/data/kitti/output/images/raw_masks.npz`
- `/workspace/sam_calibration/data/kitti/output/images/raw_masks_metadata.json`
- `/workspace/sam_calibration/data/kitti/output/images/raw_overlay.png`
- `/workspace/sam_calibration/data/kitti/output/images/raw_mask_pngs/`

### 2) Run calibration

```bash
python scripts/calibration.py \
  --image /workspace/sam_calibration/data/kitti/images/000002.png \
  --pc /workspace/sam_calibration/data/kitti/pc/000002.pcd \
  --sam-npz /workspace/sam_calibration/data/kitti/output/images/raw_masks.npz \
  --sam-meta /workspace/sam_calibration/data/kitti/output/images/raw_masks_metadata.json \
  --output-dir /workspace/sam_calibration/data/kitti/output/calibration_run \
  --debug-progress
```

Main output files:
- `calibration_result.json`: initial and optimized transform, optimizer history
- `final_projected_clusters.png`: projected cluster masks after optimization
- `final_group_matches.png`: selected mask-group and cluster matching visualization
- `final_pointcloud_projection.png`: final point cloud reprojection
- `final_matches.txt`: top ranked cluster-mask matches
- `gt_pointcloud_projection.png`, `gt_pose_error.json` (if GT block is available in code)

## Useful Optional Scripts

### Cluster projection only

```bash
python scripts/cluster_projection.py
```

Note: this script currently uses hardcoded input/output paths and camera/extrinsic values in code.

### Mask-group matching only

```bash
python scripts/find_mask_cluster_groups.py \
  --image /workspace/sam_calibration/data/kitti/images/000002.png \
  --sam-npz /workspace/sam_calibration/data/kitti/output/images/raw_masks.npz \
  --sam-meta /workspace/sam_calibration/data/kitti/output/images/raw_masks_metadata.json \
  --cluster-binary-dir /workspace/sam_calibration/data/kitti/output/pc/projection_overlay_cluster_masks/binary_masks \
  --cluster-boundary-dir /workspace/sam_calibration/data/kitti/output/pc/projection_overlay_cluster_masks/boundary_masks \
  --output-dir /workspace/sam_calibration/data/kitti/output/pairs \
  --max-group-size 3 \
  --top-k 15
```

## Notes and Current Limitations

- `scripts/calibration.py` currently uses hardcoded camera intrinsics `K` and initial transform `T0` in code.
- `scripts/calibration.py` also includes a hardcoded GT transform block used for error reporting/visualization.
- `scripts/automatic_mask_generator.py` currently saves raw masks only (no extra post-filter CLI in the current version).
- `scripts/lidar_img.py` is currently empty.

## Data Convention

Example KITTI paths used by scripts:
- image: `data/kitti/images/<frame>.png`
- point cloud: `data/kitti/pc/<frame>.pcd` (or `.bin`)
- outputs: `data/kitti/output/...`

## Troubleshooting

- If `open3d` import fails, install Open3D for your Python version.
- If SAM2 is slow on CPU, run with CUDA and verify checkpoint exists at:
  `sam2/checkpoints/sam2.1_hiera_large.pt`
- If projection results look off, first verify image/point-cloud frame pairing and initial pose assumptions.
