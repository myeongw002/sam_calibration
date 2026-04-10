#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import importlib
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree, Delaunay, QhullError

try:
    o3d = importlib.import_module("open3d")
except ImportError:  # pragma: no cover
    o3d = None


# =========================================================
# Configuration & Constants
# =========================================================
# Point cloud projection visualization
POINTCLOUD_POINT_RADIUS = 1
POINTCLOUD_MAX_POINTS = 60000
POINTCLOUD_BLEND_RATIO = 0.82  # blend: 82% projection, 18% original

# Point support filtering
SOFT_SIGMA_PX = 10.0
HARD_INSIDE_THRESH = 0.08
SOFT_SUPPORT_THRESH = 0.30
SUPPORT_SCORE_HARD_WEIGHT = 0.55
SUPPORT_SCORE_SOFT_WEIGHT = 0.45

# Mask grouping (support-global beam search)
BEAM_WIDTH = 6
MAX_GROUP_SIZE = 3
QUICK_BBOX_MARGIN = 24
TOP_K_CANDIDATES = 20
SUPPORT_SPLAT_RADIUS = 3

# Projected cluster building
CLUSTER_SPLAT_RADIUS = 2

# =========================================================
# Data classes
# =========================================================
@dataclass
class ProjectedClusterMask:
    cluster_id: int
    points_3d: np.ndarray
    points_2d: np.ndarray
    depths: np.ndarray
    valid_mask: np.ndarray
    raw_point_mask: np.ndarray
    binary_mask: np.ndarray
    boundary_mask: np.ndarray
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    bbox_fill_ratio: float


@dataclass
class MatchResult:
    cluster_id: int
    mask_ids: Tuple[int, ...]
    selection_score: float
    iou: float
    boundary_iou: float
    containment: float
    bbox_iou: float
    area_ratio: float
    centroid_score: float


@dataclass
class SamMaskFeature:
    mask_id: int
    raw_id: int
    mask_bool: np.ndarray
    mask_u8: np.ndarray
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    centroid_xy: Optional[Tuple[float, float]]
    area: int
    sdf: np.ndarray


@dataclass
class SupportStats:
    hard_inside_ratio: float
    soft_support: float
    centroid_score: float
    bbox_scale_score: float
    unsupported_ratio: float
    fragment_penalty: float
    oversize_penalty: float
    total_score: float


@dataclass
class GroupProposal:
    cluster_id: int
    mask_ids: Tuple[int, ...]
    support: SupportStats
    confidence_gap: float = 0.0


@dataclass
class ContinuousFixedPair:
    cluster_id: int
    cluster_points_sampled: np.ndarray   # (K, 3) fixed 3D points
    weight: float
    selection_score: float
    mask_ids: Tuple[int, ...]

    group_mask: np.ndarray               # uint8 0/255
    sdf: np.ndarray                      # float32, positive outside / negative inside
    mask_centroid_xy: np.ndarray         # (2,)
    mask_cov_xy: np.ndarray              # (2,2)
    norm_xy: np.ndarray                  # (2,) for normalization
    outside_sdf_value: float             # penalty for invalid projection


FixedPair = ContinuousFixedPair


# =========================================================
# Logging helpers
# =========================================================
def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def debug(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[DEBUG] {msg}")


# =========================================================
# I/O
# =========================================================
def load_image_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def load_point_cloud_xyz(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    if p.suffix.lower() == ".pcd":
        pcd = o3d.io.read_point_cloud(str(p))
        if pcd.is_empty():
            raise ValueError(f"Loaded PCD is empty: {path}")
        return np.asarray(pcd.points, dtype=np.float32)

    if p.suffix.lower() == ".bin":
        pts = np.fromfile(str(p), dtype=np.float32)
        if pts.size % 4 != 0:
            raise ValueError(f"BIN file does not contain Nx4 float32 points: {path}")
        pts = pts.reshape(-1, 4)
        return pts[:, :3].astype(np.float32)

    raise ValueError(f"Unsupported point cloud format: {p.suffix}")


def parse_kitti_calib(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = np.array([float(x) for x in value.split()], dtype=np.float64)
            data[key] = vals

    if "P2" not in data:
        raise ValueError("KITTI calib missing P2")
    if "Tr_velo_to_cam" not in data and "Tr" not in data:
        raise ValueError("KITTI calib missing Tr_velo_to_cam or Tr")

    P2 = data["P2"].reshape(3, 4)
    K = P2[:, :3]

    tr_key = "Tr_velo_to_cam" if "Tr_velo_to_cam" in data else "Tr"
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = data[tr_key].reshape(3, 4)
    return K, T


def load_gt_transform_from_calib_json(calib_path: str) -> Optional[np.ndarray]:
    p = Path(calib_path)
    if p.suffix.lower() != ".json" or not p.exists():
        return None

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt = data.get("T_lidar_to_cam_gt")
    if not isinstance(gt, dict):
        return None
    if not gt.get("available", False):
        return None

    gt_data = gt.get("data")
    if gt_data is None:
        return None

    arr = np.asarray(gt_data, dtype=np.float64)
    if arr.shape != (4, 4):
        return None
    return arr


def pose_error_metrics(T_est: np.ndarray, T_gt: np.ndarray) -> Dict[str, float]:
    delta = np.linalg.inv(T_gt) @ T_est
    delta_t = float(np.linalg.norm(delta[:3, 3]))

    rot = delta[:3, :3]
    trace_val = float(np.trace(rot))
    cos_theta = (trace_val - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    rot_deg = float(np.degrees(math.acos(cos_theta)))

    return {
        "translation_error_m": delta_t,
        "rotation_error_deg": rot_deg,
    }


def load_sam_masks(npz_path: str, metadata_path: Optional[str] = None) -> List[dict]:
    npz = np.load(npz_path)
    if "masks" not in npz:
        raise ValueError(f"'masks' key not found: {npz_path}")

    stack = npz["masks"]
    if stack.ndim != 3:
        raise ValueError(f"Expected masks with shape (N,H,W), got {stack.shape}")

    metadata: List[dict] = []
    if metadata_path is not None and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    masks: List[dict] = []
    for i in range(stack.shape[0]):
        ann = dict(metadata[i]) if i < len(metadata) else {}
        ann["segmentation"] = stack[i].astype(bool)
        ann["raw_id"] = i
        masks.append(ann)
    return masks


# =========================================================
# Geometry / transforms
# =========================================================
def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v.astype(np.float64)
    return np.array([
        [0.0, -z,  y],
        [z,  0.0, -x],
        [-y, x,  0.0]
    ], dtype=np.float64)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    I = np.eye(3, dtype=np.float64)
    if theta < 1e-12:
        W = skew(phi)
        return I + W + 0.5 * (W @ W)

    W = skew(phi)
    W2 = W @ W
    A = math.sin(theta) / theta
    B = (1.0 - math.cos(theta)) / (theta * theta)
    return I + A * W + B * W2


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    I = np.eye(3, dtype=np.float64)
    if theta < 1e-12:
        W = skew(phi)
        return I + 0.5 * W + (1.0 / 6.0) * (W @ W)

    W = skew(phi)
    W2 = W @ W
    A = (1.0 - math.cos(theta)) / (theta * theta)
    B = (theta - math.sin(theta)) / (theta * theta * theta)
    return I + A * W + B * W2


def se3_exp(delta: np.ndarray) -> np.ndarray:
    """
    delta = [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]
    where rho is translation part in se(3), phi is rotation part.
    """
    if delta.shape != (6,):
        raise ValueError("delta must have shape (6,)")

    rho = delta[:3].astype(np.float64)
    phi = delta[3:].astype(np.float64)

    R = so3_exp(phi)
    V = so3_left_jacobian(phi)
    t = V @ rho

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_delta_pose(T0: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return se3_exp(delta) @ T0


def project_points(points_lidar: np.ndarray, K: np.ndarray, T_lidar_to_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = points_lidar.shape[0]
    pts_h = np.concatenate([points_lidar.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    pts_cam_h = (T_lidar_to_cam @ pts_h.T).T
    pts_cam = pts_cam_h[:, :3]
    depth = pts_cam[:, 2]

    eps = 1e-8
    uvw = (K @ pts_cam.T).T
    u = uvw[:, 0] / np.maximum(uvw[:, 2], eps)
    v = uvw[:, 1] / np.maximum(uvw[:, 2], eps)
    uv = np.stack([u, v], axis=1)

    valid_depth = depth > eps
    return uv, depth, valid_depth


# =========================================================
# Basic mask utilities
# =========================================================


# =========================================================
# Basic mask utilities
# =========================================================
def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_area(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return max(1, x2 - x1 + 1) * max(1, y2 - y1 + 1)


def bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    union = bbox_area(b1) + bbox_area(b2) - inter
    return 0.0 if union <= 0 else float(inter) / float(union)


def expand_bbox(b: Tuple[int, int, int, int], margin: int, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape
    x1, y1, x2, y2 = b
    return (max(0, x1 - margin), max(0, y1 - margin), min(w - 1, x2 + margin), min(h - 1, y2 + margin))


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = mask1 > 0
    m2 = mask2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def mask_containment(cluster_mask: np.ndarray, image_mask: np.ndarray) -> float:
    c = cluster_mask > 0
    m = image_mask > 0
    inter = np.logical_and(c, m).sum()
    denom = c.sum()
    return 0.0 if denom == 0 else float(inter) / float(denom)


def area_ratio(mask1: np.ndarray, mask2: np.ndarray) -> float:
    a1 = int((mask1 > 0).sum())
    a2 = int((mask2 > 0).sum())
    if a1 == 0 or a2 == 0:
        return 0.0
    return min(a1 / a2, a2 / a1)


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_score(mask1: np.ndarray, mask2: np.ndarray, sigma: float = 120.0) -> float:
    c1 = mask_centroid(mask1)
    c2 = mask_centroid(mask2)
    if c1 is None or c2 is None:
        return 0.0
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    d2 = dx * dx + dy * dy
    return float(np.exp(-d2 / (sigma * sigma)))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, flood_inv)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(np.uint8) * 255
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary, dtype=np.uint8)
    out[labels == largest_idx] = 255
    return out


def make_boundary_band(mask: np.ndarray, band_width: int = 3) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return cv2.subtract(mask_u8, eroded)


def boundary_iou(mask1: np.ndarray, mask2: np.ndarray, band_width: int = 3) -> float:
    b1 = make_boundary_band(mask1, band_width) > 0
    b2 = make_boundary_band(mask2, band_width) > 0
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def oversize_penalty(group_mask: np.ndarray, cluster_mask: np.ndarray) -> float:
    ga = int((group_mask > 0).sum())
    ca = int((cluster_mask > 0).sum())
    if ca == 0:
        return 1.0
    return max(0.0, float(ga) / float(ca) - 1.0)


def mask_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def compute_signed_distance_field(mask: np.ndarray) -> np.ndarray:
    """
    positive outside, negative inside, ~0 near boundary
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    outside = cv2.distanceTransform((1 - mask_u8), cv2.DIST_L2, 3)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    sdf = outside - inside
    return sdf.astype(np.float32)


def bilinear_sample(image: np.ndarray, uv: np.ndarray, outside_value: float) -> np.ndarray:
    """
    image: (H, W)
    uv: (N, 2), float subpixel coordinates
    returns: (N,)
    """
    h, w = image.shape[:2]
    if len(uv) == 0:
        return np.empty((0,), dtype=np.float64)

    u = uv[:, 0]
    v = uv[:, 1]

    out = np.full((len(uv),), outside_value, dtype=np.float64)

    valid = (u >= 0.0) & (u <= w - 1.0) & (v >= 0.0) & (v <= h - 1.0)
    if not np.any(valid):
        return out

    uu = u[valid]
    vv = v[valid]

    x0 = np.floor(uu).astype(np.int32)
    y0 = np.floor(vv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = uu - x0
    dy = vv - y0

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    wa = (1.0 - dx) * (1.0 - dy)
    wb = dx * (1.0 - dy)
    wc = (1.0 - dx) * dy
    wd = dx * dy

    out[valid] = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


# =========================================================
# Point cloud preprocessing
# =========================================================
def prefilter_points_by_fov(points_lidar: np.ndarray, K: np.ndarray, T_lidar_to_cam: np.ndarray,
                            image_shape: Tuple[int, int], margin_px: int = 120,
                            min_depth: float = 1.0, max_depth: float = 80.0) -> np.ndarray:
    h, w = image_shape[:2]
    uv, depth, valid_depth = project_points(points_lidar, K, T_lidar_to_cam)
    u = uv[:, 0]
    v = uv[:, 1]
    valid = valid_depth
    valid &= (depth >= min_depth) & (depth <= max_depth)
    valid &= (u >= -margin_px) & (u < w + margin_px)
    valid &= (v >= -margin_px) & (v < h + margin_px)
    return points_lidar[valid]


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.10) -> np.ndarray:
    if len(points) == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points, dtype=np.float32)


def remove_statistical_outliers(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    if len(points) == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd.points, dtype=np.float32)


def choose_ground_plane(candidate_points: np.ndarray, plane_model: np.ndarray,
                        min_ground_inliers: int = 800,
                        max_ground_abs_height: float = 2.5,
                        min_up_dot: float = 0.85) -> bool:
    if len(candidate_points) < min_ground_inliers:
        return False
    a, b, c, _ = plane_model.astype(np.float64)
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        return False
    n = n / n_norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_dot = abs(float(np.dot(n, z_axis)))
    if up_dot < min_up_dot:
        return False
    z_vals = candidate_points[:, 2]
    if np.median(np.abs(z_vals)) > max_ground_abs_height:
        return False
    return True


def remove_ground_only(points: np.ndarray,
                       distance_threshold: float = 0.15,
                       ransac_n: int = 3,
                       num_iterations: int = 1000,
                       min_ground_inliers: int = 800,
                       max_ground_abs_height: float = 2.5,
                       min_up_dot: float = 0.85) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if len(points) < max(ransac_n, min_ground_inliers):
        return np.empty((0, 3), dtype=np.float32), points, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

    if len(inliers) == 0:
        return np.empty((0, 3), dtype=np.float32), points, None

    inliers = np.array(inliers, dtype=np.int64)
    ground_points = points[inliers]
    mask = np.ones(len(points), dtype=bool)
    mask[inliers] = False
    non_ground_points = points[mask]

    ok = choose_ground_plane(ground_points, np.asarray(plane_model, dtype=np.float64),
                             min_ground_inliers=min_ground_inliers,
                             max_ground_abs_height=max_ground_abs_height,
                             min_up_dot=min_up_dot)
    if not ok:
        return np.empty((0, 3), dtype=np.float32), points, None

    return ground_points, non_ground_points, np.asarray(plane_model, dtype=np.float64)


def euclidean_clustering(points: np.ndarray,
                         tolerance: float = 0.8,
                         min_cluster_size: int = 30,
                         max_cluster_size: int = 30000) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    clusters: List[np.ndarray] = []

    for i in range(len(points)):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        cluster_idx: List[int] = []

        while queue:
            idx = queue.pop()
            cluster_idx.append(idx)
            neighbors = tree.query_ball_point(points[idx], r=tolerance)
            for n_idx in neighbors:
                if not visited[n_idx]:
                    visited[n_idx] = True
                    queue.append(n_idx)

        if min_cluster_size <= len(cluster_idx) <= max_cluster_size:
            clusters.append(points[np.array(cluster_idx)])

    return clusters


def cluster_extent(points: np.ndarray) -> Tuple[float, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ext = maxs - mins
    return float(ext[0]), float(ext[1]), float(ext[2])


def filter_object_clusters(clusters: List[np.ndarray], min_extent: float = 0.20,
                           max_extent: float = 8.0, max_flatness_ratio: float = 20.0) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for c in clusters:
        ex, ey, ez = cluster_extent(c)
        ext = np.array([ex, ey, ez], dtype=np.float64)
        max_e = ext.max()
        min_e = max(ext.min(), 1e-6)
        if max_e < min_extent:
            continue
        if max_e > max_extent:
            continue
        if max_e / min_e > max_flatness_ratio:
            continue
        out.append(c)
    return out


# =========================================================
# Cluster projection mask building
# =========================================================
def render_projected_points_mask(uv_valid: np.ndarray, image_shape: Tuple[int, int], splat_radius: int = 2) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if uv_valid.shape[0] == 0:
        return mask

    pts = np.round(uv_valid).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    for px, py in pts:
        cv2.circle(mask, (int(px), int(py)), splat_radius, 255, thickness=-1)
    return mask


def render_convex_hull_mask(uv_valid: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if uv_valid.shape[0] < 3:
        return mask
    pts = np.round(uv_valid).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def postprocess_projected_mask(raw_mask: np.ndarray, close_kernel: int = 5,
                               close_iter: int = 0, dilate_kernel: int = 3,
                               dilate_iter: int = 0) -> np.ndarray:
    mask = raw_mask.copy()
    if dilate_kernel > 1 and dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    if close_kernel > 1 and close_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    mask = fill_holes(mask)
    mask = keep_largest_component(mask)
    return mask.astype(np.uint8)


def render_alpha_shape_mask(uv_valid: np.ndarray, image_shape: Tuple[int, int],
                            alpha_radius: float = 20.0,
                            min_valid_points: int = 8,
                            fallback_to_convex: bool = True) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if uv_valid.shape[0] < min_valid_points:
        return mask

    pts = np.round(uv_valid).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts = np.unique(pts, axis=0)

    if pts.shape[0] < 4:
        return render_convex_hull_mask(pts.astype(np.float32), image_shape) if fallback_to_convex else mask

    pts_f = pts.astype(np.float64)
    try:
        tri = Delaunay(pts_f)
    except QhullError:
        return render_convex_hull_mask(pts.astype(np.float32), image_shape) if fallback_to_convex else mask

    kept_triangles = []
    for simplex in tri.simplices:
        p1, p2, p3 = pts_f[simplex[0]], pts_f[simplex[1]], pts_f[simplex[2]]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-8:
            continue
        area = math.sqrt(area_sq)
        circum_r = (a * b * c) / (4.0 * area)
        if circum_r <= alpha_radius:
            kept_triangles.append(pts[simplex].reshape(-1, 1, 2))

    if len(kept_triangles) == 0:
        return render_convex_hull_mask(pts.astype(np.float32), image_shape) if fallback_to_convex else mask

    for tri_pts in kept_triangles:
        cv2.fillConvexPoly(mask, tri_pts, 255)
    return mask


def compute_bbox_fill_ratio(mask: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    area = mask_area(mask)
    bbox_area_val = max(1, x2 - x1 + 1) * max(1, y2 - y1 + 1)
    return float(area) / float(bbox_area_val)


def build_projected_cluster_mask(cluster_id: int, cluster_points: np.ndarray,
                                 image_shape: Tuple[int, int], K: np.ndarray,
                                 T_lidar_to_cam: np.ndarray, splat_radius: int = 2,
                                 min_valid_points: int = 20,
                                 alpha_radius: float = 20.0) -> Optional[ProjectedClusterMask]:
    uv, depth, valid_depth = project_points(cluster_points, K, T_lidar_to_cam)
    h, w = image_shape
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    valid = valid_depth & in_bounds
    uv_valid = uv[valid]
    depth_valid = depth[valid]

    if uv_valid.shape[0] < min_valid_points:
        return None

    raw_mask = render_projected_points_mask(uv_valid, image_shape, splat_radius=splat_radius)
    hull_mask = render_alpha_shape_mask(uv_valid, image_shape, alpha_radius=alpha_radius,
                                        min_valid_points=min_valid_points, fallback_to_convex=True)
    binary_mask = postprocess_projected_mask(hull_mask, close_kernel=5, close_iter=0, dilate_kernel=3, dilate_iter=0)
    bbox = mask_bbox(binary_mask)
    if bbox is None:
        return None
    boundary_mask = make_boundary_band(binary_mask, band_width=3)
    fill_ratio = compute_bbox_fill_ratio(binary_mask, bbox)
    return ProjectedClusterMask(
        cluster_id=cluster_id,
        points_3d=cluster_points,
        points_2d=uv_valid,
        depths=depth_valid,
        valid_mask=valid,
        raw_point_mask=raw_mask,
        binary_mask=binary_mask,
        boundary_mask=boundary_mask,
        bbox_xyxy=bbox,
        bbox_fill_ratio=fill_ratio,
    )


def build_all_projected_cluster_masks(object_clusters: List[np.ndarray], image_shape: Tuple[int, int],
                                      K: np.ndarray, T_lidar_to_cam: np.ndarray,
                                      splat_radius: int = 2, min_valid_points: int = 20,
                                      alpha_radius: float = 20.0) -> List[ProjectedClusterMask]:
    results: List[ProjectedClusterMask] = []
    for cid, cluster_points in enumerate(object_clusters):
        proj = build_projected_cluster_mask(
            cluster_id=cid,
            cluster_points=cluster_points,
            image_shape=image_shape,
            K=K,
            T_lidar_to_cam=T_lidar_to_cam,
            splat_radius=splat_radius,
            min_valid_points=min_valid_points,
            alpha_radius=alpha_radius,
        )
        if proj is not None:
            results.append(proj)
    return results


def border_touch_ratio(mask: np.ndarray) -> float:
    m = mask > 0
    h, w = m.shape
    border = np.zeros_like(m, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border_pixels = np.logical_and(m, border).sum()
    area = m.sum()
    return 1.0 if area == 0 else float(border_pixels) / float(area)


def filter_projected_clusters(projected_clusters: List[ProjectedClusterMask], image_shape: Tuple[int, int],
                              min_mask_area: int = 500, max_area_ratio: float = 0.18,
                              max_border_touch_ratio: float = 0.12, min_points_2d: int = 20,
                              max_aspect_ratio: float = 8.0) -> List[ProjectedClusterMask]:
    h, w = image_shape
    image_area = h * w
    out: List[ProjectedClusterMask] = []
    for proj in projected_clusters:
        area = mask_area(proj.binary_mask)
        if area < min_mask_area:
            continue
        if area / float(image_area) > max_area_ratio:
            continue
        if proj.points_2d.shape[0] < min_points_2d:
            continue
        if border_touch_ratio(proj.binary_mask) > max_border_touch_ratio:
            continue
        bbox = proj.bbox_xyxy
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bw = x2 - x1 + 1
            bh = y2 - y1 + 1
            aspect = max(bw / max(bh, 1), bh / max(bw, 1))
            if aspect > max_aspect_ratio:
                continue
        out.append(proj)
    return out


# =========================================================
# SAM filtering / group construction
# =========================================================
def filter_sam_masks_basic(sam_masks: List[dict], image_shape: Tuple[int, int],
                           min_area: int = 200, max_area_ratio: float = 0.35) -> List[dict]:
    h, w = image_shape
    image_area = h * w
    out: List[dict] = []
    for ann in sam_masks:
        seg = ann["segmentation"] > 0
        area = int(seg.sum())
        if area < min_area:
            continue
        if area / float(image_area) > max_area_ratio:
            continue
        out.append(ann)
    return out


def union_mask_from_ids(sam_masks: List[dict], mask_ids: Tuple[int, ...], close_kernel: int = 7, close_iter: int = 1) -> np.ndarray:
    seg = np.zeros_like(sam_masks[0]["segmentation"], dtype=np.uint8)
    for mid in mask_ids:
        seg = np.logical_or(seg > 0, sam_masks[mid]["segmentation"] > 0).astype(np.uint8) * 255
    if len(mask_ids) > 1 and close_kernel > 1 and close_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return seg


def build_mask_adjacency(sam_masks: List[dict], image_shape: Tuple[int, int], near_margin: int = 20,
                         min_bbox_iou: float = 0.0, max_centroid_dist: float = 150.0) -> Dict[int, List[int]]:
    n = len(sam_masks)
    bboxes: List[Optional[Tuple[int, int, int, int]]] = []
    centroids: List[Optional[Tuple[float, float]]] = []
    for ann in sam_masks:
        m = (ann["segmentation"] > 0).astype(np.uint8) * 255
        bboxes.append(mask_bbox(m))
        centroids.append(mask_centroid(m))

    graph: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        if bboxes[i] is None:
            continue
        b1e = expand_bbox(bboxes[i], near_margin, image_shape)
        for j in range(i + 1, n):
            if bboxes[j] is None:
                continue
            biou = bbox_iou(b1e, bboxes[j])
            ci, cj = centroids[i], centroids[j]
            cd = 1e9 if ci is None or cj is None else float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
            if biou > min_bbox_iou or cd <= max_centroid_dist:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def projected_points_bbox(points_2d: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if len(points_2d) == 0:
        return None
    xs = points_2d[:, 0]
    ys = points_2d[:, 1]
    return (
        int(np.floor(xs.min())),
        int(np.floor(ys.min())),
        int(np.ceil(xs.max())),
        int(np.ceil(ys.max())),
    )


def render_projected_support_mask(points_2d: np.ndarray, image_shape: Tuple[int, int],
                                  splat_radius: int = 3) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points_2d) == 0:
        return mask

    pts = np.round(points_2d).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    for x, y in pts:
        cv2.circle(mask, (int(x), int(y)), splat_radius, 255, thickness=-1)
    return mask


def build_sam_mask_feature_cache(sam_masks: List[dict]) -> Dict[int, SamMaskFeature]:
    feature_map: Dict[int, SamMaskFeature] = {}
    for mid, ann in enumerate(sam_masks):
        mask_bool = ann["segmentation"] > 0
        mask_u8 = mask_bool.astype(np.uint8) * 255
        feature_map[mid] = SamMaskFeature(
            mask_id=mid,
            raw_id=int(ann.get("raw_id", mid)),
            mask_bool=mask_bool,
            mask_u8=mask_u8,
            bbox_xyxy=mask_bbox(mask_u8),
            centroid_xy=mask_centroid(mask_u8),
            area=mask_area(mask_u8),
            sdf=compute_signed_distance_field(mask_u8),
        )
    return feature_map


def union_mask_from_feature_ids(feature_map: Dict[int, SamMaskFeature],
                                mask_ids: Tuple[int, ...]) -> np.ndarray:
    if len(mask_ids) == 0:
        raise ValueError("mask_ids must not be empty")
    acc = np.zeros_like(next(iter(feature_map.values())).mask_u8, dtype=np.uint8)
    for mid in mask_ids:
        acc = np.logical_or(acc > 0, feature_map[mid].mask_bool).astype(np.uint8) * 255
    return acc


def bbox_scale_score_from_bboxes(cluster_bbox: Optional[Tuple[int, int, int, int]],
                                 mask_bbox_: Optional[Tuple[int, int, int, int]]) -> float:
    if cluster_bbox is None or mask_bbox_ is None:
        return 0.0

    c_area = float(bbox_area(cluster_bbox))
    m_area = float(bbox_area(mask_bbox_))
    if c_area <= 0.0 or m_area <= 0.0:
        return 0.0

    area_term = math.exp(-abs(math.log(c_area / m_area)))

    cx1, cy1, cx2, cy2 = cluster_bbox
    mx1, my1, mx2, my2 = mask_bbox_

    c_w = max(1.0, float(cx2 - cx1 + 1))
    c_h = max(1.0, float(cy2 - cy1 + 1))
    m_w = max(1.0, float(mx2 - mx1 + 1))
    m_h = max(1.0, float(my2 - my1 + 1))

    c_ar = c_w / c_h
    m_ar = m_w / m_h
    ar_term = math.exp(-abs(math.log(c_ar / m_ar)))

    return 0.7 * area_term + 0.3 * ar_term


def unsupported_area_ratio(group_mask: np.ndarray, support_mask: np.ndarray) -> float:
    g = group_mask > 0
    s = support_mask > 0
    g_area = int(g.sum())
    if g_area == 0:
        return 1.0
    supported = int(np.logical_and(g, s).sum())
    return float(max(0, g_area - supported)) / float(g_area)


def fragment_penalty_from_mask(mask: np.ndarray) -> float:
    binary = (mask > 0).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
    ncc = max(0, num_labels - 1)
    if ncc <= 1:
        return 0.0
    return min(1.0, float(ncc - 1) / 3.0)


def candidate_mask_ids_for_cluster_point_support(
    mask_feature_map: Dict[int, SamMaskFeature],
    cluster_proj: ProjectedClusterMask,
    image_shape: Tuple[int, int],
    quick_bbox_margin: int = QUICK_BBOX_MARGIN,
    hard_inside_thresh: float = HARD_INSIDE_THRESH,
    soft_support_thresh: float = SOFT_SUPPORT_THRESH,
    soft_sigma_px: float = SOFT_SIGMA_PX,
    top_k: int = TOP_K_CANDIDATES,
) -> Tuple[List[int], Dict[int, Tuple[float, float]]]:
    """
    Returns:
      candidate_ids
      per_mask_support[mid] = (hard_inside_ratio, soft_support)
    """
    uv = cluster_proj.points_2d
    if len(uv) == 0:
        return [], {}

    h, w = image_shape[:2]
    cluster_bbox = cluster_proj.bbox_xyxy
    if cluster_bbox is None:
        cluster_bbox = projected_points_bbox(uv)

    rounded = np.round(uv).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, w - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, h - 1)

    per_mask_support: Dict[int, Tuple[float, float]] = {}
    scored: List[Tuple[int, float]] = []

    for mid, feat in mask_feature_map.items():
        if feat.bbox_xyxy is None:
            continue

        if cluster_bbox is not None:
            cb = expand_bbox(cluster_bbox, quick_bbox_margin, (h, w))
            if bbox_iou(cb, feat.bbox_xyxy) <= 0.0:
                continue

        inside = feat.mask_bool[rounded[:, 1], rounded[:, 0]]
        hard_inside_ratio = float(np.mean(inside.astype(np.float64)))

        sdf_vals = bilinear_sample(feat.sdf, uv, outside_value=float(max(h, w)))
        outside_dist = np.maximum(0.0, sdf_vals)
        soft_support = float(np.mean(np.exp(-(outside_dist ** 2) / (2.0 * soft_sigma_px * soft_sigma_px))))

        if hard_inside_ratio >= hard_inside_thresh or soft_support >= soft_support_thresh:
            per_mask_support[mid] = (hard_inside_ratio, soft_support)
            quick_score = SUPPORT_SCORE_HARD_WEIGHT * hard_inside_ratio + SUPPORT_SCORE_SOFT_WEIGHT * soft_support
            scored.append((mid, quick_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    candidate_ids = [mid for mid, _ in scored[:top_k]]
    per_mask_support = {mid: per_mask_support[mid] for mid in candidate_ids}
    return candidate_ids, per_mask_support


def score_mask_group_for_cluster_support(
    feature_map: Dict[int, SamMaskFeature],
    cluster_proj: ProjectedClusterMask,
    mask_ids: Tuple[int, ...],
    image_shape: Tuple[int, int],
    soft_sigma_px: float = SOFT_SIGMA_PX,
    support_splat_radius: int = SUPPORT_SPLAT_RADIUS,
) -> SupportStats:
    uv = cluster_proj.points_2d
    h, w = image_shape[:2]

    if len(uv) == 0:
        return SupportStats(
            hard_inside_ratio=0.0,
            soft_support=0.0,
            centroid_score=0.0,
            bbox_scale_score=0.0,
            unsupported_ratio=1.0,
            fragment_penalty=1.0,
            oversize_penalty=1.0,
            total_score=-1e9,
        )

    group_mask = union_mask_from_feature_ids(feature_map, mask_ids)
    group_bbox = mask_bbox(group_mask)
    group_cent = mask_centroid(group_mask)
    group_sdf = compute_signed_distance_field(group_mask)

    rounded = np.round(uv).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, w - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, h - 1)

    inside = (group_mask[rounded[:, 1], rounded[:, 0]] > 0)
    hard_inside_ratio = float(np.mean(inside.astype(np.float64)))

    sdf_vals = bilinear_sample(group_sdf, uv, outside_value=float(max(h, w)))
    outside_dist = np.maximum(0.0, sdf_vals)
    soft_support = float(np.mean(np.exp(-(outside_dist ** 2) / (2.0 * soft_sigma_px * soft_sigma_px))))

    pts_cent = mask_centroid(render_projected_support_mask(uv, image_shape, splat_radius=1))
    if pts_cent is None or group_cent is None or group_bbox is None:
        centroid_score_val = 0.0
    else:
        gx1, gy1, gx2, gy2 = group_bbox
        bw = max(20.0, float(gx2 - gx1 + 1))
        bh = max(20.0, float(gy2 - gy1 + 1))
        dx = (pts_cent[0] - group_cent[0]) / bw
        dy = (pts_cent[1] - group_cent[1]) / bh
        centroid_score_val = float(np.exp(-(dx * dx + dy * dy) / 0.08))

    cluster_bbox = cluster_proj.bbox_xyxy if cluster_proj.bbox_xyxy is not None else projected_points_bbox(uv)
    bbox_scale_score_val = bbox_scale_score_from_bboxes(cluster_bbox, group_bbox)

    support_mask = render_projected_support_mask(uv, image_shape, splat_radius=support_splat_radius)
    unsupported_ratio_val = unsupported_area_ratio(group_mask, support_mask)

    fragment_penalty_val = fragment_penalty_from_mask(group_mask)

    oversize_penalty_val = 0.0
    if cluster_proj.binary_mask is not None:
        oversize_penalty_val = min(1.5, oversize_penalty(group_mask, cluster_proj.binary_mask))

    total = (
        0.42 * hard_inside_ratio +
        0.28 * soft_support +
        0.12 * centroid_score_val +
        0.08 * bbox_scale_score_val -
        0.15 * unsupported_ratio_val -
        0.05 * fragment_penalty_val -
        0.08 * oversize_penalty_val
    )

    return SupportStats(
        hard_inside_ratio=hard_inside_ratio,
        soft_support=soft_support,
        centroid_score=centroid_score_val,
        bbox_scale_score=bbox_scale_score_val,
        unsupported_ratio=unsupported_ratio_val,
        fragment_penalty=fragment_penalty_val,
        oversize_penalty=oversize_penalty_val,
        total_score=float(total),
    )


def propose_mask_groups_for_one_cluster_support(
    feature_map: Dict[int, SamMaskFeature],
    mask_graph: Dict[int, List[int]],
    cluster_proj: ProjectedClusterMask,
    image_shape: Tuple[int, int],
    candidate_top_k: int = TOP_K_CANDIDATES,
    beam_width: int = BEAM_WIDTH,
    max_group_size: int = MAX_GROUP_SIZE,
    min_group_improve: float = 0.015,
    max_unsupported_ratio: float = 0.72,
    neighbor_hard_thresh: float = 0.03,
    neighbor_soft_thresh: float = 0.18,
) -> List[GroupProposal]:
    candidate_ids, per_mask_support = candidate_mask_ids_for_cluster_point_support(
        mask_feature_map=feature_map,
        cluster_proj=cluster_proj,
        image_shape=image_shape,
        quick_bbox_margin=QUICK_BBOX_MARGIN,
        hard_inside_thresh=HARD_INSIDE_THRESH,
        soft_support_thresh=SOFT_SUPPORT_THRESH,
        soft_sigma_px=SOFT_SIGMA_PX,
        top_k=candidate_top_k,
    )

    if len(candidate_ids) == 0:
        return []

    @lru_cache(maxsize=4096)
    def eval_group(mask_ids: Tuple[int, ...]) -> SupportStats:
        key = tuple(sorted(mask_ids))
        return score_mask_group_for_cluster_support(
            feature_map=feature_map,
            cluster_proj=cluster_proj,
            mask_ids=key,
            image_shape=image_shape,
            soft_sigma_px=SOFT_SIGMA_PX,
            support_splat_radius=SUPPORT_SPLAT_RADIUS,
        )

    singles: List[GroupProposal] = []
    for mid in candidate_ids:
        s = eval_group((mid,))
        if s.total_score <= 0.0:
            continue
        singles.append(GroupProposal(cluster_id=cluster_proj.cluster_id, mask_ids=(mid,), support=s))

    if len(singles) == 0:
        return []

    singles.sort(key=lambda p: p.support.total_score, reverse=True)
    all_proposals: Dict[Tuple[int, ...], GroupProposal] = {p.mask_ids: p for p in singles}
    beam: List[GroupProposal] = singles[:beam_width]

    for _group_size in range(2, max_group_size + 1):
        next_candidates: Dict[Tuple[int, ...], GroupProposal] = {}

        for base in beam:
            frontier: set[int] = set()
            for mid in base.mask_ids:
                frontier.update(mask_graph.get(mid, []))

            frontier = {
                mid for mid in frontier
                if mid in candidate_ids and mid not in base.mask_ids
            }

            for next_mid in frontier:
                h_in, s_soft = per_mask_support.get(next_mid, (0.0, 0.0))
                if h_in < neighbor_hard_thresh and s_soft < neighbor_soft_thresh:
                    continue

                new_ids = tuple(sorted(base.mask_ids + (next_mid,)))
                if new_ids in next_candidates:
                    continue

                new_support = eval_group(new_ids)

                if new_support.total_score < base.support.total_score + min_group_improve:
                    continue
                if new_support.unsupported_ratio > max_unsupported_ratio:
                    continue

                next_candidates[new_ids] = GroupProposal(
                    cluster_id=cluster_proj.cluster_id,
                    mask_ids=new_ids,
                    support=new_support,
                )

        if len(next_candidates) == 0:
            break

        next_beam = list(next_candidates.values())
        next_beam.sort(key=lambda p: p.support.total_score, reverse=True)
        beam = next_beam[:beam_width]

        for p in beam:
            if p.mask_ids not in all_proposals:
                all_proposals[p.mask_ids] = p
            elif p.support.total_score > all_proposals[p.mask_ids].support.total_score:
                all_proposals[p.mask_ids] = p

    proposals = list(all_proposals.values())
    proposals.sort(key=lambda p: p.support.total_score, reverse=True)

    if len(proposals) >= 2:
        best = proposals[0].support.total_score
        second = proposals[1].support.total_score
        proposals[0].confidence_gap = best - second
    elif len(proposals) == 1:
        proposals[0].confidence_gap = proposals[0].support.total_score

    return proposals


def select_group_proposals_globally_conflict_aware(
    proposals_per_cluster: Dict[int, List[GroupProposal]],
    per_cluster_keep: int = 3,
    min_total_score: float = 0.10,
    min_conf_gap: float = 0.02,
) -> List[GroupProposal]:
    """
    Exact Hungarian is not valid here because proposals are mask sets and can overlap.
    This is a conflict-aware greedy selector:
      - one proposal per cluster
      - no mask can be reused across selected proposals
    """
    pool: List[GroupProposal] = []
    for cluster_id, plist in proposals_per_cluster.items():
        plist = sorted(plist, key=lambda p: p.support.total_score, reverse=True)
        if len(plist) == 0:
            continue

        if len(plist) >= 2:
            best = plist[0].support.total_score
            second = plist[1].support.total_score
            plist[0].confidence_gap = best - second
        else:
            plist[0].confidence_gap = plist[0].support.total_score

        pool.extend(plist[:per_cluster_keep])

    def priority(p: GroupProposal) -> float:
        return p.support.total_score + 0.25 * p.confidence_gap

    pool = [p for p in pool if p.support.total_score >= min_total_score]
    pool.sort(key=priority, reverse=True)

    selected: List[GroupProposal] = []
    used_clusters: set[int] = set()
    used_masks: set[int] = set()

    for p in pool:
        if p.cluster_id in used_clusters:
            continue
        if p.confidence_gap < min_conf_gap:
            continue
        if any(mid in used_masks for mid in p.mask_ids):
            continue

        selected.append(p)
        used_clusters.add(p.cluster_id)
        used_masks.update(p.mask_ids)

    selected.sort(key=lambda p: p.support.total_score, reverse=True)
    return selected


def group_proposal_to_match_result(
    proposal: GroupProposal,
    feature_map: Dict[int, SamMaskFeature],
    cluster_proj_map: Dict[int, ProjectedClusterMask],
) -> Optional[MatchResult]:
    proj = cluster_proj_map.get(proposal.cluster_id)
    if proj is None:
        return None

    group_mask = union_mask_from_feature_ids(feature_map, proposal.mask_ids)
    bbox1 = mask_bbox(group_mask)
    bbox2 = proj.bbox_xyxy
    if bbox1 is None or bbox2 is None:
        return None

    iou_val = mask_iou(group_mask, proj.binary_mask)
    bi_val = boundary_iou(group_mask, proj.binary_mask, band_width=3)
    contain_val = mask_containment(proj.binary_mask, group_mask)
    bbox_iou_val = bbox_iou(bbox1, bbox2)
    area_ratio_val = area_ratio(group_mask, proj.binary_mask)

    return MatchResult(
        cluster_id=proposal.cluster_id,
        mask_ids=proposal.mask_ids,
        selection_score=proposal.support.total_score,
        iou=iou_val,
        boundary_iou=bi_val,
        containment=contain_val,
        bbox_iou=bbox_iou_val,
        area_ratio=area_ratio_val,
        centroid_score=proposal.support.centroid_score,
    )


def find_best_mask_groups_for_clusters_support_global(
    sam_masks: List[dict],
    projected_clusters: List[ProjectedClusterMask],
    image_shape: Tuple[int, int],
    debug_progress: bool = False,
) -> List[MatchResult]:
    feature_map = build_sam_mask_feature_cache(sam_masks)
    mask_graph = build_mask_adjacency(
        sam_masks=sam_masks,
        image_shape=image_shape,
        near_margin=20,
        min_bbox_iou=0.0,
        max_centroid_dist=150.0,
    )

    proposals_per_cluster: Dict[int, List[GroupProposal]] = {}

    total = len(projected_clusters)
    for idx, proj in enumerate(projected_clusters):
        debug(debug_progress, f"support_group_search {idx + 1}/{total} cluster_id={proj.cluster_id}")
        proposals = propose_mask_groups_for_one_cluster_support(
            feature_map=feature_map,
            mask_graph=mask_graph,
            cluster_proj=proj,
            image_shape=image_shape,
            candidate_top_k=20,
            beam_width=6,
            max_group_size=3,
            min_group_improve=0.015,
            max_unsupported_ratio=0.72,
            neighbor_hard_thresh=0.03,
            neighbor_soft_thresh=0.18,
        )
        if len(proposals) > 0:
            proposals_per_cluster[proj.cluster_id] = proposals

    selected = select_group_proposals_globally_conflict_aware(
        proposals_per_cluster=proposals_per_cluster,
        per_cluster_keep=3,
        min_total_score=0.10,
        min_conf_gap=0.02,
    )

    cluster_proj_map = {p.cluster_id: p for p in projected_clusters}
    matches: List[MatchResult] = []
    for p in selected:
        m = group_proposal_to_match_result(
            proposal=p,
            feature_map=feature_map,
            cluster_proj_map=cluster_proj_map,
        )
        if m is not None:
            matches.append(m)

    matches.sort(key=lambda x: x.selection_score, reverse=True)
    return matches
def is_background_like_mask_by_size(mask: np.ndarray, image_shape: Tuple[int, int], max_area_ratio: float = 0.22) -> bool:
    h, w = image_shape
    area_ratio_val = mask_area(mask) / float(h * w)
    return area_ratio_val > max_area_ratio


def is_background_like_cluster_by_size(cluster: ProjectedClusterMask, image_shape: Tuple[int, int], max_area_ratio: float = 0.18) -> bool:
    h, w = image_shape
    area_ratio_val = mask_area(cluster.binary_mask) / float(h * w)
    return area_ratio_val > max_area_ratio


def filter_background_like_group_matches(best_matches: List[MatchResult], sam_masks: List[dict],
                                         projected_clusters: List[ProjectedClusterMask], image_shape: Tuple[int, int],
                                         max_mask_area_ratio: float = 0.20,
                                         max_cluster_area_ratio: float = 0.30,
                                         min_score: float = 0.10) -> List[MatchResult]:
    cluster_id_to_proj = {p.cluster_id: p for p in projected_clusters}
    filtered: List[MatchResult] = []
    for match in best_matches:
        proj = cluster_id_to_proj.get(match.cluster_id)
        if proj is None:
            continue
        union = union_mask_from_ids(sam_masks, match.mask_ids)
        if is_background_like_mask_by_size(union, image_shape=image_shape, max_area_ratio=max_mask_area_ratio):
            continue
        if is_background_like_cluster_by_size(proj, image_shape=image_shape, max_area_ratio=max_cluster_area_ratio):
            continue
        if match.selection_score < min_score:
            continue
        filtered.append(match)
    filtered.sort(key=lambda x: x.selection_score, reverse=True)
    return filtered


# =========================================================
# Objective and optimization
# =========================================================
def build_continuous_fixed_pairs(matches: List[MatchResult],
                                 projected_clusters: List[ProjectedClusterMask],
                                 object_clusters: List[np.ndarray],
                                 sam_masks: List[dict],
                                 image_shape: Tuple[int, int],
                                 top_k: int = 5,
                                 sample_points_per_cluster: int = 128) -> List[ContinuousFixedPair]:
    cluster_id_to_points = {i: c for i, c in enumerate(object_clusters)}
    selected = matches[:top_k]
    if len(selected) == 0:
        return []

    total_score = sum(max(1e-6, m.selection_score) for m in selected)
    h, w = image_shape[:2]
    image_diag = float(np.hypot(w, h))

    pairs: List[ContinuousFixedPair] = []
    for m in selected:
        cluster_points = cluster_id_to_points.get(m.cluster_id)
        if cluster_points is None or len(cluster_points) == 0:
            continue

        sampled_points = farthest_point_sampling(cluster_points, sample_points_per_cluster)

        group_mask = union_mask_from_ids(sam_masks, m.mask_ids)
        mu, cov = compute_mask_centroid_and_cov(group_mask)
        sdf = compute_signed_distance_field(group_mask)

        bbox = mask_bbox(group_mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(20.0, float(x2 - x1 + 1))
        bh = max(20.0, float(y2 - y1 + 1))
        norm_xy = np.array([bw, bh], dtype=np.float64)

        pairs.append(
            ContinuousFixedPair(
                cluster_id=m.cluster_id,
                cluster_points_sampled=sampled_points.astype(np.float32),
                weight=max(1e-6, m.selection_score) / total_score,
                selection_score=float(m.selection_score),
                mask_ids=m.mask_ids,
                group_mask=group_mask.astype(np.uint8),
                sdf=sdf,
                mask_centroid_xy=mu,
                mask_cov_xy=cov,
                norm_xy=norm_xy,
                outside_sdf_value=0.5 * image_diag,
            )
        )
    return pairs


def compute_mask_centroid_and_cov(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(2, dtype=np.float64), np.eye(2, dtype=np.float64)

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mu = pts.mean(axis=0)

    xc = pts - mu
    if len(pts) >= 2:
        cov = (xc.T @ xc) / float(len(pts))
    else:
        cov = np.eye(2, dtype=np.float64)

    cov += 1e-6 * np.eye(2, dtype=np.float64)
    return mu, cov

def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    z = beta * x
    return np.where(z > 20.0, x, np.log1p(np.exp(z)) / beta)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    """
    Simple O(Nk) FPS. Deterministic.
    points: (N,3)
    returns sampled points: (min(N,k),3)
    """
    n = len(points)
    if n == 0:
        return points
    if n <= k:
        return points.copy()

    pts = points.astype(np.float64)
    selected = np.zeros((k,), dtype=np.int64)
    dist2 = np.full((n,), np.inf, dtype=np.float64)

    # start from centroid-nearest point
    centroid = pts.mean(axis=0)
    start = np.argmin(np.sum((pts - centroid) ** 2, axis=1))
    selected[0] = start

    cur = pts[start]
    dist2 = np.minimum(dist2, np.sum((pts - cur) ** 2, axis=1))

    for i in range(1, k):
        idx = np.argmax(dist2)
        selected[i] = idx
        cur = pts[idx]
        dist2 = np.minimum(dist2, np.sum((pts - cur) ** 2, axis=1))

    return points[selected]

def compute_projected_uv_stats(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    uv: (N,2)
    returns mean (2,), cov (2,2)
    """
    if len(uv) == 0:
        return np.zeros(2, dtype=np.float64), np.eye(2, dtype=np.float64)

    mu = uv.mean(axis=0)
    xc = uv - mu
    if len(uv) >= 2:
        cov = (xc.T @ xc) / float(len(uv))
    else:
        cov = np.eye(2, dtype=np.float64)
    cov += 1e-6 * np.eye(2, dtype=np.float64)
    return mu, cov

def residuals_for_delta_continuous(
    delta: np.ndarray,
    fixed_pairs: List[ContinuousFixedPair],
    K: np.ndarray,
    T_current: np.ndarray,
    image_shape: Tuple[int, int],
    trans_reg: float = 0.05,
    rot_reg: float = 0.05,
    trans_bound: float = 0.5,
    rot_bound_deg: float = 5.0,

    lambda_occ: float = 1.0,
    lambda_cent: float = 0.25,
    lambda_cov: float = 0.10,
    lambda_inside_ratio: float = 0.15,

    tau_occ_px: float = 2.0,
    tau_inside_px: float = 2.0,
    target_inside_ratio: float = 0.85,
) -> np.ndarray:
    """
    Smooth residuals for LM:
      1) occupancy residual via signed distance field
      2) centroid residual
      3) covariance residual
      4) inside-ratio residual
      + regularization / soft bounds
    """
    T_eval = apply_delta_pose(T_current, delta)
    h, w = image_shape[:2]

    residuals: List[float] = []

    for pair in fixed_pairs:
        pts3d = pair.cluster_points_sampled
        uv, depth, valid_depth = project_points(pts3d, K, T_eval)

        in_bounds = (
            (uv[:, 0] >= 0.0) & (uv[:, 0] <= w - 1.0) &
            (uv[:, 1] >= 0.0) & (uv[:, 1] <= h - 1.0)
        )
        valid = valid_depth & in_bounds

        # SDF sampling for all points (fixed residual size)
        sdf_vals = bilinear_sample(pair.sdf, uv, outside_value=pair.outside_sdf_value)

        # Invalid points => strong outside penalty
        sdf_vals = np.where(valid, sdf_vals, pair.outside_sdf_value)

        # 1) occupancy residual:
        #    zero-ish when point is inside mask (sdf < 0),
        #    increases smoothly when outside (sdf > 0).
        occ = softplus(sdf_vals / tau_occ_px)
        occ = occ / max(1.0, tau_occ_px)

        occ_weight = math.sqrt(pair.weight * lambda_occ / float(len(occ)))
        residuals.extend((occ_weight * occ).tolist())

        # valid projected set for moment statistics
        uv_valid = uv[valid]
        if len(uv_valid) < 6:
            # if almost everything invalid, add fixed penalties
            bad = math.sqrt(pair.weight)
            residuals.extend([2.0 * bad, 2.0 * bad])           # centroid
            residuals.extend([1.5 * bad, 1.5 * bad, 1.5 * bad])  # covariance
            residuals.append(1.5 * bad)                        # inside ratio
            continue

        # 2) centroid residual
        mu_proj, cov_proj = compute_projected_uv_stats(uv_valid)
        r_cent = (mu_proj - pair.mask_centroid_xy) / pair.norm_xy
        residuals.extend((math.sqrt(pair.weight * lambda_cent) * r_cent).tolist())

        # 3) covariance residual
        norm_x = pair.norm_xy[0]
        norm_y = pair.norm_xy[1]

        r_cov = np.array([
            (cov_proj[0, 0] - pair.mask_cov_xy[0, 0]) / (norm_x * norm_x),
            (cov_proj[0, 1] - pair.mask_cov_xy[0, 1]) / (norm_x * norm_y),
            (cov_proj[1, 1] - pair.mask_cov_xy[1, 1]) / (norm_y * norm_y),
        ], dtype=np.float64)
        residuals.extend((math.sqrt(pair.weight * lambda_cov) * r_cov).tolist())

        # 4) soft inside ratio residual
        inside_prob = sigmoid(-sdf_vals / tau_inside_px)
        inside_ratio = float(np.mean(inside_prob))
        ratio_gap = target_inside_ratio - inside_ratio
        r_ratio = softplus(np.array([ratio_gap * 10.0], dtype=np.float64))[0] / 10.0
        residuals.append(math.sqrt(pair.weight * lambda_inside_ratio) * r_ratio)

    # regularization
    residuals.extend((math.sqrt(trans_reg) * delta[:3]).tolist())
    residuals.extend((math.sqrt(rot_reg) * delta[3:]).tolist())

    # soft bounds
    dt = np.linalg.norm(delta[:3])
    dr = np.linalg.norm(delta[3:])
    rot_bound = math.radians(rot_bound_deg)

    residuals.append(math.sqrt(10.0) * max(0.0, dt - trans_bound))
    residuals.append(math.sqrt(10.0) * max(0.0, dr - rot_bound))

    return np.asarray(residuals, dtype=np.float64)

def optimize_pose_for_continuous_fixed_pairs(
    fixed_pairs: List[ContinuousFixedPair],
    K: np.ndarray,
    T_current: np.ndarray,
    image_shape: Tuple[int, int],
    debug_progress: bool = False,
    lm_max_nfev: int = 80,
    trans_reg: float = 0.05,
    rot_reg: float = 0.05,
    trans_bound: float = 0.5,
    rot_bound_deg: float = 5.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if len(fixed_pairs) == 0:
        return T_current.copy(), {"success": False, "fun": float("inf"), "cost": float("inf"), "nit": 0, "nfev": 0}

    x0 = np.zeros(6, dtype=np.float64)

    result = least_squares(
        residuals_for_delta_continuous,
        x0=x0,
        args=(
            fixed_pairs,
            K,
            T_current,
            image_shape,
            trans_reg,
            rot_reg,
            trans_bound,
            rot_bound_deg,
        ),
        method="lm",
        x_scale="jac",
        max_nfev=lm_max_nfev,
        verbose=2 if debug_progress else 0,
    )

    T_new = apply_delta_pose(T_current, result.x.astype(np.float64))
    info_dict = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.cost),
        "cost": float(result.cost),
        "nfev": int(getattr(result, "nfev", -1)),
        "delta_tx": float(result.x[0]),
        "delta_ty": float(result.x[1]),
        "delta_tz": float(result.x[2]),
        "delta_rx": float(result.x[3]),
        "delta_ry": float(result.x[4]),
        "delta_rz": float(result.x[5]),
    }
    return T_new, info_dict

# =========================================================
# Visualization / outputs
# =========================================================
def draw_projected_cluster_masks_overlay(image_bgr: np.ndarray, projected_clusters: List[ProjectedClusterMask],
                                         alpha: float = 0.30, seed: int = 7,
                                         draw_bbox: bool = True, draw_label: bool = True) -> np.ndarray:
    vis = image_bgr.copy().astype(np.float32)
    rng = np.random.default_rng(seed)
    for proj in projected_clusters:
        color = rng.integers(50, 255, size=3).tolist()
        color_arr = np.array(color, dtype=np.float32)
        mask = proj.binary_mask > 0
        vis[mask] = (1.0 - alpha) * vis[mask] + alpha * color_arr
        contours, _ = cv2.findContours(proj.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
        if draw_bbox and proj.bbox_xyxy is not None:
            x1, y1, x2, y2 = proj.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
            if draw_label:
                cv2.putText(vis, f"cluster:{proj.cluster_id}", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return np.clip(vis, 0, 255).astype(np.uint8)


def draw_projected_pointcloud_overlay(
    image_bgr: np.ndarray,
    points_lidar: np.ndarray,
    K: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    point_radius: int = POINTCLOUD_POINT_RADIUS,
    max_points: int = POINTCLOUD_MAX_POINTS,
) -> np.ndarray:
    vis = image_bgr.copy().astype(np.float32)
    uv, depth, valid_depth = project_points(points_lidar, K, T_lidar_to_cam)
    h, w = image_bgr.shape[:2]
    in_bounds = (
        (uv[:, 0] >= 0.0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0.0) & (uv[:, 1] < h)
    )
    valid = valid_depth & in_bounds
    if not np.any(valid):
        return image_bgr.copy()

    uv_valid = uv[valid]
    depth_valid = depth[valid]

    if len(uv_valid) > max_points:
        idx = np.linspace(0, len(uv_valid) - 1, max_points).astype(np.int64)
        uv_valid = uv_valid[idx]
        depth_valid = depth_valid[idx]

    depth_min = float(np.percentile(depth_valid, 5.0))
    depth_max = float(np.percentile(depth_valid, 95.0))
    if depth_max <= depth_min + 1e-6:
        depth_max = depth_min + 1.0

    norm = np.clip((depth_valid - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    colors = cv2.applyColorMap((255.0 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)

    pts = np.round(uv_valid).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    for (x, y), color in zip(pts, colors):
        cv2.circle(vis, (int(x), int(y)), point_radius, tuple(int(c) for c in color[0]), thickness=-1)

    # help the projection read as a point cloud rather than speckle noise
    vis = cv2.addWeighted(vis, POINTCLOUD_BLEND_RATIO, image_bgr.astype(np.float32), 1.0 - POINTCLOUD_BLEND_RATIO, 0.0)
    return np.clip(vis, 0, 255).astype(np.uint8)


def overlay_mask_group_matches(image_bgr: np.ndarray, sam_masks: List[dict],
                               projected_clusters: List[ProjectedClusterMask],
                               matches: List[MatchResult], top_k: int = 10,
                               alpha_mask: float = 0.20,
                               alpha_cluster: float = 0.25) -> np.ndarray:
    vis = image_bgr.copy().astype(np.float32)
    rng = np.random.default_rng(11)
    cluster_id_to_proj = {p.cluster_id: p for p in projected_clusters}

    for rank, match in enumerate(matches[:top_k]):
        color = rng.integers(0, 255, size=3).tolist()
        color_arr = np.array(color, dtype=np.float32)
        union = union_mask_from_ids(sam_masks, match.mask_ids)
        union_bool = union > 0
        vis[union_bool] = (1 - alpha_mask) * vis[union_bool] + alpha_mask * color_arr
        contours_union, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_union, -1, (255, 255, 255), 1)

        proj = cluster_id_to_proj.get(match.cluster_id)
        if proj is None:
            continue
        cmask = proj.binary_mask > 0
        vis[cmask] = (1 - alpha_cluster) * vis[cmask] + alpha_cluster * color_arr
        contours_cluster, _ = cv2.findContours(proj.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_cluster, -1, tuple(int(c) for c in color), 2)

        if proj.bbox_xyxy is not None:
            x1, y1, x2, y2 = proj.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), tuple(int(c) for c in color), 1)
            raw_ids = [int(sam_masks[mid].get("raw_id", mid)) for mid in match.mask_ids]
            txt = f"r{rank} c{match.cluster_id} m{raw_ids} s={match.selection_score:.2f}"
            cv2.putText(vis, txt, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        tuple(int(c) for c in color), 1, cv2.LINE_AA)

    return np.clip(vis, 0, 255).astype(np.uint8)


def print_group_matches(matches: List[MatchResult], sam_masks: List[dict], top_k: int = 15) -> None:
    print("=" * 130)
    print(f"{'rank':<6} {'cluster':<8} {'raw_ids':<20} {'score':<10} {'IoU':<10} {'B-IoU':<10} {'Contain':<10} {'BBoxIoU':<10} {'AreaRatio':<10}")
    print("-" * 130)
    for rank, m in enumerate(matches[:top_k]):
        raw_ids = [int(sam_masks[mid].get('raw_id', mid)) for mid in m.mask_ids]
        print(f"{rank:<6} {m.cluster_id:<8} {str(raw_ids):<20} {m.selection_score:<10.4f} {m.iou:<10.4f} {m.boundary_iou:<10.4f} {m.containment:<10.4f} {m.bbox_iou:<10.4f} {m.area_ratio:<10.4f}")
    print("=" * 130)


def save_pose_json(path: str, K: np.ndarray, T0: np.ndarray, T_opt: np.ndarray,
                   optimizer_history: List[dict], fixed_pairs: List[ContinuousFixedPair],
                   gt_pose_info: Optional[Dict[str, object]] = None) -> None:
    data = {
        "K": K.tolist(),
        "T_init": T0.tolist(),
        "T_opt": T_opt.tolist(),
        "optimizer_history": optimizer_history,
        "fixed_pairs": [
            {
                "cluster_id": p.cluster_id,
                "mask_ids": list(p.mask_ids),
                "weight": float(p.weight),
                "selection_score": float(p.selection_score),
            }
            for p in fixed_pairs
        ],
    }
    if gt_pose_info is not None:
        data["gt_pose"] = gt_pose_info
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =========================================================
# Calibration pipeline
# =========================================================
def run_initial_preprocessing(image_bgr: np.ndarray, points_raw: np.ndarray, K: np.ndarray, T0: np.ndarray,
                              args: argparse.Namespace) -> List[np.ndarray]:
    points = prefilter_points_by_fov(
        points_raw, K=K, T_lidar_to_cam=T0, image_shape=image_bgr.shape[:2],
        margin_px=args.margin_px, min_depth=args.min_depth, max_depth=args.max_depth,
    )
    info(f"after FOV prefilter: {len(points)}")

    points = voxel_downsample(points, voxel_size=args.voxel_size)
    info(f"after voxel downsample: {len(points)}")

    points = remove_statistical_outliers(points, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    info(f"after outlier removal: {len(points)}")

    ground_points, non_ground_points, plane_model = remove_ground_only(
        points,
        distance_threshold=args.ground_dist_th,
        ransac_n=args.ground_ransac_n,
        num_iterations=args.ground_iters,
        min_ground_inliers=args.min_ground_inliers,
        max_ground_abs_height=args.max_ground_abs_height,
        min_up_dot=args.min_ground_up_dot,
    )
    info(f"ground points: {len(ground_points)}")
    info(f"non-ground points: {len(non_ground_points)}")
    if plane_model is not None:
        info(f"accepted ground plane model: {plane_model.tolist()}")

    object_clusters = euclidean_clustering(
        non_ground_points,
        tolerance=args.cluster_tolerance,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
    )
    info(f"raw object clusters: {len(object_clusters)}")

    object_clusters = filter_object_clusters(
        object_clusters,
        min_extent=args.cluster_min_extent,
        max_extent=args.cluster_max_extent,
        max_flatness_ratio=args.cluster_max_flatness,
    )
    info(f"filtered object clusters: {len(object_clusters)}")
    return object_clusters


def run_pair_selection(image_bgr: np.ndarray, sam_masks: List[dict], object_clusters: List[np.ndarray],
                       K: np.ndarray, T_current: np.ndarray, args: argparse.Namespace,
                       debug: bool = False) -> Tuple[List[ProjectedClusterMask], List[MatchResult]]:
    projected_clusters = build_all_projected_cluster_masks(
        object_clusters=object_clusters,
        image_shape=image_bgr.shape[:2],
        K=K,
        T_lidar_to_cam=T_current,
        splat_radius=2,
        min_valid_points=args.cluster_min_points_2d,
        alpha_radius=args.alpha_radius,
    )
    debug and info(f"projected clusters before filter: {len(projected_clusters)}")

    projected_clusters = filter_projected_clusters(
        projected_clusters,
        image_shape=image_bgr.shape[:2],
        min_mask_area=args.cluster_min_mask_area,
        max_area_ratio=args.cluster_max_area_ratio,
        max_border_touch_ratio=args.cluster_max_border_touch_ratio,
        min_points_2d=args.cluster_min_points_2d,
        max_aspect_ratio=args.cluster_max_aspect_ratio,
    )
    info(f"projected clusters after filter: {len(projected_clusters)}")

    matches = find_best_mask_groups_for_clusters_support_global(
        sam_masks=sam_masks,
        projected_clusters=projected_clusters,
        image_shape=image_bgr.shape[:2],
        debug_progress=args.debug_progress,
    )
    info(f"best matches before size filtering: {len(matches)}")

    matches = filter_background_like_group_matches(
        best_matches=matches,
        sam_masks=sam_masks,
        projected_clusters=projected_clusters,
        image_shape=image_bgr.shape[:2],
        max_mask_area_ratio=args.max_mask_area_ratio_filter,
        max_cluster_area_ratio=args.max_cluster_area_ratio_filter,
        min_score=args.group_min_score,
    )
    info(f"matches after size filtering: {len(matches)}")
    return projected_clusters, matches


def run_calibration(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    image_bgr = load_image_bgr(args.image)
    points_raw = load_point_cloud_xyz(args.pc)
    K = np.array([ [7.188560000000e+02,0.000000000000e+00,6.071928000000e+02], 
                  [0.000000000000e+00,7.188560000000e+02,1.852157000000e+02], 
                  [0.0, 0.0, 1.0], ], 
                  dtype=np.float64)

    T0 = np.array([ [0,-1,0,0],
                                [0,0,-1,0],
                                [1,0,0,0],
                                [0,0,0,1], ], 
                               dtype=np.float64)
    info(f"raw points: {len(points_raw)}")

    sam_masks = load_sam_masks(args.sam_npz, args.sam_meta if args.sam_meta else None)
    sam_masks = filter_sam_masks_basic(sam_masks, image_shape=image_bgr.shape[:2],
                                       min_area=args.min_mask_area,
                                       max_area_ratio=args.max_mask_area_ratio)
    info(f"loaded SAM masks after filter: {len(sam_masks)}")

    object_clusters = run_initial_preprocessing(image_bgr, points_raw, K, T0, args)
    if len(object_clusters) == 0:
        raise RuntimeError("No object clusters after preprocessing.")

    T_current = T0.copy()
    optimizer_history: List[dict] = []
    final_pairs: List[ContinuousFixedPair] = []
    final_matches: List[MatchResult] = []
    final_projected_clusters: List[ProjectedClusterMask] = []

    for outer_iter in range(args.outer_iters):
        info(f"outer iteration {outer_iter + 1}/{args.outer_iters}")

        projected_clusters, matches = run_pair_selection(
            image_bgr=image_bgr,
            sam_masks=sam_masks,
            object_clusters=object_clusters,
            K=K,
            T_current=T_current,
            args=args,
            debug=args.debug_progress,
        )
        print_group_matches(matches, sam_masks, top_k=args.top_k)

        fixed_pairs = build_continuous_fixed_pairs(
            matches=matches,
            projected_clusters=projected_clusters,
            object_clusters=object_clusters,
            sam_masks=sam_masks,
            image_shape=image_bgr.shape[:2],
            top_k=args.opt_top_k_pairs,
            sample_points_per_cluster=args.sample_points_per_cluster,
        )

        info(f"continuous fixed pairs for optimization: {len(fixed_pairs)}")
        if len(fixed_pairs) == 0:
            info("No fixed pairs available. Stop.")
            final_matches = matches
            final_projected_clusters = projected_clusters
            final_pairs = []
            break

        T_new, opt_info = optimize_pose_for_continuous_fixed_pairs(
            fixed_pairs=fixed_pairs,
            K=K,
            T_current=T_current,
            image_shape=image_bgr.shape[:2],
            debug_progress=args.debug_progress,
            lm_max_nfev=args.lm_max_nfev,
            trans_reg=args.trans_reg,
            rot_reg=args.rot_reg,
            trans_bound=args.trans_bound,
            rot_bound_deg=args.rot_bound_deg,
        )
        optimizer_history.append({"outer_iter": outer_iter, **opt_info})
        opt_fun = opt_info.get("fun", opt_info.get("cost", float("inf")))
        info(f"optimizer success={opt_info['success']} fun={opt_fun:.6f} nfev={opt_info.get('nfev', -1)}")

        delta_t = np.linalg.norm(T_new[:3, 3] - T_current[:3, 3])
        R_delta = T_new[:3, :3] @ T_current[:3, :3].T
        rvec, _ = cv2.Rodrigues(R_delta)
        delta_r = float(np.linalg.norm(rvec))
        info(f"pose update: dt={delta_t:.6f} m, dr={math.degrees(delta_r):.6f} deg")

        T_current = T_new
        final_pairs = fixed_pairs
        final_matches = matches
        final_projected_clusters = projected_clusters

        if delta_t < args.stop_trans_eps and delta_r < math.radians(args.stop_rot_eps_deg):
            info("Converged by small pose update.")
            break

    # Final reprojection and pair refresh at optimized pose
    final_projected_clusters, final_matches = run_pair_selection(
        image_bgr=image_bgr,
        sam_masks=sam_masks,
        object_clusters=object_clusters,
        K=K,
        T_current=T_current,
        args=args,
        debug=args.debug_progress,
    )
    final_pairs = build_continuous_fixed_pairs(
        matches=final_matches,
        projected_clusters=final_projected_clusters,
        object_clusters=object_clusters,
        sam_masks=sam_masks,
        image_shape=image_bgr.shape[:2],
        top_k=args.opt_top_k_pairs,
        sample_points_per_cluster=args.sample_points_per_cluster,
    )

    # Save outputs
    final_cluster_overlay = draw_projected_cluster_masks_overlay(image_bgr, final_projected_clusters, alpha=0.28, seed=7)
    cv2.imwrite(os.path.join(args.output_dir, "final_projected_clusters.png"), final_cluster_overlay)

    final_match_overlay = overlay_mask_group_matches(
        image_bgr=image_bgr,
        sam_masks=sam_masks,
        projected_clusters=final_projected_clusters,
        matches=final_matches,
        top_k=args.top_k,
    )
    cv2.imwrite(os.path.join(args.output_dir, "final_group_matches.png"), final_match_overlay)

    final_pointcloud_overlay = draw_projected_pointcloud_overlay(
        image_bgr=image_bgr,
        points_lidar=points_raw,
        K=K,
        T_lidar_to_cam=T_current,
        point_radius=getattr(args, 'pointcloud_point_radius', POINTCLOUD_POINT_RADIUS),
        max_points=getattr(args, 'pointcloud_max_points', POINTCLOUD_MAX_POINTS),
    )
    cv2.imwrite(os.path.join(args.output_dir, "final_pointcloud_projection.png"), final_pointcloud_overlay)

    gt_pose = np.array([ [ 4.276802385584e-04,-9.999672484946e-01,-8.084491683471e-03,-1.198459927713e-02], 
                        [-7.210626507497e-03,8.081198471645e-03,-9.999413164504e-01,-5.403984729748e-02], 
                        [9.999738645903e-01,4.859485810390e-04,-7.206933692422e-03,-2.921968648686e-01], 
                        [ 0.0, 0.0, 0.0, 1.0], ], 
                        dtype=np.float64)
    gt_pose_info = None
    if gt_pose is not None:
        gt_error = pose_error_metrics(T_current, gt_pose)
        gt_pose_info = {
            "T_gt": gt_pose.tolist(),
            **gt_error,
        }

        info(
            "GT error: "
            f"translation={gt_error['translation_error_m']:.6f} m, "
            f"rotation={gt_error['rotation_error_deg']:.6f} deg"
        )

        gt_pointcloud_overlay = draw_projected_pointcloud_overlay(
            image_bgr=image_bgr,
            points_lidar=points_raw,
            K=K,
            T_lidar_to_cam=gt_pose,
            point_radius=getattr(args, 'pointcloud_point_radius', POINTCLOUD_POINT_RADIUS),
            max_points=getattr(args, 'pointcloud_max_points', POINTCLOUD_MAX_POINTS),
        )
        cv2.imwrite(os.path.join(args.output_dir, "gt_pointcloud_projection.png"), gt_pointcloud_overlay)

        with open(os.path.join(args.output_dir, "gt_pose_error.json"), "w", encoding="utf-8") as f:
            json.dump(gt_pose_info, f, indent=2)
    else:
        info("GT transform not available in kitti calib json; skipping GT error and GT projection output.")

    save_pose_json(
        path=os.path.join(args.output_dir, "calibration_result.json"),
        K=K,
        T0=T0,
        T_opt=T_current,
        optimizer_history=optimizer_history,
        fixed_pairs=final_pairs,
        gt_pose_info=gt_pose_info,
    )

    with open(os.path.join(args.output_dir, "final_matches.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 130 + "\n")
        f.write(f"{'rank':<6} {'cluster':<8} {'raw_ids':<20} {'score':<10} {'IoU':<10} {'B-IoU':<10} {'Contain':<10} {'BBoxIoU':<10} {'AreaRatio':<10}\n")
        f.write("-" * 130 + "\n")
        for rank, m in enumerate(final_matches[:args.top_k]):
            raw_ids = [int(sam_masks[mid].get('raw_id', mid)) for mid in m.mask_ids]
            f.write(f"{rank:<6} {m.cluster_id:<8} {str(raw_ids):<20} {m.selection_score:<10.4f} {m.iou:<10.4f} {m.boundary_iou:<10.4f} {m.containment:<10.4f} {m.bbox_iou:<10.4f} {m.area_ratio:<10.4f}\n")
        f.write("=" * 130 + "\n")

    info(f"Saved outputs to: {args.output_dir}")
    info(f"Optimized pose written to: {os.path.join(args.output_dir, 'calibration_result.json')}")


# =========================================================
# CLI
# =========================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibration pipeline with SAM-mask group ↔ projected cluster alternating optimization")

    p.add_argument("--image", type=str, required=True)
    p.add_argument("--pc", type=str, required=True)
    p.add_argument("--sam-npz", type=str, required=True)
    p.add_argument("--sam-meta", type=str, default="")
    p.add_argument("--output-dir", type=str, required=True)

    # preprocessing
    p.add_argument("--margin-px", type=int, default=120)
    p.add_argument("--min-depth", type=float, default=1.0)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument("--voxel-size", type=float, default=0.0)
    p.add_argument("--nb-neighbors", type=int, default=20)
    p.add_argument("--std-ratio", type=float, default=2.0)
    p.add_argument("--ground-dist-th", type=float, default=0.10)
    p.add_argument("--ground-ransac-n", type=int, default=3)
    p.add_argument("--ground-iters", type=int, default=1000)
    p.add_argument("--min-ground-inliers", type=int, default=800)
    p.add_argument("--max-ground-abs-height", type=float, default=2.5)
    p.add_argument("--min-ground-up-dot", type=float, default=0.85)
    p.add_argument("--cluster-tolerance", type=float, default=0.30)
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--max-cluster-size", type=int, default=30000)
    p.add_argument("--cluster-min-extent", type=float, default=0.20)
    p.add_argument("--cluster-max-extent", type=float, default=8.0)
    p.add_argument("--cluster-max-flatness", type=float, default=20.0)

    # projection / masks
    p.add_argument("--alpha-radius", type=float, default=20.0)
    p.add_argument("--cluster-min-points-2d", type=int, default=20)
    p.add_argument("--cluster-min-mask-area", type=int, default=500)
    p.add_argument("--cluster-max-area-ratio", type=float, default=0.18)
    p.add_argument("--cluster-max-border-touch-ratio", type=float, default=0.12)
    p.add_argument("--cluster-max-aspect-ratio", type=float, default=8.0)

    # sam filtering
    p.add_argument("--min-mask-area", type=int, default=200)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.35)

    # group matching
    p.add_argument("--greedy-min-improve", type=float, default=0.001)
    p.add_argument("--max-group-size", type=int, default=0, help="0 means unlimited greedy growth")
    p.add_argument("--group-min-score", type=float, default=0.10)
    p.add_argument("--group-min-iou", type=float, default=0.01)
    p.add_argument("--group-min-containment", type=float, default=0.10)
    p.add_argument("--group-improve-margin", type=float, default=0.02)
    p.add_argument("--max-mask-area-ratio-filter", type=float, default=0.20)
    p.add_argument("--max-cluster-area-ratio-filter", type=float, default=0.30)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--pointcloud-point-radius", type=int, default=1)
    p.add_argument("--pointcloud-max-points", type=int, default=60000)

    # optimization
    p.add_argument("--outer-iters", type=int, default=7)
    p.add_argument("--opt-top-k-pairs", type=int, default=5)
    p.add_argument("--lm-max-nfev", type=int, default=80)
    p.add_argument("--trans-reg", type=float, default=0.05)
    p.add_argument("--rot-reg", type=float, default=0.05)
    p.add_argument("--trans-bound", type=float, default=0.5)
    p.add_argument("--rot-bound-deg", type=float, default=5.0)
    p.add_argument("--stop-trans-eps", type=float, default=1e-3)
    p.add_argument("--stop-rot-eps-deg", type=float, default=0.005)
    p.add_argument("--sample-points-per-cluster", type=int, default=128)
    p.add_argument("--debug-progress", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()