#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, Delaunay, QhullError

from typing import Optional

@dataclass
class Segment3D:
    segment_id: int
    kind: str  # "ground" or "cluster"
    points: np.ndarray  # (N, 3)
    color: Tuple[int, int, int]  # BGR

@dataclass
class ProjectedClusterMask:
    cluster_id: int
    points_3d: np.ndarray              # (N, 3)
    points_2d: np.ndarray              # (M, 2), valid projected points
    depths: np.ndarray                 # (M,)
    valid_mask: np.ndarray             # (N,)
    raw_point_mask: np.ndarray         # (H, W), uint8 {0,255}
    binary_mask: np.ndarray            # (H, W), uint8 {0,255}
    boundary_mask: np.ndarray          # (H, W), uint8 {0,255}
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    bbox_fill_ratio: float


# -----------------------------
# I/O
# -----------------------------
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
        pts = np.fromfile(str(p), dtype=np.float32).reshape(-1, 4)
        return pts[:, :3].astype(np.float32)

    raise ValueError(f"Unsupported point cloud format: {p.suffix}")


def parse_kitti_calib(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        K: (3, 3)
        T_velo_to_cam: (4, 4)
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"KITTI calib file not found: {calib_path}")

    data = {}
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            vals = np.array([float(x) for x in value.split()], dtype=np.float64)
            data[key] = vals

    if "P2" not in data:
        raise ValueError("KITTI calib file missing P2")
    if "Tr" not in data and "Tr_velo_to_cam" not in data:
        raise ValueError("KITTI calib file missing Tr or Tr_velo_to_cam")

    P2 = data["P2"].reshape(3, 4)
    K = P2[:, :3]

    tr_key = "Tr_velo_to_cam" if "Tr_velo_to_cam" in data else "Tr"
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = data[tr_key].reshape(3, 4)

    return K, T


# -----------------------------
# Geometry / projection
# -----------------------------
def project_points(
    points_lidar: np.ndarray,
    K: np.ndarray,
    T_lidar_to_cam: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        points_lidar: (N, 3)
        K: (3, 3)
        T_lidar_to_cam: (4, 4)

    Returns:
        uv: (N, 2)
        depth: (N,)
        valid_depth: (N,)
    """
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


def prefilter_points_by_fov(
    points_lidar: np.ndarray,
    K: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    image_shape: Tuple[int, int],
    margin_px: int = 120,
    min_depth: float = 1.0,
    max_depth: float = 80.0,
) -> np.ndarray:
    """
    Rough-init 기반 FOV prefilter.
    이미지 경계 바깥으로 조금 넓게 margin을 두고 점들을 남김.
    """
    h, w = image_shape[:2]
    uv, depth, valid_depth = project_points(points_lidar, K, T_lidar_to_cam)

    u = uv[:, 0]
    v = uv[:, 1]

    valid = valid_depth
    valid &= (depth >= min_depth) & (depth <= max_depth)
    valid &= (u >= -margin_px) & (u < w + margin_px)
    valid &= (v >= -margin_px) & (v < h + margin_px)

    return points_lidar[valid]


# -----------------------------
# Point cloud preprocessing
# -----------------------------
def voxel_downsample(points: np.ndarray, voxel_size: float = 0.10) -> np.ndarray:
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points, dtype=np.float32)


def remove_statistical_outliers(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    if len(points) == 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd.points, dtype=np.float32)


def estimate_normals(
    points: np.ndarray,
    knn: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, np.empty((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return points, normals


# -----------------------------
# Ground-only removal
# -----------------------------
def choose_ground_plane(
    candidate_points: np.ndarray,
    plane_model: np.ndarray,
    lidar_height_hint: float = 1.8,
    min_ground_inliers: int = 800,
    max_ground_abs_height: float = 2.5,
    min_up_dot: float = 0.85,
) -> bool:
    """
    plane_model: ax + by + cz + d = 0 in LiDAR frame

    Heuristic:
    - ground normal should be roughly vertical in LiDAR frame
      (for KITTI Velodyne frame, z is usually up/down reference candidate)
    - plane should contain enough points
    - plane points should not be too high above sensor
    """
    if len(candidate_points) < min_ground_inliers:
        return False

    a, b, c, d = plane_model.astype(np.float64)
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        return False
    n = n / n_norm

    # assume LiDAR z-axis is vertical-ish reference for ground test
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_dot = abs(float(np.dot(n, z_axis)))
    if up_dot < min_up_dot:
        return False

    # ground should be relatively low in LiDAR frame
    z_vals = candidate_points[:, 2]
    if np.median(np.abs(z_vals)) > max_ground_abs_height:
        return False

    return True


def remove_ground_only(
    points: np.ndarray,
    distance_threshold: float = 0.15,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_ground_inliers: int = 800,
    max_ground_abs_height: float = 2.5,
    min_up_dot: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Returns:
        ground_points: (Ng, 3)
        non_ground_points: (Nng, 3)
        plane_model: (4,) or None
    """
    if len(points) < max(ransac_n, min_ground_inliers):
        return np.empty((0, 3), dtype=np.float32), points, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    if len(inliers) == 0:
        return np.empty((0, 3), dtype=np.float32), points, None

    inliers = np.array(inliers, dtype=np.int64)
    ground_points = points[inliers]
    mask = np.ones(len(points), dtype=bool)
    mask[inliers] = False
    non_ground_points = points[mask]

    ok = choose_ground_plane(
        candidate_points=ground_points,
        plane_model=np.asarray(plane_model, dtype=np.float64),
        min_ground_inliers=min_ground_inliers,
        max_ground_abs_height=max_ground_abs_height,
        min_up_dot=min_up_dot,
    )

    if not ok:
        return np.empty((0, 3), dtype=np.float32), points, None

    return ground_points, non_ground_points, np.asarray(plane_model, dtype=np.float64)


# -----------------------------
# Euclidean clustering
# -----------------------------
def euclidean_clustering(
    points: np.ndarray,
    tolerance: float = 0.8,
    min_cluster_size: int = 30,
    max_cluster_size: int = 30000,
) -> List[np.ndarray]:
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
        cluster_idx = []

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


# -----------------------------
# Cluster filtering
# -----------------------------
def cluster_extent(points: np.ndarray) -> Tuple[float, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ext = maxs - mins
    return float(ext[0]), float(ext[1]), float(ext[2])


def filter_object_clusters(
    clusters: List[np.ndarray],
    min_extent: float = 0.20,
    max_extent: float = 8.0,
    max_flatness_ratio: float = 20.0,
) -> List[np.ndarray]:
    """
    Remove obviously bad clusters:
    - too tiny
    - too huge
    - extremely flat / strip-like
    """
    out = []
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
            # too strip-like / wall-strip-like
            continue

        out.append(c)
    return out


# -----------------------------
# Visualization
# -----------------------------
def make_color_table(n: int, seed: int = 7) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    colors = []
    for _ in range(n):
        c = rng.integers(50, 255, size=3).tolist()
        colors.append((int(c[0]), int(c[1]), int(c[2])))  # BGR
    return colors


def project_segments_overlay(
    image_bgr: np.ndarray,
    segments: List[Segment3D],
    K: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    point_radius: int = 1,
    alpha: float = 0.65,
    draw_bbox: bool = True,
) -> np.ndarray:
    vis = image_bgr.copy().astype(np.float32)
    h, w = image_bgr.shape[:2]

    for seg in segments:
        uv, depth, valid_depth = project_points(seg.points, K, T_lidar_to_cam)
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)

        valid = valid_depth
        valid &= (u >= 0) & (u < w)
        valid &= (v >= 0) & (v < h)

        if valid.sum() == 0:
            continue

        uu = u[valid]
        vv = v[valid]

        color = np.array(seg.color, dtype=np.float32)

        for px, py in zip(uu, vv):
            cv2.circle(vis, (int(px), int(py)), point_radius, color.tolist(), thickness=-1)

        if draw_bbox:
            x1, x2 = int(uu.min()), int(uu.max())
            y1, y2 = int(vv.min()), int(vv.max())
            cv2.rectangle(vis, (x1, y1), (x2, y2), seg.color, 1)
            cv2.putText(
                vis,
                f"{seg.kind}:{seg.segment_id}",
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                seg.color,
                1,
                cv2.LINE_AA,
            )

    blended = cv2.addWeighted(image_bgr.astype(np.float32), 1.0 - alpha, vis, alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def visualize_segments_open3d(
    segments: List[Segment3D],
    show_frame: bool = True,
    frame_size: float = 2.0,
    window_name: str = "Segmented Point Cloud",
) -> None:
    geometries: List[o3d.geometry.Geometry] = []

    for seg in segments:
        if seg.points.size == 0:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg.points.astype(np.float64))

        # BGR (OpenCV) -> RGB (Open3D)
        b, g, r = seg.color
        rgb = np.array([r, g, b], dtype=np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.tile(rgb, (seg.points.shape[0], 1)))
        geometries.append(pcd)

    if show_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    if not geometries:
        print("[WARN] no points to visualize in Open3D")
        return

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=720,
    )

def build_segments(
    ground_points: np.ndarray,
    object_clusters: List[np.ndarray],
    seed: int = 7,
    include_ground: bool = True,
) -> List[Segment3D]:
    total = len(object_clusters) + (1 if include_ground and len(ground_points) > 0 else 0)
    colors = make_color_table(max(total, 1), seed=seed)

    segments: List[Segment3D] = []
    seg_id = 0

    if include_ground and len(ground_points) > 0:
        segments.append(Segment3D(seg_id, "ground", ground_points, colors[seg_id]))
        seg_id += 1

    for cluster in object_clusters:
        segments.append(Segment3D(seg_id, "cluster", cluster, colors[seg_id]))
        seg_id += 1

    return segments

def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return (binary * 255).astype(np.uint8)

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary, dtype=np.uint8)
    out[labels == largest_idx] = 255
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape

    flood = mask_u8.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, flood_inv)
    return filled.astype(np.uint8)


def make_boundary_band(mask: np.ndarray, band_width: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1)
    )
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.subtract(mask, eroded)


def compute_bbox_fill_ratio(mask: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    area = int((mask > 0).sum())
    bbox_area = max(1, x2 - x1 + 1) * max(1, y2 - y1 + 1)
    return float(area) / float(bbox_area)


def render_projected_points_mask(
    uv_valid: np.ndarray,
    image_shape: Tuple[int, int],
    splat_radius: int = 2,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if uv_valid.shape[0] == 0:
        return mask

    x = np.round(uv_valid[:, 0]).astype(np.int32)
    y = np.round(uv_valid[:, 1]).astype(np.int32)

    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = x[valid]
    y = y[valid]

    for px, py in zip(x, y):
        cv2.circle(mask, (int(px), int(py)), splat_radius, 255, thickness=-1)

    return mask


def postprocess_projected_mask(
    raw_mask: np.ndarray,
    close_kernel: int = 9,
    close_iter: int = 1,
    dilate_kernel: int = 5,
    dilate_iter: int = 1,
    keep_largest_cc_flag: bool = True,
    fill_holes_flag: bool = True,
) -> np.ndarray:
    mask = raw_mask.copy()

    if dilate_kernel > 1 and dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    if close_kernel > 1 and close_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    if fill_holes_flag:
        mask = fill_holes(mask)

    if keep_largest_cc_flag:
        mask = keep_largest_component(mask)

    return mask.astype(np.uint8)

def render_convex_hull_mask(
    uv_valid: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
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

def render_alpha_shape_mask(
    uv_valid: np.ndarray,
    image_shape: Tuple[int, int],
    alpha_radius: float = 25.0,
    min_valid_points: int = 8,
    fallback_to_convex: bool = True,
) -> np.ndarray:
    """
    True alpha-shape-like mask in 2D image plane using Delaunay triangulation.

    alpha_radius:
        circumradius threshold in pixel space.
        smaller -> tighter / more concave
        larger  -> closer to convex hull
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if uv_valid.shape[0] < min_valid_points:
        return mask

    pts = np.round(uv_valid).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    # 중복점 제거
    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 4:
        if fallback_to_convex:
            return render_convex_hull_mask(pts.astype(np.float32), image_shape)
        return mask

    pts_f = pts.astype(np.float64)

    try:
        tri = Delaunay(pts_f)
    except QhullError:
        if fallback_to_convex:
            return render_convex_hull_mask(pts.astype(np.float32), image_shape)
        return mask

    kept_triangles = []

    for simplex in tri.simplices:
        p1 = pts_f[simplex[0]]
        p2 = pts_f[simplex[1]]
        p3 = pts_f[simplex[2]]

        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)

        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)

        if area_sq <= 1e-8:
            continue

        area = np.sqrt(area_sq)
        circum_r = (a * b * c) / (4.0 * area)

        if circum_r <= alpha_radius:
            kept_triangles.append(pts[simplex].reshape(-1, 1, 2))

    if len(kept_triangles) == 0:
        if fallback_to_convex:
            return render_convex_hull_mask(pts.astype(np.float32), image_shape)
        return mask

    for t in kept_triangles:
        cv2.fillConvexPoly(mask, t, 255)

    return mask

def build_projected_cluster_mask(
    cluster_id: int,
    cluster_points: np.ndarray,
    image_shape: Tuple[int, int],
    K: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    splat_radius: int = 2,
    close_kernel: int = 9,
    close_iter: int = 1,
    dilate_kernel: int = 5,
    dilate_iter: int = 1,
    min_valid_points: int = 8,
    alpha_radius: float = 25.0,
) -> Optional[ProjectedClusterMask]:
    uv, depth, valid_depth = project_points(cluster_points, K, T_lidar_to_cam)

    h, w = image_shape
    u = uv[:, 0]
    v = uv[:, 1]

    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid = valid_depth & in_bounds

    uv_valid = uv[valid]
    depth_valid = depth[valid]

    if uv_valid.shape[0] < min_valid_points:
        return None

    # 디버깅용 raw point mask는 유지
    raw_mask = render_projected_points_mask(
        uv_valid=uv_valid,
        image_shape=image_shape,
        splat_radius=splat_radius,
    )

    # 핵심: binary mask는 concave hull 사용
    hull_mask = render_alpha_shape_mask(
        uv_valid=uv_valid,
        image_shape=image_shape,
        alpha_radius=alpha_radius,
        min_valid_points=min_valid_points,
        fallback_to_convex=True,
    )

    # 필요하면 hull에도 약간의 후처리만 적용
    binary_mask = postprocess_projected_mask(
        raw_mask=hull_mask,
        close_kernel=5,
        close_iter=0,
        dilate_kernel=3,
        dilate_iter=0,
        keep_largest_cc_flag=True,
        fill_holes_flag=True,
    )

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

def build_all_projected_cluster_masks(
    object_clusters: List[np.ndarray],
    image_shape: Tuple[int, int],
    K: np.ndarray,
    T_lidar_to_cam: np.ndarray,
    splat_radius: int = 2,
    close_kernel: int = 9,
    close_iter: int = 1,
    dilate_kernel: int = 5,
    dilate_iter: int = 1,
    min_valid_points: int = 8,
    alpha_radius: float = 25.0,
) -> List[ProjectedClusterMask]:
    results: List[ProjectedClusterMask] = []

    for cid, cluster_points in enumerate(object_clusters):
        proj = build_projected_cluster_mask(
            cluster_id=cid,
            cluster_points=cluster_points,
            image_shape=image_shape,
            K=K,
            T_lidar_to_cam=T_lidar_to_cam,
            splat_radius=splat_radius,
            close_kernel=close_kernel,
            close_iter=close_iter,
            dilate_kernel=dilate_kernel,
            dilate_iter=dilate_iter,
            min_valid_points=min_valid_points,
            alpha_radius=alpha_radius,
        )
        if proj is not None:
            results.append(proj)

    return results

def draw_projected_cluster_masks_overlay(
    image_bgr: np.ndarray,
    projected_clusters: List[ProjectedClusterMask],
    alpha: float = 0.30,
    seed: int = 7,
    draw_bbox: bool = True,
    draw_label: bool = True,
    draw_points: bool = True,
    point_radius: int = 1,
) -> np.ndarray:
    vis = image_bgr.copy().astype(np.float32)
    rng = np.random.default_rng(seed)

    for proj in projected_clusters:
        color = rng.integers(50, 255, size=3).tolist()
        color_arr = np.array(color, dtype=np.float32)

        mask = proj.binary_mask > 0
        vis[mask] = (1 - alpha) * vis[mask] + alpha * color_arr

        contours, _ = cv2.findContours(proj.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

        if draw_points and proj.points_2d.shape[0] > 0:
            uv = np.round(proj.points_2d).astype(np.int32)
            h, w = vis.shape[:2]
            valid = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < h)
            )
            uv = uv[valid]

            for px, py in uv:
                cv2.circle(vis, (int(px), int(py)), max(1, point_radius), color, thickness=-1)

        if draw_bbox and proj.bbox_xyxy is not None:
            x1, y1, x2, y2 = proj.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

            if draw_label:
                label = f"cluster:{proj.cluster_id}"
                cv2.putText(
                    vis,
                    label,
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    return np.clip(vis, 0, 255).astype(np.uint8)

def save_projected_cluster_masks(
    projected_clusters: List[ProjectedClusterMask],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    raw_dir = os.path.join(output_dir, "raw_point_masks")
    binary_dir = os.path.join(output_dir, "binary_masks")
    boundary_dir = os.path.join(output_dir, "boundary_masks")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)
    os.makedirs(boundary_dir, exist_ok=True)

    for proj in projected_clusters:
        cv2.imwrite(
            os.path.join(raw_dir, f"cluster_{proj.cluster_id:03d}_raw.png"),
            proj.raw_point_mask,
        )
        cv2.imwrite(
            os.path.join(binary_dir, f"cluster_{proj.cluster_id:03d}_binary.png"),
            proj.binary_mask,
        )
        cv2.imwrite(
            os.path.join(boundary_dir, f"cluster_{proj.cluster_id:03d}_boundary.png"),
            proj.boundary_mask,
        )



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Required input/output paths (set these before running).
    image_path = "/workspace/sam_calibration/data/kitti/images/000002.png"
    pointcloud_path = "/workspace/sam_calibration/data/kitti/pc/000002.pcd"
    output_path = "/workspace/sam_calibration/data/kitti/output/pc/projection_overlay.png"

    parser = argparse.ArgumentParser(description="CalibAnything-style preprocessing + projection overlay")

    parser.add_argument("--margin-px", type=int, default=120)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)

    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--nb-neighbors", type=int, default=20)
    parser.add_argument("--std-ratio", type=float, default=2.0)
    parser.add_argument("--normal-knn", type=int, default=40)

    parser.add_argument("--ground-dist-th", type=float, default=0.1)
    parser.add_argument("--ground-ransac-n", type=int, default=3)
    parser.add_argument("--ground-iters", type=int, default=1000)
    parser.add_argument("--min-ground-inliers", type=int, default=800)
    parser.add_argument("--max-ground-abs-height", type=float, default=2.5)
    parser.add_argument("--min-ground-up-dot", type=float, default=0.85)

    parser.add_argument("--cluster-tolerance", type=float, default=0.3)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--max-cluster-size", type=int, default=30000)

    parser.add_argument("--cluster-min-extent", type=float, default=0.20)
    parser.add_argument("--cluster-max-extent", type=float, default=8.0)
    parser.add_argument("--cluster-max-flatness", type=float, default=20.0)

    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-ground", action="store_true")
    parser.add_argument("--show-o3d", action="store_true", help="Show Open3D segmented point cloud view")
    parser.add_argument("--o3d-frame-size", type=float, default=2.0, help="Coordinate frame size in Open3D view")

    args = parser.parse_args()

    image_bgr = load_image_bgr(image_path)
    points = load_point_cloud_xyz(pointcloud_path)

    K = np.array([ [7.188560000000e+02,0.000000000000e+00,6.071928000000e+02], 
                  [0.000000000000e+00,7.188560000000e+02,1.852157000000e+02], 
                  [0.0, 0.0, 1.0], ], 
                  dtype=np.float64)
     
    # lidar -> camera extrinsic 
    # T_lidar_to_cam = np.array([ [ 4.276802385584e-04,-9.999672484946e-01,-8.084491683471e-03,-1.198459927713e-02], 
    #                            [-7.210626507497e-03,8.081198471645e-03,-9.999413164504e-01,-5.403984729748e-02], 
    #                            [9.999738645903e-01,4.859485810390e-04,-7.206933692422e-03,-2.921968648686e-01], 
    #                            [ 0.0, 0.0, 0.0, 1.0], ], 
    #                            dtype=np.float64)

    T_lidar_to_cam = np.array([ [0,-1,0,0],
                                [0,0,-1,0],
                                [1,0,0,0],
                                [0,0,0,1], ], 
                               dtype=np.float64)

    print(f"[INFO] raw points: {len(points)}")

    # 1) rough-init 기반 FOV prefilter
    points = prefilter_points_by_fov(
        points,
        K=K,
        T_lidar_to_cam=T_lidar_to_cam,
        image_shape=image_bgr.shape[:2],
        margin_px=args.margin_px,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    print(f"[INFO] after FOV prefilter: {len(points)}")

    # 2) downsample + outlier removal
    points = voxel_downsample(points, voxel_size=args.voxel_size)
    print(f"[INFO] after voxel downsample: {len(points)}")

    points = remove_statistical_outliers(
        points,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
    )
    print(f"[INFO] after outlier removal: {len(points)}")

    # 3) normal estimation (지금은 디버그용/향후 확장용)
    points, normals = estimate_normals(points, knn=args.normal_knn)
    print(f"[INFO] estimated normals: {normals.shape}")

    # 4) ground only
    ground_points, non_ground_points, plane_model = remove_ground_only(
        points,
        distance_threshold=args.ground_dist_th,
        ransac_n=args.ground_ransac_n,
        num_iterations=args.ground_iters,
        min_ground_inliers=args.min_ground_inliers,
        max_ground_abs_height=args.max_ground_abs_height,
        min_up_dot=args.min_ground_up_dot,
    )
    print(f"[INFO] ground points: {len(ground_points)}")
    print(f"[INFO] non-ground points: {len(non_ground_points)}")
    if plane_model is not None:
        print(f"[INFO] accepted ground plane model: {plane_model}")
    else:
        print("[INFO] no valid ground plane accepted")

    # 5) clustering on non-ground
    object_clusters = euclidean_clustering(
        non_ground_points,
        tolerance=args.cluster_tolerance,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
    )
    print(f"[INFO] raw object clusters: {len(object_clusters)}")

    # 6) object cluster filtering
    object_clusters = filter_object_clusters(
        object_clusters,
        min_extent=args.cluster_min_extent,
        max_extent=args.cluster_max_extent,
        max_flatness_ratio=args.cluster_max_flatness,
    )
    print(f"[INFO] filtered object clusters: {len(object_clusters)}")

    # 6.5) build projected cluster masks
    projected_clusters = build_all_projected_cluster_masks(
        object_clusters=object_clusters,
        image_shape=image_bgr.shape[:2],
        K=K,
        T_lidar_to_cam=T_lidar_to_cam,
        splat_radius=2,
        close_kernel=5,
        close_iter=1,
        dilate_kernel=3,
        dilate_iter=0,
        min_valid_points=20,
        alpha_radius=15.0,
    )
    print(f"[INFO] projected cluster masks: {len(projected_clusters)}")

    projected_overlay = draw_projected_cluster_masks_overlay(
        image_bgr=image_bgr,
        projected_clusters=projected_clusters,
        alpha=0.28,
        seed=args.seed,
        draw_bbox=True,
        draw_label=True,
        draw_points=True,
        point_radius=max(1, args.point_radius),
    )

    projected_overlay_path = output_path.replace(".png", "_cluster_masks.png")
    cv2.imwrite(projected_overlay_path, projected_overlay)
    print(f"[INFO] saved projected cluster mask overlay: {projected_overlay_path}")

    projected_mask_dir = output_path.replace(".png", "_cluster_masks")
    save_projected_cluster_masks(projected_clusters, projected_mask_dir)
    print(f"[INFO] saved projected cluster masks to: {projected_mask_dir}")

    # 7) build segment list
    segments = build_segments(
        ground_points=ground_points,
        object_clusters=object_clusters,
        seed=args.seed,
        include_ground=args.include_ground,
    )
    print(f"[INFO] total segments to project: {len(segments)}")

    if args.show_o3d:
        print("[INFO] opening Open3D segmentation view...")
        visualize_segments_open3d(
            segments=segments,
            show_frame=True,
            frame_size=args.o3d_frame_size,
            window_name="Ground + Object Clusters",
        )

    # 8) overlay projection
    overlay = project_segments_overlay(
        image_bgr=image_bgr,
        segments=segments,
        K=K,
        T_lidar_to_cam=T_lidar_to_cam,
        point_radius=args.point_radius,
        alpha=args.alpha,
        draw_bbox=True,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"[INFO] saved overlay: {output_path}")


if __name__ == "__main__":
    main()