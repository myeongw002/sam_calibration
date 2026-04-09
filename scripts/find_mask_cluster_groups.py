#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# Data classes
# =========================================================
@dataclass
class LoadedClusterMask:
    cluster_id: int
    binary_mask: np.ndarray          # (H, W), uint8 {0,255}
    boundary_mask: Optional[np.ndarray]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    bbox_fill_ratio: float


@dataclass
class MaskGroupMatch:
    cluster_id: int
    mask_ids: Tuple[int, ...]
    score: float
    iou: float
    boundary_iou: float
    containment: float
    bbox_iou: float
    area_ratio: float
    centroid_score: float


def log_progress(
    enabled: bool,
    stage: str,
    index: int,
    total: int,
    steps: int = 10,
    extra: str = "",
) -> None:
    if not enabled or total <= 0:
        return

    steps = max(1, steps)
    interval = max(1, total // steps)
    done = index + 1

    if done == 1 or done == total or (done % interval == 0):
        msg = f"[DEBUG] {stage}: {done}/{total}"
        if extra:
            msg += f" | {extra}"
        print(msg)


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

    a1 = bbox_area(b1)
    a2 = bbox_area(b2)
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def expand_bbox(
    b: Tuple[int, int, int, int],
    margin: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    h, w = image_shape
    x1, y1, x2, y2 = b
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(w - 1, x2 + margin),
        min(h - 1, y2 + margin),
    )


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = mask1 > 0
    m2 = mask2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_containment(cluster_mask: np.ndarray, image_mask: np.ndarray) -> float:
    c = cluster_mask > 0
    m = image_mask > 0
    inter = np.logical_and(c, m).sum()
    denom = c.sum()
    if denom == 0:
        return 0.0
    return float(inter) / float(denom)


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


def make_boundary_band(mask: np.ndarray, band_width: int = 3) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (band_width * 2 + 1, band_width * 2 + 1),
    )
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return cv2.subtract(mask_u8, eroded)


def boundary_iou(mask1: np.ndarray, mask2: np.ndarray, band_width: int = 3) -> float:
    b1 = make_boundary_band(mask1, band_width=band_width) > 0
    b2 = make_boundary_band(mask2, band_width=band_width) > 0
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def compute_bbox_fill_ratio(mask: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    area = int((mask > 0).sum())
    bbox_area_val = max(1, x2 - x1 + 1) * max(1, y2 - y1 + 1)
    return float(area) / float(bbox_area_val)


# =========================================================
# Loaders
# =========================================================
def load_sam_masks(npz_path: str, metadata_path: Optional[str] = None) -> List[dict]:
    npz = np.load(npz_path)
    if "masks" not in npz:
        raise ValueError(f"'masks' key not found in npz: {npz_path}")

    stack = npz["masks"]  # (N, H, W)
    if stack.ndim != 3:
        raise ValueError(f"Expected mask stack shape (N,H,W), got {stack.shape}")

    metadata = []
    if metadata_path is not None and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    masks: List[dict] = []
    for i in range(stack.shape[0]):
        ann = dict(metadata[i]) if i < len(metadata) else {}
        ann["segmentation"] = stack[i].astype(bool)
        masks.append(ann)

    return masks


def parse_cluster_id_from_filename(name: str) -> int:
    m = re.search(r"cluster_(\d+)", name)
    if m is None:
        raise ValueError(f"Cannot parse cluster id from filename: {name}")
    return int(m.group(1))


def load_cluster_masks(
    binary_dir: str,
    boundary_dir: Optional[str] = None,
    debug_progress: bool = False,
    progress_steps: int = 10,
) -> List[LoadedClusterMask]:
    binary_dir_p = Path(binary_dir)
    if not binary_dir_p.exists():
        raise FileNotFoundError(f"binary_dir not found: {binary_dir}")

    binary_files = sorted(binary_dir_p.glob("cluster_*_binary.png"))
    clusters: List[LoadedClusterMask] = []

    for idx, bf in enumerate(binary_files):
        log_progress(debug_progress, "load_cluster_masks", idx, len(binary_files), progress_steps)

        binary = cv2.imread(str(bf), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            continue

        cluster_id = parse_cluster_id_from_filename(bf.name)
        bbox = mask_bbox(binary)
        fill_ratio = compute_bbox_fill_ratio(binary, bbox)

        boundary = None
        if boundary_dir is not None:
            boundary_path = Path(boundary_dir) / f"cluster_{cluster_id:03d}_boundary.png"
            if boundary_path.exists():
                boundary = cv2.imread(str(boundary_path), cv2.IMREAD_GRAYSCALE)

        clusters.append(
            LoadedClusterMask(
                cluster_id=cluster_id,
                binary_mask=binary,
                boundary_mask=boundary,
                bbox_xyxy=bbox,
                bbox_fill_ratio=fill_ratio,
            )
        )

    return clusters


# =========================================================
# SAM prefilter
# =========================================================
def filter_sam_masks_basic(
    sam_masks: List[dict],
    image_shape: Tuple[int, int],
    min_area: int = 200,
    max_area_ratio: float = 0.35,
) -> List[dict]:
    h, w = image_shape
    image_area = h * w

    out: List[dict] = []
    for ann in sam_masks:
        seg = (ann["segmentation"] > 0)
        area = int(seg.sum())
        if area < min_area:
            continue
        if area / image_area > max_area_ratio:
            continue
        out.append(ann)
    return out


# =========================================================
# Group mask generation
# =========================================================
def union_mask_from_ids(
    sam_masks: List[dict],
    mask_ids: Tuple[int, ...],
    close_kernel: int = 7,
    close_iter: int = 1,
) -> np.ndarray:
    seg = np.zeros_like(sam_masks[0]["segmentation"], dtype=np.uint8)
    for mid in mask_ids:
        seg = np.logical_or(seg > 0, sam_masks[mid]["segmentation"] > 0).astype(np.uint8) * 255

    if len(mask_ids) > 1 and close_kernel > 1 and close_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return seg


# =========================================================
# Mask adjacency / candidate generation
# =========================================================
def build_mask_adjacency(
    sam_masks: List[dict],
    image_shape: Tuple[int, int],
    near_margin: int = 20,
    min_bbox_iou: float = 0.0,
    max_centroid_dist: float = 120.0,
    debug_progress: bool = False,
    progress_steps: int = 10,
) -> Dict[int, List[int]]:
    n = len(sam_masks)
    bboxes: List[Optional[Tuple[int, int, int, int]]] = []
    centroids: List[Optional[Tuple[float, float]]] = []

    for ann in sam_masks:
        m = (ann["segmentation"] > 0).astype(np.uint8) * 255
        bboxes.append(mask_bbox(m))
        centroids.append(mask_centroid(m))

    graph: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        log_progress(debug_progress, "build_mask_adjacency", i, n, progress_steps)

        if bboxes[i] is None:
            continue
        b1 = bboxes[i]
        b1e = expand_bbox(b1, near_margin, image_shape)

        for j in range(i + 1, n):
            if bboxes[j] is None:
                continue
            b2 = bboxes[j]

            biou = bbox_iou(b1e, b2)
            ci = centroids[i]
            cj = centroids[j]
            if ci is None or cj is None:
                cd = 1e9
            else:
                cd = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))

            if biou > min_bbox_iou or cd <= max_centroid_dist:
                graph[i].append(j)
                graph[j].append(i)

    return graph

def candidate_mask_ids_for_cluster_overlap_all(
    sam_masks: List[dict],
    cluster_proj: LoadedClusterMask,
    image_shape: Tuple[int, int],
    bbox_margin: int = 30,
) -> List[int]:
    """
    Use all masks whose bbox overlaps the expanded cluster bbox.
    No top-K truncation here.
    """
    cluster_bbox = cluster_proj.bbox_xyxy
    if cluster_bbox is None:
        return []

    cluster_bbox_exp = expand_bbox(cluster_bbox, bbox_margin, image_shape)

    candidates = []
    for mid, ann in enumerate(sam_masks):
        m = (ann["segmentation"] > 0).astype(np.uint8) * 255
        b = mask_bbox(m)
        if b is None:
            continue

        # any overlap
        if bbox_iou(cluster_bbox_exp, b) > 0.0:
            candidates.append(mid)

    return candidates

def score_single_masks_for_cluster(
    sam_masks: List[dict],
    cluster_proj: LoadedClusterMask,
    candidate_ids: List[int],
) -> List[MaskGroupMatch]:
    singles: List[MaskGroupMatch] = []

    for mid in candidate_ids:
        g = (mid,)
        group_mask = union_mask_from_ids(sam_masks, g)
        scored = score_mask_group_against_cluster(group_mask, cluster_proj)
        if scored is None:
            continue

        score, iou, bi, contain, biou, ar, cent = scored
        singles.append(
            MaskGroupMatch(
                cluster_id=cluster_proj.cluster_id,
                mask_ids=g,
                score=score,
                iou=iou,
                boundary_iou=bi,
                containment=contain,
                bbox_iou=biou,
                area_ratio=ar,
                centroid_score=cent,
            )
        )

    singles.sort(key=lambda x: x.score, reverse=True)
    return singles

def greedy_expand_group_for_cluster(
    sam_masks: List[dict],
    cluster_proj: LoadedClusterMask,
    seed_match: MaskGroupMatch,
    candidate_ids: List[int],
    mask_graph: Dict[int, List[int]],
    min_improve: float = 0.01,
    max_group_size: Optional[int] = None,
) -> MaskGroupMatch:
    """
    Start from one single-mask seed and greedily add one adjacent mask at a time
    if it improves the score.
    """
    current_ids = list(seed_match.mask_ids)
    current_best = seed_match
    candidate_set = set(candidate_ids)

    while True:
        if max_group_size is not None and len(current_ids) >= max_group_size:
            break

        # 현재 group에 인접한 mask만 후보로 사용
        frontier = set()
        for mid in current_ids:
            frontier.update(mask_graph.get(mid, []))

        frontier = [mid for mid in frontier if mid in candidate_set and mid not in current_ids]

        if len(frontier) == 0:
            break

        best_next_match: Optional[MaskGroupMatch] = None
        best_next_improve = -1e9

        for next_mid in frontier:
            new_ids = tuple(sorted(current_ids + [next_mid]))
            group_mask = union_mask_from_ids(sam_masks, new_ids)
            scored = score_mask_group_against_cluster(group_mask, cluster_proj)
            if scored is None:
                continue

            score, iou, bi, contain, biou, ar, cent = scored
            improvement = score - current_best.score

            if improvement > best_next_improve:
                best_next_improve = improvement
                best_next_match = MaskGroupMatch(
                    cluster_id=cluster_proj.cluster_id,
                    mask_ids=new_ids,
                    score=score,
                    iou=iou,
                    boundary_iou=bi,
                    containment=contain,
                    bbox_iou=biou,
                    area_ratio=ar,
                    centroid_score=cent,
                )

        if best_next_match is None:
            break

        if best_next_improve < min_improve:
            break

        current_ids = list(best_next_match.mask_ids)
        current_best = best_next_match

    return current_best

def find_best_mask_group_for_one_cluster_greedy(
    sam_masks: List[dict],
    cluster_proj: LoadedClusterMask,
    image_shape: Tuple[int, int],
    mask_graph: Dict[int, List[int]],
    min_score: float = 0.10,
    min_iou: float = 0.03,
    min_containment: float = 0.20,
    improve_margin: float = 0.05,
    greedy_min_improve: float = 0.01,
    max_group_size: Optional[int] = None,
) -> Optional[MaskGroupMatch]:
    """
    For one cluster:
    1) collect all overlapping masks
    2) score all singles
    3) greedily expand from each single seed
    4) keep the best final group
    """
    candidate_ids = candidate_mask_ids_for_cluster_overlap_all(
        sam_masks=sam_masks,
        cluster_proj=cluster_proj,
        image_shape=image_shape,
        bbox_margin=30,
    )

    if len(candidate_ids) == 0:
        return None

    single_matches = score_single_masks_for_cluster(
        sam_masks=sam_masks,
        cluster_proj=cluster_proj,
        candidate_ids=candidate_ids,
    )

    if len(single_matches) == 0:
        return None

    best_single = single_matches[0]
    best_group = best_single

    # 모든 single을 seed로 greedy 확장
    for seed in single_matches:
        grown = greedy_expand_group_for_cluster(
            sam_masks=sam_masks,
            cluster_proj=cluster_proj,
            seed_match=seed,
            candidate_ids=candidate_ids,
            mask_graph=mask_graph,
            min_improve=greedy_min_improve,
            max_group_size=max_group_size,   # None이면 score가 오를 때까지 계속
        )

        if grown.score > best_group.score:
            best_group = grown

    # 최종 quality gate
    if best_group.score < min_score:
        return None
    if best_group.iou < min_iou and best_group.containment < min_containment:
        return None

    # multi-mask가 single보다 별로 안 좋아지면 single 유지
    if len(best_group.mask_ids) > 1:
        if best_group.score < best_single.score + improve_margin:
            best_group = best_single

    return best_group

def find_best_mask_groups_for_clusters_greedy(
    sam_masks: List[dict],
    projected_clusters: List[LoadedClusterMask],
    image_shape: Tuple[int, int],
    min_score: float = 0.10,
    min_iou: float = 0.03,
    min_containment: float = 0.20,
    improve_margin: float = 0.05,
    greedy_min_improve: float = 0.01,
    max_group_size: Optional[int] = None,
    debug_progress: bool = False,
    progress_steps: int = 10,
) -> List[MaskGroupMatch]:
    mask_graph = build_mask_adjacency(
        sam_masks=sam_masks,
        image_shape=image_shape,
        near_margin=20,
        min_bbox_iou=0.0,
        max_centroid_dist=300.0,
        debug_progress=debug_progress,
        progress_steps=progress_steps,
    )

    matches: List[MaskGroupMatch] = []

    total_clusters = len(projected_clusters)
    for idx, proj in enumerate(projected_clusters):
        log_progress(
            debug_progress,
            "cluster_group_search",
            idx,
            total_clusters,
            progress_steps,
            extra=f"cluster_id={proj.cluster_id}",
        )

        best = find_best_mask_group_for_one_cluster_greedy(
            sam_masks=sam_masks,
            cluster_proj=proj,
            image_shape=image_shape,
            mask_graph=mask_graph,
            min_score=min_score,
            min_iou=min_iou,
            min_containment=min_containment,
            improve_margin=improve_margin,
            greedy_min_improve=greedy_min_improve,
            max_group_size=max_group_size,
        )
        if best is not None:
            matches.append(best)

    matches.sort(key=lambda x: x.score, reverse=True)
    return matches


def candidate_mask_ids_for_cluster(
    sam_masks: List[dict],
    cluster_proj: LoadedClusterMask,
    image_shape: Tuple[int, int],
    bbox_margin: int = 30,
    min_candidate_bbox_iou: float = 0.0,
    max_candidates: int = 12,
) -> List[int]:
    cluster_bbox = cluster_proj.bbox_xyxy
    if cluster_bbox is None:
        return []

    cluster_bbox_exp = expand_bbox(cluster_bbox, bbox_margin, image_shape)

    candidates = []
    for mid, ann in enumerate(sam_masks):
        m = (ann["segmentation"] > 0).astype(np.uint8) * 255
        b = mask_bbox(m)
        if b is None:
            continue

        biou = bbox_iou(cluster_bbox_exp, b)
        if biou >= min_candidate_bbox_iou:
            candidates.append((mid, biou))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in candidates[:max_candidates]]


def generate_group_candidates_for_cluster(
    sam_masks: List[dict],
    cluster_proj: LoadedClusterMask,
    image_shape: Tuple[int, int],
    mask_graph: Dict[int, List[int]],
    max_group_size: int = 2,
    max_candidates: int = 12,
) -> List[Tuple[int, ...]]:
    cand_ids = candidate_mask_ids_for_cluster(
        sam_masks=sam_masks,
        cluster_proj=cluster_proj,
        image_shape=image_shape,
        bbox_margin=30,
        min_candidate_bbox_iou=0.0,
        max_candidates=max_candidates,
    )

    group_set = set()

    # singles
    for mid in cand_ids:
        group_set.add((mid,))

    # pairs: only nearby masks
    if max_group_size >= 2:
        cand_id_set = set(cand_ids)
        for i in cand_ids:
            for j in mask_graph.get(i, []):
                if j in cand_id_set and i < j:
                    group_set.add((i, j))

    # triples: only from connected neighbors
    if max_group_size >= 3:
        cand_id_set = set(cand_ids)
        for i in cand_ids:
            neigh_i = [x for x in mask_graph.get(i, []) if x in cand_id_set]
            for j, k in combinations(sorted(neigh_i), 2):
                if i < j < k:
                    if (j in mask_graph.get(k, [])) or (k in mask_graph.get(j, [])):
                        group_set.add((i, j, k))

    return sorted(group_set, key=lambda x: (len(x), x))


# =========================================================
# Group scoring
# =========================================================
def score_mask_group_against_cluster(
    group_mask: np.ndarray,
    cluster_proj: LoadedClusterMask,
    iou_weight: float = 0.45,
    boundary_weight: float = 0.25,
    containment_weight: float = 0.20,
    bbox_weight: float = 0.06,
    area_weight: float = 0.03,
    centroid_weight: float = 0.01,
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    cluster_mask = cluster_proj.binary_mask
    bbox1 = mask_bbox(group_mask)
    bbox2 = cluster_proj.bbox_xyxy

    if bbox1 is None or bbox2 is None:
        return None

    iou = mask_iou(group_mask, cluster_mask)
    contain = mask_containment(cluster_mask, group_mask)
    bi = boundary_iou(group_mask, cluster_mask, band_width=3)
    biou = bbox_iou(bbox1, bbox2)
    ar = area_ratio(group_mask, cluster_mask)
    cent = centroid_score(group_mask, cluster_mask, sigma=120.0)

    score = (
        iou_weight * iou
        + boundary_weight * bi
        + containment_weight * contain
        + bbox_weight * biou
        + area_weight * ar
        + centroid_weight * cent
    )

    return score, iou, bi, contain, biou, ar, cent



# =========================================================
# Visualization / reporting
# =========================================================
def overlay_mask_group_matches(
    image_bgr: np.ndarray,
    sam_masks: List[dict],
    projected_clusters: List[LoadedClusterMask],
    matches: List[MaskGroupMatch],
    top_k: int = 10,
    alpha_mask: float = 0.20,
    alpha_cluster: float = 0.25,
) -> np.ndarray:
    vis = image_bgr.copy().astype(np.float32)
    rng = np.random.default_rng(11)

    cluster_id_to_proj = {p.cluster_id: p for p in projected_clusters}

    for rank, match in enumerate(matches[:top_k]):
        color = rng.integers(0, 255, size=3).tolist()
        color_arr = np.array(color, dtype=np.float32)

        # union mask
        union = union_mask_from_ids(sam_masks, match.mask_ids)
        union_bool = union > 0
        vis[union_bool] = (1 - alpha_mask) * vis[union_bool] + alpha_mask * color_arr

        contours_union, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_union, -1, (255, 255, 255), 1)

        # cluster mask
        proj = cluster_id_to_proj[match.cluster_id]
        cmask = proj.binary_mask > 0
        vis[cmask] = (1 - alpha_cluster) * vis[cmask] + alpha_cluster * color_arr

        contours_cluster, _ = cv2.findContours(proj.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_cluster, -1, tuple(int(c) for c in color), 2)

        # bbox + text
        if proj.bbox_xyxy is not None:
            x1, y1, x2, y2 = proj.bbox_xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), tuple(int(c) for c in color), 1)
            txt = f"r{rank} c{match.cluster_id} m{list(match.mask_ids)} s={match.score:.2f}"
            cv2.putText(
                vis,
                txt,
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tuple(int(c) for c in color),
                1,
                cv2.LINE_AA,
            )

    return np.clip(vis, 0, 255).astype(np.uint8)


def print_group_matches(matches: List[MaskGroupMatch], top_k: int = 15) -> None:
    print("=" * 130)
    print(f"{'rank':<6} {'cluster':<8} {'mask_ids':<20} {'score':<10} {'IoU':<10} {'B-IoU':<10} {'Contain':<10} {'BBoxIoU':<10} {'AreaRatio':<10}")
    print("-" * 130)
    for rank, m in enumerate(matches[:top_k]):
        print(
            f"{rank:<6} {m.cluster_id:<8} {str(list(m.mask_ids)):<20} "
            f"{m.score:<10.4f} {m.iou:<10.4f} {m.boundary_iou:<10.4f} "
            f"{m.containment:<10.4f} {m.bbox_iou:<10.4f} {m.area_ratio:<10.4f}"
        )
    print("=" * 130)

def mask_area(mask: np.ndarray) -> int:
    return int((mask > 0).sum())


def is_background_like_mask_by_size(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    max_area_ratio: float = 0.22,
) -> bool:
    h, w = image_shape
    area_ratio = mask_area(mask) / float(h * w)
    return area_ratio > max_area_ratio


def is_background_like_cluster_by_size(
    cluster: LoadedClusterMask,
    image_shape: Tuple[int, int],
    max_area_ratio: float = 0.18,
) -> bool:
    h, w = image_shape
    area_ratio = mask_area(cluster.binary_mask) / float(h * w)
    return area_ratio > max_area_ratio

def filter_background_like_group_matches(
    best_matches: List[MaskGroupMatch],
    sam_masks: List[dict],
    projected_clusters: List[LoadedClusterMask],
    image_shape: Tuple[int, int],
    max_mask_area_ratio: float = 0.22,
    max_cluster_area_ratio: float = 0.18,
    min_score: float = 0.10,
) -> List[MaskGroupMatch]:
    cluster_id_to_proj = {p.cluster_id: p for p in projected_clusters}
    filtered: List[MaskGroupMatch] = []

    for match in best_matches:
        proj = cluster_id_to_proj.get(match.cluster_id, None)
        if proj is None:
            continue

        union = union_mask_from_ids(sam_masks, match.mask_ids)

        # 1) mask 조합이 너무 크면 배경으로 보고 제거
        if is_background_like_mask_by_size(
            union,
            image_shape=image_shape,
            max_area_ratio=max_mask_area_ratio,
        ):
            continue

        # 2) cluster mask가 너무 크면 배경으로 보고 제거
        if is_background_like_cluster_by_size(
            proj,
            image_shape=image_shape,
            max_area_ratio=max_cluster_area_ratio,
        ):
            continue

        # 3) score는 최소한만 확인
        if match.score < min_score:
            continue

        filtered.append(match)

    filtered.sort(key=lambda x: x.score, reverse=True)
    return filtered

# =========================================================
# Main
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Find mask-group ↔ cluster pairs from saved masks")
    parser.add_argument("--image", type=str, required=False, default="", help="Optional image path for visualization")
    parser.add_argument("--sam-npz", type=str, required=True, help="Path to SAM masks npz (e.g. images/raw_masks.npz)")
    parser.add_argument("--sam-meta", type=str, required=False, default="", help="Optional SAM metadata json")
    parser.add_argument("--cluster-binary-dir", type=str, required=True, help="Path to cluster binary mask dir")
    parser.add_argument("--cluster-boundary-dir", type=str, required=False, default="", help="Optional path to cluster boundary mask dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")

    parser.add_argument("--greedy-min-improve", type=float, default=0.001)
    parser.add_argument("--max-group-size", type=int, default=0,
                        help="0 means unlimited greedy growth")
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--min-iou", type=float, default=0.01)
    parser.add_argument("--min-containment", type=float, default=0.10)
    parser.add_argument("--improve-margin", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=200)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--debug-progress", action="store_true", help="Print stage progress logs")
    parser.add_argument("--progress-steps", type=int, default=10, help="Approximate number of progress prints per stage")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load image if given
    image_bgr = None
    image_shape = None
    if args.image:
        image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {args.image}")
        image_shape = image_bgr.shape[:2]

    # Load SAM masks
    sam_masks = load_sam_masks(
        npz_path=args.sam_npz,
        metadata_path=args.sam_meta if args.sam_meta else None,
    )

    # Infer image shape from masks if image not given
    if image_shape is None:
        if len(sam_masks) == 0:
            raise ValueError("No SAM masks loaded.")
        image_shape = sam_masks[0]["segmentation"].shape

    sam_masks = filter_sam_masks_basic(
        sam_masks=sam_masks,
        image_shape=image_shape,
        min_area=args.min_mask_area,
        max_area_ratio=args.max_mask_area_ratio,
    )
    print(f"[INFO] loaded SAM masks: {len(sam_masks)}")

    # Load cluster masks
    projected_clusters = load_cluster_masks(
        binary_dir=args.cluster_binary_dir,
        boundary_dir=args.cluster_boundary_dir if args.cluster_boundary_dir else None,
        debug_progress=args.debug_progress,
        progress_steps=args.progress_steps,
    )
    print(f"[INFO] loaded cluster masks: {len(projected_clusters)}")
    max_group_size = None if args.max_group_size == 0 else args.max_group_size
    # Find best mask groups
    matches = find_best_mask_groups_for_clusters_greedy(
        sam_masks=sam_masks,
        projected_clusters=projected_clusters,
        image_shape=image_shape,
        min_score=args.min_score,
        min_iou=args.min_iou,
        min_containment=args.min_containment,
        improve_margin=args.improve_margin,
        greedy_min_improve=args.greedy_min_improve,
        max_group_size=max_group_size,
        debug_progress=args.debug_progress,
        progress_steps=args.progress_steps,
    )

    print(f"[INFO] best matches before size filtering: {len(matches)}")

    matches = filter_background_like_group_matches(
        best_matches=matches,
        sam_masks=sam_masks,
        projected_clusters=projected_clusters,
        image_shape=image_shape,
        max_mask_area_ratio=0.2,
        max_cluster_area_ratio=0.3,
        min_score=0.10,
    )

    print(f"[INFO] matches after size filtering: {len(matches)}")
    print_group_matches(matches, top_k=args.top_k)

    # Save text results
    txt_path = os.path.join(args.output_dir, "group_matches.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 130 + "\n")
        f.write(f"{'rank':<6} {'cluster':<8} {'mask_ids':<20} {'score':<10} {'IoU':<10} {'B-IoU':<10} {'Contain':<10} {'BBoxIoU':<10} {'AreaRatio':<10}\n")
        f.write("-" * 130 + "\n")
        for rank, m in enumerate(matches[:args.top_k]):
            f.write(
                f"{rank:<6} {m.cluster_id:<8} {str(list(m.mask_ids)):<20} "
                f"{m.score:<10.4f} {m.iou:<10.4f} {m.boundary_iou:<10.4f} "
                f"{m.containment:<10.4f} {m.bbox_iou:<10.4f} {m.area_ratio:<10.4f}\n"
            )
        f.write("=" * 130 + "\n")

    # Save visualization if image provided
    if image_bgr is not None:
        vis = overlay_mask_group_matches(
            image_bgr=image_bgr,
            sam_masks=sam_masks,
            projected_clusters=projected_clusters,
            matches=matches,
            top_k=args.top_k,
            alpha_mask=0.20,
            alpha_cluster=0.25,
        )
        vis_path = os.path.join(args.output_dir, "group_matches.png")
        cv2.imwrite(vis_path, vis)
        print(f"[INFO] saved visualization: {vis_path}")

    print(f"[INFO] saved text results: {txt_path}")


if __name__ == "__main__":
    main()