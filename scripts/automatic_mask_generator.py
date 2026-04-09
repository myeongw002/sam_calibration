#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAM2 masks for one image and postprocess them for calibration."
    )

    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")

    parser.add_argument(
        "--sam2-repo",
        type=str,
        default="/workspace/sam_calibration/sam2",
        help="Path to SAM2 repository root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/sam_calibration/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path relative to repo config root.",
    )

    # SAM2 generation params
    parser.add_argument("--points-per-side", type=int, default=50)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.9)
    parser.add_argument("--stability-score-offset", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument(
        "--use-m2m",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SAM2 mask-to-mask refinement.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--apply-postprocessing",
        action="store_true",
        help="Enable SAM2 build postprocessing behavior.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_torch(device: torch.device):
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return torch.autocast("cuda", dtype=torch.bfloat16)

    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    return nullcontext()


def load_image_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def add_sam2_repo_to_path(repo_dir: Path) -> None:
    if not repo_dir.exists():
        raise FileNotFoundError(f"SAM2 repo not found: {repo_dir}")
    repo_str = str(repo_dir.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def mask_bbox_from_seg(seg: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(seg)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = bbox
    return max(1, x1 - x0 + 1) * max(1, y1 - y0 + 1)


def border_touch_ratio(seg: np.ndarray) -> float:
    h, w = seg.shape
    border = np.zeros_like(seg, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    border_pixels = np.logical_and(seg, border).sum()
    area = seg.sum()
    if area == 0:
        return 1.0
    return float(border_pixels) / float(area)


def mask_iou(seg1: np.ndarray, seg2: np.ndarray) -> float:
    inter = np.logical_and(seg1, seg2).sum()
    union = np.logical_or(seg1, seg2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_containment_ratio(inner: np.ndarray, outer: np.ndarray) -> float:
    inter = np.logical_and(inner, outer).sum()
    area_inner = inner.sum()
    if area_inner == 0:
        return 0.0
    return float(inter) / float(area_inner)


def bbox_fill_ratio(seg: np.ndarray) -> float:
    bbox = mask_bbox_from_seg(seg)
    if bbox is None:
        return 0.0
    a = seg.sum()
    ba = bbox_area(bbox)
    if ba <= 0:
        return 0.0
    return float(a) / float(ba)


def keep_largest_connected_component(seg: np.ndarray) -> np.ndarray:
    seg_u8 = seg.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg_u8, connectivity=8)

    if num_labels <= 1:
        return seg

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == largest_label)
    return out


def fill_mask_holes(seg: np.ndarray) -> np.ndarray:
    seg_u8 = (seg.astype(np.uint8) * 255)
    h, w = seg_u8.shape

    flood = seg_u8.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(seg_u8, flood_inv)
    return filled > 0


def clean_mask(
    seg: np.ndarray,
    close_kernel: int = 7,
    open_kernel: int = 3,
    close_iter: int = 1,
    open_iter: int = 1,
    fill_holes: bool = False,
    keep_largest_cc: bool = False,
) -> np.ndarray:
    seg_u8 = (seg.astype(np.uint8) * 255)

    if close_kernel > 1 and close_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    if open_kernel > 1 and open_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        seg_u8 = cv2.morphologyEx(seg_u8, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    seg_bool = seg_u8 > 0

    if fill_holes:
        seg_bool = fill_mask_holes(seg_bool)

    if keep_largest_cc:
        seg_bool = keep_largest_connected_component(seg_bool)

    return seg_bool


def boundary_band(seg: np.ndarray, band_width: int = 3) -> np.ndarray:
    seg_u8 = (seg.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1))
    eroded = cv2.erode(seg_u8, kernel, iterations=1)
    band = cv2.subtract(seg_u8, eroded)
    return band > 0


def quality_score(ann: dict[str, Any], area_ratio: float) -> float:
    pred_iou = float(ann.get("predicted_iou", 0.0))
    stability = float(ann.get("stability_score", 0.0))

    # calibration용은 너무 작은 것도, 너무 큰 것도 불리함
    # 중간 크기에 약간 가점
    area_prior = 1.0 - abs(area_ratio - 0.08) / 0.08
    area_prior = max(0.0, area_prior)

    return 0.45 * pred_iou + 0.45 * stability + 0.10 * area_prior


def prune_and_refine_masks(
    masks: list[dict[str, Any]],
    image_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    max_border_touch_ratio: float,
    max_aspect_ratio: float,
    min_bbox_fill_ratio: float,
    nested_iou_thresh: float,
    containment_thresh: float,
    close_kernel: int,
    open_kernel: int,
    close_iter: int,
    open_iter: int,
    fill_holes_flag: bool,
    keep_largest_cc_flag: bool,
) -> list[dict[str, Any]]:
    h, w = image_shape
    image_area = h * w

    refined: list[dict[str, Any]] = []

    for ann in masks:
        seg = ann["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)

        seg = clean_mask(
            seg,
            close_kernel=close_kernel,
            open_kernel=open_kernel,
            close_iter=close_iter,
            open_iter=open_iter,
            fill_holes=fill_holes_flag,
            keep_largest_cc=keep_largest_cc_flag,
        )

        area = int(seg.sum())
        if area == 0:
            continue

        area_ratio = area / image_area
        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            continue

        btr = border_touch_ratio(seg)
        if btr > max_border_touch_ratio:
            continue

        bbox = mask_bbox_from_seg(seg)
        if bbox is None:
            continue

        x0, y0, x1, y1 = bbox
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        aspect = max(bw / bh, bh / bw)
        if aspect > max_aspect_ratio:
            continue

        fill_ratio = bbox_fill_ratio(seg)
        if fill_ratio < min_bbox_fill_ratio:
            continue

        ann_new = dict(ann)
        ann_new["segmentation"] = seg
        ann_new["area"] = area
        ann_new["bbox"] = [float(x0), float(y0), float(bw), float(bh)]
        ann_new["_area_ratio"] = area_ratio
        ann_new["_quality"] = quality_score(ann_new, area_ratio)
        refined.append(ann_new)

    refined.sort(key=lambda x: (x["_quality"], x["area"]), reverse=True)

    kept: list[dict[str, Any]] = []

    for ann in refined:
        seg = ann["segmentation"]
        drop = False

        for kept_ann in kept:
            kept_seg = kept_ann["segmentation"]

            iou = mask_iou(seg, kept_seg)
            contain_small = mask_containment_ratio(seg, kept_seg)
            contain_large = mask_containment_ratio(kept_seg, seg)

            # 거의 같은 마스크
            if iou > nested_iou_thresh:
                drop = True
                break

            # 작은 part-mask가 큰 mask에 거의 포함됨
            if contain_small > containment_thresh:
                drop = True
                break

            # 혹시 기존 kept mask가 지금 마스크에 거의 포함되면
            # 더 품질 좋은 현재 마스크로 교체
            if contain_large > containment_thresh and ann["_quality"] > kept_ann["_quality"]:
                kept.remove(kept_ann)
                break

        if not drop:
            kept.append(ann)

    for ann in kept:
        ann.pop("_area_ratio", None)
        ann.pop("_quality", None)

    # object-level merge
    kept = merge_related_masks(kept, image_shape)

    # merged result cleaning
    final_masks: list[dict[str, Any]] = []
    for ann in kept:
        seg = ann["segmentation"].astype(bool)

        seg = clean_mask(
            seg,
            close_kernel=close_kernel,
            open_kernel=open_kernel,
            close_iter=close_iter,
            open_iter=open_iter,
            fill_holes=fill_holes_flag,
            keep_largest_cc=keep_largest_cc_flag,
        )

        if is_background_like_mask(seg, image_shape):
            continue

        area = int(seg.sum())
        if area == 0:
            continue

        ann["segmentation"] = seg
        ann["area"] = area

        bbox = mask_bbox_from_seg(seg)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        ann["bbox"] = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
        final_masks.append(ann)

    return final_masks

def generate_overlay(image_rgb: np.ndarray, masks: list[dict[str, Any]], seed: int) -> np.ndarray:
    if len(masks) == 0:
        return image_rgb.copy()

    rng = np.random.default_rng(seed)
    overlay = image_rgb.astype(np.float32).copy()

    masks_sorted = sorted(masks, key=lambda x: x.get("area", 0), reverse=True)
    for ann in masks_sorted:
        seg = ann["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)

        color = rng.uniform(0, 255, size=3).astype(np.float32)
        alpha = 0.38
        overlay[seg] = (1.0 - alpha) * overlay[seg] + alpha * color

        seg_u8 = (seg.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), thickness=1)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def serialize_mask_metadata(masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ann in masks:
        out.append(
            {
                "area": int(ann.get("area", 0)),
                "bbox": [float(v) for v in ann.get("bbox", [])],
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "point_coords": ann.get("point_coords", []),
                "stability_score": float(ann.get("stability_score", 0.0)),
                "crop_box": [float(v) for v in ann.get("crop_box", [])],
            }
        )
    return out


def save_outputs(
    output_dir: Path,
    image_rgb: np.ndarray,
    masks: list[dict[str, Any]],
    seed: int,
    save_individual_masks: bool = False,
    save_boundary_masks: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_stack = (
        np.stack([m["segmentation"].astype(np.uint8) for m in masks], axis=0)
        if masks
        else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    )
    np.savez_compressed(output_dir / "masks.npz", masks=mask_stack)

    metadata = serialize_mask_metadata(masks)
    with (output_dir / "masks_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    overlay_rgb = generate_overlay(image_rgb, masks, seed)
    cv2.imwrite(str(output_dir / "overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "input.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    if save_individual_masks:
        mask_dir = output_dir / "mask_pngs"
        mask_dir.mkdir(exist_ok=True)
        for idx, ann in enumerate(masks):
            seg = (ann["segmentation"].astype(np.uint8) * 255)
            cv2.imwrite(str(mask_dir / f"mask_{idx:04d}.png"), seg)

    if save_boundary_masks:
        boundary_dir = output_dir / "boundary_mask_pngs"
        boundary_dir.mkdir(exist_ok=True)
        for idx, ann in enumerate(masks):
            b = boundary_band(ann["segmentation"], band_width=3)
            cv2.imwrite(str(boundary_dir / f"boundary_{idx:04d}.png"), (b.astype(np.uint8) * 255))


def save_raw_outputs(output_dir: Path, image_rgb: np.ndarray, raw_masks: list[dict[str, Any]], seed: int) -> None:
    raw_dir = output_dir 
    raw_dir.mkdir(parents=True, exist_ok=True)

    mask_stack = (
        np.stack([m["segmentation"].astype(np.uint8) for m in raw_masks], axis=0)
        if raw_masks
        else np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    )
    np.savez_compressed(raw_dir / "raw_masks.npz", masks=mask_stack)

    metadata = serialize_mask_metadata(raw_masks)
    with (raw_dir / "raw_masks_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    overlay_rgb = generate_overlay(image_rgb, raw_masks, seed)
    cv2.imwrite(str(raw_dir / "raw_overlay.png"), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

def mask_centroid(seg: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def boundary_band(seg: np.ndarray, band_width: int = 3) -> np.ndarray:
    seg_u8 = (seg.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_width * 2 + 1, band_width * 2 + 1)
    )
    eroded = cv2.erode(seg_u8, kernel, iterations=1)
    band = cv2.subtract(seg_u8, eroded)
    return band > 0


def boundary_iou(seg1: np.ndarray, seg2: np.ndarray, band_width: int = 3) -> float:
    b1 = boundary_band(seg1, band_width=band_width)
    b2 = boundary_band(seg2, band_width=band_width)
    inter = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def centroid_distance(seg1: np.ndarray, seg2: np.ndarray) -> float:
    c1 = mask_centroid(seg1)
    c2 = mask_centroid(seg2)
    if c1 is None or c2 is None:
        return 1e9
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

def should_merge_masks(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    image_shape: tuple[int, int],
    contain_thresh: float = 0.70,
    biou_thresh: float = 0.20,
    boundary_iou_thresh: float = 0.18,
    centroid_dist_thresh: float = 60.0,
) -> bool:
    bbox_a = mask_bbox_from_seg(seg_a)
    bbox_b = mask_bbox_from_seg(seg_b)
    if bbox_a is None or bbox_b is None:
        return False

    # bbox IoU
    def _bbox_iou(b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        ix1, iy1 = max(x11, x21), max(y11, y21)
        ix2, iy2 = min(x12, x22), min(y12, y22)
        iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
        inter = iw * ih
        a1 = max(1, x12 - x11 + 1) * max(1, y12 - y11 + 1)
        a2 = max(1, x22 - x21 + 1) * max(1, y22 - y21 + 1)
        union = a1 + a2 - inter
        return 0.0 if union <= 0 else float(inter) / float(union)

    contain_ab = mask_containment_ratio(seg_a, seg_b)
    contain_ba = mask_containment_ratio(seg_b, seg_a)
    biou = _bbox_iou(bbox_a, bbox_b)
    bi = boundary_iou(seg_a, seg_b, band_width=3)
    cd = centroid_distance(seg_a, seg_b)

    # 작은 part-mask 흡수 또는 매우 가까운 분할 조각 병합
    if max(contain_ab, contain_ba) > contain_thresh and biou > biou_thresh:
        return True
    if bi > boundary_iou_thresh and cd < centroid_dist_thresh:
        return True
    return False


def merge_mask_group(mask_group: list[dict[str, Any]]) -> dict[str, Any]:
    seg = np.zeros_like(mask_group[0]["segmentation"], dtype=bool)
    best = max(mask_group, key=lambda x: (x.get("predicted_iou", 0.0), x.get("stability_score", 0.0)))
    for ann in mask_group:
        seg |= ann["segmentation"].astype(bool)

    area = int(seg.sum())
    bbox = mask_bbox_from_seg(seg)
    if bbox is None:
        bbox_out = [0.0, 0.0, 0.0, 0.0]
    else:
        x0, y0, x1, y1 = bbox
        bbox_out = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]

    merged = dict(best)
    merged["segmentation"] = seg
    merged["area"] = area
    merged["bbox"] = bbox_out
    return merged


def merge_related_masks(
    masks: list[dict[str, Any]],
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    used = [False] * len(masks)
    merged_out: list[dict[str, Any]] = []

    for i in range(len(masks)):
        if used[i]:
            continue
        group = [masks[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            current_union = merge_mask_group(group)["segmentation"]

            for j in range(len(masks)):
                if used[j]:
                    continue
                if should_merge_masks(current_union, masks[j]["segmentation"], image_shape):
                    group.append(masks[j])
                    used[j] = True
                    changed = True

        merged_out.append(merge_mask_group(group))

    return merged_out

def is_background_like_mask(seg: np.ndarray, image_shape: tuple[int, int]) -> bool:
    h, w = image_shape
    bbox = mask_bbox_from_seg(seg)
    if bbox is None:
        return True

    x0, y0, x1, y1 = bbox
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    area_ratio = seg.sum() / float(h * w)

    # 너무 넓고, 화면 상/하단을 강하게 차지하는 mask 제거
    if area_ratio > 0.18 and bw / w > 0.45:
        if y0 < 0.12 * h or y1 > 0.88 * h:
            return True

    # 도로처럼 매우 넓고 낮은 mask
    if bw / w > 0.55 and bh / h < 0.35 and y1 > 0.65 * h:
        return True

    return False


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    sam2_repo = Path(args.sam2_repo)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = resolve_device(args.device)
    print(f"[INFO] using device: {device}")

    add_sam2_repo_to_path(sam2_repo)

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    image_rgb = load_image_rgb(image_path)

    autocast_ctx = setup_torch(device)

    with autocast_ctx:
        print("[INFO] building SAM2 model...")
        model = build_sam2(
            config_file=args.model_cfg,
            ckpt_path=args.checkpoint,
            device=str(device),
            apply_postprocessing=args.apply_postprocessing,
        )

        print("[INFO] creating mask generator...")
        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            stability_score_offset=args.stability_score_offset,
            crop_n_layers=args.crop_n_layers,
            box_nms_thresh=args.box_nms_thresh,
            crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
            min_mask_region_area=args.min_mask_region_area,
            use_m2m=args.use_m2m,
            output_mode="binary_mask",
        )

        print("[INFO] generating raw masks...")
        raw_masks = mask_generator.generate(image_rgb)

    print(f"[INFO] generated {len(raw_masks)} raw masks")

    print("[INFO] saving raw masks...")
    save_raw_outputs(output_dir, image_rgb, raw_masks, seed = 42)


if __name__ == "__main__":
    main()