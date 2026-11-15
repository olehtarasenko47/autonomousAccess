#!/usr/bin/env python3
"""Adaptive 4-Tier Torso Isolation with Foreground Occlusion Subtraction.

Extracted from color_classifier.py (automated-distributed-group-search-system).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

# COCO 17-keypoint upper-body indices
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

DEFAULT_KP_CONF = 0.50


@dataclass(frozen=True)
class TorsoIsolationResult:
    """Output of Stage 3 Adaptive 4-Tier Torso Isolation for a single person."""

    tier: int  # 1, 2, 3, or 4
    tier_name: str
    polygon: np.ndarray  # Shape (N, 2) int32 vertices in full-frame coordinates
    crop_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) clipped to frame bounds
    crop_mask: np.ndarray  # uint8 binary mask (255 = valid torso pixel) within crop_bbox


def _kp_valid(kp: np.ndarray, kp_conf: float = DEFAULT_KP_CONF) -> bool:
    """Return True if keypoint [x, y, conf] is visible above confidence threshold."""
    return bool(kp[2] > kp_conf and (kp[0] > 0 or kp[1] > 0))


def shrink_polygon_toward_centroid(pts: np.ndarray, shrink_ratio: float = 0.15) -> np.ndarray:
    """Shrink polygon vertices by `shrink_ratio` (default 15%) toward its centroid."""
    pts_f = pts.astype(np.float32)
    centroid = np.mean(pts_f, axis=0, keepdims=True)
    scale = 1.0 - shrink_ratio
    shrunk = centroid + scale * (pts_f - centroid)
    return np.round(shrunk).astype(np.int32)


def compute_torso_polygon(
    box: Sequence[float],
    keypoints: np.ndarray | None,
    kp_conf: float = DEFAULT_KP_CONF,
) -> tuple[int, str, np.ndarray]:
    """
    Stage 3: Select the torso sampling region dynamically across 4 tiers
    based on upper-body keypoint visibility (`KP_CONF = 0.5`).
    """
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    if keypoints is not None and len(keypoints) >= 13:
        ls = keypoints[KP_LEFT_SHOULDER]
        rs = keypoints[KP_RIGHT_SHOULDER]
        lh = keypoints[KP_LEFT_HIP]
        rh = keypoints[KP_RIGHT_HIP]

        ls_ok = _kp_valid(ls, kp_conf)
        rs_ok = _kp_valid(rs, kp_conf)
        lh_ok = _kp_valid(lh, kp_conf)
        rh_ok = _kp_valid(rh, kp_conf)

        # Tier 1: Full Torso Visible (LS, RS, LH, RH all > 0.5)
        if ls_ok and rs_ok and lh_ok and rh_ok:
            raw_poly = np.array(
                [
                    [ls[0], ls[1]],
                    [rs[0], rs[1]],
                    [rh[0], rh[1]],
                    [lh[0], lh[1]],
                ],
                dtype=np.float32,
            )
            poly = shrink_polygon_toward_centroid(raw_poly, shrink_ratio=0.15)
            return 1, "T1:FullPoly", poly

        # Tier 2: Staircase / Railing Mode (LS, RS > 0.5, Hips Occluded < 0.5)
        if ls_ok and rs_ok:
            p_ls = np.array([ls[0], ls[1]], dtype=np.float32)
            p_rs = np.array([rs[0], rs[1]], dtype=np.float32)
            w_s = float(np.linalg.norm(p_ls - p_rs))
            if w_s >= 4.0:
                mid_s = 0.5 * (p_ls + p_rs)
                s1 = mid_s + 0.85 * (p_ls - mid_s)
                s2 = mid_s + 0.85 * (p_rs - mid_s)
                drop = 0.8 * w_s
                b1 = np.array([s1[0], min(y2, s1[1] + drop)], dtype=np.float32)
                b2 = np.array([s2[0], min(y2, s2[1] + drop)], dtype=np.float32)
                poly = np.round(np.array([s1, s2, b2, b1])).astype(np.int32)
                return 2, "T2:ShoulderDrop", poly

        # Tier 3: Crowd Side-Occlusion (Only 1 Shoulder + 1 Hip, or 1 Shoulder visible > 0.5)
        if ls_ok or rs_ok:
            vis_s = ls if ls_ok else rs
            sx, sy = float(vis_s[0]), float(vis_s[1])
            center_x = 0.5 * (x1 + x2)
            patch_w = 0.25 * box_w

            if sx < center_x - 1.0:
                px1 = sx
                px2 = min(x2, sx + patch_w)
            elif sx > center_x + 1.0:
                px1 = max(x1, sx - patch_w)
                px2 = sx
            else:
                px1 = max(x1, center_x - 0.5 * patch_w)
                px2 = min(x2, center_x + 0.5 * patch_w)

            py1 = max(y1, sy + 0.03 * box_h)
            if lh_ok or rh_ok:
                vis_h = lh if lh_ok else rh
                hy = float(vis_h[1])
                py2 = hy - 0.05 * box_h if hy > py1 + 4.0 else min(y2, sy + 0.35 * box_h)
            else:
                py2 = min(y2, sy + 0.35 * box_h)

            if py2 <= py1 + 2.0:
                py2 = min(y2, py1 + max(4.0, 0.25 * box_h))

            poly = np.round(
                np.array(
                    [
                        [px1, py1],
                        [px2, py1],
                        [px2, py2],
                        [px1, py2],
                    ],
                    dtype=np.float32,
                )
            ).astype(np.int32)
            return 3, "T3:SidePatch", poly

    # Tier 4: Far-Distance Fallback (< 2 torso keypoints > 0.5)
    cx1 = x1 + 0.35 * box_w
    cx2 = x1 + 0.65 * box_w
    cy1 = y1 + 0.28 * box_h
    cy2 = y1 + 0.52 * box_h
    poly = np.round(
        np.array(
            [
                [cx1, cy1],
                [cx2, cy1],
                [cx2, cy2],
                [cx1, cy2],
            ],
            dtype=np.float32,
        )
    ).astype(np.int32)
    return 4, "T4:ChestCore", poly


def isolate_torso_mask(
    frame_shape: tuple[int, ...],
    person_idx: int,
    all_boxes: Sequence[Sequence[float]],
    keypoints: np.ndarray | None,
    kp_conf: float = DEFAULT_KP_CONF,
    subtract_foreground_occlusion: bool = True,
) -> TorsoIsolationResult:
    """
    Build the isolated binary torso mask inside a tight crop bounding box,
    applying foreground occlusion subtraction from overlapping people closer to the camera.
    """
    frame_h, frame_w = frame_shape[:2]
    box = all_boxes[person_idx]
    tier, tier_name, poly = compute_torso_polygon(box, keypoints, kp_conf=kp_conf)

    poly_clipped = poly.copy()
    poly_clipped[:, 0] = np.clip(poly_clipped[:, 0], 0, max(0, frame_w - 1))
    poly_clipped[:, 1] = np.clip(poly_clipped[:, 1], 0, max(0, frame_h - 1))

    min_x = int(np.clip(np.min(poly_clipped[:, 0]), 0, max(0, frame_w - 1)))
    max_x = int(np.clip(np.max(poly_clipped[:, 0]) + 1, min_x + 1, frame_w))
    min_y = int(np.clip(np.min(poly_clipped[:, 1]), 0, max(0, frame_h - 1)))
    max_y = int(np.clip(np.max(poly_clipped[:, 1]) + 1, min_y + 1, frame_h))

    crop_w = max(1, max_x - min_x)
    crop_h = max(1, max_y - min_y)

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    local_poly = poly_clipped - np.array([[min_x, min_y]], dtype=np.int32)
    cv2.fillPoly(mask, [local_poly], 255)

    # Foreground Occlusion Subtraction:
    # Mask out pixels falling inside the bounding box of another person closer to the camera.
    if subtract_foreground_occlusion and len(all_boxes) > 1:
        self_y2 = float(box[3])
        self_area = max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))
        mask_after_sub = mask.copy()
        for other_idx, other_box in enumerate(all_boxes):
            if other_idx == person_idx:
                continue
            other_y2 = float(other_box[3])
            other_area = max(
                1.0, (float(other_box[2]) - float(other_box[0])) * (float(other_box[3]) - float(other_box[1]))
            )
            both_bottom_clipped = (self_y2 >= frame_h - 4) and (other_y2 >= frame_h - 4)
            is_closer = (other_y2 > self_y2 + 2.0) or (both_bottom_clipped and other_area > self_area * 1.15)
            if is_closer:
                ox1 = int(np.clip(other_box[0] - min_x, 0, crop_w))
                oy1 = int(np.clip(other_box[1] - min_y, 0, crop_h))
                ox2 = int(np.clip(other_box[2] - min_x, 0, crop_w))
                oy2 = int(np.clip(other_box[3] - min_y, 0, crop_h))
                if ox2 > ox1 and oy2 > oy1:
                    mask_after_sub[oy1:oy2, ox1:ox2] = 0

        if np.count_nonzero(mask_after_sub) >= 4:
            mask = mask_after_sub

    return TorsoIsolationResult(
        tier=tier,
        tier_name=tier_name,
        polygon=poly_clipped,
        crop_bbox=(min_x, min_y, max_x, max_y),
        crop_mask=mask,
    )
