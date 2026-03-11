#!/usr/bin/env python3
"""
Conversational VLA loop for text-guided segmentation and optional volume/mass estimation.
"""

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from lang_sam import LangSAM
from lang_sam.utils import draw_image


def load_rgb(path: Path) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(path).convert("RGB")
    return image, np.array(image)


def load_depth(path: Path, scale_factor: float = 10000.0) -> np.ndarray:
    depth = np.array(Image.open(path), dtype=np.float64)
    return depth / scale_factor


def estimate_height_map(
    depth_map: np.ndarray,
    target_ground_depth: Optional[float],
    ground_percentile: float,
) -> tuple[np.ndarray, float, float]:
    valid_mask = depth_map > 0
    valid_depths = depth_map[valid_mask]
    if valid_depths.size == 0:
        return np.zeros_like(depth_map), 0.0, 1.0

    percentile = max(0.0, min(100.0, ground_percentile))
    if percentile < 100.0:
        ground_depth_est = float(np.percentile(valid_depths, percentile))
    else:
        ground_depth_est = float(np.max(valid_depths))

    depth_scale = 1.0
    if target_ground_depth is not None and target_ground_depth > 0 and ground_depth_est > 0:
        depth_scale = float(target_ground_depth) / ground_depth_est

    ground_depth = ground_depth_est * depth_scale
    height_map = ground_depth - (depth_map * depth_scale)
    height_map[~valid_mask] = 0.0
    return height_map, ground_depth, depth_scale


def pixel_area_m2(image_hw: tuple[int, int], real_world_h: float, real_world_w: float) -> float:
    h, w = image_hw
    return (real_world_h * real_world_w) / float(h * w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversational VLA using LangSAM + optional depth.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument(
        "--depth",
        type=str,
        default=None,
        help="Optional aligned 16-bit depth map (meters stored with scale_factor).",
    )
    parser.add_argument("--sam_type", type=str, default="sam2.1_hiera_large")
    parser.add_argument("--gdino_model_id", type=str, default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--box_threshold", type=float, default=0.30)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--min_height_threshold", type=float, default=0.005, help="meters")
    parser.add_argument("--real_world_height", type=float, default=1.0, help="meters in FOV")
    parser.add_argument("--real_world_width", type=float, default=1.0, help="meters in FOV")
    parser.add_argument("--target_ground_depth", type=float, default=None, help="meters")
    parser.add_argument("--ground_percentile", type=float, default=99.9)
    parser.add_argument("--density_kg_per_m3", type=float, default=180.0)
    parser.add_argument("--depth_scale_factor", type=float, default=10000.0)
    parser.add_argument("--save_dir", type=str, default="outputs/conversational")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    image_pil, image_np = load_rgb(image_path)

    depth_map = None
    if args.depth:
        depth_map = load_depth(Path(args.depth), scale_factor=args.depth_scale_factor)

    model = LangSAM(
        sam_type=args.sam_type,
        gdino_model_id=args.gdino_model_id,
    )

    print("\nConversational VLA ready.")
    print("Type natural-language prompts like: 'waste pile', 'plastic bottles', 'cardboard'.")
    print("Commands: /help, /save, /exit")
    if depth_map is not None:
        print("Depth map loaded: volume/mass estimates will be shown per turn.")
    else:
        print("No depth map loaded: running segmentation-only conversation.")

    last_overlay = image_np.copy()
    turn_idx = 0

    while True:
        user_text = input("\nVLA> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Session ended.")
            break
        if user_text.lower() == "/help":
            print("Enter object/material prompts directly.")
            print("Example: waste, garbage bag, plastic, cardboard, metal can")
            print("Use /save to write current overlay PNG.")
            print("Use /exit to stop.")
            continue
        if user_text.lower().startswith("/save"):
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = save_dir / f"turn_{turn_idx:03d}_{ts}.png"
            Image.fromarray(last_overlay).save(out)
            print(f"Saved: {out}")
            continue

        turn_idx += 1
        prompt = user_text
        result = model.predict([image_pil], [prompt], args.box_threshold, args.text_threshold)[0]

        masks = result.get("masks", np.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=bool))
        boxes = result.get("boxes", np.zeros((0, 4), dtype=np.float32))
        scores = result.get("scores", np.zeros((0,), dtype=np.float32))
        labels = result.get("labels", [])

        if len(masks) == 0:
            last_overlay = image_np.copy()
            print("No detections for this prompt.")
            continue

        union_mask = np.any(masks.astype(bool), axis=0)
        last_overlay = draw_image(image_np, masks, boxes, scores, labels).astype(np.uint8)

        coverage = float(np.mean(union_mask) * 100.0)
        print(f"Detections: {len(masks)} | Mask coverage: {coverage:.2f}%")

        if depth_map is None:
            continue

        if depth_map.shape != union_mask.shape:
            print(f"Depth/image mismatch: depth={depth_map.shape}, mask={union_mask.shape}")
            print("Skipping volume/mass for this turn.")
            continue

        height_map, ground_depth, depth_scale = estimate_height_map(
            depth_map=depth_map,
            target_ground_depth=args.target_ground_depth,
            ground_percentile=args.ground_percentile,
        )
        object_mask = union_mask & (height_map > args.min_height_threshold)
        object_heights = height_map[object_mask]

        if object_heights.size == 0:
            print("No positive-height region after thresholding.")
            continue

        area_per_px = pixel_area_m2(height_map.shape, args.real_world_height, args.real_world_width)
        volume_m3 = float(np.sum(object_heights) * area_per_px)
        volume_l = volume_m3 * 1000.0
        mass_kg = volume_m3 * args.density_kg_per_m3

        print(
            "Volume {:.3f} L | Mass {:.3f} kg | Mean height {:.1f} mm | Ground {:.3f} m | Scale {:.4f}".format(
                volume_l,
                mass_kg,
                float(np.mean(object_heights) * 1000.0),
                ground_depth,
                depth_scale,
            )
        )


if __name__ == "__main__":
    main()
