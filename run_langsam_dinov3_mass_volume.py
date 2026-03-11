#!/usr/bin/env python3
"""
Evaluate LangSAM (latest SAM2.1 family) + DINOv3 depth for volume/mass estimation.

This script:
1) loads RGB images and aligned DINOv3 depth maps
2) obtains waste masks from LangSAM text prompts
3) computes masked volume using calibrated depth-to-height conversion
4) optionally converts volume to mass using density prior
5) optionally computes MAE/RMSE/MAPE/R2 against a GT CSV
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_image(image_path: Path) -> Image.Image:
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except Exception:
        pass
    return Image.open(image_path).convert("RGB")


def load_16bit_depth_map(file_path: Path, scale_factor: float = 10000.0) -> np.ndarray:
    depth_array = np.array(Image.open(file_path), dtype=np.float64)
    return depth_array / scale_factor


def calculate_pixel_area(
    image_height: int,
    image_width: int,
    real_world_height: float,
    real_world_width: float,
) -> float:
    total_real_area = real_world_height * real_world_width
    return total_real_area / float(image_height * image_width)


def estimate_height_map(
    depth_map: np.ndarray,
    target_ground_depth: Optional[float],
    ground_percentile: float,
) -> Tuple[np.ndarray, float, float]:
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


def _masks_to_numpy(raw_masks: Any) -> np.ndarray:
    if raw_masks is None:
        return np.zeros((0, 0, 0), dtype=bool)

    if hasattr(raw_masks, "detach") and hasattr(raw_masks, "cpu"):
        masks_np = raw_masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(raw_masks)

    if masks_np.size == 0:
        return np.zeros((0, 0, 0), dtype=bool)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if masks_np.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {masks_np.shape}")

    return masks_np > 0


def _extract_masks_from_prediction(pred: Any) -> np.ndarray:
    if isinstance(pred, tuple) and len(pred) >= 1:
        return _masks_to_numpy(pred[0])

    if isinstance(pred, dict):
        return _masks_to_numpy(pred.get("masks"))

    if isinstance(pred, list):
        if not pred:
            return np.zeros((0, 0, 0), dtype=bool)
        if isinstance(pred[0], dict):
            return _masks_to_numpy(pred[0].get("masks"))
        return _masks_to_numpy(pred)

    return _masks_to_numpy(pred)


def infer_langsam_union_mask(
    model: Any,
    image_pil: Image.Image,
    prompts: List[str],
    box_threshold: float,
    text_threshold: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    w, h = image_pil.size
    union_mask = np.zeros((h, w), dtype=bool)
    prompt_hits: Dict[str, int] = {}

    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue

        try:
            pred = model.predict(
                image_pil,
                prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except TypeError:
            pred = model.predict(
                [image_pil],
                [prompt],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

        masks = _extract_masks_from_prediction(pred)
        count = 0 if masks.size == 0 else int(masks.shape[0])
        prompt_hits[prompt] = count
        if count > 0:
            union_mask |= np.any(masks, axis=0)

    return union_mask, prompt_hits


def resize_mask_if_needed(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if mask.shape == (target_h, target_w):
        return mask

    pil_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    pil_mask = pil_mask.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.array(pil_mask) > 0)


def compute_volume_from_height(
    height_map: np.ndarray,
    area_per_pixel: float,
    min_height_threshold: float,
    object_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    valid_object_mask = height_map > min_height_threshold
    if object_mask is not None:
        valid_object_mask &= object_mask

    object_heights = height_map[valid_object_mask]
    if object_heights.size == 0:
        return {
            "total_volume_m3": 0.0,
            "total_volume_liters": 0.0,
            "object_pixels": 0,
            "coverage_percentage": 0.0,
            "mean_height_mm": 0.0,
            "max_height_mm": 0.0,
        }

    total_volume_m3 = float(np.sum(object_heights) * area_per_pixel)
    total_pixels = height_map.size
    object_pixels = int(object_heights.size)

    return {
        "total_volume_m3": total_volume_m3,
        "total_volume_liters": total_volume_m3 * 1000.0,
        "object_pixels": object_pixels,
        "coverage_percentage": (object_pixels / total_pixels) * 100.0,
        "mean_height_mm": float(np.mean(object_heights) * 1000.0),
        "max_height_mm": float(np.max(object_heights) * 1000.0),
    }


def load_gt_table(csv_path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if csv_path is None:
        return {}
    if not csv_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    out: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = (
                row.get("image_name")
                or row.get("filename")
                or row.get("file")
                or row.get("image")
                or ""
            ).strip()
            if not image_name:
                continue
            key = Path(image_name).stem

            gt_volume = (
                row.get("gt_volume_liters")
                or row.get("gt_volume_l")
                or row.get("volume_liters")
                or row.get("volume_l")
                or row.get("volume")
                or ""
            ).strip()
            gt_mass = (
                row.get("gt_mass_kg")
                or row.get("mass_kg")
                or row.get("mass")
                or ""
            ).strip()

            entry: Dict[str, float] = {}
            if gt_volume:
                entry["gt_volume_liters"] = float(gt_volume)
            if gt_mass:
                entry["gt_mass_kg"] = float(gt_mass)
            if entry:
                out[key] = entry

    return out


def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    if not y_true:
        return {}

    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))

    nz = np.abs(yt) > 1e-12
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100.0) if np.any(nz) else float("nan")

    if yt.size >= 2:
        ss_res = float(np.sum(np.square(err)))
        ss_tot = float(np.sum(np.square(yt - np.mean(yt))))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")

    return {"mae": mae, "rmse": rmse, "mape_percent": mape, "r2": r2}


def save_debug_figure(
    image_np: np.ndarray,
    langsam_mask: np.ndarray,
    height_map: np.ndarray,
    masked_height: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(langsam_mask, cmap="gray")
    axes[0, 1].set_title("LangSAM Union Mask")
    axes[0, 1].axis("off")

    im_h = axes[1, 0].imshow(height_map, cmap="hot")
    axes[1, 0].set_title("Height Map (m)")
    axes[1, 0].axis("off")
    plt.colorbar(im_h, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_m = axes[1, 1].imshow(masked_height, cmap="hot")
    axes[1, 1].set_title("Masked Height (m)")
    axes[1, 1].axis("off")
    plt.colorbar(im_m, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangSAM + DINOv3 depth-based volume/mass estimation"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with RGB images")
    parser.add_argument(
        "--depth_dir",
        type=str,
        required=True,
        help="Directory with DINOv3 depth maps (*_depth_raw.png)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--depth_suffix",
        type=str,
        default="_depth_raw.png",
        help="Depth filename suffix relative to image stem",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="waste,garbage,trash,rubbish,debris",
        help="Comma-separated text prompts for LangSAM",
    )
    parser.add_argument(
        "--sam_type",
        type=str,
        default="sam2.1_hiera_large",
        help="LangSAM SAM type (e.g. sam2.1_hiera_large)",
    )
    parser.add_argument("--box_threshold", type=float, default=0.30)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--min_height_threshold", type=float, default=0.005, help="meters")
    parser.add_argument("--real_world_height", type=float, default=1.0, help="meters in FOV")
    parser.add_argument("--real_world_width", type=float, default=1.0, help="meters in FOV")
    parser.add_argument(
        "--target_ground_depth",
        type=float,
        default=None,
        help="Optional metric calibration target depth (meters)",
    )
    parser.add_argument(
        "--ground_percentile",
        type=float,
        default=99.9,
        help="High percentile to estimate ground depth",
    )
    parser.add_argument(
        "--density_kg_per_m3",
        type=float,
        default=None,
        help="If provided, converts predicted volume to mass",
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        default=None,
        help="Optional CSV with columns like image_name, gt_volume_liters, gt_mass_kg",
    )
    parser.add_argument(
        "--image_extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
    )
    parser.add_argument("--max_images", type=int, default=None, help="Debug limit")
    parser.add_argument("--save_debug", action="store_true", help="Save per-image figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    gt_table = load_gt_table(Path(args.gt_csv) if args.gt_csv else None)

    try:
        from lang_sam import LangSAM  # type: ignore
    except Exception as exc:
        logger.error("Failed to import lang_sam. Install requirements_langsam.txt first.")
        raise RuntimeError("lang_sam import failed") from exc

    logger.info("Loading LangSAM model with sam_type=%s", args.sam_type)
    model = LangSAM(sam_type=args.sam_type)

    image_files: List[Path] = []
    for ext in args.image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))
    if args.max_images is not None:
        image_files = image_files[: max(0, args.max_images)]

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    per_image_results: List[Dict[str, Any]] = []
    gt_volume_true: List[float] = []
    gt_volume_pred: List[float] = []
    gt_mass_true: List[float] = []
    gt_mass_pred: List[float] = []

    logger.info("Processing %d images", len(image_files))
    for image_path in image_files:
        image_name = image_path.stem
        depth_path = depth_dir / f"{image_name}{args.depth_suffix}"
        if not depth_path.exists():
            logger.warning("Skipping %s: depth map missing (%s)", image_name, depth_path.name)
            continue

        image_pil = load_image(image_path)
        image_np = np.array(image_pil)
        depth_map = load_16bit_depth_map(depth_path)

        height_map, ground_depth, depth_scale = estimate_height_map(
            depth_map=depth_map,
            target_ground_depth=args.target_ground_depth,
            ground_percentile=args.ground_percentile,
        )

        langsam_mask, prompt_hits = infer_langsam_union_mask(
            model=model,
            image_pil=image_pil,
            prompts=prompts,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        langsam_mask = resize_mask_if_needed(langsam_mask, height_map.shape)

        area_per_pixel = calculate_pixel_area(
            image_height=height_map.shape[0],
            image_width=height_map.shape[1],
            real_world_height=args.real_world_height,
            real_world_width=args.real_world_width,
        )

        masked_volume = compute_volume_from_height(
            height_map=height_map,
            area_per_pixel=area_per_pixel,
            min_height_threshold=args.min_height_threshold,
            object_mask=langsam_mask,
        )
        baseline_volume = compute_volume_from_height(
            height_map=height_map,
            area_per_pixel=area_per_pixel,
            min_height_threshold=args.min_height_threshold,
            object_mask=None,
        )

        mass_kg = None
        if args.density_kg_per_m3 is not None:
            mass_kg = masked_volume["total_volume_m3"] * args.density_kg_per_m3

        gt_entry = gt_table.get(image_name, {})
        if "gt_volume_liters" in gt_entry:
            gt_volume_true.append(float(gt_entry["gt_volume_liters"]))
            gt_volume_pred.append(float(masked_volume["total_volume_liters"]))
        if "gt_mass_kg" in gt_entry and mass_kg is not None:
            gt_mass_true.append(float(gt_entry["gt_mass_kg"]))
            gt_mass_pred.append(float(mass_kg))

        reduction = 0.0
        baseline_l = baseline_volume["total_volume_liters"]
        if baseline_l > 1e-9:
            reduction = (1.0 - masked_volume["total_volume_liters"] / baseline_l) * 100.0

        result = {
            "image_name": image_name,
            "depth_file": depth_path.name,
            "ground_depth_m": ground_depth,
            "depth_scale": depth_scale,
            "prompt_hits": prompt_hits,
            "langsam_mask_pixels": int(np.sum(langsam_mask)),
            "langsam_mask_coverage_percent": float(np.mean(langsam_mask) * 100.0),
            "masked_volume_liters": float(masked_volume["total_volume_liters"]),
            "baseline_volume_liters": float(baseline_volume["total_volume_liters"]),
            "volume_reduction_vs_baseline_percent": float(reduction),
            "masked_coverage_percent": float(masked_volume["coverage_percentage"]),
            "mean_height_mm": float(masked_volume["mean_height_mm"]),
            "max_height_mm": float(masked_volume["max_height_mm"]),
        }
        if mass_kg is not None:
            result["estimated_mass_kg"] = float(mass_kg)
        if gt_entry:
            result["ground_truth"] = gt_entry

        per_image_results.append(result)

        if args.save_debug:
            masked_height = np.where(langsam_mask, height_map, 0.0)
            save_debug_figure(
                image_np=image_np,
                langsam_mask=langsam_mask,
                height_map=height_map,
                masked_height=masked_height,
                out_path=output_dir / "debug" / f"{image_name}_langsam_depth_debug.png",
                title=f"{image_name} | masked_volume={masked_volume['total_volume_liters']:.3f} L",
            )

        logger.info(
            "%s | masked=%.3f L | baseline=%.3f L | mask_cov=%.1f%%",
            image_name,
            masked_volume["total_volume_liters"],
            baseline_volume["total_volume_liters"],
            np.mean(langsam_mask) * 100.0,
        )

    summary: Dict[str, Any] = {
        "config": {
            "prompts": prompts,
            "sam_type": args.sam_type,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "min_height_threshold_m": args.min_height_threshold,
            "real_world_height_m": args.real_world_height,
            "real_world_width_m": args.real_world_width,
            "target_ground_depth_m": args.target_ground_depth,
            "ground_percentile": args.ground_percentile,
            "density_kg_per_m3": args.density_kg_per_m3,
        },
        "images_processed": len(per_image_results),
        "results": per_image_results,
    }

    if gt_volume_true:
        summary["volume_metrics"] = regression_metrics(gt_volume_true, gt_volume_pred)
    if gt_mass_true:
        summary["mass_metrics"] = regression_metrics(gt_mass_true, gt_mass_pred)

    summary_path = output_dir / "langsam_dinov3_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved summary: %s", summary_path)
    if "volume_metrics" in summary:
        logger.info("Volume metrics: %s", summary["volume_metrics"])
    if "mass_metrics" in summary:
        logger.info("Mass metrics: %s", summary["mass_metrics"])

    print("\nLangSAM + DINOv3 evaluation complete.")
    print(f"Processed images: {len(per_image_results)}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
