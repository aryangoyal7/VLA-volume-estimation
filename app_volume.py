#!/usr/bin/env python3
"""
Simplified Gradio app with 2 modes:
1) Segmentation only
2) Segmentation + Volume

Note: existing app.py remains unchanged.
"""

import os
from functools import lru_cache
from typing import Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lang_sam import LangSAM
from lang_sam.utils import draw_image


# ----------------------------
# Presets (kept simple)
# ----------------------------
PRESET_SAM_TYPE = "sam2.1_hiera_large"
PRESET_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
PRESET_BOX_THRESHOLD = 0.30
PRESET_TEXT_THRESHOLD = 0.25

# Volume presets
PRESET_REAL_WORLD_HEIGHT_M = 1.0
PRESET_REAL_WORLD_WIDTH_M = 1.0
PRESET_MIN_HEIGHT_THRESHOLD_M = 0.005
PRESET_GROUND_PERCENTILE = 99.9
PRESET_TARGET_GROUND_DEPTH_M: Optional[float] = None
PRESET_DEPTH_SCALE_FACTOR = 10000.0

# DINOv3 depth presets (override via environment variables)
def _autodetect_dinov3_repo() -> str:
    """Best-effort local DINOv3 repo detection (for torch.hub source='local')."""
    candidates = [
        os.getenv("DINOV3_REPO_DIR", ""),
        os.path.abspath(os.path.dirname(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    ]
    for cand in candidates:
        if not cand:
            continue
        if os.path.isfile(os.path.join(cand, "hubconf.py")) and os.path.isdir(os.path.join(cand, "dinov3")):
            return cand
    return ""


PRESET_DINOV3_REPO_DIR = _autodetect_dinov3_repo()
PRESET_DINOV3_GITHUB_REPO = os.getenv("DINOV3_GITHUB_REPO", "facebookresearch/dinov3")
PRESET_DINOV3_DEPTHER_WEIGHTS = os.getenv("DINOV3_DEPTHER_WEIGHTS", "SYNTHMIX")
PRESET_DINOV3_BACKBONE_WEIGHTS = os.getenv("DINOV3_BACKBONE_WEIGHTS", "LVD1689M")
PRESET_DINOV3_RESIZE = int(os.getenv("DINOV3_RESIZE", "896"))
PRESET_DINOV3_MIN_DEPTH = float(os.getenv("DINOV3_MIN_DEPTH", "0.85"))
PRESET_DINOV3_MAX_DEPTH = float(os.getenv("DINOV3_MAX_DEPTH", "1.0"))
PRESET_DINOV3_USE_CPU = os.getenv("DINOV3_USE_CPU", "0") == "1"


# ----------------------------
# Cached loaders
# ----------------------------
@lru_cache(maxsize=4)
def get_langsam_model() -> LangSAM:
    return LangSAM(sam_type=PRESET_SAM_TYPE, gdino_model_id=PRESET_GDINO_MODEL_ID)


@lru_cache(maxsize=2)
def get_dinov3_depther(
    repo_dir: str,
    depther_weights: str,
    backbone_weights: str,
    min_depth: float,
    max_depth: float,
    use_cpu: bool,
):
    if repo_dir:
        depther = torch.hub.load(
            repo_dir,
            "dinov3_vit7b16_dd",
            source="local",
            pretrained=False,
            weights=depther_weights,
            backbone_weights=backbone_weights,
            depth_range=(min_depth, max_depth),
        )
    else:
        depther = torch.hub.load(
            PRESET_DINOV3_GITHUB_REPO,
            "dinov3_vit7b16_dd",
            source="github",
            pretrained=False,
            weights=depther_weights,
            backbone_weights=backbone_weights,
            depth_range=(min_depth, max_depth),
        )
    depther.eval()
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    depther = depther.to(device)
    return depther, device


# ----------------------------
# Helpers
# ----------------------------
def make_transform(resize_size: int = 896) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_image_rgb(path: str) -> Tuple[Image.Image, np.ndarray]:
    image = Image.open(path).convert("RGB")
    return image, np.array(image)


def load_16bit_depth_map(path: str, scale_factor: float = PRESET_DEPTH_SCALE_FACTOR) -> np.ndarray:
    depth_array = np.array(Image.open(path), dtype=np.float64)
    return depth_array / scale_factor


def estimate_depth_with_dinov3(image_pil: Image.Image) -> np.ndarray:
    depther, device = get_dinov3_depther(
        repo_dir=PRESET_DINOV3_REPO_DIR,
        depther_weights=PRESET_DINOV3_DEPTHER_WEIGHTS,
        backbone_weights=PRESET_DINOV3_BACKBONE_WEIGHTS,
        min_depth=PRESET_DINOV3_MIN_DEPTH,
        max_depth=PRESET_DINOV3_MAX_DEPTH,
        use_cpu=PRESET_DINOV3_USE_CPU,
    )

    transform = make_transform(PRESET_DINOV3_RESIZE)
    x = transform(image_pil)[None].to(device)
    h0, w0 = image_pil.size[1], image_pil.size[0]

    with torch.inference_mode():
        depth_pred = depther(x)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred,
            size=(h0, w0),
            mode="bilinear",
            align_corners=False,
        )

    return depth_pred[0, 0].detach().cpu().numpy()


def estimate_height_map(
    depth_map: np.ndarray,
    target_ground_depth: Optional[float],
    ground_percentile: float,
) -> Tuple[np.ndarray, float, float]:
    valid = depth_map > 0
    valid_depths = depth_map[valid]
    if valid_depths.size == 0:
        return np.zeros_like(depth_map), 0.0, 1.0

    if ground_percentile < 100.0:
        ground_est = float(np.percentile(valid_depths, ground_percentile))
    else:
        ground_est = float(np.max(valid_depths))

    depth_scale = 1.0
    if target_ground_depth is not None and target_ground_depth > 0 and ground_est > 0:
        depth_scale = float(target_ground_depth) / ground_est

    ground_depth = ground_est * depth_scale
    height_map = ground_depth - (depth_map * depth_scale)
    height_map[~valid] = 0.0
    return height_map, ground_depth, depth_scale


def pixel_area_m2(image_hw: Tuple[int, int], real_h: float, real_w: float) -> float:
    h, w = image_hw
    return (real_h * real_w) / float(h * w)


def resize_mask_if_needed(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if mask.shape == (target_h, target_w):
        return mask
    pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
    return np.array(pil) > 0


def height_to_colormap(height_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = np.where(mask, height_map, 0.0)
    vmax = float(np.max(vis)) if np.max(vis) > 0 else 1.0
    cmap = plt.get_cmap("hot")
    colored = cmap(np.clip(vis / (vmax + 1e-12), 0, 1))[..., :3]
    return (colored * 255).astype(np.uint8)


# ----------------------------
# Inference function
# ----------------------------
def run_app(image_path, text_prompt, mode, depth_map_path, density_kg_per_m3):
    if not image_path:
        raise gr.Error("Please provide an input image.")
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please provide a prompt.")

    image_pil, image_np = load_image_rgb(image_path)

    # Segmentation (always)
    model = get_langsam_model()
    pred = model.predict(
        [image_pil],
        [text_prompt],
        box_threshold=PRESET_BOX_THRESHOLD,
        text_threshold=PRESET_TEXT_THRESHOLD,
    )[0]

    masks = pred.get("masks", np.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=bool))
    boxes = pred.get("boxes", np.zeros((0, 4), dtype=np.float32))
    scores = pred.get("scores", np.zeros((0,), dtype=np.float32))
    labels = pred.get("labels", [])

    if len(masks):
        union_mask = np.any(masks.astype(bool), axis=0)
        overlay = draw_image(image_np, masks, boxes, scores, labels).astype(np.uint8)
    else:
        union_mask = np.zeros(image_np.shape[:2], dtype=bool)
        overlay = image_np.copy()

    # Mode 1: segmentation only
    if mode == "Segmentation only":
        msg = (
            "### Segmentation Only\n"
            f"- Prompt: `{text_prompt}`\n"
            f"- Detections: `{len(masks)}`\n"
            f"- Mask coverage: `{float(np.mean(union_mask)*100.0):.2f}%`\n"
            f"- SAM: `{PRESET_SAM_TYPE}`\n"
            f"- DINO used for segmentation: `{PRESET_GDINO_MODEL_ID}` (GroundingDINO)\n"
        )
        blank = np.zeros_like(image_np)
        return overlay, blank, msg

    # Mode 2: segmentation + volume
    # Default: compute depth using DINOv3.
    # Upload is optional override/fallback.
    depth_error = None
    try:
        depth_map = estimate_depth_with_dinov3(image_pil)
        depth_source = (
            f"DINOv3 depth inference ({'local repo' if PRESET_DINOV3_REPO_DIR else PRESET_DINOV3_GITHUB_REPO})"
        )
        if depth_map_path:
            depth_map = load_16bit_depth_map(depth_map_path, PRESET_DEPTH_SCALE_FACTOR)
            depth_source = f"Uploaded depth override: {os.path.basename(depth_map_path)}"
    except Exception as exc:
        depth_map = None
        depth_error = str(exc)
        if depth_map_path:
            try:
                depth_map = load_16bit_depth_map(depth_map_path, PRESET_DEPTH_SCALE_FACTOR)
                depth_source = f"Uploaded depth fallback: {os.path.basename(depth_map_path)}"
            except Exception as upload_exc:
                depth_error = f"{depth_error} | upload failed: {upload_exc}"
                depth_source = "Unavailable"
        else:
            depth_source = "Unavailable"

    # Graceful fallback if depth is missing/unavailable
    if depth_map is None:
        msg = (
            "### Segmentation + Volume (Depth Unavailable)\n"
            f"- Prompt: `{text_prompt}`\n"
            f"- Detections: `{len(masks)}`\n"
            f"- Prompt mask coverage: `{float(np.mean(union_mask)*100.0):.2f}%`\n"
            f"- SAM: `{PRESET_SAM_TYPE}`\n"
            f"- DINO for segmentation: `{PRESET_GDINO_MODEL_ID}` (GroundingDINO)\n"
            "- Volume/mass: not computed (depth not available)\n"
            "- How to fix:\n"
            "  1. Allow DINOv3 model loading (internet for torch.hub or local repo), or\n"
            "  2. Upload aligned `*_depth_raw.png` in the depth box.\n"
            f"- Depth error: `{depth_error or 'No depth provided'}`\n"
        )
        blank = np.zeros_like(image_np)
        return overlay, blank, msg

    union_mask = resize_mask_if_needed(union_mask, depth_map.shape)
    height_map, ground_depth, depth_scale = estimate_height_map(
        depth_map=depth_map,
        target_ground_depth=PRESET_TARGET_GROUND_DEPTH_M,
        ground_percentile=PRESET_GROUND_PERCENTILE,
    )

    object_mask = union_mask & (height_map > PRESET_MIN_HEIGHT_THRESHOLD_M)
    object_heights = height_map[object_mask]

    if object_heights.size == 0:
        volume_m3 = 0.0
        volume_l = 0.0
        mass_kg = 0.0
        mean_h = 0.0
        max_h = 0.0
    else:
        area_px = pixel_area_m2(
            height_map.shape,
            PRESET_REAL_WORLD_HEIGHT_M,
            PRESET_REAL_WORLD_WIDTH_M,
        )
        volume_m3 = float(np.sum(object_heights) * area_px)
        volume_l = volume_m3 * 1000.0
        mass_kg = volume_m3 * float(density_kg_per_m3)
        mean_h = float(np.mean(object_heights) * 1000.0)
        max_h = float(np.max(object_heights) * 1000.0)

    heat = height_to_colormap(height_map, mask=object_mask)
    msg = (
        "### Segmentation + Volume\n"
        f"- Prompt: `{text_prompt}`\n"
        f"- Detections: `{len(masks)}`\n"
        f"- Prompt mask coverage: `{float(np.mean(union_mask)*100.0):.2f}%`\n"
        f"- Object mask coverage: `{float(np.mean(object_mask)*100.0):.2f}%`\n"
        f"- Estimated volume: `{volume_l:.4f} L`\n"
        f"- Estimated mass: `{mass_kg:.4f} kg` @ `{float(density_kg_per_m3):.1f} kg/m^3`\n"
        f"- Mean/Max height: `{mean_h:.2f}/{max_h:.2f} mm`\n"
        f"- Ground depth: `{ground_depth:.4f} m`\n"
        f"- Depth scale: `{depth_scale:.4f}`\n"
        f"- Depth source: `{depth_source}`\n"
        f"- SAM: `{PRESET_SAM_TYPE}`\n"
        f"- DINO for segmentation: `{PRESET_GDINO_MODEL_ID}` (GroundingDINO)\n"
        f"- DINO for depth (default): `dinov3_vit7b16_dd`\n"
    )
    return overlay, heat, msg


# ----------------------------
# Gradio UI (simple)
# ----------------------------
with gr.Blocks(title="VLA App (Simple)") as blocks:
    gr.Markdown(
        f"""
# VLA App (Simple)

Two modes only:
1. **Segmentation only**
2. **Segmentation + Volume**

Presets in this app:
- SAM: `{PRESET_SAM_TYPE}`
- Segmentation DINO: `{PRESET_GDINO_MODEL_ID}`
- Depth model (default): `dinov3_vit7b16_dd`
- Depth loading source: `{'local repo' if PRESET_DINOV3_REPO_DIR else PRESET_DINOV3_GITHUB_REPO}`
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input RGB Image")
            text_prompt = gr.Textbox(lines=1, label="Text Prompt", value="waste pile")
            mode = gr.Radio(
                choices=["Segmentation only", "Segmentation + Volume"],
                value="Segmentation only",
                label="Mode",
            )
            depth_map_path = gr.Image(
                type="filepath",
                label="Optional 16-bit depth map override/fallback (*_depth_raw.png)",
            )
            density_kg_per_m3 = gr.Number(value=180.0, label="Density (kg/m^3) for mass")
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=1):
            seg_output = gr.Image(type="numpy", label="Segmentation Overlay")
            height_output = gr.Image(type="numpy", label="Masked Height Heatmap")
            metrics_md = gr.Markdown(label="Result")

    run_btn.click(
        fn=run_app,
        inputs=[image_input, text_prompt, mode, depth_map_path, density_kg_per_m3],
        outputs=[seg_output, height_output, metrics_md],
    )


if __name__ == "__main__":
    blocks.launch(server_name="0.0.0.0", server_port=7861)
