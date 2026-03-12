#!/usr/bin/env python3
"""
Gradio app for text-guided segmentation + volume/mass estimation.

This app does NOT modify existing app.py.
"""

import os
from functools import lru_cache
from typing import Any, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lang_sam import LangSAM, SAM_MODELS
from lang_sam.utils import draw_image


# ----------------------------
# Model loading helpers
# ----------------------------
@lru_cache(maxsize=8)
def get_langsam_model(sam_type: str, gdino_model_id: str) -> LangSAM:
    return LangSAM(sam_type=sam_type, gdino_model_id=gdino_model_id)


@lru_cache(maxsize=4)
def get_dinov3_depther(
    repo_dir: str,
    depther_weights: str,
    backbone_weights: str,
    min_depth: float,
    max_depth: float,
    use_cpu: bool,
):
    depther = torch.hub.load(
        repo_dir,
        "dinov3_vit7b16_dd",
        source="local",
        pretrained=False,
        weights=depther_weights,
        backbone_weights=backbone_weights,
        depth_range=(min_depth, max_depth),
    )
    depther.eval()

    if use_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    depther = depther.to(device)
    return depther, device


# ----------------------------
# Geometry helpers
# ----------------------------
def load_image_rgb(path: str) -> Tuple[Image.Image, np.ndarray]:
    image = Image.open(path).convert("RGB")
    return image, np.array(image)


def load_16bit_depth_map(path: str, scale_factor: float = 10000.0) -> np.ndarray:
    depth_array = np.array(Image.open(path), dtype=np.float64)
    return depth_array / scale_factor


def estimate_height_map(
    depth_map: np.ndarray,
    target_ground_depth: Optional[float],
    ground_percentile: float,
) -> Tuple[np.ndarray, float, float]:
    valid = depth_map > 0
    valid_depths = depth_map[valid]
    if valid_depths.size == 0:
        return np.zeros_like(depth_map), 0.0, 1.0

    percentile = max(0.0, min(100.0, ground_percentile))
    if percentile < 100.0:
        ground_est = float(np.percentile(valid_depths, percentile))
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


def height_to_colormap(height_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        vis = np.where(mask, height_map, 0.0)
    else:
        vis = height_map

    vmax = float(np.max(vis)) if np.max(vis) > 0 else 1.0
    cmap = plt.get_cmap("hot")
    colored = cmap(np.clip(vis / (vmax + 1e-12), 0, 1))[..., :3]
    return (colored * 255).astype(np.uint8)


# ----------------------------
# DINOv3 depth inference
# ----------------------------
def make_transform(resize_size: int = 896) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def estimate_depth_with_dinov3(
    image_pil: Image.Image,
    repo_dir: str,
    depther_weights: str,
    backbone_weights: str,
    resize: int,
    min_depth: float,
    max_depth: float,
    use_cpu: bool,
) -> np.ndarray:
    depther, device = get_dinov3_depther(
        repo_dir=repo_dir,
        depther_weights=depther_weights,
        backbone_weights=backbone_weights,
        min_depth=min_depth,
        max_depth=max_depth,
        use_cpu=use_cpu,
    )
    transform = make_transform(resize)
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


# ----------------------------
# Main inference
# ----------------------------
def infer_with_volume(
    image_path,
    text_prompt,
    sam_type,
    gdino_model_id,
    box_threshold,
    text_threshold,
    depth_source,
    depth_map_path,
    dinov3_repo_dir,
    depther_weights,
    backbone_weights,
    dino_resize,
    dino_min_depth,
    dino_max_depth,
    dino_use_cpu,
    depth_scale_factor,
    real_world_height,
    real_world_width,
    min_height_threshold,
    ground_percentile,
    target_ground_depth,
    density_kg_per_m3,
):
    if not image_path:
        raise gr.Error("Please provide an input image.")
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please provide a prompt.")

    image_pil, image_np = load_image_rgb(image_path)

    # 1) Text-guided segmentation
    model = get_langsam_model(sam_type=sam_type, gdino_model_id=gdino_model_id)
    pred = model.predict(
        [image_pil],
        [text_prompt],
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
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

    # 2) Depth source
    depth_map = None
    depth_note = "No depth source selected"

    if depth_source == "Upload depth map" and depth_map_path:
        depth_map = load_16bit_depth_map(depth_map_path, scale_factor=float(depth_scale_factor))
        depth_note = f"Uploaded 16-bit depth map: {os.path.basename(depth_map_path)}"

    if depth_source == "DINOv3 inference":
        if not dinov3_repo_dir:
            raise gr.Error("Set DINOv3 repo directory for depth inference mode.")
        if not depther_weights:
            raise gr.Error("Set depther weights (e.g., SYNTHMIX).")
        if not backbone_weights:
            raise gr.Error("Set DINOv3 backbone weights path for depth inference mode.")

        depth_map = estimate_depth_with_dinov3(
            image_pil=image_pil,
            repo_dir=dinov3_repo_dir,
            depther_weights=depther_weights,
            backbone_weights=backbone_weights,
            resize=int(dino_resize),
            min_depth=float(dino_min_depth),
            max_depth=float(dino_max_depth),
            use_cpu=bool(dino_use_cpu),
        )
        depth_note = "Depth estimated with DINOv3 (dinov3_vit7b16_dd)"

    # 3) If no depth, return segmentation-only
    if depth_map is None:
        msg = (
            f"### Segmentation only\n"
            f"- Prompt: `{text_prompt}`\n"
            f"- Detections: `{len(masks)}`\n"
            f"- Mask coverage: `{float(np.mean(union_mask)*100.0):.2f}%`\n"
            f"- Segmentation DINO backbone: `{gdino_model_id}` (GroundingDINO)\n"
            f"- Depth source: not provided\n"
            f"\nTo get volume, upload aligned depth map or enable DINOv3 inference."
        )
        blank = np.zeros_like(image_np)
        return overlay, blank, msg

    # 4) Compute height and masked volume
    union_mask = resize_mask_if_needed(union_mask, depth_map.shape)
    height_map, ground_depth, depth_scale = estimate_height_map(
        depth_map=depth_map,
        target_ground_depth=(None if target_ground_depth in (None, "") else float(target_ground_depth)),
        ground_percentile=float(ground_percentile),
    )

    object_mask = union_mask & (height_map > float(min_height_threshold))
    object_heights = height_map[object_mask]

    if object_heights.size == 0:
        volume_m3 = 0.0
        volume_l = 0.0
        mass_kg = 0.0
        mean_h = 0.0
        max_h = 0.0
    else:
        area_px = pixel_area_m2(height_map.shape, float(real_world_height), float(real_world_width))
        volume_m3 = float(np.sum(object_heights) * area_px)
        volume_l = volume_m3 * 1000.0
        mass_kg = volume_m3 * float(density_kg_per_m3)
        mean_h = float(np.mean(object_heights) * 1000.0)
        max_h = float(np.max(object_heights) * 1000.0)

    heat = height_to_colormap(height_map, mask=object_mask)

    summary = (
        f"### Segmentation + Volume\n"
        f"- Prompt: `{text_prompt}`\n"
        f"- Detections: `{len(masks)}`\n"
        f"- Prompt mask coverage: `{float(np.mean(union_mask)*100.0):.2f}%`\n"
        f"- Object mask coverage (height-thresholded): `{float(np.mean(object_mask)*100.0):.2f}%`\n"
        f"- Estimated volume: `{volume_l:.4f} L`\n"
        f"- Estimated mass: `{mass_kg:.4f} kg` @ `{float(density_kg_per_m3):.1f} kg/m^3`\n"
        f"- Mean/Max height: `{mean_h:.2f} / {max_h:.2f} mm`\n"
        f"- Ground depth: `{ground_depth:.4f} m`\n"
        f"- Depth scale: `{depth_scale:.4f}`\n"
        f"- Depth source: `{depth_note}`\n"
        f"- Segmentation DINO backbone: `{gdino_model_id}` (GroundingDINO)\n"
    )
    return overlay, heat, summary


# ----------------------------
# Gradio app
# ----------------------------
with gr.Blocks(title="VLA Segmentation + Volume") as blocks:
    gr.Markdown(
        """
# VLA App (Segmentation + Volume)

This app keeps prompt-based segmentation and adds volume/mass estimates.

- Segmentation stack: **GroundingDINO + SAM2.1** (`LangSAM`)
- Depth for volume: **uploaded depth map** or **DINOv3 depth inference**

`app.py` remains unchanged; this is a separate app.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input RGB Image")
            text_prompt = gr.Textbox(lines=1, label="Text Prompt", value="waste pile")
            run_btn = gr.Button("Run Segmentation + Volume", variant="primary")

            with gr.Accordion("Segmentation Settings", open=False):
                sam_type = gr.Dropdown(
                    choices=list(SAM_MODELS.keys()),
                    label="SAM Model",
                    value="sam2.1_hiera_large",
                )
                gdino_model_id = gr.Textbox(
                    label="GroundingDINO Model ID",
                    value="IDEA-Research/grounding-dino-base",
                )
                box_threshold = gr.Slider(0.0, 1.0, value=0.30, label="Box Threshold")
                text_threshold = gr.Slider(0.0, 1.0, value=0.25, label="Text Threshold")

            with gr.Accordion("Depth Source", open=True):
                depth_source = gr.Radio(
                    choices=["Upload depth map", "DINOv3 inference"],
                    value="Upload depth map",
                    label="Depth Mode",
                )
                depth_map_path = gr.Image(
                    type="filepath",
                    label="Depth map (16-bit PNG, *_depth_raw.png)",
                )

                dinov3_repo_dir = gr.Textbox(
                    label="DINOv3 Repo Directory (for depth mode)",
                    value="",
                    placeholder="/abs/path/to/dinov3/repo",
                )
                depther_weights = gr.Textbox(
                    label="DINOv3 Depther Weights",
                    value="SYNTHMIX",
                )
                backbone_weights = gr.Textbox(
                    label="DINOv3 Backbone Weights Path",
                    value="",
                    placeholder="/abs/path/to/dinov3_vit7b16_...pth",
                )
                dino_resize = gr.Slider(256, 1536, value=896, step=16, label="DINO Resize")
                dino_min_depth = gr.Number(value=0.85, label="DINO Min Depth (m)")
                dino_max_depth = gr.Number(value=1.0, label="DINO Max Depth (m)")
                dino_use_cpu = gr.Checkbox(value=False, label="Force CPU for DINOv3")
                depth_scale_factor = gr.Number(value=10000.0, label="Depth scale factor (for uploaded depth)")

            with gr.Accordion("Volume/Mass Settings", open=False):
                real_world_height = gr.Number(value=1.0, label="Real-world coverage height (m)")
                real_world_width = gr.Number(value=1.0, label="Real-world coverage width (m)")
                min_height_threshold = gr.Number(value=0.005, label="Min height threshold (m)")
                ground_percentile = gr.Number(value=99.9, label="Ground percentile")
                target_ground_depth = gr.Textbox(value="", label="Target ground depth (m, optional)")
                density_kg_per_m3 = gr.Number(value=180.0, label="Density (kg/m^3)")

        with gr.Column(scale=1):
            seg_output = gr.Image(type="numpy", label="Segmentation Overlay")
            height_output = gr.Image(type="numpy", label="Masked Height Heatmap")
            metrics_md = gr.Markdown(label="Metrics")

    run_btn.click(
        fn=infer_with_volume,
        inputs=[
            image_input,
            text_prompt,
            sam_type,
            gdino_model_id,
            box_threshold,
            text_threshold,
            depth_source,
            depth_map_path,
            dinov3_repo_dir,
            depther_weights,
            backbone_weights,
            dino_resize,
            dino_min_depth,
            dino_max_depth,
            dino_use_cpu,
            depth_scale_factor,
            real_world_height,
            real_world_width,
            min_height_threshold,
            ground_percentile,
            target_ground_depth,
            density_kg_per_m3,
        ],
        outputs=[seg_output, height_output, metrics_md],
    )


if __name__ == "__main__":
    blocks.launch(server_name="0.0.0.0", server_port=7861)
