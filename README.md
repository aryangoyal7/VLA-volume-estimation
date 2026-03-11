# VLA Volume Estimation

Vision-language pipeline for waste segmentation and volume/mass estimation:

- `LangSAM` (GroundingDINO + SAM2.1) for text-guided masks
- DINOv3 depth maps (generated separately) for geometry
- Calibrated depth-to-height integration for volume and mass

This repo is conversational: you can run interactive prompt turns on one image (`conversational_vla.py`) and get per-turn segmentation + optional volume/mass estimates.

![person.png](/assets/outputs/person.png)

## Defaults in this repo

- SAM: `sam2.1_hiera_large`
- GroundingDINO: `IDEA-Research/grounding-dino-base`

These defaults are set in:

- `lang_sam/lang_sam.py`
- `lang_sam/models/gdino.py`
- `lang_sam/server.py`
- `app.py`

## Setup (GPU recommended)

Prerequisites:

- Python 3.10+
- NVIDIA GPU + CUDA (recommended for SAM2.1 large)

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you need a specific CUDA torch wheel, install torch first, then run `pip install -e .`.

## Run Modes

### 1) Gradio app (interactive)

```bash
python app.py
```

Open: `http://0.0.0.0:8000/gradio`

### 2) Conversational VLA (CLI)

This is the multi-turn workflow.

```bash
python conversational_vla.py \
  --image /path/to/image.png \
  --depth /path/to/aligned_depth_raw.png \
  --sam_type sam2.1_hiera_large \
  --gdino_model_id IDEA-Research/grounding-dino-base \
  --real_world_height 1.0 \
  --real_world_width 1.0 \
  --density_kg_per_m3 180
```

Then type prompts like:

- `mixed waste pile`
- `plastic bottles`
- `cardboard`

Commands:

- `/help`
- `/save`
- `/exit`

If `--depth` is passed, each turn also reports estimated volume and mass.

### 3) Batch eval (DINOv3 depth + LangSAM masks)

```bash
python run_langsam_dinov3_mass_volume.py \
  --input_dir /path/to/rgb_dir \
  --depth_dir /path/to/depth_dir \
  --output_dir /path/to/output_dir \
  --sam_type sam2.1_hiera_large \
  --prompts "waste,garbage,trash,rubbish,debris" \
  --real_world_height 1.0 \
  --real_world_width 1.0 \
  --min_height_threshold 0.005 \
  --ground_percentile 99.9 \
  --density_kg_per_m3 180 \
  --save_debug
```

With ground truth:

```bash
--gt_csv /path/to/ground_truth.csv
```

This computes MAE/RMSE/MAPE/R2 for volume and optional mass.

## DINOv3 depth notes

This repo consumes aligned depth maps (`*_depth_raw.png`).  
Generate these from your DINOv3 pipeline first, then run the batch eval here.

Detailed workflow: `LANGSAM_DINOV3_EXPERIMENT.md`

## Included image datasets

This repo now includes the image-only subsets from your previous project:

- `datasets/spectralwaste/` (5 images)
- `datasets/zero_dataset/` (4 images)

## Library usage

```python
from PIL import Image
from lang_sam import LangSAM

model = LangSAM(
    sam_type="sam2.1_hiera_large",
    gdino_model_id="IDEA-Research/grounding-dino-base",
)
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
results = model.predict([image_pil], ["wheel"])
```

## Acknowledgments

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything)
- [LitServe](https://github.com/Lightning-AI/LitServe/)
- [Supervision](https://github.com/roboflow/supervision)

## License

Apache 2.0
