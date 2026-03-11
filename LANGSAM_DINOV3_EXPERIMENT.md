# LangSAM + DINOv3 Volume/Mass Experiment

This experiment evaluates the idea from "Physically Guided Visual Mass Estimation from a Single RGB Image" with your current setup:

1. DINOv3 depth map generation
2. LangSAM text-guided masking (SAM2.1 by default)
3. Masked depth integration for volume
4. Optional density-based mass conversion
5. Optional MAE/RMSE/MAPE/R2 if GT CSV is provided

## 1) Environment

```bash
python3.11 -m venv .venv_langsam
source .venv_langsam/bin/activate
pip install --upgrade pip
pip install -e .
```

## 2) Generate depth maps (DINOv3)

Use your existing DINOv3 depth pipeline/repo to create aligned depth maps:

- Input RGB image: `image_name.png` (or jpg/jpeg)
- Depth map: `image_name_depth_raw.png` (16-bit PNG, default scale factor `10000`)

Example (from your DINOv3 repo, not this repo):

```bash
python run_depth_batch.py \
  --input_dir "./spectralwaste" \
  --output_dir "./exp_langsam_dino/depth" \
  --depther_weights "SYNTHMIX" \
  --backbone_weights "<PATH_TO_DINOV3_BACKBONE_WEIGHTS>" \
  --min_depth 0.85 \
  --max_depth 1.0
```

## 3) Run LangSAM + masked volume/mass evaluation

```bash
python run_langsam_dinov3_mass_volume.py \
  --input_dir "./spectralwaste" \
  --depth_dir "./exp_langsam_dino/depth" \
  --output_dir "./exp_langsam_dino/eval" \
  --sam_type "sam2.1_hiera_large" \
  --prompts "waste,garbage,trash,rubbish,debris" \
  --real_world_height 1.0 \
  --real_world_width 1.0 \
  --min_height_threshold 0.005 \
  --ground_percentile 99.9 \
  --density_kg_per_m3 180 \
  --save_debug
```

## 4) Optional GT metrics

Provide CSV with columns (any matching aliases are accepted):

- `image_name`
- `gt_volume_liters` (or `gt_volume_l`, `volume_liters`, `volume`)
- `gt_mass_kg` (optional)

Then add:

```bash
--gt_csv "./ground_truth.csv"
```

## Output

Main result file:

- `exp_langsam_dino/eval/langsam_dinov3_eval_summary.json`

Contains:

- per-image masked and baseline volumes
- mask coverage and prompt hits
- optional mass estimate (with density prior)
- optional volume/mass regression metrics
