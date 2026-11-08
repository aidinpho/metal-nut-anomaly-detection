# Metal Nut Anomaly Detection

A focused industrial vision project for detecting and localizing visual anomalies in **metal nut images**. The Streamlit app lets a user choose PatchCore or EfficientAD-S, upload one image, and view the model decision, anomaly score, heatmap, overlay, and predicted mask.

## Current Scope

- Category: `metal_nut` from MVTec AD.
- Input: one image (`PNG`, `JPG/JPEG`, `BMP`, or `TIFF`).
- App models: PatchCore (recommended) and EfficientAD-S.
- PaDiM is included in the research comparison, but not in the app selector.
- This release is image-only. Video analysis and other object categories are outside the current app scope.
- The app reports when 80% or more of the predicted mask covers the image; this broad response can indicate a scene/background mismatch and is not reliable defect localization.

## Results

The models were evaluated on the same 220/57/58 train/validation/test split.

| Model | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|---|---:|---:|---:|---:|
| **PatchCore** | **0.9942** | **0.9787** | **0.9842** | **0.8328** |
| EfficientAD-S | 0.9439 | 0.9583 | 0.9412 | 0.7562 |
| PaDiM | 0.9710 | 0.9388 | 0.9314 | 0.6313 |

These are exploratory results, not an untouched final benchmark: the official test split was reused during model and preprocessing comparisons. The scores should not be presented as a production performance guarantee.

### PatchCore Results by Defect Type

The following breakdown uses the 58-image test subset from the split above. Image recall is the share of defective images flagged. Pixel AUROC and pixel F1 measure localization against the ground-truth masks. One of the 11 normal images was incorrectly flagged (9.1% false-positive rate).

| Test class | Images | Image recall | Pixel AUROC | Pixel F1 |
|---|---:|---:|---:|---:|
| Bent | 11 | 100.0% | 0.9924 | 0.6115 |
| Color | 12 | 91.7% | 0.9857 | 0.6070 |
| Flip | 13 | 100.0% | 0.9560 | 0.8885 |
| Scratch | 11 | 100.0% | 0.9628 | 0.6004 |

These per-class results show strong image-level detection on this split, while pixel-mask quality varies by defect type. `color` had one missed defective image. This remains a category-specific research demo, not a general-purpose detector for arbitrary metal parts or scenes.

## Run the App

The project was developed with Python 3.10 on Windows. For the tested CUDA 12.6 setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-cu126.txt
pip install -r requirements.txt
streamlit run app.py
```

The app needs the local dataset and checkpoints listed below. They are intentionally excluded from Git because they are large and have separate dataset terms.

## Required Local Assets

- MVTec AD under `datasets/MVTecAD/`, including the `metal_nut` category.
- PatchCore checkpoint at `results/patchcore_metal_nut/Patchcore/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt`.
- EfficientAD-S checkpoint at `results/efficientad_metal_nut/EfficientAd/MVTecAD/metal_nut/v0/weights/lightning/model.ckpt`.
- ImageNette data under `datasets/imagenette/` for EfficientAD-S.

The training and evaluation scripts are listed in [`GITHUB_FILE_SELECTION.md`](GITHUB_FILE_SELECTION.md). Checkpoints can be regenerated from the training scripts if the required data and environment are available.

## Dataset Terms

MVTec AD is distributed under **CC BY-NC-SA 4.0**. Its terms prohibit commercial use without separate permission from MVTec. Review the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) before using or redistributing the dataset or dataset-derived artifacts.

## Repository Contents

The public source selection is intentionally limited to the image app, setup documentation, pinned requirements, and `metal_nut` training/evaluation scripts. Datasets, model weights, generated outputs, temporary files, and unrelated experiments are excluded by `.gitignore`.

No license has been selected for this project's own source code yet. Choose one before inviting reuse or accepting contributions.
