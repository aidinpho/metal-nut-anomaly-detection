from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Patchcore


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
IMAGENETTE_ROOT = PROJECT_ROOT / "datasets" / "imagenette"

CHECKPOINTS = {
    "PatchCore": (
        PROJECT_ROOT / "results" / "patchcore_metal_nut"
        / "Patchcore" / "MVTecAD" / "metal_nut"
        / "v0" / "weights" / "lightning" / "model.ckpt"
    ),
    "EfficientAD-S": (
        PROJECT_ROOT / "results" / "efficientad_metal_nut"
        / "EfficientAd" / "MVTecAD" / "metal_nut"
        / "v0" / "weights" / "lightning" / "model.ckpt"
    ),
}


st.set_page_config(
    page_title="Metal Nut Inspection",
    page_icon="🔍",
    layout="wide",
)
st.title("🔍 Metal Nut Inspection")
st.caption("Image-based industrial anomaly detection")


@st.cache_resource(show_spinner=False)
def load_model_and_engine(model_name):
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category="metal_nut",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        seed=42,
        val_split_mode="from_test",
        val_split_ratio=0.5,
    )
    datamodule.prepare_data()
    datamodule.setup()

    if model_name == "EfficientAD-S":
        model = EfficientAd(
            model_size="small",
            imagenet_dir=IMAGENETTE_ROOT,
            visualizer=False,
        )
    elif model_name == "PatchCore":
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            precision="float32",
            visualizer=False,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = CHECKPOINTS[model_name]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{model_name} checkpoint is missing: {checkpoint}"
        )

    engine = Engine(
        accelerator=accelerator,
        devices=1,
        default_root_dir=PROJECT_ROOT / "results" / "streamlit_runs",
    )
    engine.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpoint,
        verbose=False,
    )
    image_threshold = float(
        model.post_processor.normalized_image_threshold
    )
    pixel_threshold = float(
        model.post_processor.normalized_pixel_threshold
    )
    del datamodule
    return model, engine, image_threshold, pixel_threshold


def flatten_predictions(predictions):
    for item in predictions:
        if isinstance(item, (list, tuple)):
            yield from flatten_predictions(item)
        else:
            yield item


def get_prediction(predictions):
    batches = list(flatten_predictions(predictions))
    if not batches:
        raise RuntimeError("The model did not return a prediction.")

    batch = batches[0]
    score = float(batch.pred_score.detach().cpu().reshape(-1)[0].item())
    label = int(batch.pred_label.detach().cpu().reshape(-1)[0].item())
    anomaly_map = (
        batch.anomaly_map[0]
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    predicted_mask = (
        batch.pred_mask[0]
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return score, label, anomaly_map, predicted_mask


def create_visuals(original_rgb, anomaly_map, predicted_mask):
    height, width = original_rgb.shape[:2]
    anomaly_map = cv2.resize(
        anomaly_map, (width, height), interpolation=cv2.INTER_LINEAR
    )
    predicted_mask = cv2.resize(
        predicted_mask, (width, height), interpolation=cv2.INTER_NEAREST
    )

    low, high = np.percentile(anomaly_map, [1, 99])
    if high > low:
        display_map = np.clip((anomaly_map - low) / (high - low), 0, 1)
    else:
        display_map = np.zeros_like(anomaly_map)

    heatmap = cv2.applyColorMap(
        (display_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb, 0.60, heatmap, 0.40, 0)
    mask = predicted_mask * 255
    return heatmap, overlay, mask


with st.sidebar:
    selected_model = st.selectbox(
        "Detection model",
        ["PatchCore", "EfficientAD-S"],
    )

st.subheader("Analyze an image")
st.caption(
    "For best results, upload one metal nut that fills most of the frame "
    "against a plain background. Busy scenes or other objects may be marked "
    "as anomalous across much of the image."
)
uploaded_image = st.file_uploader(
    "Upload a metal nut image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded_image is not None:
    st.image(uploaded_image.getvalue(), caption="Uploaded image", width=500)

    if st.button("Analyze image", type="primary"):
        try:
            with st.spinner(f"Loading {selected_model} and analyzing image..."):
                model, engine, image_threshold, pixel_threshold = (
                    load_model_and_engine(selected_model)
                )
                suffix = Path(uploaded_image.name).suffix.lower() or ".png"
                with tempfile.TemporaryDirectory() as temp_dir:
                    image_path = Path(temp_dir) / f"input{suffix}"
                    image_path.write_bytes(uploaded_image.getvalue())
                    predictions = engine.predict(
                        model=model,
                        data_path=image_path,
                        return_predictions=True,
                    )
                    score, label, anomaly_map, predicted_mask = get_prediction(
                        predictions
                    )
                    original = np.asarray(
                        Image.open(image_path).convert("RGB")
                    )

            heatmap, overlay, mask = create_visuals(
                original, anomaly_map, predicted_mask
            )
            mask_coverage = float((predicted_mask > 0).mean())
            st.divider()
            if mask_coverage >= 0.80:
                st.warning(
                    "⚠️ BROAD ANOMALY RESPONSE — most of the image is marked. "
                    "This can mean the input differs from the metal_nut training "
                    "images; it does not localize a specific defect reliably. "
                    "Try a single nut on a plain background."
                )
            elif label == 1:
                st.error("⚠️ POTENTIAL DEFECT DETECTED")
            else:
                st.success("✅ NO ANOMALY FLAGGED")

            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Anomaly score", f"{score:.4f}")
            metric2.metric("Image threshold", f"{image_threshold:.4f}")
            metric3.metric("Pixel threshold", f"{pixel_threshold:.4f}")
            metric4.metric("Image marked anomalous", f"{mask_coverage:.1%}")

            st.subheader("Analysis results")
            col1, col2 = st.columns(2)
            col1.image(original, caption="Original image", width="stretch")
            col2.image(heatmap, caption="Anomaly heatmap", width="stretch")
            col3, col4 = st.columns(2)
            col3.image(overlay, caption="Heatmap overlay", width="stretch")
            col4.image(
                mask,
                caption="Predicted anomaly mask",
                clamp=True,
                width="stretch",
            )
            st.caption(
                "Heatmap colors are normalized for display. The model's "
                "classification and mask use its original predictions. "
                "Results are specific to the MVTec AD metal_nut category; "
                "they are not validated for arbitrary scenes or backgrounds."
            )
        except Exception as error:
            st.error(f"Image analysis failed: {error}")
