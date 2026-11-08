
from pathlib import Path
import csv

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.metrics import AUROC, F1Score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "patchcore_metal_nut"
    / "Patchcore"
    / "MVTecAD"
    / "metal_nut"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

CATEGORIES = ["bent", "color", "flip", "scratch"]


def calculate_metrics(samples, device):

    metrics = {
        "image_AUROC": AUROC(
            fields=["pred_score", "gt_label"]
        ),
        "image_F1Score": F1Score(
            fields=["pred_label", "gt_label"]
        ),
        "pixel_AUROC": AUROC(
            fields=["anomaly_map", "gt_mask"]
        ),
        "pixel_F1Score": F1Score(
            fields=["pred_mask", "gt_mask"]
        ),
    }

    for metric in metrics.values():
        metric.to(device)

    for sample in samples:
        for metric in metrics.values():
            metric.update(sample)

    return {
        name: metric.compute().item()
        for name, metric in metrics.items()
    }


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    print("GPU:", torch.cuda.get_device_name(0))

    # 1. Initialize dataset with the same split as EfficientAD.
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

    print("\nDataset information:")
    print("Training:", len(datamodule.train_data))
    print("Validation:", len(datamodule.val_data))
    print("Test:", len(datamodule.test_data))

    # 2. Initialize PatchCore with training configuration.
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        precision="float32",
    )

    # 3. Initialize Engine.
    engine = Engine(
        accelerator="gpu",
        devices=1,
    )

    # 4. Calibrate thresholds using validation data.
    print("\nCalibrating PatchCore on validation data...")

    validation_results = engine.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=CHECKPOINT_PATH,
    )

    print("Validation results:", validation_results)

    print("\nUpdated thresholds:")
    print(
        "Image:",
        float(model.post_processor.image_threshold)
    )
    print(
        "Pixel:",
        float(model.post_processor.pixel_threshold)
    )

    # 5. Predict on the independent test split.
    print("\nGenerating PatchCore predictions...")

    predictions = engine.predict(
        model=model,
        dataloaders=datamodule.test_dataloader(),
        return_predictions=True,
    )

    if not predictions:
        raise RuntimeError("No predictions were returned.")

    # 6. Group predictions by defect category.
    grouped = {
        category: []
        for category in ["bent", "color", "flip", "good", "scratch"]
    }

    for prediction in predictions:
        category = Path(prediction.image_path[0]).parent.name.lower()
        grouped[category].append(prediction)

    print("\nTest image counts:")

    for category, samples in grouped.items():
        print(f"{category}: {len(samples)}")

    device = predictions[0].pred_score.device

    # 7. Overall evaluation.
    rows = []

    overall = calculate_metrics(predictions, device)

    rows.append({
        "comparison": "overall",
        **overall,
    })

    print("\nOVERALL PATCHCORE RESULTS")

    for name, value in overall.items():
        print(f"{name}: {value:.4f}")

    # 8. Category-wise evaluation.
    print("\nCATEGORY-WISE RESULTS")
    print("-" * 65)

    from sklearn.metrics import confusion_matrix

    for category in CATEGORIES:

        samples = grouped["good"] + grouped[category]

        results = calculate_metrics(samples, device)

        rows.append({
            "comparison": f"good_vs_{category}",
            **results,
        })

        print(f"\nGOOD vs {category.upper()}")

        for name, value in results.items():
            print(f"{name}: {value:.4f}")

        y_true = [
            int(sample.gt_label.item())
            for sample in samples
        ]

        y_pred = [
            int(sample.pred_label.item())
            for sample in samples
        ]

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        print(
            f"TN={tn} | FP={fp} | FN={fn} | TP={tp}"
        )

    # 9. False positive rate on normal images.
    good_samples = grouped["good"]

    false_positive_images = sum(
        int(sample.pred_label.item() == 1)
        for sample in good_samples
    )

    false_positive_pixels = sum(
        int(sample.pred_mask.sum().item())
        for sample in good_samples
    )

    total_pixels = sum(
        sample.pred_mask.numel()
        for sample in good_samples
    )

    print("\nFALSE POSITIVES ON GOOD IMAGES")

    print(
        f"False positive images: "
        f"{false_positive_images}/{len(good_samples)}"
    )

    print(
        "Image FP rate:",
        f"{false_positive_images / len(good_samples):.4%}",
    )

    print(
        "Pixel FP rate:",
        f"{false_positive_pixels / total_pixels:.4%}",
    )

    # 10. Save metrics to CSV.
    output_path = PROJECT_ROOT / "results" / "patchcore_category_metrics.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "comparison",
                "image_AUROC",
                "image_F1Score",
                "pixel_AUROC",
                "pixel_F1Score",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
