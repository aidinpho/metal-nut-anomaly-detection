"""Evaluate the trained PaDiM metal_nut checkpoint on the held-out test split."""

import json
from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "padim_metal_nut"
    / "Padim"
    / "MVTecAD"
    / "metal_nut"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "padim_metal_nut_evaluation"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Evaluation output already exists; refusing to overwrite it:\n{OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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

    expected_test_count = 58
    if len(datamodule.test_data) != expected_test_count:
        raise RuntimeError(
            f"Expected {expected_test_count} held-out test images, "
            f"found {len(datamodule.test_data)}."
        )

    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
    )
    engine = Engine(
        accelerator="gpu",
        devices=1,
        default_root_dir=OUTPUT_ROOT,
    )

    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=CHECKPOINT_PATH,
    )

    report = {
        "model": "PaDiM",
        "category": "metal_nut",
        "checkpoint": str(CHECKPOINT_PATH),
        "split": {
            "validation_mode": str(datamodule.val_split_mode),
            "validation_ratio": datamodule.val_split_ratio,
            "validation_images": len(datamodule.val_data),
            "test_images": len(datamodule.test_data),
        },
        "metrics": results,
    }
    report_path = OUTPUT_ROOT / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("Test metrics:", results)
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()
