"""Compare EfficientAD, PatchCore, and PaDiM on one fixed metal_nut split."""

import json
from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Padim, Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
IMAGENETTE_ROOT = PROJECT_ROOT / "datasets" / "imagenette"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "metal_nut_model_comparison_v2"

CHECKPOINTS = {
    "EfficientAD-S": (
        PROJECT_ROOT / "results" / "efficientad_metal_nut"
        / "EfficientAd" / "MVTecAD" / "metal_nut" / "v0"
        / "weights" / "lightning" / "model.ckpt"
    ),
    "PatchCore": (
        PROJECT_ROOT / "results" / "patchcore_metal_nut"
        / "Patchcore" / "MVTecAD" / "metal_nut" / "v0"
        / "weights" / "lightning" / "model.ckpt"
    ),
    "PaDiM": (
        PROJECT_ROOT / "results" / "padim_metal_nut"
        / "Padim" / "MVTecAD" / "metal_nut" / "v0"
        / "weights" / "lightning" / "model.ckpt"
    ),
}


def build_model(name: str):
    if name == "EfficientAD-S":
        return EfficientAd(model_size="small", imagenet_dir=IMAGENETTE_ROOT)
    if name == "PatchCore":
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            precision="float32",
        )
    return Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    missing = [path for path in CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(map(str, missing)))
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Comparison output already exists; refusing to overwrite it:\n{OUTPUT_ROOT}"
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
    split_counts = {
        "train": len(datamodule.train_data),
        "validation": len(datamodule.val_data),
        "test": len(datamodule.test_data),
    }
    if split_counts != {"train": 220, "validation": 57, "test": 58}:
        raise RuntimeError(f"Unexpected split counts: {split_counts}")

    comparison = {
        "category": "metal_nut",
        "split": split_counts,
        "threshold_handling": (
            "Validation was run before testing. AUROC is threshold-independent; "
            "F1 uses each checkpoint's postprocessor threshold. Anomalib returned "
            "empty validation metric dictionaries for PatchCore and PaDiM."
        ),
        "models": {},
    }

    for name, checkpoint in CHECKPOINTS.items():
        print(f"\n=== {name} ===")
        model_output = OUTPUT_ROOT / name.replace("-", "_").replace("/", "_")
        model_output.mkdir(parents=True, exist_ok=True)
        engine = Engine(
            accelerator="gpu",
            devices=1,
            default_root_dir=model_output,
        )
        model = build_model(name)
        validation = engine.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint,
            verbose=False,
        )
        test = engine.test(
            model=model,
            datamodule=datamodule,
            # Validation already restored the checkpoint and calibrated the
            # postprocessor. Do not reload it here, which would discard that
            # calibration and can fail for EfficientAD inference tensors.
            ckpt_path=None,
            verbose=False,
        )
        comparison["models"][name] = {
            "checkpoint": str(checkpoint),
            "validation_metrics": validation,
            "test_metrics": test,
        }
        print("Validation:", validation)
        print("Test:", test)
        del engine, model
        torch.cuda.empty_cache()

    report_path = OUTPUT_ROOT / "comparison.json"
    report_path.write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    print("\nComparison saved to:", report_path)


if __name__ == "__main__":
    main()
