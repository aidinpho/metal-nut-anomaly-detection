from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
IMAGENETTE_ROOT = PROJECT_ROOT / "datasets" / "imagenette"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "efficientad_metal_nut"
    / "EfficientAd"
    / "MVTecAD"
    / "metal_nut"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    print("GPU:", torch.cuda.get_device_name(0))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category="metal_nut",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        seed=42,
    )

    model = EfficientAd(
        model_size="small",
        imagenet_dir=IMAGENETTE_ROOT,
    )

    engine = Engine(
        accelerator="gpu",
        devices=1,
    )

    print("\nStarting model evaluation...")

    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=CHECKPOINT_PATH,
    )

    print("\nEvaluation completed!")
    print("Results:", results)


if __name__ == "__main__":
    main()