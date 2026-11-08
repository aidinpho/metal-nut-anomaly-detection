from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
IMAGENETTE_ROOT = PROJECT_ROOT / "datasets" / "imagenette"
RESULTS_ROOT = PROJECT_ROOT / "results" / "efficientad_metal_nut"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Initializing MVTec AD DataModule...")

    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category="metal_nut",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        seed=42,
    )

    print("Initializing EfficientAD-S...")

    model = EfficientAd(
        model_size="small",
        imagenet_dir=IMAGENETTE_ROOT,
    )

    engine = Engine(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        default_root_dir=RESULTS_ROOT,
    )

    print("\nStarting EfficientAD training...")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()