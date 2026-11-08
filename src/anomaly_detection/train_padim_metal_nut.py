"""Train PaDiM on the MVTec AD metal_nut category."""

from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "padim_metal_nut"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)

    # Never reuse an output directory: anomalib may write checkpoints there.
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files; choose a new path to "
            f"preserve them:\n{OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    print("GPU:", torch.cuda.get_device_name(0))

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

    train_count = len(datamodule.train_data)
    val_count = len(datamodule.val_data)
    test_count = len(datamodule.test_data)
    print("Train:", train_count)
    print("Validation:", val_count)
    print("Test:", test_count)

    if val_count == 0 or test_count == 0:
        raise RuntimeError("Validation or test split is empty.")
    if val_count + test_count != 115:
        raise RuntimeError(
            "Validation and test splits must cover all 115 official test images."
        )

    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
    )
    engine = Engine(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        default_root_dir=OUTPUT_ROOT,
    )

    engine.fit(model=model, datamodule=datamodule)

    checkpoints = sorted(OUTPUT_ROOT.rglob("*.ckpt"))
    if not checkpoints:
        raise RuntimeError(f"Training finished without a checkpoint in {OUTPUT_ROOT}")

    print("\nCheckpoint files:")
    for checkpoint in checkpoints:
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"{checkpoint} ({size_mb:.2f} MB)")
    print("\nOutput directory:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
