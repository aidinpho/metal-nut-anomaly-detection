
from pathlib import Path

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
RESULTS_ROOT = PROJECT_ROOT / "results" / "patchcore_metal_nut"


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    torch.manual_seed(42)

    print("GPU:", torch.cuda.get_device_name(0))

    # 1. Initialize the dataset
    print("\nInitializing MVTec AD...")

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
    print("Training samples:", len(datamodule.train_data))
    print("Validation samples:", len(datamodule.val_data))
    print("Test samples:", len(datamodule.test_data))

    # 2. Initialize PatchCore
    print("\nInitializing PatchCore...")

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        precision="float32",
    )

    # 3. Initialize training engine
    engine = Engine(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        default_root_dir=RESULTS_ROOT,
    )

    # 4. Fit PatchCore
    print("\nStarting PatchCore feature extraction...")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nPatchCore training completed!")

    print("Results directory:", RESULTS_ROOT)


if __name__ == "__main__":
    main()
