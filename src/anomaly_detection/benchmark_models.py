
from pathlib import Path
import csv
import gc
import statistics
import time

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "MVTecAD"
IMAGENETTE_ROOT = PROJECT_ROOT / "datasets" / "imagenette"

EFFICIENTAD_CHECKPOINT = (
    PROJECT_ROOT / "results" / "efficientad_metal_nut"
    / "EfficientAd" / "MVTecAD" / "metal_nut"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)

PATCHCORE_CHECKPOINT = (
    PROJECT_ROOT / "results" / "patchcore_metal_nut"
    / "Patchcore" / "MVTecAD" / "metal_nut"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)

OUTPUT_PATH = PROJECT_ROOT / "results" / "inference_benchmark.csv"

WARMUP_RUNS = 1
BENCHMARK_RUNS = 3


def create_datamodule():
    """Create the same dataset split for both models."""

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

    return datamodule


def create_efficientad():
    return EfficientAd(
        model_size="small",
        imagenet_dir=IMAGENETTE_ROOT,
        visualizer=False,
    )


def create_patchcore():
    return Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        precision="float32",
        visualizer=False,
    )


def benchmark_model(name, model_factory, checkpoint_path):

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    print("\n" + "=" * 65)
    print(f"BENCHMARKING: {name}")
    print("=" * 65)

    datamodule = create_datamodule()
    number_of_images = len(datamodule.test_data)

    model = model_factory()

    engine = Engine(
        accelerator="gpu",
        devices=1,
        default_root_dir=PROJECT_ROOT / "results" / "benchmark_runs" / name,
    )

    # Load trained weights and calibrate thresholds on validation data.
    print("\nLoading checkpoint and calibrating thresholds...")

    engine.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path=checkpoint_path,
        verbose=False,
    )

    print("Validation completed.")
    print("Test images:", number_of_images)

    test_loader = datamodule.test_dataloader()

    # Warm-up: initializes CUDA kernels and model execution.
    print("\nRunning warm-up...")

    warmup_predictions = engine.predict(
        model=model,
        dataloaders=test_loader,
        return_predictions=True,
    )

    if len(warmup_predictions) != number_of_images:
        raise RuntimeError("Warm-up prediction count is incorrect.")

    del warmup_predictions
    gc.collect()
    torch.cuda.synchronize()

    # Record memory already occupied by the loaded model.
    baseline_allocated = torch.cuda.memory_allocated(0)
    baseline_reserved = torch.cuda.memory_reserved(0)

    # Reset peak memory counters before timed inference.
    torch.cuda.reset_peak_memory_stats(0)

    run_times = []

    print("\nRunning timed inference...")

    for run_index in range(BENCHMARK_RUNS):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        predictions = engine.predict(
            model=model,
            dataloaders=test_loader,
            return_predictions=True,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if len(predictions) != number_of_images:
            raise RuntimeError(
                f"Incorrect prediction count: {len(predictions)}"
            )

        run_times.append(elapsed)

        print(
            f"Run {run_index + 1}: "
            f"{elapsed:.3f} seconds"
        )

        del predictions
        gc.collect()
        torch.cuda.synchronize()

    # Final memory measurements.
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)

    median_time = statistics.median(run_times)

    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)

    result = {
        "model": name,
        "test_images": number_of_images,
        "checkpoint_mb": checkpoint_size_mb,
        "median_total_seconds": median_time,
        "average_ms_per_image": (
            median_time / number_of_images * 1000
        ),
        "throughput_images_per_second": (
            number_of_images / median_time
        ),
        "baseline_vram_allocated_mb": (
            baseline_allocated / (1024 ** 2)
        ),
        "peak_vram_allocated_mb": (
            peak_allocated / (1024 ** 2)
        ),
        "additional_peak_allocated_mb": (
            max(0, peak_allocated - baseline_allocated) / (1024 ** 2)
        ),
        "peak_vram_reserved_mb": (
            peak_reserved / (1024 ** 2)
        ),
    }

    print("\nRESULT:", name)

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    # Release model and trainer before loading the next model.
    del engine, model, datamodule, test_loader
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return result


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"Total GPU memory: {gpu_memory:.2f} GB")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")

    results = []

    results.append(
        benchmark_model(
            name="EfficientAD-S",
            model_factory=create_efficientad,
            checkpoint_path=EFFICIENTAD_CHECKPOINT,
        )
    )

    results.append(
        benchmark_model(
            name="PatchCore",
            model_factory=create_patchcore,
            checkpoint_path=PATCHCORE_CHECKPOINT,
        )
    )

    # Save results to CSV.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("FINAL BENCHMARK COMPARISON")
    print("=" * 65)

    for result in results:

        print(f"\n{result['model']}")

        print(
            f"Average inference: "
            f"{result['average_ms_per_image']:.2f} ms/image"
        )

        print(
            f"Throughput: "
            f"{result['throughput_images_per_second']:.2f} images/sec"
        )

        print(
            f"Peak allocated VRAM: "
            f"{result['peak_vram_allocated_mb']:.2f} MB"
        )

    print("\nBenchmark CSV saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
