"""Olive-compatible calibration inputs stored as reproducible NPZ fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry


class NpzCalibrationDataset:
    """A directory of one-input-dictionary-per-``.npz`` calibration sample."""

    def __init__(self, calibration_directory: str, max_samples: int | None = None) -> None:
        samples = sorted(Path(calibration_directory).glob("*.npz"))
        if max_samples is not None:
            samples = samples[:max_samples]
        if not samples:
            raise ValueError(f"No .npz files found in {calibration_directory!r}.")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict[str, np.ndarray], int]:
        with np.load(self.samples[index], allow_pickle=False) as sample:
            return {name: sample[name] for name in sample.files}, 0


class NpzCalibrationDataLoader:
    """Yield batch-one ``(inputs, label)`` pairs required by Neural Compressor."""

    batch_size = 1

    def __init__(self, dataset: NpzCalibrationDataset) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[tuple[dict[str, np.ndarray], int]]:
        for index in range(len(self.dataset)):
            yield self.dataset[index]


@Registry.register_dataset()
def npz_calibration_dataset(
    calibration_directory: str, max_samples: int | None = None, **_: Any
) -> NpzCalibrationDataset:
    """Create the registered Olive dataset component."""
    return NpzCalibrationDataset(calibration_directory, max_samples)


@Registry.register_dataloader()
def npz_calibration_dataloader(
    dataset: NpzCalibrationDataset, **_: Any
) -> NpzCalibrationDataLoader:
    """Create the registered Olive dataloader component."""
    return NpzCalibrationDataLoader(dataset)


def create_calibration_config(
    calibration_directory: Path, calibration_samples: int
) -> DataConfig:
    """Build the data configuration consumed by Olive's INC static pass."""
    return DataConfig(
        name="whisper_onnx_npz_calibration",
        load_dataset_config=DataComponentConfig(
            type="npz_calibration_dataset",
            params={
                "calibration_directory": str(calibration_directory),
                "max_samples": calibration_samples,
            },
        ),
        dataloader_config=DataComponentConfig(type="npz_calibration_dataloader"),
    )
