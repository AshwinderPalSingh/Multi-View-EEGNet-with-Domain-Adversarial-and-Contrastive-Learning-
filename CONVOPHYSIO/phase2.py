from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from eeg_preprocessing_core import (
    EEGPreprocessor,
    _default_sample_path,
    load_eeg_data,
    set_global_seed,
    subject_split,
    validate_data,
)
from multiview_features import (
    build_feature_views_torch,
    compute_frequency_features_torch,
    prepare_band_definitions,
)


LOGGER = logging.getLogger("phase2")

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


ArrayLike = np.ndarray | np.memmap
FeatureMapping = dict[str, ArrayLike]
TensorBatch = torch.Tensor | dict[str, torch.Tensor]
REQUIRED_MULTIVIEW_VIEWS = ("time", "freq", "bands")


def _writable_float32(array: ArrayLike) -> np.ndarray:
    candidate = np.asarray(array, dtype=np.float32)
    if candidate.flags.c_contiguous and candidate.flags.writeable:
        return candidate
    return np.array(candidate, dtype=np.float32, copy=True, order="C")


def _sanitize_split_ratio(val_ratio: float, n_samples: int) -> int:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")
    if n_samples <= 1 or val_ratio <= 0.0:
        return 0
    target_val = int(round(n_samples * val_ratio))
    return min(max(target_val, 1), n_samples - 1)


def _non_stratified_split_indices(
    n_samples: int,
    target_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    if target_val == 0:
        return indices.astype(np.int64, copy=False), np.empty(0, dtype=np.int64)
    val_idx = indices[:target_val]
    train_idx = indices[target_val:]
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def _stratified_split_indices(
    y: np.ndarray,
    target_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(y.shape[0], dtype=np.int64)
    unique_labels, inverse = np.unique(y, return_inverse=True)

    grouped_indices = [
        indices[inverse == class_index][rng.permutation(np.sum(inverse == class_index))]
        for class_index in range(unique_labels.size)
    ]
    class_counts = np.asarray([group.size for group in grouped_indices], dtype=np.int64)
    ideal_val = class_counts.astype(np.float64) * (target_val / max(y.shape[0], 1))
    val_counts = np.floor(ideal_val).astype(np.int64)
    val_counts = np.minimum(val_counts, np.maximum(class_counts - 1, 0))

    current_val = int(val_counts.sum())
    remainder = ideal_val - val_counts

    while current_val < target_val:
        capacities = np.maximum(class_counts - 1 - val_counts, 0)
        candidates = np.where(capacities > 0)[0]
        if candidates.size == 0:
            break
        candidates = np.asarray(
            sorted(
                candidates.tolist(),
                key=lambda idx: (remainder[idx], class_counts[idx], -val_counts[idx]),
                reverse=True,
            ),
            dtype=np.int64,
        )
        updated = False
        for class_index in candidates:
            if current_val >= target_val:
                break
            if capacities[class_index] <= 0:
                continue
            val_counts[class_index] += 1
            current_val += 1
            updated = True
        if not updated:
            break

    while current_val > target_val:
        candidates = np.where(val_counts > 0)[0]
        if candidates.size == 0:
            break
        candidates = np.asarray(
            sorted(
                candidates.tolist(),
                key=lambda idx: (val_counts[idx] - ideal_val[idx], val_counts[idx]),
                reverse=True,
            ),
            dtype=np.int64,
        )
        val_counts[candidates[0]] -= 1
        current_val -= 1

    val_parts = [
        grouped_indices[class_index][: val_counts[class_index]]
        for class_index in range(unique_labels.size)
        if val_counts[class_index] > 0
    ]
    train_parts = [
        grouped_indices[class_index][val_counts[class_index] :]
        for class_index in range(unique_labels.size)
        if grouped_indices[class_index][val_counts[class_index] :].size > 0
    ]
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)

    if val_idx.size == 0 or train_idx.size == 0:
        LOGGER.warning("Falling back to non-stratified split because stratification was not feasible.")
        return _non_stratified_split_indices(y.shape[0], target_val, seed)

    train_idx = train_idx[rng.permutation(train_idx.size)]
    val_idx = val_idx[rng.permutation(val_idx.size)]
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray | None,
    val_ratio: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, time), got {X.shape}.")

    n_samples = int(X.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot split an empty dataset.")
    target_val = _sanitize_split_ratio(val_ratio, n_samples)

    y_array = None if y is None else np.asarray(y).reshape(-1)
    if y_array is not None and y_array.shape[0] != n_samples:
        raise ValueError(f"Label count {y_array.shape[0]} does not match sample count {n_samples}.")

    if target_val == 0:
        X_train = X.astype(np.float32, copy=False)
        X_val = np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32)
        y_train = None if y_array is None else y_array.copy()
        y_val = None if y_array is None else y_array[:0].copy()
        return X_train, X_val, y_train, y_val

    if y_array is None or not stratify:
        train_idx, val_idx = _non_stratified_split_indices(n_samples, target_val, seed)
    else:
        train_idx, val_idx = _stratified_split_indices(y_array, target_val, seed)

    X_train = X[train_idx].astype(np.float32, copy=False)
    X_val = X[val_idx].astype(np.float32, copy=False)
    y_train = None if y_array is None else y_array[train_idx]
    y_val = None if y_array is None else y_array[val_idx]
    return X_train, X_val, y_train, y_val


def _feature_length_from_path(path: Path) -> int:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        return int(array.shape[0])
    finally:
        del array


def _validate_loaded_views(
    views: FeatureMapping,
    *,
    metadata: Mapping[str, Any],
    split: str,
) -> FeatureMapping:
    time_view = views.get("time")
    if time_view is None:
        raise ValueError(f"Prepared split {split!r} is missing the required time view.")

    expected_views = metadata.get("available_views")
    if isinstance(expected_views, Sequence) and not isinstance(expected_views, (str, bytes)):
        missing_expected = [str(view) for view in expected_views if str(view) not in views]
        if missing_expected:
            raise ValueError(
                f"Prepared split {split!r} is missing expected multiview files: {missing_expected}."
            )

    reference_length = int(time_view.shape[0])
    reference_channels = int(time_view.shape[1])
    expected_shapes = metadata.get("shapes", {}) if isinstance(metadata.get("shapes"), Mapping) else {}
    for key, array in views.items():
        array_shape = tuple(int(dim) for dim in np.asarray(array).shape)
        if len(array_shape) != 3:
            raise ValueError(f"Prepared split {split!r} view {key!r} has invalid shape {array_shape}.")
        if array_shape[0] != reference_length:
            raise ValueError(
                f"Prepared split {split!r} view {key!r} has {array_shape[0]} samples but expected {reference_length}."
            )
        if int(array_shape[1]) == 0 or int(array_shape[2]) == 0:
            raise ValueError(f"Prepared split {split!r} view {key!r} is empty: {array_shape}.")
        if key in {"time", "freq", "bands"} and array_shape[1] != reference_channels:
            raise ValueError(
                f"Prepared split {split!r} view {key!r} has {array_shape[1]} channels but expected {reference_channels}."
            )
        expected_shape = expected_shapes.get(f"{split}_{key}_X")
        if expected_shape is None and key == "time":
            expected_shape = expected_shapes.get(f"{split}_X")
        if isinstance(expected_shape, Sequence) and not isinstance(expected_shape, (str, bytes)):
            expected_tuple = tuple(int(dim) for dim in expected_shape)
            if expected_tuple != array_shape:
                raise ValueError(
                    f"Prepared split {split!r} view {key!r} shape mismatch: {array_shape} vs metadata {expected_tuple}."
                )
    return views


def load_prepared_split(
    data_dir: str | os.PathLike[str],
    split: str,
) -> tuple[FeatureMapping, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    base_dir = Path(data_dir)
    time_path = base_dir / f"{split}_time_X.npy"
    if not time_path.exists():
        time_path = base_dir / f"{split}_X.npy"
    if not time_path.exists():
        raise FileNotFoundError(f"Missing prepared split array: {time_path}")

    meta_path = base_dir / "prepared_meta.json"
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                metadata = loaded

    views: FeatureMapping = {"time": np.load(time_path, mmap_mode="r", allow_pickle=False)}
    for name in ("freq", "bands"):
        candidate = base_dir / f"{split}_{name}_X.npy"
        if candidate.exists():
            views[name] = np.load(candidate, mmap_mode="r", allow_pickle=False)

    length = _feature_length_from_path(time_path)
    for name, array in views.items():
        if int(array.shape[0]) != length:
            raise ValueError(
                f"Prepared view {name!r} has {array.shape[0]} samples but expected {length} from {time_path.name}."
            )

    y_path = base_dir / f"{split}_y.npy"
    y = None
    if y_path.exists():
        y = np.load(y_path, mmap_mode="r", allow_pickle=False)
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != length:
            raise ValueError(f"Label count {y.shape[0]} does not match sample count {length} for split {split}.")

    subjects_path = base_dir / f"{split}_subjects.npy"
    subjects = None
    if subjects_path.exists():
        subjects = np.load(subjects_path, mmap_mode="r", allow_pickle=False)
        subjects = np.asarray(subjects).reshape(-1)
        if subjects.shape[0] != length:
            raise ValueError(
                f"Subject count {subjects.shape[0]} does not match sample count {length} for split {split}."
            )
    views = _validate_loaded_views(views, metadata=metadata, split=split)
    return views, y, subjects, metadata


def load_prepared_data(
    data_dir: str | os.PathLike[str],
) -> tuple[FeatureMapping, np.ndarray | None, np.ndarray | None, FeatureMapping, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    train_views, y_train, train_subjects, metadata = load_prepared_split(data_dir, "train")
    val_views, y_val, val_subjects, val_metadata = load_prepared_split(data_dir, "val")
    merged_meta = dict(metadata)
    for key, value in val_metadata.items():
        merged_meta.setdefault(key, value)
    return train_views, y_train, train_subjects, val_views, y_val, val_subjects, merged_meta


def load_canonical_data(
    data_dir: str | os.PathLike[str],
) -> tuple[FeatureMapping, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    return load_prepared_split(data_dir, "all")


def select_feature_views(
    feature_views: Mapping[str, ArrayLike],
    *,
    use_multiview: bool = True,
    ablation: str = "full",
    strict: bool = False,
) -> FeatureMapping:
    time_view = feature_views.get("time", feature_views.get("raw"))
    if time_view is None:
        raise ValueError("Feature views must include a time or raw tensor.")

    selected: FeatureMapping = {"time": time_view}
    if not use_multiview:
        return selected

    normalized_ablation = str(ablation).lower()
    include_freq = normalized_ablation in {"full", "time_freq", "time+freq", "freq"}
    include_bands = normalized_ablation in {"full", "time_bands", "time+bands", "bands"}

    if normalized_ablation in {"time", "time_only"}:
        return selected
    missing_requested: list[str] = []
    if include_freq and "freq" in feature_views:
        selected["freq"] = feature_views["freq"]
    elif include_freq:
        missing_requested.append("freq")
    if include_bands and "bands" in feature_views:
        selected["bands"] = feature_views["bands"]
    elif include_bands:
        missing_requested.append("bands")
    if missing_requested:
        if strict:
            raise ValueError(
                f"Multiview input was requested but the following prepared views are missing: {missing_requested}."
            )
        LOGGER.warning(
            "Falling back to time-only or partial multiview because prepared views are missing: %s",
            missing_requested,
        )
    return selected


def _split_feature_views(
    feature_views: Mapping[str, ArrayLike],
    y: np.ndarray | None,
    subjects: np.ndarray | None,
    *,
    val_ratio: float,
    stratify: bool,
    seed: int,
) -> tuple[FeatureMapping, FeatureMapping, np.ndarray | None, np.ndarray | None]:
    """Splits feature views for training and validation, ensuring subject-wise separation."""
    time_view = np.asarray(feature_views.get("time", feature_views.get("raw")))
    n_samples = int(time_view.shape[0])
    y_array = None if y is None else np.asarray(y).reshape(-1)
    subjects_array = None if subjects is None else np.asarray(subjects).reshape(-1)

    if subjects_array is not None and subjects_array.shape[0] == n_samples and len(np.unique(subjects_array)) > 1:
        # PRIORITY 1: Subject-wise split to prevent data leakage.
        train_idx, val_idx = subject_split(subjects_array, val_ratio=val_ratio, seed=seed)
    else:
        # Fallback to sample-wise split for single-subject or unlabeled data.
        target_val = _sanitize_split_ratio(val_ratio, n_samples)
        if target_val == 0:
            train_idx, val_idx = (np.arange(n_samples, dtype=np.int64), np.empty(0, dtype=np.int64))
        elif y_array is None or not stratify:
            train_idx, val_idx = _non_stratified_split_indices(n_samples, target_val, seed)
        else:
            train_idx, val_idx = _stratified_split_indices(y_array, target_val, seed)

    train_views = {
        key: np.asarray(value)[train_idx]
        for key, value in feature_views.items()
    }
    val_views = {
        key: np.asarray(value)[val_idx]
        for key, value in feature_views.items()
    }

    y_train = None if y_array is None else y_array[train_idx]
    y_val = None if y_array is None else y_array[val_idx]
    return train_views, val_views, y_train, y_val


class MultiViewEEGDataset(Dataset):
    def __init__(
        self,
        X_time: ArrayLike,
        y: np.ndarray | None = None,
        subjects: np.ndarray | None = None,
        feature_views: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        self.views: FeatureMapping = {"time": X_time}
        if feature_views is not None:
            for name, array in feature_views.items():
                if name in {"raw", "time"}:
                    continue
                self.views[str(name)] = array

        self.length = int(np.asarray(X_time).shape[0])
        if np.asarray(X_time).ndim != 3:
            raise ValueError(f"Expected X_time with shape (N, C, T), got {np.asarray(X_time).shape}.")
        for name, array in self.views.items():
            array_shape = np.asarray(array).shape
            if len(array_shape) != 3:
                raise ValueError(f"Expected view {name!r} to have shape (N, C, T), got {array_shape}.")
            if int(array_shape[0]) != self.length:
                raise ValueError(
                    f"View {name!r} has {array_shape[0]} samples but expected {self.length}."
                )

        self.y = None if y is None else np.asarray(y)
        if self.y is not None and self.y.shape[0] != self.length:
            raise ValueError(f"Label count {self.y.shape[0]} does not match sample count {self.length}.")
        
        self.subjects = None if subjects is None else np.asarray(subjects)
        if self.subjects is not None and self.subjects.shape[0] != self.length:
            raise ValueError(f"Subject ID count {self.subjects.shape[0]} does not match sample count {self.length}.")

        self.return_mapping = any(name != "time" for name in self.views)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        sample_views = {
            name: torch.from_numpy(_writable_float32(view[index]))
            for name, view in self.views.items()
        }
        if self.return_mapping:
            sample: torch.Tensor | dict[str, torch.Tensor] = sample_views
        else:
            sample = sample_views["time"]

        subject_tensor: torch.Tensor | None = (
            torch.tensor(int(self.subjects[index]), dtype=torch.long)
            if self.subjects is not None else None
        )

        if self.y is None:
            return sample, None, subject_tensor
        label = self.y[index]
        return sample, torch.tensor(int(label), dtype=torch.long), subject_tensor


def _stack_batch_inputs(samples: Sequence[torch.Tensor | dict[str, torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
    first = samples[0]
    if isinstance(first, dict):
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in first}
    return torch.stack(list(samples), dim=0)


def _collate_eeg_batch(
    batch: list[tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]]
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    samples = _stack_batch_inputs([sample for sample, _, _ in batch])
    labels = [label for _, label, _ in batch]
    subjects = [subject for _, _, subject in batch]
    if all(label is None for label in labels):
        return samples, None, None
    stacked_labels = torch.stack([l for l in labels if l is not None])
    stacked_subjects = (
        torch.stack([s for s in subjects if s is not None])
        if any(s is not None for s in subjects) else None
    )
    return samples, stacked_labels, stacked_subjects


def _recommended_num_workers(batch_size: int, dataset_size: int) -> int:
    cpu_count = os.cpu_count() or 1
    if dataset_size < max(batch_size * 2, 64):
        return 0
    # Unify logic to be consistent with evaluation script.
    if sys.platform == "darwin":
        return min(2, max(0, cpu_count - 1))
    return min(4, cpu_count)


def create_dataloaders(
    X_train: np.ndarray | Mapping[str, ArrayLike],
    y_train: np.ndarray | None,
    X_val: np.ndarray | Mapping[str, ArrayLike],
    y_val: np.ndarray | None,
    batch_size: int = 32,
    train_subjects: np.ndarray | None = None,
    val_subjects: np.ndarray | None = None,
    train_feature_views: Mapping[str, ArrayLike] | None = None,
    val_feature_views: Mapping[str, ArrayLike] | None = None,
) -> tuple[DataLoader, DataLoader]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    if isinstance(X_train, Mapping):
        train_time = np.asarray(X_train.get("time", X_train.get("raw")), dtype=np.float32)
        train_feature_views = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in X_train.items()
            if key not in {"time", "raw"}
        }
    else:
        train_time = np.asarray(X_train, dtype=np.float32)
        train_feature_views = None if train_feature_views is None else {
            key: np.asarray(value, dtype=np.float32) for key, value in train_feature_views.items()
        }

    if isinstance(X_val, Mapping):
        val_time = np.asarray(X_val.get("time", X_val.get("raw")), dtype=np.float32)
        val_feature_views = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in X_val.items()
            if key not in {"time", "raw"}
        }
    else:
        val_time = np.asarray(X_val, dtype=np.float32)
        val_feature_views = None if val_feature_views is None else {
            key: np.asarray(value, dtype=np.float32) for key, value in val_feature_views.items()
        }

    train_dataset = MultiViewEEGDataset(train_time, y_train, train_subjects, train_feature_views)
    val_dataset = MultiViewEEGDataset(val_time, y_val, val_subjects, val_feature_views)

    pin_memory = bool(torch.cuda.is_available())
    num_workers = _recommended_num_workers(batch_size, len(train_dataset) + len(val_dataset))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_eeg_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_drop_last = len(train_dataset) > batch_size
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=train_drop_last, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader


class ChannelNorm(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class HybridNorm1d(nn.Module):
    def __init__(self, channels: int, use_layernorm: bool = True) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(channels)
        self.layer_norm = ChannelLayerNorm(channels) if use_layernorm else nn.Identity()
        self.use_layernorm = bool(use_layernorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_out = self.batch_norm(x)
        if not self.use_layernorm:
            return batch_out
        layer_out = self.layer_norm(x)
        return 0.5 * (batch_out + layer_out)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


# === NOVEL: Domain Adversarial Training (DANN) Components ===
# Reference: Ganin et al., 'Domain-Adversarial Training of Neural Networks', JMLR 2016
# Adapted for cross-subject EEG motor imagery classification.

class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients during backward pass for domain adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for gradient reversal with configurable lambda."""

    def __init__(self):
        super().__init__()
        self.lambda_val = 0.0

    def set_lambda(self, val: float) -> None:
        self.lambda_val = float(val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)


class SubjectAdversarialHead(nn.Module):
    """DANN subject classifier with gradient reversal for domain-invariant features."""

    def __init__(self, in_features: int, num_subjects: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        hidden = max(in_features // 2, 32)
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_subjects),
        )

    def set_lambda(self, val: float) -> None:
        self.grl.set_lambda(val)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.grl(features))


class ContrastiveProjectionHead(nn.Module):
    """Projects features to a normalized embedding for supervised contrastive loss."""

    def __init__(self, in_features: int, out_features: int = 128) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(features), dim=1, eps=1e-6)


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020) for EEG class separability."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        contrast_dot = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = contrast_dot.max(dim=1, keepdim=True)
        logits = contrast_dot - logits_max.detach()
        self_mask = 1.0 - torch.eye(batch_size, device=device)
        pos_mask = mask * self_mask
        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        num_positives = torch.clamp(pos_mask.sum(1), min=1.0)
        mean_log_prob = (pos_mask * log_prob).sum(1) / num_positives
        return -mean_log_prob.mean()


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, eca_kernel_size: int = 3) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.eca = nn.Conv1d(1, 1, kernel_size=eca_kernel_size, padding=(eca_kernel_size - 1) // 2, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).squeeze(-1)
        se_weights = self.fc(pooled)
        eca_weights = self.eca(pooled.unsqueeze(1)).squeeze(1)
        weights = self.activation(se_weights + eca_weights).unsqueeze(-1)
        return x * weights


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, use_layernorm: bool = True) -> None:
        super().__init__()
        hidden = max(channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            HybridNorm1d(hidden, use_layernorm=use_layernorm),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(self.net(x))
        return x * (1.0 + attention)


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float,
        *,
        drop_path: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            HybridNorm1d(channels, use_layernorm=use_layernorm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            HybridNorm1d(channels, use_layernorm=use_layernorm),
            nn.Dropout(dropout),
        )
        self.channel_attention = SEBlock(channels)
        self.drop_path = DropPath(drop_path)
        self.output_norm = ChannelLayerNorm(channels) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.block(x)
        residual = self.channel_attention(residual)
        residual = self.drop_path(residual)
        return F.gelu(self.output_norm(x + residual))


class MultiScaleBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        target_length: int,
        dropout: float,
        *,
        drop_path: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        kernels = (3, 7, 15)
        branch_width = max(out_channels // len(kernels), 8)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_width, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                    HybridNorm1d(branch_width, use_layernorm=use_layernorm),
                    nn.GELU(),
                )
                for kernel_size in kernels
            ]
        )
        merged_channels = branch_width * len(kernels)
        self.project = nn.Sequential(
            nn.Conv1d(merged_channels, out_channels, kernel_size=1, bias=False),
            HybridNorm1d(out_channels, use_layernorm=use_layernorm),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine = ResidualTCNBlock(
            out_channels,
            dilation=1,
            dropout=dropout,
            drop_path=drop_path,
            use_layernorm=use_layernorm,
        )
        self.pool = nn.AdaptiveAvgPool1d(target_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.project(x)
        x = self.refine(x)
        return self.pool(x)


class BranchFusion(nn.Module):
    def __init__(
        self,
        branch_channels: int,
        fused_channels: int,
        num_branches: int,
        *,
        dropout: float = 0.22,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(branch_channels, max(branch_channels // 2, 16)),
                    nn.GELU(),
                    nn.Linear(max(branch_channels // 2, 16), 1),
                )
                for _ in range(num_branches)
            ]
        )
        self.project = nn.Sequential(
            nn.Conv1d(branch_channels * num_branches, fused_channels, kernel_size=1, bias=False),
            HybridNorm1d(fused_channels, use_layernorm=use_layernorm),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.cat([head(feature) for head, feature in zip(self.score_heads, features)], dim=1)
        weights = torch.softmax(scores, dim=1)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        weighted_features = [
            feature * weights[:, index].view(-1, 1, 1) for index, feature in enumerate(features)
        ]
        return self.project(torch.cat(weighted_features, dim=1)), entropy


class TemporalCrossAttention(nn.Module):
    """Cross-attention to dynamically align spectral bands to temporal positions."""
    def __init__(self, channels: int, use_layernorm: bool = True, temperature: float = 0.7) -> None:
        super().__init__()
        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.scale = (channels ** -0.5) / temperature
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = ChannelLayerNorm(channels) if use_layernorm else nn.Identity()

    def forward(self, query_feat: torch.Tensor, band_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(query_feat)  # (B, C, T)
        k = self.k_proj(band_feat)  # (B, C, n_bands)
        v = self.v_proj(band_feat)  # (B, C, n_bands)

        # (B, T, C) @ (B, C, n_bands) -> (B, T, n_bands)
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn_probs = F.softmax(attn, dim=-1)
        
        # Band attention entropy: track to avoid dominant band (one-hot collapse)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()

        # (B, T, n_bands) @ (B, n_bands, C) -> (B, T, C) -> (B, C, T)
        context = torch.bmm(attn_probs, v.transpose(1, 2)).transpose(1, 2)
        # Residual from query_feat (canonical cross-attention residual path):
        # output keeps query temporal length and remains shape-compatible.
        return self.norm(query_feat + self.out_proj(context)), entropy


class EEGNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 3,
        hidden_channels: int = 64,  # P95: 64 with D=3 + 64ch input = full capacity
        band_channels: int | None = None,
        branch_channels: int | None = None,
        fusion_channels: int | None = None,
        dropout: float = 0.25,
        sampling_frequency: float = 160.0,
        include_gamma: bool = False,
        band_definitions: Mapping[str, tuple[float, float]] | None = None,
        band_method: str = "fft_segments",
        band_segments: int = 8,
        inference_temperature: float = 1.0,
        fuse_length: int = 96,   # P95: shorter fuse reduces over-smoothing
        pool_out: int = 6,       # P95: more temporal resolution in pooled features
        tcn_depth: int = 3,      # P95: 3-layer TCN for deeper temporal context
        use_layernorm: bool = True,
        drop_path_rate: float = 0.12,  # P95: slightly higher to compensate for capacity
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.branch_channels = int(branch_channels if branch_channels is not None else max(hidden_channels * 2, 32))
        self.fusion_channels = int(fusion_channels if fusion_channels is not None else max(self.branch_channels * 2, 64))
        self.dropout = float(dropout)
        self.sampling_frequency = float(sampling_frequency)
        self.include_gamma = bool(include_gamma)
        self.band_definitions = prepare_band_definitions(
            band_definitions,
            include_gamma=include_gamma,
        )
        self.band_method = str(band_method).strip().lower()
        self.band_segments = max(int(band_segments), 1)
        self.band_names = list(self.band_definitions.keys())
        self.pool_out = int(pool_out)
        self.use_layernorm = bool(use_layernorm)
        self.drop_path_rate = float(drop_path_rate)
        self.band_channels = int(band_channels if band_channels is not None else self.in_channels)
        self.inference_temperature = float(max(inference_temperature, 1e-6))
        self.can_refresh_multiview = self.band_method == "fft_segments"

        # Keep per-sample channel-wise normalization in-model to reduce
        # cross-subject amplitude shift after feature extraction.
        self.time_norm = ChannelNorm()
        self.freq_norm = ChannelNorm()
        self.band_norm = ChannelNorm()
        self.expected_band_channels = self.band_channels
        branch_drop_paths = np.linspace(0.0, self.drop_path_rate, num=3, dtype=np.float32)
        tcn_drop_paths = np.linspace(self.drop_path_rate * 0.25, self.drop_path_rate, num=max(int(tcn_depth), 1), dtype=np.float32)

        # EEGNet-style Conv2d temporal->spatial factorization.
        # Block 1: Temporal filtering -- NO channel mixing (preserves spatial topology)
        # Conv2d(1, F1, (1, kernel_length)) treats each channel independently.
        _temporal_kernel = max(int(round(self.sampling_frequency * 0.25)), 16)  # ~250ms
        self.temporal_conv = nn.Conv2d(
            1, self.hidden_channels,
            kernel_size=(1, _temporal_kernel),
            padding=(0, _temporal_kernel // 2),
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(self.hidden_channels)

        # Block 2: Depthwise spatial filtering -- learns CSP-like spatial filters
        # Conv2d(F1, F1*D, (C, 1), groups=F1) learns D spatial filters per temporal feature.
        _D = 2  # P95v3: D=2 — D=3 caused subject memorization with 6192 samples
        self.spatial_conv = nn.Conv2d(
            self.hidden_channels, self.hidden_channels * _D,
            kernel_size=(self.in_channels, 1),
            groups=self.hidden_channels,
            bias=False,
        )
        self.spatial_bn = nn.BatchNorm2d(self.hidden_channels * _D)
        self.spatial_pool = nn.AvgPool2d(kernel_size=(1, 4))  # temporal downsampling
        self.spatial_dropout = nn.Dropout(self.dropout)

        # Block 3: Separable convolution for feature refinement
        _sep_channels = self.hidden_channels * _D
        self.separable_conv = nn.Sequential(
            # Depthwise: temporal filtering per feature map
            nn.Conv2d(
                _sep_channels, _sep_channels,
                kernel_size=(1, 16), padding=(0, 8),
                groups=_sep_channels, bias=False,
            ),
            # Pointwise: cross-feature mixing
            nn.Conv2d(_sep_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(self.dropout),
        )

        # Freq stem: same temporal->spatial factorization for spectral view
        _freq_kernel = max(_temporal_kernel // 2, 8)
        self.freq_temporal_conv = nn.Conv2d(
            1, self.hidden_channels,
            kernel_size=(1, _freq_kernel),
            padding=(0, _freq_kernel // 2),
            bias=False,
        )
        self.freq_temporal_bn = nn.BatchNorm2d(self.hidden_channels)
        self.freq_spatial_conv = nn.Conv2d(
            self.hidden_channels, self.hidden_channels * _D,
            kernel_size=(self.in_channels, 1),
            groups=self.hidden_channels,
            bias=False,
        )
        self.freq_spatial_bn = nn.BatchNorm2d(self.hidden_channels * _D)
        self.freq_spatial_pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.freq_spatial_dropout = nn.Dropout(self.dropout)
        self.freq_separable_conv = nn.Sequential(
            nn.Conv2d(
                _sep_channels, _sep_channels,
                kernel_size=(1, 16), padding=(0, 8),
                groups=_sep_channels, bias=False,
            ),
            nn.Conv2d(_sep_channels, self.hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(self.dropout),
        )

        # Band stem stays as Conv1d (band features are already spatially extracted)
        self.band_stem = nn.Sequential(
            nn.Conv1d(self.expected_band_channels, self.hidden_channels, kernel_size=1, bias=False),
            HybridNorm1d(self.hidden_channels, use_layernorm=self.use_layernorm),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.5),
        )

        self.time_branch = MultiScaleBranch(
            self.hidden_channels,
            self.branch_channels,
            fuse_length,
            self.dropout,
            drop_path=float(branch_drop_paths[0]),
            use_layernorm=self.use_layernorm,
        )
        self.freq_branch = MultiScaleBranch(
            self.hidden_channels,
            self.branch_channels,
            fuse_length,
            self.dropout,
            drop_path=float(branch_drop_paths[1]),
            use_layernorm=self.use_layernorm,
        )
        # FIX: Upgrade band_branch to MultiScaleBranch to match other branches.
        # This gives it the capacity to learn complex spectral patterns.
        self.band_branch = MultiScaleBranch(
            self.hidden_channels,
            self.branch_channels,
            target_length=16,  # Shorter target length for static band features
            dropout=self.dropout,
            drop_path=float(branch_drop_paths[2]),
            use_layernorm=self.use_layernorm,
        )
        
        # FIX 4 Upgrade (SOTA): Dynamic cross-attention alignment
        # Replaces static interpolation/repetition by dynamically pulling
        # relevant spectral band features for each temporal position.
        self.band_cross_attn = TemporalCrossAttention(
            self.branch_channels, use_layernorm=self.use_layernorm
        )
        # Learn the time-vs-freq query interpolation instead of hard-coding 0.5.
        self.query_alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.fusion = BranchFusion(
            self.branch_channels,
            self.fusion_channels,
            num_branches=3,
            dropout=self.dropout,
            use_layernorm=self.use_layernorm,
        )
        self.post_fusion_attention = SEBlock(self.fusion_channels)
        dilations = [2**index for index in range(max(int(tcn_depth), 1))]
        self.temporal_stack = nn.Sequential(
            *[
                ResidualTCNBlock(
                    self.fusion_channels,
                    dilation=dilation,
                    dropout=self.dropout,
                    drop_path=float(tcn_drop_paths[index]),
                    use_layernorm=self.use_layernorm,
                )
                for index, dilation in enumerate(dilations)
            ]
        )
        self.temporal_attention = TemporalAttention(self.fusion_channels, use_layernorm=self.use_layernorm)
        self.head = nn.Sequential(
            nn.Conv1d(self.fusion_channels, self.fusion_channels, kernel_size=1, bias=False),
            HybridNorm1d(self.fusion_channels, use_layernorm=self.use_layernorm),
            nn.GELU(),
            nn.Dropout(min(self.dropout * 1.3, 0.50)),  # P90: head conv dropout
        )
        self.pool = nn.AdaptiveAvgPool1d(self.pool_out)
        self.pre_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fusion_channels * self.pool_out, self.fusion_channels),
            nn.GELU(),
            nn.Dropout(min(self.dropout * 1.7, 0.55)),  # P90: strong pre-classifier dropout
        )
        self.classifier_dropout = nn.Dropout(min(self.dropout * 1.4, 0.50))  # P90: explicit classifier dropout
        self.classifier = nn.Linear(self.fusion_channels, self.num_classes)

        self._reset_parameters()
        self._dropout_modules: list[tuple[nn.Module, float]] = []
        self._last_aux_losses: dict[str, torch.Tensor] = {}
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                self._dropout_modules.append((module, float(module.p)))

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def init_auxiliary_heads(self, num_subjects: int) -> None:
        """Initialize DANN subject adversarial head and contrastive projection.

        Called after model creation once the number of training subjects is known.
        """
        device = next(self.parameters()).device
        self.subject_adversarial = SubjectAdversarialHead(
            self.fusion_channels, num_subjects, dropout=self.dropout
        ).to(device)
        self.contrastive_head = ContrastiveProjectionHead(
            self.fusion_channels, out_features=128
        ).to(device)

    def _resolve_time_input(self, x: TensorBatch) -> torch.Tensor:
        if isinstance(x, dict):
            for key in ("time", "raw"):
                value = x.get(key)
                if value is not None:
                    return value.float()
            raise ValueError("Input mapping must contain a 'time' or 'raw' tensor.")
        return x.float()

    def _build_fft_view(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return compute_frequency_features_torch(x.float()).detach()

    def _build_band_view(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.can_refresh_multiview:
                return build_feature_views_torch(
                    x.float(),
                    self.sampling_frequency,
                    self.band_definitions,
                    n_segments=self.band_segments,
                )["bands"].detach()
            n_bands_fallback = getattr(self, "_n_bands_fallback", 12)
            return F.adaptive_avg_pool1d(x, n_bands_fallback).detach()

    def refresh_input_views(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.can_refresh_multiview:
            return dict(x)
        time_x = self._resolve_time_input(x)
        refreshed = dict(x)
        shared_views = build_feature_views_torch(
            time_x,
            self.sampling_frequency,
            self.band_definitions,
            n_segments=self.band_segments,
        )
        refreshed["time"] = shared_views["time"]
        refreshed["raw"] = shared_views["time"]
        refreshed["freq"] = shared_views["freq"]
        refreshed["bands"] = shared_views["bands"]
        return refreshed

    def apply_temperature(self, logits: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        temp = float(self.inference_temperature if temperature is None else temperature)
        return logits / max(temp, 1e-6)

    def set_dropout_scale(self, scale: float) -> None:
        scale = float(max(scale, 0.1))
        for module, base_p in self._dropout_modules:
            module.p = min(max(base_p * scale, 0.0), 0.6)  # P95: cap at 0.6 for stronger late regularization

    def _align_feature_scale(self, feature: torch.Tensor) -> torch.Tensor:
        # Detach scale so it acts as a constant normalizer (no gradient through it).
        std = feature.detach().std(dim=(1, 2), keepdim=True, unbiased=False)

        # Per-sample adaptive epsilon (NOT batch-based).
        # This makes the normalization robust to the feature's own scale,
        # respecting the subject-dependent nature of EEG data.
        # A floor on the epsilon is critical for FP16 stability.
        mean_per_sample_std = std.mean(dim=(1, 2), keepdim=True)
        adaptive_eps = (1e-3 * mean_per_sample_std).clamp_min(1e-6)

        # SOTA+ formulation: smoother gradients and better for FP16 stability,
        # behaving closer to a principled LayerNorm.
        scale = torch.sqrt(std.square() + adaptive_eps)
        return feature / scale

    def get_aux_losses(self) -> dict[str, torch.Tensor]:
        return dict(self._last_aux_losses)

    def forward(self, x: TensorBatch, return_features: bool = False) -> torch.Tensor:
        if not isinstance(x, dict):
            # Single-view mode: DataLoader returns a raw tensor -- wrap it
            x = {"time": x}
        if "time" not in x and "raw" not in x:
            raise ValueError("Input mapping must contain a 'time' or 'raw' tensor.")
        time_x = self._resolve_time_input(x)

        # If multiview inputs are missing, compute them on-the-fly from time signal.
        if "freq" not in x:
            x["freq"] = self._build_fft_view(time_x)
        if "bands" not in x:
            x["bands"] = self._build_band_view(time_x)

        freq_x = x["freq"].float()
        band_x = x["bands"].float()
        band_feature_channels = int(band_x.shape[1])  # only electrode axis; band-time dim is free
        if band_feature_channels != self.expected_band_channels:
            raise ValueError(
                f"Band channel mismatch: got {tuple(band_x.shape)}, expected {self.expected_band_channels} electrode channels "
                f"(n_bands free). Pass --hidden-channels to match in_channels."
            )
        if not (torch.isfinite(time_x).all() and torch.isfinite(freq_x).all() and torch.isfinite(band_x).all()):
            raise ValueError("Non-finite values detected in multiview inputs.")

        time_x = self.time_norm(time_x)
        freq_x = self.freq_norm(freq_x)
        band_x = self.band_norm(band_x)
        # band_x: (B, in_channels, n_bands) -- spatial topology preserved, no reshape needed

        # EEGNet-style temporal->spatial factorization for time branch
        # (B, C, T) -> (B, 1, C, T) -> temporal->spatial->separable -> (B, H, 1, T') -> squeeze -> (B, H, T')
        _time_2d = time_x.unsqueeze(1)  # (B, 1, C, T)
        _time_2d = F.elu(self.temporal_bn(self.temporal_conv(_time_2d)))
        _time_2d = self.spatial_dropout(self.spatial_pool(F.elu(self.spatial_bn(self.spatial_conv(_time_2d)))))
        _time_2d = self.separable_conv(_time_2d)
        time_stem_out = _time_2d.squeeze(2)  # (B, H, T')

        # Same factorization for freq branch
        _freq_2d = freq_x.unsqueeze(1)
        _freq_2d = F.elu(self.freq_temporal_bn(self.freq_temporal_conv(_freq_2d)))
        _freq_2d = self.freq_spatial_dropout(self.freq_spatial_pool(F.elu(self.freq_spatial_bn(self.freq_spatial_conv(_freq_2d)))))
        _freq_2d = self.freq_separable_conv(_freq_2d)
        freq_stem_out = _freq_2d.squeeze(2)  # (B, H, T')

        time_feat = self.time_branch(time_stem_out)
        freq_feat = self.freq_branch(freq_stem_out)
        band_feat = self.band_branch(self.band_stem(band_x))
        
        # FIX: Scale alignment BEFORE attention to balance representations
        time_feat = self._align_feature_scale(time_feat)
        freq_feat = self._align_feature_scale(freq_feat)
        band_feat = self._align_feature_scale(band_feat)
        
        # SOTA Upgrade: Spectral-temporal cross attention using both time and freq contexts
        # FIX: Normalize and weight before addition to prevent oscillation and query imbalance
        # FIX: Use channel-wise normalization (dim=1) WITHOUT flatten to preserve temporal structure.
        alpha = torch.sigmoid(self.query_alpha_logit).clamp(0.25, 0.75)  # GOD-TIER: prevent single-branch collapse
        query_feat = alpha * F.normalize(time_feat, dim=1, eps=1e-6) + \
                     (1.0 - alpha) * F.normalize(freq_feat, dim=1, eps=1e-6)
        band_feat, band_attn_entropy = self.band_cross_attn(query_feat, band_feat)

        # Re-align band_feat scale after cross-attention transformation
        band_feat = self._align_feature_scale(band_feat)

        fused, fusion_entropy = self.fusion([time_feat, freq_feat, band_feat])
        # Simplified auxiliary losses: only track entropies that are used for regularization.
        # Diversity and alignment are removed to simplify the optimization objective.
        self._last_aux_losses = {
            "fusion_entropy": fusion_entropy,
            "band_attn_entropy": band_attn_entropy,
        }
        fused = self.post_fusion_attention(fused)
        fused = self.temporal_stack(fused)
        fused = self.temporal_attention(fused)
        fused = self.head(fused)
        fused = self.pool(fused)
        features = self.pre_classifier(fused)
        logits = self.classifier(self.classifier_dropout(features))
        if return_features:
            return logits, features
        return logits


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    model.eval()
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()
    return model


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_input_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(_writable_float32(x))
    else:
        tensor = x.detach()
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected input with shape (batch, channels, time) or (channels, time), got {tuple(tensor.shape)}."
        )
    return tensor.to(device, non_blocking=True)


def _to_batch_input(
    x: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    device: torch.device,
) -> TensorBatch:
    if isinstance(x, Mapping):
        batch = {
            str(key): _to_input_tensor(value, device)
            for key, value in x.items()
            if value is not None
        }
        if not batch:
            raise ValueError("Input mapping is empty.")
        return batch
    return _to_input_tensor(x, device)


def refresh_multiview_batch(
    model: nn.Module,
    batch_x: TensorBatch,
) -> TensorBatch:
    if not isinstance(batch_x, dict):
        return batch_x
    refresh_fn = getattr(model, "refresh_input_views", None)
    if callable(refresh_fn):
        return refresh_fn(batch_x)
    return batch_x


def apply_model_temperature(
    model: nn.Module,
    logits: torch.Tensor,
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    apply_fn = getattr(model, "apply_temperature", None)
    if callable(apply_fn):
        return apply_fn(logits, temperature=temperature)
    temp = float(1.0 if temperature is None else temperature)
    return logits / max(temp, 1e-6)


def predict_proba(
    model: nn.Module,
    x: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
) -> torch.Tensor:
    device = _model_device(model)
    inputs = _to_batch_input(x, device)
    with torch.no_grad():
        logits = apply_model_temperature(model, model(inputs))
        probs = torch.softmax(logits, dim=-1)
    return probs


def compute_entropy(probs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    eps = 1e-8
    if isinstance(probs, torch.Tensor):
        stable_probs = probs.clamp_min(eps)
        return -(stable_probs * stable_probs.log()).sum(dim=-1)
    probs_array = np.asarray(probs, dtype=np.float32)
    stable_probs = np.clip(probs_array, eps, 1.0)
    return -(stable_probs * np.log(stable_probs)).sum(axis=-1)


def mc_dropout_predict(
    model: nn.Module,
    x: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    T: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    device = _model_device(model)
    inputs = _to_batch_input(x, device)
    was_training = model.training

    enable_mc_dropout(model)
    with torch.no_grad():
        predictions = [torch.softmax(apply_model_temperature(model, model(inputs)), dim=-1).unsqueeze(0) for _ in range(T)]
    stacked = torch.cat(predictions, dim=0)
    mean_probs = stacked.mean(dim=0)
    uncertainty = compute_entropy(mean_probs)

    if was_training:
        model.train()
    else:
        model.eval()
    return mean_probs, uncertainty


def reject_predictions(
    probs: np.ndarray | torch.Tensor,
    entropy: np.ndarray | torch.Tensor,
    threshold: float,
) -> np.ndarray | torch.Tensor:
    if isinstance(probs, torch.Tensor):
        predictions = torch.argmax(probs, dim=-1)
        rejected = predictions.clone()
        rejected[entropy > threshold] = -1
        return rejected
    probs_array = np.asarray(probs)
    entropy_array = np.asarray(entropy)
    predictions = np.argmax(probs_array, axis=-1).astype(np.int64, copy=False)
    rejected = predictions.copy()
    rejected[entropy_array > threshold] = -1
    return rejected


def smooth_targets(y: torch.Tensor, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError(f"smoothing must be in [0, 1), got {smoothing}.")
    if num_classes <= 1:
        return torch.ones((y.shape[0], 1), device=y.device, dtype=torch.float32)
    off_value = smoothing / max(num_classes - 1, 1)
    on_value = 1.0 - smoothing
    target = torch.full((y.size(0), num_classes), off_value, device=y.device, dtype=torch.float32)
    target.scatter_(1, y.unsqueeze(1), on_value)
    return target


def _mix_inputs(inputs: TensorBatch, lam: float, index: torch.Tensor) -> TensorBatch:
    if isinstance(inputs, dict):
        mixed = {key: lam * value + (1.0 - lam) * value[index] for key, value in inputs.items()}
        if "time" in mixed and "raw" in mixed:
            mixed["raw"] = mixed["time"]
        return mixed
    return lam * inputs + (1.0 - lam) * inputs[index]


def mixup_batch(
    inputs: TensorBatch,
    y: torch.Tensor,
    num_classes: int,
    alpha: float = 0.2,
    smoothing: float = 0.1,
) -> tuple[TensorBatch, torch.Tensor]:
    if alpha <= 0.0:
        return inputs, smooth_targets(y, num_classes=num_classes, smoothing=smoothing)
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(y.size(0), device=y.device)
    mixed_inputs = _mix_inputs(inputs, lam, index)
    y_a = smooth_targets(y, num_classes=num_classes, smoothing=smoothing)
    y_b = smooth_targets(y[index], num_classes=num_classes, smoothing=smoothing)
    return mixed_inputs, lam * y_a + (1.0 - lam) * y_b


def _clone_mapping_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in inputs.items()}


def _apply_shared_channel_mask(
    inputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    updated = _clone_mapping_inputs(inputs)
    for key, value in inputs.items():
        if key == "raw":
            continue
        if value.ndim != 3:
            continue
        if value.shape[1] == mask.shape[1]:
            updated[key] = value * mask
    if "time" in updated:
        updated["raw"] = updated["time"]
    return updated


def apply_time_mask(
    inputs: TensorBatch,
    prob: float = 0.5,
    max_ratio: float = 0.1,
    masks_per_sample: int = 2,
) -> TensorBatch:
    if prob <= 0.0 or masks_per_sample <= 0:
        return inputs

    if isinstance(inputs, dict):
        if "time" not in inputs and "raw" not in inputs:
            return inputs
        key = "time" if "time" in inputs else "raw"
        time_x = inputs[key]
    else:
        key = None
        time_x = inputs

    if torch.rand(1, device=time_x.device).item() > prob:
        return inputs

    batch_size, _, time_steps = time_x.shape
    max_width = max(1, int(round(time_steps * max_ratio)))
    positions = torch.arange(time_steps, device=time_x.device).view(1, 1, time_steps)
    masked = time_x.clone()
    for _ in range(masks_per_sample):
        widths = torch.randint(1, max_width + 1, (batch_size,), device=time_x.device)
        starts = torch.randint(0, time_steps, (batch_size,), device=time_x.device)
        ends = torch.clamp(starts + widths, max=time_steps)
        mask = (positions >= starts.view(batch_size, 1, 1)) & (positions < ends.view(batch_size, 1, 1))
        masked = masked.masked_fill(mask, 0.0)

    if isinstance(inputs, dict):
        updated = _clone_mapping_inputs(inputs)
        updated[key] = masked
        if key == "time":
            updated["raw"] = masked
        return updated
    return masked


def apply_gaussian_noise(
    inputs: TensorBatch,
    std: float = 0.01,
    prob: float = 0.5,
) -> TensorBatch:
    if std <= 0.0 or prob <= 0.0:
        return inputs
    if isinstance(inputs, dict):
        key = "time" if "time" in inputs else "raw"
        if key not in inputs or torch.rand(1, device=inputs[key].device).item() > prob:
            return inputs
        updated = _clone_mapping_inputs(inputs)
        updated[key] = inputs[key] + torch.randn_like(inputs[key]) * std
        if key == "time" and "raw" in updated:
            updated["raw"] = updated["time"]
        return updated
    if torch.rand(1, device=inputs.device).item() > prob:
        return inputs
    return inputs + torch.randn_like(inputs) * std


def apply_channel_dropout(
    inputs: TensorBatch,
    drop_ratio: float = 0.1,
    prob: float = 0.5,
) -> TensorBatch:
    if drop_ratio <= 0.0 or prob <= 0.0:
        return inputs
    if isinstance(inputs, dict):
        key = "time" if "time" in inputs else "raw"
        if key not in inputs or torch.rand(1, device=inputs[key].device).item() > prob:
            return inputs
        time_x = inputs[key]
    else:
        key = None
        time_x = inputs
        if torch.rand(1, device=time_x.device).item() > prob:
            return inputs

    batch_size, channels, _ = time_x.shape
    num_drop = max(1, int(round(channels * drop_ratio)))
    drop_mask = torch.ones((batch_size, channels, 1), device=time_x.device, dtype=time_x.dtype)
    for batch_index in range(batch_size):
        channel_indices = torch.randperm(channels, device=time_x.device)[:num_drop]
        drop_mask[batch_index, channel_indices] = 0.0
    dropped = time_x * drop_mask

    if isinstance(inputs, dict):
        updated = _apply_shared_channel_mask(inputs, drop_mask)
        updated[key] = dropped
        if key == "time":
            updated["raw"] = dropped
        return updated
    return dropped


def apply_temporal_shift(
    inputs: TensorBatch,
    max_shift_ratio: float = 0.02,
    prob: float = 0.5,
) -> TensorBatch:
    if max_shift_ratio <= 0.0 or prob <= 0.0:
        return inputs
    if isinstance(inputs, dict):
        key = "time" if "time" in inputs else "raw"
        if key not in inputs or torch.rand(1, device=inputs[key].device).item() > prob:
            return inputs
        time_x = inputs[key]
    else:
        key = None
        time_x = inputs
        if torch.rand(1, device=time_x.device).item() > prob:
            return inputs

    max_shift = max(1, int(round(time_x.shape[-1] * max_shift_ratio)))
    shifts = torch.randint(-max_shift, max_shift + 1, (time_x.shape[0],), device=time_x.device)
    shifted = torch.stack([torch.roll(sample, int(shift.item()), dims=-1) for sample, shift in zip(time_x, shifts)], dim=0)

    if isinstance(inputs, dict):
        updated = _clone_mapping_inputs(inputs)
        updated[key] = shifted
        if key == "time":
            updated["raw"] = shifted
        return updated
    return shifted


def apply_random_scaling(
    inputs: TensorBatch,
    scale_range: tuple[float, float] = (0.9, 1.1),
    prob: float = 0.3,
) -> TensorBatch:
    if prob <= 0.0 or scale_range[0] <= 0.0 or scale_range[1] <= 0.0:
        return inputs
    if isinstance(inputs, dict):
        key = "time" if "time" in inputs else "raw"
        if key not in inputs or torch.rand(1, device=inputs[key].device).item() > prob:
            return inputs
        time_x = inputs[key]
    else:
        key = None
        time_x = inputs
        if torch.rand(1, device=time_x.device).item() > prob:
            return inputs

    low, high = float(scale_range[0]), float(scale_range[1])
    scales = torch.empty((time_x.shape[0], 1, 1), device=time_x.device, dtype=time_x.dtype).uniform_(low, high)
    scaled = time_x * scales
    if isinstance(inputs, dict):
        updated = _clone_mapping_inputs(inputs)
        updated[key] = scaled
        if key == "time" and "raw" in updated:
            updated["raw"] = updated["time"]
        return updated
    return scaled


class WeightedSoftTargetCrossEntropy(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        if self.class_weights is None:
            weights = 1.0
        else:
            weights = self.class_weights.unsqueeze(0)
        loss = -(targets * weights * log_probs).sum(dim=-1)
        return loss.mean()


class FocalSoftTargetCrossEntropy(nn.Module):
    """Focal loss variant for soft (mixup) targets. gamma=0 reduces to CE."""

    def __init__(self, class_weights: torch.Tensor | None = None, gamma: float = 1.5) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if self.class_weights is None:
            weights = 1.0
        else:
            weights = self.class_weights.unsqueeze(0)
        ce = -(targets * weights * log_probs).sum(dim=-1)
        if self.gamma > 0.0:
            focal_factor = (1.0 - (targets * probs).sum(dim=-1)).pow(self.gamma)
            return (focal_factor * ce).mean()
        return ce.mean()


class FocalCrossEntropy(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if self.label_smoothing > 0:
            soft_targets = smooth_targets(targets, num_classes=logits.shape[1], smoothing=self.label_smoothing)
            ce = -(soft_targets * log_probs).sum(dim=-1)
            focal_factor = (1.0 - (soft_targets * probs).sum(dim=-1)).pow(self.gamma)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")
            focal_factor = (1.0 - probs.gather(1, targets.unsqueeze(1)).squeeze(1)).pow(self.gamma)
        if self.class_weights is not None:
            ce = ce * self.class_weights[targets]
        return (focal_factor * ce).mean()


def save_preprocessed_data(
    X: np.ndarray | Mapping[str, np.ndarray],
    y: np.ndarray | None,
    save_path: str | os.PathLike[str],
) -> dict[str, str | None]:
    base_path = Path(save_path)
    base = base_path.with_suffix("") if base_path.suffix == ".npy" else base_path
    base.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(X, Mapping):
        if "time" not in X and "raw" not in X:
            raise ValueError("Preprocessed mappings must include a time/raw view.")
        time_array = np.ascontiguousarray(np.asarray(X.get("time", X.get("raw")), dtype=np.float32))
        if time_array.ndim != 3:
            raise ValueError(f"Expected time view with shape (N, C, T), got {time_array.shape}.")
        time_alias_path = base.parent / f"{base.name}_time_X.npy"
        np.save(time_alias_path, time_array, allow_pickle=False)
        paths: dict[str, str | None] = {
            "time": str(time_alias_path.resolve()),
        }
        for name, array in X.items():
            if name in {"time", "raw"}:
                continue
            view_path = base.parent / f"{base.name}_{name}_X.npy"
            view_array = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
            if view_array.ndim != 3:
                raise ValueError(f"Expected view {name!r} to have shape (N, C, T), got {view_array.shape}.")
            np.save(view_path, view_array, allow_pickle=False)
            paths[name] = str(view_path.resolve())
    else:
        time_alias_path = base.parent / f"{base.name}_time_X.npy"
        time_array = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        np.save(time_alias_path, time_array, allow_pickle=False)
        paths = {"time": str(time_alias_path.resolve())}

    y_path: Path | None = None
    if y is not None:
        y_array = np.ascontiguousarray(np.asarray(y), dtype=np.int64)
        y_path = base.parent / f"{base.name}_y.npy"
        np.save(y_path, y_array, allow_pickle=False)
    paths["y"] = None if y_path is None else str(y_path.resolve())
    return paths


def _notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def export_training_notebook(output_path: str = "train.ipynb") -> str:
    notebook = {
        "cells": [
            _notebook_cell(
                "markdown",
                "# EEG Research Training\n"
                "Colab-ready notebook for loading prepared arrays, constructing the multi-branch EEGNet, "
                "and running uncertainty-aware inference. Training code is provided as a template only.",
            ),
            _notebook_cell(
                "code",
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import torch\n"
                "from phase2 import EEGNet, create_dataloaders, load_prepared_data, mc_dropout_predict\n"
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                "print('Device:', device)\n",
            ),
            _notebook_cell(
                "code",
                "DATA_DIR = Path('./prepared')\n"
                "train_views, y_train, _, val_views, y_val, _, meta = load_prepared_data(DATA_DIR)\n"
                "print('Train time shape:', train_views['time'].shape)\n"
                "print('Available views:', list(train_views.keys()))\n"
                "print('Metadata keys:', list(meta.keys()))\n",
            ),
            _notebook_cell(
                "code",
                "train_loader, val_loader = create_dataloaders(train_views, y_train, val_views, y_val, batch_size=64)\n"
                "num_classes = int(np.unique(y_train).size) if y_train is not None else 1\n"
                "band_channels = int(train_views['bands'].shape[1]) if 'bands' in train_views else int(train_views['time'].shape[1])\n"
                "model = EEGNet(\n"
                "    in_channels=train_views['time'].shape[1],\n"
                "    num_classes=num_classes,\n"
                "    band_channels=band_channels,\n"
                "    sampling_frequency=float(meta.get('sampling_frequency_inferred', 160.0) or 160.0),\n"
                ").to(device)\n"
                "print(model)\n",
            ),
            _notebook_cell(
                "code",
                "def training_loop_template(model, loader, optimizer, criterion, device):\n"
                "    model.train()\n"
                "    for batch_x, batch_y in loader:\n"
                "        if batch_y is None:\n"
                "            raise ValueError('Labels are required for supervised training.')\n"
                "        if isinstance(batch_x, dict):\n"
                "            batch_x = {k: v.to(device) for k, v in batch_x.items()}\n"
                "        else:\n"
                "            batch_x = batch_x.to(device)\n"
                "        batch_y = batch_y.to(device)\n"
                "        optimizer.zero_grad(set_to_none=True)\n"
                "        logits = model(batch_x)\n"
                "        loss = criterion(logits, batch_y)\n"
                "        loss.backward()\n"
                "        optimizer.step()\n",
            ),
            _notebook_cell(
                "code",
                "batch_x, _ = next(iter(val_loader))\n"
                "if isinstance(batch_x, dict):\n"
                "    batch_x = {k: v.to(device) for k, v in batch_x.items()}\n"
                "else:\n"
                "    batch_x = batch_x.to(device)\n"
                "mean_probs, entropy = mc_dropout_predict(model, batch_x, T=20)\n"
                "print('MC probs:', mean_probs.shape)\n"
                "print('Entropy:', entropy[:5])\n",
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=2)
    return str(output_file.resolve())


def _infer_num_classes(y_train: np.ndarray | None, y_val: np.ndarray | None) -> int:
    if y_train is None and y_val is None:
        return 1
    label_arrays = [array for array in (y_train, y_val) if array is not None and array.size > 0]
    if not label_arrays:
        return 1
    labels = np.concatenate(label_arrays)
    return int(np.unique(labels).size)


def _describe_distribution(y: np.ndarray | None, label_mapping: Mapping[str, Any] | Mapping[int, Any]) -> dict[str, int]:
    if y is None:
        return {}
    mapping = {int(key): str(value) for key, value in label_mapping.items()} if label_mapping else {}
    counts = Counter(np.asarray(y).reshape(-1).tolist())
    return {
        mapping.get(int(label), str(label)): int(count)
        for label, count in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def summarize_model(model: nn.Module, input_shape: tuple[int, int]) -> dict[str, Any]:
    device = _model_device(model)
    n_channels, n_times = input_shape
    # Build multiview dummy matching EEGNet.forward() expectations
    dummy_time = torch.zeros((1, n_channels, n_times), dtype=torch.float32, device=device)
    dummy_freq = torch.zeros((1, n_channels, n_times), dtype=torch.float32, device=device)
    band_ch = getattr(model, "expected_band_channels", n_channels)
    dummy_bands = torch.zeros((1, band_ch, 1), dtype=torch.float32, device=device)
    dummy: TensorBatch = {"time": dummy_time, "freq": dummy_freq, "bands": dummy_bands}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(dummy)
    if was_training:
        model.train()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "output_shape": tuple(int(dim) for dim in output.shape),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def _save_phase2_artifacts(
    train_views: Mapping[str, np.ndarray],
    y_train: np.ndarray | None,
    val_views: Mapping[str, np.ndarray],
    y_val: np.ndarray | None,
    save_dir: str,
) -> dict[str, dict[str, str | None]]:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = save_preprocessed_data(train_views, y_train, output_dir / "train")
    val_paths = save_preprocessed_data(val_views, y_val, output_dir / "val")
    return {"train": train_paths, "val": val_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG phase 2 preparation for training and uncertainty inference.")
    parser.add_argument("--path", type=str, default=None, help="Path to an EEG file or dataset directory.")
    parser.add_argument("--low", type=float, default=7.0, help="Low cutoff for bandpass filtering.")
    parser.add_argument("--high", type=float, default=30.0, help="High cutoff for bandpass filtering.")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Epoch length for sliding windows.")
    parser.add_argument("--overlap-sec", type=float, default=0.0, help="Overlap between adjacent windows in seconds.")
    parser.add_argument("--include-gamma", action="store_true", help="Include gamma-band processing.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataloaders.")
    parser.add_argument("--hidden-channels", type=int, default=32, help="Base hidden width for EEGNet.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout rate for EEGNet.")
    parser.add_argument("--use-multiview", type=str, default="true", help="Whether to save/use multiview tensors.")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=("full", "time", "time_freq", "time_bands"),
        help="Select which prepared views to keep.",
    )
    parser.add_argument("--save-dir", type=str, default="prepared", help="Directory for saved NumPy arrays.")
    parser.add_argument("--notebook-path", type=str, default="train.ipynb", help="Output path for the exported notebook.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic preparation.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_global_seed(args.seed)

    sample_path = Path(args.path) if args.path is not None else _default_sample_path()
    X, y, meta = load_eeg_data(str(sample_path))

    preprocessor = EEGPreprocessor(
        low=args.low,
        high=args.high,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        include_gamma=args.include_gamma,
        seed=args.seed,
        verbose=True,
    )
    feature_views = preprocessor.preprocess_with_views(X, meta)
    X_valid, y_valid, validation_report = validate_data(feature_views["time"], y)
    feature_views = preprocessor.create_feature_views(X_valid, meta, already_preprocessed=True)
    use_multiview = str(args.use_multiview).lower() not in {"0", "false", "no"}
    all_train_views, all_val_views, y_train, y_val = _split_feature_views(
        feature_views,
        y_valid,
        None,  # subjects not available in standalone phase2 run
        val_ratio=args.val_ratio,
        stratify=y_valid is not None,
        seed=args.seed,
    )
    train_views = select_feature_views(
        all_train_views,
        use_multiview=use_multiview,
        ablation=args.ablation,
        strict=use_multiview and args.ablation != "time",
    )
    val_views = select_feature_views(
        all_val_views,
        use_multiview=use_multiview,
        ablation=args.ablation,
        strict=use_multiview and args.ablation != "time",
    )
    train_loader, val_loader = create_dataloaders(train_views, y_train, val_views, y_val, batch_size=args.batch_size)

    num_classes = _infer_num_classes(y_train, y_val)
    if y_valid is None:
        LOGGER.warning("No labels detected. Initializing a placeholder model with num_classes=1.")

    train_time = train_views["time"]
    band_channels = (
        int(train_views["bands"].shape[1])
        if "bands" in train_views
        else int(train_time.shape[1])
    )
    inferred_sfreq = float(meta.get("sampling_frequency", train_time.shape[-1] / max(args.window_sec, 1e-6)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(
        in_channels=int(train_time.shape[1]),
        num_classes=num_classes,
        hidden_channels=args.hidden_channels,
        band_channels=band_channels,
        dropout=args.dropout,
        sampling_frequency=inferred_sfreq,
        include_gamma=args.include_gamma,
    ).to(device)
    model_summary = summarize_model(model, (int(train_time.shape[1]), int(train_time.shape[2])))

    train_batch_shape: tuple[int, ...] | dict[str, tuple[int, ...]] | None = None
    val_batch_shape: tuple[int, ...] | dict[str, tuple[int, ...]] | None = None
    example_logits_shape: tuple[int, ...] | None = None
    if len(train_loader) > 0:
        train_batch, _, _ = next(iter(train_loader))
        if isinstance(train_batch, dict):
            train_batch_shape = {key: tuple(int(dim) for dim in value.shape) for key, value in train_batch.items()}
            train_batch_device: TensorBatch = {key: value.to(device, non_blocking=True) for key, value in train_batch.items()}
        else:
            train_batch_shape = tuple(int(dim) for dim in train_batch.shape)
            train_batch_device = train_batch.to(device, non_blocking=True)
        with torch.no_grad():
            example_logits = model(train_batch_device)
        example_logits_shape = tuple(int(dim) for dim in example_logits.shape)
    if len(val_loader) > 0:
        val_batch, _, _ = next(iter(val_loader))
        if isinstance(val_batch, dict):
            val_batch_shape = {key: tuple(int(dim) for dim in value.shape) for key, value in val_batch.items()}
        else:
            val_batch_shape = tuple(int(dim) for dim in val_batch.shape)

    saved_paths = _save_phase2_artifacts(all_train_views, y_train, all_val_views, y_val, args.save_dir)
    notebook_path = export_training_notebook(args.notebook_path)

    print("\nPhase 2 summary")
    print(f"Input file: {sample_path}")
    print(f"Dataset type: {meta.get('dataset_type')}")
    print(f"Feature views available from preprocessing: {list(feature_views.keys())}")
    print(f"Selected views: {list(train_views.keys())}")
    print(f"Train shape: {tuple(int(dim) for dim in train_views['time'].shape)}")
    print(f"Val shape: {tuple(int(dim) for dim in val_views['time'].shape)}")
    print(f"Train distribution: {_describe_distribution(y_train, meta.get('label_mapping', {})) or 'unlabeled'}")
    print(f"Val distribution: {_describe_distribution(y_val, meta.get('label_mapping', {})) or 'unlabeled'}")
    print(f"Validation report: {validation_report}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    print(f"Example train batch: {train_batch_shape}")
    print(f"Example val batch: {val_batch_shape}")
    print(f"Example logits shape: {example_logits_shape}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Saved arrays: {saved_paths}")
    print(f"Notebook path: {notebook_path}")
    print(
        "Model summary: "
        f"output_shape={model_summary['output_shape']}, "
        f"total_params={model_summary['total_params']}, "
        f"trainable_params={model_summary['trainable_params']}"
    )
    print(model)


if __name__ == "__main__":
    main()
