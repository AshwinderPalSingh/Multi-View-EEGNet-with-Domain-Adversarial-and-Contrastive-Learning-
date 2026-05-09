from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_USE_SHM", "0")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate EEGNet predictions, calibration, uncertainty, or aggregate fold reports.")
    parser.add_argument("--model-path", type=str, default="", help="Path to the trained model checkpoint (.pt).")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing prepared arrays for single-run evaluation.")
    parser.add_argument("--run-dir", type=str, default="", help="Cross-validation run directory containing fold_*/fold_metrics.json.")
    parser.add_argument("--aggregate-folds", type=str, default="false", help="Aggregate fold reports from --run-dir instead of loading one model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--ece-bins", type=int, default=10, help="Number of bins for ECE and reliability diagram.")
    parser.add_argument("--mc-passes", type=int, default=30, help="Number of stochastic forward passes for MC Dropout.")
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs", help="Directory where plots and reports will be saved.")
    parser.add_argument("--use-adabn", type=str, default="true", help="Apply AdaBN on evaluation data before inference.")
    return parser


if __name__ == "__main__" and any(arg in sys.argv for arg in ("-h", "--help")):
    _build_arg_parser().print_help()
    raise SystemExit(0)

import numpy as np
import torch
import torch.nn.functional as F

try:
    if os.environ.get("DISPLAY", "") == "" and "COLAB_RELEASE_TAG" not in os.environ:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

from torch.utils.data import DataLoader
from phase2 import (
    EEGNet,
    MultiViewEEGDataset,
    apply_model_temperature,
    enable_mc_dropout,
    load_prepared_split,
    select_feature_views,
)


LOGGER = logging.getLogger("evaluation")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _recommended_num_workers(batch_size: int, dataset_size: int) -> int:
    cpu_count = os.cpu_count() or 1
    if dataset_size < max(batch_size * 2, 64):
        return 0
    if sys.platform == "darwin":
        return min(2, max(0, cpu_count - 1))
    return min(4, max(1, cpu_count // 2))


def _stack_eval_inputs(samples: list[torch.Tensor | dict[str, torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
    first = samples[0]
    if isinstance(first, dict):
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in first}
    return torch.stack(samples, dim=0)


def _collate_eval_batch(
    batch: list[tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]]
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor | None]:
    samples = _stack_eval_inputs([sample for sample, _, _ in batch])
    labels = [label for _, label, _ in batch]
    if all(label is None for label in labels):
        return samples, None
    return samples, torch.stack([label for label in labels if label is not None], dim=0)


def create_eval_dataloader(
    X: np.ndarray | Mapping[str, np.ndarray],
    y: np.ndarray | None,
    batch_size: int = 64,
) -> DataLoader:
    if isinstance(X, Mapping):
        time_view = X.get("time", X.get("raw"))
        if time_view is None:
            raise ValueError("Evaluation mappings must include a time/raw view.")
        feature_views = {key: value for key, value in X.items() if key not in {"time", "raw"}}
        dataset = MultiViewEEGDataset(time_view, y, None, feature_views)
    else:
        dataset = MultiViewEEGDataset(X, y, None, None)
    pin_memory = bool(torch.cuda.is_available())
    num_workers = _recommended_num_workers(batch_size, len(dataset))
    loader_kwargs: dict[str, Any] = {
        "batch_size": min(batch_size, max(1, len(dataset))),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_eval_batch,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model_weights"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                metadata: dict[str, Any] = {}
                for candidate_key in ("config", "meta"):
                    candidate = checkpoint.get(candidate_key)
                    if isinstance(candidate, Mapping):
                        metadata.update(candidate)
                for candidate_key in (
                    "num_classes",
                    "hidden_channels",
                    "sampling_frequency",
                    "band_channels",
                    "band_method",
                    "band_segments",
                    "temperature",
                    "band_definitions",
                ):
                    if candidate_key in checkpoint:
                        metadata[candidate_key] = checkpoint[candidate_key]
                return value, metadata
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint, {}
    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict or a mapping containing "
        "`state_dict` / `model_state_dict`."
    )


def _sanitize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned[cleaned_key] = value
    return cleaned


def _infer_hidden_channels(state_dict: Mapping[str, torch.Tensor], default: int = 32) -> int:
    # Check new Conv2d temporal factorization first, then old Conv1d stem
    for key in ("temporal_conv.weight", "stem.0.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 3:
            return int(weight.shape[0])
    return default


def _infer_num_classes(
    labels: np.ndarray | None,
    state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> int:
    if labels is not None and labels.size > 0:
        return int(np.unique(labels).size)
    if "num_classes" in metadata:
        return int(metadata["num_classes"])
    weight = state_dict.get("classifier.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    raise ValueError("Could not infer number of classes from labels or checkpoint.")


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_input_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
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


def _move_eval_batch_to_device(
    batch: torch.Tensor | dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(batch, dict):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return batch.to(device, non_blocking=True)


@torch.no_grad()
def update_batchnorm_stats(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    batchnorm_layers = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not batchnorm_layers:
        return
    original_momenta = {}
    for module in batchnorm_layers:
        original_momenta[module] = module.momentum
        module.reset_running_stats()
        module.momentum = None
        module.num_batches_tracked.zero_()
    was_training = model.training
    model.train()
    for batch_x, _ in loader:
        batch_x = _move_eval_batch_to_device(batch_x, device)
        model(batch_x)
    for module in batchnorm_layers:
        module.momentum = original_momenta[module]
    model.train(was_training)


def load_model_and_data(
    model_path: str,
    data_dir: str,
) -> tuple[EEGNet, np.ndarray | Mapping[str, np.ndarray], np.ndarray | None, torch.device, np.ndarray | None]:
    model_file = Path(model_path).expanduser()
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")

    val_views, y_val, val_subjects, data_meta = load_prepared_split(data_dir, "val")

    checkpoint = torch.load(model_file, map_location="cpu")
    raw_state_dict, metadata = _extract_state_dict(checkpoint)
    state_dict = _sanitize_state_dict_keys(raw_state_dict)

    use_multiview = bool(metadata.get("use_multiview", True))
    ablation = str(metadata.get("ablation", "full"))
    selected_views = select_feature_views(
        val_views,
        use_multiview=use_multiview,
        ablation=ablation,
        strict=use_multiview,
    )
    time_view = selected_views["time"]

    in_channels = int(time_view.shape[1])
    num_classes = _infer_num_classes(y_val, state_dict, metadata)
    hidden_channels = int(metadata.get("hidden_channels", _infer_hidden_channels(state_dict)))
    sampling_frequency = float(metadata.get("sampling_frequency", 160.0))
    band_method = str(metadata.get("band_method", "fft_segments"))
    band_segments = int(metadata.get("band_segments", 4))
    temperature = float(metadata.get("temperature", 1.0))
    band_channels = int(
        metadata.get(
            "band_channels",
            selected_views["bands"].shape[1] if "bands" in selected_views else in_channels,
        )
    )
    band_definitions = metadata.get("band_definitions")
    parsed_band_definitions = None
    if isinstance(band_definitions, Mapping):
        parsed_band_definitions = {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in band_definitions.items()
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2
        }

    model = EEGNet(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        band_channels=band_channels,
        sampling_frequency=sampling_frequency,
        band_method=band_method,
        band_segments=band_segments,
        band_definitions=parsed_band_definitions,
        inference_temperature=temperature,
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with EEGNet initialization. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, selected_views, y_val, device, val_subjects


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    temperature: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    device = _model_device(model)
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    total_correct = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = _move_eval_batch_to_device(batch_x, device)
            logits = apply_model_temperature(model, model(batch_x), temperature=temperature)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            predictions.append(preds.cpu().numpy())
            probabilities.append(probs.cpu().numpy())

            if batch_y is not None:
                batch_y = batch_y.to(device, non_blocking=True)
                total_correct += int((preds == batch_y).sum().item())
                total_samples += int(batch_y.size(0))

    predictions_np = np.concatenate(predictions, axis=0) if predictions else np.empty(0, dtype=np.int64)
    probabilities_np = (
        np.concatenate(probabilities, axis=0).astype(np.float32, copy=False)
        if probabilities
        else np.empty((0, 0), dtype=np.float32)
    )
    accuracy = float(total_correct / total_samples) if total_samples > 0 else float("nan")
    return accuracy, predictions_np, probabilities_np


def compute_entropy(probs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    eps = 1e-8
    if isinstance(probs, torch.Tensor):
        stable_probs = probs.clamp_min(eps)
        return -(stable_probs * stable_probs.log()).sum(dim=-1)
    probs_array = np.asarray(probs, dtype=np.float32)
    stable_probs = np.clip(probs_array, eps, 1.0)
    return -(stable_probs * np.log(stable_probs)).sum(axis=-1)


def compute_confidence_entropy(
    probs: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"Expected probability array with shape (n_samples, n_classes), got {probs.shape}.")
    confidence = probs.max(axis=1)
    entropy = np.asarray(compute_entropy(probs), dtype=np.float32)
    if labels is None:
        return confidence, entropy, None
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError(
            f"Label count {labels.shape[0]} does not match probability count {probs.shape[0]}."
        )
    correctness = (np.argmax(probs, axis=1) == labels).astype(np.int64, copy=False)
    return confidence, entropy, correctness


def _bin_statistics(
    scores: np.ndarray,
    outcomes: np.ndarray | None,
    n_bins: int,
    *,
    value_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return {
            "centers": np.empty(0, dtype=np.float32),
            "counts": np.empty(0, dtype=np.int64),
            "mean_score": np.empty(0, dtype=np.float32),
            "mean_outcome": np.empty(0, dtype=np.float32),
            "edges": np.empty(0, dtype=np.float32),
        }

    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}.")

    if value_range is None:
        lower = float(np.min(scores))
        upper = float(np.max(scores))
        if np.isclose(lower, upper):
            lower -= 1e-6
            upper += 1e-6
    else:
        lower, upper = value_range
        if lower >= upper:
            raise ValueError(f"Invalid value_range: {value_range}.")

    edges = np.linspace(lower, upper, n_bins + 1, dtype=np.float32)
    bin_ids = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.int64, copy=False)
    score_sums = np.bincount(bin_ids, weights=scores, minlength=n_bins).astype(np.float32, copy=False)
    mean_score = np.divide(
        score_sums,
        counts,
        out=np.full(n_bins, np.nan, dtype=np.float32),
        where=counts > 0,
    )

    if outcomes is None:
        mean_outcome = np.full(n_bins, np.nan, dtype=np.float32)
    else:
        outcomes = np.asarray(outcomes, dtype=np.float32).reshape(-1)
        outcome_sums = np.bincount(bin_ids, weights=outcomes, minlength=n_bins).astype(np.float32, copy=False)
        mean_outcome = np.divide(
            outcome_sums,
            counts,
            out=np.full(n_bins, np.nan, dtype=np.float32),
            where=counts > 0,
        )

    return {
        "centers": centers,
        "counts": counts,
        "mean_score": mean_score,
        "mean_outcome": mean_outcome,
        "edges": edges,
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray | None, n_bins: int = 10) -> float:
    if labels is None or len(labels) == 0:
        return float("nan")
    confidence, _, correctness = compute_confidence_entropy(probs, labels)
    assert correctness is not None
    stats = _bin_statistics(confidence, correctness, n_bins, value_range=(0.0, 1.0))
    valid = stats["counts"] > 0
    if not np.any(valid):
        return float("nan")
    weights = stats["counts"][valid].astype(np.float32) / max(int(np.sum(stats["counts"][valid])), 1)
    ece = np.sum(weights * np.abs(stats["mean_outcome"][valid] - stats["mean_score"][valid]))
    return float(ece)


def compute_auc(probs: np.ndarray, labels: np.ndarray | None, num_classes: int) -> float:
    if roc_auc_score is None:
        LOGGER.warning("scikit-learn not found. Skipping AUC calculation.")
        return float("nan")
    if labels is None or len(labels) == 0 or probs.shape[0] != len(labels) or probs.shape[1] != num_classes:
        return float("nan")
    if num_classes <= 1:
        return float("nan")
    if num_classes == 2:
        return float(roc_auc_score(labels, probs[:, 1]))
    # Multi-class case: One-vs-Rest macro-average
    try:
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # This can happen if a class is missing in a small validation set
        return float("nan")


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is None or len(labels) == 0:
        ax.text(0.5, 0.5, "Labels unavailable\nECE not defined", ha="center", va="center")
        ax.set_axis_off()
        return fig

    confidence, _, correctness = compute_confidence_entropy(probs, labels)
    assert correctness is not None
    stats = _bin_statistics(confidence, correctness, n_bins=10, value_range=(0.0, 1.0))
    ece = compute_ece(probs, labels, n_bins=10)
    width = 1.0 / max(len(stats["centers"]), 1)

    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect calibration")
    valid = stats["counts"] > 0
    ax.bar(
        stats["centers"][valid],
        stats["mean_outcome"][valid],
        width=width * 0.9,
        color="#4C72B0",
        alpha=0.65,
        edgecolor="white",
        label="Observed accuracy",
    )
    ax.plot(
        stats["centers"][valid],
        stats["mean_score"][valid],
        color="#DD8452",
        marker="o",
        linewidth=2,
        label="Mean confidence",
    )
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability Diagram (ECE={ece:.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_accuracy_vs_uncertainty(entropy: np.ndarray, correct: np.ndarray | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    if correct is None or len(entropy) == 0:
        ax.text(0.5, 0.5, "Labels unavailable\nAccuracy curve not defined", ha="center", va="center")
        ax.set_axis_off()
        return fig

    stats = _bin_statistics(entropy, correct, n_bins=min(10, max(3, len(entropy))), value_range=None)
    valid = stats["counts"] > 0
    ax.plot(stats["centers"][valid], stats["mean_outcome"][valid], marker="o", linewidth=2, color="#C44E52")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Uncertainty")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_confidence_vs_accuracy(confidence: np.ndarray, correct: np.ndarray | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    if correct is None or len(confidence) == 0:
        ax.text(0.5, 0.5, "Labels unavailable\nAccuracy curve not defined", ha="center", va="center")
        ax.set_axis_off()
        return fig

    stats = _bin_statistics(confidence, correct, n_bins=10, value_range=(0.0, 1.0))
    valid = stats["counts"] > 0
    ax.plot(stats["centers"][valid], stats["mean_outcome"][valid], marker="o", linewidth=2, color="#55A868")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence vs Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_entropy_histogram(entropy: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    entropy = np.asarray(entropy, dtype=np.float32).reshape(-1)
    if entropy.size == 0:
        ax.text(0.5, 0.5, "No entropy values available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    bins = min(30, max(10, int(np.sqrt(entropy.size))))
    ax.hist(entropy, bins=bins, color="#8172B2", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Histogram")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def evaluate_reject_option(
    probs: np.ndarray,
    entropy: np.ndarray,
    labels: np.ndarray | None,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float32)
    entropy = np.asarray(entropy, dtype=np.float32).reshape(-1)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if probs.shape[0] != entropy.shape[0]:
        raise ValueError(
            f"Probability count {probs.shape[0]} does not match entropy count {entropy.shape[0]}."
        )

    preds = np.argmax(probs, axis=1)
    coverage = np.empty(thresholds.shape[0], dtype=np.float32)
    accuracy = np.full(thresholds.shape[0], np.nan, dtype=np.float32)

    for index, threshold in enumerate(thresholds):
        keep_mask = entropy <= threshold
        coverage[index] = float(np.mean(keep_mask)) if keep_mask.size > 0 else np.nan
        if labels is not None and np.any(keep_mask):
            accuracy[index] = float(np.mean(preds[keep_mask] == labels[keep_mask]))
    return coverage, accuracy


def compute_subject_auc(
    probs: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
    """Computes the mean and std of per-subject AUC scores."""
    if roc_auc_score is None:
        return float("nan"), float("nan")

    unique_subjects = np.unique(subjects)
    subject_aucs: list[float] = []

    for subject_id in unique_subjects:
        mask = subjects == subject_id
        if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
            continue  # AUC is not defined for single samples or single-class data.
        try:
            auc = compute_auc(probs[mask], labels[mask], num_classes)
            if not np.isnan(auc):
                subject_aucs.append(auc)
        except ValueError:
            continue  # May occur if a class is missing for a subject.

    if not subject_aucs:
        return float("nan"), float("nan")

    return float(np.mean(subject_aucs)), float(np.std(subject_aucs))


def compute_per_subject_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> dict[int, float]:
    """Computes accuracy for each subject."""
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1)
    results: dict[int, float] = {}
    for subject_id in np.unique(subjects):
        mask = subjects == subject_id
        results[int(subject_id)] = float(np.mean(predictions[mask] == labels[mask]))
    return results


def predict_proba(
    model: torch.nn.Module,
    x: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    *,
    temperature: float | None = None,
) -> torch.Tensor:
    device = _model_device(model)
    if isinstance(x, Mapping):
        inputs = {str(key): _to_input_tensor(value, device) for key, value in x.items() if value is not None}
    else:
        inputs = _to_input_tensor(x, device)
    model.eval()
    with torch.no_grad():
        logits = apply_model_temperature(model, model(inputs), temperature=temperature)
        probs = torch.softmax(logits, dim=-1)
    return probs


def mc_dropout_analysis(
    model: torch.nn.Module,
    X: np.ndarray | Mapping[str, np.ndarray],
    T: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    device = _model_device(model)
    if isinstance(X, Mapping):
        reference = X.get("time", X.get("raw"))
        if reference is None:
            raise ValueError("MC Dropout mappings must include a time/raw view.")
        n_samples = int(reference.shape[0])
    else:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (trials, channels, time), got {X.shape}.")
        n_samples = int(X.shape[0])

    batch_size = min(64, max(1, n_samples))
    mean_chunks: list[np.ndarray] = []
    var_chunks: list[np.ndarray] = []
    entropy_chunks: list[np.ndarray] = []
    mutual_info_chunks: list[np.ndarray] = []
    was_training = model.training

    enable_mc_dropout(model)
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            stop = start + batch_size
            if isinstance(X, Mapping):
                batch = {
                    str(key): torch.from_numpy(np.asarray(value[start:stop], dtype=np.float32)).to(device, non_blocking=True)
                    for key, value in X.items()
                }
            else:
                batch = torch.from_numpy(X[start:stop]).to(device, non_blocking=True)
            samples = [torch.softmax(apply_model_temperature(model, model(batch)), dim=-1).unsqueeze(0) for _ in range(T)]
            stacked = torch.cat(samples, dim=0)
            mean_probs = stacked.mean(dim=0)
            variance = stacked.var(dim=0, unbiased=False)
            predictive_entropy = compute_entropy(mean_probs)
            expected_entropy = compute_entropy(stacked).mean(dim=0)
            mutual_info = predictive_entropy - expected_entropy

            mean_chunks.append(mean_probs.cpu().numpy().astype(np.float32, copy=False))
            var_chunks.append(variance.cpu().numpy().astype(np.float32, copy=False))
            entropy_chunks.append(predictive_entropy.cpu().numpy().astype(np.float32, copy=False))
            mutual_info_chunks.append(mutual_info.cpu().numpy().astype(np.float32, copy=False))

    if was_training:
        model.train()
    else:
        model.eval()

    mean_probs_np = np.concatenate(mean_chunks, axis=0) if mean_chunks else np.empty((0, 0), dtype=np.float32)
    variance_np = np.concatenate(var_chunks, axis=0) if var_chunks else np.empty((0, 0), dtype=np.float32)
    entropy_np = np.concatenate(entropy_chunks, axis=0) if entropy_chunks else np.empty(0, dtype=np.float32)
    mutual_info_np = np.concatenate(mutual_info_chunks, axis=0) if mutual_info_chunks else np.empty(0, dtype=np.float32)
    return mean_probs_np, variance_np, entropy_np, mutual_info_np


def analyze_failures(
    probs: np.ndarray,
    entropy: np.ndarray,
    labels: np.ndarray | None,
) -> dict[str, Any]:
    if labels is None or len(labels) == 0:
        LOGGER.warning("Labels unavailable; skipping failure analysis.")
        return {
            "high_confidence_wrong_indices": np.empty(0, dtype=np.int64),
            "low_confidence_correct_indices": np.empty(0, dtype=np.int64),
        }

    labels = np.asarray(labels).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32)
    entropy = np.asarray(entropy, dtype=np.float32).reshape(-1)
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = preds == labels
    wrong_mask = ~correct

    wrong_indices = np.where(wrong_mask)[0]
    correct_indices = np.where(correct)[0]

    high_conf_cutoff = (
        float(np.quantile(confidence[wrong_indices], 0.75)) if wrong_indices.size > 0 else float("nan")
    )
    low_conf_cutoff = (
        float(np.quantile(confidence[correct_indices], 0.25)) if correct_indices.size > 0 else float("nan")
    )

    high_conf_wrong = (
        wrong_indices[confidence[wrong_indices] >= high_conf_cutoff] if wrong_indices.size > 0 else np.empty(0, dtype=np.int64)
    )
    low_conf_correct = (
        correct_indices[confidence[correct_indices] <= low_conf_cutoff]
        if correct_indices.size > 0
        else np.empty(0, dtype=np.int64)
    )

    def _print_cases(name: str, indices: np.ndarray) -> None:
        if indices.size == 0:
            print(f"{name}: none")
            return
        preview = indices[: min(10, indices.size)]
        print(f"{name}: {indices.size} samples")
        for idx in preview.tolist():
            print(
                f"  idx={idx}, label={int(labels[idx])}, pred={int(preds[idx])}, "
                f"conf={confidence[idx]:.4f}, entropy={entropy[idx]:.4f}"
            )

    _print_cases("High-confidence wrong predictions", high_conf_wrong)
    _print_cases("Low-confidence correct predictions", low_conf_correct)

    return {
        "high_confidence_wrong_indices": high_conf_wrong,
        "low_confidence_correct_indices": low_conf_correct,
        "high_confidence_threshold": high_conf_cutoff,
        "low_confidence_threshold": low_conf_cutoff,
    }


def _save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def _plot_reject_option_curve(
    thresholds: np.ndarray,
    coverage: np.ndarray,
    accuracy: np.ndarray,
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(thresholds, coverage, color="#4C72B0", linewidth=2, marker="o", label="Coverage")
    ax1.set_xlabel("Entropy threshold")
    ax1.set_ylabel("Coverage", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, accuracy, color="#DD8452", linewidth=2, marker="s", label="Accuracy")
    ax2.set_ylabel("Accuracy", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    fig.suptitle("Reject Option Analysis")
    fig.tight_layout()
    return fig


def _plot_mc_dropout_comparison(single_probs: np.ndarray, mc_entropy: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    if single_probs.size == 0 or mc_entropy.size == 0:
        ax.text(0.5, 0.5, "MC Dropout results unavailable", ha="center", va="center")
        ax.set_axis_off()
        return fig

    single_conf = np.max(single_probs, axis=1)
    ax.scatter(single_conf, mc_entropy, alpha=0.6, s=20, color="#937860")
    ax.set_xlabel("Single-pass confidence")
    ax.set_ylabel("MC Dropout entropy")
    ax.set_title("Single-pass Confidence vs MC Entropy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _describe_label_distribution(labels: np.ndarray | None) -> dict[str, int]:
    if labels is None:
        return {}
    counts = Counter(np.asarray(labels).reshape(-1).tolist())
    return {str(label): int(count) for label, count in sorted(counts.items(), key=lambda item: item[0])}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_fold_reports(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    report_paths = sorted(run_dir.glob("fold_*/fold_metrics.json"))
    if not report_paths:
        raise FileNotFoundError(f"No fold_metrics.json files were found under {run_dir}.")

    fold_reports: list[dict[str, Any]] = []
    for report_path in report_paths:
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                fold_reports.append(payload)

    metrics_to_aggregate = [
        "test_accuracy",
        "test_macro_f1",
        "test_auc",
        "test_ece",
        "pre_temperature_ece",
        "temperature",
        "subject_acc_mean",
        "subject_auc_mean",
    ]
    summary: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "num_folds": len(fold_reports),
        "folds": fold_reports,
        "metrics": {},
    }
    for metric_name in metrics_to_aggregate:
        values = np.asarray(
            [float(report.get(metric_name, float("nan"))) for report in fold_reports],
            dtype=np.float64,
        )
        valid = np.isfinite(values)
        mean_value = float(np.mean(values[valid])) if np.any(valid) else float("nan")
        std_value = float(np.std(values[valid])) if np.any(valid) else float("nan")
        summary["metrics"][metric_name] = {
            "mean": mean_value,
            "std": std_value,
            "values": [float(value) for value in values.tolist()],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "aggregate_summary.json", summary)
    _write_csv(output_dir / "fold_metrics.csv", fold_reports)
    summary_row = {"num_folds": int(len(fold_reports))}
    for metric_name in metrics_to_aggregate:
        summary_row[f"{metric_name}_mean"] = summary["metrics"][metric_name]["mean"]
        summary_row[f"{metric_name}_std"] = summary["metrics"][metric_name]["std"]
    _write_csv(output_dir / "aggregate_summary.csv", [summary_row])
    return summary


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.aggregate_folds = parse_bool(args.aggregate_folds)
    args.use_adabn = parse_bool(args.use_adabn)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_folds:
        if not args.run_dir:
            parser.error("--run-dir is required when --aggregate-folds true.")
        if args.mc_passes != 30 or args.ece_bins != 10:
            LOGGER.warning("--mc-passes and --ece-bins are ignored in --aggregate-folds mode.")
        summary = _aggregate_fold_reports(Path(args.run_dir), output_dir)
        print("\nAggregate summary")
        for metric_name, metric_payload in summary["metrics"].items():
            print(f"{metric_name}: {metric_payload['mean']:.4f} +- {metric_payload['std']:.4f}")
        print(f"Saved reports: {output_dir}")
        return

    if not args.model_path or not args.data_dir:
        parser.error("--model-path and --data-dir are required for single-run evaluation.")

    # Load train subjects for leakage check, and validation data for evaluation.
    _subjects_path = Path(args.data_dir) / "train_subjects.npy"
    train_subjects = (
        np.load(_subjects_path, mmap_mode="r", allow_pickle=False)
        if _subjects_path.exists() else None
    )
    model, X_val, y_val, device, val_subjects = load_model_and_data(
        args.model_path, args.data_dir
    )

    # CRITICAL: Verify no subject leakage between train and validation sets
    if train_subjects is not None and val_subjects is not None:
        train_set = set(train_subjects)
        val_set = set(val_subjects)
        if not train_set.isdisjoint(val_set):
            raise ValueError(f"DATA LEAKAGE DETECTED! Overlapping subjects found: {train_set.intersection(val_set)}")
        else:
            print("[INFO] Data leakage check passed: Train and validation subject sets are disjoint.")


    dataloader = create_eval_dataloader(X_val, y_val, batch_size=args.batch_size)
    if args.use_adabn:
        update_batchnorm_stats(dataloader, model, device)

    raw_accuracy, raw_predictions, raw_probabilities = evaluate_model(model, dataloader, temperature=1.0)
    accuracy, predictions, probabilities = evaluate_model(model, dataloader, temperature=None)
    confidence, entropy, correctness = compute_confidence_entropy(probabilities, y_val)
    raw_ece = compute_ece(raw_probabilities, y_val, n_bins=args.ece_bins)
    ece = compute_ece(probabilities, y_val, n_bins=args.ece_bins)
    auc = compute_auc(probabilities, y_val, model.num_classes)
    
    per_class_auc = None
    if roc_auc_score and y_val is not None and probabilities.shape[0] == y_val.shape[0]:
        try:
            per_class_auc = roc_auc_score(y_val, probabilities, multi_class="ovr", average=None)
        except ValueError:
            per_class_auc = None

    plot_paths = {
        "reliability": _save_figure(plot_reliability_diagram(probabilities, y_val), output_dir, "reliability_diagram.png"),
        "accuracy_vs_uncertainty": _save_figure(
            plot_accuracy_vs_uncertainty(entropy, correctness),
            output_dir,
            "accuracy_vs_uncertainty.png",
        ),
        "confidence_vs_accuracy": _save_figure(
            plot_confidence_vs_accuracy(confidence, correctness),
            output_dir,
            "confidence_vs_accuracy.png",
        ),
        "entropy_histogram": _save_figure(plot_entropy_histogram(entropy), output_dir, "entropy_histogram.png"),
    }

    if entropy.size > 0:
        thresholds = np.linspace(float(np.min(entropy)), float(np.max(entropy)), num=min(20, max(5, entropy.size)))
    else:
        thresholds = np.linspace(0.0, 1.0, num=10, dtype=np.float32)
    coverage, reject_accuracy = evaluate_reject_option(probabilities, entropy, y_val, thresholds)
    plot_paths["reject_option"] = _save_figure(
        _plot_reject_option_curve(thresholds, coverage, reject_accuracy),
        output_dir,
        "reject_option.png",
    )

    mc_mean_probs, mc_variance, mc_entropy, mc_mutual_info = mc_dropout_analysis(model, X_val, T=args.mc_passes)
    plot_paths["mc_dropout_comparison"] = _save_figure(
        _plot_mc_dropout_comparison(probabilities, mc_entropy),
        output_dir,
        "mc_dropout_comparison.png",
    )

    failure_summary = analyze_failures(probabilities, entropy, y_val)

    subject_accuracies: dict[int, float] | None = None
    if y_val is not None and val_subjects is not None:
        subject_accuracies = compute_per_subject_metrics(predictions, y_val, val_subjects)

    subject_auc_mean, subject_auc_std = float("nan"), float("nan")
    if y_val is not None and val_subjects is not None:
        subject_auc_mean, subject_auc_std = compute_subject_auc(
            probabilities, y_val, val_subjects, model.num_classes
        )

    avg_entropy = float(np.mean(entropy)) if entropy.size > 0 else float("nan")
    avg_confidence = float(np.mean(confidence)) if confidence.size > 0 else float("nan")
    mc_avg_entropy = float(np.mean(mc_entropy)) if mc_entropy.size > 0 else float("nan")
    mc_avg_variance = float(np.mean(mc_variance)) if mc_variance.size > 0 else float("nan")
    mc_avg_mutual_info = float(np.mean(mc_mutual_info)) if mc_mutual_info.size > 0 else float("nan")
    single_vs_mc_gap = (
        float(np.mean(np.max(probabilities, axis=1)) - np.mean(np.max(mc_mean_probs, axis=1)))
        if probabilities.size > 0 and mc_mean_probs.size > 0
        else float("nan")
    )

    summary_payload = {
        "accuracy": float(accuracy),
        "raw_accuracy": float(raw_accuracy),
        "ece": float(ece),
        "raw_ece": float(raw_ece),
        "auc": float(auc),
        "avg_confidence": float(avg_confidence),
        "avg_entropy": float(avg_entropy),
        "mc_avg_entropy": float(mc_avg_entropy),
        "mc_avg_variance": float(mc_avg_variance),
        "mc_avg_mutual_info": float(mc_avg_mutual_info),
        "single_vs_mc_gap": float(single_vs_mc_gap),
        "temperature": float(getattr(model, "inference_temperature", 1.0)),
        "subject_auc_mean": float(subject_auc_mean),
        "subject_auc_std": float(subject_auc_std),
        "plots": plot_paths,
    }
    _save_json(output_dir / "evaluation_summary.json", summary_payload)

    print("\nFinal summary")
    print(f"Device: {device}")
    val_shape = (
        {key: tuple(int(dim) for dim in value.shape) for key, value in X_val.items()}
        if isinstance(X_val, Mapping)
        else tuple(int(dim) for dim in X_val.shape)
    )
    print(f"Validation shape: {val_shape}")
    print(f"Label distribution: {_describe_label_distribution(y_val) or 'unlabeled'}")
    print(f"Accuracy: {accuracy:.4f}" if not np.isnan(accuracy) else "Accuracy: nan")
    print(f"Raw Accuracy: {raw_accuracy:.4f}" if not np.isnan(raw_accuracy) else "Raw Accuracy: nan")
    print(f"ECE: {ece:.4f}" if not np.isnan(ece) else "ECE: nan")
    print(f"Pre-Temp ECE: {raw_ece:.4f}" if not np.isnan(raw_ece) else "Pre-Temp ECE: nan")
    print(f"AUC (Macro OvR): {auc:.4f}" if not np.isnan(auc) else "AUC: nan")
    if per_class_auc is not None:
        auc_str = " | ".join([f"C{i}: {v:.3f}" for i, v in enumerate(per_class_auc)])
        print(f"Per-Class AUC: [ {auc_str} ]")

    # Per-class F1 scores and confusion matrix
    if y_val is not None and len(predictions) > 0:
        n_cls = model.num_classes
        cm = np.zeros((n_cls, n_cls), dtype=np.int64)
        for t, p in zip(y_val.astype(np.int64), predictions.astype(np.int64)):
            cm[t, p] += 1
        tp = np.diag(cm).astype(np.float64)
        precision_arr = tp / np.clip(cm.sum(axis=0).astype(np.float64), 1, None)
        recall_arr = tp / np.clip(cm.sum(axis=1).astype(np.float64), 1, None)
        f1_arr = 2.0 * precision_arr * recall_arr / np.clip(precision_arr + recall_arr, 1e-8, None)
        macro_f1 = float(np.mean(f1_arr))
        print(f"Macro F1: {macro_f1:.4f}")
        f1_str = " | ".join([f"C{i}: {v:.3f}" for i, v in enumerate(f1_arr)])
        print(f"Per-Class F1: [ {f1_str} ]")
        prec_str = " | ".join([f"C{i}: {v:.3f}" for i, v in enumerate(precision_arr)])
        rec_str = " | ".join([f"C{i}: {v:.3f}" for i, v in enumerate(recall_arr)])
        print(f"Per-Class Precision: [ {prec_str} ]")
        print(f"Per-Class Recall:    [ {rec_str} ]")
        print(f"Confusion Matrix:\n{cm}")

    print(f"Avg Confidence: {avg_confidence:.4f}" if not np.isnan(avg_confidence) else "Avg Confidence: nan")
    print(f"Avg Entropy: {avg_entropy:.4f}" if not np.isnan(avg_entropy) else "Avg Entropy: nan")
    print(f"MC Avg Entropy: {mc_avg_entropy:.4f}" if not np.isnan(mc_avg_entropy) else "MC Avg Entropy: nan")
    print(f"MC Avg Mutual Info: {mc_avg_mutual_info:.4f}" if not np.isnan(mc_avg_mutual_info) else "MC Avg Mutual Info: nan")
    if subject_accuracies is not None:
        subject_values = np.asarray(list(subject_accuracies.values()), dtype=np.float32)
        print(f"Mean Subject Acc: {subject_values.mean():.4f}")
        print(f"Std Subject Acc: {subject_values.std():.4f}")
        print(f"Min/Max Subject Acc: {subject_values.min():.4f} / {subject_values.max():.4f}")
        print("--- Per-Subject Accuracy ---")
        for subject_id, acc in sorted(subject_accuracies.items(), key=lambda item: item[0]):
            print(f"  - Subject {subject_id}: {acc:.4f}")
        print("--------------------------")
    if not np.isnan(subject_auc_mean):
        print(f"Mean Subject AUC: {subject_auc_mean:.4f}")
        print(f"Std Subject AUC: {subject_auc_std:.4f}")
    print(f"MC Avg Variance: {mc_avg_variance:.6f}" if not np.isnan(mc_avg_variance) else "MC Avg Variance: nan")
    print(
        f"Single-vs-MC confidence gap: {single_vs_mc_gap:.4f}"
        if not np.isnan(single_vs_mc_gap)
        else "Single-vs-MC confidence gap: nan"
    )
    print(f"High-confidence errors: {len(failure_summary.get('high_confidence_wrong_indices', []))} samples")
    print(f"High-uncertainty correct: {len(failure_summary.get('low_confidence_correct_indices', []))} samples")
    print(f"Saved plots: {plot_paths}")


if __name__ == "__main__":
    main()