from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
import tempfile
from collections import Counter          # FIX-1: was missing -- caused NameError crash
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence         # FIX-2: Sequence needed for _indices_for_subjects

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_USE_SHM", "0")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the research-grade multi-branch EEG model.")
    parser.add_argument("--data-dir", type=str, required=True, help="Prepared dataset directory with train/val or all_* arrays.")
    parser.add_argument("--output-dir", type=str, default="training_outputs", help="Directory for checkpoints and plots.")
    parser.add_argument("--epochs", type=int, default=200)         # P95v6: 200 epochs — folds 1-2 still climbing at 150
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)          # P95v3 proven: 1e-3
    parser.add_argument("--weight-decay", type=float, default=5e-3) # P95v3: stronger regularization
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)      # P95v3: reduced for D=2
    parser.add_argument("--scheduler", choices=("onecycle", "cosine", "cosine_restarts", "cosine_single_restart"), default="cosine_single_restart")
    parser.add_argument("--loss", choices=("weighted_ce", "focal"), default="focal")
    parser.add_argument("--focal-gamma", type=float, default=1.5)  # P95v3: less aggressive for 3-class
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup alpha. Disabled by default (0.0) as it can harm EEG signals.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing.")
    parser.add_argument("--cutmix-alpha", type=float, default=0.3, help="CutMix alpha for EEG-safe temporal segment swapping.")
    parser.add_argument("--cutmix-prob", type=float, default=0.20, help="P95v3: reduced CutMix probability.")
    parser.add_argument("--rdrop-alpha", type=float, default=0.0, help="R-Drop KL consistency regularization weight.")
    parser.add_argument("--time-mask-prob", type=float, default=0.3, help="Probability of time masking.")
    parser.add_argument("--time-mask-ratio", type=float, default=0.1)
    parser.add_argument("--time-masks-per-sample", type=int, default=1)
    parser.add_argument("--noise-prob", type=float, default=0.6)   # P95v3: more aggressive
    parser.add_argument("--noise-std", type=float, default=0.04)   # P95v3: stronger noise
    parser.add_argument("--channel-dropout-prob", type=float, default=0.35, help="Probability of channel dropout.")
    parser.add_argument("--channel-dropout-ratio", type=float, default=0.2)
    parser.add_argument("--temporal-shift-prob", type=float, default=0.4, help="Probability of temporal shift.")
    parser.add_argument("--temporal-shift-ratio", type=float, default=0.05)
    parser.add_argument("--scaling-prob", type=float, default=0.4, help="Probability of random scaling.")
    parser.add_argument("--scaling-min", type=float, default=0.85)
    parser.add_argument("--scaling-max", type=float, default=1.15)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=50)          # P95v6: more patience for 200 epochs
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9997)  # P95v6: slower EMA for longer training
    parser.add_argument("--use-ema", type=str, default="true")
    parser.add_argument("--use-swa", type=str, default="false")
    parser.add_argument("--swa-start", type=float, default=0.75)
    parser.add_argument("--tta-passes", type=int, default=15)        # P95: more TTA views
    parser.add_argument("--tta-noise-std", type=float, default=0.015)
    parser.add_argument("--tta-shift-ratio", type=float, default=0.03)
    parser.add_argument("--disable-tta", type=str, default="false")
    parser.add_argument("--use-multiview", type=str, default="true")
    parser.add_argument("--eval-model", choices=("auto", "base", "ema", "swa"), default="auto")
    parser.add_argument("--use-amp", type=str, default="true")
    parser.add_argument("--debug-validation", type=str, default="false")
    parser.add_argument("--fail-on-nonfinite", type=str, default="false")
    parser.add_argument("--use-adabn", type=str, default="true", help="Apply AdaBN using calibration subjects before test inference.")
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--ablation", type=str, default="full", choices=("full", "time", "time_freq", "time_bands"))
    parser.add_argument("--split-mode", type=str, default="subject_kfold", choices=("holdout", "subject_kfold"))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=-1, help="0-based fold index to run, or -1 to run all folds.")
    parser.add_argument("--inner-val-ratio", type=float, default=0.10)
    parser.add_argument("--calibration-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal-start-epoch", type=int, default=1)
    parser.add_argument("--aug-warmup-epochs", type=int, default=0)  # P95v3: no warmup — full aug from epoch 1
    parser.add_argument("--aug-rampup-epochs", type=int, default=10) # P95v6: slightly longer rampup for cleaner early learning
    parser.add_argument("--balanced-sampling", type=str, default="false", help="Use WeightedRandomSampler to address class imbalance.")
    # NOVEL: Domain Adversarial Training (DANN) for cross-subject invariance
    parser.add_argument("--dann-lambda", type=float, default=0.08, help="NOVEL: domain adversarial for cross-subject invariance.")
    parser.add_argument("--dann-warmup-epochs", type=int, default=5, help="P95v3: early DANN activation.")
    # NOVEL: Supervised contrastive loss for class separability
    parser.add_argument("--supcon-weight", type=float, default=0.10, help="NOVEL: contrastive clustering for class separability.")
    # P95v3: SAM disabled by default
    parser.add_argument("--use-sam", type=str, default="false", help="P95v3: disabled — no benefit at this model size.")
    parser.add_argument("--sam-rho", type=float, default=0.02, help="SAM neighborhood radius.")
    parser.add_argument("--sam-warmup-epochs", type=int, default=30, help="SAM warmup epochs.")
    # Head LR multiplier (tune from cell)
    parser.add_argument("--head-lr-mult", type=float, default=0.5, help="P95v6: 0.5x head LR — compromise between 0.3 (too slow) and 1.0 (overshoot).")
    # P95: MC-Dropout inference
    parser.add_argument("--mc-dropout-passes", type=int, default=5, help="MC-Dropout passes at inference.")
    # P95: Frequency augmentations
    parser.add_argument("--freq-aug-prob", type=float, default=0.3, help="Probability of frequency-domain augmentation.")
    return parser


if __name__ == "__main__" and any(arg in sys.argv for arg in ("-h", "--help")):
    _build_arg_parser().print_help()
    raise SystemExit(0)

import numpy as np

if os.environ.get("DISPLAY", "") == "" and "COLAB_RELEASE_TAG" not in os.environ:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import WeightedRandomSampler

from phase2 import (
    EEGNet,
    FocalCrossEntropy,
    FocalSoftTargetCrossEntropy,
    MultiViewEEGDataset,
    WeightedSoftTargetCrossEntropy,
    _collate_eeg_batch,
    apply_channel_dropout,
    apply_gaussian_noise,
    apply_random_scaling,
    apply_time_mask,
    apply_temporal_shift,
    apply_model_temperature,
    create_dataloaders,
    load_canonical_data,
    load_prepared_data,
    mixup_batch,
    refresh_multiview_batch,
    select_feature_views,
)
from evaluation import (
    compute_auc,
    compute_ece,
    compute_subject_auc
)

LOGGER = logging.getLogger("train")

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "training_outputs"
    epochs: int = 200                    # P95v6: 200 epochs for full convergence
    batch_size: int = 128
    lr: float = 1e-3                     # P95v3 proven
    weight_decay: float = 5e-3           # P95v3: stronger L2 to fight subject memorization
    hidden_channels: int = 64            # P95v3: keep 64ch but with D=2 (128 spatial maps)
    dropout: float = 0.30                # P95v3: reduced — D=2 needs less regularization
    scheduler: str = "cosine_single_restart"  # P95v6: single restart to escape plateaus
    loss: str = "focal"
    focal_gamma: float = 1.5             # P95v3: less aggressive for 3-class (2.0 over-weighted noise)
    mixup_alpha: float = 0.0
    label_smoothing: float = 0.05
    cutmix_alpha: float = 0.3
    cutmix_prob: float = 0.20            # P95v3: reduced from 0.30
    rdrop_alpha: float = 0.0
    # Augmentation
    time_mask_prob: float = 0.3
    time_mask_ratio: float = 0.1
    time_masks_per_sample: int = 1
    noise_prob: float = 0.6              # P95v3: more aggressive noise for robustness
    noise_std: float = 0.04              # P95v3: stronger noise forces robust features
    channel_dropout_prob: float = 0.35
    channel_dropout_ratio: float = 0.2
    temporal_shift_prob: float = 0.4
    temporal_shift_ratio: float = 0.05
    scaling_prob: float = 0.4
    scaling_min: float = 0.85
    scaling_max: float = 1.15
    max_grad_norm: float = 1.0
    patience: int = 50                   # P95v6: more patience for 200 epochs
    min_delta: float = 1e-4
    ema_decay: float = 0.9997            # P95v6: slower EMA for longer training
    use_ema: bool = True
    use_swa: bool = False
    swa_start: float = 0.75
    tta_passes: int = 15                 # P95: more TTA views
    tta_noise_std: float = 0.015
    tta_shift_ratio: float = 0.03
    disable_tta: bool = False
    use_multiview: bool = True
    ablation: str = "full"
    split_mode: str = "subject_kfold"
    num_folds: int = 5
    fold_index: int = -1
    inner_val_ratio: float = 0.10
    calibration_ratio: float = 0.10
    eval_model: str = "auto"
    use_amp: bool = True
    debug_validation: bool = False
    fail_on_nonfinite: bool = False
    use_adabn: bool = True
    max_val_batches: int = 0
    seed: int = 42
    focal_start_epoch: int = 1
    aug_warmup_epochs: int = 0           # P95v3: NO warmup — full augmentation from epoch 1
    aug_rampup_epochs: int = 10          # P95v6: slightly longer rampup
    balanced_sampling: bool = False
    # NOVEL: Domain Adversarial Training (DANN)
    dann_lambda: float = 0.08            # NOVEL: domain adversarial for cross-subject invariance
    dann_warmup_epochs: int = 5          # P95v3: activate DANN early to prevent subject memorization
    # NOVEL: Supervised contrastive loss
    supcon_weight: float = 0.10          # NOVEL: contrastive clustering for class separability
    # P95v3: SAM disabled — caused instability, no benefit at this model size
    use_sam: bool = False
    sam_rho: float = 0.02
    sam_warmup_epochs: int = 30
    # Head LR multiplier (tune from cell)
    head_lr_mult: float = 0.5            # P95v6: compromise between 0.3 (too slow) and 1.0 (overshoot)
    # P95: MC-Dropout inference
    mc_dropout_passes: int = 5
    # P95: Frequency augmentations
    freq_aug_prob: float = 0.3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _feature_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    preprocessing = metadata.get("preprocessing", {}) if isinstance(metadata.get("preprocessing"), dict) else {}
    feature_config = metadata.get("feature_config", {}) if isinstance(metadata.get("feature_config"), dict) else {}
    return {
        "sampling_frequency": float(
            metadata.get("sampling_frequency_inferred")
            or metadata.get("sampling_frequency")
            or preprocessing.get("sampling_frequency")
            or 160.0
        ),
        "band_method": str(feature_config.get("band_method", preprocessing.get("band_method", "fft_segments"))),
        "band_segments": int(feature_config.get("band_segments", preprocessing.get("band_segments", 8))),
        "include_gamma": bool(feature_config.get("include_gamma", preprocessing.get("include_gamma", False))),
    }


def _subset_feature_views(
    feature_views: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        key: np.ascontiguousarray(np.asarray(value)[indices], dtype=np.float32)
        for key, value in feature_views.items()
    }


def _subset_labels(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(labels)[indices], dtype=np.int64)


def _subset_subjects(subjects: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(subjects)[indices], dtype=np.int32)


def _subject_kfold_subjects(subjects: np.ndarray, num_folds: int, seed: int) -> list[np.ndarray]:
    subjects_array = np.asarray(subjects).reshape(-1)
    unique_subjects = np.unique(subjects_array)
    if unique_subjects.size < 2:
        raise ValueError("Subject k-fold training requires at least two unique subjects.")
    if num_folds < 2:
        raise ValueError(f"num_folds must be at least 2, got {num_folds}.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_subjects)
    return [np.asarray(fold, dtype=np.int32) for fold in np.array_split(shuffled, num_folds) if len(fold) > 0]


def _split_inner_subjects(
    subjects: np.ndarray,
    *,
    inner_val_ratio: float,
    calibration_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_subjects = np.asarray(np.unique(np.asarray(subjects).reshape(-1)), dtype=np.int32)
    if unique_subjects.size < 3:
        raise ValueError("Inner split requires at least three unique subjects.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_subjects)

    n_subjects = int(shuffled.size)
    n_val = max(1, int(round(n_subjects * float(inner_val_ratio))))
    n_calib = max(1, int(round(n_subjects * float(calibration_ratio))))
    if n_val + n_calib >= n_subjects:
        overflow = (n_val + n_calib) - (n_subjects - 1)
        n_calib = max(1, n_calib - overflow)
        if n_val + n_calib >= n_subjects:
            n_val = max(1, n_subjects - n_calib - 1)

    val_subjects = np.asarray(shuffled[:n_val], dtype=np.int32)
    calib_subjects = np.asarray(shuffled[n_val:n_val + n_calib], dtype=np.int32)
    train_subjects = np.asarray(shuffled[n_val + n_calib:], dtype=np.int32)
    if train_subjects.size == 0:
        raise ValueError("Inner subject split produced an empty training subject set.")
    if train_subjects.size < 5:
        LOGGER.warning(
            "Inner split has only %d training subjects (val=%d, calib=%d, total=%d). "
            "This can produce high-variance fold results.",
            int(train_subjects.size),
            int(val_subjects.size),
            int(calib_subjects.size),
            int(n_subjects),
        )
    return train_subjects, val_subjects, calib_subjects


def _indices_for_subjects(subjects: np.ndarray, target_subjects: Sequence[int]) -> np.ndarray:
    subject_array = np.asarray(subjects).reshape(-1)
    mask = np.isin(subject_array, np.asarray(list(target_subjects), dtype=subject_array.dtype))
    indices = np.where(mask)[0].astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError(f"No samples matched target subjects: {list(target_subjects)}")
    return indices


def _macro_f1_from_predictions(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(preds, targets, num_classes)
    return macro_f1_from_cm(cm)


def _per_subject_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> tuple[float, float, float, float]:
    subject_array = np.asarray(subjects).reshape(-1)
    accuracies: list[float] = []
    for subject_id in np.unique(subject_array):
        mask = subject_array == subject_id
        accuracies.append(float(np.mean(predictions[mask] == labels[mask])))
    if not accuracies:
        return float("nan"), float("nan"), float("nan"), float("nan")
    values = np.asarray(accuracies, dtype=np.float32)
    return float(values.mean()), float(values.std()), float(values.min()), float(values.max())


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _recommended_num_workers(batch_size: int, dataset_size: int) -> int:
    cpu_count = os.cpu_count() or 1
    if dataset_size < max(batch_size * 2, 64):
        return 0
    if sys.platform == "darwin":
        return min(2, max(0, cpu_count - 1))
    return min(4, cpu_count)


def _dataset_components(
    feature_views: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time_view = np.asarray(feature_views.get("time", feature_views.get("raw")), dtype=np.float32)
    other_views = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in feature_views.items()
        if key not in {"time", "raw"}
    }
    return time_view, other_views


def _make_loader(
    feature_views: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
) -> torch.utils.data.DataLoader:
    time_view, other_views = _dataset_components(feature_views, labels, subjects)
    dataset = MultiViewEEGDataset(time_view, labels, subjects, other_views)
    num_workers = _recommended_num_workers(batch_size, len(dataset))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "collate_fn": _collate_eeg_batch,
        "drop_last": bool((shuffle or sampler is not None) and len(dataset) > batch_size),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    if sampler is not None:
        return torch.utils.data.DataLoader(dataset, sampler=sampler, **loader_kwargs)
    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_kwargs)


def _create_train_val_loaders(
    train_views: dict[str, np.ndarray],
    y_train: np.ndarray,
    train_subjects: np.ndarray,
    val_views: dict[str, np.ndarray],
    y_val: np.ndarray,
    val_subjects: np.ndarray,
    cfg: TrainConfig,
    num_classes: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # FIX-3: WeightedRandomSampler equalises training batches but validation/test
    # retain the TRUE class distribution. If imbalance ratio >1.5 this mismatch
    # causes the model to miscalibrate class 0, collapsing val_acc below chance.
    # Prefer --balanced-sampling false with focal loss class weights instead.
    if cfg.balanced_sampling:
        counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
        max_imbalance = float(counts.max() / max(float(counts.min()), 1.0))
        if max_imbalance >= 1.5:
            LOGGER.warning(
                "[BALANCED] class imbalance ratio=%.2fx -- sampler will cause val-set "
                "calibration mismatch. Consider --balanced-sampling false + focal weights.",
                max_imbalance,
            )
        class_weight = 1.0 / np.maximum(counts, 1.0)
        sample_weights = class_weight[y_train.astype(np.int64)]
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(y_train),
            replacement=True,
        )
        print(f"[BALANCED] WeightedRandomSampler: class_counts={counts.astype(int).tolist()}")
        train_loader = _make_loader(
            train_views,
            y_train,
            train_subjects,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
        )
    else:
        train_loader = _make_loader(
            train_views,
            y_train,
            train_subjects,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
    val_loader = _make_loader(
        val_views,
        y_val,
        val_subjects,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def _learn_temperature(
    model: EEGNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    temp_optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    all_logits_list: list[torch.Tensor] = []
    all_labels_list: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for bx, by, _ in loader:
            if by is None:
                continue
            bx = _move_batch_to_device(bx, device)
            by = by.to(device, non_blocking=True)
            with autocast_context(device, enabled=use_amp):
                logits = torch.clamp(model(bx), -20.0, 20.0)
            all_logits_list.append(logits)
            all_labels_list.append(by)

    if not all_logits_list:
        return 1.0

    all_logits_cat = torch.cat(all_logits_list, dim=0)
    all_labels_cat = torch.cat(all_labels_list, dim=0)
    nll_criterion = torch.nn.CrossEntropyLoss()

    def _temp_closure():
        temp_optimizer.zero_grad()
        loss = nll_criterion(all_logits_cat / temperature.clamp(min=0.1), all_labels_cat)
        loss.backward()
        return loss

    temp_optimizer.step(_temp_closure)
    return float(temperature.item())


def create_grad_scaler(
    device: torch.device,
    *,
    enabled: bool = True,
) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, *, enabled: bool = True):
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(ema_value):
                ema_value.copy_(model_value)
                continue
            ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)


@torch.no_grad()
def update_batchnorm_stats(
    loader: torch.utils.data.DataLoader,
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
    for batch_x, _, _ in loader:
        batch_x = _move_batch_to_device(batch_x, device)
        model(batch_x)

    for module in batchnorm_layers:
        module.momentum = original_momenta[module]
    model.train(was_training)


def _move_batch_to_device(batch: torch.Tensor | dict[str, torch.Tensor], device: torch.device):
    if isinstance(batch, dict):
        return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return batch.to(device, non_blocking=True)


def _batch_time_shape(batch: torch.Tensor | dict[str, torch.Tensor]) -> tuple[int, int]:
    if isinstance(batch, dict):
        tensor = batch["time"] if "time" in batch else batch["raw"]
    else:
        tensor = batch
    return int(tensor.shape[1]), int(tensor.shape[2])


def _first_nonfinite_tensor_in_module(model: torch.nn.Module) -> tuple[str, tuple[int, ...], str] | None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            return name, tuple(int(dim) for dim in parameter.shape), "parameter"
    for name, buffer in model.named_buffers():
        if not torch.isfinite(buffer).all():
            return name, tuple(int(dim) for dim in buffer.shape), "buffer"
    return None


def _tensor_debug_stats(tensor: torch.Tensor) -> str:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return f"shape={tuple(int(dim) for dim in tensor.shape)} finite=0"
    return (
        f"shape={tuple(int(dim) for dim in tensor.shape)} "
        f"min={float(finite.min()):.6g} "
        f"max={float(finite.max()):.6g} "
        f"mean={float(finite.mean()):.6g} "
        f"std={float(finite.std(unbiased=False)):.6g}"
    )


def _debug_check_batch_inputs(
    batch_x: torch.Tensor | dict[str, torch.Tensor],
    *,
    prefix: str,
    fail_on_nonfinite: bool,
) -> None:
    if isinstance(batch_x, dict):
        bad_keys = [key for key, value in batch_x.items() if not torch.isfinite(value).all()]
        if bad_keys:
            print(f"[VAL-DEBUG] {prefix}: non-finite values in views {bad_keys}")
            for key in bad_keys:
                print(f"[VAL-DEBUG] {prefix}:{key} {_tensor_debug_stats(batch_x[key])}")
            if fail_on_nonfinite:
                raise FloatingPointError(f"{prefix}: non-finite values in input views {bad_keys}")
    else:
        if not torch.isfinite(batch_x).all():
            print(f"[VAL-DEBUG] {prefix}: non-finite values in tensor input")
            print(f"[VAL-DEBUG] {prefix} {_tensor_debug_stats(batch_x)}")
            if fail_on_nonfinite:
                raise FloatingPointError(f"{prefix}: non-finite values in input tensor")


def _select_validation_model(
    *,
    cfg: TrainConfig,
    model: EEGNet,
    ema: ModelEMA | None,
    swa_model: AveragedModel | None,
    swa_start_epoch: int | None,
    epoch: int,
) -> tuple[torch.nn.Module, str]:
    mode = str(getattr(cfg, "eval_model", "auto")).lower()
    if mode == "base":
        return model, "base"
    if mode == "ema":
        if ema is None:
            raise ValueError("Validation requested EMA model, but EMA is disabled.")
        return ema.ema_model, "ema"
    if mode == "swa":
        if swa_model is None or swa_start_epoch is None or epoch < swa_start_epoch:
            raise ValueError("Validation requested SWA model, but SWA is unavailable at this epoch.")
        return swa_model, "swa"
    if swa_model is not None and swa_start_epoch is not None and epoch >= swa_start_epoch:
        return swa_model, "swa-auto"
    if ema is not None:
        return ema.ema_model, "ema-auto"
    return model, "base-auto"


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1e-3)
    imbalance_ratio = float(counts.max() / max(float(counts.min()), 1e-3))
    if imbalance_ratio < 1.1:
        return torch.ones(num_classes, dtype=torch.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets.astype(np.int64), preds.astype(np.int64)):
        cm[target, pred] += 1
    return cm


def accuracy_from_cm(cm: np.ndarray) -> float:
    return float(np.trace(cm) / max(cm.sum(), 1))


def macro_f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.clip(cm.sum(axis=0), 1, None)
    recall = tp / np.clip(cm.sum(axis=1), 1, None)
    f1 = 2.0 * precision * recall / np.clip(precision + recall, 1e-8, None)
    return float(np.mean(f1))


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    plt.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = -float("inf")
        self.counter = 0

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _build_hard_criterion(
    loss_name: str,
    class_weights: torch.Tensor,
    focal_gamma: float,
    label_smoothing: float,
) -> torch.nn.Module:
    if loss_name == "focal":
        return FocalCrossEntropy(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    return torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def infer_sampling_frequency(metadata: dict[str, Any], time_length: int) -> float:
    for key in ("sampling_frequency_inferred", "sampling_frequency"):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    preprocessing = metadata.get("preprocessing", {})
    if isinstance(preprocessing, dict):
        window_sec = preprocessing.get("window_sec")
        if window_sec is not None and float(window_sec) > 0:
            return float(time_length) / float(window_sec)
    common_lengths = {
        320: 160.0,
        500: 250.0,
        625: 250.0,
        640: 160.0,
        1000: 250.0,
        1250: 250.0,
        1375: 250.0,
    }
    if time_length in common_lengths:
        return common_lengths[time_length]
    return max(float(time_length) / 2.0, 1.0)


def compute_dropout_scale(epoch: int, total_epochs: int) -> float:
    # FIX: Stronger regularization late in training.
    progress = float(epoch) / max(float(total_epochs - 1), 1.0)
    return 0.6 + 0.4 * progress


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper for any base optimizer.

    SAM seeks parameters that lie in neighborhoods having uniformly low loss,
    rather than just finding the sharpest local minimum. This produces models
    that generalize better across subjects (flatter loss landscape = less
    sensitivity to subject-specific input variations).

    Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
    Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        """Ascent step: perturb weights to find the worst-case neighborhood."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the worst point
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        """Descent step: update weights using gradients at the perturbed point."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # restore original weights
        self.base_optimizer.step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # For compatibility — SAM uses first_step/second_step explicitly
        self.base_optimizer.step(closure)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


def apply_bandstop_noise(
    batch_x: torch.Tensor | dict[str, torch.Tensor],
    prob: float = 0.3,
    attenuation: tuple[float, float] = (0.3, 0.7),
    bandwidth_bins: int = 8,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """P95: Band-stop noise — randomly attenuate a narrow frequency band.

    Forces the model to not rely on a single spectral peak (e.g., just mu rhythm).
    Applied in FFT domain to preserve phase structure in other bands.
    """
    if random.random() > prob:
        return batch_x

    if isinstance(batch_x, dict):
        time_key = "time" if "time" in batch_x else "raw"
        x = batch_x[time_key]
    else:
        x = batch_x

    # FFT → attenuate random band → iFFT
    X_fft = torch.fft.rfft(x, dim=-1)
    n_freqs = X_fft.shape[-1]
    if n_freqs <= bandwidth_bins:
        return batch_x

    # Random center frequency
    center = random.randint(bandwidth_bins, n_freqs - bandwidth_bins - 1)
    start = max(0, center - bandwidth_bins // 2)
    end = min(n_freqs, center + bandwidth_bins // 2)

    # Random attenuation factor per sample
    atten = random.uniform(attenuation[0], attenuation[1])
    X_fft_modified = X_fft.clone()
    X_fft_modified[..., start:end] = X_fft_modified[..., start:end] * atten

    result = torch.fft.irfft(X_fft_modified, n=x.shape[-1], dim=-1)

    if isinstance(batch_x, dict):
        modified = dict(batch_x)
        modified[time_key] = result
        return modified
    return result


def apply_phase_perturbation(
    batch_x: torch.Tensor | dict[str, torch.Tensor],
    prob: float = 0.3,
    max_phase_shift: float = 0.4,  # ~±π/8 radians
) -> torch.Tensor | dict[str, torch.Tensor]:
    """P95: Phase perturbation — add small random phase shifts to FFT coefficients.

    Preserves spectral power (amplitude) while changing temporal phase alignment.
    This is the spectral equivalent of 'jittering' and forces the model to learn
    phase-invariant features (critical for cross-subject generalization where
    phase varies with electrode placement).
    """
    if random.random() > prob:
        return batch_x

    if isinstance(batch_x, dict):
        time_key = "time" if "time" in batch_x else "raw"
        x = batch_x[time_key]
    else:
        x = batch_x

    X_fft = torch.fft.rfft(x, dim=-1)
    # Random phase shifts (uniform in [-max_phase_shift, +max_phase_shift])
    phase_noise = torch.empty_like(X_fft.real).uniform_(-max_phase_shift, max_phase_shift)
    phase_rotation = torch.complex(torch.cos(phase_noise), torch.sin(phase_noise))
    X_fft_perturbed = X_fft * phase_rotation

    result = torch.fft.irfft(X_fft_perturbed, n=x.shape[-1], dim=-1)

    if isinstance(batch_x, dict):
        modified = dict(batch_x)
        modified[time_key] = result
        return modified
    return result


def apply_training_augmentations(
    model: EEGNet,
    batch_x: torch.Tensor | dict[str, torch.Tensor],
    cfg: TrainConfig,
    epoch: int,
) -> torch.Tensor | dict[str, torch.Tensor]:
    # FIX 2: Progressive 3-phase augmentation scheduling
    warmup = getattr(cfg, "aug_warmup_epochs", 20)
    rampup = getattr(cfg, "aug_rampup_epochs", 40)
    if epoch <= warmup:
        # Phase 1: minimal augmentation -- clean representation learning
        aug_scale = 0.1
    elif epoch <= warmup + rampup:
        # Phase 2: linear ramp 0.1 -> 1.0
        aug_scale = 0.1 + 0.9 * float(epoch - warmup) / max(float(rampup), 1.0)
    else:
        # Phase 3: full augmentation for generalization
        aug_scale = 1.0
    batch_x = apply_temporal_shift(
        batch_x,
        max_shift_ratio=cfg.temporal_shift_ratio * aug_scale,
        prob=cfg.temporal_shift_prob * aug_scale,
    )
    batch_x = apply_random_scaling(
        batch_x,
        scale_range=(
            1.0 - (1.0 - cfg.scaling_min) * aug_scale,
            1.0 + (cfg.scaling_max - 1.0) * aug_scale,
        ),
        prob=cfg.scaling_prob * aug_scale,
    )
    batch_x = apply_time_mask(
        batch_x,
        prob=cfg.time_mask_prob * aug_scale,
        max_ratio=cfg.time_mask_ratio * aug_scale,
        masks_per_sample=cfg.time_masks_per_sample,
    )
    batch_x = apply_channel_dropout(
        batch_x,
        drop_ratio=cfg.channel_dropout_ratio * aug_scale,
        prob=cfg.channel_dropout_prob * aug_scale,
    )
    batch_x = apply_gaussian_noise(
        batch_x,
        std=cfg.noise_std * aug_scale,
        prob=cfg.noise_prob * aug_scale,
    )
    # P95: Frequency-domain augmentations (applied after time-domain augs)
    freq_aug_prob = getattr(cfg, 'freq_aug_prob', 0.3) * aug_scale
    if freq_aug_prob > 0.0 and epoch > warmup:
        batch_x = apply_bandstop_noise(batch_x, prob=freq_aug_prob)
        batch_x = apply_phase_perturbation(batch_x, prob=freq_aug_prob * 0.7)
    batch_x = refresh_multiview_batch(model, batch_x)
    return batch_x


def cutmix_batch(
    batch_x: torch.Tensor | dict[str, torch.Tensor],
    batch_y: torch.Tensor,
    num_classes: int,
    alpha: float = 0.3,
    prob: float = 0.4,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
    """EEG-safe CutMix: replace temporal segments instead of blending.

    Unlike MixUp which corrupts EEG phase structure by linear interpolation,
    CutMix replaces a contiguous temporal region from one sample with another.
    Each segment retains its original phase/frequency structure intact.

    Returns: (mixed_x, y_a, y_b, lam) where lam is the mixing ratio.
    """
    if random.random() > prob:
        return batch_x, batch_y, batch_y, 1.0

    # Sample mixing ratio from Beta distribution
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 (keep majority of original)

    if isinstance(batch_x, dict):
        time_x = batch_x["time"]
    else:
        time_x = batch_x

    batch_size = time_x.shape[0]
    T = time_x.shape[-1]  # temporal dimension

    # Compute cut region
    cut_len = int(T * (1.0 - lam))
    if cut_len < 1:
        return batch_x, batch_y, batch_y, 1.0

    cut_start = random.randint(0, T - cut_len)
    cut_end = cut_start + cut_len

    # Random permutation for pairing samples
    indices = torch.randperm(batch_size, device=time_x.device)

    if isinstance(batch_x, dict):
        mixed = {}
        for key, val in batch_x.items():
            mixed_val = val.clone()
            if val.dim() >= 2:  # Apply cut to all views along temporal dim
                mixed_val[..., cut_start:cut_end] = val[indices][..., cut_start:cut_end]
            mixed[key] = mixed_val
        # Adjust lambda to actual area ratio
        lam = 1.0 - float(cut_len) / float(T)
        return mixed, batch_y, batch_y[indices], lam
    else:
        mixed_x = time_x.clone()
        mixed_x[..., cut_start:cut_end] = time_x[indices][..., cut_start:cut_end]
        lam = 1.0 - float(cut_len) / float(T)
        return mixed_x, batch_y, batch_y[indices], lam


def compute_rdrop_loss(
    logits_1: torch.Tensor,
    logits_2: torch.Tensor,
) -> torch.Tensor:
    """R-Drop: KL divergence between two forward passes with different dropout masks.

    Enforces consistency: the model should produce similar predictions regardless
    of which neurons are dropped. This extracts more regularization value from
    existing dropout without adding parameters.

    Reference: Liang et al., "R-Drop: Regularized Dropout for Neural Networks", NeurIPS 2021.
    """
    p = F.log_softmax(logits_1, dim=-1)
    q = F.log_softmax(logits_2, dim=-1)
    # Symmetric KL: 0.5 * (KL(p||q) + KL(q||p))
    kl_pq = F.kl_div(p, q.exp(), reduction="batchmean")
    kl_qp = F.kl_div(q, p.exp(), reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)

def _tta_views(
    model: EEGNet,
    batch_x: torch.Tensor | dict[str, torch.Tensor],
    noise_std: float,
    shift_ratio: float,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(batch_x, dict):
        if "time" not in batch_x and "raw" not in batch_x:
            return batch_x
        time_key = "time" if "time" in batch_x else "raw"
        time_x = apply_temporal_shift(batch_x[time_key], max_shift_ratio=shift_ratio, prob=1.0)
        time_x = apply_gaussian_noise(time_x, std=noise_std, prob=1.0)
        augmented = dict(batch_x)
        augmented[time_key] = time_x
        if "raw" in augmented:
            augmented["raw"] = augmented[time_key]
        return refresh_multiview_batch(model, augmented)
    augmented = apply_temporal_shift(batch_x, max_shift_ratio=shift_ratio, prob=1.0)
    augmented = apply_gaussian_noise(augmented, std=noise_std, prob=1.0)
    return augmented


def _compute_dann_lambda(epoch: int, warmup: int, total_epochs: int, max_lambda: float) -> float:
    """DANN lambda schedule: sigmoid ramp from 0 to max_lambda after warmup."""
    if epoch <= warmup:
        return 0.0
    p = min(1.0, (epoch - warmup) / max(total_epochs - warmup, 1))
    return max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def train_one_epoch(
    model: EEGNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler,
    device: torch.device,
    num_classes: int,
    cfg: TrainConfig,
    soft_criterion: WeightedSoftTargetCrossEntropy,
    hard_criterion: torch.nn.Module,
    epoch: int,
    ema: ModelEMA | None = None,
    swa_model: AveragedModel | None = None,
    swa_start_epoch: int | None = None,
    subject_id_map: dict[int, int] | None = None,
    supcon_criterion: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.train()
    if hasattr(model, "set_dropout_scale"):
        model.set_dropout_scale(compute_dropout_scale(epoch - 1, cfg.epochs))
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    swa_was_updated = False

    # DANN lambda schedule for this epoch
    dann_lambda_val = _compute_dann_lambda(
        epoch, getattr(cfg, 'dann_warmup_epochs', 30),
        cfg.epochs, getattr(cfg, 'dann_lambda', 0.0)
    )
    use_dann = dann_lambda_val > 0.0 and hasattr(model, 'subject_adversarial') and subject_id_map is not None
    if use_dann:
        model.subject_adversarial.set_lambda(dann_lambda_val)

    use_supcon = getattr(cfg, 'supcon_weight', 0.0) > 0.0 and hasattr(model, 'contrastive_head') and supcon_criterion is not None

    for batch_x, batch_y, batch_subj in loader:
        if batch_y is None:
            raise ValueError("Training requires labeled batches.")
        batch_x = _move_batch_to_device(batch_x, device)
        batch_y = batch_y.to(device, non_blocking=True)
        if isinstance(batch_x, dict):
            n = int(batch_x["time"].shape[0])
            if not all(int(value.shape[0]) == n for value in batch_x.values()):
                raise ValueError("Cross-view batch size mismatch.")
            if not all(torch.isfinite(value).all() for value in batch_x.values()):
                raise ValueError("Non-finite values detected in inputs.")
            if not float(batch_x["time"].detach().std().item()) > 1e-6:
                raise ValueError("Time-view std is too small.")
        else:
            if not torch.isfinite(batch_x).all():
                raise ValueError("Non-finite values detected in inputs.")
            if not float(batch_x.detach().std().item()) > 1e-6:
                raise ValueError("Input std is too small.")

        batch_x = apply_training_augmentations(model, batch_x, cfg, epoch=epoch)
        if isinstance(batch_x, dict):
            n = int(batch_x["time"].shape[0])
            if not all(int(value.shape[0]) == n for value in batch_x.values()):
                raise ValueError("Cross-view batch size mismatch after augmentation.")
            if not all(torch.isfinite(value).all() for value in batch_x.values()):
                raise ValueError("Non-finite values detected after augmentation.")
            if not float(batch_x["time"].detach().std().item()) > 1e-6:
                raise ValueError("Augmented time-view std is too small.")
        else:
            if not torch.isfinite(batch_x).all():
                raise ValueError("Non-finite values detected after augmentation.")
            if not float(batch_x.detach().std().item()) > 1e-6:
                raise ValueError("Augmented input std is too small.")

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=cfg.use_amp):
            # --- RUN6: EEG-safe CutMix ---
            use_cutmix = (
                getattr(cfg, "cutmix_alpha", 0.0) > 0.0
                and epoch > getattr(cfg, "aug_warmup_epochs", 5)
            )
            if use_cutmix:
                cm_x, y_a, y_b, lam = cutmix_batch(
                    batch_x, batch_y, num_classes=num_classes,
                    alpha=cfg.cutmix_alpha,
                    prob=getattr(cfg, "cutmix_prob", 0.4),
                )
                # Refresh multiview features after CutMix modification
                cm_x = refresh_multiview_batch(model, cm_x)
                logits = model(cm_x)
                logits = torch.clamp(logits, -20.0, 20.0)
                loss = lam * hard_criterion(logits, y_a) + (1.0 - lam) * hard_criterion(logits, y_b)
                features = None  # Skip DANN/SupCon for CutMix batches
            elif cfg.mixup_alpha > 0.0:
                mixed_x, soft_targets = mixup_batch(
                    batch_x,
                    batch_y,
                    num_classes=num_classes,
                    alpha=cfg.mixup_alpha,
                    smoothing=cfg.label_smoothing,
                )
                logits = model(mixed_x)
                logits = torch.clamp(logits, -20.0, 20.0)
                loss = soft_criterion(logits, soft_targets)
                features = None
            else:
                result = model(batch_x, return_features=(use_dann or use_supcon))
                if isinstance(result, tuple):
                    logits, features = result
                else:
                    logits, features = result, None
                logits = torch.clamp(logits, -20.0, 20.0)
                loss = hard_criterion(logits, batch_y)

            # --- RUN6: R-Drop consistency regularization ---
            rdrop_alpha = getattr(cfg, "rdrop_alpha", 0.0)
            if rdrop_alpha > 0.0 and epoch > getattr(cfg, "aug_warmup_epochs", 5):
                # Second forward pass with different dropout mask
                if use_cutmix:
                    logits_2 = model(cm_x)
                else:
                    logits_2 = model(batch_x)
                logits_2 = torch.clamp(logits_2, -20.0, 20.0)
                rdrop_loss = compute_rdrop_loss(logits, logits_2)
                loss = loss + rdrop_alpha * rdrop_loss

            if hasattr(model, "get_aux_losses"):
                aux_losses = model.get_aux_losses()
                warmup_done = epoch > getattr(cfg, "aug_warmup_epochs", 20)
                if warmup_done and aux_losses:
                    fusion_entropy = aux_losses.get("fusion_entropy", logits.new_tensor(0.0))
                    band_attn_entropy = aux_losses.get("band_attn_entropy", logits.new_tensor(0.0))
                    loss = (
                        loss
                        - 0.005 * fusion_entropy     # Maximize to prevent branch collapse
                        - 0.002 * band_attn_entropy  # Maximize to prevent attention collapse
                    )

            # === NOVEL: Domain Adversarial Training (DANN) ===
            if use_dann and features is not None and batch_subj is not None:
                mapped_subj = torch.tensor(
                    [subject_id_map.get(int(s), 0) for s in batch_subj.cpu().numpy()],
                    dtype=torch.long, device=device
                )
                subj_logits = model.subject_adversarial(features)
                dann_loss = F.cross_entropy(subj_logits, mapped_subj)
                loss = loss + dann_loss  # GRL already reverses backbone gradient

            # === NOVEL: Supervised Contrastive Loss ===
            if use_supcon and features is not None and supcon_criterion is not None:
                proj = model.contrastive_head(features)
                supcon_loss = supcon_criterion(proj, batch_y)
                loss = loss + cfg.supcon_weight * supcon_loss

        # P95v2: SAM-compatible gradient computation with delayed activation
        is_sam = isinstance(optimizer, SAM)
        sam_warmup = getattr(cfg, 'sam_warmup_epochs', 30)
        sam_active = is_sam and epoch > sam_warmup  # SAM only after warmup
        if sam_active and cfg.use_amp:
            # SAM with AMP: two-step gradient computation
            # GradScaler.unscale_() can only be called ONCE per optimizer between update() calls.
            # Solution: manually unscale for step 1, use official unscale_() only for step 2.

            # Step 1: compute gradient at current weights → perturb to worst-case
            scaler.scale(loss).backward()
            # Manual unscale: divide all grads by the current scale factor
            inv_scale = 1.0 / scaler.get_scale()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(inv_scale)
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.first_step()  # perturb weights to worst-case neighborhood
            optimizer.zero_grad(set_to_none=True)

            # Step 2: compute gradient at perturbed weights → apply actual update
            with autocast_context(device, enabled=cfg.use_amp):
                if use_cutmix:
                    logits_2nd = model(cm_x)
                elif cfg.mixup_alpha > 0.0:
                    logits_2nd = model(mixed_x)
                else:
                    result_2nd = model(batch_x, return_features=False)
                    logits_2nd = result_2nd if not isinstance(result_2nd, tuple) else result_2nd[0]
                logits_2nd = torch.clamp(logits_2nd, -20.0, 20.0)
                if use_cutmix:
                    loss_2nd = lam * hard_criterion(logits_2nd, y_a) + (1.0 - lam) * hard_criterion(logits_2nd, y_b)
                elif cfg.mixup_alpha > 0.0:
                    loss_2nd = soft_criterion(logits_2nd, soft_targets)
                else:
                    loss_2nd = hard_criterion(logits_2nd, batch_y)
            scaler.scale(loss_2nd).backward()
            scaler.unscale_(optimizer.base_optimizer)  # official unscale — only call between update()s
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.second_step()  # restore original weights + apply perturbed gradient update
            scaler.update()
            optimizer_step_was_skipped = False
        elif sam_active:
            # SAM without AMP (rare path)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.first_step()
            optimizer.zero_grad(set_to_none=True)
            if use_cutmix:
                logits_2nd = model(cm_x)
            elif cfg.mixup_alpha > 0.0:
                logits_2nd = model(mixed_x)
            else:
                result_2nd = model(batch_x, return_features=False)
                logits_2nd = result_2nd if not isinstance(result_2nd, tuple) else result_2nd[0]
            logits_2nd = torch.clamp(logits_2nd, -20.0, 20.0)
            if use_cutmix:
                loss_2nd = lam * hard_criterion(logits_2nd, y_a) + (1.0 - lam) * hard_criterion(logits_2nd, y_b)
            elif cfg.mixup_alpha > 0.0:
                loss_2nd = soft_criterion(logits_2nd, soft_targets)
            else:
                loss_2nd = hard_criterion(logits_2nd, batch_y)
            loss_2nd.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.second_step()
            optimizer_step_was_skipped = False
        else:
            # Standard gradient step (AdamW or SAM-during-warmup)
            # When SAM is in warmup, step through its base_optimizer
            active_optim = optimizer.base_optimizer if is_sam else optimizer
            scaler.scale(loss).backward()
            scaler.unscale_(active_optim)
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scale_before_step = float(scaler.get_scale())
            scaler.step(active_optim)
            scaler.update()
            optimizer_step_was_skipped = float(scaler.get_scale()) < scale_before_step

        if ema is not None:
            ema.update(model)
        if swa_model is not None and swa_start_epoch is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_was_updated = True

        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) and not optimizer_step_was_skipped:
            scheduler.step()

        total_loss += loss.item() * batch_y.size(0)
        preds = logits.detach().argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    cm = confusion_matrix(preds, targets, num_classes)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "acc": accuracy_from_cm(cm),
        "f1": macro_f1_from_cm(cm),
        "swa_updated": float(swa_was_updated),
    }


@torch.no_grad()
def predict_logits_with_tta(
    model: EEGNet,
    batch_x: torch.Tensor | dict[str, torch.Tensor],
    device: torch.device,
    tta_passes: int,
    noise_std: float,
    shift_ratio: float,
    temperature: float | None = None,
    *,
    use_amp: bool = True,
    debug_validation: bool = False,
    fail_on_nonfinite: bool = False,
    mc_dropout_passes: int = 0,
) -> torch.Tensor:
    """P95: Enhanced TTA with MC-Dropout ensemble.

    The first pass uses the clean input in eval mode.
    Subsequent TTA passes use augmented views in eval mode.
    MC-Dropout passes re-enable dropout (train mode) for
    Bayesian averaging across different dropout masks.
    """
    logits = []
    passes = max(int(tta_passes), 1)

    # Standard TTA passes (eval mode — deterministic dropout)
    model.eval()
    for pass_index in range(passes):
        if pass_index == 0:
            current_x = batch_x
        elif passes > 1:
            current_x = _tta_views(model, batch_x, noise_std=noise_std, shift_ratio=shift_ratio)
        if debug_validation:
            _debug_check_batch_inputs(
                current_x,
                prefix=f"val_batch_tta_pass_{pass_index + 1}",
                fail_on_nonfinite=fail_on_nonfinite,
            )
        with autocast_context(device, enabled=use_amp):
            current_logits = apply_model_temperature(model, model(current_x), temperature=temperature)
        if debug_validation and not torch.isfinite(current_logits).all():
            print(f"[VAL-DEBUG] non-finite logits at TTA pass {pass_index + 1}/{passes}")
            print(f"[VAL-DEBUG] raw_logits {_tensor_debug_stats(current_logits)}")
            if fail_on_nonfinite:
                raise FloatingPointError(f"Non-finite logits at TTA pass {pass_index + 1}/{passes}")
        current_logits = torch.clamp(current_logits, -20.0, 20.0)
        logits.append(current_logits)

    # P95: MC-Dropout passes (train mode — stochastic dropout masks)
    mc_passes = max(int(mc_dropout_passes), 0)
    if mc_passes > 0:
        model.train()  # Re-enable dropout for Bayesian uncertainty estimation
        for mc_idx in range(mc_passes):
            # Use clean input with different dropout masks
            with autocast_context(device, enabled=use_amp):
                mc_logits = apply_model_temperature(model, model(batch_x), temperature=temperature)
            mc_logits = torch.clamp(mc_logits, -20.0, 20.0)
            logits.append(mc_logits)
        model.eval()  # Restore eval mode

    return torch.stack(logits, dim=0).mean(dim=0)


@torch.no_grad()
def validate_one_epoch(
    model: EEGNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: torch.nn.Module,
    tta_passes: int = 1,
    tta_noise_std: float = 0.0,
    tta_shift_ratio: float = 0.0,
    temperature: float | None = None,
    *,
    use_amp: bool = True,
    debug_validation: bool = False,
    fail_on_nonfinite: bool = False,
    max_batches: int = 0,
    mc_dropout_passes: int = 0,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch_index, (batch_x, batch_y, _) in enumerate(loader, start=1):
        if batch_y is None:
            raise ValueError("Validation requires labeled batches.")
        batch_x = _move_batch_to_device(batch_x, device)
        batch_y = batch_y.to(device, non_blocking=True)
        if isinstance(batch_x, dict):
            n = int(batch_x["time"].shape[0])
            if not all(int(value.shape[0]) == n for value in batch_x.values()):
                raise ValueError("Cross-view batch size mismatch.")
            if not all(torch.isfinite(value).all() for value in batch_x.values()):
                raise ValueError("Non-finite values detected in validation inputs.")
            if not float(batch_x["time"].detach().std().item()) > 1e-6:
                raise ValueError("Validation time-view std is too small.")
        else:
            if not torch.isfinite(batch_x).all():
                raise ValueError("Non-finite values detected in validation inputs.")
            if not float(batch_x.detach().std().item()) > 1e-6:
                raise ValueError("Validation input std is too small.")

        logits = predict_logits_with_tta(
            model,
            batch_x,
            device=device,
            tta_passes=tta_passes,
            noise_std=tta_noise_std,
            shift_ratio=tta_shift_ratio,
            temperature=temperature,
            use_amp=use_amp,
            debug_validation=debug_validation,
            fail_on_nonfinite=fail_on_nonfinite,
            mc_dropout_passes=mc_dropout_passes,
        )
        if debug_validation and not torch.isfinite(logits).all():
            print(f"[VAL-DEBUG] batch {batch_index}: non-finite logits after TTA aggregation")
            print(f"[VAL-DEBUG] agg_logits {_tensor_debug_stats(logits)}")
            if fail_on_nonfinite:
                raise FloatingPointError(f"Validation batch {batch_index}: non-finite logits after TTA aggregation")
        with autocast_context(device, enabled=use_amp):
            probs = torch.softmax(logits, dim=-1)
            loss = criterion(logits, batch_y)
        if debug_validation and not torch.isfinite(probs).all():
            print(f"[VAL-DEBUG] batch {batch_index}: non-finite softmax output")
            print(f"[VAL-DEBUG] probs {_tensor_debug_stats(probs)}")
            if fail_on_nonfinite:
                raise FloatingPointError(f"Validation batch {batch_index}: non-finite softmax output")
        if debug_validation and not torch.isfinite(loss):
            print(f"[VAL-DEBUG] batch {batch_index}: non-finite validation loss")
            print(f"[VAL-DEBUG] loss={float(loss.detach().cpu())}")
            if fail_on_nonfinite:
                raise FloatingPointError(f"Validation batch {batch_index}: non-finite loss")

        total_loss += loss.item() * batch_y.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        if max_batches > 0 and batch_index >= max_batches:
            if debug_validation:
                print(f"[VAL-DEBUG] stopping early after {batch_index} validation batch(es)")
            break

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)
    cm = confusion_matrix(preds, targets, num_classes)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "acc": accuracy_from_cm(cm),
        "f1": macro_f1_from_cm(cm),
        "cm": cm,
        "preds": preds,
        "targets": targets,
        "probs": probs,
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
    cfg: TrainConfig,
):
    if scheduler_name == "cosine_single_restart":
        # P95: Single restart cosine — two-phase training with one strategic restart
        # Phase 1: warmup(10ep) → cosine decay to eta_min over first half
        # Phase 2: jump back to peak LR → cosine decay to eta_min over second half
        # This gives deep convergence in each phase unlike frequent warm restarts.
        warmup_epochs = 10
        half_epochs = max((epochs - warmup_epochs) // 2, 1)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs
        )
        phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=half_epochs, eta_min=1e-6
        )
        phase2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs - half_epochs, 1), eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, phase1, phase2],
            milestones=[warmup_epochs, warmup_epochs + half_epochs],
        )
    if scheduler_name == "cosine_restarts":
        # Cosine warm restarts -- escape local minima
        warmup_epochs = 5
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        restarts = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=2, eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, restarts], milestones=[warmup_epochs]
        )
    if scheduler_name == "cosine":
        warmup_epochs = 5
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(epochs - warmup_epochs), 1), eta_min=1e-6
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    # Default to OneCycleLR which has built-in warmup
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, epochs=epochs, steps_per_epoch=steps_per_epoch)


def save_checkpoint(
    path: Path,
    model: EEGNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    cfg: TrainConfig,
    num_classes: int,
    sampling_frequency: float,
    temperature: float = 1.0,
) -> None:
    # Sanitize state dict keys (strip 'module.' prefix from SWA/DDP models)
    state_dict = model.state_dict()
    sanitized = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
    torch.save(
        {
            "model_state_dict": sanitized,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg.__dict__,
            "temperature": temperature,
            "meta": {
                "num_classes": num_classes,
                "hidden_channels": cfg.hidden_channels,
                "band_channels": int(getattr(model, "band_channels", getattr(model, "in_channels", 0))),
                "sampling_frequency": sampling_frequency,
                "use_multiview": cfg.use_multiview,
                "ablation": cfg.ablation,
                "band_method": str(getattr(model, "band_method", "fft_segments")),
                "band_segments": int(getattr(model, "band_segments", 4)),
                "band_definitions": {
                    str(name): [float(low), float(high)]
                    for name, (low, high) in getattr(model, "band_definitions", {}).items()
                },
                "temperature": temperature,
            },
            "num_classes": num_classes,
            "hidden_channels": cfg.hidden_channels,
            "band_channels": int(getattr(model, "band_channels", getattr(model, "in_channels", 0))),
            "sampling_frequency": sampling_frequency,
        },
        path,
    )


def parse_args() -> TrainConfig:
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.use_ema = parse_bool(args.use_ema)
    args.use_swa = parse_bool(args.use_swa)
    args.disable_tta = parse_bool(args.disable_tta)
    args.use_multiview = parse_bool(args.use_multiview)
    args.use_amp = parse_bool(args.use_amp)
    args.debug_validation = parse_bool(args.debug_validation)
    args.fail_on_nonfinite = parse_bool(args.fail_on_nonfinite)
    args.use_adabn = parse_bool(args.use_adabn)
    args.balanced_sampling = parse_bool(args.balanced_sampling)
    args.use_sam = parse_bool(args.use_sam)  # P95: SAM optimizer toggle
    return TrainConfig(**vars(args))


def _label_distribution(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(np.asarray(labels).reshape(-1).tolist())
    return {str(label): int(count) for label, count in sorted(counts.items(), key=lambda item: item[0])}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _run_training_session(
    *,
    cfg: TrainConfig,
    output_dir: Path,
    train_views: dict[str, np.ndarray],
    y_train: np.ndarray,
    train_subjects: np.ndarray,
    val_views: dict[str, np.ndarray],
    y_val: np.ndarray,
    val_subjects: np.ndarray,
    calib_views: dict[str, np.ndarray],
    y_calib: np.ndarray,
    calib_subjects: np.ndarray,
    test_views: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_subjects: np.ndarray,
    metadata: dict[str, Any],
    fold_name: str,
    fold_subject_splits: dict[str, list[int]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    cm_path = output_dir / "confusion_matrix.png"

    if cfg.use_multiview and cfg.ablation != "full":
        raise ValueError("Strict multiview training requires --ablation full. Use --use-multiview false for time-only runs.")

    train_views = select_feature_views(train_views, use_multiview=cfg.use_multiview, ablation=cfg.ablation, strict=cfg.use_multiview)
    val_views = select_feature_views(val_views, use_multiview=cfg.use_multiview, ablation=cfg.ablation, strict=cfg.use_multiview)
    calib_views = select_feature_views(calib_views, use_multiview=cfg.use_multiview, ablation=cfg.ablation, strict=cfg.use_multiview)
    test_views = select_feature_views(test_views, use_multiview=cfg.use_multiview, ablation=cfg.ablation, strict=cfg.use_multiview)

    num_classes = int(np.unique(np.concatenate([y_train, y_val, y_calib, y_test])).size)
    class_names = [str(index) for index in range(num_classes)]
    feature_cfg = _feature_config_from_metadata(metadata)
    sampling_frequency = float(feature_cfg["sampling_frequency"])

    train_loader, val_loader = _create_train_val_loaders(
        train_views,
        y_train,
        train_subjects,
        val_views,
        y_val,
        val_subjects,
        cfg,
        num_classes,
    )
    calib_loader = _make_loader(calib_views, y_calib, calib_subjects, batch_size=cfg.batch_size, shuffle=False)
    test_loader = _make_loader(test_views, y_test, test_subjects, batch_size=cfg.batch_size, shuffle=False)

    first_batch, _, _ = next(iter(train_loader))
    in_channels, _ = _batch_time_shape(first_batch)
    band_channels = (
        int(first_batch["bands"].shape[1])
        if isinstance(first_batch, dict) and "bands" in first_batch
        else in_channels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=cfg.hidden_channels,
        band_channels=band_channels,
        dropout=cfg.dropout,
        sampling_frequency=sampling_frequency,
        include_gamma=bool(feature_cfg["include_gamma"]),
        band_method=str(feature_cfg["band_method"]),
        band_segments=int(feature_cfg["band_segments"]),
    ).to(device)
    if isinstance(first_batch, dict) and "bands" in first_batch:
        model._n_bands_fallback = int(first_batch["bands"].shape[2])
    effective_tta_passes = 1 if cfg.disable_tta else max(int(cfg.tta_passes), 1)
    if effective_tta_passes > 1 and not getattr(model, "can_refresh_multiview", False):
        raise ValueError("TTA with more than one pass requires online multiview refresh support.")

    # === NOVEL: Initialize DANN + Contrastive auxiliary heads ===
    subject_id_map: dict[int, int] | None = None
    supcon_criterion_obj = None
    unique_train_subs = np.unique(train_subjects)
    if getattr(cfg, 'dann_lambda', 0.0) > 0.0 or getattr(cfg, 'supcon_weight', 0.0) > 0.0:
        model.init_auxiliary_heads(num_subjects=len(unique_train_subs))
        subject_id_map = {int(sid): idx for idx, sid in enumerate(unique_train_subs)}
        from phase2 import SupConLoss
        supcon_criterion_obj = SupConLoss(temperature=0.07).to(device)
        print(f"[NOVEL] DANN enabled: lambda={cfg.dann_lambda}, warmup={cfg.dann_warmup_epochs}, subjects={len(unique_train_subs)}")
        print(f"[NOVEL] SupCon enabled: weight={cfg.supcon_weight}")

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    swa_model = AveragedModel(model).to(device) if cfg.use_swa else None
    swa_start_epoch = max(1, int(round(cfg.epochs * cfg.swa_start))) if cfg.use_swa else None

    if cfg.balanced_sampling:
        class_weights = torch.ones(num_classes, device=device)
        soft_criterion = WeightedSoftTargetCrossEntropy(class_weights)
        focal_soft_criterion = FocalSoftTargetCrossEntropy(class_weights, gamma=cfg.focal_gamma)
        hard_criterion = FocalCrossEntropy(class_weights=class_weights, gamma=cfg.focal_gamma, label_smoothing=0.0)
        val_criterion = torch.nn.CrossEntropyLoss()
        print(f"[LOSS] FocalLoss(gamma={cfg.focal_gamma}) -- NO class weights (sampler handles balance)")
    else:
        computed_weights = compute_class_weights(y_train, num_classes).to(device)
        class_weights = computed_weights
        if cfg.loss == "focal":
            print(f"[LOSS] FocalLoss(gamma={cfg.focal_gamma}) with class_weights={class_weights.cpu().numpy()}")
        else:
            print(f"[LOSS] {cfg.loss} with class_weights={class_weights.cpu().numpy()}")
            
        soft_criterion = WeightedSoftTargetCrossEntropy(class_weights)
        focal_soft_criterion = FocalSoftTargetCrossEntropy(class_weights, gamma=cfg.focal_gamma)
        hard_criterion = _build_hard_criterion(cfg.loss, class_weights, cfg.focal_gamma, cfg.label_smoothing)
        val_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # P90: Differential weight decay — backbone gets cfg.weight_decay (2e-3),
    # classifier head gets 5x more (1e-2) to prevent head memorization.
    head_param_names = {'classifier', 'pre_classifier', 'classifier_dropout'}
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if parameter belongs to the classifier head
        is_head = any(hp in name for hp in head_param_names)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)
    use_sam = parse_bool(getattr(cfg, 'use_sam', 'true'))
    if use_sam:
        # P95: SAM wraps AdamW for flatter minima → better cross-subject generalization
        optimizer = SAM(
            [
                {'params': backbone_params, 'weight_decay': cfg.weight_decay, 'rho': cfg.sam_rho},
                {'params': head_params, 'weight_decay': cfg.weight_decay * 5.0, 'rho': cfg.sam_rho},
            ],
            base_optimizer_cls=torch.optim.AdamW,
            rho=cfg.sam_rho,
            lr=cfg.lr,
            betas=(0.9, 0.999),
        )
        print(f"[P95] SAM optimizer: rho={cfg.sam_rho}, base=AdamW")
        print(f"[P95] Differential WD: backbone={cfg.weight_decay}, head={cfg.weight_decay * 5.0}, "
              f"backbone_params={len(backbone_params)}, head_params={len(head_params)}")
    else:
        # Head LR multiplier — controllable from CLI via --head-lr-mult
        head_lr_mult = getattr(cfg, 'head_lr_mult', 1.0)
        optimizer = torch.optim.AdamW(
            [
                {'params': backbone_params, 'weight_decay': cfg.weight_decay, 'lr': cfg.lr},
                {'params': head_params, 'weight_decay': cfg.weight_decay * 5.0, 'lr': cfg.lr * head_lr_mult},
            ],
            lr=cfg.lr,
            betas=(0.9, 0.999),
        )
        print(f"[P95v6] AdamW optimizer (SAM disabled)")
        if head_lr_mult != 1.0:
            print(f"[P95v6] Differential LR: backbone={cfg.lr}, head={cfg.lr * head_lr_mult:.6f} ({head_lr_mult}x)")
        print(f"[P95v6] Differential WD: backbone={cfg.weight_decay}, head={cfg.weight_decay * 5.0}, "
              f"backbone_params={len(backbone_params)}, head_params={len(head_params)}")
    scheduler = _build_scheduler(
        optimizer.base_optimizer if use_sam else optimizer,
        scheduler_name=cfg.scheduler,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        cfg=cfg,
    )
    scaler = create_grad_scaler(device, enabled=cfg.use_amp)
    early_stopping = EarlyStopping(cfg.patience, cfg.min_delta)

    best_auc = -float("inf")
    best_f1 = -float("inf")
    best_acc = -float("inf")
    focal_start = getattr(cfg, "focal_start_epoch", 1)

    print(f"\n[{fold_name}] Device: {device}")
    print(f"[{fold_name}] Train shape: {tuple(int(dim) for dim in train_views['time'].shape)}")
    print(f"[{fold_name}] Val shape:   {tuple(int(dim) for dim in val_views['time'].shape)}")
    print(f"[{fold_name}] Calib shape: {tuple(int(dim) for dim in calib_views['time'].shape)}")
    print(f"[{fold_name}] Test shape:  {tuple(int(dim) for dim in test_views['time'].shape)}")
    print(f"[{fold_name}] Class weights: {class_weights.detach().cpu().numpy()}")
    print(f"[{fold_name}] Sampling frequency: {sampling_frequency:.4f}")
    print(f"[{fold_name}] TTA passes: {effective_tta_passes}")

    for epoch in range(1, cfg.epochs + 1):
        # P95v2: Log SAM activation point
        sam_warmup = getattr(cfg, 'sam_warmup_epochs', 30)
        if isinstance(optimizer, SAM) and epoch == sam_warmup + 1:
            print(f"[{fold_name}] [P95v2] SAM activated at epoch {epoch} (warmup={sam_warmup})")
        current_soft = focal_soft_criterion if epoch >= focal_start else soft_criterion
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            num_classes=num_classes,
            cfg=cfg,
            soft_criterion=current_soft,
            hard_criterion=hard_criterion,
            epoch=epoch,
            ema=ema,
            swa_model=swa_model,
            swa_start_epoch=swa_start_epoch,
            subject_id_map=subject_id_map,
            supcon_criterion=supcon_criterion_obj,
        )

        val_model, val_model_name = _select_validation_model(
            cfg=cfg,
            model=model,
            ema=ema,
            swa_model=swa_model,
            swa_start_epoch=swa_start_epoch,
            epoch=epoch,
        )
        val_metrics = validate_one_epoch(
            model=val_model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            criterion=val_criterion,
            tta_passes=effective_tta_passes,
            tta_noise_std=cfg.tta_noise_std,
            tta_shift_ratio=cfg.tta_shift_ratio,
            temperature=1.0,
            use_amp=cfg.use_amp,
            debug_validation=cfg.debug_validation,
            fail_on_nonfinite=cfg.fail_on_nonfinite,
            max_batches=cfg.max_val_batches,
        )
        val_auc = compute_auc(val_metrics["probs"], val_metrics["targets"], num_classes)
        val_metrics["auc"] = val_auc

        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        improved = val_auc > best_auc + cfg.min_delta
        if improved:
            best_auc = val_auc
            best_f1 = float(val_metrics["f1"])
            best_acc = float(val_metrics["acc"])
            save_checkpoint(
                checkpoint_path,
                val_model,
                optimizer,
                scheduler,
                epoch,
                {"val_loss": float(val_metrics["loss"]), "val_acc": best_acc, "val_f1": best_f1, "val_auc": best_auc},
                cfg,
                num_classes,
                sampling_frequency,
                temperature=1.0,
            )

        print(
            f"[{fold_name}] Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_auc:.4f} | best_val_auc={best_auc:.4f} | model={val_model_name}"
        )

        if early_stopping.step(val_auc):
            print(f"[{fold_name}] Early stopping triggered at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if swa_model is not None and swa_start_epoch is not None:
        update_batchnorm_stats(train_loader, swa_model, device=device)
    if cfg.use_adabn:
        update_batchnorm_stats(calib_loader, model, device=device)

    learned_temp = _learn_temperature(model, calib_loader, device, use_amp=cfg.use_amp)
    model.inference_temperature = learned_temp

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint["temperature"] = learned_temp
    checkpoint.setdefault("meta", {})["temperature"] = learned_temp
    torch.save(checkpoint, checkpoint_path)

    pre_metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes,
        criterion=val_criterion,
        tta_passes=effective_tta_passes,
        tta_noise_std=cfg.tta_noise_std,
        tta_shift_ratio=cfg.tta_shift_ratio,
        temperature=1.0,
        use_amp=cfg.use_amp,
        debug_validation=cfg.debug_validation,
        fail_on_nonfinite=cfg.fail_on_nonfinite,
        max_batches=cfg.max_val_batches,
        mc_dropout_passes=getattr(cfg, 'mc_dropout_passes', 0),
    )
    post_metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes,
        criterion=val_criterion,
        tta_passes=effective_tta_passes,
        tta_noise_std=cfg.tta_noise_std,
        tta_shift_ratio=cfg.tta_shift_ratio,
        temperature=learned_temp,
        use_amp=cfg.use_amp,
        debug_validation=cfg.debug_validation,
        fail_on_nonfinite=cfg.fail_on_nonfinite,
        max_batches=cfg.max_val_batches,
        mc_dropout_passes=getattr(cfg, 'mc_dropout_passes', 0),
    )
    final_auc = compute_auc(post_metrics["probs"], post_metrics["targets"], num_classes)
    pre_ece = compute_ece(pre_metrics["probs"], pre_metrics["targets"], n_bins=15)
    post_ece = compute_ece(post_metrics["probs"], post_metrics["targets"], n_bins=15)
    plot_confusion_matrix(post_metrics["cm"], class_names, cm_path)

    subject_acc_mean, subject_acc_std, subject_acc_min, subject_acc_max = _per_subject_accuracy(
        post_metrics["preds"],
        post_metrics["targets"],
        test_subjects,
    )
    subject_auc_mean, subject_auc_std = compute_subject_auc(
        post_metrics["probs"],
        post_metrics["targets"],
        test_subjects,
        num_classes,
    )

    report = {
        "fold_name": fold_name,
        "split_mode": cfg.split_mode,
        "best_val_auc": float(best_auc),
        "best_val_acc": float(best_acc),
        "best_val_f1": float(best_f1),
        "test_accuracy": float(post_metrics["acc"]),
        "test_macro_f1": float(post_metrics["f1"]),
        "test_auc": float(final_auc),
        "test_ece": float(post_ece),
        "pre_temperature_ece": float(pre_ece),
        "temperature": float(learned_temp),
        "subject_acc_mean": float(subject_acc_mean),
        "subject_acc_std": float(subject_acc_std),
        "subject_acc_min": float(subject_acc_min),
        "subject_acc_max": float(subject_acc_max),
        "subject_auc_mean": float(subject_auc_mean),
        "subject_auc_std": float(subject_auc_std),
        "sampling_frequency": float(sampling_frequency),
        "train_samples": int(y_train.shape[0]),
        "val_samples": int(y_val.shape[0]),
        "calib_samples": int(y_calib.shape[0]),
        "test_samples": int(y_test.shape[0]),
        "train_subjects": fold_subject_splits["train"],
        "val_subjects": fold_subject_splits["val"],
        "calib_subjects": fold_subject_splits["calib"],
        "test_subjects": fold_subject_splits["test"],
        "train_label_distribution": _label_distribution(y_train),
        "val_label_distribution": _label_distribution(y_val),
        "calib_label_distribution": _label_distribution(y_calib),
        "test_label_distribution": _label_distribution(y_test),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "confusion_matrix_path": str(cm_path.resolve()),
    }
    _save_json(output_dir / "fold_metrics.json", report)
    return report


def _run_subject_kfold_training(cfg: TrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_views, y_all, all_subjects, metadata = load_canonical_data(cfg.data_dir)
    if y_all is None or all_subjects is None:
        raise ValueError("Subject-kfold training requires all_y.npy and all_subjects.npy.")

    all_views = {key: np.asarray(value, dtype=np.float32) for key, value in all_views.items()}
    y_all = np.asarray(y_all, dtype=np.int64)
    all_subjects = np.asarray(all_subjects, dtype=np.int32)
    fold_subject_groups = _subject_kfold_subjects(all_subjects, cfg.num_folds, cfg.seed)
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, test_subject_ids in enumerate(fold_subject_groups):
        if cfg.fold_index >= 0 and fold_idx != cfg.fold_index:
            continue

        remaining_subject_ids = np.asarray(
            [subject_id for subject_id in np.unique(all_subjects) if subject_id not in set(test_subject_ids.tolist())],
            dtype=np.int32,
        )
        train_subject_ids, val_subject_ids, calib_subject_ids = _split_inner_subjects(
            remaining_subject_ids,
            inner_val_ratio=cfg.inner_val_ratio,
            calibration_ratio=cfg.calibration_ratio,
            seed=cfg.seed + fold_idx + 1,
        )

        train_idx = _indices_for_subjects(all_subjects, train_subject_ids)
        val_idx = _indices_for_subjects(all_subjects, val_subject_ids)
        calib_idx = _indices_for_subjects(all_subjects, calib_subject_ids)
        test_idx = _indices_for_subjects(all_subjects, test_subject_ids)

        fold_dir = output_dir / f"fold_{fold_idx + 1:02d}"
        report = _run_training_session(
            cfg=cfg,
            output_dir=fold_dir,
            train_views=_subset_feature_views(all_views, train_idx),
            y_train=_subset_labels(y_all, train_idx),
            train_subjects=_subset_subjects(all_subjects, train_idx),
            val_views=_subset_feature_views(all_views, val_idx),
            y_val=_subset_labels(y_all, val_idx),
            val_subjects=_subset_subjects(all_subjects, val_idx),
            calib_views=_subset_feature_views(all_views, calib_idx),
            y_calib=_subset_labels(y_all, calib_idx),
            calib_subjects=_subset_subjects(all_subjects, calib_idx),
            test_views=_subset_feature_views(all_views, test_idx),
            y_test=_subset_labels(y_all, test_idx),
            test_subjects=_subset_subjects(all_subjects, test_idx),
            metadata=dict(metadata),
            fold_name=f"fold_{fold_idx + 1:02d}",
            fold_subject_splits={
                "train": [int(value) for value in train_subject_ids.tolist()],
                "val": [int(value) for value in val_subject_ids.tolist()],
                "calib": [int(value) for value in calib_subject_ids.tolist()],
                "test": [int(value) for value in np.asarray(test_subject_ids).tolist()],
            },
        )
        fold_reports.append(report)

    if not fold_reports:
        raise ValueError("No folds were executed. Check --fold-index and --num-folds.")

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
    aggregate_summary: dict[str, Any] = {
        "split_mode": cfg.split_mode,
        "num_folds_executed": len(fold_reports),
        "folds": fold_reports,
        "metrics": {},
    }
    for metric_name in metrics_to_aggregate:
        values = np.asarray([float(report[metric_name]) for report in fold_reports], dtype=np.float64)
        aggregate_summary["metrics"][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": [float(value) for value in values.tolist()],
        }

    _save_json(output_dir / "aggregate_summary.json", aggregate_summary)
    _write_metrics_csv(output_dir / "fold_metrics.csv", fold_reports)
    summary_row = {
        "num_folds_executed": int(len(fold_reports)),
    }
    for metric_name in metrics_to_aggregate:
        summary_row[f"{metric_name}_mean"] = aggregate_summary["metrics"][metric_name]["mean"]
        summary_row[f"{metric_name}_std"] = aggregate_summary["metrics"][metric_name]["std"]
    _write_metrics_csv(output_dir / "aggregate_summary.csv", [summary_row])

    print("\nCross-validation summary")
    for metric_name in metrics_to_aggregate:
        metric = aggregate_summary["metrics"][metric_name]
        print(f"{metric_name}: {metric['mean']:.4f} Â± {metric['std']:.4f}")


def main() -> None:
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_seed(cfg.seed)
    if cfg.split_mode == "subject_kfold":
        _run_subject_kfold_training(cfg)
        return

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_views, y_train, train_subjects, val_views, y_val, val_subjects, metadata = load_prepared_data(cfg.data_dir)
    if y_train is None or y_val is None:
        raise ValueError("Holdout training requires train_y.npy and val_y.npy.")
    if train_subjects is None:
        train_subjects = np.zeros(y_train.shape[0], dtype=np.int32)
    if val_subjects is None:
        val_subjects = np.zeros(y_val.shape[0], dtype=np.int32)

    report = _run_training_session(
        cfg=cfg,
        output_dir=output_dir,
        train_views={key: np.asarray(value, dtype=np.float32) for key, value in train_views.items()},
        y_train=np.asarray(y_train, dtype=np.int64),
        train_subjects=np.asarray(train_subjects, dtype=np.int32),
        val_views={key: np.asarray(value, dtype=np.float32) for key, value in val_views.items()},
        y_val=np.asarray(y_val, dtype=np.int64),
        val_subjects=np.asarray(val_subjects, dtype=np.int32),
        calib_views={key: np.asarray(value, dtype=np.float32) for key, value in val_views.items()},
        y_calib=np.asarray(y_val, dtype=np.int64),
        calib_subjects=np.asarray(val_subjects, dtype=np.int32),
        test_views={key: np.asarray(value, dtype=np.float32) for key, value in val_views.items()},
        y_test=np.asarray(y_val, dtype=np.int64),
        test_subjects=np.asarray(val_subjects, dtype=np.int32),
        metadata=dict(metadata),
        fold_name="holdout",
        fold_subject_splits={
            "train": [int(value) for value in np.unique(train_subjects).tolist()],
            "val": [int(value) for value in np.unique(val_subjects).tolist()],
            "calib": [int(value) for value in np.unique(val_subjects).tolist()],
            "test": [int(value) for value in np.unique(val_subjects).tolist()],
        },
    )

    print("\nHoldout summary")
    print(f"Accuracy: {report['test_accuracy']:.4f}")
    print(f"Macro F1: {report['test_macro_f1']:.4f}")
    print(f"AUC: {report['test_auc']:.4f}")
    print(f"ECE: {report['test_ece']:.4f}")
    print(f"Temperature: {report['temperature']:.4f}")
    print(f"Checkpoint: {report['checkpoint_path']}")


if __name__ == "__main__":
    main()