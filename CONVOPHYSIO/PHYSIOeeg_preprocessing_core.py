from __future__ import annotations
import argparse
import gc
import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from importlib import import_module
from numpy.lib.format import open_memmap
from pathlib import Path
from pprint import pprint
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from multiview_features import (
    compute_frequency_features_numpy,
    compute_segmented_bandpower_numpy,
    prepare_band_definitions,
)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_USE_SHM", "0")

try:
    import mne
except ImportError:
    mne = None

try:
    from scipy import io as scipy_io
    from scipy import signal
except ImportError:
    scipy_io = None
    signal = None

LOGGER = logging.getLogger("eeg_preprocessing")

SUPPORTED_FILE_SUFFIXES = {".edf", ".gdf", ".mat", ".npy", ".npz", ".csv", ".txt"}
NON_EEG_CHANNEL_TOKENS = {
    "eog",
    "emg",
    "ecg",
    "ekg",
    "stim",
    "status",
    "marker",
    "event",
    "trigger",
    "resp",
    "gsr",
    "audio",
    "mic",
}

# Standard 21-channel motor cortex strip for MI classification
# Fronto-central + Central + Centro-parietal: covers primary/secondary motor cortex
# and somatosensory cortex on both hemispheres.
# Names match after _normalize_channel_name (strips dots, uppercases).
PHYSIONET_MOTOR_CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
]
BCI_IV_2A_LABEL_NAMES = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
}
PHYSIONET_LABEL_NAMES = {
    "T0": "rest",
    "T1": "task_1",
    "T2": "task_2",
}
PHYSIONET_GLOBAL_LABEL_TO_INDEX = {
    "rest": 0,
    "task_1": 1,
    "task_2": 2,
}
BCI_IV_2A_GLOBAL_LABEL_TO_INDEX = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}
DEFAULT_EEG_BANDS = {
    "mu": (8.0, 12.0),       # MI-specific ERD band
    "beta": (13.0, 30.0),   # motor rebound ERS band
    "gamma": (30.0, 45.0),  # motor activation gamma band
}
REQUIRED_FEATURE_VIEW_KEYS = ("time", "freq", "bands")

# â”€â”€â”€ PhysioNet Run-Type Filtering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Mapping from task-filter name to the set of run numbers to include.
# PhysioNet EEG Motor Movement/Imagery Dataset run definitions:
#   R01, R02:       Baseline (eyes open / eyes closed)
#   R03, R07, R11:  Motor imagery -- left hand (T1) / right hand (T2)
#   R04, R08, R12:  Motor imagery -- both fists (T1) / both feet (T2)
#   R05, R09, R13:  Motor execution -- left hand (T1) / right hand (T2)
#   R06, R10, R14:  Motor execution -- both fists (T1) / both feet (T2)
PHYSIONET_TASK_RUNS: dict[str, set[int]] = {
    "mi_hand":     {3, 7, 11},         # Imagined left/right hand (RECOMMENDED)
    "mi_fistfeet": {4, 8, 12},         # Imagined fists/feet
    "me_hand":     {5, 9, 13},         # Actual left/right hand
    "me_fistfeet": {6, 10, 14},        # Actual fists/feet
    "all_mi":      {3, 4, 7, 8, 11, 12},  # All motor imagery
    "all_me":      {5, 6, 9, 10, 13, 14}, # All motor execution
    "all":         set(range(1, 15)),   # All 14 runs (LEGACY -- do NOT use)
}
PHYSIONET_MI_HAND_GLOBAL_LABEL_TO_INDEX = {
    "rest": 0,
    "left_hand": 1,
    "right_hand": 2,
}
PHYSIONET_ME_HAND_GLOBAL_LABEL_TO_INDEX = {
    "rest": 0,
    "left_hand": 1,
    "right_hand": 2,
}
PHYSIONET_MI_FISTFEET_GLOBAL_LABEL_TO_INDEX = {
    "rest": 0,
    "fists": 1,
    "feet": 2,
}
PHYSIONET_ME_FISTFEET_GLOBAL_LABEL_TO_INDEX = {
    "rest": 0,
    "fists": 1,
    "feet": 2,
}


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _get_torch() -> Any:
    torch_module = import_module("torch")
    try:
        torch_module.set_num_threads(1)
    except RuntimeError:
        pass
    if hasattr(torch_module, "set_num_interop_threads"):
        try:
            torch_module.set_num_interop_threads(1)
        except RuntimeError:
            pass
    return torch_module


def _require_dependency(module: Any, package_name: str) -> None:
    if module is None:
        raise ImportError(
            f"{package_name} is required for this operation. "
            f"Install it before running this function."
        )


def _natural_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    if re.fullmatch(r"-?\d+", text):
        return (0, int(text))
    return (1, text)


def _sorted_labels(labels: Iterable[Any]) -> list[str]:
    return [str(label) for label in sorted({str(label) for label in labels}, key=_natural_key)]


def _normalize_channel_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def _physionet_run_number(file_path: Path) -> int | None:
    match = re.search(r"R(\d{2})", file_path.name, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _physionet_label_name(file_path: Path, raw_label: str) -> str:
    raw_label = str(raw_label)
    if raw_label == "T0":
        return "rest"

    run_num = _physionet_run_number(file_path)
    if run_num in PHYSIONET_TASK_RUNS["mi_hand"] or run_num in PHYSIONET_TASK_RUNS["me_hand"]:
        if raw_label == "T1":
            return "left_hand"
        if raw_label == "T2":
            return "right_hand"
    if run_num in PHYSIONET_TASK_RUNS["mi_fistfeet"] or run_num in PHYSIONET_TASK_RUNS["me_fistfeet"]:
        if raw_label == "T1":
            return "fists"
        if raw_label == "T2":
            return "feet"

    return PHYSIONET_LABEL_NAMES.get(raw_label, raw_label)


def _physionet_global_label_to_index(task_filter: str) -> Mapping[str, int]:
    normalized = str(task_filter).strip().lower()
    if normalized == "mi_hand":
        return PHYSIONET_MI_HAND_GLOBAL_LABEL_TO_INDEX
    if normalized == "me_hand":
        return PHYSIONET_ME_HAND_GLOBAL_LABEL_TO_INDEX
    if normalized == "mi_fistfeet":
        return PHYSIONET_MI_FISTFEET_GLOBAL_LABEL_TO_INDEX
    if normalized == "me_fistfeet":
        return PHYSIONET_ME_FISTFEET_GLOBAL_LABEL_TO_INDEX
    return PHYSIONET_GLOBAL_LABEL_TO_INDEX


def _is_probable_eeg_channel(name: str) -> bool:
    lowered = str(name).strip().lower()
    return not any(token in lowered for token in NON_EEG_CHANNEL_TOKENS)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _synthesize_channel_names(n_channels: int) -> list[str]:
    return [f"ch_{index:03d}" for index in range(n_channels)]


def _as_float32_contiguous(array: np.ndarray | Sequence[Any]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32))


def _as_int64_contiguous(array: np.ndarray | Sequence[Any]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.int64))


def _validate_feature_views(
    feature_views: Mapping[str, np.ndarray] | None,
    *,
    expected_length: int,
    prefix: str,
) -> dict[str, np.ndarray]:
    if not isinstance(feature_views, Mapping):
        raise ValueError(f"{prefix} feature views are required but were not provided.")

    validated: dict[str, np.ndarray] = {}
    missing_keys = [key for key in REQUIRED_FEATURE_VIEW_KEYS if key not in feature_views]
    if missing_keys:
        raise ValueError(f"{prefix} feature views are missing required keys: {missing_keys}.")

    for key in REQUIRED_FEATURE_VIEW_KEYS:
        array = np.asarray(feature_views[key], dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"{prefix} view {key!r} must have shape (N, C, T), got {array.shape}.")
        if int(array.shape[0]) != int(expected_length):
            raise ValueError(
                f"{prefix} view {key!r} has {array.shape[0]} samples but expected {expected_length}."
            )
        if key in {"freq", "bands"} and int(array.shape[1]) == 0:
            raise ValueError(f"{prefix} view {key!r} is empty after feature extraction.")
        validated[key] = np.ascontiguousarray(array)
    return validated


def _verify_saved_outputs(
    saved_paths: Mapping[str, str],
    *,
    required_keys: Sequence[str],
) -> None:
    missing_keys = [str(key) for key in required_keys if str(key) not in saved_paths]
    if missing_keys:
        raise RuntimeError(f"Saving failed. Missing expected output entries: {missing_keys}.")

    missing_files: list[str] = []
    empty_files: list[str] = []
    for key in required_keys:
        output_path = Path(saved_paths[str(key)])
        if not output_path.exists():
            missing_files.append(str(output_path))
            continue
        if output_path.is_file() and output_path.stat().st_size == 0:
            empty_files.append(str(output_path))

    if missing_files or empty_files:
        details: list[str] = []
        if missing_files:
            details.append(f"missing files={missing_files}")
        if empty_files:
            details.append(f"empty files={empty_files}")
        raise RuntimeError("Saving failed verification: " + "; ".join(details))


def _current_ram_usage_mb() -> float | None:
    try:
        import resource

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if usage <= 0:
        return None
    if os.name == "posix" and "darwin" in os.sys.platform:
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _log_memory(message: str) -> None:
    usage_mb = _current_ram_usage_mb()
    if usage_mb is None:
        print(message)
        return
    print(f"{message} | RAM={usage_mb:.1f} MB")


def _create_time_alias(source_path: Path, alias_path: Path) -> None:
    if alias_path.exists() or alias_path.is_symlink():
        alias_path.unlink()
    try:
        os.link(source_path, alias_path)
    except OSError:
        shutil.copyfile(source_path, alias_path)


def _resolve_input_path(path: str | Path) -> tuple[Path, dict[str, Any]]:
    input_path = Path(path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    resolution_meta: dict[str, Any] = {
        "input_path": str(input_path.resolve()),
        "input_is_directory": input_path.is_dir(),
    }

    if input_path.is_file():
        resolution_meta["selected_file"] = str(input_path.resolve())
        return input_path, resolution_meta

    candidates = sorted(
        file_path
        for file_path in input_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(
            f"No supported EEG files found under directory: {input_path}"
        )

    selected = candidates[0]
    resolution_meta["selected_file"] = str(selected.resolve())
    resolution_meta["available_file_count"] = len(candidates)
    resolution_meta["available_file_formats"] = dict(
        sorted(Counter(file_path.suffix.lower().lstrip(".") for file_path in candidates).items())
    )
    LOGGER.info(
        "Directory input detected. Using %s for inspection/loading.",
        selected,
    )
    return selected, resolution_meta


def _extract_path_metadata(file_path: Path) -> dict[str, Any]:
    subject_match = re.search(r"[\\/](?:S|A)(\d{2,3})", str(file_path))
    session_match = re.search(r"([RTE])(\d{1,2})\.(?:edf|gdf|mat)$", file_path.name, flags=re.IGNORECASE)
    metadata: dict[str, Any] = {
        "source_file": str(file_path.resolve()),
        "subject_id": None,
        "session_id": None,
        "recording_id": file_path.stem,
    }
    if subject_match:
        metadata["subject_id"] = int(subject_match.group(1))
    if session_match:
        prefix = session_match.group(1).upper()
        number = int(session_match.group(2))
        metadata["session_id"] = f"{prefix}{number:02d}"
    return metadata


def _detect_dataset_type(file_path: Path) -> str:
    path_text = str(file_path).lower()
    suffix = file_path.suffix.lower()
    if suffix == ".edf" or "physionet" in path_text or re.search(r"s\d{3}r\d+\.edf$", path_text):
        return "physionet_eeg"
    if suffix in {".gdf", ".mat"} and (
        "bciciv" in path_text or re.search(r"a\d{2}[te]\.(gdf|mat)$", path_text)
    ):
        return "bci_competition_iv_2a"
    return "unknown"


def _read_raw_file(file_path: Path, preload: bool) -> Any:
    _require_dependency(mne, "mne")
    suffix = file_path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(file_path), preload=preload, verbose="ERROR")
    if suffix == ".gdf":
        return mne.io.read_raw_gdf(str(file_path), preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported raw file format: {file_path.suffix}")


def _annotation_mask(dataset_type: str, descriptions: np.ndarray) -> np.ndarray:
    desc = np.asarray(descriptions, dtype=str)
    if desc.size == 0:
        return np.zeros(0, dtype=bool)
    if dataset_type == "physionet_eeg":
        return np.char.startswith(desc.astype("U"), "T")
    if dataset_type == "bci_competition_iv_2a":
        return np.isin(desc, list(BCI_IV_2A_LABEL_NAMES))
    return desc != ""


def _summarize_annotations(
    dataset_type: str,
    descriptions: np.ndarray,
) -> tuple[dict[str, int], str | None]:
    if descriptions.size == 0:
        return {}, None

    mask = _annotation_mask(dataset_type, descriptions)
    selected = descriptions[mask] if mask.any() else descriptions
    distribution = dict(sorted(Counter(map(str, selected)).items(), key=lambda item: _natural_key(item[0])))
    labels_format = None
    if distribution:
        if dataset_type == "bci_competition_iv_2a":
            labels_format = "GDF class event codes"
        elif dataset_type == "physionet_eeg":
            labels_format = "EDF annotation labels"
        else:
            labels_format = "Annotation descriptions"
    return distribution, labels_format


def _inspect_raw_file(file_path: Path, resolution_meta: Mapping[str, Any]) -> dict[str, Any]:
    raw = _read_raw_file(file_path, preload=False)
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    label_distribution, labels_format = _summarize_annotations(
        _detect_dataset_type(file_path), descriptions
    )
    mask = _annotation_mask(_detect_dataset_type(file_path), descriptions)

    metadata: dict[str, Any] = {
        **dict(resolution_meta),
        "dataset_type": _detect_dataset_type(file_path),
        "file_format": file_path.suffix.lower().lstrip("."),
        "sampling_frequency": float(raw.info["sfreq"]),
        "num_channels": int(raw.info["nchan"]),
        "channel_names": list(raw.ch_names),
        "channel_types": list(raw.get_channel_types()),
        "data_shape": (int(raw.info["nchan"]), int(raw.n_times)),
        "axis_order": "channels x time",
        "num_trials": int(mask.sum()) if mask.any() else None,
        "labels_format": labels_format,
        "label_distribution": label_distribution,
        "labels_available": bool(label_distribution),
        "needs_label_extraction": bool(mask.any()),
        "annotation_count": int(len(raw.annotations)),
        "annotation_descriptions": _sorted_labels(descriptions),
        "is_continuous": True,
        "is_epoched": False,
        "inspected_path": str(file_path.resolve()),
    }
    return metadata


def _flatten_numeric_candidates(
    obj: Any,
    prefix: str = "root",
    depth: int = 0,
    max_depth: int = 5,
) -> list[tuple[str, np.ndarray]]:
    if depth > max_depth:
        return []

    candidates: list[tuple[str, np.ndarray]] = []
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            candidates.append((prefix, obj))
        elif obj.dtype == object and obj.ndim == 0:
            candidates.extend(_flatten_numeric_candidates(obj.item(), prefix, depth + 1, max_depth))
        return candidates

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key).startswith("__"):
                continue
            candidates.extend(_flatten_numeric_candidates(value, f"{prefix}.{key}", depth + 1, max_depth))
        return candidates

    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            if str(key).startswith("_"):
                continue
            candidates.extend(_flatten_numeric_candidates(value, f"{prefix}.{key}", depth + 1, max_depth))
        return candidates

    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj[:16]):
            candidates.extend(_flatten_numeric_candidates(value, f"{prefix}[{index}]", depth + 1, max_depth))
        return candidates

    return candidates


def _is_integer_like(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.number) or array.size == 0:
        return False
    flat = np.ravel(array)
    if flat.size > 2048:
        flat = flat[:2048]
    return np.allclose(flat, np.round(flat), equal_nan=False)


def _score_data_candidate(name: str, array: np.ndarray) -> float:
    lowered = name.lower()
    score = 0.0
    if any(token in lowered for token in ("data", "eeg", "signal", "signals", "x", "trial", "trials")):
        score += 10.0
    if array.ndim >= 2:
        score += 5.0
    score += min(math.log10(max(array.size, 1)), 8.0)
    return score


def _score_label_candidate(name: str, array: np.ndarray) -> float:
    lowered = name.lower()
    score = 0.0
    if any(token in lowered for token in ("label", "labels", "target", "targets", "class", "classes", "y")):
        score += 10.0
    if array.ndim <= 2:
        score += 2.0
    if _is_integer_like(array):
        score += 4.0
    if array.size <= 100000:
        score += 1.0
    return score


def _infer_2d_axis_order(array: np.ndarray) -> tuple[np.ndarray, str]:
    if array.ndim != 2:
        raise ValueError("Expected a 2D array for axis inference.")
    if array.shape[0] <= array.shape[1]:
        return array, "channels x time"
    return array.T, "channels x time (transposed from time x channels)"


def _standardize_epoched_array(
    array: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if array.ndim != 3:
        raise ValueError("Expected a 3D array for epoched standardization.")

    shape = array.shape
    trial_axis: int | None = None
    if labels is not None:
        label_count = int(np.asarray(labels).size)
        matches = [axis for axis, size in enumerate(shape) if size == label_count]
        if len(matches) == 1:
            trial_axis = matches[0]

    if trial_axis is None:
        trial_axis = int(np.argmin(shape[:2])) if max(shape[:2]) > shape[2] else 0
        if shape[trial_axis] > 512:
            trial_axis = int(np.argmin(shape))

    time_axis = int(np.argmax(shape))
    if time_axis == trial_axis:
        remaining = [axis for axis in range(3) if axis != trial_axis]
        time_axis = remaining[int(np.argmax([shape[axis] for axis in remaining]))]

    channel_axis = next(axis for axis in range(3) if axis not in {trial_axis, time_axis})
    standardized = np.transpose(array, (trial_axis, channel_axis, time_axis))
    axis_order = "trials x channels x time"
    return standardized, axis_order


def _encode_labels(raw_labels: Sequence[Any]) -> tuple[np.ndarray, dict[int, str], dict[str, int]]:
    labels = np.asarray(raw_labels).reshape(-1)
    labels_as_str = np.asarray([str(value) for value in labels], dtype=object)
    unique = _sorted_labels(labels_as_str)
    mapping = {label: index for index, label in enumerate(unique)}
    encoded = np.asarray([mapping[value] for value in labels_as_str], dtype=np.int64)
    inverse_mapping = {index: label for label, index in mapping.items()}
    distribution = {
        label: int((labels_as_str == label).sum())
        for label in unique
    }
    return encoded, inverse_mapping, distribution


def _extract_epochs(
    data: np.ndarray,
    start_samples: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    if window_samples <= 0:
        raise ValueError("Window length must be positive for epoch extraction.")
    indices = start_samples[:, None] + np.arange(window_samples, dtype=np.int64)[None, :]
    flattened = np.take(data, indices.ravel(), axis=1)
    return flattened.reshape(data.shape[0], len(start_samples), window_samples).transpose(1, 0, 2)


def _build_mne_trial_metadata(
    *,
    file_path: Path,
    raw: Any,
    X: np.ndarray,
    y: np.ndarray | None,
    raw_labels: Sequence[Any] | None,
    label_distribution: dict[str, int],
    label_mapping: dict[int, str] | None,
    extraction_method: str,
    is_continuous: bool,
) -> dict[str, Any]:
    dataset_type = _detect_dataset_type(file_path)
    metadata: dict[str, Any] = {
        "dataset_type": dataset_type,
        "file_format": file_path.suffix.lower().lstrip("."),
        "sampling_frequency": float(raw.info["sfreq"]),
        "num_channels": int(X.shape[1]),
        "channel_names": list(raw.ch_names),
        "channel_types": list(raw.get_channel_types()),
        "data_shape": tuple(int(dim) for dim in X.shape),
        "axis_order": "trials x channels x time",
        "num_trials": int(X.shape[0]),
        "labels_format": "encoded integer labels" if y is not None else None,
        "label_distribution": label_distribution,
        "labels_available": y is not None,
        "label_mapping": label_mapping or {},
        "raw_label_names": list(raw_labels) if raw_labels is not None else [],
        "is_continuous": is_continuous,
        "is_epoched": not is_continuous,
        "annotation_count": int(len(raw.annotations)),
        "annotation_descriptions": _sorted_labels(raw.annotations.description),
        "extraction_method": extraction_method,
        "inspected_path": str(file_path.resolve()),
        "uncertainty_metadata": {
            "source_dataset_type": dataset_type,
            "source_file": str(file_path.resolve()),
            "original_channel_names": list(raw.ch_names),
            "sampling_frequency": float(raw.info["sfreq"]),
            "label_distribution": label_distribution,
            "label_mapping": label_mapping or {},
        },
        "preprocessing_log": [],
        "seed": None,
    }
    return metadata


def _extract_physionet_trials(file_path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    raw = _read_raw_file(file_path, preload=True)
    data = raw.get_data().astype(np.float32, copy=False)
    sfreq = float(raw.info["sfreq"])
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durations = np.asarray(raw.annotations.duration, dtype=float)

    mask = _annotation_mask("physionet_eeg", descriptions)
    if not mask.any():
        X = data[np.newaxis, :, :]
        meta = _build_mne_trial_metadata(
            file_path=file_path,
            raw=raw,
            X=X,
            y=None,
            raw_labels=None,
            label_distribution={},
            label_mapping=None,
            extraction_method="continuous_raw_fallback",
            is_continuous=True,
        )
        return X, None, meta

    usable_durations = durations[mask]
    window_samples = int(np.floor(np.nanmin(usable_durations) * sfreq))
    if window_samples <= 0:
        raise ValueError(f"Could not infer a positive trial window from {file_path}.")

    start_samples = np.rint(onsets[mask] * sfreq).astype(np.int64)
    valid_mask = start_samples + window_samples <= data.shape[-1]
    if not valid_mask.any():
        raise ValueError(f"No valid PhysioNet annotation windows fit within {file_path}.")

    selected_labels = descriptions[mask][valid_mask]
    X = _extract_epochs(data, start_samples[valid_mask], window_samples)
    y, inverse_mapping, distribution = _encode_labels(selected_labels)
    label_mapping = {
        index: _physionet_label_name(file_path, raw_label)
        for index, raw_label in inverse_mapping.items()
    }
    meta = _build_mne_trial_metadata(
        file_path=file_path,
        raw=raw,
        X=X,
        y=y,
        raw_labels=selected_labels.tolist(),
        label_distribution=distribution,
        label_mapping=label_mapping,
        extraction_method="annotation_duration_min_crop",
        is_continuous=False,
    )
    meta["raw_label_mapping"] = inverse_mapping
    meta["trial_window_samples"] = int(window_samples)
    meta["trial_window_sec"] = float(window_samples / sfreq)
    return X.astype(np.float32, copy=False), y, meta


def _extract_bci_trials(file_path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    raw = _read_raw_file(file_path, preload=True)
    data = raw.get_data().astype(np.float32, copy=False)
    sfreq = float(raw.info["sfreq"])
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durations = np.asarray(raw.annotations.duration, dtype=float)

    cue_mask = np.isin(descriptions, list(BCI_IV_2A_LABEL_NAMES))
    trial_mask = descriptions == "768"

    if cue_mask.any():
        cue_onsets = onsets[cue_mask]
        cue_labels = descriptions[cue_mask]

        if trial_mask.any():
            trial_onsets = onsets[trial_mask]
            trial_ends = trial_onsets + np.maximum(durations[trial_mask], 0.0)
            cue_to_trial = np.searchsorted(trial_onsets, cue_onsets, side="right") - 1
            valid_mask = cue_to_trial >= 0
            available = np.full(cue_onsets.shape, np.nan, dtype=float)
            valid_indices = np.where(valid_mask)[0]
            available[valid_indices] = trial_ends[cue_to_trial[valid_indices]] - cue_onsets[valid_indices]
        else:
            next_onsets = np.r_[cue_onsets[1:], data.shape[-1] / sfreq]
            available = next_onsets - cue_onsets

        positive_available = available[np.isfinite(available) & (available > 0)]
        if positive_available.size == 0:
            positive_available = durations[cue_mask][durations[cue_mask] > 0]
        if positive_available.size == 0:
            raise ValueError(f"Could not infer BCI trial windows from annotations in {file_path}.")

        window_samples = int(np.floor(np.min(positive_available) * sfreq))
        if window_samples <= 0:
            raise ValueError(f"Could not derive a positive BCI epoch length from {file_path}.")

        start_samples = np.rint(cue_onsets * sfreq).astype(np.int64)
        valid_mask = start_samples + window_samples <= data.shape[-1]
        if not valid_mask.any():
            raise ValueError(f"No valid BCI cue windows fit within {file_path}.")

        selected_labels = cue_labels[valid_mask]
        X = _extract_epochs(data, start_samples[valid_mask], window_samples)
        y, inverse_mapping, distribution = _encode_labels(selected_labels)
        label_mapping = {
            index: BCI_IV_2A_LABEL_NAMES.get(raw_label, raw_label)
            for index, raw_label in inverse_mapping.items()
        }
        meta = _build_mne_trial_metadata(
            file_path=file_path,
            raw=raw,
            X=X,
            y=y,
            raw_labels=selected_labels.tolist(),
            label_distribution=distribution,
            label_mapping=label_mapping,
            extraction_method="cue_to_trial_end_min_crop",
            is_continuous=False,
        )
        meta["raw_label_mapping"] = inverse_mapping
        meta["trial_window_samples"] = int(window_samples)
        meta["trial_window_sec"] = float(window_samples / sfreq)
        return X.astype(np.float32, copy=False), y, meta

    if trial_mask.any():
        window_samples = int(np.floor(np.nanmin(durations[trial_mask]) * sfreq))
        if window_samples > 0:
            start_samples = np.rint(onsets[trial_mask] * sfreq).astype(np.int64)
            valid_mask = start_samples + window_samples <= data.shape[-1]
            X = _extract_epochs(data, start_samples[valid_mask], window_samples)
            meta = _build_mne_trial_metadata(
                file_path=file_path,
                raw=raw,
                X=X,
                y=None,
                raw_labels=None,
                label_distribution={},
                label_mapping=None,
                extraction_method="trial_start_unlabeled_epochs",
                is_continuous=False,
            )
            meta["trial_window_samples"] = int(window_samples)
            meta["trial_window_sec"] = float(window_samples / sfreq)
            return X.astype(np.float32, copy=False), None, meta

    X = data[np.newaxis, :, :]
    meta = _build_mne_trial_metadata(
        file_path=file_path,
        raw=raw,
        X=X,
        y=None,
        raw_labels=None,
        label_distribution={},
        label_mapping=None,
        extraction_method="continuous_raw_fallback",
        is_continuous=True,
    )
    return X, None, meta


def _extract_generic_annotation_trials(file_path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    raw = _read_raw_file(file_path, preload=True)
    data = raw.get_data().astype(np.float32, copy=False)
    sfreq = float(raw.info["sfreq"])
    descriptions = np.asarray(raw.annotations.description, dtype=str)
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durations = np.asarray(raw.annotations.duration, dtype=float)

    if descriptions.size == 0:
        X = data[np.newaxis, :, :]
        meta = _build_mne_trial_metadata(
            file_path=file_path,
            raw=raw,
            X=X,
            y=None,
            raw_labels=None,
            label_distribution={},
            label_mapping=None,
            extraction_method="continuous_raw_fallback",
            is_continuous=True,
        )
        return X, None, meta

    positive_durations = durations[durations > 0]
    if positive_durations.size == 0:
        X = data[np.newaxis, :, :]
        meta = _build_mne_trial_metadata(
            file_path=file_path,
            raw=raw,
            X=X,
            y=None,
            raw_labels=None,
            label_distribution={},
            label_mapping=None,
            extraction_method="continuous_raw_no_annotation_duration",
            is_continuous=True,
        )
        return X, None, meta

    window_samples = int(np.floor(np.min(positive_durations) * sfreq))
    if window_samples <= 0:
        raise ValueError(f"Could not derive a positive window length from {file_path}.")

    start_samples = np.rint(onsets * sfreq).astype(np.int64)
    valid_mask = start_samples + window_samples <= data.shape[-1]
    if not valid_mask.any():
        raise ValueError(f"No valid annotation windows fit within {file_path}.")

    selected_labels = descriptions[valid_mask]
    X = _extract_epochs(data, start_samples[valid_mask], window_samples)
    y, inverse_mapping, distribution = _encode_labels(selected_labels)
    meta = _build_mne_trial_metadata(
        file_path=file_path,
        raw=raw,
        X=X,
        y=y,
        raw_labels=selected_labels.tolist(),
        label_distribution=distribution,
        label_mapping=inverse_mapping,
        extraction_method="generic_annotation_duration_min_crop",
        is_continuous=False,
    )
    meta["trial_window_samples"] = int(window_samples)
    meta["trial_window_sec"] = float(window_samples / sfreq)
    return X.astype(np.float32, copy=False), y, meta


def _load_mat_file(file_path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    _require_dependency(scipy_io, "scipy")
    mat = scipy_io.loadmat(
        str(file_path),
        squeeze_me=True,
        struct_as_record=False,
        simplify_cells=True,
    )

    candidates = _flatten_numeric_candidates(mat)
    if not candidates:
        raise ValueError(f"No numeric arrays found in MAT file: {file_path}")

    data_name, data_array = max(candidates, key=lambda item: _score_data_candidate(item[0], item[1]))
    label_candidates = [
        (name, array)
        for name, array in candidates
        if array.shape != data_array.shape
    ]
    label_name: str | None = None
    raw_labels: np.ndarray | None = None
    if label_candidates:
        label_name, raw_labels = max(
            label_candidates,
            key=lambda item: _score_label_candidate(item[0], item[1]),
        )
        raw_labels = np.asarray(raw_labels).reshape(-1)

    data_array = np.asarray(data_array)
    axis_order = ""
    if data_array.ndim == 2:
        standardized, detected_order = _infer_2d_axis_order(data_array)
        X = standardized[np.newaxis, :, :].astype(np.float32, copy=False)
        axis_order = detected_order
        is_continuous = True
    elif data_array.ndim == 3:
        standardized, detected_order = _standardize_epoched_array(data_array, raw_labels)
        X = standardized.astype(np.float32, copy=False)
        axis_order = detected_order
        is_continuous = False
    else:
        raise ValueError(
            f"Unsupported MAT data rank {data_array.ndim} in {file_path}. "
            "Expected 2D or 3D EEG arrays."
        )

    y: np.ndarray | None = None
    label_mapping: dict[int, str] = {}
    distribution: dict[str, int] = {}
    if raw_labels is not None:
        if raw_labels.size == X.shape[0]:
            y, label_mapping, distribution = _encode_labels(raw_labels)
        else:
            LOGGER.warning(
                "Ignoring MAT labels from %s because label count %s does not match trial count %s.",
                label_name,
                raw_labels.size,
                X.shape[0],
            )

    n_channels = int(X.shape[1])
    meta: dict[str, Any] = {
        "dataset_type": _detect_dataset_type(file_path),
        "file_format": "mat",
        "sampling_frequency": None,
        "num_channels": n_channels,
        "channel_names": _synthesize_channel_names(n_channels),
        "channel_types": ["eeg"] * n_channels,
        "data_shape": tuple(int(dim) for dim in X.shape),
        "axis_order": "trials x channels x time",
        "detected_input_axis_order": axis_order,
        "num_trials": int(X.shape[0]),
        "labels_format": "encoded integer labels" if y is not None else None,
        "label_distribution": distribution,
        "labels_available": y is not None,
        "label_mapping": label_mapping,
        "is_continuous": is_continuous,
        "is_epoched": not is_continuous,
        "inspected_path": str(file_path.resolve()),
        "source_data_key": data_name,
        "source_label_key": label_name,
        "uncertainty_metadata": {
            "source_dataset_type": _detect_dataset_type(file_path),
            "source_file": str(file_path.resolve()),
            "source_data_key": data_name,
            "source_label_key": label_name,
            "label_distribution": distribution,
            "label_mapping": label_mapping,
        },
        "preprocessing_log": [],
        "seed": None,
    }
    return X, y, meta


def _load_generic_array_file(file_path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(str(file_path), allow_pickle=True)
        source_key = None
    elif suffix == ".npz":
        loaded = np.load(str(file_path), allow_pickle=True)
        keys = list(loaded.files)
        if not keys:
            raise ValueError(f"No arrays found in NPZ file: {file_path}")
        source_key = max(keys, key=lambda key: loaded[key].size)
        array = loaded[source_key]
    elif suffix in {".csv", ".txt"}:
        array = np.loadtxt(str(file_path), delimiter="," if suffix == ".csv" else None)
        source_key = None
    else:
        raise ValueError(f"Unsupported generic file format: {file_path.suffix}")

    array = np.asarray(array)
    if array.ndim == 2:
        standardized, detected_order = _infer_2d_axis_order(array)
        X = standardized[np.newaxis, :, :].astype(np.float32, copy=False)
        is_continuous = True
    elif array.ndim == 3:
        standardized, detected_order = _standardize_epoched_array(array)
        X = standardized.astype(np.float32, copy=False)
        is_continuous = False
    else:
        raise ValueError(
            f"Unsupported array rank {array.ndim} in {file_path}. Expected 2D or 3D data."
        )

    n_channels = int(X.shape[1])
    meta: dict[str, Any] = {
        "dataset_type": _detect_dataset_type(file_path),
        "file_format": suffix.lstrip("."),
        "sampling_frequency": None,
        "num_channels": n_channels,
        "channel_names": _synthesize_channel_names(n_channels),
        "channel_types": ["eeg"] * n_channels,
        "data_shape": tuple(int(dim) for dim in X.shape),
        "axis_order": "trials x channels x time",
        "detected_input_axis_order": detected_order,
        "num_trials": int(X.shape[0]),
        "labels_format": None,
        "label_distribution": {},
        "labels_available": False,
        "label_mapping": {},
        "is_continuous": is_continuous,
        "is_epoched": not is_continuous,
        "inspected_path": str(file_path.resolve()),
        "source_data_key": source_key,
        "uncertainty_metadata": {
            "source_dataset_type": _detect_dataset_type(file_path),
            "source_file": str(file_path.resolve()),
            "source_data_key": source_key,
        },
        "preprocessing_log": [],
        "seed": None,
    }
    return X, None, meta


def inspect_dataset(path: str) -> dict[str, Any]:
    file_path, resolution_meta = _resolve_input_path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".edf", ".gdf"}:
        metadata = _inspect_raw_file(file_path, resolution_meta)
    elif suffix == ".mat":
        X, y, metadata = _load_mat_file(file_path)
        metadata = {
            **resolution_meta,
            **metadata,
            "data_shape": tuple(int(dim) for dim in X.shape),
            "num_trials": int(X.shape[0]),
            "labels_available": y is not None,
            "label_distribution": metadata.get("label_distribution", {}),
        }
    else:
        X, y, metadata = _load_generic_array_file(file_path)
        metadata = {
            **resolution_meta,
            **metadata,
            "data_shape": tuple(int(dim) for dim in X.shape),
            "num_trials": int(X.shape[0]),
            "labels_available": y is not None,
            "label_distribution": metadata.get("label_distribution", {}),
        }

    serializable_metadata = _to_serializable(metadata)
    pprint(serializable_metadata)
    return serializable_metadata


def load_eeg_data(path: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    file_path, resolution_meta = _resolve_input_path(path)
    dataset_type = _detect_dataset_type(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".edf":
        if dataset_type == "physionet_eeg":
            X, y, meta = _extract_physionet_trials(file_path)
        else:
            X, y, meta = _extract_generic_annotation_trials(file_path)
    elif suffix == ".gdf":
        if dataset_type == "bci_competition_iv_2a":
            X, y, meta = _extract_bci_trials(file_path)
        else:
            X, y, meta = _extract_generic_annotation_trials(file_path)
    elif suffix == ".mat":
        X, y, meta = _load_mat_file(file_path)
    else:
        X, y, meta = _load_generic_array_file(file_path)

    meta = {**resolution_meta, **_extract_path_metadata(file_path), **meta}
    meta["data_shape"] = tuple(int(dim) for dim in X.shape)
    meta["num_trials"] = int(X.shape[0])
    meta["num_channels"] = int(X.shape[1])
    meta["dtype"] = str(X.dtype)
    meta["seed"] = meta.get("seed")

    return X.astype(np.float32, copy=False), y, _to_serializable(meta)


class EEGPreprocessor:
    def __init__(
        self,
        *,
        low: float = 4.0,
        high: float = 30.0,
        window_sec: float = 2.0,
        overlap_sec: float = 0.0,
        target_channels: Sequence[str] | None = None,
        filter_order: int = 4,
        normalization_scope: str = "trial",  # kept for API compat; always per-trial
        include_gamma: bool = False,
        band_definitions: Mapping[str, tuple[float, float]] | None = None,
        band_method: str = "fft_segments",
        band_segments: int = 8,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.low = low
        self.high = high
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.target_channels = list(target_channels) if target_channels is not None else None
        self.filter_order = filter_order
        self.normalization_scope = str(normalization_scope)
        self.band_method = str(band_method).strip().lower()
        self.band_segments = max(int(band_segments), 1)
        self.band_definitions = prepare_band_definitions(
            band_definitions,
            include_gamma=include_gamma,
        )
        self.seed = seed
        self.verbose = verbose
        set_global_seed(seed)

    def _log(self, meta: dict[str, Any], message: str) -> None:
        meta.setdefault("preprocessing_log", []).append(message)
        if self.verbose:
            LOGGER.info(message)

    def bandpass_filter(
        self,
        X: np.ndarray,
        sfreq: float,
        low: float | None = None,
        high: float | None = None,
    ) -> np.ndarray:
        # FIX 1: use instance defaults when not explicitly specified,
        # preventing silent 7 Hz cutoff from old hardcoded defaults
        low = self.low if low is None else float(low)
        high = self.high if high is None else float(high)
        _require_dependency(signal, "scipy")
        if sfreq <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {sfreq}.")

        X = np.asarray(X, dtype=np.float32)
        nyquist = sfreq / 2.0
        adj_low = max(float(low), 0.01)
        adj_high = min(float(high), nyquist - 0.01)
        if adj_low >= adj_high:
            raise ValueError(
                f"Invalid bandpass range after adjustment: low={adj_low}, high={adj_high}, nyquist={nyquist}."
            )

        sos = signal.butter(
            self.filter_order,
            [adj_low / nyquist, adj_high / nyquist],
            btype="bandpass",
            output="sos",
        )
        try:
            filtered = signal.sosfiltfilt(sos, X, axis=-1)
        except ValueError:
            filtered = signal.sosfilt(sos, X, axis=-1)
        return filtered.astype(np.float32, copy=False)

    def _window_step_samples(self, sfreq: float, window_sec: float) -> tuple[int, int]:
        window_samples = int(round(window_sec * sfreq))
        if window_samples <= 0:
            raise ValueError(f"window_sec={window_sec} yields an invalid epoch length.")
        overlap_samples = int(round(self.overlap_sec * sfreq))
        if overlap_samples < 0:
            raise ValueError(f"overlap_sec must be non-negative, got {self.overlap_sec}.")
        if overlap_samples >= window_samples:
            raise ValueError(
                f"overlap_sec={self.overlap_sec} must be smaller than window_sec={window_sec}."
            )
        step_samples = max(window_samples - overlap_samples, 1)
        return window_samples, step_samples

    def _window_continuous_data(
        self,
        X: np.ndarray,
        window_samples: int,
        step_samples: int,
    ) -> np.ndarray:
        if X.shape[-1] < window_samples:
            raise ValueError(
                f"Signal is too short for epoching: time samples={X.shape[-1]}, window_samples={window_samples}."
            )
        windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=window_samples, axis=-1)
        windows = windows[:, ::step_samples, :]
        if windows.shape[1] == 0:
            raise ValueError("No windows were produced from the continuous recording.")
        return windows.transpose(1, 0, 2).astype(np.float32, copy=False)

    def _window_epoched_data(
        self,
        X: np.ndarray,
        window_samples: int,
        step_samples: int,
    ) -> np.ndarray:
        if X.shape[-1] == window_samples and step_samples == window_samples:
            return X.astype(np.float32, copy=False)
        if X.shape[-1] < window_samples:
            raise ValueError(
                f"Epoched trial length {X.shape[-1]} is shorter than requested window length {window_samples}."
            )
        # Guard: if trial is already epoched, truncate to window_samples
        # instead of sliding-window which multiplies trials and breaks labels
        if step_samples == window_samples:
            return X[:, :, :window_samples].astype(np.float32, copy=False)
        windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=window_samples, axis=-1)
        windows = windows[:, :, ::step_samples, :]
        if windows.shape[2] == 0:
            raise ValueError("No windows were produced from the epoched recording.")
        return windows.transpose(0, 2, 1, 3).reshape(-1, X.shape[1], window_samples).astype(np.float32, copy=False)

    def epoch_data(
        self,
        X: np.ndarray,
        sfreq: float,
        window_sec: float = 2.0,
    ) -> np.ndarray:
        if sfreq <= 0:
            raise ValueError("Sampling frequency is required for epoch extraction.")

        X = np.asarray(X, dtype=np.float32)
        window_samples, step_samples = self._window_step_samples(float(sfreq), window_sec)
        if X.ndim == 2:
            return self._window_continuous_data(X, window_samples, step_samples)
        if X.ndim == 3:
            return self._window_epoched_data(X, window_samples, step_samples)
        raise ValueError(f"Expected 2D continuous data or 3D epoched data, got shape {X.shape}.")

    def select_channels(
        self,
        X: np.ndarray,
        ch_names: Sequence[str],
        target_channels: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D epoched data for channel selection, got shape {X.shape}.")
        if len(ch_names) != X.shape[1]:
            raise ValueError(
                f"Channel-name mismatch: len(ch_names)={len(ch_names)} vs X.shape[1]={X.shape[1]}."
            )

        normalized_to_index = {
            _normalize_channel_name(name): index for index, name in enumerate(ch_names)
        }
        if target_channels:
            indices: list[int] = []
            selected_names: list[str] = []
            for target in target_channels:
                normalized = _normalize_channel_name(target)
                if normalized in normalized_to_index:
                    index = normalized_to_index[normalized]
                    indices.append(index)
                    selected_names.append(ch_names[index])
            if not indices:
                raise ValueError("None of the requested target channels were found in the recording.")
        else:
            indices = [index for index, name in enumerate(ch_names) if _is_probable_eeg_channel(name)]
            if not indices:
                indices = list(range(len(ch_names)))
            selected_names = [ch_names[index] for index in indices]

        return X[:, indices, :].astype(np.float32, copy=False), selected_names

    def normalize(
        self,
        X: np.ndarray,
        *,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        X = np.asarray(X, dtype=np.float32)
        mean = X.mean(axis=2, keepdims=True, dtype=np.float32)
        std = X.std(axis=2, keepdims=True, dtype=np.float32)
        std = np.maximum(std, 1e-6).astype(np.float32, copy=False)
        X = (X - mean) / std
        if return_stats:
            stats = {
                "mean": mean.astype(np.float32, copy=False),
                "std": std.astype(np.float32, copy=False),
                "scope": np.asarray("per_trial_per_channel_zscore"),
            }
            return X.astype(np.float32, copy=False), stats
        return X.astype(np.float32, copy=False)

    def compute_frequency_features(
        self,
        X: np.ndarray,
        *,
        chunk_size: int = 128,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n_trials, n_channels, n_time = X.shape
        freq_length = (n_time // 2) + 1
        output = np.empty((n_trials, n_channels, freq_length), dtype=np.float32)

        for start in range(0, n_trials, max(int(chunk_size), 1)):
            stop = min(start + max(int(chunk_size), 1), n_trials)
            chunk = np.asarray(X[start:stop], dtype=np.float32)
            output[start:stop] = compute_frequency_features_numpy(chunk)
            del chunk
        # Diagnostic: ensure freq view is non-zero
        _freq_mean = float(np.abs(output).mean())
        if _freq_mean < 1e-8:
            LOGGER.warning("[FREQ-DIAG] Frequency view is all zeros! mean_abs=%.6e", _freq_mean)
        else:
            LOGGER.info("[FREQ-DIAG] Frequency view OK: mean_abs=%.4f (dB scale)", _freq_mean)
        return output

    def extract_band_signals(
        self,
        X: np.ndarray,
        sfreq: float,
        ch_names: Sequence[str],
        n_segments: int = 4,
    ) -> tuple[np.ndarray, list[str]]:
        X = np.asarray(X, dtype=np.float32)
        segment_count = max(int(n_segments), int(self.band_segments), 1)
        if self.band_method == "fft_segments":
            return compute_segmented_bandpower_numpy(
                X,
                float(sfreq),
                self.band_definitions,
                n_segments=segment_count,
            )
        nyquist = float(sfreq) / 2.0
        n_time = X.shape[-1]
        # Ensure n_time is divisible; trim to nearest multiple if needed
        seg_len = n_time // segment_count
        if seg_len < 1:
            segment_count = 1
            seg_len = n_time
        usable_len = seg_len * segment_count

        band_features: list[np.ndarray] = []
        band_names: list[str] = []
        for band_name, (low, high) in self.band_definitions.items():
            if low >= nyquist:
                continue
            high = min(float(high), nyquist - 0.01)
            if low >= high:
                continue
            filtered = self.bandpass_filter(X, sfreq, low=low, high=high)
            # Segment into n_segments temporal windows
            seg = filtered[:, :, :usable_len].reshape(
                filtered.shape[0], filtered.shape[1], segment_count, seg_len
            )
            # FIX 3: log-mean-power replaces log-variance.
            # Mean(x^2) = signal power (correct EEG energy measure).
            # log(var) under-represents steady-state oscillations where variance
            # is low but power (mu/beta ERD) is the actual discriminative feature.
            seg_power = np.log(np.mean(seg ** 2, axis=-1, dtype=np.float32) + 1e-6)
            band_features.append(seg_power.astype(np.float32, copy=False))
            # name each segment so band_channel_names remains informative
            for seg_idx in range(segment_count):
                band_names.append(f"{band_name}_s{seg_idx}")
        if not band_features:
            return np.empty((X.shape[0], X.shape[1], 0), dtype=np.float32), []
        # Concatenate along last axis: (n_trials, n_channels, n_segments * n_bands)
        X_band = np.concatenate(band_features, axis=-1).astype(np.float32, copy=False)
        # NO second normalization -- log-variance already on consistent scale
        return X_band, band_names
    def create_feature_views(
        self,
        X: np.ndarray,
        meta: dict[str, Any],
        *,
        already_preprocessed: bool = False,
    ) -> dict[str, np.ndarray]:
        sfreq = meta.get("sampling_frequency")
        if sfreq is None:
            raise ValueError("Sampling frequency is required in metadata for feature extraction.")

        X_time = np.asarray(X, dtype=np.float32) if already_preprocessed else self.preprocess(X, meta)
        ch_names = list(meta.get("channel_names", _synthesize_channel_names(int(X_time.shape[1]))))
        X_freq = self.compute_frequency_features(X_time)
        X_band, band_channel_names = self.extract_band_signals(
            X_time,
            float(sfreq),
            ch_names,
            n_segments=self.band_segments,
        )
        if X_freq.ndim != 3 or int(X_freq.shape[1]) == 0 or int(X_freq.shape[2]) == 0:
            raise ValueError("Frequency feature extraction failed to produce a valid non-empty tensor.")
        if X_band.ndim != 3 or int(X_band.shape[1]) == 0 or int(X_band.shape[2]) == 0:
            raise ValueError(
                "Band feature extraction failed to produce a valid non-empty tensor. "
                "Check the sampling frequency and requested band definitions."
            )
        for key, array in {"time": X_time, "freq": X_freq, "bands": X_band}.items():
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values detected in feature view {key!r}.")
        feature_shapes = {
            "time": tuple(int(dim) for dim in X_time.shape),
            "freq": tuple(int(dim) for dim in X_freq.shape),
            "bands": tuple(int(dim) for dim in X_band.shape),
        }
        meta.setdefault("feature_views", {})
        meta["feature_views"].update(
            {
                "band_definitions": {name: [low_hz, high_hz] for name, (low_hz, high_hz) in self.band_definitions.items()},
                "band_channel_names": band_channel_names,
                "band_method": self.band_method,
                "band_segments": int(self.band_segments),
                "shapes": feature_shapes,
            }
        )
        meta.setdefault("uncertainty_metadata", {})
        meta["uncertainty_metadata"]["feature_shapes"] = feature_shapes
        return {
            "raw": X_time,
            "time": X_time,
            "freq": X_freq,
            "bands": X_band,
        }

    def preprocess_with_views(self, X: np.ndarray, meta: dict[str, Any]) -> dict[str, np.ndarray]:
        X_time = self.preprocess(X, meta)
        return self.create_feature_views(X_time, meta, already_preprocessed=True)

    def preprocess(self, X: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
        sfreq = meta.get("sampling_frequency")
        if sfreq is None:
            raise ValueError("Sampling frequency is required in metadata for preprocessing.")

        meta["seed"] = self.seed
        meta.setdefault("uncertainty_metadata", {})
        meta["uncertainty_metadata"]["seed"] = self.seed
        self._log(meta, f"Applying bandpass filter: {self.low}-{self.high} Hz")
        X = self.bandpass_filter(X, float(sfreq), low=self.low, high=self.high)

        if bool(meta.get("is_continuous", False)):
            self._log(
                meta,
                "Detected continuous signal -> "
                f"applying sliding windows with window={self.window_sec:.2f}s overlap={self.overlap_sec:.2f}s",
            )
            X = self.epoch_data(X, float(sfreq), window_sec=self.window_sec)
            meta["is_continuous"] = False
            meta["is_epoched"] = True
        else:
            self._log(
                meta,
                "Detected epoched signal -> "
                f"re-windowing with window={self.window_sec:.2f}s overlap={self.overlap_sec:.2f}s when needed",
            )
            X = self.epoch_data(X, float(sfreq), window_sec=self.window_sec)

        self._log(meta, "Selecting EEG channels")
        X, selected_names = self.select_channels(X, meta["channel_names"], self.target_channels)
        meta["channel_names"] = selected_names
        meta["num_channels"] = len(selected_names)
        meta["uncertainty_metadata"]["selected_channels"] = selected_names

        # Common Average Reference -- removes global volume conduction
        self._log(meta, "Applying Common Average Reference (CAR)")
        X = X - X.mean(axis=1, keepdims=True)

        self._log(meta, "Applying per-trial per-channel z-score normalization")
        X, normalization_stats = self.normalize(X, return_stats=True)
        meta["normalization_scope"] = "trial_zscore"
        meta["normalization_stats"] = {
            "mean_shape": tuple(int(dim) for dim in normalization_stats["mean"].shape),
            "std_shape": tuple(int(dim) for dim in normalization_stats["std"].shape),
        }
        meta["windowing"] = {
            "window_sec": float(self.window_sec),
            "overlap_sec": float(self.overlap_sec),
            "window_samples": int(round(float(self.window_sec) * float(sfreq))),
        }
        meta["feature_config"] = {
            "band_method": self.band_method,
            "band_segments": int(self.band_segments),
            "band_definitions": {name: [low_hz, high_hz] for name, (low_hz, high_hz) in self.band_definitions.items()},
        }

        meta["data_shape"] = tuple(int(dim) for dim in X.shape)
        meta["num_trials"] = int(X.shape[0])
        meta["dtype"] = str(X.dtype)
        return X


def validate_data(
    X: np.ndarray,
    y: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    report: dict[str, Any] = {
        "replaced_non_finite": False,
        "dropped_empty_trials": 0,
        "trimmed_for_label_mismatch": False,
        "class_imbalance_ratio": None,
    }

    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[np.newaxis, :, :]
        report["added_trial_axis"] = True
    if X.ndim != 3:
        raise ValueError(f"Expected EEG data with shape (trials, channels, time), got {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
        raise ValueError(f"Empty EEG tensor detected with shape {X.shape}.")

    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        report["replaced_non_finite"] = True

    trial_std = X.std(axis=(1, 2))
    valid_trials = trial_std > 1e-8
    if not valid_trials.all():
        dropped = int((~valid_trials).sum())
        X = X[valid_trials]
        report["dropped_empty_trials"] = dropped
        if y is not None:
            y = np.asarray(y).reshape(-1)[valid_trials[: np.asarray(y).reshape(-1).size]]

    if y is not None:
        y = np.asarray(y).reshape(-1)
        if y.size != X.shape[0]:
            min_size = min(y.size, X.shape[0])
            X = X[:min_size]
            y = y[:min_size]
            report["trimmed_for_label_mismatch"] = True
        if y.size > 0:
            counts = Counter(map(str, y.tolist()))
            if counts:
                smallest = min(counts.values())
                largest = max(counts.values())
                report["class_imbalance_ratio"] = float(largest / max(smallest, 1))
    return X.astype(np.float32, copy=False), y, report


class EEGDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = None if y is None else np.asarray(y)
        if self.X.ndim != 3:
            raise ValueError(f"Expected X with shape (trials, channels, time), got {self.X.shape}.")
        if self.y is not None and self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"Label count {self.y.shape[0]} does not match number of trials {self.X.shape[0]}."
            )

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        torch_module = _get_torch()
        sample = torch_module.from_numpy(self.X[index])
        if self.y is None:
            return sample, None
        label = self.y[index]
        if np.issubdtype(np.asarray(label).dtype, np.integer):
            return sample, torch_module.tensor(int(label), dtype=torch_module.long)
        return sample, torch_module.tensor(label)


def _coerce_label_mapping(meta: Mapping[str, Any]) -> dict[int, str]:
    mapping = meta.get("label_mapping", {})
    if not isinstance(mapping, Mapping):
        return {}
    coerced: dict[int, str] = {}
    for key, value in mapping.items():
        try:
            coerced[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return coerced


def _remap_labels_to_global(
    y: np.ndarray,
    meta: Mapping[str, Any],
    global_label_to_index: Mapping[str, int],
) -> np.ndarray:
    y_array = np.asarray(y).reshape(-1)
    local_mapping = _coerce_label_mapping(meta)
    if not local_mapping:
        if np.issubdtype(y_array.dtype, np.integer):
            return y_array.astype(np.int64, copy=False)
        raise ValueError("Missing label mapping metadata for global label remapping.")

    remapped = np.empty(y_array.shape[0], dtype=np.int64)
    for index, local_label in enumerate(y_array.tolist()):
        try:
            label_name = local_mapping[int(local_label)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not resolve local label {local_label!r} using metadata mapping.") from exc
        if label_name not in global_label_to_index:
            raise ValueError(f"Unknown label name {label_name!r}; expected one of {sorted(global_label_to_index)}.")
        remapped[index] = int(global_label_to_index[label_name])
    return remapped


def _align_trials_to_target(
    X: np.ndarray,
    channel_names: Sequence[str],
    target_channels: Sequence[str],
    target_time: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected epoched EEG array with shape (trials, channels, time), got {X.shape}.")
    if len(channel_names) != X.shape[1]:
        raise ValueError(
            f"Channel-name mismatch during alignment: len(channel_names)={len(channel_names)} vs X.shape[1]={X.shape[1]}."
        )
    if target_time <= 0:
        raise ValueError(f"target_time must be positive, got {target_time}.")

    current_normalized = [_normalize_channel_name(name) for name in channel_names]
    target_normalized = [_normalize_channel_name(name) for name in target_channels]
    if X.shape[1] == len(target_channels) and X.shape[2] == target_time and current_normalized == target_normalized:
        return X.astype(np.float32, copy=False)

    aligned = np.zeros((X.shape[0], len(target_channels), target_time), dtype=np.float32)
    current_lookup = {normalized: index for index, normalized in enumerate(current_normalized)}
    copy_time = min(X.shape[2], target_time)
    for target_index, normalized_name in enumerate(target_normalized):
        source_index = current_lookup.get(normalized_name)
        if source_index is None:
            continue
        aligned[:, target_index, :copy_time] = X[:, source_index, :copy_time]
    return aligned


def _resolve_collection_root(root_dir: Path, preferred_child: str | None = None) -> Path:
    resolved = root_dir.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
    if preferred_child is not None and (resolved / preferred_child).is_dir():
        return resolved / preferred_child
    return resolved


def _infer_dataset_from_root(root_dir: Path) -> str | None:
    resolved = root_dir.expanduser()
    if not resolved.exists() or not resolved.is_dir():
        return None

    physio_root = resolved / "PhysioNet_EEG"
    if physio_root.is_dir():
        return "physionet"
    if any(path.is_dir() and re.fullmatch(r"S\d{3}", path.name) for path in resolved.iterdir()):
        return "physionet"

    bci_root = resolved / "BCICIV_2a_gdf"
    if bci_root.is_dir():
        return "bci"
    if any(
        path.is_file() and re.fullmatch(r"A\d{2}[TE]\.gdf", path.name, flags=re.IGNORECASE)
        for path in resolved.iterdir()
    ):
        return "bci"
    return None


def _physionet_subject_files(
    root_dir: Path,
    subject_id: int,
    task_filter: str = "all",
) -> list[Path]:
    """Get .edf files for a subject, optionally filtered by run type."""
    subject_dir = root_dir / f"S{subject_id:03d}"
    if not subject_dir.is_dir():
        return []
    all_files = sorted(path for path in subject_dir.glob("*R*.edf") if path.is_file())
    if not all_files:
        all_files = sorted(path for path in subject_dir.glob("*.edf") if path.is_file())
    if not all_files:
        return []

    # Apply task-filter: only include runs that match the requested task type
    allowed_runs = PHYSIONET_TASK_RUNS.get(task_filter)
    if allowed_runs is None:
        LOGGER.warning("Unknown task_filter %r, using all runs.", task_filter)
        return all_files
    if task_filter == "all":
        return all_files

    filtered = []
    for fpath in all_files:
        # Extract run number from filename like S001R04.edf -> 4
        match = re.search(r"R(\d{2})", fpath.name, re.IGNORECASE)
        if match:
            run_num = int(match.group(1))
            if run_num in allowed_runs:
                filtered.append(fpath)
    return filtered


def _bci_subject_files(root_dir: Path, subject_id: int) -> list[Path]:
    ordered_files: list[Path] = []
    for suffix in ("T", "E"):
        candidate = root_dir / f"A{subject_id:02d}{suffix}.gdf"
        if candidate.is_file():
            ordered_files.append(candidate)
    if ordered_files:
        return ordered_files
    return sorted(path for path in root_dir.glob(f"A{subject_id:02d}*.gdf") if path.is_file())


def _class_distribution_from_labels(y: np.ndarray) -> dict[int, int]:
    labels = np.asarray(y).reshape(-1)
    counts = Counter(int(label) for label in labels.tolist())
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _euclidean_align_trials(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply subject-level Euclidean Alignment: X' = R^(-1/2) X."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Euclidean Alignment expects (N, C, T), got {X.shape}.")
    n_trials, n_channels, n_time = X.shape
    if n_trials == 0 or n_channels == 0 or n_time == 0:
        return X

    covariances = np.matmul(X, np.transpose(X, (0, 2, 1))) / max(n_time, 1)
    reference_cov = np.mean(covariances, axis=0, dtype=np.float64)
    reference_cov = reference_cov + (float(eps) * np.eye(n_channels, dtype=np.float64))
    eigenvalues, eigenvectors = np.linalg.eigh(reference_cov)
    eigenvalues = np.clip(eigenvalues, float(eps), None)
    inv_sqrt = (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
    aligned = np.einsum("ab,nbt->nat", inv_sqrt.astype(np.float32, copy=False), X)
    return np.ascontiguousarray(aligned, dtype=np.float32)


def _stream_write_array(path: Path, array: np.ndarray | Sequence[Any], *, dtype: Any) -> tuple[int, ...]:
    materialized = np.asarray(array, dtype=dtype)
    mmap = open_memmap(path, mode="w+", dtype=dtype, shape=materialized.shape)
    mmap[...] = materialized
    mmap.flush()
    shape = tuple(int(dim) for dim in materialized.shape)
    del mmap
    return shape


def _subject_artifact_paths(cache_dir: Path, subject_id: int) -> dict[str, Path]:
    prefix = cache_dir / f"subject_{subject_id:03d}"
    return {
        "time": prefix.with_name(f"{prefix.name}_time.npy"),
        "freq": prefix.with_name(f"{prefix.name}_freq.npy"),
        "bands": prefix.with_name(f"{prefix.name}_bands.npy"),
        "y": prefix.with_name(f"{prefix.name}_y.npy"),
    }


def _prepare_subject_artifact(
    *,
    subject_id: int,
    files: Sequence[Path],
    dataset_name: str,
    preprocessor: EEGPreprocessor,
    global_label_to_index: Mapping[str, int],
    target_spec: dict[str, Any],
    subject_cache_dir: Path,
    fixed_target_sec: float | None,
) -> dict[str, Any] | None:
    subject_X_parts: list[np.ndarray] = []
    subject_y_parts: list[np.ndarray] = []
    valid_file_count = 0

    _log_memory(f"[STREAM] Subject {subject_id:03d}: starting ({len(files)} files)")
    for file_index, file_path in enumerate(files, start=1):
        try:
            X_file, y_file, meta = load_eeg_data(str(file_path))
            if y_file is None or np.asarray(y_file).size == 0:
                LOGGER.warning("Skipping unlabeled file %s.", file_path)
                continue

            X_file = preprocessor.preprocess(X_file, meta)
            X_file, y_file, validation_report = validate_data(X_file, y_file)
            if y_file is None or int(X_file.shape[0]) == 0:
                LOGGER.warning("Skipping empty or invalid file %s after validation.", file_path)
                continue

            y_file = _remap_labels_to_global(y_file, meta, global_label_to_index)
            current_channels = list(meta.get("channel_names", []))
            if not current_channels:
                current_channels = _synthesize_channel_names(int(X_file.shape[1]))

            if target_spec.get("channels") is None:
                target_spec["channels"] = current_channels
                if fixed_target_sec is not None and meta.get("sampling_frequency") is not None:
                    target_spec["time"] = int(
                        round(float(fixed_target_sec) * float(meta["sampling_frequency"]))
                    )
                else:
                    target_spec["time"] = int(X_file.shape[2])
                target_spec["sampling_frequency"] = float(meta.get("sampling_frequency") or 0.0)
                _log_memory(
                    "[STREAM] Alignment target initialized "
                    f"from subject {subject_id:03d} file {file_path.name}: "
                    f"channels={len(target_spec['channels'])}, time={target_spec['time']}"
                )

            aligned_X = _align_trials_to_target(
                X_file,
                current_channels,
                target_spec["channels"],
                int(target_spec["time"]),
            )
            subject_X_parts.append(np.asarray(aligned_X, dtype=np.float32))
            subject_y_parts.append(np.asarray(y_file, dtype=np.int64))
            valid_file_count += 1

            if validation_report.get("trimmed_for_label_mismatch"):
                LOGGER.warning("Label mismatch corrected while validating %s.", file_path)
            _log_memory(
                f"[STREAM] Subject {subject_id:03d}: processed file {file_index}/{len(files)} "
                f"-> {file_path.name}, trials={int(aligned_X.shape[0])}"
            )
            del X_file, y_file, aligned_X
            gc.collect()
        except Exception as exc:
            LOGGER.warning("Skipping %s due to error: %s", file_path, exc)

    if not subject_X_parts:
        LOGGER.warning("%s subject %s: no valid labeled files were processed.", dataset_name, subject_id)
        return None

    subject_time = np.ascontiguousarray(np.concatenate(subject_X_parts, axis=0), dtype=np.float32)
    subject_y = np.ascontiguousarray(np.concatenate(subject_y_parts, axis=0), dtype=np.int64)
    subject_time = _euclidean_align_trials(subject_time)
    del subject_X_parts, subject_y_parts
    gc.collect()

    subject_meta = {
        "sampling_frequency": float(target_spec["sampling_frequency"]),
        "channel_names": list(target_spec["channels"]),
        "num_channels": int(len(target_spec["channels"])),
        "is_continuous": False,
        "is_epoched": True,
        "preprocessing_log": [],
        "uncertainty_metadata": {},
    }
    _log_memory(
        f"[STREAM] Subject {subject_id:03d}: creating feature views for "
        f"{int(subject_time.shape[0])} trials"
    )
    feature_views = preprocessor.create_feature_views(subject_time, subject_meta, already_preprocessed=True)
    artifact_paths = _subject_artifact_paths(subject_cache_dir, subject_id)
    time_shape = _stream_write_array(artifact_paths["time"], feature_views["time"], dtype=np.float32)
    freq_shape = _stream_write_array(artifact_paths["freq"], feature_views["freq"], dtype=np.float32)
    bands_shape = _stream_write_array(artifact_paths["bands"], feature_views["bands"], dtype=np.float32)
    _stream_write_array(artifact_paths["y"], subject_y, dtype=np.int64)

    class_distribution = _class_distribution_from_labels(subject_y)
    artifact = {
        "subject_id": int(subject_id),
        "file_count": int(valid_file_count),
        "n_samples": int(subject_time.shape[0]),
        "time_shape": time_shape,
        "freq_shape": freq_shape,
        "bands_shape": bands_shape,
        "sampling_frequency": float(target_spec["sampling_frequency"]),
        "class_distribution": class_distribution,
        "time_path": str(artifact_paths["time"].resolve()),
        "freq_path": str(artifact_paths["freq"].resolve()),
        "bands_path": str(artifact_paths["bands"].resolve()),
        "y_path": str(artifact_paths["y"].resolve()),
    }

    del subject_time, subject_y, feature_views
    gc.collect()
    _log_memory(
        f"[STREAM] Subject {subject_id:03d}: saved to disk "
        f"(time={time_shape}, freq={freq_shape}, bands={bands_shape})"
    )
    return artifact


def _subject_artifact_view_shapes(
    subject_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    if not subject_artifacts:
        raise RuntimeError("No subject artifacts were provided.")

    first_artifact = dict(subject_artifacts[0])
    view_shapes: dict[str, tuple[int, int]] = {}
    for artifact in subject_artifacts:
        for key in REQUIRED_FEATURE_VIEW_KEYS:
            path_key = f"{key}_path"
            if key in view_shapes or path_key not in artifact:
                continue
            source = np.load(artifact[path_key], mmap_mode="r", allow_pickle=False)
            if source.ndim != 3:
                raise ValueError(f"Invalid source shape for {key}: {tuple(source.shape)}.")
            view_shapes[key] = tuple(int(dim) for dim in source.shape[1:])
            del source
    if "time" not in view_shapes:
        raise RuntimeError("Streaming merge requires a time view shape.")
    return view_shapes, first_artifact


def _merge_all_subject_artifacts(
    *,
    subject_artifacts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dataset: str,
    root_dir: Path,
    preprocessing_config: Mapping[str, Any],
) -> dict[str, str]:
    if not subject_artifacts:
        raise RuntimeError(f"No subject artifacts were generated for dataset {dataset}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    view_shapes, first_artifact = _subject_artifact_view_shapes(subject_artifacts)
    total_samples = sum(int(artifact["n_samples"]) for artifact in subject_artifacts)
    if total_samples <= 0:
        raise RuntimeError(f"No samples were available for canonical merge in dataset {dataset}.")

    all_paths: dict[str, Path] = {
        "time": output_dir / "all_time_X.npy",
        "y": output_dir / "all_y.npy",
        "subjects": output_dir / "all_subjects.npy",
    }
    for key in view_shapes:
        if key == "time":
            continue
        all_paths[key] = output_dir / f"all_{key}_X.npy"

    memmaps: dict[str, np.memmap] = {}
    for key, shape in view_shapes.items():
        memmaps[key] = open_memmap(
            all_paths["time" if key == "time" else key],
            mode="w+",
            dtype=np.float32,
            shape=(total_samples,) + tuple(shape),
        )
    memmaps["y"] = open_memmap(
        all_paths["y"],
        mode="w+",
        dtype=np.int64,
        shape=(total_samples,),
    )
    memmaps["subjects"] = open_memmap(
        all_paths["subjects"],
        mode="w+",
        dtype=np.int32,
        shape=(total_samples,),
    )

    offset = 0
    used_subjects: list[int] = []
    class_distribution = Counter()
    subject_sample_counts: dict[int, int] = {}

    try:
        for artifact_index, artifact in enumerate(subject_artifacts, start=1):
            subject_id = int(artifact["subject_id"])
            start = int(offset)
            stop = start + int(artifact["n_samples"])
            _log_memory(
                f"[STREAM] Merge {artifact_index}/{len(subject_artifacts)}: "
                f"subject {subject_id:03d} -> all offset {start}:{stop}"
            )
            for key in view_shapes:
                path_key = f"{key}_path"
                source = np.load(artifact[path_key], mmap_mode="r", allow_pickle=False)
                dest = memmaps[key]
                if int(source.shape[0]) != int(stop - start):
                    raise ValueError(
                        f"Sample-count mismatch for {key}: {tuple(source.shape)} vs expected {stop - start}."
                    )
                if dest.shape[1:] != source.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for {key}: {tuple(source.shape)} vs {tuple(dest.shape)}."
                    )
                dest[start:stop] = source
                del source

            y_source = np.load(artifact["y_path"], mmap_mode="r", allow_pickle=False)
            memmaps["y"][start:stop] = y_source
            memmaps["subjects"][start:stop] = np.int32(subject_id)
            class_distribution.update(int(label) for label in np.asarray(y_source).reshape(-1).tolist())
            subject_sample_counts[subject_id] = int(stop - start)
            used_subjects.append(subject_id)
            offset = stop
            del y_source

            for suffix in ("time_path", "freq_path", "bands_path", "y_path"):
                artifact_path = Path(str(artifact.get(suffix, "")))
                if artifact_path.exists():
                    artifact_path.unlink()
            gc.collect()
    finally:
        for array in memmaps.values():
            array.flush()
            del array
        gc.collect()

    metadata = {
        "dataset": dataset,
        "root": str(root_dir.resolve()),
        "sampling_frequency_inferred": float(first_artifact["sampling_frequency"]),
        "preprocessing": dict(preprocessing_config),
        "available_views": list(view_shapes.keys()),
        "save_mode": "all",
        "streaming": True,
        "used_subjects": sorted(used_subjects),
        "subject_sample_counts": {str(key): int(value) for key, value in sorted(subject_sample_counts.items())},
        "class_distribution": {
            "all": dict(sorted(class_distribution.items())),
        },
        "shapes": {},
    }
    metadata["shapes"]["all_X"] = (total_samples,) + tuple(view_shapes["time"])
    metadata["shapes"]["all_time_X"] = metadata["shapes"]["all_X"]
    for key, shape in view_shapes.items():
        if key == "time":
            continue
        metadata["shapes"][f"all_{key}_X"] = (total_samples,) + tuple(shape)

    meta_path = output_dir / "prepared_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(metadata), handle, indent=2)

    saved_paths = {
        "all_time_X": str(all_paths["time"].resolve()),
        "all_y": str(all_paths["y"].resolve()),
        "all_subjects": str(all_paths["subjects"].resolve()),
        "meta": str(meta_path.resolve()),
    }
    if "freq" in all_paths:
        saved_paths["all_freq_X"] = str(all_paths["freq"].resolve())
    if "bands" in all_paths:
        saved_paths["all_bands_X"] = str(all_paths["bands"].resolve())

    required_keys = ["all_time_X", "all_y", "all_subjects", "meta"]
    if "freq" in view_shapes:
        required_keys.append("all_freq_X")
    if "bands" in view_shapes:
        required_keys.append("all_bands_X")
    _verify_saved_outputs(saved_paths, required_keys=required_keys)
    return saved_paths


def _merge_subject_artifacts(
    *,
    subject_artifacts: Sequence[Mapping[str, Any]],
    output_dir: Path,
    train_subjects: Sequence[int],
    val_subjects: Sequence[int],
    dataset: str,
    root_dir: Path,
    preprocessing_config: Mapping[str, Any],
) -> dict[str, str]:
    if not subject_artifacts:
        raise RuntimeError(f"No subject artifacts were generated for dataset {dataset}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_subject_set = {int(subject_id) for subject_id in train_subjects}
    val_subject_set = {int(subject_id) for subject_id in val_subjects}
    view_shapes, first_artifact = _subject_artifact_view_shapes(subject_artifacts)

    split_counts = {
        "train": sum(
            int(artifact["n_samples"])
            for artifact in subject_artifacts
            if int(artifact["subject_id"]) in train_subject_set
        ),
        "val": sum(
            int(artifact["n_samples"])
            for artifact in subject_artifacts
            if int(artifact["subject_id"]) in val_subject_set
        ),
    }
    if split_counts["train"] == 0 or split_counts["val"] == 0:
        raise RuntimeError(
            f"Streaming merge requires both train and val samples, got counts={split_counts}."
        )

    split_paths: dict[str, dict[str, Path]] = {
        "train": {
            "time": output_dir / "train_time_X.npy",
            "y": output_dir / "train_y.npy",
            "subjects": output_dir / "train_subjects.npy",
        },
        "val": {
            "time": output_dir / "val_time_X.npy",
            "y": output_dir / "val_y.npy",
            "subjects": output_dir / "val_subjects.npy",
        },
    }
    for split_name in ("train", "val"):
        if "freq" in view_shapes:
            split_paths[split_name]["freq"] = output_dir / f"{split_name}_freq_X.npy"
        if "bands" in view_shapes:
            split_paths[split_name]["bands"] = output_dir / f"{split_name}_bands_X.npy"

    split_memmaps: dict[str, dict[str, np.memmap]] = {"train": {}, "val": {}}
    for split_name in ("train", "val"):
        for key, shape in view_shapes.items():
            per_view_shape = (split_counts[split_name],) + tuple(int(dim) for dim in shape)
            split_memmaps[split_name][key] = open_memmap(
                split_paths[split_name][key],
                mode="w+",
                dtype=np.float32,
                shape=per_view_shape,
            )
        split_memmaps[split_name]["y"] = open_memmap(
            split_paths[split_name]["y"],
            mode="w+",
            dtype=np.int64,
            shape=(split_counts[split_name],),
        )
        split_memmaps[split_name]["subjects"] = open_memmap(
            split_paths[split_name]["subjects"],
            mode="w+",
            dtype=np.int32,
            shape=(split_counts[split_name],),
        )

    split_offsets = {"train": 0, "val": 0}
    class_distribution = {"train": Counter(), "val": Counter()}
    used_subjects = {"train": [], "val": []}

    try:
        for artifact_index, artifact in enumerate(subject_artifacts, start=1):
            subject_id = int(artifact["subject_id"])
            if subject_id in train_subject_set:
                split_name = "train"
            elif subject_id in val_subject_set:
                split_name = "val"
            else:
                continue

            start = int(split_offsets[split_name])
            stop = start + int(artifact["n_samples"])
            _log_memory(
                f"[STREAM] Merge {artifact_index}/{len(subject_artifacts)}: "
                f"subject {subject_id:03d} -> {split_name} offset {start}:{stop}"
            )

            for key in view_shapes:
                path_key = f"{key}_path"
                if key not in split_memmaps[split_name] or path_key not in artifact:
                    continue
                source = np.load(artifact[path_key], mmap_mode="r", allow_pickle=False)
                dest = split_memmaps[split_name][key]
                print(f"[MERGE] split={split_name} view={key} shape={tuple(int(dim) for dim in dest.shape)}")
                if int(source.shape[0]) != int(stop - start):
                    raise ValueError(
                        f"Sample-count mismatch for {key}: {tuple(source.shape)} vs expected {stop - start}."
                    )
                if dest.shape[1:] != source.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for {key}: {tuple(source.shape)} vs {tuple(dest.shape)}"
                    )
                dest[start:stop] = source
                del source
            y_source = np.load(artifact["y_path"], mmap_mode="r", allow_pickle=False)
            split_memmaps[split_name]["y"][start:stop] = y_source
            split_memmaps[split_name]["subjects"][start:stop] = np.int32(subject_id)

            split_offsets[split_name] = stop
            class_distribution[split_name].update(
                int(label) for label in np.asarray(y_source).reshape(-1).tolist()
            )
            used_subjects[split_name].append(subject_id)
            del y_source

            for suffix in ("time_path", "freq_path", "bands_path", "y_path"):
                if suffix not in artifact:
                    continue
                artifact_path = Path(str(artifact[suffix]))
                if artifact_path.exists():
                    artifact_path.unlink()
            gc.collect()
    finally:
        for split_name in ("train", "val"):
            for array in split_memmaps[split_name].values():
                array.flush()
                del array
        gc.collect()

    metadata = {
        "dataset": dataset,
        "root": str(root_dir.resolve()),
        "train_subjects": sorted(train_subject_set),
        "val_subjects": sorted(val_subject_set),
        "sampling_frequency_inferred": float(first_artifact["sampling_frequency"]),
        "preprocessing": dict(preprocessing_config),
        "available_views": list(view_shapes.keys()),
        "shapes": {},
        "class_distribution": {
            "train": dict(sorted(class_distribution["train"].items())),
            "val": dict(sorted(class_distribution["val"].items())),
        },
        "used_subjects": {
            "train": sorted(used_subjects["train"]),
            "val": sorted(used_subjects["val"]),
        },
        "save_mode": "holdout",
        "streaming": True,
    }
    metadata["shapes"]["train_X"] = (split_counts["train"],) + tuple(view_shapes["time"])
    metadata["shapes"]["val_X"] = (split_counts["val"],) + tuple(view_shapes["time"])
    metadata["shapes"]["train_time_X"] = metadata["shapes"]["train_X"]
    metadata["shapes"]["val_time_X"] = metadata["shapes"]["val_X"]
    for key, shape in view_shapes.items():
        if key == "time":
            continue
        metadata["shapes"][f"train_{key}_X"] = (split_counts["train"],) + tuple(shape)
        metadata["shapes"][f"val_{key}_X"] = (split_counts["val"],) + tuple(shape)
    meta_path = output_dir / "prepared_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(metadata), handle, indent=2)

    saved_paths = {
        "train_time_X": str(split_paths["train"]["time"].resolve()),
        "train_y": str(split_paths["train"]["y"].resolve()),
        "train_subjects": str(split_paths["train"]["subjects"].resolve()),
        "val_time_X": str(split_paths["val"]["time"].resolve()),
        "val_y": str(split_paths["val"]["y"].resolve()),
        "val_subjects": str(split_paths["val"]["subjects"].resolve()),
        "meta": str(meta_path.resolve()),
    }
    if "freq" in split_paths["train"]:
        saved_paths["train_freq_X"] = str(split_paths["train"]["freq"].resolve())
        saved_paths["val_freq_X"] = str(split_paths["val"]["freq"].resolve())
    if "bands" in split_paths["train"]:
        saved_paths["train_bands_X"] = str(split_paths["train"]["bands"].resolve())
        saved_paths["val_bands_X"] = str(split_paths["val"]["bands"].resolve())
    required_keys = [
        "train_time_X",
        "train_y",
        "train_subjects",
        "val_time_X",
        "val_y",
        "val_subjects",
        "meta",
    ]
    if "freq" in view_shapes:
        required_keys.extend(["train_freq_X", "val_freq_X"])
    if "bands" in view_shapes:
        required_keys.extend(["train_bands_X", "val_bands_X"])
    _verify_saved_outputs(
        saved_paths,
        required_keys=required_keys,
    )
    return saved_paths


def _load_subject_collection(
    *,
    root_dir: Path,
    dataset_name: str,
    subject_ids: Sequence[int],
    file_resolver: Any,
    global_label_to_index: Mapping[str, int],
    low: float = 7.0,
    high: float = 30.0,
    window_sec: float = 2.0,
    seed: int = 42,
    fixed_target_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = EEGPreprocessor(
        low=low,
        high=high,
        window_sec=window_sec,
        seed=seed,
        verbose=False,
    )
    target_channels: list[str] | None = None
    target_time: int | None = None
    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    subject_chunks: list[np.ndarray] = []
    used_subjects: list[int] = []

    for subject_id in subject_ids:
        files = file_resolver(root_dir, subject_id)
        if not files:
            LOGGER.warning("%s subject %s: no files found.", dataset_name, subject_id)
            continue

        subject_X_parts: list[np.ndarray] = []
        subject_y_parts: list[np.ndarray] = []
        valid_file_count = 0
        for file_path in files:
            try:
                X_file, y_file, meta = load_eeg_data(str(file_path))
                if y_file is None or np.asarray(y_file).size == 0:
                    LOGGER.warning("Skipping unlabeled file %s.", file_path)
                    continue

                X_file = preprocessor.preprocess(X_file, meta)
                X_file, y_file, validation_report = validate_data(X_file, y_file)
                if y_file is None or X_file.shape[0] == 0:
                    LOGGER.warning("Skipping empty or invalid file %s after validation.", file_path)
                    continue

                y_file = _remap_labels_to_global(y_file, meta, global_label_to_index)
                current_channels = list(meta.get("channel_names", []))
                if not current_channels:
                    current_channels = _synthesize_channel_names(int(X_file.shape[1]))

                if target_channels is None:
                    target_channels = current_channels
                    if fixed_target_sec is not None and meta.get("sampling_frequency") is not None:
                        target_time = int(round(float(fixed_target_sec) * float(meta["sampling_frequency"])))
                    else:
                        target_time = int(X_file.shape[2])
                    LOGGER.info(
                        "%s alignment target initialized from %s: channels=%d, time=%d",
                        dataset_name,
                        file_path.name,
                        len(target_channels),
                        target_time,
                    )

                assert target_channels is not None
                assert target_time is not None
                aligned_X = _align_trials_to_target(X_file, current_channels, target_channels, target_time)
                subject_X_parts.append(aligned_X.astype(np.float32, copy=False))
                subject_y_parts.append(np.asarray(y_file, dtype=np.int64))
                valid_file_count += 1

                if validation_report.get("trimmed_for_label_mismatch"):
                    LOGGER.warning("Label mismatch corrected while validating %s.", file_path)
            except Exception as exc:
                LOGGER.warning("Skipping %s due to error: %s", file_path, exc)

        if not subject_X_parts:
            LOGGER.warning("%s subject %s: no valid labeled files were processed.", dataset_name, subject_id)
            continue

        subject_X = np.concatenate(subject_X_parts, axis=0).astype(np.float32, copy=False)
        subject_y = np.concatenate(subject_y_parts, axis=0).astype(np.int64, copy=False)
        subject_ids_array = np.full(subject_y.shape[0], subject_id, dtype=np.int16)

        X_chunks.append(subject_X)
        y_chunks.append(subject_y)
        subject_chunks.append(subject_ids_array)
        used_subjects.append(subject_id)

        LOGGER.info(
            "%s subject %s: files=%d, trials=%d, shape=%s, class_distribution=%s",
            dataset_name,
            subject_id,
            valid_file_count,
            int(subject_X.shape[0]),
            tuple(int(dim) for dim in subject_X.shape),
            _class_distribution_from_labels(subject_y),
        )
        subject_X = _euclidean_align_trials(subject_X)

    if not X_chunks:
        raise RuntimeError(f"No valid labeled samples were found for dataset {dataset_name}.")

    X_all = np.concatenate(X_chunks, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(y_chunks, axis=0).astype(np.int64, copy=False)
    subjects_all = np.concatenate(subject_chunks, axis=0).astype(np.int16, copy=False)

    LOGGER.info(
        "%s summary: total_samples=%d, subjects=%s, class_distribution=%s, final_shape=%s",
        dataset_name,
        int(X_all.shape[0]),
        used_subjects,
        _class_distribution_from_labels(y_all),
        tuple(int(dim) for dim in X_all.shape),
    )
    return X_all, y_all, subjects_all


def load_physionet_subjects(root_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_root = _resolve_collection_root(root_dir, preferred_child="PhysioNet_EEG")
    return _load_subject_collection(
        root_dir=resolved_root,
        dataset_name="physionet",
        subject_ids=list(range(1, 110)),
        file_resolver=_physionet_subject_files,
        global_label_to_index=PHYSIONET_GLOBAL_LABEL_TO_INDEX,
        fixed_target_sec=2.0,
    )


def load_bci_subjects(root_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_root = _resolve_collection_root(root_dir, preferred_child="BCICIV_2a_gdf")
    return _load_subject_collection(
        root_dir=resolved_root,
        dataset_name="bci",
        subject_ids=list(range(1, 10)),
        file_resolver=_bci_subject_files,
        global_label_to_index=BCI_IV_2A_GLOBAL_LABEL_TO_INDEX,
    )


def subject_wise_split(
    X: np.ndarray,
    y: np.ndarray | None,
    subjects: np.ndarray,
    train_subjects: Sequence[int],
    test_subjects: Sequence[int],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    X = np.asarray(X, dtype=np.float32)
    subjects_array = np.asarray(subjects).reshape(-1)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, time), got {X.shape}.")
    if subjects_array.shape[0] != X.shape[0]:
        raise ValueError(
            f"Subject count {subjects_array.shape[0]} does not match sample count {X.shape[0]}."
        )

    y_array = None if y is None else np.asarray(y).reshape(-1)
    if y_array is not None and y_array.shape[0] != X.shape[0]:
        raise ValueError(f"Label count {y_array.shape[0]} does not match sample count {X.shape[0]}.")

    train_mask = np.isin(subjects_array, np.asarray(list(train_subjects)))
    test_mask = np.isin(subjects_array, np.asarray(list(test_subjects)))
    if not np.any(train_mask):
        raise ValueError("No samples matched the requested training subjects.")
    if not np.any(test_mask):
        raise ValueError("No samples matched the requested test subjects.")

    X_train = X[train_mask].astype(np.float32, copy=False)
    X_test = X[test_mask].astype(np.float32, copy=False)
    y_train = None if y_array is None else y_array[train_mask].astype(np.int64, copy=False)
    y_test = None if y_array is None else y_array[test_mask].astype(np.int64, copy=False)
    return X_train, y_train, X_test, y_test


def subject_split(
    subjects: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices by subject ID so no subject appears in both sets.

    Returns (train_indices, val_indices) as arrays of sample-level indices.
    This is the function imported by phase2.py for subject-wise data splitting.
    """
    subjects_array = np.asarray(subjects).reshape(-1)
    unique_subjects = np.unique(subjects_array)
    n_unique = len(unique_subjects)

    if n_unique <= 1:
        # Cannot split by subject with only one subject
        return np.arange(len(subjects_array), dtype=np.int64), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_subjects)
    n_val = max(1, int(np.floor(n_unique * val_ratio)))
    val_subjects = set(shuffled[:n_val].tolist())
    train_subjects = set(shuffled[n_val:].tolist())

    train_mask = np.isin(subjects_array, list(train_subjects))
    val_mask = np.isin(subjects_array, list(val_subjects))

    train_idx = np.where(train_mask)[0].astype(np.int64)
    val_idx = np.where(val_mask)[0].astype(np.int64)
    return train_idx, val_idx


def _split_single_recording_indices(
    n_samples: int,
    y: np.ndarray | None,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 1:
        return np.arange(n_samples, dtype=np.int64), np.empty(0, dtype=np.int64)

    ratio = float(np.clip(val_ratio, 0.0, 0.95))
    target_val = int(round(n_samples * ratio))
    target_val = min(max(target_val, 1), n_samples - 1)
    rng = np.random.default_rng(seed)

    if y is None:
        permutation = rng.permutation(n_samples)
        val_idx = np.sort(permutation[:target_val]).astype(np.int64, copy=False)
        train_idx = np.sort(permutation[target_val:]).astype(np.int64, copy=False)
        return train_idx, val_idx

    y_array = np.asarray(y).reshape(-1)
    if y_array.shape[0] != n_samples:
        raise ValueError(f"Label count {y_array.shape[0]} does not match sample count {n_samples}.")

    val_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for label in sorted(np.unique(y_array).tolist()):
        label_indices = np.flatnonzero(y_array == label)
        label_indices = label_indices[rng.permutation(label_indices.size)]
        if label_indices.size <= 1:
            train_parts.append(label_indices.astype(np.int64, copy=False))
            continue
        label_val = int(round(label_indices.size * ratio))
        label_val = min(max(label_val, 1), label_indices.size - 1)
        val_parts.append(label_indices[:label_val].astype(np.int64, copy=False))
        train_parts.append(label_indices[label_val:].astype(np.int64, copy=False))

    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)

    if val_idx.size == 0 or train_idx.size == 0:
        permutation = rng.permutation(n_samples)
        val_idx = np.sort(permutation[:target_val]).astype(np.int64, copy=False)
        train_idx = np.sort(permutation[target_val:]).astype(np.int64, copy=False)
        return train_idx, val_idx

    train_idx = np.sort(train_idx[rng.permutation(train_idx.size)]).astype(np.int64, copy=False)
    val_idx = np.sort(val_idx[rng.permutation(val_idx.size)]).astype(np.int64, copy=False)
    return train_idx, val_idx


def _split_feature_views_for_save(
    feature_views: Mapping[str, np.ndarray],
    y: np.ndarray | None,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, dict[str, np.ndarray], np.ndarray | None]:
    time_view = np.asarray(feature_views["time"], dtype=np.float32)
    train_idx, val_idx = _split_single_recording_indices(
        int(time_view.shape[0]),
        y,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_views = {
        str(name): np.ascontiguousarray(np.asarray(array, dtype=np.float32)[train_idx])
        for name, array in feature_views.items()
        if str(name) in REQUIRED_FEATURE_VIEW_KEYS
    }
    val_views = {
        str(name): np.ascontiguousarray(np.asarray(array, dtype=np.float32)[val_idx])
        for name, array in feature_views.items()
        if str(name) in REQUIRED_FEATURE_VIEW_KEYS
    }
    y_array = None if y is None else np.asarray(y, dtype=np.int64).reshape(-1)
    y_train = None if y_array is None else np.ascontiguousarray(y_array[train_idx])
    y_val = None if y_array is None else np.ascontiguousarray(y_array[val_idx])
    return train_views, y_train, val_views, y_val


def save_prepared_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray | None,
    output_dir: Path,
    *,
    train_subjects: np.ndarray | None = None,
    val_subjects: np.ndarray | None = None,
    train_feature_views: Mapping[str, np.ndarray] | None = None,
    val_feature_views: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_views = _validate_feature_views(
        train_feature_views,
        expected_length=int(np.asarray(X_train).shape[0]),
        prefix="train",
    )
    val_views = _validate_feature_views(
        val_feature_views,
        expected_length=int(np.asarray(X_val).shape[0]),
        prefix="val",
    )
    train_time_path = output_dir / "train_time_X.npy"
    train_freq_path = output_dir / "train_freq_X.npy"
    train_bands_path = output_dir / "train_bands_X.npy"
    train_y_path = output_dir / "train_y.npy"
    val_time_path = output_dir / "val_time_X.npy"
    val_freq_path = output_dir / "val_freq_X.npy"
    val_bands_path = output_dir / "val_bands_X.npy"
    val_y_path = output_dir / "val_y.npy"

    train_time = _as_float32_contiguous(train_views["time"])
    val_time = _as_float32_contiguous(val_views["time"])
    np.save(train_time_path, train_time, allow_pickle=False)
    np.save(train_freq_path, _as_float32_contiguous(train_views["freq"]), allow_pickle=False)
    np.save(train_bands_path, _as_float32_contiguous(train_views["bands"]), allow_pickle=False)
    np.save(val_time_path, val_time, allow_pickle=False)
    np.save(val_freq_path, _as_float32_contiguous(val_views["freq"]), allow_pickle=False)
    np.save(val_bands_path, _as_float32_contiguous(val_views["bands"]), allow_pickle=False)
    if y_train is not None:
        np.save(train_y_path, _as_int64_contiguous(y_train), allow_pickle=False)
    if y_val is not None:
        np.save(val_y_path, _as_int64_contiguous(y_val), allow_pickle=False)

    saved_paths: dict[str, str] = {
        "train_time_X": str(train_time_path.resolve()),
        "train_freq_X": str(train_freq_path.resolve()),
        "train_bands_X": str(train_bands_path.resolve()),
        "train_y": str(train_y_path.resolve()),
        "val_time_X": str(val_time_path.resolve()),
        "val_freq_X": str(val_freq_path.resolve()),
        "val_bands_X": str(val_bands_path.resolve()),
        "val_y": str(val_y_path.resolve()),
    }

    if train_subjects is not None:
        train_subjects_path = output_dir / "train_subjects.npy"
        np.save(
            train_subjects_path,
            np.ascontiguousarray(np.asarray(train_subjects), dtype=np.int32),
            allow_pickle=False,
        )
        saved_paths["train_subjects"] = str(train_subjects_path.resolve())
    if val_subjects is not None:
        val_subjects_path = output_dir / "val_subjects.npy"
        np.save(
            val_subjects_path,
            np.ascontiguousarray(np.asarray(val_subjects), dtype=np.int32),
            allow_pickle=False,
        )
        saved_paths["val_subjects"] = str(val_subjects_path.resolve())

    if metadata is not None:
        meta_path = output_dir / "prepared_meta.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_serializable(dict(metadata)), handle, indent=2)
        saved_paths["meta"] = str(meta_path.resolve())
    return saved_paths


def _label_distribution_for_print(y: np.ndarray | None, meta: Mapping[str, Any]) -> dict[str, int]:
    if y is None:
        return {}
    mapping = {int(key): str(value) for key, value in meta.get("label_mapping", {}).items()}
    counts = Counter(map(int, np.asarray(y).reshape(-1).tolist()))
    return {
        mapping.get(index, str(index)): count
        for index, count in sorted(counts.items(), key=lambda item: item[0])
    }


def _default_sample_path() -> Path:
    candidates = [
        Path("PhysioNet_EEG/S001/S001R04.edf"),
        Path("BCICIV_2a_gdf/A01T.gdf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No default sample EEG file found in the current workspace.")


def _run_single_file_pipeline(
    path: Path,
    low: float,
    high: float,
    window_sec: float,
    overlap_sec: float,
    include_gamma: bool,
    band_method: str,
    band_segments: int,
    seed: int,
    output_dir: Path | None = None,
) -> None:
    print(f"Inspecting: {path}")
    inspected = inspect_dataset(str(path))

    X, y, meta = load_eeg_data(str(path))
    preprocessor = EEGPreprocessor(
        low=low,
        high=high,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        include_gamma=include_gamma,
        band_method=band_method,
        band_segments=band_segments,
        seed=seed,
        verbose=True,
    )
    X_processed = preprocessor.preprocess(X, meta)
    X_valid, y_valid, validation_report = validate_data(X_processed, y)
    feature_views = preprocessor.create_feature_views(
        X_valid,
        meta,
        already_preprocessed=True,
    )

    if output_dir is not None:
        train_views, y_train, val_views, y_val = _split_feature_views_for_save(
            feature_views,
            y_valid,
            val_ratio=0.2,
            seed=seed,
        )
        single_file_metadata = {
            "dataset": inspected.get("dataset_type"),
            "source_path": str(path.resolve()),
            "sampling_frequency": meta.get("sampling_frequency"),
            "channel_names": list(meta.get("channel_names", [])),
            "preprocessing": {
                "low": low,
                "high": high,
                "window_sec": window_sec,
                "overlap_sec": overlap_sec,
                "include_gamma": include_gamma,
                "band_method": band_method,
                "band_segments": int(band_segments),
                "seed": seed,
                "split_mode": "single_file_train_val_split",
                "val_ratio": 0.2,
            },
            "available_views": list(REQUIRED_FEATURE_VIEW_KEYS),
            "inspection": inspected,
            "validation_report": _to_serializable(validation_report),
            "class_distribution": {
                "train": _class_distribution_from_labels(y_train) if y_train is not None else {},
                "val": _class_distribution_from_labels(y_val) if y_val is not None else {},
            },
            "shapes": {
                "train_time_X": tuple(int(dim) for dim in train_views["time"].shape),
                "train_freq_X": tuple(int(dim) for dim in train_views["freq"].shape),
                "train_bands_X": tuple(int(dim) for dim in train_views["bands"].shape),
                "val_time_X": tuple(int(dim) for dim in val_views["time"].shape),
                "val_freq_X": tuple(int(dim) for dim in val_views["freq"].shape),
                "val_bands_X": tuple(int(dim) for dim in val_views["bands"].shape),
            },
        }
        saved_paths = save_prepared_arrays(
            train_views["time"],
            y_train,
            val_views["time"],
            y_val,
            output_dir,
            train_feature_views=train_views,
            val_feature_views=val_views,
            metadata=single_file_metadata,
        )
        _verify_saved_outputs(
            saved_paths,
            required_keys=[
                "train_time_X",
                "train_freq_X",
                "train_bands_X",
                "val_time_X",
                "val_freq_X",
                "val_bands_X",
                "meta",
            ],
        )
        print(f"Saved arrays: {saved_paths}")

    print("\nFinal summary")
    print(f"Final shape: {tuple(int(dim) for dim in X_valid.shape)}")
    print(f"Number of trials: {int(X_valid.shape[0])}")
    print(f"Channels: {meta['channel_names']}")
    print(f"Label distribution: {_label_distribution_for_print(y_valid, meta) or 'unlabeled'}")
    print(
        "Feature views: "
        f"time={tuple(int(dim) for dim in feature_views['time'].shape)}, "
        f"freq={tuple(int(dim) for dim in feature_views['freq'].shape)}, "
        f"bands={tuple(int(dim) for dim in feature_views['bands'].shape)}"
    )
    print(f"Validation report: {_to_serializable(validation_report)}")
    print(f"Inspection dataset type: {inspected.get('dataset_type')}")


def _run_subjectwise_preparation(
    dataset: str,
    root_dir: Path,
    output_dir: Path,
    low: float,
    high: float,
    window_sec: float,
    overlap_sec: float,
    include_gamma: bool,
    band_method: str,
    band_segments: int,
    seed: int,
    task_filter: str = "mi_hand",
    channel_set: str = "motor21",
    save_mode: str = "all",
) -> None:
    resolved_root = root_dir.expanduser()
    if dataset == "physionet":
        resolved_root = _resolve_collection_root(resolved_root, preferred_child="PhysioNet_EEG")
        subject_ids = list(range(1, 110))
        file_resolver = lambda root, sid: _physionet_subject_files(root, sid, task_filter=task_filter)
        global_label_to_index = _physionet_global_label_to_index(task_filter)
        train_subjects = list(range(1, 81))
        val_subjects = list(range(81, 110))
        fixed_target_sec = window_sec
        allowed_runs = PHYSIONET_TASK_RUNS.get(task_filter, set())
        print(f"[TASK-FILTER] task_filter={task_filter!r} -> runs={sorted(allowed_runs)}")
    elif dataset == "bci":
        resolved_root = _resolve_collection_root(resolved_root, preferred_child="BCICIV_2a_gdf")
        subject_ids = list(range(1, 10))
        file_resolver = _bci_subject_files
        global_label_to_index = BCI_IV_2A_GLOBAL_LABEL_TO_INDEX
        train_subjects = list(range(1, 7))
        val_subjects = list(range(7, 10))
        fixed_target_sec = None
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_cache_dir = output_dir / "_subject_cache"
    subject_cache_dir.mkdir(parents=True, exist_ok=True)

    # Channel selection: use all 64 EEG channels or restrict to 21-channel motor strip
    if dataset == "physionet" and str(channel_set).strip().lower() == "all":
        target_channels = None  # Use all available EEG channels
        print("[CHANNELS] Using ALL EEG channels (64 for PhysioNet)")
    elif dataset == "physionet":
        target_channels = PHYSIONET_MOTOR_CHANNELS
        print("[CHANNELS] Using 21-channel motor strip")
    else:
        target_channels = None

    preprocessor = EEGPreprocessor(
        low=low,
        high=high,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        include_gamma=include_gamma,
        band_method=band_method,
        band_segments=band_segments,
        target_channels=target_channels,
        seed=seed,
        verbose=False,
    )
    target_spec: dict[str, Any] = {
        "channels": None,
        "time": None,
        "sampling_frequency": None,
    }
    subject_artifacts: list[dict[str, Any]] = []

    try:
        for subject_index, subject_id in enumerate(subject_ids, start=1):
            files = file_resolver(resolved_root, subject_id)
            if not files:
                LOGGER.warning("%s subject %s: no files found.", dataset, subject_id)
                continue
            _log_memory(
                f"[STREAM] {dataset} subject {subject_id:03d} "
                f"({subject_index}/{len(subject_ids)})"
            )
            artifact = _prepare_subject_artifact(
                subject_id=subject_id,
                files=files,
                dataset_name=dataset,
                preprocessor=preprocessor,
                global_label_to_index=global_label_to_index,
                target_spec=target_spec,
                subject_cache_dir=subject_cache_dir,
                fixed_target_sec=fixed_target_sec,
            )
            if artifact is not None:
                n = int(artifact["n_samples"])
                # FIX: skip near-empty subjects (e.g. S088/S092 -- 128 Hz recordings
                # where all MI files fail the window check, leaving only 2 baseline
                # trials). Including them in val destroys metric reliability.
                # Threshold=50 is conservative; lowest legitimate subject (S100)=290.
                _MIN_TRIALS_THRESHOLD = 50
                if n < _MIN_TRIALS_THRESHOLD:
                    LOGGER.warning(
                        "[SKIP] %s subject %s: only %d trials (< %d threshold). "
                        "Likely corrupt/short-window recording. Excluded from dataset.",
                        dataset, subject_id, n, _MIN_TRIALS_THRESHOLD,
                    )
                else:
                    subject_artifacts.append(artifact)
    finally:
        gc.collect()

    preprocessing_config = {
        "low": low,
        "high": high,
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
        "include_gamma": include_gamma,
        "band_method": band_method,
        "band_segments": int(band_segments),
        "channel_set": str(channel_set),
        "task_filter": str(task_filter),
        "save_mode": str(save_mode),
        "save_feature_views": True,
        "seed": seed,
        "streaming": True,
    }
    if str(save_mode).strip().lower() == "all":
        saved_paths = _merge_all_subject_artifacts(
            subject_artifacts=subject_artifacts,
            output_dir=output_dir,
            dataset=dataset,
            root_dir=resolved_root,
            preprocessing_config=preprocessing_config,
        )
    else:
        saved_paths = _merge_subject_artifacts(
            subject_artifacts=subject_artifacts,
            output_dir=output_dir,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            dataset=dataset,
            root_dir=resolved_root,
            preprocessing_config=preprocessing_config,
        )

    if subject_cache_dir.exists():
        shutil.rmtree(subject_cache_dir, ignore_errors=True)
    train_total = sum(
        int(artifact["n_samples"]) for artifact in subject_artifacts if int(artifact["subject_id"]) in set(train_subjects)
    )
    val_total = sum(
        int(artifact["n_samples"]) for artifact in subject_artifacts if int(artifact["subject_id"]) in set(val_subjects)
    )
    used_subjects = sorted(int(artifact["subject_id"]) for artifact in subject_artifacts)
    merged_meta_path = Path(saved_paths["meta"])
    with merged_meta_path.open("r", encoding="utf-8") as handle:
        merged_meta = json.load(handle)

    print("\nSubject-wise preparation summary")
    print(f"Dataset: {dataset}")
    print(f"Root: {resolved_root}")
    print(f"Subjects used: {used_subjects}")
    print(f"Total samples: {int(train_total + val_total)}")
    print(f"Save mode: {save_mode}")
    if str(save_mode).strip().lower() == "all":
        print(f"All shape: {tuple(int(dim) for dim in merged_meta['shapes']['all_X'])}")
        print(f"All class distribution: {merged_meta['class_distribution']['all']}")
    else:
        print(f"Train subjects: {train_subjects}")
        print(f"Test subjects: {val_subjects}")
        print(f"Train shape: {tuple(int(dim) for dim in merged_meta['shapes']['train_X'])}")
        print(f"Val shape: {tuple(int(dim) for dim in merged_meta['shapes']['val_X'])}")
        print(f"Train class distribution: {merged_meta['class_distribution']['train']}")
        print(f"Val class distribution: {merged_meta['class_distribution']['val']}")
    print(f"Saved arrays: {saved_paths}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust EEG preprocessing pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("physionet", "bci"),
        default=None,
        help="Prepare a full subject-wise dataset split for PhysioNet or BCI IV 2a.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for multi-subject dataset preparation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for saved prepared NumPy arrays.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to an EEG file or directory. Defaults to a local sample file if omitted.",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=4.0,  # matches EEGPreprocessor default (theta lower bound)
        help="Low cutoff frequency for bandpass filtering (Hz). Default: 4.0",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=38.0,  # FIXED: widened from 30 to 38 Hz to capture low gamma
        help="High cutoff frequency for bandpass filtering. Default: 38.0",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Epoch length in seconds for continuous recordings.",
    )
    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=0.0,
        help="Overlap between adjacent sliding windows in seconds.",
    )
    parser.add_argument(
        "--include-gamma",
        action="store_true",
        help="Include gamma-band feature extraction metadata and optional saved views.",
    )
    parser.add_argument(
        "--save-mode",
        type=str,
        default="all",
        choices=("all", "holdout"),
        help="Save canonical all_* artifacts for subject CV or legacy train/val holdout arrays.",
    )
    parser.add_argument(
        "--channel-set",
        type=str,
        default="motor21",
        choices=("motor21", "all"),
        help="PhysioNet channel selection. motor21 is the research default; all keeps all EEG channels.",
    )
    parser.add_argument(
        "--task-filter",
        type=str,
        default="mi_hand",
        choices=tuple(PHYSIONET_TASK_RUNS.keys()),
        help=(
            "Which PhysioNet run types to include. "
            "mi_hand=imagined left/right hand (R03,R07,R11). "
            "mi_fistfeet=imagined fists/feet (R04,R08,R12). "
            "all=all 14 runs (LEGACY, not recommended). "
            "Default: mi_hand."
        ),
    )
    parser.add_argument(
        "--band-method",
        type=str,
        default="fft_segments",
        choices=("fft_segments", "bandpass_segments"),
        help="Band-feature construction method. fft_segments is the shared research-grade default.",
    )
    parser.add_argument(
        "--band-segments",
        type=int,
        default=8,
        help="Number of temporal segments per band feature view.",
    )
    parser.add_argument(
        "--use-all-channels",
        type=str,
        default="false",
        help="Deprecated compatibility flag. Prefer --channel-set {motor21,all}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible preprocessing decisions.",
    )
    args = parser.parse_args()
    use_all_channels = str(args.use_all_channels).strip().lower() not in {"0", "false", "no", "off"}
    if use_all_channels and args.channel_set == "motor21":
        args.channel_set = "all"
    if float(args.high) > 30.0 and not bool(args.include_gamma):
        LOGGER.warning(
            "high cutoff %.1f Hz is above beta range while --include-gamma is disabled. "
            "For tighter cross-view spectral alignment, use --high 30.0 or enable --include-gamma.",
            float(args.high),
        )

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_global_seed(args.seed)
    output_dir = None if args.output is None else Path(args.output).expanduser()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving to {output_dir}")

    if args.dataset is not None:
        if args.root is None or args.output is None:
            parser.error("--root and --output are required when --dataset is used.")
        _run_subjectwise_preparation(
            dataset=args.dataset,
            root_dir=Path(args.root),
            output_dir=output_dir,
            low=args.low,
            high=args.high,
            window_sec=args.window_sec,
            overlap_sec=args.overlap_sec,
            include_gamma=args.include_gamma,
            band_method=args.band_method,
            band_segments=args.band_segments,
            seed=args.seed,
            task_filter=args.task_filter,
            channel_set=args.channel_set,
            save_mode=args.save_mode,
        )
        return

    sample_path = Path(args.path) if args.path is not None else _default_sample_path()
    if output_dir is not None and sample_path.is_dir():
        inferred_dataset = _infer_dataset_from_root(sample_path)
        if inferred_dataset is not None:
            print(
                "[INFO] Detected dataset-style directory input without --dataset. "
                f"Routing to subject-wise {inferred_dataset} preparation."
            )
            _run_subjectwise_preparation(
                dataset=inferred_dataset,
                root_dir=sample_path,
                output_dir=output_dir,
                low=args.low,
                high=args.high,
                window_sec=args.window_sec,
                overlap_sec=args.overlap_sec,
                include_gamma=args.include_gamma,
                band_method=args.band_method,
                band_segments=args.band_segments,
                seed=args.seed,
                task_filter=args.task_filter,
                channel_set=args.channel_set,
                save_mode=args.save_mode,
            )
            return
    _run_single_file_pipeline(
        sample_path,
        args.low,
        args.high,
        args.window_sec,
        args.overlap_sec,
        args.include_gamma,
        args.band_method,
        args.band_segments,
        args.seed,
        output_dir,
    )


if __name__ == "__main__":
    main()