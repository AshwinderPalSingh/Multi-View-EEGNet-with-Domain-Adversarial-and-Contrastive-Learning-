from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_EEG_BANDS = {
    "mu": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def prepare_band_definitions(
    band_definitions: Mapping[str, tuple[float, float]] | None = None,
    *,
    include_gamma: bool = False,
) -> dict[str, tuple[float, float]]:
    band_map = dict(band_definitions) if band_definitions is not None else dict(DEFAULT_EEG_BANDS)
    if not include_gamma and band_definitions is None:
        band_map.pop("gamma", None)
    return {
        str(name): (float(low_hz), float(high_hz))
        for name, (low_hz, high_hz) in band_map.items()
        if float(low_hz) < float(high_hz)
    }


def _effective_segment_count(time_length: int, n_segments: int) -> int:
    if time_length <= 0:
        raise ValueError(f"time_length must be positive, got {time_length}.")
    return max(1, min(int(n_segments), int(time_length)))


def compute_frequency_features_numpy(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    fft = np.fft.rfft(X, axis=-1, norm="ortho")
    raw_power = (fft.real * fft.real + fft.imag * fft.imag).astype(np.float32, copy=False)
    return (10.0 * np.log10(raw_power + 1e-12)).astype(np.float32, copy=False)


def compute_segmented_bandpower_numpy(
    X: np.ndarray,
    sfreq: float,
    band_definitions: Mapping[str, tuple[float, float]],
    *,
    n_segments: int = 8,
) -> tuple[np.ndarray, list[str]]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, T), got {X.shape}.")
    if sfreq <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {sfreq}.")

    n_segments = _effective_segment_count(int(X.shape[-1]), n_segments)
    seg_len = max(int(X.shape[-1]) // n_segments, 1)
    usable_len = seg_len * n_segments
    trimmed = np.asarray(X[:, :, :usable_len], dtype=np.float32)
    segmented = trimmed.reshape(trimmed.shape[0], trimmed.shape[1], n_segments, seg_len)

    fft = np.fft.rfft(segmented, axis=-1, norm="ortho")
    power = (fft.real * fft.real + fft.imag * fft.imag).astype(np.float32, copy=False)
    freqs = np.fft.rfftfreq(seg_len, d=1.0 / float(sfreq))
    nyquist = float(sfreq) / 2.0

    band_chunks: list[np.ndarray] = []
    band_names: list[str] = []
    for band_name, (low_hz, high_hz) in band_definitions.items():
        if float(low_hz) >= nyquist:
            continue
        high = min(float(high_hz), nyquist)
        mask = (freqs >= float(low_hz)) & (freqs < (high + 1e-6))
        if not np.any(mask):
            continue
        band_power = np.log(np.mean(power[..., mask], axis=-1, dtype=np.float32) + 1e-6).astype(np.float32, copy=False)
        band_chunks.append(band_power)
        for seg_idx in range(n_segments):
            band_names.append(f"{band_name}_s{seg_idx}")

    if not band_chunks:
        return np.empty((X.shape[0], X.shape[1], 0), dtype=np.float32), []
    return np.concatenate(band_chunks, axis=-1).astype(np.float32, copy=False), band_names


def build_feature_views_numpy(
    X_time: np.ndarray,
    sfreq: float,
    band_definitions: Mapping[str, tuple[float, float]],
    *,
    n_segments: int = 8,
) -> tuple[dict[str, np.ndarray], list[str]]:
    X_time = np.asarray(X_time, dtype=np.float32)
    X_freq = compute_frequency_features_numpy(X_time)
    X_band, band_names = compute_segmented_bandpower_numpy(
        X_time,
        float(sfreq),
        band_definitions,
        n_segments=n_segments,
    )
    return {
        "raw": X_time,
        "time": X_time,
        "freq": X_freq,
        "bands": X_band,
    }, band_names


def compute_frequency_features_torch(x: Any) -> Any:
    import torch

    x = x.float()
    fft = torch.fft.rfft(x, dim=-1, norm="ortho")
    raw_power = fft.real.square() + fft.imag.square()
    return (10.0 * raw_power.clamp_min(1e-12).log10()).float()


def compute_segmented_bandpower_torch(
    x: Any,
    sfreq: float,
    band_definitions: Mapping[str, tuple[float, float]],
    *,
    n_segments: int = 8,
) -> Any:
    import torch

    x = x.float()
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (B, C, T), got {tuple(x.shape)}.")
    if sfreq <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {sfreq}.")

    n_segments = _effective_segment_count(int(x.shape[-1]), n_segments)
    seg_len = max(int(x.shape[-1]) // n_segments, 1)
    usable_len = seg_len * n_segments
    segmented = x[:, :, :usable_len].contiguous().view(x.shape[0], x.shape[1], n_segments, seg_len)

    fft = torch.fft.rfft(segmented, dim=-1, norm="ortho")
    power = fft.real.square() + fft.imag.square()
    freqs = torch.fft.rfftfreq(seg_len, d=1.0 / float(sfreq), device=x.device)
    nyquist = float(sfreq) / 2.0

    band_chunks: list[Any] = []
    for low_hz, high_hz in band_definitions.values():
        if float(low_hz) >= nyquist:
            continue
        high = min(float(high_hz), nyquist)
        mask = (freqs >= float(low_hz)) & (freqs < (high + 1e-6))
        if not bool(mask.any().item()):
            continue
        band_power = torch.log(power[..., mask].mean(dim=-1).clamp_min(1e-6))
        band_chunks.append(band_power)

    if not band_chunks:
        return torch.empty((x.shape[0], x.shape[1], 0), device=x.device, dtype=torch.float32)
    return torch.cat(band_chunks, dim=-1).float()


def build_feature_views_torch(
    time_x: Any,
    sfreq: float,
    band_definitions: Mapping[str, tuple[float, float]],
    *,
    n_segments: int = 8,
) -> dict[str, Any]:
    time_x = time_x.float()
    return {
        "time": time_x,
        "freq": compute_frequency_features_torch(time_x),
        "bands": compute_segmented_bandpower_torch(
            time_x,
            float(sfreq),
            band_definitions,
            n_segments=n_segments,
        ),
    }