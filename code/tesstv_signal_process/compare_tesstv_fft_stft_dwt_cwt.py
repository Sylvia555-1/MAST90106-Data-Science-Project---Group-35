#!/usr/bin/env python3
"""Compare FFT, STFT, DWT and CWT for fixed-parameter t-ESSTV.

Two comparisons are produced from the same common APR18/JUN18/JUL18 step
protocols:

1. signal representations: FFT spectrum, STFT spectrogram, DWT detail energy,
   and CWT scalogram;
2. model effect: reconstruct a denoised shear-rate history from each transform,
   run the same published Donor 5 t-ESSTV parameters, and score every prediction
   against the same untouched raw measured stress.

The original step records are non-uniformly sampled.  Each transform is applied
only after interpolation to a uniform physical-time grid over the transition
window (default 0-1 s).  The reconstructed signal is interpolated back to the
original measurement times; later raw plateau points are left unchanged.

Keep this file beside:

* tensorial_esstv_figure5_v3.py
* compare_tesstv_raw_baselines.py
* compare_tesstv_signal_preprocessing.py

Required packages for the full comparison:

    python3 -m pip install numpy pandas scipy matplotlib openpyxl PyWavelets ssqueezepy

Example:

    python3 compare_tesstv_fft_stft_dwt_cwt.py \
      --apr "APR18 SS SAOS STEPS LAOS UDLAOS.xlsx" \
      --jun "JUN18.xlsx" --jul "JUL18.xlsx" \
      --output "fft_stft_dwt_cwt_comparison"
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

try:
    import tensorial_esstv_figure5_v3 as tesstv
    import compare_tesstv_raw_baselines as baseline
    import compare_tesstv_signal_preprocessing as common_tools
except ImportError as exc:
    raise SystemExit(
        "Put tensorial_esstv_figure5_v3.py, compare_tesstv_raw_baselines.py "
        "and compare_tesstv_signal_preprocessing.py beside this script."
    ) from exc


METHOD_RAW = "raw"
METHOD_FFT = "FFT"
METHOD_STFT = "STFT"
METHOD_DWT = "DWT"
METHOD_CWT = "CWT"
EPS = 1.0e-12


@dataclass(frozen=True)
class Config:
    processing_window_s: float
    max_grid_points: int
    cutoff_hz: float | None
    cutoff_duration_factor: float
    taper_ratio: float
    stft_nperseg: int
    stft_overlap_fraction: float
    dwt_wavelet: str
    dwt_level: int
    dwt_threshold_scale: float
    cwt_wavelet: str
    cwt_voices: int
    cwt_threshold_scale: float
    clip_margin_fraction: float


@dataclass
class UniformSegment:
    grid: np.ndarray
    values: np.ndarray
    end_time: float
    metadata: dict[str, object]


@dataclass
class TransformResult:
    uniform_filtered: np.ndarray
    metadata: dict[str, object]
    diagnostic: dict[str, np.ndarray]


def safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return value or "item"


def protocol_name(key: tuple[float, float]) -> str:
    return f"{key[0]:g}->{key[1]:g}"


def upper_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << int(value - 1).bit_length()


def lower_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value).bit_length() - 1)


def make_uniform_segment(
    time: np.ndarray,
    shear: np.ndarray,
    config: Config,
) -> UniformSegment:
    common_tools.validate_time_signal(time, shear)
    end_time = float(min(config.processing_window_s, time[-1]))
    if end_time <= time[0]:
        raise ValueError("The processing window contains no positive time span.")
    in_window = time <= end_time
    segment_time = time[in_window]
    segment_shear = shear[in_window]
    if segment_time[-1] < end_time:
        segment_time = np.append(segment_time, end_time)
        segment_shear = np.append(segment_shear, np.interp(end_time, time, shear))

    dt = np.diff(segment_time)
    reference_dt = float(np.quantile(dt, 0.10))
    requested = max(256, int(math.ceil((end_time - time[0]) / reference_dt)) + 1)
    cap = lower_power_of_two(max(256, config.max_grid_points))
    n_grid = min(upper_power_of_two(requested), cap)
    grid = np.linspace(time[0], end_time, n_grid)
    uniform = np.interp(grid, segment_time, segment_shear)
    grid_dt = float(grid[1] - grid[0])
    return UniformSegment(
        grid=grid,
        values=uniform,
        end_time=end_time,
        metadata={
            "processing_window_start_s": float(time[0]),
            "processing_window_end_s": end_time,
            "original_n_points": int(len(time)),
            "original_points_in_processing_window": int(np.sum(in_window)),
            "uniform_n_points": int(n_grid),
            "original_dt_min_in_window_s": float(np.min(dt)),
            "original_dt_median_in_window_s": float(np.median(dt)),
            "original_dt_max_in_window_s": float(np.max(dt)),
            "uniform_dt_s": grid_dt,
            "uniform_sampling_frequency_hz": float(1.0 / grid_dt),
            "uniform_nyquist_hz": float(0.5 / grid_dt),
        },
    )


def transition_duration(
    time: np.ndarray,
    shear: np.ndarray,
    pre_rate: float,
    target_rate: float,
) -> float:
    return common_tools.estimate_transition_duration(
        time, shear, pre_rate, target_rate
    )


def cutoff_for_signal(
    segment: UniformSegment,
    time: np.ndarray,
    shear: np.ndarray,
    pre_rate: float,
    target_rate: float,
    config: Config,
) -> tuple[float, float, str]:
    duration = transition_duration(time, shear, pre_rate, target_rate)
    nyquist = float(segment.metadata["uniform_nyquist_hz"])
    if config.cutoff_hz is None:
        cutoff = config.cutoff_duration_factor / max(duration, EPS)
        rule = "duration_factor/10-90%_transition_duration"
    else:
        cutoff = config.cutoff_hz
        rule = "fixed_user_cutoff_hz"
    minimum = 1.0 / max(segment.end_time - segment.grid[0], EPS)
    return float(np.clip(cutoff, minimum, 0.90 * nyquist)), duration, rule


def cosine_lowpass(frequencies: np.ndarray, cutoff: float, taper_ratio: float) -> tuple[np.ndarray, float]:
    stop = float(cutoff * (1.0 + taper_ratio))
    transfer = np.ones_like(frequencies, dtype=float)
    transfer[frequencies >= stop] = 0.0
    transition = (frequencies > cutoff) & (frequencies < stop)
    if stop > cutoff:
        phase = (frequencies[transition] - cutoff) / (stop - cutoff)
        transfer[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return transfer, stop


def fft_transform(
    segment: UniformSegment,
    cutoff: float,
    duration: float,
    cutoff_rule: str,
    config: Config,
) -> TransformResult:
    values = segment.values
    dt = float(segment.metadata["uniform_dt_s"])
    pad = max(32, min(len(values) // 8, 512))
    padded = np.pad(values, (pad, pad), mode="edge")
    frequencies = np.fft.rfftfreq(len(padded), d=dt)
    spectrum = np.fft.rfft(padded)
    transfer, stop = cosine_lowpass(frequencies, cutoff, config.taper_ratio)
    reconstructed = np.fft.irfft(spectrum * transfer, n=len(padded))[pad:-pad]
    return TransformResult(
        reconstructed,
        {
            "method": METHOD_FFT,
            "transition_duration_10_90_s": duration,
            "cutoff_hz": cutoff,
            "cutoff_rule": cutoff_rule,
            "taper_stop_hz": stop,
            "stft_nperseg": np.nan,
            "stft_noverlap": np.nan,
            "dwt_wavelet": "not_applicable",
            "dwt_level": np.nan,
            "coefficient_threshold": np.nan,
            "cwt_wavelet": "not_applicable",
            "cwt_voices": np.nan,
            "inverse_reference_rmse": 0.0,
        },
        {
            "frequency_hz": frequencies,
            "magnitude": np.abs(spectrum),
            "transfer": transfer,
        },
    )


def stft_transform(
    segment: UniformSegment,
    cutoff: float,
    duration: float,
    cutoff_rule: str,
    config: Config,
) -> TransformResult:
    values = segment.values
    fs = float(segment.metadata["uniform_sampling_frequency_hz"])
    nperseg = min(config.stft_nperseg, len(values))
    nperseg = max(16, nperseg)
    noverlap = min(nperseg - 1, int(round(nperseg * config.stft_overlap_fraction)))
    frequencies, times, coefficients = scipy_signal.stft(
        values,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="even",
        padded=True,
    )
    transfer, stop = cosine_lowpass(frequencies, cutoff, config.taper_ratio)
    filtered_coefficients = coefficients * transfer[:, None]
    _, reconstructed = scipy_signal.istft(
        filtered_coefficients,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    if len(reconstructed) < len(values):
        reconstructed = np.pad(reconstructed, (0, len(values) - len(reconstructed)), mode="edge")
    reconstructed = reconstructed[: len(values)]
    return TransformResult(
        reconstructed,
        {
            "method": METHOD_STFT,
            "transition_duration_10_90_s": duration,
            "cutoff_hz": cutoff,
            "cutoff_rule": cutoff_rule,
            "taper_stop_hz": stop,
            "stft_nperseg": int(nperseg),
            "stft_noverlap": int(noverlap),
            "dwt_wavelet": "not_applicable",
            "dwt_level": np.nan,
            "coefficient_threshold": np.nan,
            "cwt_wavelet": "not_applicable",
            "cwt_voices": np.nan,
            "inverse_reference_rmse": 0.0,
        },
        {
            "frequency_hz": frequencies,
            "time_s": times + segment.grid[0],
            "magnitude": np.abs(coefficients),
            "transfer": transfer,
        },
    )


def dwt_transform(segment: UniformSegment, config: Config) -> TransformResult:
    values = segment.values
    if config.dwt_wavelet.lower() == "haar":
        reconstructed, wavelet_metadata = common_tools.haar_denoise(
            values, config.dwt_level, config.dwt_threshold_scale
        )
        detail_energy = np.array([], dtype=float)
    else:
        try:
            import pywt  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "DWT requires PyWavelets. Run: python3 -m pip install PyWavelets"
            ) from exc
        wavelet = pywt.Wavelet(config.dwt_wavelet)
        maximum_level = pywt.dwt_max_level(len(values), wavelet.dec_len)
        actual_level = max(1, min(config.dwt_level, maximum_level))
        coefficients = pywt.wavedec(values, wavelet, mode="symmetric", level=actual_level)
        finest = np.asarray(coefficients[-1], dtype=float)
        sigma = float(np.median(np.abs(finest - np.median(finest))) / 0.67448975)
        threshold = float(
            config.dwt_threshold_scale * sigma * math.sqrt(2.0 * math.log(len(values)))
        )
        detail_energy = np.asarray(
            [float(np.sum(np.asarray(detail, dtype=float) ** 2)) for detail in coefficients[1:]],
            dtype=float,
        )
        filtered_coefficients = [
            coefficients[0],
            *[pywt.threshold(detail, threshold, mode="soft") for detail in coefficients[1:]],
        ]
        reconstructed = pywt.waverec(
            filtered_coefficients, wavelet, mode="symmetric"
        )[: len(values)]
        wavelet_metadata = {
            "wavelet": config.dwt_wavelet,
            "wavelet_level": actual_level,
            "wavelet_sigma": sigma,
            "wavelet_threshold": threshold,
        }
    return TransformResult(
        np.asarray(reconstructed, dtype=float),
        {
            "method": METHOD_DWT,
            "transition_duration_10_90_s": np.nan,
            "cutoff_hz": np.nan,
            "cutoff_rule": "not_applicable",
            "taper_stop_hz": np.nan,
            "stft_nperseg": np.nan,
            "stft_noverlap": np.nan,
            "dwt_wavelet": wavelet_metadata["wavelet"],
            "dwt_level": wavelet_metadata["wavelet_level"],
            "coefficient_threshold": wavelet_metadata["wavelet_threshold"],
            "cwt_wavelet": "not_applicable",
            "cwt_voices": np.nan,
            "inverse_reference_rmse": 0.0,
        },
        {
            "detail_level": np.arange(1, len(detail_energy) + 1),
            "detail_energy": detail_energy[::-1],
        },
    )


def complex_soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = np.abs(coefficients)
    shrink = np.maximum(1.0 - threshold / np.maximum(magnitude, EPS), 0.0)
    return coefficients * shrink


def cwt_transform(segment: UniformSegment, config: Config) -> TransformResult:
    try:
        from ssqueezepy import cwt, icwt  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CWT reconstruction requires ssqueezepy. Run: "
            "python3 -m pip install ssqueezepy"
        ) from exc

    values = segment.values.astype(float)
    mean = float(np.mean(values))
    centered = values - mean
    coefficients, scales = cwt(
        centered,
        wavelet=config.cwt_wavelet,
        scales="log-piecewise",
        nv=config.cwt_voices,
    )
    coefficients = np.asarray(coefficients)
    scales = np.asarray(scales)
    finest_count = max(1, min(4, coefficients.shape[0] // 8))
    finest = coefficients[:finest_count]
    sigma = float(
        np.median(np.abs(finest.real - np.median(finest.real))) / 0.67448975
    )
    threshold = float(
        config.cwt_threshold_scale * sigma * math.sqrt(2.0 * math.log(len(values)))
    )
    filtered_coefficients = complex_soft_threshold(coefficients, threshold)
    reference_inverse = np.asarray(
        icwt(
            coefficients,
            wavelet=config.cwt_wavelet,
            scales=scales,
            x_len=len(values),
            x_mean=mean,
            one_int=True,
        ),
        dtype=float,
    )
    filtered_inverse = np.asarray(
        icwt(
            filtered_coefficients,
            wavelet=config.cwt_wavelet,
            scales=scales,
            x_len=len(values),
            x_mean=mean,
            one_int=True,
        ),
        dtype=float,
    )
    # Subtract the unfiltered inverse error.  With a zero threshold this makes
    # the CWT branch exactly equal to the raw uniform signal.
    reconstructed = values + (filtered_inverse - reference_inverse)
    inverse_rmse = float(np.sqrt(np.mean((reference_inverse - values) ** 2)))
    return TransformResult(
        reconstructed,
        {
            "method": METHOD_CWT,
            "transition_duration_10_90_s": np.nan,
            "cutoff_hz": np.nan,
            "cutoff_rule": "coefficient_soft_threshold",
            "taper_stop_hz": np.nan,
            "stft_nperseg": np.nan,
            "stft_noverlap": np.nan,
            "dwt_wavelet": "not_applicable",
            "dwt_level": np.nan,
            "coefficient_threshold": threshold,
            "cwt_wavelet": config.cwt_wavelet,
            "cwt_voices": config.cwt_voices,
            "inverse_reference_rmse": inverse_rmse,
        },
        {
            "scale": scales,
            "magnitude": np.abs(coefficients),
        },
    )


def match_segment_endpoints(filtered: np.ndarray, raw: np.ndarray) -> np.ndarray:
    if len(filtered) < 2:
        return filtered
    fraction = np.linspace(0.0, 1.0, len(filtered))
    endpoint_error = (filtered[0] - raw[0]) * (1.0 - fraction)
    endpoint_error += (filtered[-1] - raw[-1]) * fraction
    return filtered - endpoint_error


def splice_to_original(
    time: np.ndarray,
    raw_shear: np.ndarray,
    segment: UniformSegment,
    uniform_filtered: np.ndarray,
    pre_rate: float,
    target_rate: float,
    config: Config,
) -> np.ndarray:
    filtered = match_segment_endpoints(uniform_filtered, segment.values)
    filtered = common_tools.stable_clip(
        filtered, pre_rate, target_rate, config.clip_margin_fraction
    )
    output = raw_shear.copy()
    mask = time <= segment.end_time
    output[mask] = np.interp(time[mask], segment.grid, filtered)
    return output


def process_test(
    source: tesstv.StepTest,
    method: str,
    config: Config,
) -> tuple[tesstv.StepTest, dict[str, object], dict[str, np.ndarray]]:
    data = source.data.copy()
    time = data["time"].to_numpy(float)
    raw_shear = data["shear_rate"].to_numpy(float)
    raw_stress = data["stress"].to_numpy(float).copy()
    segment = make_uniform_segment(time, raw_shear, config)
    cutoff, duration, cutoff_rule = cutoff_for_signal(
        segment, time, raw_shear, source.pre_rate, source.target_rate, config
    )

    if method == METHOD_RAW:
        processed = raw_shear.copy()
        transform_metadata = {
            "method": METHOD_RAW,
            "transition_duration_10_90_s": duration,
            "cutoff_hz": np.nan,
            "cutoff_rule": "not_applicable",
            "taper_stop_hz": np.nan,
            "stft_nperseg": np.nan,
            "stft_noverlap": np.nan,
            "dwt_wavelet": "not_applicable",
            "dwt_level": np.nan,
            "coefficient_threshold": np.nan,
            "cwt_wavelet": "not_applicable",
            "cwt_voices": np.nan,
            "inverse_reference_rmse": 0.0,
        }
        diagnostic: dict[str, np.ndarray] = {}
    else:
        if method == METHOD_FFT:
            transformed = fft_transform(segment, cutoff, duration, cutoff_rule, config)
        elif method == METHOD_STFT:
            transformed = stft_transform(segment, cutoff, duration, cutoff_rule, config)
        elif method == METHOD_DWT:
            transformed = dwt_transform(segment, config)
        elif method == METHOD_CWT:
            transformed = cwt_transform(segment, config)
        else:
            raise ValueError(f"Unknown method: {method}")
        processed = splice_to_original(
            time,
            raw_shear,
            segment,
            transformed.uniform_filtered,
            source.pre_rate,
            source.target_rate,
            config,
        )
        transform_metadata = transformed.metadata
        diagnostic = transformed.diagnostic

    data["shear_rate"] = processed
    if not np.array_equal(data["stress"].to_numpy(float), raw_stress):
        raise AssertionError("Measured stress target changed during preprocessing.")
    delta = processed - raw_shear
    metadata = {
        **segment.metadata,
        **transform_metadata,
        "shear_rate_rmse_change_s-1": float(np.sqrt(np.mean(delta**2))),
        "shear_rate_mae_change_s-1": float(np.mean(np.abs(delta))),
        "shear_rate_max_abs_change_s-1": float(np.max(np.abs(delta))),
        "raw_stress_target_changed": False,
        "model_parameters_changed": False,
    }
    test = tesstv.StepTest(
        title=f"{source.title} [{method}]",
        pre_rate=source.pre_rate,
        target_rate=source.target_rate,
        data=data,
        title_pre_rate=source.title_pre_rate,
        title_target_rate=source.title_target_rate,
        measured_pre_rate=source.measured_pre_rate,
        measured_target_rate=source.measured_target_rate,
        source_column=source.source_column,
        protocol_warning=source.protocol_warning,
    )
    return test, metadata, diagnostic


def method_colors() -> dict[str, str]:
    return {
        METHOD_RAW: "#7F7F7F",
        METHOD_FFT: "#4472C4",
        METHOD_STFT: "#ED7D31",
        METHOD_DWT: "#70AD47",
        METHOD_CWT: "#8064A2",
    }


def plot_summary(summary: pd.DataFrame, output: Path, methods: list[str]) -> None:
    datasets = list(dict.fromkeys(summary["dataset"]))
    colors = method_colors()
    x = np.arange(len(datasets), dtype=float)
    width = 0.15
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = [
        ("fcost_trans_eq12_normalized_rms", "Eq. 12 normalized cost", "cost"),
        ("rmse", "Transient RMSE", "Pa"),
        ("r2", "Transient R²", "R²"),
        ("eq12_improvement_percent_vs_raw", "Eq. 12 improvement vs raw", "%"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel(), specs):
        for offset, method in zip(offsets, methods):
            part = summary.loc[summary["method"] == method].set_index("dataset")
            axis.bar(
                x + offset,
                part.reindex(datasets)[column].to_numpy(float),
                width=width,
                label=method,
                color=colors[method],
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        if column == "fcost_trans_eq12_normalized_rms":
            axis.axhline(
                tesstv.PAPER_FCOST_TRANS,
                color="black",
                linestyle="--",
                linewidth=1,
                label="paper Donor 5 reference",
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, datasets)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Fixed Donor 5 t-ESSTV: FFT/STFT/DWT/CWT input comparison",
        fontweight="bold",
    )
    fig.tight_layout()
    common_tools.save_figure(fig, output, "fft_stft_dwt_cwt_model_comparison")


def diagnostic_for_method(
    source: tesstv.StepTest,
    method: str,
    config: Config,
) -> tuple[UniformSegment, dict[str, np.ndarray]]:
    time = source.data["time"].to_numpy(float)
    shear = source.data["shear_rate"].to_numpy(float)
    segment = make_uniform_segment(time, shear, config)
    cutoff, duration, rule = cutoff_for_signal(
        segment, time, shear, source.pre_rate, source.target_rate, config
    )
    if method == METHOD_FFT:
        result = fft_transform(segment, cutoff, duration, rule, config)
    elif method == METHOD_STFT:
        result = stft_transform(segment, cutoff, duration, rule, config)
    elif method == METHOD_DWT:
        result = dwt_transform(segment, config)
    elif method == METHOD_CWT:
        result = cwt_transform(segment, config)
    else:
        raise ValueError(method)
    return segment, result.diagnostic


def plot_transform_comparison(
    dataset: str,
    source: tesstv.StepTest,
    config: Config,
    output: Path,
    include_cwt: bool,
) -> None:
    methods = [METHOD_FFT, METHOD_STFT, METHOD_DWT] + ([METHOD_CWT] if include_cwt else [])
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    segments: dict[str, UniformSegment] = {}
    for method in methods:
        segment, diagnostic = diagnostic_for_method(source, method, config)
        diagnostics[method] = diagnostic
        segments[method] = segment

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fft = diagnostics[METHOD_FFT]
    valid = fft["frequency_hz"] > 0
    axes[0, 0].plot(fft["frequency_hz"][valid], fft["magnitude"][valid])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("FFT: global spectrum")
    axes[0, 0].set_xlabel("frequency (Hz)")
    axes[0, 0].set_ylabel("magnitude")

    stft = diagnostics[METHOD_STFT]
    image_stft = axes[0, 1].pcolormesh(
        stft["time_s"],
        stft["frequency_hz"],
        np.log10(stft["magnitude"] + EPS),
        shading="auto",
    )
    axes[0, 1].set_ylim(0, min(250, float(np.max(stft["frequency_hz"]))))
    axes[0, 1].set_title("STFT: local spectrum")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("frequency (Hz)")
    fig.colorbar(image_stft, ax=axes[0, 1], label="log10 magnitude")

    dwt = diagnostics[METHOD_DWT]
    if len(dwt["detail_level"]):
        axes[1, 0].bar(dwt["detail_level"], dwt["detail_energy"])
    else:
        axes[1, 0].text(0.5, 0.5, "Haar fallback: detail energy unavailable", ha="center")
    axes[1, 0].set_title("DWT: detail energy by level")
    axes[1, 0].set_xlabel("detail level (1 = finest)")
    axes[1, 0].set_ylabel("energy")

    if include_cwt:
        cwt = diagnostics[METHOD_CWT]
        segment = segments[METHOD_CWT]
        image_cwt = axes[1, 1].imshow(
            np.log10(cwt["magnitude"] + EPS),
            aspect="auto",
            origin="upper",
            extent=[segment.grid[0], segment.grid[-1], len(cwt["scale"]), 1],
        )
        axes[1, 1].set_title("CWT: Morlet scalogram")
        axes[1, 1].set_xlabel("time (s)")
        axes[1, 1].set_ylabel("scale index (small = high frequency)")
        fig.colorbar(image_cwt, ax=axes[1, 1], label="log10 magnitude")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5, 0.5, "CWT skipped for validation", ha="center", va="center"
        )
    for axis in axes.ravel():
        axis.grid(alpha=0.15)
    fig.suptitle(
        f"{dataset} {protocol_name((source.pre_rate, source.target_rate))} s⁻¹: transform comparison",
        fontweight="bold",
    )
    fig.tight_layout()
    common_tools.save_figure(
        fig, output, f"transform_comparison_{safe_slug(dataset)}"
    )


def choose_diagnostic_protocol(mapping: dict[tuple[float, float], tesstv.StepTest], common: set[tuple[float, float]]) -> tesstv.StepTest:
    if (0.1, 5.0) in common:
        return mapping[(0.1, 5.0)]
    return mapping[sorted(common, key=baseline.protocol_sort_key)[0]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apr", required=True)
    parser.add_argument("--jun", required=True)
    parser.add_argument("--jul", required=True)
    parser.add_argument("--output", default="fft_stft_dwt_cwt_comparison")
    parser.add_argument("--step-sheet")
    parser.add_argument("--processing-window-s", type=float, default=1.0)
    parser.add_argument("--max-grid-points", type=int, default=4096)
    parser.add_argument("--cutoff-hz", type=float, default=None)
    parser.add_argument("--cutoff-duration-factor", type=float, default=1.0)
    parser.add_argument("--taper-ratio", type=float, default=0.5)
    parser.add_argument("--stft-nperseg", type=int, default=256)
    parser.add_argument("--stft-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--dwt-wavelet", default="db4")
    parser.add_argument("--dwt-level", type=int, default=5)
    parser.add_argument("--dwt-threshold-scale", type=float, default=1.0)
    parser.add_argument("--cwt-wavelet", default="morlet")
    parser.add_argument("--cwt-voices", type=int, default=32)
    parser.add_argument("--cwt-threshold-scale", type=float, default=1.0)
    parser.add_argument("--clip-margin-fraction", type=float, default=0.10)
    parser.add_argument(
        "--skip-cwt",
        action="store_true",
        help="Development fallback only; full project comparison should include CWT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.processing_window_s <= 0:
        raise ValueError("--processing-window-s must be positive.")
    if args.max_grid_points < 256:
        raise ValueError("--max-grid-points must be at least 256.")
    if not 0 <= args.stft_overlap_fraction < 1:
        raise ValueError("--stft-overlap-fraction must be in [0, 1).")

    workbooks = {
        "APR18": Path(args.apr).expanduser().resolve(),
        "JUN18": Path(args.jun).expanduser().resolve(),
        "JUL18": Path(args.jul).expanduser().resolve(),
    }
    missing = [f"{label}: {path}" for label, path in workbooks.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Workbook(s) not found:\n  " + "\n  ".join(missing))

    config = Config(
        processing_window_s=args.processing_window_s,
        max_grid_points=args.max_grid_points,
        cutoff_hz=args.cutoff_hz,
        cutoff_duration_factor=args.cutoff_duration_factor,
        taper_ratio=args.taper_ratio,
        stft_nperseg=args.stft_nperseg,
        stft_overlap_fraction=args.stft_overlap_fraction,
        dwt_wavelet=args.dwt_wavelet,
        dwt_level=args.dwt_level,
        dwt_threshold_scale=args.dwt_threshold_scale,
        cwt_wavelet=args.cwt_wavelet,
        cwt_voices=args.cwt_voices,
        cwt_threshold_scale=args.cwt_threshold_scale,
        clip_margin_fraction=args.clip_margin_fraction,
    )
    methods = [METHOD_RAW, METHOD_FFT, METHOD_STFT, METHOD_DWT]
    if not args.skip_cwt:
        methods.append(METHOD_CWT)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common_tools.enable_relaxed_state_cache()
    tests, common, protocol_audit = common_tools.load_common_tests(
        workbooks, args.step_sheet
    )
    protocol_audit.to_csv(output / "transform_protocol_audit.csv", index=False)
    common_text = "; ".join(
        protocol_name(key) for key in sorted(common, key=baseline.protocol_sort_key)
    )

    simulations: dict[
        tuple[str, str], dict[tuple[float, float], pd.DataFrame]
    ] = {}
    processing_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    lambda_rows: list[dict[str, object]] = []

    for dataset, mapping in tests.items():
        dataset_dir = output / safe_slug(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            simulations[(dataset, method)] = {}
        for key in sorted(common, key=baseline.protocol_sort_key):
            source = mapping[key]
            method_frames: dict[str, pd.DataFrame] = {}
            raw_shear = source.data["shear_rate"].to_numpy(float).copy()
            for method in methods:
                test, metadata, _ = process_test(source, method, config)
                result = tesstv.simulate_step(test, tesstv.PAPER_PARAMS)
                simulations[(dataset, method)][key] = result
                method_frames[method] = result
                processing_rows.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol_name(key),
                        "initial_rate_s-1": key[0],
                        "target_rate_s-1": key[1],
                        "source_excel_column_one_based": source.source_column + 1,
                        "source_title": source.title,
                        **metadata,
                    }
                )
                protocol_rows.append(
                    common_tools.score_protocol(dataset, method, key, result)
                )

            raw_lambda = method_frames[METHOD_RAW]["lambda"].to_numpy(float)
            for method in methods:
                current = method_frames[method]["lambda"].to_numpy(float)
                delta = current - raw_lambda
                lambda_rows.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol_name(key),
                        "method": method,
                        "lambda_rmse_change_vs_raw": float(np.sqrt(np.mean(delta**2))),
                        "lambda_max_abs_change_vs_raw": float(np.max(np.abs(delta))),
                        "lambda_final_change_vs_raw": float(delta[-1]),
                    }
                )

            detail = pd.DataFrame(
                {
                    "time_s": method_frames[METHOD_RAW]["time"],
                    "raw_measured_stress_Pa": method_frames[METHOD_RAW]["stress_data"],
                    "raw_measured_shear_rate_s-1": raw_shear,
                }
            )
            for method in methods:
                detail[f"{method}_input_shear_rate_s-1"] = method_frames[method]["shear_rate"]
                detail[f"{method}_predicted_stress_Pa"] = method_frames[method]["stress_model"]
                detail[f"{method}_lambda"] = method_frames[method]["lambda"]
            detail.to_csv(
                dataset_dir / f"transform_{safe_slug(protocol_name(key))}.csv",
                index=False,
            )

    processing_audit = pd.DataFrame(processing_rows)
    per_protocol = common_tools.add_improvements(
        pd.DataFrame(protocol_rows), ["dataset", "protocol"]
    )
    lambda_metrics = pd.DataFrame(lambda_rows)
    summary = common_tools.aggregate_results(simulations, processing_audit)
    summary.insert(summary.columns.get_loc("protocols") + 1, "common_protocols", common_text)

    summary.to_csv(output / "fft_stft_dwt_cwt_summary.csv", index=False)
    per_protocol.to_csv(output / "fft_stft_dwt_cwt_per_protocol.csv", index=False)
    processing_audit.to_csv(output / "fft_stft_dwt_cwt_processing_audit.csv", index=False)
    lambda_metrics.to_csv(output / "fft_stft_dwt_cwt_lambda_metrics.csv", index=False)
    plot_summary(summary, output, methods)
    for dataset, mapping in tests.items():
        plot_transform_comparison(
            dataset,
            choose_diagnostic_protocol(mapping, common),
            config,
            output,
            include_cwt=not args.skip_cwt,
        )

    print("\nFFT/STFT/DWT/CWT comparison completed")
    print(f"Output: {output}")
    print(f"Common protocols ({len(common)}): {common_text}")
    print("Stress target: raw measured stress, unchanged")
    print("Parameters: published Donor 5 values, fixed")
    print(
        summary[
            [
                "dataset",
                "method",
                "rmse",
                "r2",
                "fcost_trans_eq12_normalized_rms",
                "eq12_improvement_percent_vs_raw",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
