#!/usr/bin/env python3
"""Compare raw, Fourier-filtered and wavelet-denoised t-ESSTV inputs.

This is the second-stage sensitivity experiment that follows
``compare_tesstv_raw_baselines.py``.  It keeps the published Donor 5 t-ESSTV
parameters fixed and changes only the measured shear-rate history supplied to
the model.  Every prediction is evaluated against the same *raw measured
stress*.  The script therefore tests the contribution of input-signal
processing without mixing it with parameter fitting or target smoothing.

Important sampling detail
-------------------------
The TA Instruments step records are strongly non-uniform in physical time.
FFT and discrete wavelet transforms require uniform sampling, so each
shear-rate history is first linearly interpolated to a uniform physical-time
grid.  The filtered signal is then interpolated back to the original measured
times before the t-ESSTV ODE is solved.  The original workbook is never edited.

Keep this file beside:

* tensorial_esstv_figure5_v3.py
* compare_tesstv_raw_baselines.py

Example
-------
python3 compare_tesstv_signal_preprocessing.py \
  --apr "APR18 SS SAOS STEPS LAOS UDLAOS.xlsx" \
  --jun "JUN18.xlsx" \
  --jul "JUL18.xlsx" \
  --output "three_dataset_signal_preprocessing"

The dependency-free wavelet default is a Haar DWT.  To reproduce a db4 DWT,
install PyWavelets and add ``--wavelet db4``.
"""

from __future__ import annotations

import argparse
import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tensorial_esstv_figure5_v3 as tesstv
    import compare_tesstv_raw_baselines as baseline
except ImportError as exc:
    raise SystemExit(
        "Put tensorial_esstv_figure5_v3.py and "
        "compare_tesstv_raw_baselines.py beside this script."
    ) from exc


EPS = 1.0e-12
METHOD_RAW = "raw"
METHOD_FOURIER = "fourier"


@dataclass(frozen=True)
class ProcessingConfig:
    fourier_cutoff_hz: float | None
    fourier_duration_factor: float
    fourier_taper_ratio: float
    wavelet: str
    wavelet_level: int
    wavelet_threshold_scale: float
    max_grid_points: int
    clip_margin_fraction: float


@dataclass
class ProcessedSignal:
    values: np.ndarray
    metadata: dict[str, object]


def enable_relaxed_state_cache() -> None:
    """Cache identical pre-shear equilibrations without changing the model."""

    original = tesstv.relaxed_steady_state

    @functools.lru_cache(maxsize=None)
    def cached(rate: float, parameters: tesstv.Params) -> tuple[float, ...]:
        return tuple(original(rate, parameters).tolist())

    def wrapper(rate: float, parameters: tesstv.Params) -> np.ndarray:
        return np.asarray(cached(float(rate), parameters), dtype=float).copy()

    tesstv.relaxed_steady_state = wrapper


def safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return value or "item"


def protocol_name(key: tuple[float, float]) -> str:
    return f"{key[0]:g}->{key[1]:g}"


def validate_time_signal(time: np.ndarray, signal: np.ndarray) -> None:
    if len(time) < 8 or len(time) != len(signal):
        raise ValueError("A step signal needs at least eight aligned samples.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("Time and shear-rate values must be finite.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Time must be strictly increasing.")


def lower_power_of_two(value: int) -> int:
    if value < 2:
        return 1
    return 1 << (int(value).bit_length() - 1)


def upper_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value - 1).bit_length())


def uniform_resample(
    time: np.ndarray,
    signal: np.ndarray,
    max_grid_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Interpolate a non-uniform signal to a uniform physical-time grid."""

    validate_time_signal(time, signal)
    dt = np.diff(time)
    reference_dt = float(np.quantile(dt, 0.10))
    span = float(time[-1] - time[0])
    requested = max(256, int(math.ceil(span / reference_dt)) + 1)
    cap = lower_power_of_two(max(256, int(max_grid_points)))
    n_grid = min(upper_power_of_two(requested), cap)
    grid = np.linspace(time[0], time[-1], n_grid)
    values = np.interp(grid, time, signal)
    grid_dt = float(grid[1] - grid[0])
    metadata: dict[str, float | int] = {
        "original_n_points": int(len(time)),
        "uniform_n_points": int(n_grid),
        "original_dt_min_s": float(np.min(dt)),
        "original_dt_median_s": float(np.median(dt)),
        "original_dt_max_s": float(np.max(dt)),
        "uniform_dt_s": grid_dt,
        "uniform_nyquist_hz": float(0.5 / grid_dt),
    }
    return grid, values, metadata


def first_crossing_time(
    time: np.ndarray, progress: np.ndarray, level: float
) -> float | None:
    indices = np.flatnonzero(progress >= level)
    if not len(indices):
        return None
    index = int(indices[0])
    if index == 0:
        return float(time[0])
    x0, x1 = float(progress[index - 1]), float(progress[index])
    t0, t1 = float(time[index - 1]), float(time[index])
    if abs(x1 - x0) <= EPS:
        return t1
    weight = np.clip((level - x0) / (x1 - x0), 0.0, 1.0)
    return float(t0 + weight * (t1 - t0))


def estimate_transition_duration(
    time: np.ndarray,
    shear: np.ndarray,
    pre_rate: float,
    target_rate: float,
) -> float:
    """Estimate the measured 10-90 percent transition duration."""

    change = float(target_rate - pre_rate)
    if abs(change) <= EPS:
        return float(np.median(np.diff(time)))
    progress = np.clip((shear - pre_rate) / change, 0.0, 1.0)
    t10 = first_crossing_time(time, progress, 0.10)
    t90 = first_crossing_time(time, progress, 0.90)
    fallback = max(5.0 * float(np.min(np.diff(time))), 0.02)
    if t10 is None or t90 is None or t90 <= t10:
        return fallback
    return max(float(t90 - t10), float(np.min(np.diff(time))))


def stable_clip(
    values: np.ndarray,
    pre_rate: float,
    target_rate: float,
    margin_fraction: float,
) -> np.ndarray:
    """Prevent spectral ringing from creating nonphysical rate excursions."""

    low = min(pre_rate, target_rate)
    high = max(pre_rate, target_rate)
    margin = max((high - low) * margin_fraction, 0.03)
    return np.clip(values, max(0.0, low - margin), high + margin)


def fourier_filter(
    time: np.ndarray,
    shear: np.ndarray,
    pre_rate: float,
    target_rate: float,
    config: ProcessingConfig,
) -> ProcessedSignal:
    grid, uniform, metadata = uniform_resample(
        time, shear, config.max_grid_points
    )
    grid_dt = float(metadata["uniform_dt_s"])
    nyquist = float(metadata["uniform_nyquist_hz"])
    duration = estimate_transition_duration(time, shear, pre_rate, target_rate)
    if config.fourier_cutoff_hz is None:
        cutoff = config.fourier_duration_factor / max(duration, grid_dt)
        cutoff_rule = "duration_factor/10-90%_transition_duration"
    else:
        cutoff = float(config.fourier_cutoff_hz)
        cutoff_rule = "fixed_user_cutoff_hz"
    cutoff = float(np.clip(cutoff, 1.0 / max(time[-1] - time[0], EPS), 0.90 * nyquist))

    pad = max(128, min(len(uniform) // 8, 4096))
    padded = np.pad(uniform, (pad, pad), mode="edge")
    frequencies = np.fft.rfftfreq(len(padded), d=grid_dt)
    spectrum = np.fft.rfft(padded)

    stop = min(cutoff * (1.0 + config.fourier_taper_ratio), nyquist)
    transfer = np.ones_like(frequencies)
    transfer[frequencies >= stop] = 0.0
    transition = (frequencies > cutoff) & (frequencies < stop)
    if stop > cutoff:
        phase = (frequencies[transition] - cutoff) / (stop - cutoff)
        transfer[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    filtered_uniform = np.fft.irfft(spectrum * transfer, n=len(padded))[pad:-pad]
    filtered = np.interp(time, grid, filtered_uniform)
    filtered = stable_clip(
        filtered, pre_rate, target_rate, config.clip_margin_fraction
    )

    metadata.update(
        {
            "method": METHOD_FOURIER,
            "transition_duration_10_90_s": duration,
            "fourier_cutoff_hz": cutoff,
            "fourier_cutoff_rule": cutoff_rule,
            "fourier_taper_stop_hz": stop,
            "wavelet": "not_applicable",
            "wavelet_level": np.nan,
            "wavelet_sigma": np.nan,
            "wavelet_threshold": np.nan,
        }
    )
    return ProcessedSignal(filtered, metadata)


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def haar_denoise(
    values: np.ndarray,
    level: int,
    threshold_scale: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Dependency-free orthonormal Haar DWT with soft thresholding."""

    approximation = np.asarray(values, dtype=float).copy()
    details: list[np.ndarray] = []
    maximum_level = int(math.log2(len(approximation)))
    actual_level = max(1, min(int(level), maximum_level))
    root_two = math.sqrt(2.0)
    for _ in range(actual_level):
        even = approximation[0::2]
        odd = approximation[1::2]
        details.append((even - odd) / root_two)
        approximation = (even + odd) / root_two

    finest = details[0]
    sigma = float(np.median(np.abs(finest - np.median(finest))) / 0.67448975)
    threshold = float(
        threshold_scale * sigma * math.sqrt(2.0 * math.log(len(values)))
    )
    shrunk = [soft_threshold(detail, threshold) for detail in details]
    reconstruction = approximation
    for detail in reversed(shrunk):
        restored = np.empty(detail.size * 2, dtype=float)
        restored[0::2] = (reconstruction + detail) / root_two
        restored[1::2] = (reconstruction - detail) / root_two
        reconstruction = restored
    return reconstruction[: len(values)], {
        "wavelet": "haar",
        "wavelet_level": actual_level,
        "wavelet_sigma": sigma,
        "wavelet_threshold": threshold,
    }


def pywavelets_denoise(
    values: np.ndarray,
    wavelet: str,
    level: int,
    threshold_scale: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    try:
        import pywt  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Wavelet {wavelet!r} requires PyWavelets. Install it with: "
            "python3 -m pip install PyWavelets; or use --wavelet haar."
        ) from exc

    maximum_level = pywt.dwt_max_level(len(values), pywt.Wavelet(wavelet).dec_len)
    actual_level = max(1, min(int(level), int(maximum_level)))
    coefficients = pywt.wavedec(values, wavelet, mode="symmetric", level=actual_level)
    finest = np.asarray(coefficients[-1], dtype=float)
    sigma = float(np.median(np.abs(finest - np.median(finest))) / 0.67448975)
    threshold = float(
        threshold_scale * sigma * math.sqrt(2.0 * math.log(len(values)))
    )
    filtered_coefficients = [
        coefficients[0],
        *[pywt.threshold(detail, threshold, mode="soft") for detail in coefficients[1:]],
    ]
    reconstruction = pywt.waverec(filtered_coefficients, wavelet, mode="symmetric")
    return np.asarray(reconstruction[: len(values)], dtype=float), {
        "wavelet": wavelet,
        "wavelet_level": actual_level,
        "wavelet_sigma": sigma,
        "wavelet_threshold": threshold,
    }


def wavelet_filter(
    time: np.ndarray,
    shear: np.ndarray,
    pre_rate: float,
    target_rate: float,
    config: ProcessingConfig,
) -> ProcessedSignal:
    grid, uniform, metadata = uniform_resample(
        time, shear, config.max_grid_points
    )
    if config.wavelet.lower() == "haar":
        filtered_uniform, wavelet_metadata = haar_denoise(
            uniform, config.wavelet_level, config.wavelet_threshold_scale
        )
    else:
        filtered_uniform, wavelet_metadata = pywavelets_denoise(
            uniform,
            config.wavelet,
            config.wavelet_level,
            config.wavelet_threshold_scale,
        )
    filtered = np.interp(time, grid, filtered_uniform)
    filtered = stable_clip(
        filtered, pre_rate, target_rate, config.clip_margin_fraction
    )
    metadata.update(
        {
            "method": f"wavelet_{config.wavelet.lower()}",
            "transition_duration_10_90_s": estimate_transition_duration(
                time, shear, pre_rate, target_rate
            ),
            "fourier_cutoff_hz": np.nan,
            "fourier_cutoff_rule": "not_applicable",
            "fourier_taper_stop_hz": np.nan,
            **wavelet_metadata,
        }
    )
    return ProcessedSignal(filtered, metadata)


def processed_test(
    source: tesstv.StepTest,
    method: str,
    config: ProcessingConfig,
) -> tuple[tesstv.StepTest, dict[str, object]]:
    data = source.data.copy()
    time = data["time"].to_numpy(float)
    raw_shear = data["shear_rate"].to_numpy(float)
    raw_stress = data["stress"].to_numpy(float).copy()

    if method == METHOD_RAW:
        output = raw_shear.copy()
        dt = np.diff(time)
        metadata: dict[str, object] = {
            "method": METHOD_RAW,
            "original_n_points": len(time),
            "uniform_n_points": np.nan,
            "original_dt_min_s": float(np.min(dt)),
            "original_dt_median_s": float(np.median(dt)),
            "original_dt_max_s": float(np.max(dt)),
            "uniform_dt_s": np.nan,
            "uniform_nyquist_hz": np.nan,
            "transition_duration_10_90_s": estimate_transition_duration(
                time, raw_shear, source.pre_rate, source.target_rate
            ),
            "fourier_cutoff_hz": np.nan,
            "fourier_cutoff_rule": "not_applicable",
            "fourier_taper_stop_hz": np.nan,
            "wavelet": "not_applicable",
            "wavelet_level": np.nan,
            "wavelet_sigma": np.nan,
            "wavelet_threshold": np.nan,
        }
    elif method == METHOD_FOURIER:
        processed = fourier_filter(
            time, raw_shear, source.pre_rate, source.target_rate, config
        )
        output, metadata = processed.values, processed.metadata
    elif method.startswith("wavelet_"):
        processed = wavelet_filter(
            time, raw_shear, source.pre_rate, source.target_rate, config
        )
        output, metadata = processed.values, processed.metadata
    else:
        raise ValueError(f"Unknown processing method: {method}")

    data["shear_rate"] = output
    if not np.array_equal(data["stress"].to_numpy(float), raw_stress):
        raise AssertionError("Signal processing changed the measured stress target.")
    delta = output - raw_shear
    metadata.update(
        {
            "shear_rate_rmse_change_s-1": float(np.sqrt(np.mean(delta**2))),
            "shear_rate_mae_change_s-1": float(np.mean(np.abs(delta))),
            "shear_rate_max_abs_change_s-1": float(np.max(np.abs(delta))),
            "raw_stress_target_changed": False,
            "model_parameters_changed": False,
        }
    )
    return (
        tesstv.StepTest(
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
        ),
        metadata,
    )


def load_common_tests(
    workbooks: dict[str, Path], step_sheet: str | None
) -> tuple[
    dict[str, dict[tuple[float, float], tesstv.StepTest]],
    set[tuple[float, float]],
    pd.DataFrame,
]:
    by_dataset: dict[str, dict[tuple[float, float], tesstv.StepTest]] = {}
    audit_parts: list[pd.DataFrame] = []
    for label, workbook in workbooks.items():
        all_tests = tesstv.read_step_tests(workbook, step_sheet)
        selected = baseline.choose_unique_measured_protocols(all_tests)
        mapping = {(float(test.pre_rate), float(test.target_rate)): test for test in selected}
        by_dataset[label] = mapping
        audit = tesstv.protocol_audit_table(all_tests, selected)
        audit.insert(0, "dataset", label)
        audit_parts.append(audit)

    common = set.intersection(*(set(mapping) for mapping in by_dataset.values()))
    if not common:
        raise ValueError("APR18, JUN18 and JUL18 have no common measured step protocols.")
    return by_dataset, common, pd.concat(audit_parts, ignore_index=True)


def score_protocol(
    dataset: str,
    method: str,
    key: tuple[float, float],
    result: pd.DataFrame,
) -> dict[str, object]:
    score = tesstv.metrics(result["stress_data"], result["stress_model"])
    cost = tesstv.transient_costs_for_frame(result)
    residual = result["stress_model"].to_numpy(float) - result["stress_data"].to_numpy(float)
    maximum_index = int(np.argmax(np.abs(residual)))
    return {
        "dataset": dataset,
        "method": method,
        "protocol": protocol_name(key),
        "initial_rate_s-1": key[0],
        "target_rate_s-1": key[1],
        "direction": "step_up" if key[1] > key[0] else "step_down",
        **score,
        **cost,
        "max_abs_residual_Pa": float(np.max(np.abs(residual))),
        "time_of_max_abs_residual_s": float(result["time"].iloc[maximum_index]),
    }


def add_improvements(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    raw = output.loc[
        output["method"] == METHOD_RAW,
        [*group_columns, "rmse", "fcost_trans_eq12_normalized_rms", "r2"],
    ].rename(
        columns={
            "rmse": "_raw_rmse",
            "fcost_trans_eq12_normalized_rms": "_raw_eq12",
            "r2": "_raw_r2",
        }
    )
    output = output.merge(raw, on=group_columns, how="left", validate="many_to_one")
    output["rmse_improvement_percent_vs_raw"] = 100.0 * (
        output["_raw_rmse"] - output["rmse"]
    ) / output["_raw_rmse"]
    output["eq12_improvement_percent_vs_raw"] = 100.0 * (
        output["_raw_eq12"] - output["fcost_trans_eq12_normalized_rms"]
    ) / output["_raw_eq12"]
    output["r2_change_vs_raw"] = (
        output["r2"] - output["_raw_r2"]
    )
    return output.drop(columns=["_raw_rmse", "_raw_eq12", "_raw_r2"])


def aggregate_results(
    simulations: dict[tuple[str, str], dict[tuple[float, float], pd.DataFrame]],
    audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, method), frames in simulations.items():
        score = baseline.combined_metrics(frames)
        cost = tesstv.aggregate_transient_costs(frames)
        method_audit = audit.loc[
            (audit["dataset"] == dataset) & (audit["method"] == method)
        ]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "parameter_source": "Armstrong et al. Table 6 Donor 5; fixed",
                "parameters_optimised": False,
                "stress_target": "raw measured stress; unchanged",
                "n_protocols": int(cost["n_protocols"]),
                "n_points": int(cost["n_points"]),
                "protocols": cost["protocols"],
                **score,
                **cost,
                "paper_donor5_transient_cost_reference": tesstv.PAPER_FCOST_TRANS,
                "mean_shear_rate_rmse_change_s-1": float(
                    method_audit["shear_rate_rmse_change_s-1"].mean()
                ),
                "max_shear_rate_abs_change_s-1": float(
                    method_audit["shear_rate_max_abs_change_s-1"].max()
                ),
            }
        )
    return add_improvements(pd.DataFrame(rows), ["dataset"])


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    datasets = list(dict.fromkeys(summary["dataset"]))
    methods = list(dict.fromkeys(summary["method"]))
    colors = {METHOD_RAW: "#7F7F7F", METHOD_FOURIER: "#4472C4"}
    colors.update({method: "#70AD47" for method in methods if method.startswith("wavelet_")})
    x = np.arange(len(datasets), dtype=float)
    width = 0.24
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("fcost_trans_eq12_normalized_rms", "Eq. 12 normalized cost", "cost"),
        ("rmse", "Transient RMSE", "Pa"),
        ("r2", "Transient R²", "R²"),
        ("eq12_improvement_percent_vs_raw", "Eq. 12 improvement vs raw", "%"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel(), specs):
        for offset, method in zip(offsets, methods):
            part = summary.loc[summary["method"] == method].set_index("dataset")
            values = part.reindex(datasets)[column].to_numpy(float)
            axis.bar(x + offset, values, width=width, label=method, color=colors[method])
        axis.axhline(0.0, color="black", linewidth=0.8)
        if column == "fcost_trans_eq12_normalized_rms":
            axis.axhline(
                tesstv.PAPER_FCOST_TRANS,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="paper Donor 5 reference",
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, datasets)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Fixed Donor 5 t-ESSTV: raw versus processed shear-rate input",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output, "signal_preprocessing_summary")


def plot_lambda_summary(lambda_metrics: pd.DataFrame, output: Path) -> None:
    processed = lambda_metrics.loc[lambda_metrics["method"] != METHOD_RAW].copy()
    if processed.empty:
        return
    labels = (
        processed["dataset"].astype(str)
        + " | "
        + processed["method"].astype(str)
        + " | "
        + processed["protocol"].astype(str)
    )
    fig, axis = plt.subplots(figsize=(12, max(5, 0.28 * len(processed))))
    positions = np.arange(len(processed))
    axis.barh(positions, processed["lambda_max_abs_change_vs_raw"], color="#8064A2")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("maximum |λprocessed - λraw|")
    axis.set_title("Change in inferred structure caused only by shear-rate preprocessing")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output, "lambda_change_summary")


def choose_examples(keys: set[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(keys, key=baseline.protocol_sort_key)
    desired = [(0.1, 5.0), (5.0, 0.5)]
    selected = [key for target in desired for key in ordered if key == target]
    if len(selected) < 2:
        selected.extend(key for key in ordered if key not in selected)
    return selected[:2]


def plot_dataset_traces(
    dataset: str,
    keys: set[tuple[float, float]],
    methods: list[str],
    simulations: dict[tuple[str, str], dict[tuple[float, float], pd.DataFrame]],
    raw_shear: dict[tuple[str, tuple[float, float]], np.ndarray],
    output: Path,
) -> None:
    examples = choose_examples(keys)
    colors = {METHOD_RAW: "#7F7F7F", METHOD_FOURIER: "#4472C4"}
    colors.update({method: "#70AD47" for method in methods if method.startswith("wavelet_")})
    fig, axes = plt.subplots(3, len(examples), figsize=(12, 9), squeeze=False)
    for column, key in enumerate(examples):
        raw_frame = simulations[(dataset, METHOD_RAW)][key]
        x = raw_frame["time_plot"].to_numpy(float)
        axes[0, column].plot(
            x,
            raw_shear[(dataset, key)],
            "o",
            markersize=2.5,
            color="black",
            label="raw measured rate",
        )
        axes[1, column].plot(
            x,
            raw_frame["stress_data"],
            "o",
            markersize=3,
            markerfacecolor="none",
            color="black",
            label="raw measured stress",
        )
        for method in methods:
            frame = simulations[(dataset, method)][key]
            axes[0, column].plot(
                x, frame["shear_rate"], color=colors[method], label=method
            )
            axes[1, column].plot(
                x, frame["stress_model"], color=colors[method], label=method
            )
            axes[2, column].plot(
                x, frame["lambda"], color=colors[method], label=method
            )
        axes[0, column].set_title(f"{dataset}: {protocol_name(key)} s⁻¹")
        axes[0, column].set_ylabel("shear rate (s⁻¹)")
        axes[1, column].set_ylabel("stress (Pa)")
        axes[2, column].set_ylabel("λ")
        axes[2, column].set_xlabel("time (s)")
        for row in range(3):
            axes[row, column].set_xscale("log")
            axes[row, column].grid(alpha=0.20)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[2, 0].legend(fontsize=7)
    fig.suptitle(
        f"{dataset}: identical Donor 5 parameters and raw-stress target",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output, f"signal_traces_{safe_slug(dataset)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apr", required=True, help="Path to APR18 workbook")
    parser.add_argument("--jun", required=True, help="Path to JUN18 workbook")
    parser.add_argument("--jul", required=True, help="Path to JUL18 workbook")
    parser.add_argument(
        "--output", default="three_dataset_signal_preprocessing", help="Output folder"
    )
    parser.add_argument("--step-sheet", help="Override the detected steps sheet")
    parser.add_argument(
        "--fourier-cutoff-hz",
        type=float,
        default=None,
        help="Fixed low-pass cutoff. Default: 1/(measured 10-90%% transition duration).",
    )
    parser.add_argument(
        "--fourier-duration-factor", type=float, default=1.0,
        help="Automatic Fourier cutoff multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--fourier-taper-ratio", type=float, default=0.5,
        help="Raised-cosine transition width relative to cutoff (default: 0.5).",
    )
    parser.add_argument(
        "--wavelet", default="haar",
        help="DWT wavelet. haar needs no extra package; db4 needs PyWavelets.",
    )
    parser.add_argument("--wavelet-level", type=int, default=5)
    parser.add_argument("--wavelet-threshold-scale", type=float, default=1.0)
    parser.add_argument("--max-grid-points", type=int, default=65536)
    parser.add_argument("--clip-margin-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workbooks = {
        "APR18": Path(args.apr).expanduser().resolve(),
        "JUN18": Path(args.jun).expanduser().resolve(),
        "JUL18": Path(args.jul).expanduser().resolve(),
    }
    missing = [f"{label}: {path}" for label, path in workbooks.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Workbook(s) not found:\n  " + "\n  ".join(missing))
    if args.max_grid_points < 256:
        raise ValueError("--max-grid-points must be at least 256.")
    if args.wavelet_level < 1:
        raise ValueError("--wavelet-level must be at least 1.")

    config = ProcessingConfig(
        fourier_cutoff_hz=args.fourier_cutoff_hz,
        fourier_duration_factor=args.fourier_duration_factor,
        fourier_taper_ratio=args.fourier_taper_ratio,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        wavelet_threshold_scale=args.wavelet_threshold_scale,
        max_grid_points=args.max_grid_points,
        clip_margin_fraction=args.clip_margin_fraction,
    )
    methods = [METHOD_RAW, METHOD_FOURIER, f"wavelet_{args.wavelet.lower()}"]
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    enable_relaxed_state_cache()

    tests, common, protocol_audit = load_common_tests(workbooks, args.step_sheet)
    protocol_audit.to_csv(output / "signal_preprocessing_protocol_audit.csv", index=False)
    common_text = "; ".join(
        protocol_name(key) for key in sorted(common, key=baseline.protocol_sort_key)
    )

    simulations: dict[
        tuple[str, str], dict[tuple[float, float], pd.DataFrame]
    ] = {}
    processing_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    lambda_rows: list[dict[str, object]] = []
    raw_shear: dict[tuple[str, tuple[float, float]], np.ndarray] = {}

    for dataset, mapping in tests.items():
        dataset_dir = output / safe_slug(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            simulations[(dataset, method)] = {}

        for key in sorted(common, key=baseline.protocol_sort_key):
            source = mapping[key]
            raw_shear[(dataset, key)] = source.data["shear_rate"].to_numpy(float).copy()
            method_frames: dict[str, pd.DataFrame] = {}
            for method in methods:
                test, processing = processed_test(source, method, config)
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
                        **processing,
                    }
                )
                protocol_rows.append(score_protocol(dataset, method, key, result))

            raw_lambda = method_frames[METHOD_RAW]["lambda"].to_numpy(float)
            for method in methods:
                values = method_frames[method]["lambda"].to_numpy(float)
                delta = values - raw_lambda
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
                    "raw_measured_shear_rate_s-1": raw_shear[(dataset, key)],
                }
            )
            for method in methods:
                detail[f"{method}_input_shear_rate_s-1"] = method_frames[method]["shear_rate"]
                detail[f"{method}_predicted_stress_Pa"] = method_frames[method]["stress_model"]
                detail[f"{method}_lambda"] = method_frames[method]["lambda"]
            detail.to_csv(
                dataset_dir / f"signal_{safe_slug(protocol_name(key))}.csv", index=False
            )

    processing_audit = pd.DataFrame(processing_rows)
    per_protocol = add_improvements(pd.DataFrame(protocol_rows), ["dataset", "protocol"])
    lambda_metrics = pd.DataFrame(lambda_rows)
    summary = aggregate_results(simulations, processing_audit)
    summary.insert(summary.columns.get_loc("protocols") + 1, "common_protocols", common_text)

    summary.to_csv(output / "signal_preprocessing_summary.csv", index=False)
    per_protocol.to_csv(output / "signal_preprocessing_per_protocol.csv", index=False)
    processing_audit.to_csv(output / "signal_preprocessing_processing_audit.csv", index=False)
    lambda_metrics.to_csv(output / "signal_preprocessing_lambda_metrics.csv", index=False)

    plot_summary(summary, output)
    plot_lambda_summary(lambda_metrics, output)
    for dataset in workbooks:
        plot_dataset_traces(
            dataset, common, methods, simulations, raw_shear, output
        )

    print("\nFixed-parameter signal-processing comparison completed")
    print(f"Output: {output}")
    print(f"Common protocols ({len(common)}): {common_text}")
    print("Measured stress target: raw and unchanged for every method")
    display = summary[
        [
            "dataset",
            "method",
            "rmse",
            "r2",
            "fcost_trans_eq12_normalized_rms",
            "eq12_improvement_percent_vs_raw",
        ]
    ]
    print("\nSummary:")
    print(display.to_string(index=False))
    print(
        "\nPositive improvement percentages mean preprocessing reduced error. "
        "Negative values mean it made the model prediction worse."
    )


if __name__ == "__main__":
    main()
