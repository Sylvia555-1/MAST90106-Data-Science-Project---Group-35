"""Shared, truth-free calculations for the offline DWT study.

This module never reads simulation labels, latent moduli, or random seeds.
All detail signals are reconstructed on the original sampling grid.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/coagulation_dwt_mpl")
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MAIN_FEATURES = ["R6", "H13", "P6", "V6"]
FEATURE_UNITS = {"R6": "Pa", "H13": "Pa", "P6": "fraction", "V6": "Pa/s"}


def load_config():
    """Read the fixed analysis settings, without accessing simulation truth."""
    config = json.loads((ROOT / "config.json").read_text())
    config["source_dir"] = os.environ.get("DWT_NOISE_DATA", str((ROOT / config["source_dir"]).resolve()))
    return config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    """Reject non-standard JSON NaN values rather than hiding invalid checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_record(case, config):
    """Validate one complete CSV, preserving the original time and amplitude."""
    frame = pd.read_csv(Path(config["source_dir"]) / f"{case}.csv")
    columns = ["time_s", "strain", "shear_rate_1_s", "stress_Pa"]
    if list(frame.columns) != columns:
        raise ValueError(f"Unexpected columns in {case}: {list(frame.columns)}")
    expected_n = int(config["duration_s"] * config["sampling_hz"])
    if len(frame) != expected_n or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Invalid length or non-finite value in {case}")
    expected_time = np.arange(expected_n) / config["sampling_hz"]
    if not np.allclose(frame.time_s, expected_time, rtol=0, atol=3e-12):
        raise ValueError(f"Missing, duplicate, or irregular time samples in {case}")
    return frame


def decompose(signal, config):
    """Return A8 and D1...D8; their sum reconstructs the supplied signal."""
    signal = np.asarray(signal, dtype=float)
    level, wavelet, mode = config["level"], config["wavelet"], config["mode"]
    if level > pywt.dwt_max_level(len(signal), wavelet):
        raise ValueError("Record is too short for the requested decomposition")
    coefficients = pywt.wavedec(signal, wavelet, mode=mode, level=level)
    output = {}
    # Retain one coefficient block at a time and set the others to zero.
    names = [f"A{level}"] + [f"D{j}" for j in range(level, 0, -1)]
    for index, name in enumerate(names):
        selected = [np.zeros_like(c) for c in coefficients]
        selected[index] = coefficients[index]
        output[name] = pywt.waverec(selected, wavelet, mode=mode)[:len(signal)]
    return output


def window_slices(time, config, window_s):
    """Anchor full windows to absolute time zero, including after a crop.

    No interpolation or circular shift is used. Incomplete windows are omitted.
    """
    time = np.asarray(time)
    fs = config["sampling_hz"]
    indices = np.rint(time * fs).astype(np.int64)
    if not np.allclose(time, indices / fs, rtol=0, atol=3e-12):
        raise ValueError("Time values do not lie on the original sample grid")
    if len(time) < 2 or not np.all(np.diff(indices) == 1):
        raise ValueError("Windowing requires contiguous uniform samples")
    size = int(round(window_s * fs))
    if size < 1 or not np.isclose(size, window_s * fs):
        raise ValueError("Window duration must contain an integer sample count")
    first = ((indices[0] + size - 1) // size) * size
    stop = ((indices[-1] + 1) // size) * size
    return [(start / fs, (start + size) / fs,
             slice(int(start - indices[0]), int(start + size - indices[0])))
            for start in range(first, stop, size)]


def extract_features(time, signal, components, config, window_s=None):
    """Compute interpretable window features without labels or denoising.

    H13 is a combined activity index, not a noise estimate.
    P6 is a ratio among selected detail energies, not total signal energy.
    """
    window_s = config["window_s"] if window_s is None else window_s
    windows = window_slices(time, config, window_s)
    rows = []
    first_time = float(time[0])
    record_end = float(time[-1] + 1 / config["sampling_hz"])
    for start, end, selection in windows:
        energy = np.array([np.mean(components[f"D{j}"][selection] ** 2)
                           for j in range(1, config["level"] + 1)])
        boundary = (start < first_time + config["boundary_s"] - 1e-9 or
                    end > record_end - config["boundary_s"] + 1e-9)
        row = {"start_s": start, "end_s": end, "mid_s": (start + end) / 2,
               "boundary_window": boundary,
               "reference_window": (start >= config["reference_start_s"] and
                                    end <= config["reference_end_s"]),
               "raw_rms": float(np.sqrt(np.mean(signal[selection] ** 2)))}
        row.update({f"E{j}": value for j, value in enumerate(energy, 1)})
        row.update({f"R{j}": np.sqrt(value) for j, value in enumerate(energy, 1)})
        row["H13"] = np.sqrt(energy[:3].sum())
        row["P6"] = energy[5] / energy.sum() if energy.sum() > 0 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No complete windows are available")
    table["V6"] = table.R6.diff() / table.mid_s.diff()
    # A difference uses two windows, so both must avoid the boundary zone.
    invalid = table.boundary_window | table.boundary_window.shift(1, fill_value=True)
    table.loc[invalid, "V6"] = np.nan
    return table


def harmonic_reference(frame, config, window_s=None):
    """Fit stress = G' * strain + G'' * rate/omega + offset in each window.

    Fit residuals can include model mismatch, not only random noise.
    """
    duration = config["window_s"] if window_s is None else window_s
    omega = 2 * np.pi * config["frequency_hz"]
    rows = []
    for start, end, selection in window_slices(frame.time_s.to_numpy(), config, duration):
        part = frame.iloc[selection]
        design = np.column_stack([part.strain, part.shear_rate_1_s / omega, np.ones(len(part))])
        parameters, _, rank, _ = np.linalg.lstsq(design, part.stress_Pa, rcond=None)
        if rank != 3:
            raise ValueError("Harmonic regression is rank deficient")
        residual = part.stress_Pa - design @ parameters
        rows.append({"start_s": start, "end_s": end, "mid_s": (start + end) / 2,
                     "Gprime_Pa": parameters[0], "Gdoubleprime_Pa": parameters[1],
                     "offset_Pa": parameters[2], "residual_rms_Pa": np.sqrt(np.mean(residual ** 2))})
    return pd.DataFrame(rows)


def shade_boundaries(axes, config):
    """Mark conservative boundary regions on full-record plots."""
    for ax in np.asarray(axes).ravel():
        ax.axvspan(0, config["boundary_s"], color="gray", alpha=0.15)
        ax.axvspan(config["duration_s"] - config["boundary_s"], config["duration_s"], color="gray", alpha=0.15)
        ax.grid(alpha=0.2)


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
