"""Finite-window Morlet CWT analysis of the APR18 original LAOS data.

To run directly:
    python cwt_apr18_original.py

Dependencies:
    python -m pip install numpy pandas matplotlib openpyxl

Default input:
    E:\\pt\\dataset\\APR18 SS SAOS STEPS LAOS UDLAOS(1).xlsx

Alternatively, the workbook path can be passed as the first command-line argument, which is convenient for testing or changing the filename:
    python cwt_apr18_original.py "E:\\pt\\dataset\\APR18 SS SAOS STEPS LAOS UDLAOS.xlsx"

Output:
    A folder named `cwt_apr18_results` and a zip file of the same name in the directory where the workbook is located.

Analysis boundaries:
    The original APR18 LAOS dataset has no known transition times or condensation points. This script evaluates the CWT’s harmonic extraction,
    temporal stability, consistency with peer harmonic regression, and sensitivity to window length; it does not calculate detection delay.
"""

from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from numpy.lib.stride_tricks import sliding_window_view


DEFAULT_DATA_FILE = Path(
    r"E:\pt\dataset\APR18 SS SAOS STEPS LAOS UDLAOS.xlsx"
)
DATA_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_FILE
OUTPUT_DIR = DATA_FILE.parent / "cwt_apr18_results"

SHEET_NAME = "LAOS"
FS = 50.0
OMEGA = 1.0
F0 = OMEGA / (2.0 * np.pi)
HOP_S = 0.5
MORLET_B = 3.0
WINDOW_CYCLES = (2, 1)
FEATURES = (
    "I1", "I3", "I5", "I3_I1", "I5_I1", "power", "entropy", "bandwidth"
)
FEATURE_LABELS = ("I1", "I3", "I5", "I3/I1", "I5/I1", "Power", "Entropy", "Bandwidth")
HARMONICS = np.array([1, 3, 5])

# The starting columns and Gamma labels for the six sets of omega=1 data in the LAOS worksheet.
BLOCKS = ((0, 0.5), (4, 1.0), (8, 5.0), (12, 10.0), (16, 50.0), (20, 100.0))
COLORS = {"I1": "#007F86", "I3": "#D28A1A", "I5": "#9A4A94"}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
})


def save_csv(frame, name):
    frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig", float_format="%.10g")


def load_laos():
    """Read six sets of omega=1 LAOS data; remove the first start spike from each set and interpolate to 50 Hz."""
    sheet = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, header=None)
    signals = {}
    summary = []

    for start_col, gamma in BLOCKS:
        raw = sheet.iloc[4:, start_col:start_col + 3].copy()
        raw.columns = ["time_s", "stress_Pa", "strain_pct"]
        raw = raw.apply(pd.to_numeric, errors="coerce").dropna().iloc[1:].reset_index(drop=True)

        raw_t = raw["time_s"].to_numpy()
        time_s = np.arange(np.ceil(raw_t[0] * FS), np.floor(raw_t[-1] * FS) + 1) / FS
        stress = np.interp(time_s, raw_t, raw["stress_Pa"])
        strain = np.interp(time_s, raw_t, raw["strain_pct"])
        condition = f"gamma_{gamma:g}"
        signals[condition] = {"gamma": gamma, "time_s": time_s,
                              "stress_Pa": stress, "strain_pct": strain}
        summary.append({
            "condition": condition,
            "omega_rad_s": OMEGA,
            "gamma": gamma,
            "raw_rows_after_startup_drop": len(raw),
            "uniform_samples": len(time_s),
            "time_start_s": time_s[0],
            "time_end_s": time_s[-1],
            "sampling_rate_Hz": FS,
            "startup_rows_dropped": 1,
        })

    return signals, pd.DataFrame(summary)


def morlet_bank(n, frequencies):
    """Construct a finite-window central Morlet kernel and compute the positive-frequency amplitude response."""
    offsets = np.arange(n) - (n - 1) / 2.0
    scales = FS / np.asarray(frequencies)
    u = offsets[:, None] / scales[None, :]
    kernels = np.exp(-u * u / MORLET_B - 2j * np.pi * u)
    kernels /= np.sqrt(np.pi * MORLET_B * scales[None, :])
    kernels -= kernels.mean(axis=0, keepdims=True)
    response = np.abs(np.sum(kernels * np.exp(2j * np.pi * u), axis=0))
    return kernels, response


def make_estimator(cycles):
    n = round(cycles * FS / F0)
    fft_f = np.fft.rfftfreq(n, 1.0 / FS)
    band_f = fft_f[(fft_f >= 0.5 * F0) & (fft_f <= 8.0 * F0)]
    frequencies = np.r_[HARMONICS * F0, band_f]
    kernels, response = morlet_bank(n, frequencies)
    return {
        "cycles": cycles,
        "n": n,
        "band_f": band_f,
        "df": FS / n,
        "kernels": kernels,
        "response": response,
        "kernel_energy": np.sum(np.abs(kernels[:, 3:]) ** 2, axis=0),
    }


def estimate_cwt_features(windows, estimator):
    coefficients = np.asarray(windows) @ estimator["kernels"]
    amplitudes = 2.0 * np.abs(coefficients[:, :3]) / estimator["response"][:3]
    ratios = np.divide(
        amplitudes[:, 1:], amplitudes[:, :1],
        out=np.full_like(amplitudes[:, 1:], np.nan),
        where=amplitudes[:, :1] > 0,
    )

    spectrum = 2.0 * np.abs(coefficients[:, 3:]) ** 2
    spectrum /= FS * estimator["kernel_energy"]
    mass = spectrum * estimator["df"]
    power = mass.sum(axis=1)
    probability = np.divide(
        mass, power[:, None], out=np.full_like(mass, np.nan), where=power[:, None] > 0
    )
    log_probability = np.zeros_like(probability)
    np.log(probability, out=log_probability, where=probability > 0)
    entropy = -(probability * log_probability).sum(axis=1) / np.log(len(estimator["band_f"]))
    centroid = probability @ estimator["band_f"]
    bandwidth = np.sqrt(
        (probability * (estimator["band_f"] - centroid[:, None]) ** 2).sum(axis=1)
    )
    return np.column_stack((amplitudes, ratios, power, entropy, bandwidth))


def harmonic_projection(windows, n):
    """Perform a least-squares fit for the 1st, 3rd and 5th harmonics within the same window, to serve as an external consistency reference."""
    t = np.arange(n) / FS
    columns = [np.ones(n)]
    for harmonic in HARMONICS:
        columns.extend([
            np.cos(2.0 * np.pi * harmonic * F0 * t),
            np.sin(2.0 * np.pi * harmonic * F0 * t),
        ])
    design = np.column_stack(columns)
    coefficients = np.asarray(windows) @ np.linalg.pinv(design).T
    return np.column_stack([
        np.hypot(coefficients[:, 1], coefficients[:, 2]),
        np.hypot(coefficients[:, 3], coefficients[:, 4]),
        np.hypot(coefficients[:, 5], coefficients[:, 6]),
    ])


def extract_windows(signal, estimator):
    n = estimator["n"]
    hop = round(HOP_S * FS)
    windows = sliding_window_view(signal["stress_Pa"], n)[::hop]
    starts = np.arange(0, len(signal["stress_Pa"]) - n + 1, hop)
    values = estimate_cwt_features(windows, estimator)
    projection = harmonic_projection(windows, n)

    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "window_start_s", signal["time_s"][starts])
    frame.insert(1, "window_center_s", signal["time_s"][starts] + (n - 1) / (2 * FS))
    frame.insert(2, "window_end_s", signal["time_s"][starts] + n / FS)
    for i, harmonic in enumerate(HARMONICS):
        frame[f"projection_I{harmonic}"] = projection[:, i]
    return frame


def robust_summary(all_features):
    rows = []
    for (cycles, condition, gamma), group in all_features.groupby(
        ["cycles", "condition", "gamma"], sort=False
    ):
        for feature in FEATURES:
            values = group[feature].to_numpy()
            median = np.nanmedian(values)
            mad_scale = 1.4826 * np.nanmedian(np.abs(values - median))
            robust_cv = 100.0 * mad_scale / abs(median) if median != 0 else np.nan
            rows.append({
                "cycles": cycles,
                "condition": condition,
                "gamma": gamma,
                "feature": feature,
                "median": median,
                "mad_scale": mad_scale,
                "robust_cv_pct": robust_cv,
                "minimum": np.nanmin(values),
                "maximum": np.nanmax(values),
                "n_windows": len(values),
            })
    return pd.DataFrame(rows)


def method_agreement(all_features):
    rows = []
    for (cycles, condition, gamma), group in all_features.groupby(
        ["cycles", "condition", "gamma"], sort=False
    ):
        for harmonic in HARMONICS:
            cwt = group[f"I{harmonic}"].to_numpy()
            ref = group[f"projection_I{harmonic}"].to_numpy()
            window_difference = 100.0 * (cwt / ref - 1.0)
            cwt_median = np.median(cwt)
            ref_median = np.median(ref)
            rows.append({
                "cycles": cycles,
                "condition": condition,
                "gamma": gamma,
                "harmonic": f"I{harmonic}",
                "cwt_median_Pa": cwt_median,
                "projection_median_Pa": ref_median,
                "median_level_difference_pct": 100.0 * (cwt_median / ref_median - 1.0),
                "median_window_difference_pct": np.median(window_difference),
                "max_abs_window_difference_pct": np.max(np.abs(window_difference)),
            })
    return pd.DataFrame(rows)


def window_comparison(summary):
    two = summary[summary["cycles"] == 2].set_index(["condition", "gamma", "feature"])
    one = summary[summary["cycles"] == 1].set_index(["condition", "gamma", "feature"])
    rows = []
    for index in two.index:
        a, b = two.loc[index], one.loc[index]
        rows.append({
            "condition": index[0],
            "gamma": index[1],
            "feature": index[2],
            "two_cycle_median": a["median"],
            "one_cycle_median": b["median"],
            "one_minus_two_pct": 100.0 * (b["median"] / a["median"] - 1.0),
            "two_cycle_robust_cv_pct": a["robust_cv_pct"],
            "one_cycle_robust_cv_pct": b["robust_cv_pct"],
        })
    return pd.DataFrame(rows)


def calibration_experiment():
    """Quantify the estimation error in a finite window using a known harmonic mixed signal and an independent single-frequency signal."""
    amplitudes = np.array([1.0, 0.10, 0.03])
    phase_offsets = np.array([0.0, 0.4, -0.7])
    rows = []
    for cycles in WINDOW_CYCLES:
        estimator = make_estimator(cycles)
        n = estimator["n"]
        local_t = (np.arange(n) - (n - 1) / 2.0) / FS
        for phase_index, phase in enumerate(np.linspace(0, 2 * np.pi, 24, endpoint=False)):
            theta = 2 * np.pi * F0 * local_t[:, None] * HARMONICS
            theta += phase * HARMONICS + phase_offsets
            mixture = (np.cos(theta) * amplitudes).sum(axis=1)
            estimated = estimate_cwt_features(mixture[None, :], estimator)[0, :5]
            expected = np.r_[amplitudes, amplitudes[1:] / amplitudes[0]]
            for feature, actual, target in zip(FEATURES[:5], estimated, expected):
                rows.append({
                    "test": "harmonic_mixture", "cycles": cycles,
                    "phase_index": phase_index, "feature": feature,
                    "expected": target, "estimated": actual,
                    "error_pct": 100.0 * (actual / target - 1.0),
                })
            for j, harmonic in enumerate(HARMONICS):
                tone = np.cos(theta[:, j])
                actual = estimate_cwt_features(tone[None, :], estimator)[0, j]
                rows.append({
                    "test": "isolated_unit_tone", "cycles": cycles,
                    "phase_index": phase_index, "feature": f"I{harmonic}",
                    "expected": 1.0, "estimated": actual,
                    "error_pct": 100.0 * (actual - 1.0),
                })

    details = pd.DataFrame(rows)
    summary = (
        details.assign(abs_error_pct=details["error_pct"].abs())
        .groupby(["test", "cycles", "feature"], as_index=False)
        .agg(
            max_abs_error_pct=("abs_error_pct", "max"),
            mean_abs_error_pct=("abs_error_pct", "mean"),
        )
    )
    return details, summary


def plot_signal_overview(signals):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, layout="constrained")
    for ax, (condition, signal) in zip(axes.flat, signals.items()):
        ax.plot(signal["time_s"], signal["stress_Pa"], color="#007F86", lw=0.9)
        ax.set_title(f"Gamma = {signal['gamma']:g}")
        ax.set_ylabel("Stress (Pa)")
        ax.grid(alpha=0.14)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("APR18 LAOS stress after startup-sample removal", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "01_signal_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_scalograms(signals, estimator):
    dense_f = np.linspace(0.5 * F0, 8.0 * F0, 96)
    kernels, _ = morlet_bank(estimator["n"], dense_f)
    energy = np.sum(np.abs(kernels) ** 2, axis=0)
    hop = round(HOP_S * FS)
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True,
                             layout="constrained")
    mesh = None
    for ax, (condition, signal) in zip(axes.flat, signals.items()):
        windows = sliding_window_view(signal["stress_Pa"], estimator["n"])[::hop]
        coefficients = windows @ kernels
        power = np.abs(coefficients) ** 2 / energy
        relative_db = 10.0 * np.log10(np.maximum(power / power.max(), 1e-8))
        starts = np.arange(0, len(signal["stress_Pa"]) - estimator["n"] + 1, hop)
        centers = signal["time_s"][starts] + (estimator["n"] - 1) / (2 * FS)
        mesh = ax.pcolormesh(centers, dense_f / F0, relative_db.T, shading="auto",
                             cmap="magma", norm=Normalize(-35, 0))
        for harmonic in HARMONICS:
            ax.axhline(harmonic, color="white", lw=0.65, ls="--", alpha=0.72)
        ax.set_title(f"Gamma = {signal['gamma']:g}")
        ax.set_ylabel("Frequency / f0")
    for ax in axes[-1]:
        ax.set_xlabel("Window centre (s)")
    fig.colorbar(mesh, ax=axes, label="Relative CWT power (dB)", shrink=0.82)
    fig.suptitle("Two-cycle finite-window Morlet CWT", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "02_cwt_scalograms.png", bbox_inches="tight")
    plt.close(fig)


def plot_harmonic_summary(summary):
    part = summary[summary["cycles"] == 2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for feature in ["I1", "I3", "I5"]:
        data = part[part["feature"] == feature].sort_values("gamma")
        axes[0].errorbar(data["gamma"], data["median"], yerr=data["mad_scale"],
                         marker="o", capsize=3, lw=1.7, label=feature,
                         color=COLORS[feature])
    axes[0].set(xscale="log", yscale="log", xlabel="Gamma", ylabel="Amplitude (Pa)",
                title="CWT harmonic amplitudes")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.15, which="both")

    for feature, label, color in [("I3_I1", "I3/I1", COLORS["I3"]),
                                  ("I5_I1", "I5/I1", COLORS["I5"])]:
        data = part[part["feature"] == feature].sort_values("gamma")
        axes[1].errorbar(data["gamma"], data["median"], yerr=data["mad_scale"],
                         marker="o", capsize=3, lw=1.7, label=label, color=color)
    axes[1].set(xscale="log", yscale="log", xlabel="Gamma", ylabel="Harmonic ratio",
                title="Relative harmonic structure")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.15, which="both")
    fig.suptitle("Two-cycle CWT summary across LAOS amplitudes", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "03_harmonic_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_stability(summary):
    part = summary[summary["cycles"] == 2]
    matrix = (part.pivot(index="gamma", columns="feature", values="robust_cv_pct")
              .reindex(index=[0.5, 1, 5, 10, 50, 100], columns=FEATURES))
    display = np.log10(1.0 + matrix.to_numpy())
    fig, ax = plt.subplots(figsize=(10.5, 4.7), layout="constrained")
    image = ax.imshow(display, aspect="auto", cmap="YlGnBu", vmin=0,
                      vmax=np.nanpercentile(display, 95))
    ax.set_xticks(range(len(FEATURES)), FEATURE_LABELS)
    ax.set_yticks(range(len(matrix.index)), [f"{v:g}" for v in matrix.index])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Gamma")
    ax.set_title("Temporal variability: robust CV within each record")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            label = f"{value:.1f}%" if value < 100 else ">99%"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if display[i, j] > 0.65 * np.nanmax(display) else "black")
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{10**v - 1:.1f}%" for v in ticks])
    cbar.set_label("Robust CV")
    fig.savefig(OUTPUT_DIR / "04_feature_stability.png", bbox_inches="tight")
    plt.close(fig)


def plot_method_agreement(agreement):
    part = agreement[agreement["cycles"] == 2]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, layout="constrained")
    for ax, feature in zip(axes, ["I1", "I3", "I5"]):
        data = part[part["harmonic"] == feature].sort_values("gamma")
        ax.axhline(0, color="#6D7780", lw=0.9)
        ax.plot(data["gamma"], data["median_level_difference_pct"], marker="o",
                lw=1.7, color=COLORS[feature])
        ax.set_xscale("log")
        ax.set_title(feature)
        ax.set_xlabel("Gamma")
        ax.grid(alpha=0.15, which="both")
    axes[0].set_ylabel("CWT - harmonic regression (%)")
    fig.suptitle("Two-cycle amplitude agreement", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "05_method_agreement.png", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(calibration):
    part = calibration[calibration["test"] == "harmonic_mixture"]
    fig, ax = plt.subplots(figsize=(8.8, 4.2), layout="constrained")
    x = np.arange(5)
    for offset, cycles, color in [(-0.18, 2, "#007F86"), (0.18, 1, "#D28A1A")]:
        values = (part[part["cycles"] == cycles].set_index("feature")
                  .reindex(FEATURES[:5])["max_abs_error_pct"])
        bars = ax.bar(x + offset, values, width=0.34, color=color,
                      label=f"{cycles} cycle" + ("s" if cycles == 2 else ""))
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    ax.set_yscale("log")
    ax.set_xticks(x, FEATURE_LABELS[:5])
    ax.set_ylabel("Maximum absolute relative error (%)")
    ax.set_title("Finite-window CWT calibration with a known harmonic mixture")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15, which="major")
    fig.savefig(OUTPUT_DIR / "06_calibration.png", bbox_inches="tight")
    plt.close(fig)


def write_results_guide(input_summary, feature_summary, agreement, calibration):
    two = feature_summary[feature_summary["cycles"] == 2]
    i1_cv = two[two["feature"] == "I1"]["robust_cv_pct"]
    agreement_two = agreement[agreement["cycles"] == 2]
    mixture_two = calibration[
        (calibration["test"] == "harmonic_mixture") & (calibration["cycles"] == 2)
    ]
    text = f"""APR18 original LAOS CWT results

Scope
-----
- Six omega=1 rad/s LAOS records: Gamma 0.5, 1, 5, 10, 50 and 100.
- Each block's first row was removed as the fixed startup spike.
- Uniform grid: {FS:g} Hz; two-cycle primary window; {HOP_S:g} s output step.
- This source has no labelled change time or clotting point, so no detection delay is reported.

Files to inspect first
----------------------
1. 02_cwt_scalograms.png: time-frequency structure within each condition.
2. 03_harmonic_summary.png: I1, I3, I5 and harmonic ratios across Gamma.
3. 04_feature_stability.png: within-record temporal variability.
4. 05_method_agreement.png: CWT amplitude versus same-window harmonic regression.
5. 06_calibration.png: known-signal estimator error for one and two cycles.

Quick numerical checks
----------------------
- Number of analysed conditions: {len(input_summary)}
- Two-cycle I1 robust CV range: {i1_cv.min():.3f}% to {i1_cv.max():.3f}%
- Largest absolute two-cycle CWT/projection median difference: \
{agreement_two['median_level_difference_pct'].abs().max():.3f}%
- Largest two-cycle mixed-harmonic calibration error: \
{mixture_two['max_abs_error_pct'].max():.3f}%

Interpretation boundary
-----------------------
These outputs show repeatability and method agreement inside the supplied records. They do not by
themselves identify a gel point, clotting time, diagnostic threshold, or prospective false-alarm rate.
"""
    (OUTPUT_DIR / "RESULTS_README.txt").write_text(text, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signals, input_summary = load_laos()

    frames = []
    estimators = {cycles: make_estimator(cycles) for cycles in WINDOW_CYCLES}
    for cycles, estimator in estimators.items():
        for condition, signal in signals.items():
            frame = extract_windows(signal, estimator)
            frame.insert(0, "cycles", cycles)
            frame.insert(1, "condition", condition)
            frame.insert(2, "gamma", signal["gamma"])
            frames.append(frame)
    all_features = pd.concat(frames, ignore_index=True)

    feature_summary = robust_summary(all_features)
    agreement = method_agreement(all_features)
    comparison = window_comparison(feature_summary)
    calibration_details, calibration_summary = calibration_experiment()

    save_csv(input_summary, "input_summary.csv")
    save_csv(all_features[all_features["cycles"] == 2], "cwt_features_2cycle.csv")
    save_csv(all_features[all_features["cycles"] == 1], "cwt_features_1cycle.csv")
    save_csv(feature_summary, "feature_summary.csv")
    save_csv(agreement, "harmonic_method_agreement.csv")
    save_csv(comparison, "window_comparison.csv")
    save_csv(calibration_details, "calibration_details.csv")
    save_csv(calibration_summary, "calibration_summary.csv")

    plot_signal_overview(signals)
    plot_scalograms(signals, estimators[2])
    plot_harmonic_summary(feature_summary)
    plot_feature_stability(feature_summary)
    plot_method_agreement(agreement)
    plot_calibration(calibration_summary)
    write_results_guide(input_summary, feature_summary, agreement, calibration_summary)

    config = {
        "data_file": str(DATA_FILE),
        "sheet": SHEET_NAME,
        "sampling_rate_Hz": FS,
        "omega_rad_s": OMEGA,
        "fundamental_frequency_Hz": F0,
        "hop_s": HOP_S,
        "window_cycles": WINDOW_CYCLES,
        "window_samples": {str(c): estimators[c]["n"] for c in WINDOW_CYCLES},
        "morlet_B": MORLET_B,
        "frequency_band_Hz": [0.5 * F0, 8.0 * F0],
        "features": FEATURES,
        "startup_rows_dropped_per_condition": 1,
        "timing_definition": "window_end_s is the earliest time the complete window is available",
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    archive = shutil.make_archive(str(OUTPUT_DIR), "zip", OUTPUT_DIR)
    print(f"Results folder: {OUTPUT_DIR}")
    print(f"Upload this file for interpretation: {archive}")


if __name__ == "__main__":
    main()
