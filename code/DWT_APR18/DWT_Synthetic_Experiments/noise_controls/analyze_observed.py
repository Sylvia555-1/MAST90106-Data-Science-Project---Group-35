"""Stages 1-2: validate CSV records and extract offline DWT features.

Run this before evaluate_known_components.py. This script does not read
ground_truth.json, parameters.json, latent signals, or event times.
"""
import argparse
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import pywt
import scipy
from common import (ROOT, MAIN_FEATURES, FEATURE_UNITS, load_config, load_record,
                    decompose, extract_features, harmonic_reference, sha256,
                    write_json, shade_boundaries, save_figure, plt)


def plot_observed(case, frame, parts, features, harmonic, config, output):
    """Show observations and features without inserting simulation labels."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
    # Plot min/max in every 1 s bin; all samples contribute to this overview.
    stress = frame.stress_Pa.to_numpy().reshape(-1, config["sampling_hz"])
    seconds = np.arange(len(stress)) + 0.5
    axes[0].fill_between(seconds, stress.min(axis=1), stress.max(axis=1), color="C0", alpha=0.7)
    axes[0].set(title=f"{case}: observed stress (1 s min/max display)", ylabel="Stress (Pa)")
    for ax, interval in zip(axes[1:], [(50, 53), (1500, 1503)]):
        selected = frame.time_s.between(*interval)
        ax.plot(frame.time_s[selected], frame.stress_Pa[selected], lw=1)
        ax.set(title=f"Original samples: {interval[0]}-{interval[1]} s", ylabel="Stress (Pa)", xlabel="Time (s)")
    save_figure(fig, output / f"{case}_raw.png")

    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True, layout="constrained")
    # Each plotted point summarizes a 10 s interval, not an event time.
    for j, ax in enumerate(axes.flat, 1):
        ax.plot(features.mid_s, features[f"R{j}"], lw=1)
        ax.set(ylabel=f"D{j} RMS (Pa)", xlabel="Window midpoint (s)")
    shade_boundaries(axes, config)
    fig.suptitle(f"{case}: reconstructed detail strength; no event labels")
    save_figure(fig, output / f"{case}_detail_rms.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, name in zip(axes.flat, MAIN_FEATURES):
        ax.plot(features.mid_s, features[name])
        ax.set(title=name, ylabel=FEATURE_UNITS[name], xlabel="Window midpoint (s)")
    shade_boundaries(axes, config)
    fig.suptitle(f"{case}: four preselected features (offline)")
    save_figure(fig, output / f"{case}_main_features.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    axes[0].semilogy(features.mid_s, features.raw_rms, label="Raw stress RMS")
    for j in (5, 6, 7):
        axes[0].semilogy(features.mid_s, features[f"R{j}"], label=f"D{j} RMS")
    axes[0].set(ylabel="RMS (Pa)"); axes[0].legend(ncol=4)
    axes[1].plot(harmonic.mid_s, harmonic.Gprime_Pa, label="G'")
    axes[1].plot(harmonic.mid_s, harmonic.Gdoubleprime_Pa, label="G''")
    axes[1].set(ylabel="Harmonic estimate (Pa)"); axes[1].legend()
    axes[2].plot(harmonic.mid_s, harmonic.residual_rms_Pa)
    axes[2].set(ylabel="Fit residual RMS (Pa)", xlabel="Window midpoint (s)")
    shade_boundaries(axes, config)
    fig.suptitle(f"{case}: simple references; fit residual is not known noise")
    save_figure(fig, output / f"{case}_references.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["control", "clot_A"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = load_config()
    validation_dir = args.output_root / "01_validation"
    feature_dir = args.output_root / "02_features"
    validation_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    source = Path(config["source_dir"])
    # Hash files as bytes; no labels or metadata values enter the analysis.
    hashes = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    manifest = {"config_sha256": sha256(ROOT / "config.json"),
                "source_csv_sha256": {case: hashes[f"{case}.csv"] for case in args.cases},
                "source_files_sha256": hashes, "config": config,
                "scope": "Offline observable-only analysis; truth not used for features",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "pywavelets": pywt.__version__,
                             "scipy": scipy.__version__}, "python_executable": sys.executable}
    write_json(validation_dir / "manifest.json", manifest)
    reports, reference_inputs = {}, None
    for case in args.cases:
        frame = load_record(case, config)
        time, stress = frame.time_s.to_numpy(), frame.stress_Pa.to_numpy()
        parts = decompose(stress, config)
        reconstruction = sum(parts.values())
        error = float(np.max(np.abs(reconstruction - stress)))
        if error >= config["reconstruction_tolerance_pa"]:
            raise ValueError(f"Reconstruction failed: {case}, {error}")
        inputs = frame.iloc[:, :3].to_numpy()
        if reference_inputs is None:
            reference_inputs = inputs.copy()
        if not np.array_equal(reference_inputs, inputs):
            raise ValueError("Loading differs between supplied cases")
        features = extract_features(time, stress, parts, config)
        harmonic = harmonic_reference(frame, config)
        features.to_csv(feature_dir / f"{case}_features.csv", index=False)
        harmonic.to_csv(feature_dir / f"{case}_harmonic.csv", index=False)
        np.savez_compressed(feature_dir / f"{case}_details.npz", time_s=time, **parts)
        reports[case] = {"rows": len(frame), "finite": True, "uniform_time": True,
                         "identical_loading": True, "min_gap_s": float(np.diff(time).min()),
                         "max_gap_s": float(np.diff(time).max()),
                         "reconstruction_max_error_Pa": error,
                         "reconstruction_relative_max_error": error / float(np.max(abs(stress))),
                         "window_count": len(features),
                         "interior_window_count": int((~features.boundary_window).sum()),
                         "reference_window_count": int(features.reference_window.sum()),
                         "V6_reference_valid_count": int(features.loc[features.reference_window, "V6"].notna().sum())}
        plot_observed(case, frame, parts, features, harmonic, config, feature_dir / "figures")
        print(f"{case}: {len(frame)} samples, {len(features)} windows, reconstruction error {error:.3g} Pa")
    unchanged = all(sha256(source / p) == value for p, value in hashes.items())
    if not unchanged:
        raise ValueError("Source files changed during analysis")
    write_json(validation_dir / "data_validation.json", {"cases": reports, "source_files_unchanged": unchanged})


if __name__ == "__main__":
    main()
