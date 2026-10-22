"""Audit DWT features against known synthetic components; do not denoise.

Run analyze_observed.py first. This separate stage checks its frozen outputs
before reading generator parameters or labels. All attribution is retrospective
and depends on the known simulator; it is not blind noise identification.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/coagulation_dwt_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from common import (ROOT, decompose, extract_features, load_config, load_record,
                    sha256, write_json)


FEATURE_UNITS = {"R6": "Pa", "H13": "Pa", "P6": "dimensionless", "V6": "Pa/s"}
ENERGY_ATOL, ENERGY_RTOL = 1e-18, 1e-10
LINEARITY_ATOL = 1e-10


def verify_observed(case, config, output_root, manifest):
    """Require frozen CSV-only features and reproduce them before reading truth."""
    feature_path = output_root / "02_features" / f"{case}_features.csv"
    source_path = Path(config["source_dir"]) / f"{case}.csv"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Run analyze_observed.py first: {feature_path}")
    if manifest["config_sha256"] != sha256(ROOT / "config.json"):
        raise ValueError("The configuration changed after the observed analysis.")
    if manifest["source_csv_sha256"].get(case) != sha256(source_path):
        raise ValueError(f"The source CSV is absent from, or differs from, the manifest: {case}")

    record = load_record(case, config)
    time = record["time_s"].to_numpy()
    stress = record["stress_Pa"].to_numpy()
    components = decompose(stress, config)
    recomputed = extract_features(time, stress, components, config)
    saved = pd.read_csv(feature_path)
    if len(saved) != len(recomputed) or not set(recomputed).issubset(saved):
        raise ValueError(f"Observed feature table has an unexpected layout: {case}")
    max_errors = {}
    for column in recomputed:
        expected, actual = recomputed[column].to_numpy(), saved[column].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18,
                                   equal_nan=True, err_msg=f"{case}: {column}")
        difference = np.asarray(actual, float) - np.asarray(expected, float)
        finite = np.isfinite(difference)
        max_errors[column] = float(np.max(np.abs(difference[finite]))) if finite.any() else None
    return record, components, recomputed, {
        "observed_feature_sha256": sha256(feature_path),
        "observed_features_reproduced": True,
        "max_feature_serialization_errors": max_errors,
    }


def reconstruct_known_components(record, parameters, case_truth, config):
    """Recreate the supplied formula, preserving its arithmetic and RNG order."""
    time = record["time_s"].to_numpy()
    np.testing.assert_equal(parameters["sampling_Hz"], config["sampling_hz"])
    np.testing.assert_equal(parameters["frequency_Hz"], config["frequency_hz"])
    np.testing.assert_equal(parameters["duration_s"], config["duration_s"])
    omega = parameters["angular_frequency_rad_s"]
    gamma = parameters["strain_amplitude"]
    onset = case_truth["onset_s"]
    if onset is None:
        growth = np.zeros(len(time))
    else:
        elapsed = np.maximum(time - onset, 0)
        # Match x**shape / tau**shape used in the supplied generator.
        growth = 1 - np.exp(-elapsed ** case_truth["growth_shape"] /
                            case_truth["tau_growth_s"] ** case_truth["growth_shape"])
    increment = case_truth["plateau_increment_Gprime_Pa"]
    storage = parameters["APR18_baseline_Gprime_Pa"] + increment * growth
    loss = (parameters["APR18_baseline_Gdoubleprime_Pa"] +
            parameters["clot_Gdoubleprime_to_Gprime_increment_ratio"] * increment * growth)
    clean = gamma * (storage * np.sin(omega * time) + loss * np.cos(omega * time))
    envelope = gamma * np.hypot(storage, loss)
    correlation = parameters["AR1_correlation"]
    white = np.random.default_rng(case_truth["seed"]).standard_normal(len(time))
    colored = lfilter([np.sqrt(1 - correlation ** 2)], [1, -correlation], white)
    generated_noise = (parameters["noise_floor_Pa"] +
                       parameters["noise_relative_envelope"] * envelope) * colored
    expected = np.column_stack((time, gamma * np.sin(omega * time),
                                gamma * omega * np.cos(omega * time), clean + generated_noise))
    # CSVs were saved with %.9g; do not confuse rounding with model disagreement.
    rounded = np.array([[float(format(value, ".9g")) for value in column]
                        for column in expected.T]).T
    actual = record[["time_s", "strain", "shear_rate_1_s", "stress_Pa"]].to_numpy()
    np.testing.assert_allclose(actual, rounded, rtol=2e-15, atol=1e-15,
                               err_msg="The supplied CSV cannot be reproduced at its saved precision.")
    effective_noise = actual[:, 3] - clean
    return {
        "time_s": time, "clean_stress_Pa": clean,
        "generated_noise_Pa": generated_noise, "effective_noise_Pa": effective_noise,
        "storage_modulus_Pa": storage, "loss_modulus_Pa": loss, "growth_fraction": growth,
    }, {
        "nine_significant_digit_reproduction_passed": True,
        "serialized_csv_max_abs_difference": float(np.max(np.abs(actual - rounded))),
        "noise_effective_minus_generated_max_abs_Pa": float(np.max(np.abs(effective_noise - generated_noise))),
        "noise_effective_definition": "Observed CSV stress minus exact clean stress; includes CSV rounding.",
    }


def band_energy_audit(time, observed, clean, noise, windows):
    """Check linearity and retain the signed signal-noise cross term."""
    linear_errors = {}
    for band in observed:
        error = float(np.max(np.abs(observed[band] - clean[band] - noise[band])))
        if error >= LINEARITY_ATOL:
            raise AssertionError(f"DWT linearity failed for {band}: {error}")
        linear_errors[band] = error
    rows = []
    for window in windows.itertuples():
        left, right = np.searchsorted(time, [window.start_s, window.end_s])
        for level in range(1, 9):
            name = f"D{level}"
            y, s, n = [values[name][left:right] for values in (observed, clean, noise)]
            energy_y, energy_s, energy_n = [float(np.mean(values ** 2)) for values in (y, s, n)]
            cross = float(2 * np.mean(s * n))
            residual = energy_y - (energy_s + energy_n + cross)
            np.testing.assert_allclose(energy_y, energy_s + energy_n + cross,
                                       atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
            # Infinity and NaN are preserved; no arbitrary epsilon is added.
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = float(10 * np.log10(np.divide(energy_s, energy_n)))
            rows.append({"start_s": window.start_s, "end_s": window.end_s,
                         "mid_s": window.mid_s, "boundary_window": window.boundary_window,
                         "level": level, "observed_energy_Pa2": energy_y,
                         "clean_energy_Pa2": energy_s, "noise_energy_Pa2": energy_n,
                         "cross_term_Pa2": cross, "identity_residual_Pa2": residual,
                         "known_component_snr_dB": snr})
    energy_table = pd.DataFrame(rows)
    return energy_table, {
        "linearity_max_abs_error_Pa_by_component": linear_errors,
        "linearity_absolute_tolerance_Pa": LINEARITY_ATOL,
        "energy_identity_passed": True,
        "energy_identity_max_abs_residual_Pa2": float(energy_table.identity_residual_Pa2.abs().max()),
        "energy_identity_absolute_tolerance_Pa2": ENERGY_ATOL,
        "energy_identity_relative_tolerance": ENERGY_RTOL,
    }


def compare_features(observed, clean):
    """Decompose feature changes, not nonlinear features themselves, additively."""
    rows, checks = [], {}
    for feature, unit in FEATURE_UNITS.items():
        y, s = observed[feature].to_numpy(), clean[feature].to_numpy()
        bias = y - s
        reference = observed.reference_window.to_numpy(bool) & np.isfinite(y) & np.isfinite(s)
        if not reference.any():
            raise ValueError(f"No finite reference windows for {feature}.")
        means = [float(np.mean(values[reference])) for values in (y, s, bias)]
        delta_y, delta_s, delta_bias = [values - mean for values, mean in zip((y, s, bias), means)]
        np.testing.assert_allclose(delta_y, delta_s + delta_bias,
                                   atol=1e-12, rtol=1e-10, equal_nan=True)
        identity_error = delta_y - delta_s - delta_bias
        checks[feature] = {
            "reference_finite_windows": int(reference.sum()),
            "reference_observed_mean": means[0], "reference_clean_mean": means[1],
            "reference_bias_mean": means[2], "unit": unit,
            "change_identity_max_abs_error": float(np.nanmax(np.abs(identity_error))),
        }
        for index, window in enumerate(observed.itertuples()):
            rows.append({"start_s": window.start_s, "end_s": window.end_s,
                         "mid_s": window.mid_s, "boundary_window": window.boundary_window,
                         "reference_window": window.reference_window, "feature": feature,
                         "unit": unit, "observed": y[index], "clean": s[index],
                         "noise_induced_bias": bias[index], "delta_observed": delta_y[index],
                         "delta_clean": delta_s[index], "delta_bias": delta_bias[index],
                         "change_identity_residual": identity_error[index]})
    return pd.DataFrame(rows), checks


def interval_summary(effects, energies, onset, config):
    """Use labels only to summarize preset diagnostic intervals after analysis."""
    intervals = [("reference", config["reference_start_s"], config["reference_end_s"]),
                 ("late", 1500, 1700)]
    if onset is not None:
        intervals += [("onset_plus_0_to_30", onset, onset + 30),
                      ("onset_plus_30_to_100", onset + 30, onset + 100),
                      ("onset_plus_100_to_300", onset + 100, onset + 300)]
    feature_rows, band_rows = [], []
    for label, start, end in intervals:
        selection = effects[(effects.start_s >= start) & (effects.end_s <= end) &
                            ~effects.boundary_window]
        for feature, group in selection.groupby("feature", sort=False):
            row = {"interval": label, "start_s": start, "end_s": end,
                   "feature": feature, "unit": FEATURE_UNITS[feature],
                   "finite_windows": int(np.isfinite(group.observed).sum())}
            for column in ("observed", "clean", "noise_induced_bias",
                           "delta_observed", "delta_clean", "delta_bias"):
                values = group[column]
                row.update({f"{column}_mean": values.mean(),
                            f"{column}_min": values.min(), f"{column}_max": values.max()})
            feature_rows.append(row)
        selection = energies[(energies.start_s >= start) & (energies.end_s <= end) &
                             ~energies.boundary_window]
        for level, group in selection.groupby("level"):
            signal_mean, noise_mean = group.clean_energy_Pa2.mean(), group.noise_energy_Pa2.mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                pooled_snr = float(10 * np.log10(np.divide(signal_mean, noise_mean)))
            band_rows.append({"interval": label, "start_s": start, "end_s": end,
                              "level": level, "windows": len(group),
                              "clean_energy_mean_Pa2": signal_mean,
                              "noise_energy_mean_Pa2": noise_mean,
                              "cross_term_mean_Pa2": group.cross_term_Pa2.mean(),
                              "pooled_known_component_snr_dB": pooled_snr,
                              "window_snr_min_dB": group.known_component_snr_dB.min(),
                              "window_snr_max_dB": group.known_component_snr_dB.max()})
    return pd.DataFrame(feature_rows), pd.DataFrame(band_rows)


def mark_context(axis, config, onset):
    """Mark numerical boundaries and the known inserted label, never a detection."""
    boundary, duration = config["boundary_s"], config["duration_s"]
    axis.axvspan(0, boundary, color="grey", alpha=.18)
    axis.axvspan(duration - boundary, duration, color="grey", alpha=.18)
    if onset is not None:
        axis.axvline(onset, color="black", linestyle=":", linewidth=1)
    axis.set_xlim(0, duration)
    axis.grid(alpha=.18)


def save_figures(case, effects, energies, onset, config, folder):
    """Make English figures; all time markers describe window midpoints."""
    label = "Control: no inserted onset" if onset is None else f"Dotted line: inserted onset {onset:g} s (audit only)"
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, layout="constrained")
    for row, (feature, unit) in enumerate(FEATURE_UNITS.items()):
        data = effects[effects.feature == feature]
        left, right = axes[row]
        left.plot(data.mid_s, data.observed, label="Observed", linewidth=1.5)
        left.plot(data.mid_s, data.clean, label="Known clean", linestyle="--", linewidth=1)
        right.plot(data.mid_s, data.noise_induced_bias, color="tab:orange", linewidth=1)
        right.axhline(0, color="black", linewidth=.6)
        if feature in ("R6", "H13"):
            left.set_yscale("log")
        left.set_ylabel(f"{feature} ({unit})")
        right.set_ylabel(f"Observed - clean ({unit})")
        left.legend(fontsize=8)
        for axis in (left, right):
            mark_context(axis, config, onset)
    axes[0, 0].set_title("Feature values; R6 and H13 use logarithmic axes")
    axes[0, 1].set_title("Bias induced by this noise realization")
    for axis in axes[-1]:
        axis.set_xlabel("Window midpoint (s)")
    fig.suptitle(f"{case}: offline DWT feature audit, {config['window_s']:g} s windows\n"
                 f"{label}; grey = boundary-sensitive region", fontsize=12)
    fig.savefig(folder / "feature_values_and_noise_bias.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for axis, (feature, unit) in zip(axes, FEATURE_UNITS.items()):
        data = effects[effects.feature == feature]
        for column, name, style in (("delta_observed", "Observed change", "-"),
                                    ("delta_clean", "Clean change", "--"),
                                    ("delta_bias", "Change in noise-induced bias", ":")):
            axis.plot(data.mid_s, data[column], linestyle=style, label=name, linewidth=1.2)
        axis.set_ylabel(f"Delta {feature} ({unit})")
        axis.axhline(0, color="black", linewidth=.5)
        axis.legend(fontsize=8, loc="upper left")
        mark_context(axis, config, onset)
    axes[-1].set_xlabel("Window midpoint (s)")
    fig.suptitle(f"{case}: change from the {config['reference_start_s']:g}-{config['reference_end_s']:g} s reference mean\n"
                 f"Delta observed = Delta clean + Delta bias; {label}", fontsize=12)
    fig.savefig(folder / "feature_change_attribution.png", dpi=160)
    plt.close(fig)

    snr = energies.pivot(index="level", columns="mid_s", values="known_component_snr_dB")
    finite = snr.to_numpy()[np.isfinite(snr.to_numpy())]
    color_limit = max(10, float(np.ceil(np.max(np.abs(finite)) / 10) * 10))
    fig, axis = plt.subplots(figsize=(12, 4), layout="constrained")
    mesh = axis.pcolormesh(np.r_[energies.start_s.unique(), energies.end_s.max()],
                           np.arange(.5, 9.5), np.ma.masked_invalid(snr.to_numpy()),
                           cmap="RdBu_r", vmin=-color_limit, vmax=color_limit, shading="flat")
    fig.colorbar(mesh, ax=axis, label="Known-component SNR (dB)")
    axis.set_yticks(np.arange(1, 9), labels=[f"D{j}" for j in range(1, 9)])
    axis.set(xlabel="Time (s); cells are complete windows", ylabel="Detail level")
    mark_context(axis, config, onset)
    axis.set_title(f"{case}: clean/noise component energy ratio, not an estimated real-data SNR\n"
                   f"0 dB means equal component energies, not an event threshold. {label}", fontsize=11)
    fig.savefig(folder / "known_component_snr.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["control", "clot_A"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = load_config()
    manifest_path = args.output_root / "01_validation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = Path(config["source_dir"])
    for case in args.cases:
        record, observed_components, observed_features, validation = verify_observed(
            case, config, args.output_root, manifest)
        # Only now load truth. It is never passed to observed feature extraction.
        parameter_path, truth_path = source / "parameters.json", source / "ground_truth.json"
        parameters = json.loads(parameter_path.read_text())
        truth = {item["case"]: item for item in json.loads(truth_path.read_text())}[case]
        known, reproduction = reconstruct_known_components(record, parameters, truth, config)
        clean_components = decompose(known["clean_stress_Pa"], config)
        noise_components = decompose(known["effective_noise_Pa"], config)
        clean_features = extract_features(known["time_s"], known["clean_stress_Pa"], clean_components, config)
        noise_features = extract_features(known["time_s"], known["effective_noise_Pa"], noise_components, config)
        energies, algebra_checks = band_energy_audit(known["time_s"], observed_components,
                                                     clean_components, noise_components, observed_features)
        effects, feature_checks = compare_features(observed_features, clean_features)
        feature_summary, band_summary = interval_summary(effects, energies, truth["onset_s"], config)
        folder = args.output_root / "03_noise_attribution" / case
        folder.mkdir(parents=True, exist_ok=True)
        for name, table in {"clean_features": clean_features, "effective_noise_features": noise_features,
                            "band_component_energies": energies, "feature_noise_effects": effects,
                            "feature_interval_summary": feature_summary,
                            "band_interval_summary": band_summary}.items():
            table.to_csv(folder / f"{name}.csv", index=False, float_format="%.17g")
        np.savez_compressed(folder / "known_components.npz", **known)
        save_figures(case, effects, energies, truth["onset_s"], config, folder)
        for original, digest in manifest.get("source_files_sha256", {}).items():
            path = Path(original)
            if not path.is_absolute():
                path = source / path
            if sha256(path) != digest:
                raise AssertionError(f"A source file changed: {path}")
        audit = {
            "case": case, "scope": "Known-simulator offline attribution; no detection or denoising.",
            "source_csv_sha256": sha256(source / f"{case}.csv"),
            "parameters_sha256": sha256(parameter_path), "ground_truth_sha256": sha256(truth_path),
            "config_sha256": sha256(ROOT / "config.json"), "onset_label_for_audit_only_s": truth["onset_s"],
            "validation": {**validation, **reproduction, **algebra_checks},
            "feature_change_checks": feature_checks, "source_files_unchanged": True,
            "interpretation": [
                "A clean feature may fluctuate because of the carrier, decimation and window alignment.",
                "Noise-feature values are not the same as observed-minus-clean feature bias.",
                "Known component SNR concerns component strength, not the cause of every local change.",
                "The onset label is an inserted smooth-growth start, not a detected point or clinical label.",
                "No thresholds, classifiers, denoising or contribution percentages are fitted here.",
            ],
        }
        write_json(folder / "validation.json", audit)
        print(f"{case}: known-component reconstruction and attribution checks passed; {folder}")


if __name__ == "__main__":
    main()
