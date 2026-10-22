"""Independently check selected stage 3-4 results using NumPy and PyWavelets.

Run after run_sensitivity.py. This script does not import the study's calculation
helpers, modify observations, or repeat the full analysis. A failed numerical
check is saved in independent_review.json and causes a nonzero exit status.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
import pywt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TOLERANCE = 1e-14


def direct_features(signal, start_s, removed_samples):
    """Recompute eight detail energies from explicit absolute sample slices."""
    coefficients = pywt.wavedec(signal, "db2", mode="symmetric", level=8)
    left = int(round(start_s * 100)) - removed_samples
    right = left + 1000
    energies = []
    for level in range(1, 9):
        selected = [np.zeros_like(values) for values in coefficients]
        selected[-level] = coefficients[-level]
        detail = pywt.waverec(selected, "db2", mode="symmetric")[:len(signal)]
        energies.append(float(np.mean(detail[left:right] ** 2)))
        if level == 6:
            previous_r6 = float(np.sqrt(np.mean(detail[left - 1000:left] ** 2)))
    energy = np.array(energies)
    return {"R6": float(np.sqrt(energy[5])), "H13": float(np.sqrt(energy[:3].sum())),
            "P6": float(energy[5] / energy.sum()),
            "V6": float((np.sqrt(energy[5]) - previous_r6) / 10)}


def main():
    config = json.loads((ROOT / "config.json").read_text())
    # These independent calculations intentionally implement the fixed protocol.
    required = {"sampling_hz": 100, "frequency_hz": 1, "wavelet": "db2",
                "mode": "symmetric", "level": 8, "window_s": 10,
                "boundary_s": 20, "duration_s": 1800}
    if any(config[key] != value for key, value in required.items()):
        raise ValueError("This independent check requires the original fixed DWT protocol")
    source = Path(os.environ.get("DWT_NOISE_DATA", str((ROOT / config["source_dir"]).resolve())))
    checks, files, spots = [], {}, []

    def register(path):
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    register(ROOT / "config.json")
    register(Path(__file__).resolve())
    differences = pd.read_csv(register(RESULTS / "04_sensitivity/start_shift_differences.csv"))
    # Different times, cases and shifts test indexing independently of the helpers.
    for case, start, removed in [("control", 50, 1), ("clot_A", 320, 8), ("clot_A", 1530, 32)]:
        signal = pd.read_csv(register(source / f"{case}.csv")).stress_Pa.to_numpy()
        baseline = direct_features(signal, start, 0)
        variant = direct_features(signal[removed:], start, removed)
        table = pd.read_csv(register(RESULTS / "04_sensitivity/feature_variants" /
                                    f"{case}_start_{removed}samples.csv"))
        saved = table[table.start_s == start].iloc[0]
        delta = differences[(differences.case == case) & (differences.removed_samples == removed)
                            & (differences.start_s == start)].set_index("feature")
        values = {}
        for name, value in variant.items():
            difference = value - baseline[name]
            value_error = abs(value - saved[name])
            delta_error = abs(difference - delta.loc[name, "signed_difference"])
            passed = bool(value_error < TOLERANCE and delta_error < TOLERANCE)
            values[name] = {"independent_value": value, "saved_value": float(saved[name]),
                            "independent_signed_difference": difference,
                            "saved_signed_difference": float(delta.loc[name, "signed_difference"]),
                            "absolute_value_error": float(value_error),
                            "absolute_difference_error": float(delta_error), "passed": passed}
            checks.append({"name": f"{case}_{start}s_shift{removed}_{name}", "passed": passed})
        spots.append({"case": case, "window_start_s": start, "window_end_s": start + 10,
                      "removed_samples": removed, "features": values})

    late = {}
    for case in ["control", "clot_A"]:
        folder = RESULTS / "03_noise_attribution" / case
        clean = pd.read_csv(register(folder / "clean_features.csv"))
        with np.load(register(folder / "known_components.npz")) as known:
            time = known["time_s"]
            selected = (time >= 1500) & (time < 1700)
            features = clean[(clean.start_s >= 1500) & (clean.end_s <= 1700)]
            storage, loss = known["storage_modulus_Pa"][selected], known["loss_modulus_Pa"][selected]
            late[case] = {"interval_s": [1500, 1700], "Gprime_start_Pa": float(storage[0]),
                          "Gprime_end_Pa": float(storage[-1]), "Gprime_range_Pa": float(np.ptp(storage)),
                          "Gdoubleprime_range_Pa": float(np.ptp(loss)),
                          "clean_raw_rms_range_Pa": [float(features.raw_rms.min()), float(features.raw_rms.max())],
                          "clean_R6_range_Pa": [float(features.R6.min()), float(features.R6.max())],
                          "clean_R6_peak_to_peak_over_mean_percent":
                              float(100 * (features.R6.max() - features.R6.min()) / features.R6.mean()),
                          "clean_V6_range_Pa_per_s": [float(features.V6.min()), float(features.V6.max())]}
            if case == "control":
                coefficients = pywt.wavedec(known["clean_stress_Pa"], "db2", mode="symmetric", level=8)
                selected_coefficients = [np.zeros_like(values) for values in coefficients]
                selected_coefficients[-6] = coefficients[-6]
                detail = pywt.waverec(selected_coefficients, "db2", mode="symmetric")[:len(time)]
                rms = np.sqrt(np.mean(detail.reshape(-1, 1000) ** 2, axis=1))
                error = float(np.max(np.abs(rms - clean.R6.to_numpy())))
                interior = rms[2:178]  # Omit the first and last 20 s.
                period_error = float(np.max(np.abs(interior[8:] - interior[:-8])))
                late[case]["independent_clean_R6_max_error_Pa"] = error
                late[case]["clean_R6_80s_repeat_max_error_Pa"] = period_error
                checks.extend([
                    {"name": "control_clean_R6_direct_reproduction", "passed": error < TOLERANCE},
                    {"name": "control_clean_R6_80s_repeat", "passed": period_error < TOLERANCE},
                    {"name": "control_known_moduli_constant", "passed": bool(np.ptp(storage) == 0 and np.ptp(loss) == 0)}])

    summary = pd.read_csv(register(RESULTS / "04_sensitivity/start_shift_summary.csv"))
    aligned = summary[(summary.feature == "H13") & summary.removed_samples.isin([8, 32])]
    checks.append({"name": "H13_multiple_of_8_shift_invariance_in_saved_interiors",
                   "passed": bool(len(aligned) >= 4 and (aligned.maximum_absolute_difference == 0).all())})
    window_summary = pd.read_csv(register(RESULTS / "04_sensitivity/window_interval_summary.csv"))
    late_windows = window_summary[(window_summary.case == "clot_A") & (window_summary.interval_start_s == 1500)]
    passed = all(check["passed"] for check in checks)
    result = {
        "scope": "Independent spot checks and interpretation review of stages 3-4; no full rerun or data mutation.",
        "methods": ["Standalone numpy and pywt.wavedec/waverec; no study calculation helpers imported.",
                    "Three physical windows with distinct start removals; all eight details recomputed before explicit integer sample slicing.",
                    "R6, H13, P6, V6 values and signed differences checked against CSVs at absolute tolerance 1e-14 in native units.",
                    "Known components provide retrospective physical context; control clean D6 is independently reconstructed."],
        "all_passed": passed, "check_count": len(checks),
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "failure_behavior": "The script saves failed numerical checks and exits 1; otherwise it exits 0.",
        "checks": checks, "spot_checks": spots, "late_known_component_diagnostics": late,
        "clot_A_late_window_summary": late_windows.to_dict("records"),
        "interpretation_limits": [
            "Clean feature change is not automatically material change: constant control moduli give oscillatory clean R6 and V6.",
            "D6 decimation is 64 samples and the 1 Hz carrier has period 100 samples: their common repeat is 1600 samples. The phase of 1000-sample RMS windows repeats every 8 windows, consistent with the observed 80 s repeat.",
            "Zero H13 differences for removals of 8 or 32 samples reflect D1-D3 decimation alignment, not arbitrary shift invariance.",
            "Stage 4 V6 summaries require both contributing windows inside an interval; stage 3 may include a preceding window outside it, so their interval means should not be directly interchanged.",
            "No near-zero relative-error division, event threshold, accuracy or biological label is introduced."],
        "reviewed_file_sha256": files}
    destination = RESULTS / "01_validation/independent_review.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Independent review: {sum(check['passed'] for check in checks)}/{len(checks)} checks passed")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
