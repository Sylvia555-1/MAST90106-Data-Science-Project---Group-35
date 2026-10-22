"""Stage 4: compare window sizes and DWT starting positions, without truth.

Run only after reviewing the known-component results. This script does not
choose a best configuration, identify events, or read simulation labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import (ROOT, MAIN_FEATURES, FEATURE_UNITS, load_config, load_record,
                    decompose, extract_features, sha256, write_json,
                    shade_boundaries, save_figure, plt)

# These fixed development intervals are descriptive, not event labels.
INTERVALS = [(20, 200), (200, 400), (400, 800), (800, 1200), (1500, 1700)]


def verify_inputs(manifest, config):
    """Require the same settings and original files as stages 1-2."""
    if manifest["config_sha256"] != sha256(ROOT / "config.json"):
        raise ValueError("Configuration changed after observable feature extraction")
    source = Path(config["source_dir"])
    for name, expected in manifest["source_files_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Source file changed after stage 2: {name}")


def interval_summary(case, variant, features):
    """Summarize complete windows inside common physical time intervals.

    Different window sizes are not paired as if their midpoints were equal.
    For V6, both contributing windows must lie inside the stated interval.
    """
    rows = []
    for start, end in INTERVALS:
        for name in MAIN_FEATURES:
            included = ((features.start_s >= start) & (features.end_s <= end)
                        & ~features.boundary_window & np.isfinite(features[name]))
            support_start = features.start_s.copy()
            if name == "V6":
                support_start = features.start_s.shift(1)
                included &= support_start >= start
            values = features.loc[included, name]
            if values.empty:
                raise ValueError(f"No usable windows for {case}, {variant}, {name}, {start}-{end}")
            rows.append({"case": case, "variant": variant, "feature": name,
                         "unit": FEATURE_UNITS[name], "interval_start_s": start,
                         "interval_end_s": end, "window_count": len(values),
                         "included_support_start_s": float(support_start[included].min()),
                         "included_support_end_s": float(features.loc[included, "end_s"].max()),
                         "mean": values.mean(), "minimum": values.min(), "maximum": values.max()})
    return rows


def shift_differences(case, shift, baseline, variant):
    """Compare the same absolute 10 s windows in the shared interior."""
    paired = baseline.merge(variant, on=["start_s", "end_s"],
                            suffixes=("_base", "_variant"), validate="one_to_one")
    keep = ~paired.boundary_window_base & ~paired.boundary_window_variant
    # Use the same complete, finite window set for all four feature comparisons.
    for name in MAIN_FEATURES:
        keep &= np.isfinite(paired[f"{name}_base"]) & np.isfinite(paired[f"{name}_variant"])
    paired = paired.loc[keep]
    if paired.empty:
        raise ValueError("No shared interior windows remain after shifting")
    rows, summaries = [], []
    for name in MAIN_FEATURES:
        difference = paired[f"{name}_variant"] - paired[f"{name}_base"]
        for index, value in difference.items():
            row = paired.loc[index]
            rows.append({"case": case, "removed_samples": shift, "feature": name,
                         "unit": FEATURE_UNITS[name], "start_s": row.start_s,
                         "end_s": row.end_s, "mid_s": row.mid_s_base,
                         "baseline": row[f"{name}_base"], "variant": row[f"{name}_variant"],
                         "signed_difference": value, "absolute_difference": abs(value)})
        summaries.append({"case": case, "removed_samples": shift, "feature": name,
                          "unit": FEATURE_UNITS[name], "window_count": len(difference),
                          "mean_signed_difference": difference.mean(),
                          "mean_absolute_difference": difference.abs().mean(),
                          "maximum_absolute_difference": difference.abs().max()})
    return rows, summaries


def plot_comparisons(case, window_variants, differences, config, destination):
    """Keep physical units visible and show the full record in every panel."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for ax, name in zip(axes, MAIN_FEATURES):
        for seconds, features in window_variants.items():
            ax.plot(features.mid_s, features[name], lw=0.9, label=f"{seconds:g} s windows")
        ax.set(ylabel=f"{name} ({FEATURE_UNITS[name]})", xlim=(0, config["duration_s"]))
        ax.legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Absolute time: window midpoint (s)")
    shade_boundaries(axes, config)
    fig.suptitle(f"{case}: window-size sensitivity; different windows are not pointwise pairs")
    save_figure(fig, destination / f"{case}_window_sizes.png")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for ax, name in zip(axes, MAIN_FEATURES):
        subset = differences[differences.feature == name]
        for removed, group in subset.groupby("removed_samples"):
            ax.plot(group.mid_s, group.signed_difference, lw=0.9, label=f"Remove {removed} samples")
        ax.axhline(0, color="black", lw=0.6)
        ax.set(ylabel=f"Change in {name}\n({FEATURE_UNITS[name]})", xlim=(0, config["duration_s"]))
        ax.legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("Absolute time: common 10 s window midpoint (s)")
    shade_boundaries(axes, config)
    fig.suptitle(f"{case}: signed difference after changing DWT origin; shared interior only")
    save_figure(fig, destination / f"{case}_start_shift_differences.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["control", "clot_A"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = load_config()
    manifest_path = args.output_root / "01_validation" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_inputs(manifest, config)
    output = args.output_root / "04_sensitivity"
    variants_dir = output / "feature_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    differences, shift_summaries, interval_rows, checks, baseline_hashes = [], [], [], {}, {}
    for case in args.cases:
        frame = load_record(case, config)
        time, signal = frame.time_s.to_numpy(), frame.stress_Pa.to_numpy()
        components = decompose(signal, config)
        baseline = extract_features(time, signal, components, config)
        saved_path = args.output_root / "02_features" / f"{case}_features.csv"
        saved = pd.read_csv(saved_path)
        columns = ["start_s", "end_s", "raw_rms"] + MAIN_FEATURES
        if saved.shape != baseline.shape or not np.allclose(saved[columns], baseline[columns],
                                                            rtol=1e-12, atol=1e-15, equal_nan=True):
            raise ValueError(f"Saved observable features do not reproduce for {case}")
        baseline_hashes[case] = sha256(saved_path)
        window_variants = {}
        for seconds in sorted({config["window_s"], *config["sensitivity_window_s"]}):
            features = extract_features(time, signal, components, config, window_s=seconds)
            features.to_csv(variants_dir / f"{case}_window_{seconds:g}s.csv", index=False)
            window_variants[seconds] = features
            interval_rows.extend(interval_summary(case, f"window_{seconds:g}s", features))
        case_rows, maximum_error = [], 0.0
        for removed in config["sensitivity_start_samples"]:
            # Preserve timestamps; do not wrap the signal or fill the missing start.
            cropped = signal[removed:]
            parts = decompose(cropped, config)
            error = float(np.max(np.abs(sum(parts.values()) - cropped)))
            maximum_error = max(maximum_error, error)
            if error >= config["reconstruction_tolerance_pa"]:
                raise ValueError(f"Cropped reconstruction failed for {case}, {removed} samples")
            features = extract_features(time[removed:], cropped, parts, config)
            features.to_csv(variants_dir / f"{case}_start_{removed}samples.csv", index=False)
            rows, summary = shift_differences(case, removed, baseline, features)
            case_rows.extend(rows)
            shift_summaries.extend(summary)
        differences.extend(case_rows)
        checks[case] = {"saved_baseline_reproduced": True,
                        "maximum_cropped_reconstruction_error_Pa": maximum_error}
        plot_comparisons(case, window_variants, pd.DataFrame(case_rows), config, output / "figures")
    verify_inputs(manifest, config)
    pd.DataFrame(interval_rows).to_csv(output / "window_interval_summary.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "start_shift_differences.csv", index=False)
    pd.DataFrame(shift_summaries).to_csv(output / "start_shift_summary.csv", index=False)
    write_json(output / "manifest.json", {"scope": "Observable-only descriptive sensitivity; no event labels",
               "source_files_unchanged": True, "config_sha256": sha256(ROOT / "config.json"),
               "stage_1_2_manifest_sha256": sha256(manifest_path),
               "baseline_feature_sha256": baseline_hashes, "validation": checks,
               "source_csv_sha256": {case: sha256(Path(config["source_dir"]) / f"{case}.csv")
                                     for case in args.cases}, "descriptive_intervals_s": INTERVALS,
               "limitations": ["Finite sensitivity checks, not a parameter search or event detector.",
                               "Window-size summaries use complete windows, not paired midpoints.",
                               "Start-shift differences use shared interior windows and native units."]})
    print(f"Sensitivity outputs saved to {output}")


if __name__ == "__main__":
    main()
