"""Collect completed results without changing DWT settings or fitting a detector.

Run this after stages 1-5. The CSVs and overview support the written report;
known clean components are used only for retrospective interpretation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import ROOT, load_config, sha256, write_json, save_figure, plt


def main():
    config = load_config()
    results = ROOT / "results"
    output = results / "06_summary"
    output.mkdir(parents=True, exist_ok=True)
    cases = ["control", "clot_A", "clot_B", "clot_C"]
    feature_tables, band_tables, comparison = [], [], []
    input_hashes = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")

    for case in cases:
        stage_root = results if case in cases[:2] else results / "05_scenario_checks"
        manifest = json.loads((stage_root / "01_validation/manifest.json").read_text())
        if manifest["config_sha256"] != sha256(ROOT / "config.json"):
            raise ValueError("The saved results use a different configuration")
        for name, expected in manifest["source_files_sha256"].items():
            if sha256(Path(config["source_dir"]) / name) != expected:
                raise ValueError(f"An original source file has changed: {name}")

        observed_path = stage_root / "02_features" / f"{case}_features.csv"
        audit = stage_root / "03_noise_attribution" / case
        paths = {"observed": observed_path,
                 "features": audit / "feature_interval_summary.csv",
                 "bands": audit / "band_interval_summary.csv",
                 "effects": audit / "feature_noise_effects.csv"}
        for path in paths.values():
            input_hashes[str(path.relative_to(ROOT))] = sha256(path)
        observed, features, bands, effects = [pd.read_csv(p) for p in paths.values()]
        feature_tables.append(features.assign(case=case))
        band_tables.append(bands.assign(case=case))

        # These are physical intervals, not event times estimated by DWT.
        for interval, start, end in [("reference", 20, 200), ("late", 1500, 1700)]:
            row = features[(features.feature == "R6") & (features.interval == interval)].iloc[0]
            kept = effects[(effects.feature == "R6") & (effects.start_s >= start)
                           & (effects.end_s <= end) & ~effects.boundary_window]
            bias = kept.noise_induced_bias
            comparison.append({"case": case, "interval": interval,
                               "start_s": start, "end_s": end,
                               "observed_R6_mean_Pa": row.observed_mean,
                               "clean_R6_mean_Pa": row.clean_mean,
                               "signed_bias_mean_Pa": bias.mean(),
                               "absolute_bias_mean_Pa": bias.abs().mean(),
                               "absolute_bias_max_Pa": bias.abs().max()})

        interior = observed.loc[~observed.boundary_window]
        axes[0].plot(interior.mid_s, interior.R6, label=case, lw=1.2)

    features = pd.concat(feature_tables, ignore_index=True)
    bands = pd.concat(band_tables, ignore_index=True)
    features.to_csv(output / "all_cases_feature_intervals.csv", index=False)
    bands.to_csv(output / "all_cases_band_intervals.csv", index=False)
    pd.DataFrame(comparison).to_csv(output / "R6_comparison.csv", index=False)

    axes[0].set(title="Observed D6 RMS: strong growth in A, B and C",
                xlabel="Window midpoint (s)", ylabel="R6 (Pa; logarithmic axis)",
                yscale="log", xlim=(0, 1800))
    axes[0].legend()
    positions = np.arange(len(cases))
    for offset, interval, label in [(-0.18, "reference", "20-200 s"),
                                    (0.18, "late", "1500-1700 s")]:
        selected = bands[(bands.level == 1) & (bands.interval == interval)].set_index("case")
        values = selected.loc[cases, "pooled_known_component_snr_dB"]
        axes[1].bar(positions + offset, values, width=0.36, label=label)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xticks=positions, xticklabels=cases, ylabel="10 log10(E_clean / E_noise) (dB)",
                title="D1 component balance changes during growth")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Offline db2, 8 levels, 10 s windows; first/last 20 s excluded\n"
                 "Right: known-generator audit, not blind noise identification or an event threshold",
                 fontsize=11)
    save_figure(fig, output / "overview.png")
    write_json(output / "summary_manifest.json", {
        "source_files_unchanged": True, "config_sha256": sha256(ROOT / "config.json"),
        "input_sha256": input_hashes,
        "scope": "Descriptive aggregation of saved stages 1-5; no new detector or parameter choice",
        "R6_bias_note": "Signed means can cancel; absolute mean and maximum are also saved."
    })
    print(f"Saved combined tables and overview to {output}")


if __name__ == "__main__":
    main()
