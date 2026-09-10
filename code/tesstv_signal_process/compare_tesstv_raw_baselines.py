#!/usr/bin/env python3
"""Compare APR18, JUN18 and JUL18 with one fixed t-ESSTV parameter set.

This is a stage-one *raw baseline* comparison.  It deliberately does not fit
or optimise any parameter.  Every workbook is simulated with the published
Donor 5 tensorial t-ESSTV parameters from Armstrong et al. (2022), Table 6.

Keep this file beside ``tensorial_esstv_figure5_v3.py``.

Example
-------
python3 compare_tesstv_raw_baselines.py \
  --apr "APR18 SS SAOS STEPS LAOS UDLAOS.xlsx" \
  --jun "JUN18 SS SAOS STEPS LAOS UDLAOS.xlsx" \
  --jul "JUL18 SS SAOS STEPS LAOS UDLAOS.xlsx" \
  --output "three_dataset_raw_baseline"

Main outputs
------------
raw_baseline_summary.csv
    One row per workbook: SS metrics, all-protocol transient metrics and a
    fair common-protocol transient comparison.
raw_baseline_per_protocol.csv
    Metrics for every usable measured step protocol.
raw_baseline_time_regions.csv
    Residual metrics in early (<0.01 s), transition (0.01-1 s), and late
    (>=1 s) regions.
raw_baseline_ss_blocks.csv
    Every detected SS block, kept separate (for example original and NEW).
raw_baseline_protocol_audit.csv
    Workbook titles versus measured shear-rate endpoints.
raw_baseline_comparison.png/pdf
raw_baseline_protocol_costs.png/pdf
raw_baseline_time_region_rmse.png/pdf

Scientific interpretation
-------------------------
The APR/JUN/JUL comparison is a fixed-parameter transfer test, not a fit of
the t-ESSTV parameters to each workbook.  The paper values 0.0043 (SS) and
0.0232 (transient) are shown only as Donor 5 reference values.  If protocol
sets differ, the script reports both all-valid-protocol metrics and metrics on
the intersection of measured protocols shared by all three workbooks.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tensorial_esstv_figure5_v3 as tesstv
except ImportError as exc:
    raise SystemExit(
        "Could not import tensorial_esstv_figure5_v3.py. Put that file in "
        "the same folder as compare_tesstv_raw_baselines.py."
    ) from exc


EARLY_END_S = 0.01
TRANSITION_END_S = 1.0


@dataclass
class DatasetRun:
    label: str
    workbook: Path
    active_ss_block: str
    summary: dict[str, object]
    simulations: dict[tuple[float, float], pd.DataFrame]
    protocol_metrics: pd.DataFrame
    region_metrics: pd.DataFrame
    ss_block_metrics: pd.DataFrame
    protocol_audit: pd.DataFrame


def safe_slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return value or "dataset"


def protocol_name(key: tuple[float, float]) -> str:
    return f"{key[0]:g}->{key[1]:g}"


def protocol_sort_key(key: tuple[float, float]) -> tuple[int, float, float]:
    pre, target = key
    return (0 if target > pre else 1, pre, target)


def choose_unique_measured_protocols(
    tests: list[tesstv.StepTest],
) -> list[tesstv.StepTest]:
    """Use measured endpoints, deduplicate protocols and prefer clean titles."""

    groups: dict[tuple[float, float], list[tesstv.StepTest]] = {}
    for test in tests:
        key = (float(test.pre_rate), float(test.target_rate))
        if not np.all(np.isfinite(key)) or tesstv.rate_close(key[0], key[1]):
            continue
        groups.setdefault(key, []).append(test)

    selected: list[tesstv.StepTest] = []
    for key in sorted(groups, key=protocol_sort_key):
        candidates = groups[key]
        # Prefer a title that agrees with measured shear rate, then more data.
        chosen = max(
            candidates,
            key=lambda item: (not bool(item.protocol_warning), len(item.data)),
        )
        selected.append(chosen)
    return selected


def combined_metrics(
    simulations: dict[tuple[float, float], pd.DataFrame]
) -> dict[str, float]:
    data_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for frame in simulations.values():
        data = frame["stress_data"].to_numpy(float)
        prediction = frame["stress_model"].to_numpy(float)
        valid = np.isfinite(data) & np.isfinite(prediction)
        if np.any(valid):
            data_parts.append(data[valid])
            prediction_parts.append(prediction[valid])
    if not data_parts:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    return tesstv.metrics(
        np.concatenate(data_parts), np.concatenate(prediction_parts)
    )


def aggregate_subset(
    simulations: dict[tuple[float, float], pd.DataFrame],
    keys: set[tuple[float, float]] | None = None,
) -> tuple[dict[str, object], dict[str, float]]:
    subset = {
        key: frame
        for key, frame in simulations.items()
        if keys is None or key in keys
    }
    return tesstv.aggregate_transient_costs(subset), combined_metrics(subset)


def time_region_name(time_s: float) -> str:
    if time_s < EARLY_END_S:
        return "early_<0.01s"
    if time_s < TRANSITION_END_S:
        return "transition_0.01-1s"
    return "late_>=1s"


def time_region_rows(
    dataset: str,
    key: tuple[float, float],
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_all = frame["stress_data"].to_numpy(float)
    maximum = float(np.nanmax(np.abs(data_all)))
    work = frame.copy()
    work["time_region"] = [time_region_name(value) for value in work["time"]]

    for region in ("early_<0.01s", "transition_0.01-1s", "late_>=1s"):
        part = work.loc[work["time_region"] == region]
        data = part["stress_data"].to_numpy(float)
        prediction = part["stress_model"].to_numpy(float)
        valid = np.isfinite(data) & np.isfinite(prediction)
        data = data[valid]
        prediction = prediction[valid]
        if not len(data):
            continue
        residual = prediction - data
        score = tesstv.metrics(data, prediction)
        normalized = (
            np.nan
            if maximum <= tesstv.EPS
            else float(np.sqrt(np.mean((residual / maximum) ** 2)))
        )
        rows.append(
            {
                "dataset": dataset,
                "protocol": protocol_name(key),
                "direction": "step_up" if key[1] > key[0] else "step_down",
                "time_region": region,
                **score,
                "normalized_rms_using_protocol_max": normalized,
                "mean_residual_prediction_minus_measurement_Pa": float(
                    np.mean(residual)
                ),
                "max_abs_residual_Pa": float(np.max(np.abs(residual))),
            }
        )
    return rows


def evaluate_ss_blocks(
    dataset: str,
    blocks: dict[str, pd.DataFrame],
    active_block: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rates = np.unique(
        np.concatenate(
            [block["shear_rate"].to_numpy(float) for block in blocks.values()]
        )
    )
    shared_model = tesstv.tensorial_steady_curve(
        all_rates, tesstv.PAPER_PARAMS
    ).set_index("shear_rate")
    model_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    reference_name = "original" if "original" in blocks else next(iter(blocks))
    reference = blocks[reference_name].set_index("shear_rate")["stress"]

    for name, block in blocks.items():
        model = shared_model.loc[block["shear_rate"].to_numpy(float)].reset_index()
        model["stress_data"] = block["stress"].to_numpy(float)
        model["dataset"] = dataset
        model["ss_block"] = name
        model["selected_for_summary"] = name == active_block
        if "excel_row" in block:
            model["excel_row"] = block["excel_row"].to_numpy(int)
        model_rows.append(model)

        score = tesstv.metrics(model["stress_data"], model["stress_model"])
        cost = tesstv.steady_costs(model["stress_data"], model["stress_model"])
        aligned = reference.reindex(block["shear_rate"].to_numpy(float)).to_numpy()
        exact = int(np.sum(block["stress"].to_numpy(float) == aligned))
        metric_rows.append(
            {
                "dataset": dataset,
                "ss_block": name,
                "selected_for_summary": name == active_block,
                "reference_block": reference_name,
                "n_exact_stress_values_matching_reference": exact,
                "n_stress_values_different_from_reference": len(block) - exact,
                **score,
                **cost,
                "paper_donor5_fcost_ss_reference": tesstv.PAPER_FCOST_SS,
                "eq10_ratio_to_paper_reference": (
                    cost["fcost_ss_eq10_relative_rms"]
                    / tesstv.PAPER_FCOST_SS
                ),
            }
        )

    all_models = pd.concat(model_rows, ignore_index=True)
    all_models.to_csv(output_dir / "steady_state_predictions.csv", index=False)
    return all_models, pd.DataFrame(metric_rows)


def run_one_dataset(
    label: str,
    workbook: Path,
    output_root: Path,
    ss_sheet: str | None,
    step_sheet: str | None,
    ss_cols: tuple[int, int],
    ss_block_request: str,
) -> DatasetRun:
    dataset_dir = output_root / safe_slug(label)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    blocks = tesstv.read_steady_state_blocks(workbook, ss_sheet, ss_cols)
    active_name, _ = tesstv.choose_steady_state_block(blocks, ss_block_request)
    _, ss_metrics = evaluate_ss_blocks(label, blocks, active_name, dataset_dir)
    active_ss = ss_metrics.loc[ss_metrics["ss_block"] == active_name].iloc[0]

    all_tests = tesstv.read_step_tests(workbook, step_sheet)
    selected_tests = choose_unique_measured_protocols(all_tests)
    if not selected_tests:
        raise ValueError(f"No usable step protocols were detected in {workbook}.")

    audit = tesstv.protocol_audit_table(all_tests, selected_tests)
    audit.insert(0, "dataset", label)
    audit.to_csv(dataset_dir / "protocol_audit.csv", index=False)

    simulations: dict[tuple[float, float], pd.DataFrame] = {}
    protocol_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    for test in selected_tests:
        key = (float(test.pre_rate), float(test.target_rate))
        result = tesstv.simulate_step(test, tesstv.PAPER_PARAMS)
        simulations[key] = result
        result.to_csv(
            dataset_dir
            / f"step_{safe_slug(protocol_name(key).replace('->', '_to_'))}.csv",
            index=False,
        )

        score = tesstv.metrics(result["stress_data"], result["stress_model"])
        cost = tesstv.transient_costs_for_frame(result)
        residual = (
            result["stress_model"].to_numpy(float)
            - result["stress_data"].to_numpy(float)
        )
        maximum_index = int(np.nanargmax(np.abs(residual)))
        protocol_rows.append(
            {
                "dataset": label,
                "protocol": protocol_name(key),
                "initial_rate_s-1": key[0],
                "target_rate_s-1": key[1],
                "direction": "step_up" if key[1] > key[0] else "step_down",
                "excel_column_one_based": test.source_column + 1,
                "source_title": test.title,
                "protocol_warning": test.protocol_warning,
                **score,
                **cost,
                "max_abs_residual_Pa": float(np.nanmax(np.abs(residual))),
                "time_of_max_abs_residual_s": float(
                    result["time"].iloc[maximum_index]
                ),
            }
        )
        region_rows.extend(time_region_rows(label, key, result))

    all_cost, all_score = aggregate_subset(simulations)
    summary = {
        "dataset": label,
        "workbook": str(workbook),
        "parameter_source": "Armstrong et al. Table 6 Donor 5; fixed",
        "parameters_optimised_on_this_dataset": False,
        "active_ss_block": active_name,
        "n_ss_points": int(active_ss["n"]),
        "ss_rmse_Pa": float(active_ss["rmse"]),
        "ss_mae_Pa": float(active_ss["mae"]),
        "ss_r2": float(active_ss["r2"]),
        "ss_fcost_eq10": float(active_ss["fcost_ss_eq10_relative_rms"]),
        "paper_donor5_ss_cost_reference": tesstv.PAPER_FCOST_SS,
        "ss_cost_ratio_to_paper_reference": float(
            active_ss["fcost_ss_eq10_relative_rms"] / tesstv.PAPER_FCOST_SS
        ),
        "n_valid_protocols_all": int(all_cost["n_protocols"]),
        "n_transient_points_all": int(all_cost["n_points"]),
        "protocols_all": all_cost["protocols"],
        "transient_rmse_all_points_Pa": float(all_score["rmse"]),
        "transient_mae_all_points_Pa": float(all_score["mae"]),
        "transient_r2_all_points": float(all_score["r2"]),
        "transient_fcost_eq11_all": float(all_cost["fcost_trans_eq11_rmse_Pa"]),
        "transient_fcost_eq12_all": float(
            all_cost["fcost_trans_eq12_normalized_rms"]
        ),
        "paper_donor5_transient_cost_reference": tesstv.PAPER_FCOST_TRANS,
        "transient_eq12_all_ratio_to_paper_reference": float(
            all_cost["fcost_trans_eq12_normalized_rms"]
            / tesstv.PAPER_FCOST_TRANS
        ),
    }

    return DatasetRun(
        label=label,
        workbook=workbook,
        active_ss_block=active_name,
        summary=summary,
        simulations=simulations,
        protocol_metrics=pd.DataFrame(protocol_rows),
        region_metrics=pd.DataFrame(region_rows),
        ss_block_metrics=ss_metrics,
        protocol_audit=audit,
    )


def add_common_protocol_metrics(
    runs: list[DatasetRun],
) -> tuple[pd.DataFrame, set[tuple[float, float]]]:
    common = set(runs[0].simulations)
    for run in runs[1:]:
        common &= set(run.simulations)

    rows: list[dict[str, object]] = []
    for run in runs:
        row = dict(run.summary)
        common_cost, common_score = aggregate_subset(run.simulations, common)
        row.update(
            {
                "n_common_protocols": len(common),
                "common_protocols": "; ".join(
                    protocol_name(key)
                    for key in sorted(common, key=protocol_sort_key)
                ),
                "transient_rmse_common_Pa": (
                    float(common_score["rmse"]) if common else np.nan
                ),
                "transient_mae_common_Pa": (
                    float(common_score["mae"]) if common else np.nan
                ),
                "transient_r2_common": (
                    float(common_score["r2"]) if common else np.nan
                ),
                "transient_fcost_eq11_common": (
                    float(common_cost["fcost_trans_eq11_rmse_Pa"])
                    if common
                    else np.nan
                ),
                "transient_fcost_eq12_common": (
                    float(common_cost["fcost_trans_eq12_normalized_rms"])
                    if common
                    else np.nan
                ),
                "transient_eq12_common_ratio_to_paper_reference": (
                    float(common_cost["fcost_trans_eq12_normalized_rms"])
                    / tesstv.PAPER_FCOST_TRANS
                    if common
                    else np.nan
                ),
                "primary_transient_comparison_basis": (
                    "common measured protocols"
                    if common
                    else "all valid protocols; no common intersection"
                ),
                "primary_transient_fcost_eq12": (
                    float(common_cost["fcost_trans_eq12_normalized_rms"])
                    if common
                    else row["transient_fcost_eq12_all"]
                ),
                "primary_transient_r2": (
                    float(common_score["r2"])
                    if common
                    else row["transient_r2_all_points"]
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows), common


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["dataset"].tolist()
    colors = ["#4472C4", "#ED7D31", "#70AD47"][: len(labels)]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].bar(x, summary["ss_fcost_eq10"], color=colors)
    axes[0, 0].axhline(
        tesstv.PAPER_FCOST_SS,
        color="black",
        linestyle="--",
        label="Paper Donor 5 reference",
    )
    axes[0, 0].set_title("SS Eq. 10 relative RMS cost")
    axes[0, 0].set_ylabel("dimensionless cost")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x, summary["primary_transient_fcost_eq12"], color=colors)
    axes[0, 1].axhline(
        tesstv.PAPER_FCOST_TRANS,
        color="black",
        linestyle="--",
        label="Paper Donor 5 reference",
    )
    axes[0, 1].set_title("Transient Eq. 12 normalized cost")
    axes[0, 1].set_ylabel("dimensionless cost")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(x, summary["ss_r2"], color=colors)
    axes[1, 0].set_title("Steady-state R²")
    axes[1, 0].set_ylabel("R²")

    axes[1, 1].bar(x, summary["primary_transient_r2"], color=colors)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Transient R²")
    axes[1, 1].set_ylabel("R²")

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Fixed Donor 5 t-ESSTV parameters: raw-data baseline comparison",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "raw_baseline_comparison")


def plot_protocol_costs(protocols: pd.DataFrame, output_dir: Path) -> None:
    pivot = protocols.pivot_table(
        index="protocol",
        columns="dataset",
        values="fcost_trans_eq12_normalized_rms",
        aggfunc="first",
    )
    pivot = pivot.sort_index()
    fig, axis = plt.subplots(figsize=(max(9, 1.15 * len(pivot)), 5.5))
    pivot.plot(kind="bar", ax=axis, width=0.8)
    axis.axhline(
        tesstv.PAPER_FCOST_TRANS,
        color="black",
        linestyle="--",
        label="Paper Donor 5 reference",
    )
    axis.set_title("Transient Eq. 12 cost by measured protocol")
    axis.set_xlabel("protocol (s⁻¹)")
    axis.set_ylabel("normalized RMS cost")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "raw_baseline_protocol_costs")


def plot_time_regions(regions: pd.DataFrame, output_dir: Path) -> None:
    order = ["early_<0.01s", "transition_0.01-1s", "late_>=1s"]
    grouped = (
        regions.groupby(["dataset", "time_region"], as_index=False)["rmse"]
        .mean()
        .pivot(index="time_region", columns="dataset", values="rmse")
        .reindex(order)
    )
    fig, axis = plt.subplots(figsize=(9, 5.5))
    grouped.plot(kind="bar", ax=axis, width=0.78)
    axis.set_title("Mean transient RMSE by time region")
    axis.set_xlabel("time region")
    axis.set_ylabel("RMSE (Pa)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="dataset")
    fig.tight_layout()
    save_figure(fig, output_dir, "raw_baseline_time_region_rmse")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate APR18, JUN18 and JUL18 using fixed published Donor 5 "
            "tensorial t-ESSTV parameters."
        )
    )
    parser.add_argument("--apr", required=True, help="Path to APR18 workbook")
    parser.add_argument("--jun", required=True, help="Path to JUN18 workbook")
    parser.add_argument("--jul", required=True, help="Path to JUL18 workbook")
    parser.add_argument(
        "--output",
        default="three_dataset_raw_baseline",
        help="Output directory (default: three_dataset_raw_baseline)",
    )
    parser.add_argument("--ss-sheet", help="Override the SS sheet name")
    parser.add_argument("--step-sheet", help="Override the step sheet name")
    parser.add_argument(
        "--ss-cols",
        default="0,1",
        help="Zero-based SS shear-rate,stress columns (default: 0,1)",
    )
    parser.add_argument(
        "--ss-block",
        default="auto",
        help="SS block for summary/figures: auto, original, new, or block name",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workbooks = {
        "APR18": Path(args.apr).expanduser().resolve(),
        "JUN18": Path(args.jun).expanduser().resolve(),
        "JUL18": Path(args.jul).expanduser().resolve(),
    }
    missing = [f"{name}: {path}" for name, path in workbooks.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Workbook(s) not found:\n  " + "\n  ".join(missing))

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ss_cols = tesstv.parse_col_pair(args.ss_cols)

    runs: list[DatasetRun] = []
    for label, workbook in workbooks.items():
        print(f"\nRunning {label}: {workbook}")
        runs.append(
            run_one_dataset(
                label=label,
                workbook=workbook,
                output_root=output_dir,
                ss_sheet=args.ss_sheet,
                step_sheet=args.step_sheet,
                ss_cols=ss_cols,
                ss_block_request=args.ss_block,
            )
        )

    summary, common = add_common_protocol_metrics(runs)
    protocols = pd.concat([run.protocol_metrics for run in runs], ignore_index=True)
    regions = pd.concat([run.region_metrics for run in runs], ignore_index=True)
    ss_blocks = pd.concat([run.ss_block_metrics for run in runs], ignore_index=True)
    audits = pd.concat([run.protocol_audit for run in runs], ignore_index=True)

    summary.to_csv(output_dir / "raw_baseline_summary.csv", index=False)
    protocols.to_csv(output_dir / "raw_baseline_per_protocol.csv", index=False)
    regions.to_csv(output_dir / "raw_baseline_time_regions.csv", index=False)
    ss_blocks.to_csv(output_dir / "raw_baseline_ss_blocks.csv", index=False)
    audits.to_csv(output_dir / "raw_baseline_protocol_audit.csv", index=False)

    plot_summary(summary, output_dir)
    plot_protocol_costs(protocols, output_dir)
    plot_time_regions(regions, output_dir)

    print("\nFixed-parameter raw baseline completed")
    print(f"Output: {output_dir}")
    print(
        "Common measured protocols: "
        + (
            ", ".join(
                protocol_name(key) for key in sorted(common, key=protocol_sort_key)
            )
            if common
            else "none; primary transient comparison uses each dataset's valid protocols"
        )
    )
    columns = [
        "dataset",
        "active_ss_block",
        "ss_rmse_Pa",
        "ss_r2",
        "ss_fcost_eq10",
        "n_valid_protocols_all",
        "primary_transient_fcost_eq12",
        "primary_transient_r2",
        "primary_transient_comparison_basis",
    ]
    print("\nSummary:")
    print(summary[columns].to_string(index=False))
    print(
        "\nInterpretation: these are fixed-paper-parameter transfer results, "
        "not dataset-specific best fits."
    )


if __name__ == "__main__":
    main()
