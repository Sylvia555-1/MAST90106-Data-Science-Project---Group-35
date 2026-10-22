# DWT experiments on synthetic STEP and oscillatory data

Python code used for the Group 35 DWT studies. This archive supplements the real APR18 code in `code/DWT_APR18`. Each experiment keeps its own outputs. Comments and plots are in English; archived interpretation reports are in Chinese.

## Why synthetic data?

Real APR18 records do not provide independently verified material-change times or separate clean/noise signals. Synthetic signals provide declared onsets, clean references and no-change controls for checking feature behaviour. They do not validate a biological mechanism or establish real blood detection accuracy.

## Contents and reported findings

| Folder | Question | Main result |
|---|---|---|
| `step_growth` | Which features follow prescribed extra growth after loading settles? Compare levels 7–9. | A8 rate follows slow growth; ordinary mean rate also does. Smoothing and onset timing are different criteria. |
| `laos_features` | Compare R6 with ordinary RMS for extra-response onset. | On the control/clot_A pair, R6 and RMS both have 2 s absolute midpoint error (302 s and 298 s versus 300 s). This is offline, not a real-time alarm. |
| `noise_controls` | Does feature change reflect known response or noise? | Clean constant-material R6 also fluctuates with alignment; fine details are not automatically noise. No denoising is applied. |
| `level_benchmark` | Compare A3–A7 with fixed development/test rules under dense and APR18 sampling. | A5 is not an established optimum. Development selects A7, mainly through fewer failures; real onset error remains unknown. |

## Setup

Python 3.11 or later is recommended. Install dependencies in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

The three supplied source datasets are **not bundled**. Copy the original folders into `data/`, retaining their original layout, or set these environment variables to their absolute paths:

```bash
export DWT_STEP_DATA="/path/to/step_data"
export DWT_LAOS_DATA="/path/to/laos_synthetic"
export DWT_NOISE_DATA="/path/to/coagulation_synthetic"
```

Required inputs:

- STEP: `manifest.json`, `test/control_1.csv`, `test/test_clot_1.csv`, and corresponding `ground_truth/*_truth.csv`.
- LAOS: `metadata.json`, `observable/control.csv`, `observable/clot_A.csv`, and corresponding `truth/*_truth.csv`. The optional full audit also reads clot_B/C.
- Initial noise study: original `control.csv`, `clot_A.csv`, `clot_B.csv`, `clot_C.csv`, and simulator metadata/support files in the supplied folder.
- Level benchmark: small APR18 time/stress and prior-feature reference tables are included in `level_benchmark/inputs`; the script generates the benchmark signals with fixed seeds. This benchmark is distinct from the supplied slow-growth dataset.

Run commands from the archive root:

### 1. Synthetic STEP

```bash
python step_growth/run_features.py
python step_growth/compare_levels.py
python step_growth/plot_A8_explanation.py
```

Results go to `step_growth/results/` (observed features, truth review, sensitivity, figures, validation and level comparison).

### 2. Synthetic LAOS

```bash
python laos_features/explore_features.py
python laos_features/compare_onsets.py
# Optional four-record audit:
python laos_features/audit_data.py
```

Results are separated into `03_observed_features`, `04_truth_review`, `05_figures`, `06_validation`, and `07_onset_comparison`. The onset script needs the feature script's output. Control-only calibration precedes truth-based error evaluation; this remains an exploratory pair comparison.

### 3. Initial noise-controls experiment

```bash
python noise_controls/self_check.py
python noise_controls/analyze_observed.py
python noise_controls/evaluate_known_components.py
python noise_controls/run_sensitivity.py
python noise_controls/analyze_observed.py --cases clot_B clot_C --output-root noise_controls/results/05_scenario_checks
python noise_controls/evaluate_known_components.py --cases clot_B clot_C --output-root noise_controls/results/05_scenario_checks
python noise_controls/independent_review.py
python noise_controls/summarize_results.py
```

Outputs are grouped into `noise_controls/results/01_validation` through `06_summary`. The clean reference is reconstructed from known simulation parameters; it is not estimated by DWT from observations alone. Do not regenerate the supplied datasets using an unrelated or unavailable generator.

### 4. Controlled level-selection benchmark

```bash
python level_benchmark/run_experiment.py
python level_benchmark/validate_results.py
python level_benchmark/write_report.py
```

This is the most expensive experiment. Outputs are grouped into protocol, features, evaluation, figures and report folders. See its README for splits, thresholds, timing tolerance and failure penalty.

## Reproducibility and limits

- Calculation logic, wavelets, windows, thresholds and experimental settings are preserved from the original scripts. Packaging replaces machine-specific input paths with environment variables or relative paths.
- Re-running overwrites generated stage outputs, never input CSVs. Run in a fresh extracted folder to preserve an earlier run.
- `RESULTS_ZH.md` files describe the original reviewed runs; they are not promises of performance on new data.
- `PACKAGE_VALIDATION.json` records checks actually performed during packaging. The entire historical suite was not rerun merely to package it.
- Full-record DWT and overlapping windows use future data. Feature peaks, midpoint onsets and confirmation-window ends are different quantities.
- No training or clinical validation is claimed. Neither a high-frequency detail nor a high correlation establishes a material mechanism.

## Provenance

Original experiment folders: `step_dwt_growth_features_20261006`, `laos_dwt_feature_study_20261007`, `coagulation_dwt_feature_noise_study`, `apr18_step_level_comparison_20261011`. This archive contains analysis code, not a new synthetic data generator for the supplied STEP/LAOS datasets.
