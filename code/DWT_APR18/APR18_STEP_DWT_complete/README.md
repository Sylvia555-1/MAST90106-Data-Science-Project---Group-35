# APR18 step DWT features versus synthetic step features

This offline Python experiment studies the full APR18 0.1-to-1/s response
(steps!BH4:BK68), using the scope confirmed by the user. Read `RESULTS_ZH.md`
for findings and limitations. It does not estimate real-data detection accuracy.

## Run

```bash
/opt/anaconda3/bin/python /Users/jerry_t0mato/Desktop/codex/apr18_step_dwt_comparison_20261008/run_experiment.py
```

Required libraries: numpy, pandas, PyWavelets, openpyxl, matplotlib.
The scripts contain the original source locations near the top; change those
paths when moving the project. No package installation or source modification
is performed. `Experiment.ipynb` provides an optional readable entry point.

## Files

- `01_audit_sources.py`: check actual workbook blocks, duplicates, units, sampling,
  and the synthetic file schema; export all original samples.
- `dwt_features.py`: reconstruct physical-unit DWT components, summarize windows,
  count original observations, and compare the same windows.
- `02_extract_and_compare.py`: extract features; match physical scale; check grid,
  alignment, and boundaries; reproduce previous synthetic features; investigate
  synthetic sampling at the APR18 timestamps.
- `03_plot_results.py`: produce the four research figures and numerical summary.
- `01_source_audit/`: raw exports, workbook inventory, observation support.
- `02_features/`: base features for stress and measured input, synthetic features.
- `03_checks/`: numerical checks and sensitivity variants.
- `04_figures/`: results as PNG files.
- `05_comparison/`: physical scale mapping, interval summaries, sampling diagnostic.

## Reading feature columns

`A5` and `A13` in window CSV files are **means of reconstructed approximation
components**. `R6` through `R13` are **RMS strengths of reconstructed D6 through
D13**, not signed detail coefficients. Stress features use Pa; input features
use 1/s. A `_rate` suffix means the difference of consecutive window means divided
by midpoint spacing; stress rates have units Pa/s. The rate is labeled at the
later midpoint and is not an instantaneous derivative.

`grid_mean` uses the uniform computational grid. For APR18 this grid is linearly
interpolated. `linear_mean` is the exact time-average of the piecewise-linear
curve through original points; it is not an arithmetic mean of irregular samples.
`original_points` counts actual source timestamps in [left_s,right_s).
`max_original_gap_s` includes the source interval bracketing a window even when
the window itself has no measurement. `_gap_exceeds_scale` is an observation
support warning, not a proof that smaller gaps ensure adequate resolution.

`_interior` excludes 6.25 times the physical scale spacing from both record ends.
For rate features, both windows must pass. This conservative engineering screen
does not establish the true sample history. Masked values remain in CSV files.

No amplitude normalization, threshold denoising, causal inference, or ML is used.
The original measurement times are preserved in the raw exports; source data are
not overwritten. Dense interpolation values are explicitly computed, not measured.
