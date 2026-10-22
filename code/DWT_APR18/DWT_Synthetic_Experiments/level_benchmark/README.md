# APR18 early STEP: A3–A7 comparison

This is a separate, offline experiment. It compares approximation-rate features
at fixed 10 ms windows and includes an ordinary piecewise-linear mean-rate baseline.
It does not label real APR18 changes as ground truth.

## Run

```bash
python run_experiment.py
python validate_results.py
python write_report.py
```

Dependencies: numpy, pandas, scipy, PyWavelets, matplotlib.
The previous experiment's A5 result is included in inputs to check
implementation equivalence. Synthetic and real data retain physical units.

## Design

- 120 development positives and 99 independent development negatives.
- 240 frozen test positives and 200 test negatives.
- Positive shapes: half-cosine growth over 20 ms or 150 ms, onset 0.35–0.60 s.
- Amplitudes: 0.002, 0.006, 0.012 Pa. Gaussian correlated noise standard deviations:
  0.0003, 0.001, 0.002 Pa; 5 ms correlation constant. These are assumptions.
- Each observed signal has a dense version and a paired version sampled at all
  65 APR18 measurement times, then linearly reconstructed.
- db2 A3–A7, symmetric extension, physical reconstruction, 10 ms window means/rates.
- All detection uses a common boundary-screened interval. Full 63 s signals enter
  DWT but only the first second enters onset scoring.
- First pair of consecutive positive-rate exceedances defines a candidate.
  This is offline: the reported midpoint uses future observations.
- Each feature/arm has its own threshold from 99 development negative maxima.
  The rank-based threshold targets a 5% probability of any false candidate per
  negative record under the same null distribution. Test rates can exceed 5%.
- Positive match tolerance: ±50 ms. A missing or mistimed first candidate receives
  a 0.2 s failure penalty; otherwise the loss is absolute timing error.
- Select a DWT level by mean development loss in the APR18-sampled arm before
  evaluating test predictions. The ordinary baseline remains a comparator.
- Sensitivity uses the first 60 positives and first 60 negatives of the test
  manifest with fixed thresholds. No retuning follows the test evaluation.

## Outputs

1. `01_protocol`: settings, frozen case manifest, digests.
2. `02_features`: all simulated window features (compressed), real-data features.
3. `03_evaluation`: thresholds, predictions, metrics, uncertainty and checks.
4. `04_figures`: four result plots.
5. `05_report`: readable results and run log.

## Important limits

The benchmark excludes the first ~0.27 s to compare all levels fairly. It cannot
establish optimality for the earliest observed APR18 transient. The negative
control is a constant signal with simulated noise, not a measured blood noise
reference. The slow synthetic rise is 150 ms, not the earlier 40 s clot-growth
experiment. Fixed 10 ms averaging and the simple detector are part of the method
being evaluated. This is not a claim about all wavelets, event types or windows.

Chinese results: `05_report/RESULTS_ZH.md`.
Install recorded dependencies with `python -m pip install -r requirements.txt`.
