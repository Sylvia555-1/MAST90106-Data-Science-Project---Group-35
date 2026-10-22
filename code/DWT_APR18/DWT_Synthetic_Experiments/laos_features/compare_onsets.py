"""Compare two exploratory onset rules with the same calibration procedure.

Use control windows only to calibrate each feature's threshold. The known
clot_A onset is read after the detections and used only for descriptive error.
Offline DWT uses future samples: none of these times is a real-time alarm.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '07_onset_comparison'
FEATURES = ['R6_Pa', 'raw_RMS_Pa']


def find_runs(frame, feature, threshold, required=3):
    """Return maximal exceedance runs with at least three adjacent windows."""
    flag = frame[feature].to_numpy() > threshold
    starts = np.flatnonzero(flag & ~np.r_[False, flag[:-1]])
    ends = np.flatnonzero(flag & ~np.r_[flag[1:], False])
    events = []
    for start, end in zip(starts, ends):
        if end - start + 1 < required:
            continue
        first, confirmed = frame.iloc[start], frame.iloc[start + required - 1]
        events.append(dict(first_window_left_s=float(first.left_s),
                           first_window_right_s=float(first.right_s),
                           estimated_onset_midpoint_s=float(first.mid_s),
                           third_window_right_s=float(confirmed.right_s),
                           consecutive_windows=int(end - start + 1)))
    return events


def main():
    OUT.mkdir(exist_ok=True)
    # Check persistence handling, including a run at the end of the record.
    toy = pd.DataFrame({'x': [0, 2, 2, 0, 2, 2, 2], 'left_s': np.arange(7),
                        'right_s': np.arange(7) + 10, 'mid_s': np.arange(7) + 5})
    assert len(find_runs(toy, 'x', 1)) == 1
    assert find_runs(toy, 'x', 1)[0]['first_window_left_s'] == 4
    assert not find_runs(toy.assign(x=0), 'x', 1)

    control = pd.read_csv(ROOT / '03_observed_features/control.csv')
    baseline = control[(control.left_s >= 20) & (control.right_s <= 200)]
    thresholds = []
    for feature in FEATURES:
        center = float(baseline[feature].median())
        mad = float(np.median(np.abs(baseline[feature] - center)))
        assert mad > 0
        for multiplier in [3, 6, 9]:
            thresholds.append(dict(feature=feature, k=multiplier, baseline_median_Pa=center,
                                   baseline_MAD_Pa=mad, threshold_Pa=center + multiplier * 1.4826 * mad))
    pd.DataFrame(thresholds).to_csv(OUT / 'thresholds.csv', index=False)
    records, all_events = [], {}
    for alignment in ['full', 'crop_one_cycle']:
        for case in ['control', 'clot_A']:
            path = ROOT / ('03_observed_features' if alignment == 'full' else '06_validation') / (f'{case}.csv' if alignment == 'full' else f'{case}_crop_one_cycle.csv')
            frame = pd.read_csv(path)
            # Use the same absolute monitoring interval and frozen thresholds.
            frame = frame[(frame.left_s >= 200) & (frame.right_s <= 1780)].reset_index(drop=True)
            assert np.allclose(np.diff(frame.left_s), 1)
            for setting in thresholds:
                events = find_runs(frame, setting['feature'], setting['threshold_Pa'])
                key = f"{alignment}/{case}/{setting['feature']}/k{setting['k']}"
                all_events[key] = events
                row = dict(case=case, alignment=alignment, feature=setting['feature'], k=setting['k'],
                           candidate_run_count=len(events), threshold_Pa=setting['threshold_Pa'])
                if events:
                    row.update(events[0])
                records.append(row)
    result = pd.DataFrame(records)
    # Truth is consulted only after all threshold choices and detections above.
    meta = json.loads((Path(os.environ.get('DWT_LAOS_DATA', str(Path(__file__).resolve().parents[1] / 'data/laos_synthetic'))) / 'metadata.json').read_text())
    onset = meta['scenario']['onset_and_plateau_by_case']['clot_A']['onset_s']
    result['declared_onset_s'] = np.where(result.case == 'clot_A', onset, np.nan)
    result['midpoint_error_s'] = result.estimated_onset_midpoint_s - result.declared_onset_s
    result.to_csv(OUT / 'results.csv', index=False)
    (OUT / 'candidate_runs.json').write_text(json.dumps(all_events, indent=2))
    (OUT / 'settings.json').write_text(json.dumps(dict(
        primary_k=6, sensitivity_k=[3, 9], required_consecutive_windows=3,
        baseline='control only; windows contained in [20, 200) seconds',
        monitoring='windows contained in [200, 1780) seconds',
        threshold='median + k * 1.4826 * median absolute deviation',
        estimated_onset='midpoint of first window in first qualifying run',
        window_s=10, stride_s=1, three_windows_cover_s=12,
        false_positive_scope='candidate runs on one control record, not a population false-alarm rate',
        caveats=['Exploratory: source metadata and truth have already been reviewed.',
                 'A midpoint error is a convention-dependent summary, not proven temporal resolution.',
                 'Full-record DWT and centered windows can give early responses using future samples.',
                 'Threshold multipliers are working choices, not calibrated probability levels.'],
        persistence_check_passed=True
    ), indent=2))
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
