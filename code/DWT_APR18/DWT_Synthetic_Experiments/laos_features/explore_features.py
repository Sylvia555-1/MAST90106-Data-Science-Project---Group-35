"""Screen a small set of DWT features on control and clot_A.

This is an offline, exploratory comparison, not a fitted change detector.
Reconstruct each DWT component on the original time grid, then summarize
10-second windows every second. Keep observed and truth-review outputs apart.
"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('DWT_LAOS_DATA', str(Path(__file__).resolve().parents[1] / 'data/laos_synthetic')))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.plot_cache'))
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 100
WINDOW = 1000
STRIDE = 100


def window_mean(values):
    """Use all original points; [left, right) windows overlap by 9 seconds."""
    blocks = np.lib.stride_tricks.sliding_window_view(values, WINDOW)[::STRIDE]
    return blocks.mean(axis=1)


def extract_features(time, stress):
    """Observable-only feature extraction. No truth or event time is used."""
    coefficients = pywt.wavedec(stress, 'db2', mode='symmetric', level=8)
    components = []
    for index in range(9):
        selected = [np.zeros_like(c) for c in coefficients]
        selected[index] = coefficients[index]
        components.append(pywt.waverec(selected, 'db2', mode='symmetric')[:len(stress)])
    error = float(np.max(np.abs(np.sum(components, axis=0) - stress)))
    assert error < 1e-10
    left = time[np.arange(0, len(time) - WINDOW + 1, STRIDE)]
    result = pd.DataFrame({'left_s': left, 'right_s': left + 10, 'mid_s': left + 5})
    result['interior'] = (left >= time[0] + 16) & (left + 10 <= time[-1] + .01 - 16)
    result['raw_RMS_Pa'] = np.sqrt(window_mean(stress ** 2))
    result['raw_mean_Pa'] = window_mean(stress)
    result['A8_mean_Pa'] = window_mean(components[0])
    energies = []
    for level in range(1, 9):
        energy = window_mean(components[-level] ** 2)
        energies.append(energy)
        result[f'R{level}_Pa'] = np.sqrt(energy)
    total_detail_energy = np.sum(energies, axis=0)
    assert np.all(total_detail_energy > 0)
    result['P6'] = energies[5] / total_detail_energy
    result['H13_Pa'] = np.sqrt(energies[0] + energies[1] + energies[2])
    result['R45_Pa'] = np.sqrt(energies[3] + energies[4])
    assert np.all(result.R6_Pa > 0)
    result['R45_over_R6'] = result.R45_Pa / result.R6_Pa
    result['R6_rate_Pa_s'] = result.R6_Pa.diff() / result.mid_s.diff()
    result['raw_RMS_rate_Pa_s'] = result.raw_RMS_Pa.diff() / result.mid_s.diff()
    # Confirm the RMS implementation using one directly sliced window.
    index = int(np.flatnonzero(left >= 100)[0])
    start = int(round((left[index] - time[0]) * FS))
    direct = np.sqrt(np.mean(components[-6][start:start + WINDOW] ** 2))
    assert abs(direct - result.R6_Pa.iloc[index]) < 1e-12
    return result, error


def clean_stress(time, truth):
    """Reconstruct the declared clean signal only for truth-based review."""
    sine, cosine = np.sin(2 * np.pi * time), np.cos(2 * np.pi * time)
    return .1 * (truth.ideal_Gprime_scale_Pa.to_numpy() *
                 (sine - truth.ideal_third_order_a.to_numpy() * sine ** 3 +
                  truth.ideal_fifth_order_b.to_numpy() * sine ** 5) +
                 truth.ideal_Gdoubleprime_scale_Pa.to_numpy() * cosine)


def main():
    for folder in ['03_observed_features', '04_truth_review', '05_figures', '06_validation']:
        (ROOT / folder).mkdir(exist_ok=True)
    paths = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    observed, clean, noise = {}, {}, {}
    errors, alignment = {}, []
    for case in ['control', 'clot_A']:
        obs = pd.read_csv(SOURCE / 'observable' / f'{case}.csv')
        t, stress = obs.time_s.to_numpy(), obs.stress_Pa.to_numpy()
        observed[case], errors[case] = extract_features(t, stress)
        observed[case].to_csv(ROOT / '03_observed_features' / f'{case}.csv', index=False)
        truth = pd.read_csv(SOURCE / 'truth' / f'{case}_truth.csv')
        ideal = clean_stress(t, truth)
        clean[case], _ = extract_features(t, ideal)
        noise[case], _ = extract_features(t, stress - ideal)
        clean[case].to_csv(ROOT / '04_truth_review' / f'{case}_clean.csv', index=False)
        noise[case].to_csv(ROOT / '04_truth_review' / f'{case}_noise_only.csv', index=False)
        # Remove exactly one loading cycle. Compare identical absolute windows.
        # This preserves loading phase while changing the DWT decimation origin.
        cropped, _ = extract_features(t[100:], stress[100:])
        cropped.to_csv(ROOT / '06_validation' / f'{case}_crop_one_cycle.csv', index=False)
        joined = observed[case].merge(cropped, on='left_s', suffixes=('_full', '_crop'))
        joined = joined[(joined.left_s >= 20) & (joined.left_s + 10 <= 200)]
        alignment.append(dict(case=case, feature='R6',
                              max_abs_difference_Pa=float(np.max(np.abs(joined.R6_Pa_full - joined.R6_Pa_crop))),
                              median_full_R6_Pa=float(joined.R6_Pa_full.median())))

    features = ['raw_mean_Pa', 'A8_mean_Pa', 'raw_RMS_Pa'] + [f'R{i}_Pa' for i in range(1, 9)] + ['R6_rate_Pa_s', 'P6', 'H13_Pa', 'R45_over_R6']
    rows = []
    # Truth times define descriptive review intervals, never a detector rule.
    for case in observed:
        for version, frame in [('observed', observed[case]), ('clean', clean[case]), ('noise_only', noise[case])]:
            for label, start, end in [('baseline', 20, 200), ('first_30s', 300, 330), ('late', 1500, 1700)]:
                subset = frame[(frame.left_s >= start) & (frame.right_s <= end)]
                for feature in features:
                    values = subset[feature].dropna()
                    rows.append(dict(case=case, version=version, interval=label, feature=feature,
                                     median=float(values.median()), minimum=float(values.min()), maximum=float(values.max())))
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / '04_truth_review' / 'feature_summary.csv', index=False)
    constant = clean['control'].query('left_s >= 20 and right_s <= 200')
    constant_r6_variation = float((constant.R6_Pa.max() - constant.R6_Pa.min()) / constant.R6_Pa.mean())

    # Summaries are drawn at window centers; they are not instantaneous events.
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    plotted = [('raw_RMS_Pa', 'Raw stress RMS (Pa)'), ('R6_Pa', 'D6 strength R6 (Pa)'),
               ('R6_rate_Pa_s', 'R6 change rate (Pa/s)'), ('R45_over_R6', 'Combined D4-D5 strength / R6'),
               ('P6', 'D6 share of selected detail energy'), ('H13_Pa', 'Fast-scale strength H13 (Pa)')]
    for ax, (feature, title) in zip(axes.flat, plotted):
        for case, color in [('control', '#2878B5'), ('clot_A', '#D65A31')]:
            for frame, style, alpha in [(observed[case], '-', 1), (clean[case], '--', .8)]:
                part = frame[(frame.mid_s >= 250) & (frame.mid_s <= 360)]
                label = case + (' observed' if style == '-' else ' clean')
                ax.plot(part.mid_s, part[feature], style, color=color, alpha=alpha, lw=1.4, label=label)
        ax.axvline(300, color='.4', ls=':', label='Declared onset')
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('10-s window midpoint (s); step = 1 s')
    fig.suptitle('LAOS feature screening: control vs clot_A', fontsize=18)
    fig.text(.5, .015, 'Offline full-record DWT; overlapping windows. Curves near onset may use data from after onset. No detected event is shown.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .045, 1, .955])
    fig.savefig(ROOT / '05_figures' / 'features_near_onset.png', dpi=160)
    plt.close(fig)
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    validation = dict(reconstruction_error_Pa=errors, source_unchanged=True,
                      clean_control_R6_peak_to_peak_over_mean=constant_r6_variation,
                      crop_one_cycle_baseline_alignment=alignment,
                      settings=dict(wavelet='db2', level=8, extension='symmetric', window_s=10,
                                    stride_s=1, transform='full record offline', boundary_exclusion_s=16),
                      no_resampling=True, no_amplitude_normalization=True,
                      detector_fitted=False, feature_ranking_exploratory=True,
                      note='Noise-only RMS does not add linearly to clean RMS. B and C were audited but not used in feature screening.')
    (ROOT / '06_validation' / 'feature_checks.json').write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    print(summary[(summary.case == 'clot_A') & (summary.version == 'observed')].to_string(index=False))


if __name__ == '__main__':
    main()
