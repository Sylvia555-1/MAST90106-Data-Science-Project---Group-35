"""Compare decomposition levels without changing the existing level-8 study.

This is an offline feature comparison, not a change-point detector.
Run with python compare_levels.py
"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.plot_cache'))
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path(os.environ.get('DWT_STEP_DATA', str(Path(__file__).resolve().parents[1] / 'data/step_data')))
OUT = ROOT / 'results/level_comparison'
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = (7, 8, 9)
CASES = ('control_1', 'test_clot_1')
WAVELET = 'db2'
COMMON_LEFT, COMMON_RIGHT = 32, 88


def components(x, level):
    """Reconstruct A_J and each D_j on the original time grid."""
    coefficients = pywt.wavedec(x, WAVELET, mode='symmetric', level=level)
    parts = []
    for index in range(len(coefficients)):
        selected = [np.zeros_like(c) for c in coefficients]
        selected[index] = coefficients[index]
        parts.append(pywt.waverec(selected, WAVELET, mode='symmetric')[:len(x)])
    error = float(np.max(np.abs(np.sum(parts, axis=0) - x)))
    assert error < 1e-12
    return parts[0], dict(zip(range(level, 0, -1), parts[1:])), error


def features(t, x, level):
    """Summarize complete one-second windows; preserve time and amplitude."""
    assert np.allclose(np.diff(t), .01, atol=1e-8, rtol=0)
    approximation, details, error = components(x, level)
    rows = []
    for left in range(int(np.ceil(t[0])), int(np.floor(t[-1]))):
        mask = (t >= left) & (t < left + 1)
        assert mask.sum() == 100
        rows.append({'left_s': left, 'right_s': left + 1, 'mid_s': left + .5,
                     'raw_mean_Pa': np.mean(x[mask]),
                     'A_mean_Pa': np.mean(approximation[mask]),
                     'R_Pa': np.sqrt(np.mean(details[level][mask] ** 2))})
    frame = pd.DataFrame(rows)
    frame['A_slope_Pa_s'] = frame.A_mean_Pa.diff() / frame.mid_s.diff()
    frame['raw_slope_Pa_s'] = frame.raw_mean_Pa.diff() / frame.mid_s.diff()
    # Require both slope windows to lie in a common physical interval.
    frame['valid'] = ((frame.left_s.shift(1) >= COMMON_LEFT)
                      & (frame.right_s <= COMMON_RIGHT))
    return frame, error


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    paths = [SOURCE / f'{folder}/{case}{suffix}.csv'
             for case in CASES for folder, suffix in
             [('test', ''), ('ground_truth', '_truth')]]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    observed, clean, cropped = {}, {}, {}
    errors = []
    identity_errors = []
    for case in CASES:
        data = pd.read_csv(SOURCE / f'test/{case}.csv')
        truth = pd.read_csv(SOURCE / f'ground_truth/{case}_truth.csv')
        t = data.time_s.to_numpy()
        x = data.stress_Pa.to_numpy()
        assert np.array_equal(t, truth.time_s.to_numpy())
        clean_x = (truth.stress_nonclot_truth_Pa + truth.stress_excess_truth_Pa).to_numpy()
        a8, d8, _ = components(x, 8)
        a9, d9, _ = components(x, 9)
        identity_errors.append({'case': case,
            'A8_equals_A9_plus_D9_max_error_Pa': float(np.max(np.abs(a8 - a9 - d9[9]))),
            'shared_D1_to_D8_max_error_Pa': float(max(np.max(np.abs(d8[j] - d9[j]))
                                                    for j in range(1, 9)))})
        for level in LEVELS:
            observed[case, level], error = features(t, x, level)
            clean[case, level], clean_error = features(t, clean_x, level)
            cropped[case, level], crop_error = features(t[32:], x[32:], level)
            errors.extend((error, clean_error, crop_error))
            for kind, frames in [('observed', observed), ('clean', clean), ('crop32', cropped)]:
                frames[case, level].to_csv(OUT / f'{case}_J{level}_{kind}.csv', index=False)

    summary = []
    for level in LEVELS:
        obs = observed['test_clot_1', level]
        clean_pos = clean['test_clot_1', level]
        control = observed['control_1', level]
        # The clean baseline response is subtracted for onset-spreading checks.
        # This uses known synthetic truth and is not available in a real detector.
        clean_extra_slope = (clean_pos.A_slope_Pa_s
                             - clean['control_1', level].A_slope_Pa_s)
        valid = obs.valid
        pre = valid & (obs.right_s <= 40)
        growth = valid & (obs.left_s.shift(1) >= 40) & (obs.right_s <= 70)
        paired = obs.merge(cropped['test_clot_1', level], on='left_s', suffixes=('_a', '_b'))
        crop_valid = paired.valid_a & paired.valid_b
        peak_index = clean_extra_slope[growth].idxmax()
        summary.append({
            'level': level, 'common_interval_s': '32-88',
            'n_valid_slopes': int(valid.sum()),
            'positive_growth_mean_A_slope_Pa_s': float(obs.loc[growth, 'A_slope_Pa_s'].mean()),
            'positive_slope_noise_RMSE_Pa_s': rmse(obs.loc[valid, 'A_slope_Pa_s']
                                                  - clean_pos.loc[valid, 'A_slope_Pa_s']),
            'clean_slope_shape_RMSE_vs_raw_Pa_s': rmse(clean_pos.loc[valid, 'A_slope_Pa_s']
                                                     - clean_pos.loc[valid, 'raw_slope_Pa_s']),
            'control_slope_RMS_Pa_s': rmse(control.loc[valid, 'A_slope_Pa_s']),
            'pre_onset_clean_extra_slope_max_abs_Pa_s': float(clean_extra_slope[pre].abs().max()),
            'growth_clean_extra_slope_max_Pa_s': float(clean_extra_slope[growth].max()),
            'clean_extra_slope_peak_window_left_s': int(obs.loc[peak_index, 'left_s']),
            'crop32_slope_max_abs_difference_Pa_s': float((paired.loc[crop_valid, 'A_slope_Pa_s_b']
                                                        - paired.loc[crop_valid, 'A_slope_Pa_s_a']).abs().max()),
            'growth_R_mean_Pa': float(obs.loc[growth, 'R_Pa'].mean()),
            'control_growth_R_mean_Pa': float(control.loc[growth, 'R_Pa'].mean()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'summary.csv', index=False)

    # Boundary screening is deliberately conservative, not an exact error bound.
    filter_len = pywt.Wavelet(WAVELET).dec_len
    screening = []
    for level in range(7, 12):
        margin = 2 * (filter_len - 1) * (2 ** level - 1) / 100
        screening.append({'level': level, 'coefficient_spacing_s': 2 ** level / 100,
                          'conservative_margin_each_end_s': margin,
                          'remaining_interior_duration_s': max(0, 120 - 2 * np.ceil(margin)),
                          'included_in_feature_comparison': level in LEVELS})
    pd.DataFrame(screening).to_csv(OUT / 'boundary_screening.csv', index=False)

    # Check existing level-8 results independently, without rewriting them.
    old = pd.read_csv(ROOT / 'results/observed/test_clot_1.csv')
    current = observed['test_clot_1', 8]
    saved_match = float(np.nanmax(np.abs(old.A8_slope_Pa_s - current.A_slope_Pa_s)))
    assert saved_match < 1e-14
    assert all(row['A8_equals_A9_plus_D9_max_error_Pa'] < 1e-12 for row in identity_errors)
    assert all(row['shared_D1_to_D8_max_error_Pa'] < 1e-12 for row in identity_errors)
    constant_t = np.arange(12001) / 100
    for level in LEVELS:
        flat, _ = features(constant_t, np.full(12001, .025), level)
        assert np.max(np.abs(flat.loc[flat.valid, 'A_slope_Pa_s'])) < 1e-14
        assert np.max(np.abs(flat.loc[flat.valid, 'R_Pa'])) < 1e-14
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (OUT / 'validation.json').write_text(json.dumps({
        'wavelet': WAVELET, 'extension': 'symmetric', 'window_s': 1,
        'levels': LEVELS, 'common_interval_s': [COMMON_LEFT, COMMON_RIGHT],
        'maximum_reconstruction_error_Pa': max(errors), 'component_identity_checks': identity_errors,
        'matches_saved_level8_max_error_Pa_s': saved_match,
        'constant_signal_checks_passed': True, 'source_hashes_unchanged': True,
        'source_hashes': hashes, 'max_level': pywt.dwt_max_level(12001, filter_len),
        'limitations': ['Offline; uses future samples.', 'Known truth inspected; exploratory.',
                        'No threshold, detected onset, accuracy, or selected optimal level.',
                        'All timing labels refer to two adjacent one-second windows.']
    }, indent=2))

    colors = {7: '#278553', 8: '#2866b0', 9: '#c36a23'}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for level in LEVELS:
        f = observed['test_clot_1', level]
        c = clean['test_clot_1', level]
        valid = f.valid
        axes[0].plot(f.loc[valid, 'mid_s'], f.loc[valid, 'A_slope_Pa_s'] * 1000,
                     color=colors[level], label=f'A{level}: observed growth')
        extra = c.A_slope_Pa_s - clean['control_1', level].A_slope_Pa_s
        zoom = valid & (f.mid_s <= 50)
        axes[1].plot(f.loc[zoom, 'mid_s'], extra[zoom] * 1000,
                     color=colors[level], label=f'A{level}: clean added response')
        control = observed['control_1', level]
        axes[2].plot(f.loc[valid, 'mid_s'], control.loc[valid, 'A_slope_Pa_s'] * 1000,
                     color=colors[level], label=f'A{level}: observed control')
    base = clean['test_clot_1', 7]
    extra_raw = base.raw_slope_Pa_s - clean['control_1', 7].raw_slope_Pa_s
    zoom = base.valid & (base.mid_s <= 50)
    axes[1].plot(base.loc[zoom, 'mid_s'], extra_raw[zoom] * 1000,
                 'k--', linewidth=1, label='Clean raw window-mean rate')
    for ax in axes:
        ax.axvline(40, ls=':', color='black', label='Declared onset: 40 s')
        ax.axhline(0, color='gray', linewidth=.6)
        ax.grid(alpha=.2)
        ax.set_ylabel('Change rate (mPa/s)')
        ax.legend(fontsize=8)
    axes[0].set(title='Observed growth: smoothness does not establish onset accuracy', xlim=(32, 88))
    axes[1].set(title='Known clean added response: offline filtering can spread activity before onset', xlim=(32, 50))
    axes[2].set(title='No added growth: background fluctuations remain', xlim=(32, 88))
    axes[-1].set_xlabel('Later window midpoint (s); each rate uses two adjacent 1-s windows')
    fig.savefig(OUT / 'level_comparison.png', dpi=150)
    plt.close(fig)
    print(summary.to_string(index=False))
    print('Validation passed; outputs:', OUT)


if __name__ == '__main__':
    main()
