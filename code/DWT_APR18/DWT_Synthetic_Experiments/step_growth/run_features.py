"""Explore DWT features for added stress growth in synthetic step pair 1.

This is an offline, exploratory feature study, not a trained detector.
Run: python run_features.py
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.plot_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.plot_cache'))

import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path(os.environ.get('DWT_STEP_DATA', str(Path(__file__).resolve().parents[1] / 'data/step_data')))
CASES = ('control_1', 'test_clot_1')
FEATURES = [f'R{j}_Pa' for j in range(1, 9)] + ['A8_slope_Pa_s', 'raw_slope_Pa_s']
OUT = ROOT / 'results'
for folder in ('observed', 'truth_review', 'sensitivity', 'figures', 'validation'):
    (OUT / folder).mkdir(parents=True, exist_ok=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decompose(values):
    """Reconstruct each component on the original sample grid."""
    coefficients = pywt.wavedec(values, 'db2', mode='symmetric', level=8)
    components = []
    for index in range(len(coefficients)):
        selected = [np.zeros_like(c) for c in coefficients]
        selected[index] = coefficients[index]
        components.append(pywt.waverec(selected, 'db2', mode='symmetric')[:len(values)])
    reconstructed = np.sum(components, axis=0)
    error = float(np.max(np.abs(reconstructed - values)))
    assert error < 1e-12 * max(1., float(np.max(np.abs(values))))
    return components[0], list(reversed(components[1:])), error


def extract(t, values, seconds=1):
    """Use complete, absolute-time windows. Do not interpolate or normalize."""
    assert len(t) == len(values) and np.isfinite(values).all()
    assert np.allclose(np.diff(t), .01, atol=1e-8, rtol=0)
    approximation, details, error = decompose(values)
    rows = []
    start = int(np.ceil((t[0] - 1e-9) / seconds)) * seconds
    for left in range(start, int(np.floor(t[-1])), seconds):
        right = left + seconds
        if right > t[-1] + 1e-9:
            continue
        mask = (t >= left) & (t < right)
        assert int(mask.sum()) == int(seconds * 100)
        row = {'left_s': left, 'right_s': right,
               'mid_s': (left + right) / 2,
               'interior': bool(left >= t[0] + 16 and right <= t[-1] - 16),
               'raw_mean_Pa': float(np.mean(values[mask])),
               'A8_mean_Pa': float(np.mean(approximation[mask]))}
        row.update({f'R{j}_Pa': float(np.sqrt(np.mean(detail[mask]**2)))
                    for j, detail in enumerate(details, 1)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for name in ('raw', 'A8'):
        frame[f'{name}_slope_Pa_s'] = frame[f'{name}_mean_Pa'].diff() / frame.mid_s.diff()
    frame['slope_interior'] = frame.interior & frame.interior.shift(1, fill_value=False)
    return frame, error


def self_checks():
    """Check indexing and numerical behavior using independent toy signals."""
    t = np.arange(12001) / 100
    constant, e = extract(t, np.full(len(t), .025))
    assert len(constant) == 120 and constant.left_s.iloc[0] == 0
    assert constant.right_s.iloc[-1] == 120
    assert np.isnan(constant.A8_slope_Pa_s.iloc[0])
    assert np.max(np.abs(constant.loc[constant.interior, FEATURES[:8]].to_numpy())) < 1e-14
    slope = .003
    linear, _ = extract(t, .02 + slope * t)
    assert np.max(np.abs(linear.loc[linear.slope_interior, 'A8_slope_Pa_s'] - slope)) < 1e-12
    assert np.max(np.abs(linear.loc[linear.slope_interior, 'raw_slope_Pa_s'] - slope)) < 1e-12
    shifted, _ = extract(t[32:], (.02 + slope * t)[32:])
    common = linear.merge(shifted, on='left_s', suffixes=('_a', '_b'))
    assert np.max(np.abs(common.raw_mean_Pa_a - common.raw_mean_Pa_b)) < 1e-15
    sample = .02 + .01 * np.sin(2 * np.pi * .7 * t)
    f, _ = extract(t, sample)
    doubled, _ = extract(t, 2 * sample)
    assert np.allclose(doubled[FEATURES], 2 * f[FEATURES], atol=1e-14, equal_nan=True)
    manual = np.mean(sample[2000:2100])
    assert abs(manual - f.loc[f.left_s == 20, 'raw_mean_Pa'].iloc[0]) < 1e-15
    return {'complete_windows_and_endpoint': True, 'initial_slope_missing': True,
            'constant_details_zero': True, 'linear_slope_preserved_interior': True,
            'crop_preserves_absolute_raw_windows': True, 'amplitude_scaling': True,
            'manual_window_mean': True, 'constant_reconstruction_error_Pa': e}


def segment(frame, left, right, feature):
    valid = frame.slope_interior if 'slope' in feature else frame.interior
    # Both contributing windows must belong to the interval for slope summaries.
    prior_left = frame.left_s.shift(1) if 'slope' in feature else frame.left_s
    return frame.loc[valid & (prior_left >= left) & (frame.right_s <= right), feature]


def visible(frame, feature, left=16, right=104):
    """Mask invalid samples before plotting so hidden edge peaks do not set axes."""
    valid = frame.slope_interior if 'slope' in feature else frame.interior
    return frame.loc[valid & (frame.mid_s >= left) & (frame.mid_s <= right)]


source_paths = [SOURCE / 'manifest.json']
source_paths += [SOURCE / f'test/{case}.csv' for case in CASES]
source_paths += [SOURCE / f'ground_truth/{case}_truth.csv' for case in CASES]
hashes = {str(p.relative_to(SOURCE)): sha256(p) for p in source_paths}
settings = {
    'purpose': 'Candidate features for added growth after fixed loading; pair 1 only.',
    'cases': CASES, 'wavelet': 'db2', 'level': 8, 'extension': 'symmetric',
    'sampling_Hz': 100, 'window_s': 1, 'boundary_exclusion_s': 16,
    'variants': ['2s_windows', 'crop_1_sample', 'crop_32_samples'],
    'reference_interval_s': [20, 30], 'raw_data_normalized': False,
    'scope': 'Exploratory: metadata and truth were previously inspected. No detection threshold or accuracy.',
}
(OUT / 'validation/settings.json').write_text(json.dumps(settings, indent=2))
checks = self_checks()
observed = {}
validation = []
raw = {}
for case in CASES:
    frame = pd.read_csv(SOURCE / f'test/{case}.csv')
    raw[case] = frame
    assert list(frame) == ['time_s', 'shear_rate_1_s', 'stress_Pa']
    t = frame.time_s.to_numpy()
    observed[case], err = extract(t, frame.stress_Pa.to_numpy())
    observed[case].to_csv(OUT / f'observed/{case}.csv', index=False)
    rate_features, rate_err = extract(t, frame.shear_rate_1_s.to_numpy())
    rate_features.to_csv(OUT / f'observed/{case}_input.csv', index=False)
    validation.append({'case': case, 'reconstruction_error_Pa': err,
                       'input_reconstruction_error_1_s': rate_err})
assert np.array_equal(raw[CASES[0]].shear_rate_1_s, raw[CASES[1]].shear_rate_1_s)
observed_hashes = {p.name: sha256(p) for p in (OUT / 'observed').glob('*.csv')}
(OUT / 'validation/observed_feature_hashes.json').write_text(json.dumps(observed_hashes, indent=2))

# Only after saving observed features do we read truth for interpretation.
clean = {}
truths = {}
for case in CASES:
    truth = pd.read_csv(SOURCE / f'ground_truth/{case}_truth.csv')
    assert np.array_equal(truth.time_s, raw[case].time_s)
    truths[case] = truth
    values = (truth.stress_nonclot_truth_Pa + truth.stress_excess_truth_Pa).to_numpy()
    clean[case], _ = extract(truth.time_s.to_numpy(), values)
    clean[case].to_csv(OUT / f'truth_review/{case}_clean.csv', index=False)

summary = []
for case in CASES:
    for kind, frame in (('observed', observed[case]), ('clean', clean[case])):
        for label, left, right in [('reference', 20, 30), ('early_growth', 40, 50),
                                   ('growth', 40, 70), ('late', 90, 100)]:
            for feature in FEATURES + ['raw_mean_Pa', 'A8_mean_Pa']:
                s = segment(frame, left, right, feature)
                summary.append({'case': case, 'signal': kind, 'interval': label,
                                'feature': feature, 'n_windows': len(s),
                                'mean': s.mean(), 'min': s.min(), 'max': s.max(),
                                'std_ddof1': s.std()})
pd.DataFrame(summary).to_csv(OUT / 'truth_review/interval_summary.csv', index=False)

sensitivity = []
variants = {}
for case in CASES:
    base = observed[case]
    t = raw[case].time_s.to_numpy()
    x = raw[case].stress_Pa.to_numpy()
    for label, seconds, crop in [('2s_windows', 2, 0), ('crop_1_sample', 1, 1), ('crop_32_samples', 1, 32)]:
        variant, _ = extract(t[crop:], x[crop:], seconds)
        variants[case, label] = variant
        variant.to_csv(OUT / f'sensitivity/{case}_{label}.csv', index=False)
        if seconds != 1:
            continue
        paired = base.merge(variant, on='left_s', suffixes=('_base', '_variant'))
        for feature in FEATURES:
            flag = 'slope_interior' if 'slope' in feature else 'interior'
            valid = paired[flag + '_base'] & paired[flag + '_variant']
            delta = paired.loc[valid, feature + '_variant'] - paired.loc[valid, feature + '_base']
            sensitivity.append({'case': case, 'variant': label, 'feature': feature,
                                'max_abs_difference': float(np.max(np.abs(delta))),
                                'mean_abs_difference': float(np.mean(np.abs(delta)))})
pd.DataFrame(sensitivity).to_csv(OUT / 'sensitivity/shift_summary.csv', index=False)

colors = {'control_1': '#2878ad', 'test_clot_1': '#c45b28'}
fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
for j, ax in enumerate(axes.flat, 1):
    feature = f'R{j}_Pa'
    for case in CASES:
        frame = visible(observed[case], feature)
        ax.plot(frame.mid_s, frame[feature], color=colors[case], lw=1, label=case + ' observed')
    frame = visible(clean['test_clot_1'], feature)
    ax.plot(frame.mid_s, frame[feature], 'k--', lw=1, label='Growth clean')
    ax.axvline(40, color='gray', ls=':', label='Declared onset')
    ax.set(title=f'D{j} intensity', ylabel='RMS (Pa)', xlim=(16, 104))
    ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=7)
for ax in axes[-1]: ax.set_xlabel('Window midpoint (s)')
fig.suptitle('Detail intensity: observed controls, added growth, and clean growth response', fontsize=13)
fig.savefig(OUT / 'figures/detail_strengths.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
for ax, feature, title, unit in zip(axes, ['raw_mean_Pa', 'A8_slope_Pa_s', 'raw_slope_Pa_s'],
        ['Raw window mean', 'DWT approximation change rate', 'Raw window mean change rate'], ['Pa', 'Pa/s', 'Pa/s']):
    for case in CASES:
        frame = visible(observed[case], feature)
        ax.plot(frame.mid_s, frame[feature], color=colors[case], label=case + ' observed')
    frame = visible(clean['test_clot_1'], feature)
    ax.plot(frame.mid_s, frame[feature], 'k--', lw=1, label='Growth clean')
    ax.axvline(40, color='gray', ls=':', label='Declared onset')
    ax.set(title=title, ylabel=unit, xlim=(16, 104))
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel('Window midpoint (s); slopes use this and the previous window')
fig.suptitle('Slow growth features and a simple time-domain reference', fontsize=13)
fig.savefig(OUT / 'figures/trend_features.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
for col, feature in enumerate(('R8_Pa', 'A8_slope_Pa_s')):
    base = observed['test_clot_1']
    for row, labels in enumerate((['crop_1_sample', 'crop_32_samples'], ['2s_windows'])):
        ax = axes[row, col]
        frame = visible(base, feature, 25, 90)
        ax.plot(frame.mid_s, frame[feature], color='black', lw=1, label='Original, 1 s')
        for label in labels:
            frame = visible(variants['test_clot_1', label], feature, 25, 90)
            ax.plot(frame.mid_s, frame[feature], lw=1, label=label)
        ax.axvline(40, color='gray', ls=':')
        ax.set(title=feature, xlim=(25, 90), ylabel='Pa/s' if 'slope' in feature else 'Pa')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
for ax in axes[-1]: ax.set_xlabel('Absolute window midpoint (s)')
fig.suptitle('Sensitivity checks: same data, different DWT start or summary window', fontsize=13)
fig.savefig(OUT / 'figures/sensitivity.png', dpi=150)
plt.close(fig)

assert all(sha256(SOURCE / name) == value for name, value in hashes.items())
assert all(sha256(OUT / 'observed' / name) == value for name, value in observed_hashes.items())
report = {'self_checks': checks, 'reconstruction': validation, 'source_hashes': hashes,
          'source_unchanged': True, 'saved_observed_features_unchanged': True,
          'python': sys.executable, 'numpy': np.__version__, 'pywavelets': pywt.__version__,
          'note': 'No detector, threshold, timing score or accuracy was calculated.'}
(OUT / 'validation/checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
