"""Illustrate the existing pair-1 A8 results without changing the experiment."""
from pathlib import Path
import os
import hashlib

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.plot_cache'))
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = Path(os.environ.get('DWT_STEP_DATA', str(Path(__file__).resolve().parents[1] / 'data/step_data')))
paths = [source / f'test/{case}.csv' for case in ('control_1', 'test_clot_1')]
before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
signals = {}
for path in paths:
    frame = pd.read_csv(path)
    x = frame.stress_Pa.to_numpy()
    coefficients = pywt.wavedec(x, 'db2', mode='symmetric', level=8)
    approximation_only = [coefficients[0]] + [np.zeros_like(c) for c in coefficients[1:]]
    a8 = pywt.waverec(approximation_only, 'db2', mode='symmetric')[:len(x)]
    saved = pd.read_csv(HERE / f'results/observed/{path.stem}.csv')
    # Check the illustration against every previously saved window mean.
    for row in saved.itertuples():
        mask = (frame.time_s >= row.left_s) & (frame.time_s < row.right_s)
        assert abs(np.mean(a8[mask]) - row.A8_mean_Pa) < 1e-15
    signals[path.stem] = (frame, a8)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
growth, ga = signals['test_clot_1']
control, ca = signals['control_1']
mask = (growth.time_s >= 16) & (growth.time_s <= 104)
axes[0].plot(growth.time_s[mask], growth.stress_Pa[mask], color='#c45b28', alpha=.35,
             lw=.6, label='Growth: original stress')
axes[0].plot(growth.time_s[mask], ga[mask], color='#174d78', lw=1.7, label='Growth: reconstructed A8')
axes[0].plot(control.time_s[mask], ca[mask], color='gray', ls='--', lw=1.4,
             label='Control: reconstructed A8')
axes[0].axvline(40, color='black', ls=':', label='Declared growth onset')
axes[0].set(title='A8 retains the slow rise and the higher plateau', xlabel='Time (s)', ylabel='Stress / A8 (Pa)')
axes[0].legend(fontsize=9)
mask = (growth.time_s >= 20) & (growth.time_s <= 25)
axes[1].plot(growth.time_s[mask], growth.stress_Pa[mask], color='#c45b28', alpha=.65,
             lw=.7, label='Original stress')
axes[1].plot(growth.time_s[mask], ga[mask], color='#174d78', lw=1.7, label='Reconstructed A8')
truth = pd.read_csv(source / 'ground_truth/test_clot_1_truth.csv')
axes[1].plot(truth.time_s[mask], truth.stress_nonclot_truth_Pa[mask], 'k--', lw=1,
             label='Supplied clean baseline truth')
axes[1].set(title='Before growth: A8 is smoother, but it still differs from clean truth',
             xlabel='Time (s)', ylabel='Stress / A8 (Pa)')
axes[1].legend(fontsize=9)
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(HERE / 'results/figures/A8_explanation.png', dpi=150)
plt.close(fig)
assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
print('A8 illustration verified against all saved means; source files unchanged.')
