"""Audit the supplied synthetic LAOS records before choosing DWT features.

Keep all source files unchanged. Truth checks assess the benchmark design;
truth is never supplied to a detector. No detection rule is fitted here.
"""
from pathlib import Path
import os
import hashlib
import json

import numpy as np
import pandas as pd
import pywt

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('DWT_LAOS_DATA', str(Path(__file__).resolve().parents[1] / 'data/laos_synthetic')))
OUTPUT = ROOT / '01_data_audit'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    files = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    hashes = {str(p.relative_to(SOURCE)): sha256(p) for p in files}
    meta = json.loads((SOURCE / 'metadata.json').read_text())
    profiles, windows, truth_checks = [], [], []
    reference_input = None
    for case in ['control', 'clot_A', 'clot_B', 'clot_C']:
        obs = pd.read_csv(SOURCE / 'observable' / f'{case}.csv')
        truth = pd.read_csv(SOURCE / 'truth' / f'{case}_truth.csv')
        t = obs.time_s.to_numpy()
        stress = obs.stress_Pa.to_numpy()
        assert list(obs) == ['time_s', 'strain', 'shear_rate_1_s', 'stress_Pa']
        assert np.isfinite(obs.to_numpy()).all()
        assert np.isfinite(truth.to_numpy()).all()
        assert len(t) == 180000 and t[0] == 0 and t[-1] == 1799.99
        assert np.all(np.diff(t) > 0)
        assert np.allclose(np.diff(t), .01, rtol=0, atol=1e-10)
        assert np.array_equal(t, truth.time_s.to_numpy())
        inp = obs[['time_s', 'strain', 'shear_rate_1_s']].to_numpy()
        if reference_input is None:
            reference_input = inp
        assert np.array_equal(inp, reference_input)
        sine, cosine = np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)
        strain_error = float(np.max(np.abs(obs.strain.to_numpy() - .1 * sine)))
        rate_error = float(np.max(np.abs(obs.shear_rate_1_s.to_numpy() - .2 * np.pi * cosine)))
        assert strain_error < 1e-10 and rate_error < 1e-10

        # This verifies numerical compatibility, not detection performance.
        coefs = pywt.wavedec(stress, 'db2', mode='symmetric', level=8)
        restored = pywt.waverec(coefs, 'db2', mode='symmetric')[:len(t)]
        dwt_error = float(np.max(np.abs(restored - stress)))
        assert dwt_error < 1e-10
        profiles.append(dict(case=case, rows=len(obs), missing_values=int(obs.isna().sum().sum()),
                             duplicate_times=int(obs.time_s.duplicated().sum()),
                             recorded_time_span_s=float(t[-1] - t[0]), nominal_duration_s=1800, sampling_Hz=100,
                             strain_formula_max_error=strain_error,
                             rate_formula_max_error_1_s=rate_error,
                             dwt_reconstruction_max_error_Pa=dwt_error))

        # Full 10-second windows contain ten complete loading cycles.
        # The record is half-open: [0, 1800), with no repeated final endpoint.
        blocks = stress.reshape(-1, 1000)
        for i, block in enumerate(blocks):
            windows.append(dict(case=case, left_s=i * 10, right_s=(i + 1) * 10,
                                stress_mean_Pa=float(block.mean()),
                                stress_RMS_Pa=float(np.sqrt(np.mean(block ** 2)))))

        # Review the declared truth separately from observable-only statistics.
        info = meta['scenario']['onset_and_plateau_by_case'][case]
        onset, amplitude = info['onset_s'], info['plateau_reference_Pa']
        q = np.zeros_like(t) if onset is None else 1 - np.exp(-(np.maximum(t - onset, 0) / 300) ** 2)
        gp = meta['APR18_SAOS_baseline_at_1Hz']['Gprime_Pa'] + amplitude * q
        gpp = meta['APR18_SAOS_baseline_at_1Hz']['Gdoubleprime_Pa'] + .15 * amplitude * q
        a, b = .08 + .57 * q, .008 + .192 * q
        expected = np.column_stack([q, gp, gpp, a, b])
        truth_error = float(np.max(np.abs(expected - truth.iloc[:, 1:].to_numpy())))
        assert truth_error < 1e-9
        clean = .1 * (gp * (sine - a * sine ** 3 + b * sine ** 5) + gpp * cosine)
        residual = stress - clean
        declared_sd = .00008 + .002 * .1 * np.hypot(gp, gpp)
        normalized_residual = residual / declared_sd
        first_changed = None if not np.any(q > 0) else float(t[np.flatnonzero(q > 0)[0]])
        truth_checks.append(dict(case=case, declared_onset_s=onset,
                                 first_strictly_positive_state_sample_s=first_changed,
                                 truth_formula_max_error=truth_error,
                                 normalized_noise_SD=float(normalized_residual.std()),
                                 normalized_noise_lag1_correlation=float(np.corrcoef(normalized_residual[:-1], normalized_residual[1:])[0, 1]),
                                 initial_declared_noise_SD_Pa=float(declared_sd[0]),
                                 final_declared_noise_SD_Pa=float(declared_sd[-1])))

    window_df = pd.DataFrame(windows)
    summaries = []
    for case, rows in window_df.groupby('case', sort=False):
        for label, start, end in [('early', 20, 200), ('late', 1500, 1700)]:
            part = rows[(rows.left_s >= start) & (rows.right_s <= end)]
            summaries.append(dict(case=case, interval=label, start_s=start, end_s=end,
                                  median_abs_window_mean_Pa=float(part.stress_mean_Pa.abs().median()),
                                  median_window_RMS_Pa=float(part.stress_RMS_Pa.median())))
    pd.DataFrame(profiles).to_csv(OUTPUT / 'quality_profile.csv', index=False)
    window_df.to_csv(OUTPUT / 'observable_window_statistics.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / 'observable_summary.csv', index=False)
    (OUTPUT / 'truth_review.json').write_text(json.dumps(truth_checks, indent=2))
    assert hashes == {str(p.relative_to(SOURCE)): sha256(p) for p in files}
    (OUTPUT / 'validation.json').write_text(json.dumps(dict(
        all_checks_passed=True, source_hashes=hashes, source_files_unchanged=True,
        resampling=False, amplitude_normalization=False, detector_fitted=False,
        wavelet='db2', level=8, extension='symmetric',
        numpy_version=np.__version__, pandas_version=pd.__version__, pywavelets_version=pywt.__version__,
        scope='Data audit and numerical reconstruction only; no feature ranking or detection accuracy.'
    ), indent=2))
    print(pd.DataFrame(profiles).to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(truth_checks, indent=2))


if __name__ == '__main__':
    main()
