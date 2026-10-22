"""Compare A3--A7 onset features without selecting on the held-out test set.

Run: python run_experiment.py
All settings are explicit below. Outputs are separated by processing stage.
This is an offline synthetic benchmark, not a clinical event detector.
"""
from pathlib import Path
import hashlib
import json
import os
import platform

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl_cache'))
import numpy as np
import pandas as pd
import pywt
from scipy.signal import lfilter
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LEVELS = [3, 4, 5, 6, 7]
NAMES = [f'A{j}' for j in LEVELS] + ['ordinary']
RAW = pd.read_csv(ROOT / 'inputs/APR18_BH_raw.csv', float_precision='round_trip')
RT = RAW.time_s.to_numpy()
N_INTERVALS = int(np.ceil((RT[-1] - RT[0]) / .0003125))
GRID = np.linspace(RT[0], RT[-1], N_INTERVALS + 1)
DT = GRID[1] - GRID[0]
WIDTH = .01
LEFT = np.arange(.01, 1 - WIDTH / 2, WIDTH)
RIGHT = LEFT + WIDTH
MID = LEFT + WIDTH / 2
# All candidates must pass the same boundary screen, including both rate windows.
SAFE = (LEFT - WIDTH >= GRID[0] + .005 + 6.25 * 2**7 * DT) & (RIGHT <= 1)
ALPHA = .05
TOLERANCE = .05
MISS_PENALTY = .2


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_manifest():
    """Fix all cases before computing or looking at any feature results."""
    rows = []
    for split, n_pos, n_null, seed in [('development', 120, 99, 1101), ('test', 240, 200, 2101)]:
        rng = np.random.default_rng(seed)
        for i in range(n_pos + n_null):
            positive = i < n_pos
            rows.append(dict(
                case_id=f'{split}_{i:03d}', split=split, positive=positive,
                shape=('fast' if i % 2 == 0 else 'slow') if positive else 'null',
                onset_s=float(rng.uniform(.35, .60)) if positive else None,
                rise_s=(.02 if i % 2 == 0 else .15) if positive else None,
                amplitude_Pa=[.002, .006, .012][(i // 2) % 3] if positive else 0.,
                noise_sd_Pa=[.0003, .001, .002][(i // 6) % 3],
                seed=int(rng.integers(1, 2**31)),
            ))
    return rows


def signal_for(case):
    """A known smooth onset plus stationary correlated Gaussian noise.

    Noise is a generic simulation assumption, not an APR18 noise estimate.
    The sparse arm samples this SAME noisy dense reference at APR18 times.
    """
    clean = np.full(len(GRID), .025)
    if case['positive']:
        u = np.clip((GRID - case['onset_s']) / case['rise_s'], 0, 1)
        clean += case['amplitude_Pa'] * .5 * (1 - np.cos(np.pi * u))
    rng = np.random.default_rng(case['seed'])
    rho = np.exp(-DT / .005)
    white = rng.normal(size=len(GRID) + 1000)
    noise = lfilter([np.sqrt(1 - rho*rho)], [1, -rho], white)[1000:]
    return clean + case['noise_sd_Pa'] * noise


def piecewise_means(t, y):
    """Exact time means of the linearly connected observations."""
    area = np.r_[0., np.cumsum(np.diff(t) * (y[:-1] + y[1:]) / 2)]
    def integral(x):
        i = np.clip(np.searchsorted(t, x, side='right') - 1, 0, len(t) - 2)
        dx = x - t[i]
        slope = (y[i+1] - y[i]) / (t[i+1] - t[i])
        return area[i] + y[i]*dx + .5*slope*dx*dx
    return (integral(RIGHT) - integral(LEFT)) / WIDTH


def extract(t, y, raw_t, raw_y, mode='symmetric'):
    """Reconstruct approximations in Pa, then calculate 10 ms window rates."""
    coeff = pywt.wavedec(y, 'db2', level=7, mode=mode)
    restored = pywt.waverec(coeff, 'db2', mode=mode)[:len(y)]
    assert np.max(np.abs(restored - y)) < 1e-11
    lo = np.searchsorted(t, LEFT)
    hi = np.searchsorted(t, RIGHT)
    assert (hi > lo).all()
    means = []
    for level in LEVELS:
        keep = [v.copy() for v in coeff]
        for j in range(1, level+1):
            keep[-j] = np.zeros_like(keep[-j])
        a = pywt.waverec(keep, 'db2', mode=mode)[:len(y)]
        # Cumulative sums provide a fast version of the original window mean.
        c = np.r_[0., np.cumsum(a)]
        means.append((c[hi] - c[lo]) / (hi - lo))
    means.append(piecewise_means(raw_t, raw_y))
    means = np.stack(means, axis=1)
    rates = np.vstack([np.full((1, len(NAMES)), np.nan), np.diff(means, axis=0) / WIDTH])
    return means, rates


def detect(rates, thresholds):
    """First pair of consecutive exceedances within the common safe interval.

    Return the first midpoint in the pair as the offline estimated onset.
    This timestamp uses later data. It is not a real-time alarm timestamp.
    """
    pair = np.minimum(rates[:-1], rates[1:])
    usable = SAFE[:-1] & SAFE[1:]
    outputs = []
    for j, threshold in enumerate(thresholds):
        indices = np.flatnonzero(usable & (pair[:, j] > threshold))
        outputs.append(float(MID[indices[0]]) if len(indices) else None)
    return outputs


def null_scores(rates):
    pair = np.minimum(rates[:-1], rates[1:])
    return pair[SAFE[:-1] & SAFE[1:]].max(axis=0)


def evaluation_rows(case, arm, rates, thresholds, variant='base'):
    rows = []
    for name, prediction in zip(NAMES, detect(rates, thresholds)):
        error = None if prediction is None or not case['positive'] else prediction - case['onset_s']
        hit = case['positive'] and error is not None and abs(error) <= TOLERANCE
        rows.append(dict(**case, arm=arm, variant=variant, feature=name,
                         detected=prediction is not None, prediction_s=prediction,
                         signed_error_s=error, hit=bool(hit),
                         loss_s=(abs(error) if hit else MISS_PENALTY) if case['positive'] else None))
    return rows


def proportion_ci(k, n):
    """Exact binomial interval; independent records, not interpolation points."""
    return [0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k))]


def metrics(rows):
    df = pd.DataFrame(rows)
    output = []
    for (arm, variant, feature), g in df.groupby(['arm', 'variant', 'feature']):
        p, n = g[g.positive], g[~g.positive]
        k = int(p.hit.sum())
        f = int(n.detected.sum())
        hits = p[p.hit]
        output.append(dict(arm=arm, variant=variant, feature=feature,
                           n_positive=len(p), hits=k, detection_fraction=float(k/len(p)),
                           n_null=len(n), false_alarm_records=f,
                           null_false_alarm_fraction=float(f/len(n)),
                           null_fpr_95ci=proportion_ci(f, len(n)),
                           miss_or_wrong_time_fraction=float(1-k/len(p)),
                           no_candidate_records=int((~p.detected).sum()),
                           mean_penalized_error_s=float(p.loss_s.mean()),
                           matched_median_abs_error_s=float(hits.signed_error_s.abs().median()) if len(hits) else None,
                           matched_median_signed_error_s=float(hits.signed_error_s.median()) if len(hits) else None))
    return output


def main():
    cases = make_manifest()
    protocol = dict(
        aim='Compare A3--A7 early-response onset features, not prove universal optimality.',
        real_source_sha256=source_hash(ROOT/'inputs/APR18_BH_raw.csv'),
        dt_s=float(DT), full_record_s=[float(GRID[0]),float(GRID[-1])], grid_points=len(GRID),
        wavelet='db2', levels=LEVELS, window_s=WIDTH,
        score_interval_midpoints_s=[float(MID[SAFE][0]),float(MID[SAFE][-1])],
        onset_range_s=[.35,.60], noise_correlation_s=.005,
        negative_control='Constant stress plus correlated Gaussian noise',
        fast_rise_s=.02, slow_rise_s=.15,
        thresholds='Per arm and per feature: 95th order statistic of 99 independent development null maxima; strict exceedance.',
        decision='First two consecutive positive-rate exceedances; first midpoint in pair reported offline.',
        match_tolerance_s=TOLERANCE, miss_or_wrong_time_penalty_s=MISS_PENALTY,
        selection='Lowest mean penalized error on development positives in APR18-sampled arm; ties prefer lower level.',
        null_false_alarm_target=ALPHA,
        robustness='First 60 test positives and first 60 test nulls, APR18 sampling only; crop 1/crop 16 grid points or reflect boundary, frozen thresholds.',
        uncertainty='Paired bootstrap over independent test cases. 99% percentile CI for each of 5 A5 comparisons within an arm, 2000 resamples.',
        limitations=['No real onset labels; initial <0.27 s excluded by common boundary screen.',
                     'Stationary synthetic noise is an assumption, not an estimate of instrument noise.',
                     'Constant background simplifies real step loading and overshoot.',
                     '10 ms windows held fixed; conclusions are conditional on that window.',
                     'Full 63 s input used, but detection examines only first second.',
                     'Boundary mask is an engineering screen, not a mathematical accuracy guarantee.'])
    save_json(ROOT/'01_protocol/protocol.json', protocol)
    save_json(ROOT/'01_protocol/cases.json', cases)
    # Write a digest before any features or outcomes are computed.
    save_json(ROOT/'01_protocol/frozen_digests.json', {n:source_hash(ROOT/'01_protocol'/n) for n in ['protocol.json','cases.json']})
    all_rates, all_means = {}, {}
    for index, case in enumerate(cases):
        y = signal_for(case)
        sparse = np.interp(RT, GRID, y)
        for arm, raw_t, raw_y, vals in [('dense', GRID, y, y), ('APR18_sampling', RT, sparse, np.interp(GRID,RT,sparse))]:
            means, rates = extract(GRID, vals, raw_t, raw_y)
            all_rates[(case['case_id'],arm)] = rates
            all_means[(case['case_id'],arm)] = means
        if index % 50 == 0:
            print(f'Features {index+1}/{len(cases)}', flush=True)
    np.savez_compressed(ROOT/'02_features/synthetic_window_features.npz',
                        means=np.stack([all_means[(c['case_id'],a)] for c in cases for a in ['dense','APR18_sampling']]),
                        rates=np.stack([all_rates[(c['case_id'],a)] for c in cases for a in ['dense','APR18_sampling']]),mid_s=MID)
    save_json(ROOT/'02_features/array_order.json', dict(case_order=[c['case_id'] for c in cases],arm_order=['dense','APR18_sampling'],feature_order=NAMES))
    thresholds = {}
    dev_rows = []
    for arm in ['dense','APR18_sampling']:
        scores = np.stack([null_scores(all_rates[(c['case_id'],arm)]) for c in cases if c['split']=='development' and not c['positive']])
        rank = int(np.ceil((len(scores)+1)*(1-ALPHA)))
        thresholds[arm] = np.sort(scores,axis=0)[rank-1]
        for c in cases:
            if c['split']=='development':
                dev_rows += evaluation_rows(c, arm, all_rates[(c['case_id'],arm)], thresholds[arm])
    save_json(ROOT/'03_evaluation/frozen_thresholds.json', {k:dict(zip(NAMES,map(float,v))) for k,v in thresholds.items()})
    dev = metrics(dev_rows)
    selected = min([m for m in dev if m['arm']=='APR18_sampling' and m['feature']!='ordinary'],key=lambda x:(x['mean_penalized_error_s'],int(x['feature'][1:])))['feature']
    save_json(ROOT/'03_evaluation/selection_before_test.json', dict(selected_DWT=selected,development_metrics=dev))
    # Evaluate the frozen test set only after the selection is saved.
    test_rows = []
    for c in cases:
        if c['split']=='test':
            for arm in thresholds:
                test_rows += evaluation_rows(c, arm, all_rates[(c['case_id'],arm)], thresholds[arm])
    pd.DataFrame(dev_rows).to_csv(ROOT/'03_evaluation/development_predictions.csv',index=False)
    pd.DataFrame(test_rows).to_csv(ROOT/'03_evaluation/test_predictions.csv',index=False)
    summary = metrics(test_rows)
    save_json(ROOT/'03_evaluation/test_metrics.json', summary)
    pd.DataFrame(summary).to_csv(ROOT/'03_evaluation/test_metrics.csv',index=False)
    # Simultaneous-within-arm conservative intervals for the five A5 comparisons.
    bootstrap = []
    for arm in thresholds:
        p=pd.DataFrame(test_rows)
        matrix=p[(p.arm==arm)&p.positive].pivot(index='case_id',columns='feature',values='loss_s')
        rng=np.random.default_rng(771)
        indices=rng.integers(0,len(matrix),(2000,len(matrix)))
        for other in [x for x in NAMES if x!='A5']:
            difference=(matrix.A5-matrix[other]).to_numpy()
            means=difference[indices].mean(axis=1)
            lo,hi=np.quantile(means,[.005,.995])
            bootstrap.append(dict(arm=arm,comparison=f'A5 minus {other}',mean_loss_difference_s=float(difference.mean()),
                                  simultaneous_within_arm_95_percent_family_ci_s=[float(lo),float(hi)]))
    save_json(ROOT/'03_evaluation/paired_uncertainty.json',bootstrap)
    robustness=[]
    subset=[c for c in cases if c['split']=='test' and c['positive']][:60]+[c for c in cases if c['split']=='test' and not c['positive']][:60]
    for c in subset:
        y=signal_for(c);sparse=np.interp(RT,GRID,y);values=np.interp(GRID,RT,sparse)
        for mode,crop,label in [('symmetric',0,'base_subset'),('symmetric',1,'crop_1'),('symmetric',16,'crop_16'),('reflect',0,'reflect')]:
            rates=all_rates[(c['case_id'],'APR18_sampling')] if label=='base_subset' else extract(GRID[crop:],values[crop:],RT,sparse,mode)[1]
            robustness += evaluation_rows(c,'APR18_sampling',rates,thresholds['APR18_sampling'],label)
    pd.DataFrame(robustness).to_csv(ROOT/'03_evaluation/robustness_predictions.csv',index=False)
    save_json(ROOT/'03_evaluation/robustness_metrics.json',metrics(robustness))
    # Real data: describe features only; there is no true onset label here.
    real_values=np.interp(GRID,RT,RAW.stress_Pa)
    means,rates=extract(GRID,real_values,RT,RAW.stress_Pa.to_numpy())
    real=pd.DataFrame({'mid_s':MID,'common_interior':SAFE})
    for j,name in enumerate(NAMES):real[f'{name}_mean']=means[:,j];real[f'{name}_rate']=rates[:,j]
    real.to_csv(ROOT/'02_features/real_APR18_features.csv',index=False)
    check_real_against_original(real)
    plots(test_rows,robustness,real,cases,all_rates,thresholds)
    save_json(ROOT/'03_evaluation/environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pywavelets=pywt.__version__))
    for name,digest in json.loads((ROOT/'01_protocol/frozen_digests.json').read_text()).items():assert source_hash(ROOT/'01_protocol'/name)==digest
    print('COMPLETE. Development-selected DWT:',selected,flush=True)


def check_real_against_original(real):
    """Check optimized means and derivatives against the earlier implementation."""
    original=ROOT/'inputs/original_APR18_A5_features.csv'
    old=pd.read_csv(original)
    assert np.allclose(old.mid_s,real.mid_s,rtol=0,atol=1e-12)
    mean_error=float(np.max(abs(old.A5-real.A5_mean)))
    rate_error=float(np.nanmax(abs(old.A5_rate-real.A5_rate)))
    assert mean_error<1e-10 and rate_error<1e-8
    # A truly constant clean signal must not produce an eligible false candidate.
    constant=np.full(len(GRID),.025)
    _,rates=extract(GRID,constant,GRID,constant)
    assert np.nanmax(abs(rates[SAFE]))<1e-9
    save_json(ROOT/'03_evaluation/implementation_checks.json',dict(original_A5_mean_max_error_Pa=mean_error,original_A5_rate_max_error_Pa_s=rate_error,constant_signal_pass=True,reconstruction_assertions_pass=True))


def plots(test_rows, robustness, real, cases, rates, thresholds):
    metrics_df=pd.DataFrame(metrics(test_rows))
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for arm,color in [('dense','#2478a8'),('APR18_sampling','#d06632')]:
        m=metrics_df[metrics_df.arm==arm].set_index('feature').loc[NAMES]
        axs[0].plot(NAMES,m.detection_fraction*100,'o-',label=arm,color=color)
        axs[1].plot(NAMES,m.mean_penalized_error_s*1000,'o-',color=color)
        axs[2].plot(NAMES,m.null_false_alarm_fraction*100,'o-',color=color)
    for ax in axs:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
    axs[0].set(title='Within ±50 ms of known onset',ylabel='Matched records (%)');axs[0].legend()
    axs[1].set(title='Timing error with failure penalty',ylabel='Mean penalized error (ms)')
    axs[2].set(title='No-change controls',ylabel='Records with a false candidate (%)');axs[2].axhline(5,ls='--',color='gray')
    fig.savefig(ROOT/'04_figures/01_frozen_test_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for name in NAMES:ax.plot(real.mid_s[SAFE],real.loc[SAFE,f'{name}_rate'],label=name,lw=1.4)
    ax.set(xlabel='Window midpoint (s)',ylabel='Stress change rate (Pa/s)',title='Real APR18: common valid interval only, no onset truth')
    ax.grid(alpha=.2);ax.legend(ncol=6);fig.savefig(ROOT/'04_figures/02_real_APR18_levels.png',dpi=160);plt.close(fig)
    # The example is fixed by manifest order, not by favourable test performance.
    c=next(c for c in cases if c['split']=='test' and c['positive'])
    y=signal_for(c);sparse=np.interp(RT,GRID,y)
    fig,axs=plt.subplots(2,1,figsize=(10,6),layout='constrained',sharex=True)
    mask=(GRID>=.27)&(GRID<=1)
    axs[0].plot(GRID[mask],y[mask],alpha=.65,label='Dense observed reference')
    axs[0].plot(RT,sparse,'o-',label='APR18 timestamps and linear links')
    for name in ['A3','A5','A7','ordinary']:
        j=NAMES.index(name);axs[1].plot(MID[SAFE],rates[(c['case_id'],'APR18_sampling')][SAFE,j],label=name)
    for ax in axs:ax.axvline(c['onset_s'],color='black',ls='--',label='Known onset');ax.grid(alpha=.2);ax.legend();ax.set_xlim(.27,1)
    axs[0].set(title=f"Fixed example: {c['case_id']}, rise {c['rise_s']} s",ylabel='Stress (Pa)')
    axs[1].set(xlabel='Time / window midpoint (s)',ylabel='Rate (Pa/s)')
    fig.savefig(ROOT/'04_figures/03_fixed_example.png',dpi=160);plt.close(fig)
    m=pd.DataFrame(metrics(robustness));fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for variant in ['base_subset','crop_1','crop_16','reflect']:
        v=m[m.variant==variant].set_index('feature').loc[NAMES]
        ax.plot(NAMES,v.detection_fraction*100,'o-',label=variant)
    ax.set(title='Frozen-threshold sensitivity on the same 60 positive cases',ylabel='Matched records (%)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(ROOT/'04_figures/04_alignment_sensitivity.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
