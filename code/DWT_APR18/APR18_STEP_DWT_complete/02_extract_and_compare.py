"""Compare APR18 0.1-to-1/s with the previous synthetic step feature study.

No model fitting, event threshold, denoising, amplitude normalization, or
real-data accuracy claim is included. Real DWT inputs are explicitly interpolated.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import pywt
from dwt_features import reconstruct, windows, linear_window_means, compare

ROOT = Path(__file__).resolve().parent
SYNTHETIC = ROOT / 'inputs' / 'synthetic_step'
for folder in ['02_features','03_checks','04_figures','05_comparison']:
    (ROOT / folder).mkdir(exist_ok=True)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def real_variant(raw, multiplier=1, mode='symmetric', crop=0):
    """Nested grids have exactly equal endpoints and matched physical scales."""
    t = raw.time_s.to_numpy()
    intervals = int(np.ceil((t[-1]-t[0])/.0003125)) * multiplier
    grid = np.linspace(t[0], t[-1], intervals+1)[crop:]
    shift = int(np.log2(multiplier))
    levels = list(range(6+shift,14+shift))
    early = 5+shift
    output, fidelity = {}, []
    for channel in ['stress_Pa','shear_rate_1_s']:
        original = raw[channel].to_numpy()
        signal = np.interp(grid,t,original)
        components,error = reconstruct(signal, levels, [early,13+shift], mode)
        assert np.allclose(components[f'A{early}'],
                           components[f'A{13+shift}'] + sum(components[f'D{j}'] for j in levels),
                           rtol=1e-12,atol=1e-12)
        slow = windows(grid,signal,components,1,1,63,t)
        fast_components = {f'A{early}':components[f'A{early}'],
                           f'D{6+shift}':components[f'D{6+shift}'],
                           f'D{7+shift}':components[f'D{7+shift}']}
        fast = windows(grid,signal,fast_components,.01,.01,1,t)
        for frame in [slow,fast]:
            frame['linear_mean'] = linear_window_means(t,original,frame)
            frame['linear_mean_rate'] = frame.linear_mean.diff()/frame.mid_s.diff()
        output[channel] = (slow,fast)
        if crop == 0:
            span = np.ptp(original)
            extrema = np.r_[0, np.flatnonzero(np.diff(original)[:-1]*np.diff(original)[1:]<0)+1, len(t)-1]
            extrema_errors=[]
            for k in extrema:
                nearest = np.argsort(np.abs(grid-t[k]))[:2]
                extrema_errors.append(float(np.min(np.abs(signal[nearest]-original[k]))/span))
            global_extrema=[]
            for label,operation in [('minimum',np.argmin),('maximum',np.argmax)]:
                original_index=int(operation(original)); grid_index=int(operation(signal))
                time_error=float(abs(t[original_index]-grid[grid_index]))
                assert time_error <= grid[1]-grid[0]+1e-12
                global_extrema.append(dict(kind=label,raw_time_s=float(t[original_index]),
                                           grid_time_s=float(grid[grid_index]),time_error_s=time_error))
            fidelity.append(dict(channel=channel, grid_intervals=intervals,
                                 grid_dt_s=float(grid[1]-grid[0]),
                                 all_65_knots_used=True, exact_original_endpoints=True,
                                 max_roundtrip_fraction=float(np.max(abs(np.interp(t,grid,signal)-original))/span),
                                 local_extrema_including_endpoints=len(extrema),
                                 maximum_extremum_nearby_amplitude_fraction=max(extrema_errors),
                                 reconstruction_error=error,
                                 global_extrema=global_extrema,
                                 passes_one_percent_fidelity=bool(max(extrema_errors)<.01 and np.max(abs(np.interp(t,grid,signal)-original))/span<.01)))
    return output,fidelity,grid


def main():
    audit = json.loads((ROOT/'01_source_audit/audit.json').read_text())
    hashes = dict(audit['source_hashes'])
    assert all(hash_file(Path(p))==h for p,h in hashes.items())
    raw = pd.read_csv(ROOT/'01_source_audit/BH_raw.csv',float_precision='round_trip')
    base,fidelity,grid = real_variant(raw)
    dt = grid[1]-grid[0]
    for channel,(slow,fast) in base.items():
        slow.to_csv(ROOT/f'02_features/APR18_{channel}_1s.csv',index=False)
        fast.to_csv(ROOT/f'02_features/APR18_{channel}_10ms.csv',index=False)
    np.savez_compressed(ROOT/'02_features/interpolated_inputs.npz',time_s=grid,
                        stress_Pa=np.interp(grid,raw.time_s,raw.stress_Pa),
                        shear_rate_1_s=np.interp(grid,raw.time_s,raw.shear_rate_1_s))

    # Refinement changes level as well as sample interval to preserve scale.
    stability=[]
    for multiplier in [2,4]:
        refined,checks,_ = real_variant(raw,multiplier)
        fidelity.extend(checks)
        shift=int(np.log2(multiplier))
        for channel in base:
            for index, label, names in [(0,'1s',['A13_rate',*[f'R{j}' for j in range(6,14)]]),
                                         (1,'10ms',['A5_rate','R6','R7'])]:
                mapping={n:n[0]+str(int(n.split('_')[0][1:])+shift)+('_rate' if '_rate' in n else '') for n in names}
                rows=compare(base[channel][index],refined[channel][index],names,mapping)
                stability.extend(dict(variant=f'grid_x{multiplier}',channel=channel,window=label,**r) for r in rows)
                refined[channel][index].to_csv(ROOT/f'03_checks/grid_x{multiplier}_{channel}_{label}.csv',index=False)
    for mode,crop in [('constant',0),('reflect',0),('symmetric',1),('symmetric',16),('symmetric',32),('symmetric',1024)]:
        alternative,_,_=real_variant(raw,mode=mode,crop=crop)
        label_variant=f'{mode}_crop{crop}'
        for channel in base:
            for index,label,names in [(0,'1s',['A13_rate',*[f'R{j}' for j in range(6,14)]]),
                                      (1,'10ms',['A5_rate','R6','R7'])]:
                rows=compare(base[channel][index],alternative[channel][index],names)
                stability.extend(dict(variant=label_variant,channel=channel,window=label,**r) for r in rows)
                alternative[channel][index].to_csv(ROOT/f'03_checks/{label_variant}_{channel}_{label}.csv',index=False)
    pd.DataFrame(stability).to_csv(ROOT/'03_checks/feature_sensitivity.csv',index=False)
    assert all(item['passes_one_percent_fidelity'] for item in fidelity)

    # Recompute the previous synthetic feature definitions independently.
    synthetic={}
    prior_checks=[]
    for case in ['control_1','test_clot_1']:
        path=SYNTHETIC/f'test/{case}.csv'
        frame=pd.read_csv(path)
        t=frame.time_s.to_numpy(); signal=frame.stress_Pa.to_numpy()
        comp,error=reconstruct(signal,list(range(1,9)),[8])
        features=windows(t,signal,comp,1,0,120,t)
        features.to_csv(ROOT/f'02_features/synthetic_{case}_1s.csv',index=False)
        synthetic[case]=features
        previous=pd.read_csv(ROOT/f'inputs/previous_synthetic_features/{case}.csv')
        checks=[np.max(abs(features[f'R{j}']-previous[f'R{j}_Pa'])) for j in range(1,9)]
        checks += [np.nanmax(abs(features.A8_rate-previous.A8_slope_Pa_s)),
                   np.max(abs(features.grid_mean-previous.raw_mean_Pa))]
        assert max(checks)<1e-12
        prior_checks.append(dict(case=case,maximum_feature_difference=max(checks),reconstruction_error=error))

        # Isolate the effect of observation spacing: compare dense and sparsified
        # versions on the SAME 63 s grid, origin, levels, and analysis windows.
        sparse_values=np.interp(raw.time_s,t,signal)
        dense_grid=np.interp(grid,t,signal)
        sparse_grid=np.interp(grid,raw.time_s,sparse_values)
        for kind,values,support_time in [('dense',dense_grid,t),('APR18_sampling',sparse_grid,raw.time_s.to_numpy())]:
            comps,_=reconstruct(values,list(range(6,14)),[13])
            f=windows(grid,values,comps,1,1,63,support_time)
            f.to_csv(ROOT/f'05_comparison/{case}_{kind}_63s.csv',index=False)
        pd.DataFrame({'time_s':raw.time_s,'stress_Pa':sparse_values}).to_csv(ROOT/f'05_comparison/{case}_sampled_at_APR18_times.csv',index=False)

    scale_map=pd.DataFrame([dict(synthetic_level=j,synthetic_scale_s=2**j*.01,
                                APR18_level=j+5,APR18_scale_s=2**(j+5)*dt,
                                interpretation='Scale spacing, not exact event duration') for j in range(1,9)])
    scale_map.to_csv(ROOT/'05_comparison/physical_scale_mapping.csv',index=False)
    assert np.max(abs(scale_map.APR18_scale_s/scale_map.synthetic_scale_s-1))<.00001

    # Readable checks target numerical correctness, not success of the detector.
    toy_time=np.arange(10001)*.01
    constant=np.full(len(toy_time),.03)
    toy,_=reconstruct(constant,[1,8],[8])
    assert max(np.max(abs(toy['D1'])),np.max(abs(toy['D8'])))<1e-14
    linear=.01+.002*toy_time
    toy,_=reconstruct(linear,[8],[8])
    f=windows(toy_time,linear,toy,1,0,100,toy_time)
    assert np.max(abs(f.loc[f.A8_rate_interior,'A8_rate']-.002))<1e-12
    assert np.isnan(f.A8_rate.iloc[0])
    exact=linear_window_means(toy_time,linear,f)
    assert np.max(abs(exact-(.01+.002*f.mid_s.to_numpy())))<1e-12
    assert all(hash_file(Path(p))==h for p,h in hashes.items())
    result=dict(config={'wavelet':'db2','mode':'symmetric','analysis':'offline descriptive features',
                        'real_grid_dt_s':dt,'slow_actual_level':13,'early_actual_level':5,
                        'slow_window_s':1,'early_window_s':.01,
                        'boundary_margin_scale_multiplier':6.25,
                        'normalization':False,'threshold_detector':False,
                        'raw_time_origin_preserved':True,'original_points':65,
                        'real_input':'all original points as linear interpolation knots',
                        'user_scope':'APR18 BH 0.1 to 1/s, complete response versus synthetic growth'},
                fidelity=fidelity,prior_synthetic_reproduction=prior_checks,
                checks={'constant_details_zero':True,'linear_trend_preserved_interior':True,
                        'initial_difference_is_missing':True,'source_hashes_unchanged':True,
                        'scale_matching_within_10ppm':True,'exact_linear_window_integration':True,
                        'approximation_component_sum':True,'global_extrema_time_within_one_grid_step':True},
                environment={'python':platform.python_version(),'numpy':np.__version__,
                             'pandas':pd.__version__,'PyWavelets':pywt.__version__},
                source_hashes=hashes,
                limitations=['No independent real event labels or noise-only record.',
                             'Interpolation does not recover unobserved fluctuations.',
                             'Original first point is not a verified loading trigger.',
                             'Matched scales do not make the physical protocols identical.',
                             'Engineering boundary screens do not establish genuine prehistory.'])
    (ROOT/'03_checks/validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'checks':result['checks'],'fidelity':fidelity,'prior':prior_checks},indent=2))


if __name__=='__main__':
    main()
