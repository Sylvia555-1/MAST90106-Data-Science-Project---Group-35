"""Independent checks for detection rules, data splits, and DWT means."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
import run_experiment as ex

root=Path(__file__).resolve().parent
cases=json.loads((root/'01_protocol/cases.json').read_text())
assert len({c['seed'] for c in cases})==len(cases)
assert len({c['case_id'] for c in cases})==len(cases)
thresholds=json.loads((root/'03_evaluation/frozen_thresholds.json').read_text())
arr=np.load(root/'02_features/synthetic_window_features.npz')
order=json.loads((root/'02_features/array_order.json').read_text())
pred=pd.read_csv(root/'03_evaluation/test_predictions.csv')
rows=[]
# Independently locate the first two adjacent exceedances by a literal loop.
for i,c in enumerate(cases):
 if c['split']!='test':continue
 for a,arm in enumerate(order['arm_order']):
  rates=arr['rates'][2*i+a]
  for j,name in enumerate(order['feature_order']):
   found=np.nan
   for k in range(len(ex.MID)-1):
    if ex.SAFE[k] and ex.SAFE[k+1] and rates[k,j]>thresholds[arm][name] and rates[k+1,j]>thresholds[arm][name]:
     found=ex.MID[k];break
   saved=pred[(pred.case_id==c['case_id'])&(pred.arm==arm)&(pred.feature==name)].iloc[0]
   assert (np.isnan(found) and np.isnan(saved.prediction_s)) or abs(found-saved.prediction_s)<1e-12
# Compute each level directly rather than truncating a level-7 coefficient tree.
y=np.interp(ex.GRID,ex.RT,ex.RAW.stress_Pa)
real=pd.read_csv(root/'02_features/real_APR18_features.csv')
for j in ex.LEVELS:
 co=pywt.wavedec(y,'db2',level=j,mode='symmetric')
 co=[co[0]]+[np.zeros_like(c) for c in co[1:]]
 x=pywt.waverec(co,'db2',mode='symmetric')[:len(y)]
 manual=np.array([np.mean(x[(ex.GRID>=l)&(ex.GRID<r)]) for l,r in zip(ex.LEFT,ex.RIGHT)])
 err=float(np.max(abs(manual-real[f'A{j}_mean'])))
 assert err<1e-10
 rows.append(dict(level=j,maximum_direct_mean_error_Pa=err))
for filename,digest in json.loads((root/'01_protocol/frozen_digests.json').read_text()).items():
 assert ex.source_hash(root/'01_protocol'/filename)==digest
out=dict(independent_prediction_checks=len(pred),all_predictions_match=True,distinct_seeds=len(cases),direct_approximation_checks=rows,protocol_digests_unchanged=True)
(root/'03_evaluation/independent_validation.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
