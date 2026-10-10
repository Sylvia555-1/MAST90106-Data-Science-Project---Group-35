"""Make source-backed figures and a compact numerical comparison summary."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.plot_cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE='#24658a'; ORANGE='#cb642b'; GREY='#888888'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def read(relative):
    return pd.read_csv(ROOT/relative)


def save(fig,name):
    fig.savefig(ROOT/f'04_figures/{name}.png',dpi=160)
    plt.close(fig)


def main():
    raw=read('01_source_audit/BH_raw.csv')
    early=read('02_features/APR18_stress_Pa_10ms.csv')
    real=read('02_features/APR18_stress_Pa_1s.csv')
    synth=read('02_features/synthetic_test_clot_1_1s.csv')
    control=read('02_features/synthetic_control_1_1s.csv')
    validation=json.loads((ROOT/'03_checks/validation.json').read_text())
    dt=validation['config']['real_grid_dt_s']

    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,(limits,title) in enumerate([((0,63),'Full record'),((0,1),'First second')]):
        for row,(channel,unit,color) in enumerate([('shear_rate_1_s','Shear rate (1/s)',BLUE),('stress_Pa','Stress (Pa)',ORANGE)]):
            ax=axes[row,col]
            ax.plot(raw.time_s,raw[channel],'.-',color=color,lw=1,ms=5,label='Original samples')
            ax.set(xlim=limits,ylabel=unit,xlabel='Recorded time (s)')
            ax.grid(alpha=.2)
        axes[0,col].set_title(title)
    peak=raw.loc[raw.stress_Pa.idxmax()]
    axes[1,1].annotate(f'Observed peak: {peak.time_s:.3f} s',xy=(peak.time_s,peak.stress_Pa),
                       xytext=(.53,.043),arrowprops={'arrowstyle':'->','color':GREY})
    fig.suptitle('APR18 0.1 to 1/s: input and stress response\n65 original points; lines only guide the eye; no event labels')
    save(fig,'01_original_response')

    fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True,layout='constrained')
    axes[0].plot(raw.time_s,raw.stress_Pa,'o',ms=3,color=GREY,label='Original stress samples')
    axes[0].plot(early.mid_s,early.A5,color=ORANGE,label='A5 window mean (scale ~0.01 s)')
    axes[0].plot(early.mid_s,early.linear_mean,'--',color=BLUE,label='Linear-interpolant window mean')
    axes[0].set_ylabel('Stress (Pa)')
    axes[1].plot(early.mid_s,early.A5_rate,color=ORANGE,label='A5 mean change rate')
    axes[1].plot(early.mid_s,early.linear_mean_rate,'--',color=BLUE,label='Ordinary trend change rate')
    axes[1].axvspan(0,.01+6.25*(2**5)*dt,color=GREY,alpha=.15,label='Boundary-sensitive rate windows')
    axes[1].axhline(0,color=GREY,lw=.7)
    axes[1].set_ylabel('Change rate (Pa/s)')
    for level,color in [(6,BLUE),(7,ORANGE)]:
        axes[2].plot(early.mid_s,early[f'R{level}'],color=color,label=f'R{level}: scale ~{2**level*dt:.2f} s')
        axes[2].axvline(6.25*2**level*dt,color=color,ls=':',lw=1)
    axes[2].set_ylabel('Detail RMS (Pa)')
    axes[2].text(.48,.86,'Left of each dotted line:\nboundary-sensitive scale',transform=axes[2].transAxes,fontsize=9)
    axes[3].bar(early.mid_s,early.original_points,width=.008,color=BLUE)
    axes[3].set(ylabel='Original points\nper 10 ms',xlabel='Recorded time / window midpoint (s)',xlim=(0,1))
    for ax in axes[:3]:ax.legend(fontsize=8,loc='upper right')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Early real response: feature definitions retained, faster scales added\n10 ms feature spacing does not imply 10 ms measurement resolution')
    save(fig,'02_early_features')

    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    axes[0,0].plot(synth.mid_s,synth.grid_mean,color=ORANGE,label='Synthetic added growth')
    axes[0,0].plot(control.mid_s,control.grid_mean,color=BLUE,label='Synthetic control')
    axes[0,1].plot(raw.time_s,raw.stress_Pa,'.-',color=ORANGE,label='APR18 original stress')
    axes[0,0].set_title('Synthetic: 0.5 to 1/s, added growth at 40 s')
    axes[0,1].set_title('APR18: 0.1 to 1/s, no known extra-growth label')
    for ax in axes[0]:ax.set_ylabel('Stress (Pa)');ax.legend(fontsize=8)
    for col,frame,approx,detail,reference in [(0,synth,'A8','R8','grid_mean_rate'),(1,real,'A13','R13','linear_mean_rate')]:
        axes[1,col].plot(frame.mid_s,frame[f'{approx}_rate'],color=ORANGE,label=f'{approx} change rate')
        axes[1,col].plot(frame.mid_s,frame[reference],'--',color=BLUE,label='Ordinary trend change rate')
        axes[1,col].axhline(0,color=GREY,lw=.7)
        axes[1,col].set_ylabel('Change rate (Pa/s)')
        axes[1,col].legend(fontsize=8)
        axes[2,col].plot(frame.mid_s,frame[detail],color=ORANGE,label=f'{detail} (same ~2.56 s scale)')
        axes[2,col].set(ylabel='Detail RMS (Pa)',xlabel='Recorded time / window midpoint (s)')
        axes[2,col].legend(fontsize=8)
        end=120 if col==0 else 63
        for ax in axes[1:,col]:
            ax.axvspan(0,16,color=GREY,alpha=.15)
            ax.axvspan(end-16,end,color=GREY,alpha=.15)
            ax.set_xlim(0,end)
        # Show only screened values so boundary peaks cannot set the y-axis.
        for row,name in [(1,f'{approx}_rate'),(2,detail)]:
            mask=frame[f'{name}_interior']
            values=frame.loc[mask,name].to_numpy()
            if row==1: values=np.r_[values,frame.loc[mask,reference]]
            low,high=min(0,values.min()),max(0,values.max())
            pad=max((high-low)*.15,1e-10)
            axes[row,col].set_ylim(low-pad,high+pad)
            # Draw invalid portions only as gaps: all values remain in CSV.
            for line in axes[row,col].lines[:(2 if row==1 else 1)]:
                y=line.get_ydata().copy();y[~mask]=np.nan;line.set_ydata(y)
        for ax in axes[:,col]:
            if col==0:ax.axvline(40,color=GREY,ls=':',lw=1)
            ax.grid(alpha=.2)
    fig.suptitle('Same feature definitions and matched time scales, different response shapes\nIndependent time and vertical ranges; grey areas excluded for the slow DWT scale')
    save(fig,'03_synthetic_real_comparison')

    dense=read('05_comparison/test_clot_1_dense_63s.csv')
    sparse=read('05_comparison/test_clot_1_APR18_sampling_63s.csv')
    sampled=read('05_comparison/test_clot_1_sampled_at_APR18_times.csv')
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    axes[0].plot(dense.mid_s,dense.grid_mean,color=BLUE,label='Dense synthetic input')
    axes[0].plot(sparse.mid_s,sparse.grid_mean,color=ORANGE,label='Same signal sampled at APR18 times, then interpolated')
    axes[0].plot(sampled.time_s,sampled.stress_Pa,'ko',ms=4,label='Retained synthetic samples')
    axes[0].set_ylabel('Stress / window mean (Pa)')
    for frame,color,label in [(dense,BLUE,'Dense input A13 rate'),(sparse,ORANGE,'Sparse input A13 rate')]:
        axes[1].plot(frame.mid_s,frame.A13_rate,color=color,label=label)
    axes[1].set(ylabel='Change rate (Pa/s)',xlabel='Recorded time / window midpoint (s)',xlim=(30,50))
    axes[0].set_ylim(.023,.034)
    shown=np.r_[dense.loc[dense.mid_s.between(30,50),'A13_rate'],sparse.loc[sparse.mid_s.between(30,50),'A13_rate']]
    axes[1].set_ylim(min(0,shown.min())-.0001,max(0,shown.max())+.00015)
    for ax in axes:
        ax.axvline(40,color=GREY,ls=':',label='Known synthetic onset: 40 s')
        ax.axvspan(47,50,color=GREY,alpha=.15)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1].axhline(0,color=GREY,lw=.7)
    fig.suptitle('Sampling diagnostic: the same synthetic signal, two observation densities\nSame 63 s domain and DWT settings; grey region is boundary-sensitive')
    save(fig,'04_sampling_diagnostic')

    sensitivity=read('03_checks/feature_sensitivity.csv')
    summary_rows=[]
    for label,frame,feature,lo,hi in [('synthetic_growth',synth,'A8_rate',40,70),
                                     ('synthetic_control',control,'A8_rate',40,70),
                                     ('APR18_late',real,'A13_rate',18,47)]:
        mask=frame[f'{feature}_interior'] & (frame.left_s>=lo) & (frame.right_s<=hi)
        # The rate uses two windows; enforce both within the descriptive interval.
        mask &= frame.left_s.shift(1)>=lo
        values=frame.loc[mask,feature]
        summary_rows.append(dict(case=label,feature=feature,left_s=lo,right_s=hi,
                                 windows=len(values),mean_rate=float(values.mean()),
                                 negative_windows=int((values<0).sum()),positive_windows=int((values>0).sum())))
    pd.DataFrame(summary_rows).to_csv(ROOT/'05_comparison/interval_summary.csv',index=False)
    known_gap=raw.loc[(raw.time_s<=40),'time_s'].max(),raw.loc[(raw.time_s>=40),'time_s'].min()
    initial_input=raw[(raw.time_s>=.042)&(raw.time_s<=1)]
    summary=dict(real_observed_stress_min={'time_s':float(raw.loc[raw.stress_Pa.idxmin(),'time_s']),'stress_Pa':float(raw.stress_Pa.min())},
                 real_observed_stress_peak={'time_s':float(peak.time_s),'stress_Pa':float(peak.stress_Pa)},
                 real_last_stress_Pa=float(raw.stress_Pa.iloc[-1]),
                 peak_to_last_fraction=float((peak.stress_Pa-raw.stress_Pa.iloc[-1])/peak.stress_Pa),
                 measured_input_range_0p042_to_1_s=[float(initial_input.shear_rate_1_s.min()),float(initial_input.shear_rate_1_s.max())],
                 APR18_sampling_bracket_around_synthetic_onset=list(known_gap),
                 interval_results=summary_rows,
                 feature_sensitivity=sensitivity[(sensitivity.channel=='stress_Pa') & sensitivity.feature.isin(['A13_rate','R13','A5_rate','R6','R7'])].to_dict('records'))
    (ROOT/'05_comparison/result_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='feature_sensitivity'},indent=2))


if __name__=='__main__':
    main()
