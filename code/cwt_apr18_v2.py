"""APR18 semi-synthetic v2: finite-window Morlet CWT benchmark. based on v2 of synthetic data

python cwt_apr18_v2.py
python -m pip install numpy pandas matplotlib
E:\\pt\\apr18_semisynthetic_laos_v2

1. The principal analysis utilised an observation window of 628 points / 12.56 s, with a step size of 25 points / 0.5 s,
   consistent with the LAOS_FFT_STFT_Progress report. The data was first interpolated to a 50 Hz grid in absolute time.
2. The complex Morlet CWT is computed directly using discrete calculations, without calling the removed `scipy.signal.cwt`.
   Mother wavelet: exp(-u**2/B) * exp(2j*pi*u) / sqrt(pi*B), where B = 3.
   Scale = fs/f. The Gaussian width at different frequencies varies with scale; therefore, this is not an STFT.
3. This is a finite-observation window CWT: each coefficient uses only the data within the current window; no data is interpolated from outside the window.
   This differs from an infinitely supported CWT applied to the entire signal; low-frequency kernel truncation and weak harmonic crosstalk may introduce bias.
   628 points is the observation buffer length; this does not imply that all frequencies have the same effective temporal resolution.
   The same centre is used for all scales within the same window; all alarm times are recorded as the right-hand end of the window (exclusive end), 
   and the centre time is not treated as a detection time. Short-window analysis uses a fixed mother wavelet and does not adjust parameters based on detection results. 
4. In-window mean subtraction is equivalent to subtracting the mean of the kernel from each discrete kernel.
   C = sum(x*k); harmonic amplitude = 2*abs(C)/abs(sum(k*exp(2j*pi*f*t))).
   This calibrates the positive-frequency gain but cannot completely eliminate negative-frequency response and harmonic crosstalk.
   Quantify these errors using a known independent calibration signal (24 initial phases).
5. CWT spectral-density proxy S = 2 * abs(C)² / (fs * sum(abs(k)²)).
   S is expressed in Pa²/Hz; the expected level of white noise is normalised by the kernel energy.
   It is not the Hann FFT PSD, nor is it guaranteed that its integral equals the total variance of the signal.
   Power = sum(S * df), p = S * df / Power, H = -sum(p * log(p)) / log(M),
   BW = sqrt(sum(p * (f - sum(p * f))**2)). Calculated only within the range 0.5*f0 to 8*f0,
   using the rFFT frequency grid corresponding to the STFT to align the frequency intervals and the number of entropy bins.
   The dense time-frequency plot is for display purposes only and is not used to calculate these features.
6. For each window length, the median and 1.4826*MAD are calculated separately using all outputs from the ‘control’ channel.
   Main rule: z ≥ 5 or z ≤ -5 on the same side persists for at least 1 s, i.e. across three consecutive outputs.
   NaN, direction changes or discontinuous outputs will interrupt the sequence; each instance of duration exceeding the threshold counts as a single alarm.
   When MAD is zero, retain the invalid fraction; do not add an arbitrary epsilon to generate large z-values.
7. Read `case_summary.csv` for truth-table evaluation only after all features and alarms have been frozen.
   Baseline stress and state changes in `truth/` shall not be used for features, thresholds or detection.
8. The control reference for the entire period constitutes offline calibration; the number of alarms from the control itself constitutes an in-calibration diagnosis.
   Three injection levels against the same background cannot be used to estimate population detection rates or clinical clotting/gelation times.
   Results from one cycle constitute an exploratory sensitivity analysis; in particular, attention should be paid to weak harmonic calibration errors.

Source: User-provided LAOS_FFT_STFT_Progress report, pp. 1–4; data package README.
Morlet https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
"""

from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


#usually only need to make changes here; there is no need to search directories or carry out additional checks, such as verifying whether files exist.
DATA_DIR = Path(r"E:\pt\apr18_semisynthetic_laos_v2")
FS = 50.0
F0 = 1.0 / (2.0 * np.pi)  # omega = 1 rad/s
HOP_S = 0.5
MORLET_B = 3.0
WINDOW_CYCLES = (2, 1)  
PRIMARY_Z = 5.0
PRIMARY_PERSISTENCE_S = 1.0
Z_THRESHOLDS = (3.0, 5.0, 8.0)
PERSISTENCE_SECONDS = (0.0, 1.0, 3.0)
CASES = ("control", "change_10pct", "change_20pct", "change_30pct")
FEATURES = ("I1", "I3", "I5", "I3_I1", "I5_I1", "power", "entropy", "bandwidth")
LABELS = ("I1", "I3", "I5", "I3/I1", "I5/I1", "Power proxy", "Entropy", "Bandwidth")
UNITS = ("Pa", "Pa", "Pa", "ratio", "ratio", "Pa²", "unitless", "Hz")
COLORS = ("#67727E", "#007F86", "#D28A1A", "#9A4A94")
CASE_LABELS = ("Control", "+10% gain", "+20% gain", "+30% gain")

# The main results for the two cycles on page 3 of the report, for comparison; these are not STFT outputs calculated by this script.<\n>
# This table corresponds to FS=50, N=628, hop=0.5, z=5, a duration of 1 second, and the complete control reference.
STFT_REPORT_TIMES = {
    "change_10pct": [26.12, 29.62, np.nan, np.nan, np.nan, 26.12, 28.12, np.nan],
    "change_20pct": [24.12, 25.62, 27.12, np.nan, np.nan, 24.62, 20.62, np.nan],
    "change_30pct": [23.12, 23.62, 25.62, np.nan, np.nan, 23.62, 20.12, np.nan],
}


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def load_observables(data_dir):
    raw = {case: pd.read_csv(data_dir / "observable" / f"{case}.csv") for case in CASES}
    base_t = raw["control"]["time_s"].to_numpy()
    # The data grid for this experiment is 0.06, 0.08, …, 39.98 s; this is consistent with the reported timing method.
    t = np.arange(np.ceil(base_t[0] * FS), np.floor(base_t[-1] * FS) + 1) / FS
    signals = {
        case: np.interp(t, frame["time_s"], frame["stress_Pa"])
        for case, frame in raw.items()
    }
    return t, signals


def morlet_bank(n, frequencies):
    """CWT kernels centred on a finite window, and positive frequency-amplitude gain."""
    offset = np.arange(n) - (n - 1) / 2.0
    scales = FS / np.asarray(frequencies)
    u = offset[:, None] / scales[None, :]
    kernels = np.exp(-u * u / MORLET_B - 2j * np.pi * u)
    kernels /= np.sqrt(np.pi * MORLET_B * scales[None, :])
    kernels -= kernels.mean(axis=0, keepdims=True)  # This is equivalent to removing the mean from the signal in each window.
    response = np.abs(np.sum(kernels * np.exp(2j * np.pi * u), axis=0))
    return kernels, response


def make_estimator(n):
    fft_f = np.fft.rfftfreq(n, 1.0 / FS)
    band_f = fft_f[(fft_f >= 0.5 * F0) & (fft_f <= 8.0 * F0)]
    frequencies = np.r_[np.array([1, 3, 5]) * F0, band_f]
    kernels, response = morlet_bank(n, frequencies)
    return {
        "n": n, "frequencies": band_f, "df": FS / n,
        "kernels": kernels, "response": response,
        "kernel_energy": np.sum(np.abs(kernels[:, 3:]) ** 2, axis=0),
    }


def estimate_features(windows, estimator):
    coefficients = np.asarray(windows) @ estimator["kernels"]
    amplitudes = 2.0 * np.abs(coefficients[:, :3]) / estimator["response"][:3]
    ratios = np.divide(
        amplitudes[:, 1:], amplitudes[:, :1],
        out=np.full_like(amplitudes[:, 1:], np.nan), where=amplitudes[:, :1] > 0,
    )
    spectrum = 2.0 * np.abs(coefficients[:, 3:]) ** 2
    spectrum /= FS * estimator["kernel_energy"]
    mass = spectrum * estimator["df"]
    power = mass.sum(axis=1)
    p = np.divide(mass, power[:, None], out=np.full_like(mass, np.nan), where=power[:, None] > 0)
    log_p = np.zeros_like(p)
    np.log(p, out=log_p, where=p > 0)
    entropy = -(p * log_p).sum(axis=1) / np.log(len(estimator["frequencies"]))
    centroid = p @ estimator["frequencies"]
    bandwidth = np.sqrt((p * (estimator["frequencies"] - centroid[:, None]) ** 2).sum(axis=1))
    values = np.column_stack((amplitudes, ratios, power, entropy, bandwidth))
    return values


def extract_case(t, x, estimator):
    n = estimator["n"]
    hop = round(HOP_S * FS)
    starts = np.arange(0, len(x) - n + 1, hop)
    windows = sliding_window_view(x, n)[::hop]
    frame = pd.DataFrame(estimate_features(windows, estimator), columns=FEATURES)
    frame.insert(0, "window_start_s", t[starts])
    frame.insert(1, "center_s", t[starts] + (n - 1) / (2.0 * FS))
    frame.insert(2, "window_end_s", t[starts] + n / FS)
    return frame


def calibrate_reference(control):
    values = control[list(FEATURES)].to_numpy()
    median = np.nanmedian(values, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(values - median), axis=0)
    return pd.DataFrame({
        "feature": FEATURES, "median": median, "mad_scale": scale,
        "valid_scale": np.isfinite(scale) & (scale > 0),
        "n_reference_outputs": len(control),
    })


def add_scores(frame, reference):
    result = frame.copy()
    values = result[list(FEATURES)].to_numpy()
    med = reference["median"].to_numpy()
    scale = reference["mad_scale"].to_numpy()
    z = np.divide(values - med, scale, out=np.full_like(values, np.nan),
                  where=reference["valid_scale"].to_numpy())
    effect = np.divide(100.0 * (values - med), np.abs(med),
                       out=np.full_like(values, np.nan), where=np.isfinite(med) & (med != 0))
    for j, feature in enumerate(FEATURES):
        result[f"z_{feature}"] = z[:, j]
        result[f"effect_pct_{feature}"] = effect[:, j]
    return result


def alarm_episodes(frame, feature, threshold, persistence):
    """Do not use true values; record all persistent same-side out-of-bounds events, recording the confirmation only once per segment."""
    times = frame["window_end_s"].to_numpy()
    scores = frame[f"z_{feature}"].to_numpy()
    episodes = []
    previous_side, run_start, confirmed = 0, 0, False
    for i, score in enumerate(scores):
        side = 0 if not np.isfinite(score) else (1 if score >= threshold else (-1 if score <= -threshold else 0))
        adjacent = i > 0 and np.isclose(times[i] - times[i - 1], HOP_S, atol=1e-8, rtol=0)
        if side == 0:
            previous_side, confirmed = 0, False
            continue
        if side != previous_side or not adjacent:
            run_start, confirmed = i, False
        if not confirmed and times[i] - times[run_start] >= persistence - 1e-8:
            episodes.append({
                "run_start_s": times[run_start], "confirmation_s": times[i],
                "direction": "increase" if side > 0 else "decrease",
                "score_at_confirmation": score,
                "feature_at_confirmation": frame.iloc[i][feature],
                "effect_pct_at_confirmation": frame.iloc[i][f"effect_pct_{feature}"],
            })
            confirmed = True
        previous_side = side
    return episodes


def calibration_experiment():
    """Calibration based on independent known signals, without selecting parameters based on injected true values."""
    amplitudes = np.array([1.0, 0.10, 0.03])
    harmonics = np.array([1, 3, 5])
    rows = []
    for cycles in (0.5, 1, 2):
        n = round(cycles * FS / F0)
        estimator = make_estimator(n)
        local_t = (np.arange(n) - (n - 1) / 2) / FS
        for phase_index, phase in enumerate(np.linspace(0, 2 * np.pi, 24, endpoint=False)):
            theta = 2 * np.pi * F0 * local_t[:, None] * harmonics
            theta += phase * harmonics + np.array([0.0, 0.4, -0.7])
            mixture = (np.cos(theta) * amplitudes).sum(axis=1)
            estimated = estimate_features(mixture[None, :], estimator)[0, :5]
            expected = np.r_[amplitudes, amplitudes[1:] / amplitudes[0]]
            for feature, actual, target in zip(FEATURES[:5], estimated, expected):
                rows.append({"test": "harmonic_mixture", "cycles": cycles,
                             "phase_index": phase_index, "feature": feature,
                             "expected": target, "estimated": actual,
                             "error_pct": 100 * (actual / target - 1)})
            for j, harmonic in enumerate(harmonics):
                tone = np.cos(theta[:, j])
                actual = estimate_features(tone[None, :], estimator)[0, j]
                rows.append({"test": "isolated_unit_tone", "cycles": cycles,
                             "phase_index": phase_index, "feature": f"I{harmonic}",
                             "expected": 1.0, "estimated": actual,
                             "error_pct": 100 * (actual - 1)})
    details = pd.DataFrame(rows)
    summary = (details.assign(abs_error_pct=details["error_pct"].abs())
               .groupby(["test", "cycles", "feature"], as_index=False)
               .agg(max_abs_error_pct=("abs_error_pct", "max"),
                    mean_abs_error_pct=("abs_error_pct", "mean")))
    return details, summary


def evaluate_alarms(frozen, case_info):
    """The alarm has been frozen; this function uses ‘onset’ to calculate the delay and the pre-injection alarm."""
    summary, events = [], []
    for (cycles, case, feature, threshold, persistence), episodes in frozen.items():
        info = case_info.loc[case]
        onset = info["ground_truth_onset_s"]
        first = episodes[0] if episodes else {}
        first_time = first.get("confirmation_s", np.nan)
        is_control = case == "control"
        if is_control:
            status = "control_alarm" if episodes else "control_no_alarm"
        elif not episodes:
            status = "no_alarm"
        elif first_time < onset:
            status = "pre_onset_first_alarm"
        elif first["run_start_s"] < onset:
            status = "run_started_before_onset"
        else:
            status = "post_onset_first_alarm"
        post = [e for e in episodes if np.isfinite(onset) and e["run_start_s"] >= onset]
        pre_count = sum(e["confirmation_s"] < onset for e in episodes) if not is_control else 0
        row = {
            "cycles": cycles, "case": case, "feature": feature,
            "threshold": threshold, "persistence_s": persistence,
            "gain_fraction": info["final_amplitude_change_fraction"],
            "truth_onset_s": onset, "status": status,
            "first_confirmation_s": first_time,
            "first_alarm_delay_s": first_time - onset if status == "post_onset_first_alarm" else np.nan,
            "first_eligible_post_confirmation_s": post[0]["confirmation_s"] if post else np.nan,
            "n_pre_onset_alarms": pre_count, "n_alarm_episodes": len(episodes),
            "first_direction": first.get("direction", ""),
            "first_score": first.get("score_at_confirmation", np.nan),
            "first_effect_pct": first.get("effect_pct_at_confirmation", np.nan),
        }
        summary.append(row)
        for episode in episodes:
            events.append({"cycles": cycles, "case": case, "feature": feature,
                           "threshold": threshold, "persistence_s": persistence, **episode})
    return pd.DataFrame(summary), pd.DataFrame(events)


def comparison_with_report(primary):
    rows = []
    for case, stft_times in STFT_REPORT_TIMES.items():
        for feature, stft_time in zip(FEATURES, stft_times):
            result = primary[(primary["case"] == case) & (primary["feature"] == feature)].iloc[0]
            stft_delay = stft_time - result["truth_onset_s"]
            cwt_delay = result["first_alarm_delay_s"]
            rows.append({"case": case, "feature": feature,
                         "cwt_status": result["status"],
                         "cwt_confirmation_s": result["first_confirmation_s"],
                         "cwt_delay_s": cwt_delay,
                         "stft_report_status": "no_alarm" if np.isnan(stft_time) else "alarm",
                         "stft_report_confirmation_s": stft_time,
                         "stft_report_delay_s": stft_delay,
                         "cwt_minus_stft_delay_s": cwt_delay - stft_delay})
    return pd.DataFrame(rows)


def plot_outputs(t, signals, frames, primary, case_info, estimator, calibration, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    onset = case_info.loc["change_10pct", "ground_truth_onset_s"]

    def save(fig, name):
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, sharey=True, layout="constrained")
    for ax, case, label, color in zip(axes.flat, CASES, CASE_LABELS, COLORS):
        ax.plot(t, signals[case], color=color, lw=0.9)
        if case != "control":
            ax.axvline(onset, color="#B63D45", ls="--", lw=1)
        ax.set(title=label, xlabel="Time (s)", ylabel="Stress (Pa)")
    fig.suptitle("APR18 v2: observed stress")
    save(fig, "01_stress_signals.png")

    for score_plot, name in ((False, "03_eight_features.png"), (True, "04_robust_scores.png")):
        fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, layout="constrained")
        for j, (ax, feature) in enumerate(zip(axes.flat, FEATURES)):
            for case, label, color in zip(CASES, CASE_LABELS, COLORS):
                frame = frames[case]
                ax.plot(frame["window_end_s"], frame[f"z_{feature}" if score_plot else feature],
                        label=label, color=color, lw=1.3)
                if score_plot:
                    row = primary[(primary["case"] == case) & (primary["feature"] == feature)].iloc[0]
                    if np.isfinite(row["first_confirmation_s"]):
                        ax.scatter(row["first_confirmation_s"], row["first_score"], color=color,
                                   marker="o", s=32, zorder=5)
            if score_plot:
                for level in (-PRIMARY_Z, PRIMARY_Z):
                    ax.axhline(level, color="#777777", ls=":", lw=0.9)
            ax.axvline(onset, color="#B63D45", ls="--", lw=1)
            ax.set(title=LABELS[j], ylabel="Robust z score" if score_plot else UNITS[j])
            ax.grid(alpha=0.15)
        axes[0, 0].legend(ncol=2, fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("Available time: exclusive window end (s)")
        fig.suptitle("CWT: two-cycle primary analysis" + ("; dots = first confirmation" if score_plot else ""))
        save(fig, name)

    # Only the dense grid is displayed; common colour scale. The x-axis continues to use the time to output, to avoid misinterpretation as early detection.
    n = estimator["n"]
    dense_f = np.linspace(0.5 * F0, 8 * F0, 151)
    dense_k, _ = morlet_bank(n, dense_f)
    density = []
    for case in CASES:
        windows = sliding_window_view(signals[case], n)[::round(HOP_S * FS)]
        c = windows @ dense_k
        s = 2 * np.abs(c) ** 2 / (FS * np.sum(np.abs(dense_k) ** 2, axis=0))
        density.append(10 * np.log10(np.maximum(s.T, 1e-30)))
    vmin, vmax = np.min(density), np.max(density)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, layout="constrained")
    for ax, case, label, spectrum in zip(axes.flat, CASES, CASE_LABELS, density):
        pcm = ax.pcolormesh(frames[case]["window_end_s"], dense_f / F0, spectrum,
                            cmap="magma", shading="auto", vmin=vmin, vmax=vmax)
        for harmonic in (1, 3, 5):
            ax.axhline(harmonic, color="white", lw=0.6, ls=":", alpha=0.7)
        if case != "control":
            ax.axvline(onset, color="#55E0D1", ls="--", lw=1)
        ax.set(title=label, xlabel="Exclusive window end (s)", ylabel="Frequency / f0")
    fig.colorbar(pcm, ax=axes, label="CWT density proxy (dB re 1 Pa²/Hz)", shrink=0.8)
    fig.suptitle(f"Finite-window CWT; coefficients centred {(n + 1) / (2 * FS):.2f} s before the displayed time")
    save(fig, "02_cwt_scalograms.png")

    # NaN indicates a ‘not detected’ or ‘fail’ initial alarm; if left blank, it is treated as a zero delay.
    delay = primary[primary["case"] != "control"].pivot(index="feature", columns="case", values="first_alarm_delay_s")
    matrix = delay.reindex(index=FEATURES, columns=CASES[1:]).to_numpy()
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#ECEEF1")
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    finite = matrix[np.isfinite(matrix)]
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(1.0, finite.max()) if finite.size else 1)
    for i, feature in enumerate(FEATURES):
        for j, case in enumerate(CASES[1:]):
            row = primary[(primary["feature"] == feature) & (primary["case"] == case)].iloc[0]
            value = matrix[i, j]
            label = f"{value:.2f} s" if np.isfinite(value) else ("No alarm" if row["status"] == "no_alarm" else "Earlier/background run")
            dark = finite.size and np.isfinite(value) and value > 0.6 * finite.max()
            ax.text(j, i, label, ha="center", va="center", color="white" if dark else "#252B34", fontsize=9)
    ax.set_xticks(range(3), CASE_LABELS[1:])
    ax.set_yticks(range(8), LABELS)
    ax.set_title("CWT confirmation delay from known injection onset")
    fig.colorbar(image, ax=ax, label="Delay (s)")
    save(fig, "05_confirmation_delays.png")

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for offset, cycles in enumerate((0.5, 1, 2)):
        part = calibration[(calibration["test"] == "harmonic_mixture") & (calibration["cycles"] == cycles)]
        errors = part.set_index("feature").reindex(FEATURES[:5])["max_abs_error_pct"]
        bars = ax.bar(np.arange(5) + (offset - 1) * 0.24, errors, width=0.24, label=f"{cycles:g} cycles")
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set_xticks(range(5), LABELS[:5])
    ax.set_yscale("log")
    ax.margins(y=0.25)
    ax.set(ylabel="Maximum absolute error (%, log scale)", title="Calibration: stationary known mixture, 24 starting phases")
    ax.legend()
    save(fig, "06_harmonic_calibration.png")


def main(data_dir=DATA_DIR, output_dir=None):
    data_dir = Path(data_dir)
    out = Path(output_dir) if output_dir is not None else data_dir / "cwt_results"
    out.mkdir(parents=True, exist_ok=True)
    t, signals = load_observables(data_dir)
    frozen, all_frames, references, run_times = {}, {}, [], []

    for cycles in WINDOW_CYCLES:
        n = round(cycles * FS / F0)
        estimator = make_estimator(n)
        extracted = {}
        for case in CASES:
            start = perf_counter()
            extracted[case] = extract_case(t, signals[case], estimator)
            run_times.append({"cycles": cycles, "case": case,
                              "feature_extraction_seconds": perf_counter() - start})
        reference = calibrate_reference(extracted["control"])
        references.append(reference.assign(cycles=cycles))
        all_frames[cycles] = {case: add_scores(extracted[case], reference) for case in CASES}
        for case, frame in all_frames[cycles].items():
            save_csv(frame, out / f"features_{cycles}cycle_{case}.csv")
            for feature in FEATURES:
                for threshold in Z_THRESHOLDS:
                    for persistence in PERSISTENCE_SECONDS:
                        key = (cycles, case, feature, threshold, persistence)
                        frozen[key] = alarm_episodes(frame, feature, threshold, persistence)

    # At this stage, all test results have been frozen and the true values have not been read.
    case_info = pd.read_csv(data_dir / "case_summary.csv").set_index("case")
    sensitivity, events = evaluate_alarms(frozen, case_info)
    primary = sensitivity[(sensitivity["cycles"] == 2)
                          & (sensitivity["threshold"] == PRIMARY_Z)
                          & (sensitivity["persistence_s"] == PRIMARY_PERSISTENCE_S)].copy()
    comparison = comparison_with_report(primary)
    details, calibration = calibration_experiment()

    # The interval [26.00, 38.56), which corresponds to the late reporting window: used solely to explain changes in magnitude/proportion.
    n = round(2 * FS / F0)
    estimator = make_estimator(n)
    late_start = round((26.0 - t[0]) * FS)
    late_values = estimate_features(np.stack([signals[c][late_start:late_start+n] for c in CASES]), estimator)
    late_rows = []
    for i, case in enumerate(CASES):
        for j, feature in enumerate(FEATURES):
            ref = late_values[0, j]
            late_rows.append({"case": case, "feature": feature,
                              "window_start_s": 26.0, "window_end_s": 26.0 + n / FS,
                              "value": late_values[i, j], "control_value": ref,
                              "change_vs_control_pct": 100 * (late_values[i, j] / ref - 1) if ref != 0 else np.nan})

    # This refers only to the smallest tested gear ratio detected in this context; it is not referred to as the ‘reliable detection limit’.
    minimum = []
    for feature in FEATURES:
        part = primary[primary["feature"] == feature]
        control_alarms = int(part[part["case"] == "control"]["n_alarm_episodes"].iloc[0])
        eligible = part[(part["status"] == "post_onset_first_alarm") & (part["n_pre_onset_alarms"] == 0)]
        minimum.append({"feature": feature, "control_alarm_episodes": control_alarms,
                        "minimum_observed_tested_gain": eligible["gain_fraction"].min() if control_alarms == 0 else np.nan,
                        "independent_backgrounds": 1})

    outputs = {
        "primary_detection_summary.csv": primary,
        "alarm_events_all_settings.csv": events,
        "parameter_sensitivity.csv": sensitivity,
        "cwt_vs_stft_report.csv": comparison,
        "reference_statistics.csv": pd.concat(references, ignore_index=True),
        "calibration_details.csv": details,
        "calibration_summary.csv": calibration,
        "late_window_summary.csv": pd.DataFrame(late_rows),
        "minimum_observed_tested_gain.csv": pd.DataFrame(minimum),
        "feature_extraction_runtime.csv": pd.DataFrame(run_times),
    }
    for filename, frame in outputs.items():
        save_csv(frame, out / filename)

    config = {
        "method": "finite-window centred complex Morlet CWT, explicit sampled kernels",
        "wavelet_B": MORLET_B, "wavelet_C": 1.0, "fs_Hz": FS, "f0_Hz": F0,
        "time_grid_start_s": float(t[0]), "time_grid_end_s": float(t[-1]),
        "n_resampled": len(t), "primary_n_samples": n, "primary_window_s": n / FS,
        "hop_s": HOP_S, "primary_threshold": PRIMARY_Z,
        "primary_persistence_s": PRIMARY_PERSISTENCE_S,
        "primary_reference_outputs": len(all_frames[2]["control"]),
        "primary_spectral_bins_Hz": estimator["frequencies"].tolist(),
        "primary_band_requested_Hz": [0.5 * F0, 8 * F0],
        "primary_band_df_Hz": estimator["df"],
        "amplitude_calibration": "discrete positive-frequency response; interference remains",
        "power_definition": "integrated L2-normalized wavelet density proxy, not Fourier PSD power",
        "reference": "all matched-control outputs; offline; control alarms are in-reference",
        "timing": "exclusive observation-window end; no backdating to centre or first crossing",
        "short_window_scope": "exploratory; read calibration errors before comparing delays",
        "stft_reference": "user report p.3, transcribed primary times; valid for report settings only",
        "scope": "one measured background, synthetic mechanical gain; not physiological clotting labels",
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_outputs(t, signals, all_frames[2], primary, case_info, estimator, calibration, out)
    print("\nPrimary CWT confirmation times (s); NaN means no confirmed alarm:")
    print(primary.pivot(index="feature", columns="case", values="first_confirmation_s").reindex(FEATURES).round(2).to_string())
    print("\nCWT vs report: positive delay difference means CWT is later:")
    print(comparison[["case", "feature", "cwt_delay_s", "stft_report_delay_s", "cwt_minus_stft_delay_s"]].round(2).to_string(index=False))
    print("\nMixture calibration, maximum absolute error (%):")
    print(calibration[calibration["test"] == "harmonic_mixture"].pivot(index="feature", columns="cycles", values="max_abs_error_pct").round(2).to_string())
    print(f"\nSaved to: {out}")
    print("One-cycle results are exploratory. Control alarms reuse calibration data.")
    return out


if __name__ == "__main__":
    main()
