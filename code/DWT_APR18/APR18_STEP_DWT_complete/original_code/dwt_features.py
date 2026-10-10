"""Small, readable DWT helpers shared by real and synthetic comparisons."""
import numpy as np
import pandas as pd
import pywt


def reconstruct(values, levels, approximation_levels, mode='symmetric'):
    """Keep physical units and reconstruct requested components on the input grid."""
    values = np.asarray(values, dtype=float)
    largest = max([*levels, *approximation_levels])
    assert largest <= pywt.dwt_max_level(len(values), 'db2')
    coefficients = pywt.wavedec(values, 'db2', mode=mode, level=largest)
    restored = pywt.waverec(coefficients, 'db2', mode=mode)[:len(values)]
    error = float(np.max(np.abs(restored - values)))
    assert error < 1e-11 * max(1, np.max(np.abs(values)))
    components = {}
    for level in levels:
        kept = [np.zeros_like(c) for c in coefficients]
        kept[-level] = coefficients[-level]
        components[f'D{level}'] = pywt.waverec(kept, 'db2', mode=mode)[:len(values)]
    for level in approximation_levels:
        # A_j contains A_largest and all details coarser than j.
        kept = [c.copy() for c in coefficients]
        for j in range(1, level + 1):
            kept[-j] = np.zeros_like(kept[-j])
        components[f'A{level}'] = pywt.waverec(kept, 'db2', mode=mode)[:len(values)]
    return components, error


def windows(time, signal, components, width, start, stop, raw_time):
    """Summarize absolute-time windows and retain their original observation support."""
    dt = float(np.median(np.diff(time)))
    assert np.allclose(np.diff(time), dt, atol=1e-11)
    rows = []
    for left in np.arange(start, stop - width / 2, width):
        right = left + width
        if left < time[0] or right > time[-1] + 1e-10:
            continue
        lo, hi = np.searchsorted(time, [left, right], side='left')
        if hi <= lo:
            continue
        crossing = (raw_time[:-1] < right) & (raw_time[1:] > left)
        gap = float(np.diff(raw_time)[crossing].max()) if crossing.any() else np.nan
        row = dict(left_s=left, right_s=right, mid_s=(left + right) / 2,
                   grid_points=hi-lo,
                   original_points=int(((raw_time >= left) & (raw_time < right)).sum()),
                   max_original_gap_s=gap,
                   grid_mean=float(np.mean(signal[lo:hi])))
        for name, values in components.items():
            feature = f'R{name[1:]}' if name[0] == 'D' else name
            row[feature] = float(np.sqrt(np.mean(values[lo:hi]**2))) if name[0] == 'D' else float(np.mean(values[lo:hi]))
            scale = (2 ** int(name[1:])) * dt
            # Six scale spacings is a conservative engineering screen, not a
            # measurement uncertainty or an exact filter support theorem.
            row[f'{feature}_interior'] = bool(left >= time[0] + 6.25 * scale and right <= time[-1] - 6.25 * scale)
            row[f'{feature}_gap_exceeds_scale'] = bool(gap > scale)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for name in ['grid_mean', *[n for n in components if n[0] == 'A']]:
        frame[f'{name}_rate'] = frame[name].diff() / frame.mid_s.diff()
        if name != 'grid_mean':
            frame[f'{name}_rate_interior'] = frame[f'{name}_interior'] & frame[f'{name}_interior'].shift(1, fill_value=False)
    return frame


def linear_window_means(raw_time, values, frame):
    """Integrate the piecewise-linear curve exactly over each complete window."""
    means = []
    for row in frame.itertuples():
        knots = np.r_[row.left_s, raw_time[(raw_time > row.left_s) & (raw_time < row.right_s)], row.right_s]
        means.append(np.trapz(np.interp(knots, raw_time, values), knots) / (row.right_s-row.left_s))
    return np.asarray(means)


def compare(reference, other, names, name_mapping=None):
    """Compare the same physical windows, after screening both boundaries."""
    joined = reference.merge(other, on=['left_s', 'right_s', 'mid_s'], suffixes=('_ref', '_other'))
    result = []
    for name in names:
        alternative = (name_mapping or {}).get(name, name)
        left = f'{name}_ref' if name in other else name
        right = f'{alternative}_other' if alternative in reference else alternative
        mask_name = f'{name}_interior' if '_rate' not in name else f'{name}_interior'
        alt_mask = f'{alternative}_interior'
        a_mask = mask_name + '_ref' if mask_name in other else mask_name
        b_mask = alt_mask + '_other' if alt_mask in reference else alt_mask
        mask = joined[a_mask].to_numpy(bool) & joined[b_mask].to_numpy(bool)
        a = joined.loc[mask, left].to_numpy()
        b = joined.loc[mask, right].to_numpy()
        result.append(dict(feature=name, alternative_feature=alternative, windows=len(a),
                           maximum_absolute_difference=float(np.max(np.abs(a-b))) if len(a) else None,
                           relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)) if len(a) else None))
    return result
