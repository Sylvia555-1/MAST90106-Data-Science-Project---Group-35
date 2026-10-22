"""Check scientific identities and feature behavior using known toy signals.

These checks validate implementation and interpretation, not detection accuracy.
Run this script after creating common.py and config.json.
"""
import numpy as np
import pywt

from common import ROOT, decompose, extract_features, load_config, write_json


def main():
    config = load_config()
    fs, level = config["sampling_hz"], config["level"]
    time = np.arange(int(300 * fs)) / fs
    rng = np.random.default_rng(741)
    clean = (0.2 + 0.001 * time) * np.sin(2 * np.pi * time)
    clean += 0.03 * np.cos(8 * np.pi * time)
    # A correlated component makes omission of the cross term detectable.
    noise = 0.3 * clean + 0.02 * rng.standard_normal(len(time))
    observed = clean + noise
    bands = decompose(observed, config)
    clean_bands, noise_bands = decompose(clean, config), decompose(noise, config)
    features = extract_features(time, observed, bands, config)
    checks = []

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    # Complete coefficients and reconstructed components must recover the input.
    coefficients = pywt.wavedec(observed, config["wavelet"],
                               mode=config["mode"], level=level)
    inverse = pywt.waverec(coefficients, config["wavelet"],
                          mode=config["mode"])[:len(time)]
    inverse_error = float(np.max(np.abs(inverse - observed)))
    sum_error = float(np.max(np.abs(sum(bands.values()) - observed)))
    check("perfect_reconstruction", max(inverse_error, sum_error) < 1e-12,
          inverse_max_error=inverse_error, component_sum_max_error=sum_error)
    linear_error = max(float(np.max(np.abs(bands[key] - clean_bands[key]
                                          - noise_bands[key]))) for key in bands)
    check("component_linearity", linear_error < 1e-12, max_error=linear_error)

    # RMS scales with amplitude; relative activity P6 does not.
    scaled = extract_features(time, 3 * observed, decompose(3 * observed, config), config)
    rms_columns = [f"R{j}" for j in range(1, level + 1)] + ["H13", "raw_rms"]
    scale_error = float(np.max(np.abs(scaled[rms_columns].to_numpy()
                                     - 3 * features[rms_columns].to_numpy())))
    share_error = float(np.max(np.abs(scaled.P6 - features.P6)))
    check("amplitude_scaling_and_share_invariance", scale_error < 1e-12
          and share_error < 1e-12, rms_max_error=scale_error, P6_max_error=share_error)
    expected_speed = features.R6.diff() / features.mid_s.diff()
    valid = features.V6.notna()
    check("growth_rate_uses_physical_seconds", valid.any()
          and np.allclose(features.loc[valid, "V6"], expected_speed[valid], atol=1e-14),
          compared_windows=int(valid.sum()))

    # Directly check the local energy identity; energies are not additive alone.
    energy_error, omitted_cross_error = 0.0, 0.0
    for row in features.itertuples():
        selected = (time >= row.start_s) & (time < row.end_s)
        for j in range(1, level + 1):
            y, s, n = [item[f"D{j}"][selected]
                       for item in (bands, clean_bands, noise_bands)]
            ey, es, en = [np.mean(item ** 2) for item in (y, s, n)]
            cross = 2 * np.mean(s * n)
            energy_error = max(energy_error, abs(ey - es - en - cross))
            omitted_cross_error = max(omitted_cross_error, abs(ey - es - en))
    check("local_energy_includes_cross_term", energy_error < 1e-12
          and omitted_cross_error > 1e-5, max_identity_error=float(energy_error),
          max_error_without_cross_term=float(omitted_cross_error))

    zeros = np.zeros_like(time)
    zero_features = extract_features(time, zeros, decompose(zeros, config), config)
    check("zero_signal_has_undefined_share", zero_features.P6.isna().all()
          and (zero_features.R6 == 0).all())
    check("first_growth_rate_is_undefined", np.isnan(features.V6.iloc[0]))
    # No event and no noise: a 1 Hz carrier still produces fine-scale details.
    carrier = decompose(np.sin(2 * np.pi * time), config)
    margin = int(config["boundary_s"] * fs)
    carrier_rms = {key: float(np.sqrt(np.mean(value[margin:-margin] ** 2)))
                   for key, value in carrier.items() if key.startswith("D")}
    check("pure_carrier_has_multiscale_details", all(v > 1e-5 for v in carrier_rms.values()),
          rms_by_detail=carrier_rms,
          implication="Nonzero fine-scale activity is not sufficient evidence of noise.")

    # Crop already computed components: common absolute windows must be unchanged.
    # This checks indexing without adding DWT shift sensitivity to the comparison.
    offset = 7
    cropped = extract_features(time[offset:], observed[offset:],
                               {key: value[offset:] for key, value in bands.items()}, config)
    common = features.merge(cropped, on=["start_s", "end_s"], suffixes=("_full", "_crop"))
    columns = rms_columns + ["P6"] + [f"E{j}" for j in range(1, level + 1)]
    crop_error = max(float(np.max(np.abs(common[f"{key}_full"] - common[f"{key}_crop"])))
                     for key in columns)
    check("cropping_preserves_absolute_window_features", len(common) > 0 and crop_error < 1e-12,
          common_windows=len(common), max_error=crop_error)
    irregular = time.copy()
    irregular[400] += 0.002
    try:
        extract_features(irregular, observed, bands, config)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check("nonuniform_time_is_rejected", rejected)

    # Analysis and synthesis each have a finite cascaded support span.
    # Their sum is a conservative distance bound, not a minimal exclusion width.
    wavelet = pywt.Wavelet(config["wavelet"])
    support_samples = (wavelet.dec_len + wavelet.rec_len - 2) * (2 ** level - 1)
    check("configured_boundary_exceeds_support_bound", margin >= support_samples,
          support_bound_samples=support_samples, support_bound_s=support_samples / fs,
          configured_margin_s=config["boundary_s"])
    # Compare equal-phase extension modes. Periodization has a different convention
    # and is deliberately excluded; wrapping would also change the physical record.
    random_signal = rng.standard_normal(len(time))
    reference = decompose(random_signal, config)
    extension_errors = {}
    for mode in ("zero", "reflect"):
        variant = decompose(random_signal, {**config, "mode": mode})
        extension_errors[mode] = max(float(np.max(np.abs(reference[key][margin:-margin]
                                        - variant[key][margin:-margin]))) for key in reference)
    check("extension_changes_do_not_reach_interior", max(extension_errors.values()) < 1e-12,
          interior_max_error_by_extension=extension_errors)

    result = {"scope": "Implementation and interpretation checks; no detector evaluated",
              "all_passed": all(item["passed"] for item in checks),
              "check_count": len(checks), "checks": checks}
    destination = ROOT / "results" / "01_validation" / "self_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    print(f"Scientific self-checks: {sum(item['passed'] for item in checks)}/{len(checks)} passed")
    if not result["all_passed"]:
        raise AssertionError(f"Read failed checks in {destination}")


if __name__ == "__main__":
    main()
