# Illustrative LAOS benchmark (synthetic, not a measured clot experiment)

Four independent 30-min strain-controlled, 1-Hz, 10% strain records are sampled uniformly at 100 Hz (100 points/cycle): three injected changes and a matched no-change control. These are simulated scenarios, not donor measurements. The 10% strain is within the 0.01–1 LAOS amplitude range studied by van Kempen et al.; it is not a continuous LAOS test during actual clot formation. The published experiment monitored formation at small strain, then ran LAOS on the matured clot. A continuous 10% LAOS protocol during clot growth may alter the clot. Use this package as a signal-processing benchmark, not as a validated physiology model.

## Model

Let s=sin(2πt), c=cos(2πt), γ=0.1s, γdot=0.2πc, q=0 until prescribed onset and q=1−exp(−((t−onset)/300)^2) thereafter. Let Gp=0.0021502273569187027 + A q Pa, Gpp=0.06202740369600379 + 0.15 A q Pa. A is 210, 182, 238 Pa for A/B/C and zero for control. The base Gp/Gpp were interpolated from APR18 SS SAOS at 1 Hz; they are linear-regime numbers and their transfer to LAOS is a scenario assumption. The A reference is motivated by published whole-blood low-deformation G' plateau 210±29 Pa, not a measured LAOS modulus. For a=0.08+0.57q and b=0.008+0.192q, ideal stress is σ=0.1[ Gp(s−a s³+b s⁵)+Gpp c ] Pa. The cubic and quintic terms provide odd harmonics; their coefficients and evolution are illustrative assumptions, not fitted to APR18 or published clot harmonics. Colored AR(1) measurement noise is added with rho 0.8, fixed seeds, SD=0.00008+0.002*0.1*hypot(Gp,Gpp) Pa (also assumed). No t-ESSTV parameter fitting data are included.

## Files and use

`observable/*.csv` contains time_s, strain, shear_rate_1_s, stress_Pa. The same observable columns can go into FFT/STFT/CWT/DWT. `truth/*_truth.csv` records the injected state and coefficients. `metadata.json` records seeds, full parameters and independent FFT sanity-check results. Keep truth separate while choosing detection features. Controls and clot runs are scenarios from the same synthetic construction, not independent donors; do not split adjacent windows randomly across training and test. The known onset is the start of an inserted mechanical change, not a clinical gel point.

For FFT, use cycle-aligned windows, e.g. 10 s at 100 Hz; compare 1, 3, 5 Hz amplitudes and I3/I1. For STFT use a window containing multiple 1-Hz cycles and examine the same bands over time. A single FFT over all 30 min cannot localize onset. In the included basic check, a threshold fixed using the complete control run (max control I3/I1 + 0.01) with three consecutive 10-s windows tests detectable change. This threshold check is only a sanity check and uses a complete control case for calibration; separate train/test cases are needed for any generalization claim.

This is deliberately phenomenological. It validates whether your signal-analysis implementation recovers an injected increase in odd harmonics and stiffness under stated noise. It does not validate human clot detection, match a specific donor, or predict whether APR18 LAOS actually contains clotting.

References: van Kempen et al. 2016, DOI 10.1007/s10237-015-0686-9; Windberger et al. 2020, DOI 10.3390/molecules25173890; Ewoldt et al. 2008, DOI 10.1122/1.2970095.
