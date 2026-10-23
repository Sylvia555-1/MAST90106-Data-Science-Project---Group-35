# Conventional Rheology Analysis — SAOS and LAOS

This package contains two Jupyter notebooks for conventional oscillatory rheology analysis. **Use the SAOS notebook for precomputed modulus sweeps in the Excel `SS SAOS` layout; use the LAOS notebook for raw, time-resolved stress–strain CSV files.** Both use matching plot styles and save figures as PNG and PDF.

## Notebooks and outputs

| Notebook | Analysis | Output folder |
| --- | --- | --- |
| `SAOS_APR18_Analysis.ipynb` | Reads an Excel `SS SAOS` sheet; analyses frequency sweeps, loss tangent, exploratory Winter–Chambon consistency, complex modulus/viscosity, and amplitude-sweep harmonics. Also supports an **optional, separate** time-resolved SAOS CSV. The filename says APR18, but another workbook with the **same sheet and column layout** can be selected. | `results/saos_apr18/` |
| `Conventional_Gel_Point_Detection.ipynb` | Performs cycle-wise FFT on raw oscillatory CSVs, estimates first-harmonic storage/loss moduli and higher-harmonic ratios, and identifies persistent LAOS modulus crossovers. Uses synthetic truth files **only for evaluation after detection**. Also retains a separate legacy raw-signal SAOS mode; this is **not** the Excel `SS SAOS` reader. | `results/conventional_gel_point/<dataset>/` |

The SAOS notebook exports figures for moduli versus frequency, loss tangent, power-law diagnostics, complex modulus and viscosity, and amplitude-dependent harmonic response. The LAOS notebook exports full-range and, when an event is available, zoomed crossover plots. Figures use consistent colours and are saved in **250-DPI PNG** and **vector PDF** formats.

## Included data

All paths below are relative to `Conventional_Analysis_Updated/`, the inner folder of the extracted ZIP.

| File or folder | Purpose |
| --- | --- |
| `data/APR18 SS SAOS STEPS LAOS UDLAOS.xlsx` | Example Excel workbook; `SS SAOS` is used by the SAOS notebook. No conversion to raw stress/strain is required. |
| `laos_synthetic/observable/` | Time-resolved `clot_A.csv`, `clot_B.csv`, `clot_C.csv`, and `control.csv` for the LAOS notebook. |
| `laos_synthetic/truth/` | Matching synthetic reference files for post-detection comparison only. |
| `APR18_converted/observable/` | Six converted LAOS examples from the APR18 workbook, stored in the required raw-signal CSV layout. |
| `results/` | Example results already included in the archive. Re-running notebooks may overwrite files of the same names. |

## SAOS Excel input format

The SAOS notebook reads the worksheet named **`SS SAOS`**, using column positions (not column headings):

| Excel columns | Expected values | Requirement |
| --- | --- | --- |
| **E–G** | Angular frequency in rad/s, storage modulus $G'$ in Pa, loss modulus $G''$ in Pa | Required |
| **I–N** | Strain amplitude (%), decimal strain, first-harmonic $G'_1$ and $G''_1$, third-harmonic $G'_3$ and $G''_3$ | Required by the current notebook |
| **A–C** | Steady-shear measurements in the example workbook | Present in the example, but **not processed** by this notebook |

The first worksheet row is treated as a header and skipped. Non-numeric entries are ignored. The notebook requires usable positive values for the frequency sweep and a non-empty amplitude sweep; it **does not currently skip the amplitude analysis** when I–N are absent. For another dataset, retain the `SS SAOS` sheet name and column positions or edit the loading function.

**Do not feed this precomputed SAOS Excel format into `Conventional_Gel_Point_Detection.ipynb`**: its raw-signal loader expects different columns and data.

## Requirements and usage

Python 3.10+ is recommended. The included `requirements.txt` lists `numpy`, `pandas`, `matplotlib`, `scipy`, `openpyxl`, `ipykernel`, `notebook`, and `nbformat`.

1. Extract the ZIP and open the **inner** `Conventional_Analysis_Updated/` folder in VS Code or Jupyter. Keep the supplied subfolders together.
2. Open a terminal in that folder and install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. For **SAOS**, open `SAOS_APR18_Analysis.ipynb` and run all cells. By default it finds the included workbook in `data/`. To analyse another file with the same layout, change the configuration cell:

   ```python
   APR18_PATH = Path(r"C:\path\to\your_saos_workbook.xlsx")
   ```

   The worksheet must still be named `SS SAOS`. The notebook's output directory and CSV filenames currently include `apr18` / `APR18` **even when a different workbook is loaded**; adjust `OUT_DIR` and the CSV filename strings if you want results named after your sample.

4. For **LAOS**, open `Conventional_Gel_Point_Detection.ipynb` and configure its first code cell:

   ```python
   DATA_ROOT = Path("laos_synthetic")    # or Path("APR18_converted")
   SAMPLE_NAMES = None                   # or ["clot_A"]
   DEFAULT_MODE = "LAOS"
   ```

   Each file in `<DATA_ROOT>/observable/` must contain `time_s`, `strain` (dimensionless), `shear_rate_1_s`, and `stress_Pa`. Any matching `<DATA_ROOT>/truth/` files are optional and are used for comparison only. Run all cells after changing the data root.

## Optional time-resolved SAOS experiment

For **repeated multi-frequency SAOS measurements over coagulation time**, set `SAOS_TIME_CSV` in the SAOS notebook's configuration cell to a CSV containing these columns:

| Column | Meaning |
| --- | --- |
| `time_s` | Measurement/coagulation time in seconds |
| `omega_rad_s` | Angular frequency (rad/s) |
| `G_prime_Pa` | Storage modulus (Pa) |
| `G_double_prime_Pa` | Loss modulus (Pa) |
| `sweep_id` | Optional: group measurements that belong to the same approximately simultaneous frequency sweep |

At least **three distinct frequencies per sweep** are needed for the notebook's diagnostic calculation. When supplied, it writes `SAOS_Winter_Chambon_time_diagnostics.csv` and `Fig_06_time_resolved_WC_diagnostics` to the SAOS result folder. The consistency flags rely on **heuristic thresholds**, not a validated gel-point determination; measurements in a sweep should represent approximately the same material age.

## Interpreting results

- **SAOS frequency sweeps** describe the measured viscoelastic state. Frequency-dependent $\tan\delta=G''/G'$ and fitted power-law exponents can assess **consistency with** Winter–Chambon behaviour, but a single sweep **cannot give clotting time**.
- **SAOS amplitude sweeps** may extend beyond the linear regime. First- and third-harmonic components and $I_3/I_1$ are useful nonlinear diagnostics; do not treat the entire amplitude sweep as linear SAOS.
- **LAOS first-harmonic crossovers** ($G'_1=G''_1$) are **operational crossover estimates**, not automatically critical gel points or the injected onset of clot formation. `observable_detection_s` identifies the estimated crossover; `observable_confirmation_s` denotes the later time when the required persistent cycles have been observed.
- A synthetic truth file must not be used to determine the observable detection time. Ground-truth onset and ideal crossover columns are for **evaluation**, and real APR18 data do not include those synthetic ground-truth annotations.

The included results are examples. Review the plots and per-sample CSV summaries before drawing conclusions from a different experiment.
