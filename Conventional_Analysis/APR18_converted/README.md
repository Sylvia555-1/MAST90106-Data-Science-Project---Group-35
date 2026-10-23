# APR18 LAOS — Notebook-Compatible CSVs

## Files

The six CSV files in `observable/` are converted from the populated runs in the `LAOS` worksheet of `APR18 SS SAOS STEPS LAOS UDLAOS.xlsx`.

| File pattern | Measured experiment |
| --- | --- |
| `APR18_LAOS_w1_gamma0p5.csv` | Angular frequency 1 rad/s; nominal strain amplitude 0.5 (50%) |
| `APR18_LAOS_w1_gamma1.csv` | Angular frequency 1 rad/s; nominal strain amplitude 1 (100%) |
| `APR18_LAOS_w1_gamma5.csv` | Angular frequency 1 rad/s; nominal strain amplitude 5 (500%) |
| `APR18_LAOS_w1_gamma10.csv` | Angular frequency 1 rad/s; nominal strain amplitude 10 (1,000%) |
| `APR18_LAOS_w1_gamma50.csv` | Angular frequency 1 rad/s; nominal strain amplitude 50 (5,000%) |
| `APR18_LAOS_w1_gamma100.csv` | Angular frequency 1 rad/s; nominal strain amplitude 100 (10,000%) |

The `omega=5` headings in the uploaded workbook have no stress/strain/time observations and are not exported. `conversion_summary.csv` documents the exports.

## Format

Each CSV contains the four columns required by `Conventional_Gel_Point_Detection.ipynb`:

| Column | Meaning |
| --- | --- |
| `time_s` | Original measurement time, seconds |
| `strain` | Original strain percent divided by 100 (dimensionless) |
| `shear_rate_1_s` | **Estimated** shear rate: numerical derivative of strain with respect to time, 1/s; not an instrument-measured signal |
| `stress_Pa` | Original recorded shear stress, pascals |

The converter removes the **first recorded sample from each run** by default to avoid initialisation transients; use `--keep-first-point` to preserve it. Each experiment is kept separate; they do **not** represent one continuous clotting timeline.

## Using the notebook

Extract the ZIP, then edit the first configuration cell in `Conventional_Gel_Point_Detection.ipynb`:

```python
DATA_ROOT = Path(r"C:\path\to\APR18_converted")
SAMPLE_NAMES = None
DEFAULT_MODE = "LAOS"
# Recommended: save APR18 output separately from synthetic results
OUT_DIR = DATA_ROOT / "results" / "conventional_gel_point"
```

`DATA_ROOT` must be the folder containing `observable/`, **not** the `observable/` folder itself. No `truth/` folder is needed. The notebook's FFT cycle calculation and LAOS plots will work, but APR18 has no known coagulation onset or ground-truth gel point. Do not interpret crossovers between different strain conditions as temporal gelation.

## Re-running the converter

Place `convert_apr18_laos.py` next to the original Excel workbook, and run:

```bash
python convert_apr18_laos.py
```

Or specify paths:

```bash
python convert_apr18_laos.py --excel "APR18 SS SAOS STEPS LAOS UDLAOS.xlsx" --output-dir "APR18_converted"
```

Requires Python 3 and NumPy. This export already contains the converted CSVs, so running the script is optional.
