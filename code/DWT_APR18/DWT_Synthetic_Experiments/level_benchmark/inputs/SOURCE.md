# Real input provenance

`APR18_BH_raw.csv` is the unchanged original-point export from the prior experiment.
Source workbook: APR18 SS SAOS STEPS LAOS UDLAOS.xlsx, steps!BH4:BK68, 0.1 to 1/s.
`original_APR18_A5_features.csv` is a frozen implementation-equivalence reference,
not synthetic test data and not ground-truth event labels.

Synthetic data are fully reproducible from 01_protocol/cases.json and signal_for().
No synthetic onset is inferred from real APR18 stress.
