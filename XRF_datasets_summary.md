# Datasets Summary — `real_datasets/xrf`

This table summarises every dataset available in the `real_datasets/xrf` folder.  
**Spectral Variables** are identified as all columns whose header is a numeric value, with the column values monotonically increasing from left to right across the dataset.  
All datasets are binary classification tasks with classes **A** and **B**.

| Dataset | Samples | Class A | Class B | Spectral Variables | Spectral Range |
|---|---|---|---|---|---|
| bank_notes | 407 | 251 | 156 | 985 | 1.00 – 26.07 |
| ecigar | 36 | 23 | 13 | 3800 | 1.00 – 20.01 |
| forage | 195 | 58 | 137 | 1450 | 1.00 – 30.00 |
| milk | 383 | 143 | 240 | 900 | 1.00 – 24.00 |
| soil | 212 | 110 | 102 | 701 | 1.00 – 15.00 |
| soil_types | 717 | 354 | 363 | 701 | 1.00 – 15.00 |
| soil_types_unir | 261 | 120 | 141 | 619 | 1351.54 – 2150.47 |
| soil_vnir | 394 | 336 | 58 | 1050 | 400.00 – 2498.00 |
| soyben | 93 | 52 | 41 | 2048 | 0.00 – 40.98 |
| soyben15 | 91 | 51 | 40 | 2048 | 0.00 – 40.89 |
| sweet_pepper | 32 | 17 | 15 | 2047 | 0.00 – 40.92 |
| tomato | 52 | 20 | 32 | 2047 | 0.00 – 40.92 |

## Notes

- **Spectral Range**: the first and last spectral wavelength/wavenumber values (column names) present in each dataset.
- `soil_types_unir` and `soil_vnir` use different spectral measurement ranges (NIR/VNIR), whereas the remaining datasets share the XRF range starting near 1 keV or 0 keV.
- `soyben` and `soyben15` are variants of the same soybean dataset with slightly different spectral coverages; `sweet_pepper` and `tomato` share an identical spectral range.
- The `ecigar` dataset has the highest number of spectral variables (3,800) despite a small sample size (36).
