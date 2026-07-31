# VADER Dashboard

A small Streamlit dashboard for CSV files produced from rheology experiments.

## Data layout

Place CSV files in `data`, next to `app.py`.

Expected filename format:

```text
material_name_velocity_sample_PROC1_FAST_processed.csv
```

The material name may contain underscores. The app parses filenames from the right, so this works:

```text
xanthan_gel_batch_A_5mms_S01_PROC1_FAST_processed.csv
```

Expected columns:

```text
time_from_onset_s
vertical_distance_L
radial_Hencky_strain
vertical_strain
D_over_D0
force_g
diameter_mm
```

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\generate_synthetic_data.py
streamlit run app.py
```

The app sets its working directory to the folder containing `app.py`, then reads `data/*.csv`.

## Standalone file

`vader_dashboard.py` contains the complete application, analysis code, styles,
and embedded VADER icon. Place it beside the `data` folder and run:

```powershell
streamlit run vader_dashboard.py
```

It still requires the packages listed in `requirements.txt` to be installed.

## Interface

- A fixed icon rail switches between Filtered / processed, Raw data, Frequency analysis, and Summary plots.
- Hovering or focusing the rail expands it as an overlay, so the plot area never resizes.
- Physical settings remain available from the fixed top-right toolbar without taking plot space.
- Raw data contains the original CSV variables and no processing controls.
- Filtered / processed contains derived variables and independent background-processing controls for both plots.
- Frequency analysis contains FFT, Welch PSD, energy histograms, detected peaks, and optional preprocessing.
- Summary plots contains binned mean curves with standard-deviation bands and per-experiment peak distributions.
- CSV files in `data` are ignored by Git so confidential work data is not accidentally committed.

## Processing formulas

Each processed plot has independent X and Y processing panels. Both build
Excel-style nested formulas:

```text
force_g
MA(force_g)
LP(MA(force_g),20)
SG(LP(MA(force_g),20),31,3)
```

Select a filter, set its parameters, and press Add. The new filter wraps the
existing formula. Remove peels off the outer filter and Reset returns to the
selected signal. `LP` and `HP` use Hz; `MA` and `SG` windows use samples.
## Experiment preprocessing

The Physics menu also controls the preprocessing applied before derived
quantities, filtering, frequency analysis, and summaries. Raw data is never
changed.

For each source file, the crop keeps rows where:

```text
time > minimum_time
time < max(time where crop_strain <= crop_threshold)
```

The optional tail-force correction subtracts the mean `force_g` measured where
`crop_strain >= crop_threshold`. If no tail samples exist, the correction is
skipped and the app reports a nonfatal warning. Sampling frequency is estimated
from the median positive time step after sorting and removing duplicate times.

## Derived quantities

The `Physical settings` menu controls the surface tension, capillary factor,
force zero, strain source, and minimum strain-rate magnitude.

The default calculations are:

```text
force_N = (force_g - force_zero_g) * 9.80665e-3
area_m2 = pi * (diameter_mm * 1e-3 / 2)^2
stress_Pa = force_N / area_m2
surface_tension_stress_Pa = capillary_factor * surface_tension_N_m / diameter_m
net_stress_Pa = stress_Pa - surface_tension_stress_Pa
hencky_strain = -2 * radial_Hencky_strain
hencky_strain_rate_1_s = d(hencky_strain) / dt
extensional_viscosity_Pa_s = net_stress_Pa / hencky_strain_rate_1_s
```

The default capillary factor is `2`. The force column is assumed to contain
gram-force values.