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

## Standalone versions

Both standalone files contain the complete application, analysis code, styles,
and embedded VADER icon:

- `vader_dashboard_v1.py` is the stable checkpoint restored from commit `73f1df7`.
- `vader_dashboard_v2.py` is the current version with custom formulas available
  in Data, Frequency analysis, and Postprocessing.

Place either file beside the `data` folder and run the selected version:

```powershell
streamlit run vader_dashboard_v2.py
```

Both versions require the packages listed in `requirements.txt` to be installed.

## One-file architecture

The standalone file is divided by two large markers:

    PHYSICS, RHEOLOGY, AND SIGNAL PROCESSING (EXPERT EDIT ZONE)
    STREAMLIT FRONTEND / GUI (APPLICATION LAYER)

Scientific contributors work only above the frontend marker. The Streamlit code
below it reads shared registries, so the application still runs from one Python
file without duplicating scientific definitions in the GUI.

The main extension points are:

- `PhysicalSettings` and `PHYSICS_CONTROL_SPECS`: add a physical parameter and
  its generated Physics control.
- `CUSTOM_DERIVED_QUANTITIES` and `add_custom_derived_columns()`: register and
  calculate a new constitutive or derived quantity. Its axis label, unit, and
  log-axis default then become available to the plots automatically.
- `CUSTOM_FILTER_CONTROL_SPECS` and `CUSTOM_FILTER_EXECUTORS`: register a new
  smoother/filter, its formula aliases and parameters, and its numerical
  implementation. The formula parser and Add filter panel use the registry
  automatically.

Built-in equations, preprocessing, smoothing, filtering, FFT, PSD, and energy
analysis also remain together in that expert zone. GUI request state, layouts,
widgets, Plotly rendering, styling, and navigation remain below the frontend
marker.

## Interface

- A fixed icon rail follows the workflow: Data, Frequency analysis, Postprocessing, and Summary.
- Hovering or focusing the rail expands it as an overlay, so the plot area never resizes.
- Physical settings remain available from the fixed top-right toolbar without taking plot space.
- Data contains measured and derived variables with independent background-processing controls for both plots.
- Postprocessing is reserved for the upcoming feature-extraction workflow.
- Plot View menus control raw overlays, legends, and independent linear/log axis scales.
- Velocity is available as a numeric X or Y variable and velocity choices sort numerically.
- Plot labels use scientific symbols, subscripts, superscripts, and bracketed units.
- Data-series panels stay open while selecting multiple runs.
- Applied plot, filter, and frequency settings are restored when returning to a workspace.
- Frequency analysis contains FFT, Welch PSD, energy histograms, optional peak markers, and compact per-plot controls.
- Summary contains binned mean curves with standard-deviation bands and per-experiment peak distributions.
- The IFF-friendly palette uses `#0075CF` as its primary accent and avoids red plot colors.
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

## Custom formulas

The function button beside a Data Y selector, the Frequency variable, or the
Postprocessing Y selector opens the same formula editor. The formula remains
fully editable by hand. Variable and symbol insertion controls can append any
token repeatedly, while Wrap applies a selected function to the current formula.
Numeric constants are typed directly. Available short names are:

```text
t, Lv, HS, eps_z, D, D0, F, A, sig, sig_surf,
delta_sig, HS_rate, visc_e, v, pi, ST
```

`F` is also accepted anywhere `force_g` is accepted in processing formulas.
`D0` is reconstructed as `D / (D/D0)`, and `ST` uses the Physics-panel surface
tension value in mN/m. Supported functions are `abs`, `sqrt`, `log`, `log10`,
`exp`, `sin`, `cos`, `minimum`, and `maximum`. Expressions are evaluated by a
restricted parser; Python attributes, imports, indexing, and arbitrary function
calls are rejected.

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
hencky_strain_rate_1_s = d(radial_Hencky_strain) / dt
extensional_viscosity_Pa_s = net_stress_Pa / hencky_strain_rate_1_s
```

The default capillary factor is `2 / pi`. The force column is assumed to contain
gram-force values.