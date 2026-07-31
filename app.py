from __future__ import annotations

import base64
import hashlib
import math
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    DERIVED_COLUMNS,
    FilterStep,
    FilterSettings,
    FrequencyResult,
    PhysicalSettings,
    PreprocessResult,
    ProcessedFrame,
    add_derived_columns,
    analyze_frequency,
    format_filter_formula,
    format_filter_workflow,
    parse_filter_formula,
    preprocess_experiments,
    process_axes_frame,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
BRAND_ICON_PATH = ROOT_DIR / "assets" / "vader-icon.png"
FILENAME_SUFFIX = "_PROC1_FAST_processed.csv"

REQUIRED_COLUMNS = [
    "time_from_onset_s",
    "vertical_distance_L",
    "radial_Hencky_strain",
    "vertical_strain",
    "D_over_D0",
    "force_g",
    "diameter_mm",
]

AVAILABLE_COLUMNS = REQUIRED_COLUMNS + DERIVED_COLUMNS
METADATA_COLUMNS = ["material", "velocity", "sample", "source_file"]
RAW_DEFAULT_PLOTS = [
    {"x": "time_from_onset_s", "y": "force_g"},
    {"x": "time_from_onset_s", "y": "diameter_mm"},
]
PROCESSED_DEFAULT_PLOTS = [
    {"x": "time_from_onset_s", "y": "force_g"},
    {"x": "hencky_strain", "y": "net_stress_Pa"},
]


def main() -> None:
    os.chdir(ROOT_DIR)
    st.set_page_config(
        page_title="VADER Dashboard",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    workspace = render_navigation()
    render_header(workspace)
    physical_settings = render_physical_settings()

    DATA_DIR.mkdir(exist_ok=True)
    signature = get_data_signature(DATA_DIR)
    raw_data, file_summary, issues = load_dataset(str(DATA_DIR), signature)
    for issue in issues:
        st.warning(issue)
    if raw_data.empty:
        st.info("No compatible CSV files were found in the data folder.")
        st.code(str(DATA_DIR), language="text")
        return

    if workspace == "Raw data":
        render_dashboard(
            raw_data, file_summary, signature, physical_settings,
            "raw", REQUIRED_COLUMNS, RAW_DEFAULT_PLOTS,
            allow_processing=False,
        )
        return

    preprocessed = preprocess_dataset(raw_data, physical_settings)
    warning_key = (signature, physical_settings)
    if (
        preprocessed.warnings
        and st.session_state.get("preprocess_warning_key") != warning_key
    ):
        first_warning = preprocessed.warnings[0]
        remaining = len(preprocessed.warnings) - 1
        suffix = f" (+{remaining} more)" if remaining else ""
        st.toast(f"Preprocessing: {first_warning}{suffix}", icon=":material/warning:")
        st.session_state.preprocess_warning_key = warning_key
    data = derive_dataset(preprocessed.frame, physical_settings)
    if workspace == "Filtered / processed":
        render_dashboard(
            data, file_summary, signature, physical_settings,
            "processed", AVAILABLE_COLUMNS, PROCESSED_DEFAULT_PLOTS,
            allow_processing=True,
        )
    elif workspace == "Frequency analysis":
        render_frequency_workspace(
            data, file_summary, signature, physical_settings
        )
    else:
        render_summary_workspace(data, file_summary)

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vader-ink: #172033;
            --vader-muted: #667085;
            --vader-border: #d9dee8;
            --vader-panel: #ffffff;
            --vader-accent: #df3d4f;
            --vader-rail: 4rem;
        }
        header[data-testid="stHeader"],
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none;
        }
        div[data-testid="stElementContainer"]:has(.vader-header),
        div[data-testid="stElementContainer"]:has(.st-key-nav_rail),
        div[data-testid="stElementContainer"]:has(.st-key-top_physical_settings) {
            position: absolute;
            width: 0;
            height: 0;
            margin: 0;
            padding: 0;
        }
        .block-container {
            max-width: 100%;
            padding: 3rem 0.7rem 0.7rem calc(var(--vader-rail) + 0.7rem);
        }
        .block-container > div[data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }
        .vader-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            display: flex;
            align-items: center;
            height: 3rem;
            padding: 0 11rem 0 calc(var(--vader-rail) + 0.85rem);
            color: var(--vader-ink);
            background: rgba(248, 250, 252, 0.98);
            border-bottom: 1px solid var(--vader-border);
            backdrop-filter: blur(8px);
            white-space: nowrap;
        }
        .vader-header__mark {
            width: 0.28rem;
            height: 1.25rem;
            margin-right: 0.7rem;
            background: var(--vader-accent);
            border-radius: 2px;
        }
        .vader-header__title {
            font-size: 1rem;
            font-weight: 700;
        }
        .vader-header__divider {
            width: 1px;
            height: 1.1rem;
            margin: 0 0.65rem;
            background: var(--vader-border);
        }
        .vader-header__workspace {
            color: var(--vader-muted);
            font-size: 0.82rem;
            font-weight: 500;
        }
        .st-key-nav_rail {
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 1100;
            width: var(--vader-rail);
            padding: 0.45rem;
            overflow: hidden;
            background: #ffffff;
            border-right: 1px solid var(--vader-border);
            transition: width 160ms ease, box-shadow 160ms ease;
        }
        .st-key-nav_rail:hover,
        .st-key-nav_rail:focus-within {
            width: 15.75rem;
            box-shadow: 12px 0 28px rgba(23, 32, 51, 0.13);
        }
        .st-key-nav_rail [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }
        .nav-rail__brand {
            display: flex;
            align-items: center;
            width: 14.85rem;
            height: 3rem;
            margin-bottom: 0.65rem;
            white-space: nowrap;
        }
        .nav-rail__brand-mark {
            display: inline-flex;
            flex: 0 0 3.1rem;
            align-items: center;
            justify-content: center;
            width: 3.1rem;
            height: 3rem;
            background: #ffffff;
        }
        .nav-rail__brand-label {
            margin-left: 0.65rem;
            color: var(--vader-ink);
            font-size: 0.88rem;
            font-weight: 700;
        }
        .nav-rail__brand-mark img {
            display: block;
            width: 2.7rem;
            height: 2.7rem;
            object-fit: contain;
        }
        .st-key-nav_rail button {
            justify-content: flex-start;
            width: 14.85rem;
            min-height: 3rem;
            padding: 0;
            gap: 0;
            overflow: hidden;
            border-color: transparent;
            white-space: nowrap;
        }
        .st-key-nav_rail button > div {
            justify-content: flex-start;
            width: 100%;
            overflow: hidden;
        }
        .st-key-nav_rail button > div > span {
            justify-content: flex-start;
            width: 100%;
            min-width: 0;
            gap: 0;
        }
        .st-key-nav_rail button > div > span > span:first-child {
            display: inline-flex;
            flex: 0 0 3.1rem;
            align-items: center;
            justify-content: center;
            width: 3.1rem;
            height: 3rem;
        }
        .st-key-nav_rail button > div > span > div {
            flex: 1 1 auto;
            min-width: 0;
            padding-left: 0.75rem;
        }
        .st-key-nav_rail button p {
            flex: 1 1 auto;
            margin: 0;
            padding-right: 0.8rem;
            text-align: left;
            opacity: 0;
            transition: opacity 100ms ease;
        }
        .st-key-nav_rail:hover button p,
        .st-key-nav_rail:focus-within button p {
            opacity: 1;
        }
        .st-key-nav_rail button [data-testid="stIconMaterial"] {
            display: inline-flex;
            flex: 0 0 1.5rem;
            align-items: center;
            justify-content: center;
            width: 1.5rem;
            height: 1.5rem;
            margin: 0;
            font-size: 1.45rem;
            line-height: 1;
            overflow: visible;
        }
        .st-key-top_physical_settings {
            position: fixed;
            top: 0.34rem;
            right: 0.7rem;
            z-index: 1200;
            width: 8.7rem;
        }
        .st-key-top_physical_settings button {
            min-height: 2.25rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.55rem 0.65rem 0.2rem;
            border-color: var(--vader-border);
            border-radius: 6px;
            background: var(--vader-panel);
        }
        div[data-testid="stSelectbox"] label,
        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {
            font-size: 0.76rem;
        }
        div[data-testid="stExpander"] {
            margin-top: 0.05rem;
            border: 0;
            border-top: 1px solid #edf0f5;
            border-radius: 0;
        }
        div[data-testid="stExpander"] details summary {
            padding: 0.45rem 0;
        }
        div[data-testid="stPlotlyChart"] {
            margin-top: -0.45rem;
        }
        .plot-heading {
            margin-bottom: 0.15rem;
            color: var(--vader-muted);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }
        .inline-control-label {
            color: var(--vader-muted);
            font-size: 0.74rem;
            font-weight: 600;
            white-space: nowrap;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-right: 0.45rem;
                padding-left: calc(var(--vader-rail) + 0.45rem);
            }
            .vader-header__workspace,
            .vader-header__divider {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_header(workspace: str) -> None:
    st.markdown(
        f"""
        <div class="vader-header">
            <span class="vader-header__mark"></span>
            <span class="vader-header__title">VADER Dashboard</span>
            <span class="vader-header__divider"></span>
            <span class="vader-header__workspace">{workspace}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navigation() -> str:
    workspaces = [
        ("Filtered / processed", ":material/auto_fix_high:"),
        ("Raw data", ":material/table_chart:"),
        ("Frequency analysis", ":material/graphic_eq:"),
        ("Summary plots", ":material/analytics:"),
    ]
    current = st.session_state.get("workspace", "Filtered / processed")
    icon_data = base64.b64encode(BRAND_ICON_PATH.read_bytes()).decode("ascii")
    with st.container(key="nav_rail"):
        st.markdown(
            f"""
            <div class="nav-rail__brand">
                <span class="nav-rail__brand-mark">
                    <img src="data:image/png;base64,{icon_data}" alt="VADER">
                </span>
                <span class="nav-rail__brand-label">VADER Dashboard</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, icon in workspaces:
            if st.button(
                label,
                icon=icon,
                key=f"navigate_{label}",
                type="primary" if label == current else "secondary",
                width="stretch",
                help=label,
            ):
                st.session_state.workspace = label
                st.rerun()
    return current

def render_physical_settings() -> PhysicalSettings:
    with st.container(key="top_physical_settings"):
        with st.popover(
            "Physics",
            icon=":material/tune:",
            width="stretch",
        ):
            surface_tension = st.number_input(
                "Surface tension (mN/m)", min_value=0.0, value=72.0, step=1.0,
                key="physics_surface_tension",
            )
            capillary_factor = st.number_input(
                "Capillary factor", min_value=0.0, value=2.0, step=0.1,
                key="physics_capillary_factor",
                help="Surface-tension stress = factor x gamma / diameter.",
            )
            force_zero = st.number_input(
                "Force zero (g)", value=0.0, step=0.1,
                key="physics_force_zero",
                help="force_g is treated as gram-force and converted to newtons.",
            )
            strain_source = st.selectbox(
                "Hencky strain source",
                ["Diameter / radial strain", "Vertical strain"],
                key="physics_strain_source",
            )
            minimum_rate = st.number_input(
                "Minimum |strain rate| (1/s)", min_value=1e-9, value=1e-5,
                format="%.2e", key="physics_min_rate",
                help="Viscosity is empty below this magnitude.",
            )
            st.divider()
            crop_enabled = st.toggle(
                "Crop experiments", value=True, key="physics_crop_enabled"
            )
            crop_strain_column = st.selectbox(
                "Crop strain",
                ["radial_Hencky_strain", "vertical_strain"],
                key="physics_crop_strain_column",
            )
            crop_threshold = st.number_input(
                "Crop threshold", value=7.0, key="physics_crop_threshold"
            )
            crop_min_time = st.number_input(
                "Minimum time (s)",
                min_value=0.0,
                value=0.0001,
                format="%.4f",
                key="physics_crop_min_time",
            )
            force_tail_offset = st.toggle(
                "Tail force offset",
                value=True,
                key="physics_force_tail_offset",
                help=(
                    "Subtract the mean force where crop strain is at or above "
                    "the threshold."
                ),
            )
            st.caption(
                "Stress = force / area. Net stress subtracts capillary stress. "
                "Extensional viscosity = net stress / Hencky rate."
            )
    return PhysicalSettings(
        surface_tension_mN_m=float(surface_tension),
        capillary_factor=float(capillary_factor),
        force_zero_g=float(force_zero),
        min_abs_strain_rate=float(minimum_rate),
        strain_source=strain_source,
        crop_enabled=bool(crop_enabled),
        crop_strain_column=crop_strain_column,
        crop_threshold=float(crop_threshold),
        crop_min_time_s=float(crop_min_time),
        force_tail_offset_enabled=bool(force_tail_offset),
    )

def get_data_signature(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    csv_paths = sorted(data_dir.glob("*.csv"))
    return tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in csv_paths
    )


@st.cache_data(show_spinner="Loading data")
def load_dataset(
    data_dir_text: str, signature: tuple[tuple[str, int, int], ...]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    del data_dir_text
    frames: list[pd.DataFrame] = []
    file_records: list[dict[str, Any]] = []
    issues: list[str] = []

    for path_text, _, _ in signature:
        path = Path(path_text)
        metadata = parse_filename(path)

        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - surfaced in the app
            issues.append(f"{path.name}: could not be read ({exc})")
            continue

        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            issues.append(f"{path.name}: missing columns: {', '.join(missing)}")
            continue

        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        for column in REQUIRED_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        for key, value in metadata.items():
            frame[key] = value

        frames.append(frame)
        file_records.append(
            {
                "source_file": path.name,
                "material": metadata["material"],
                "velocity": metadata["velocity"],
                "sample": metadata["sample"],
            }
        )

    data = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=REQUIRED_COLUMNS + METADATA_COLUMNS)
    )
    file_summary = pd.DataFrame.from_records(
        file_records,
        columns=["source_file", "material", "velocity", "sample"],
    )
    return data, file_summary, issues


@st.cache_data(show_spinner=False)
def preprocess_dataset(
    data: pd.DataFrame,
    settings: PhysicalSettings,
) -> PreprocessResult:
    return preprocess_experiments(data, settings)


@st.cache_data(show_spinner=False)
def derive_dataset(
    data: pd.DataFrame,
    settings: PhysicalSettings,
) -> pd.DataFrame:
    return add_derived_columns(data, settings)

def parse_filename(path: Path) -> dict[str, str]:
    name = path.name
    core = name[: -len(FILENAME_SUFFIX)] if name.endswith(FILENAME_SUFFIX) else path.stem
    parts = core.rsplit("_", 2)

    if len(parts) == 3:
        material, velocity, sample = parts
    else:
        material, velocity, sample = core, "unknown", "unknown"

    return {
        "material": material,
        "velocity": velocity,
        "sample": sample,
        "source_file": name,
    }


def render_dashboard(
    data: pd.DataFrame,
    file_summary: pd.DataFrame,
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
    scope: str,
    axis_options: list[str],
    default_plots: list[dict[str, str]],
    allow_processing: bool,
) -> None:
    plot_columns = st.columns(2, gap="small")
    for index, defaults in enumerate(default_plots):
        with plot_columns[index]:
            render_plot_window(
                index, data, file_summary, defaults, signature,
                physical_settings, scope, axis_options, allow_processing,
            )


def render_plot_window(
    index: int,
    data: pd.DataFrame,
    file_summary: pd.DataFrame,
    defaults: dict[str, str],
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
    scope: str,
    axis_options: list[str],
    allow_processing: bool,
) -> None:
    key_prefix = f"{scope}_plot_{index}"
    with st.container(border=True):
        st.markdown(
            f'<div class="plot-heading">Plot {index + 1}</div>',
            unsafe_allow_html=True,
        )
        materials = sorted(file_summary["material"].dropna().unique().tolist())
        velocities = sorted(file_summary["velocity"].dropna().unique().tolist())
        x_settings = FilterSettings()
        y_settings = FilterSettings()
        if allow_processing:
            toolbar = st.columns(
                [0.38, 1.65, 1.15, 0.38, 1.65, 1.15, 1.15, 1.15],
                gap="small",
                vertical_alignment="center",
            )
            with toolbar[0]:
                render_inline_label("X:")
            with toolbar[1]:
                x_column = st.selectbox(
                    "X axis", axis_options,
                    index=get_option_index(axis_options, defaults["x"]),
                    key=f"{key_prefix}_x", label_visibility="collapsed",
                )
            with toolbar[2]:
                x_settings = render_processing_controls(
                    f"{key_prefix}_x", x_column, "X"
                )
            with toolbar[3]:
                render_inline_label("Y:")
            with toolbar[4]:
                y_column = st.selectbox(
                    "Y axis", axis_options,
                    index=get_option_index(axis_options, defaults["y"]),
                    key=f"{key_prefix}_y", label_visibility="collapsed",
                )
            with toolbar[5]:
                y_settings = render_processing_controls(
                    f"{key_prefix}_y", y_column, "Y"
                )
            selector_scope = f"{scope}_{index}"
            with toolbar[6]:
                selected_materials = render_filter_dropdown(
                    selector_scope, "Material", materials
                )
            with toolbar[7]:
                selected_velocities = render_filter_dropdown(
                    selector_scope, "Velocity", velocities
                )
        else:
            toolbar = st.columns(
                [0.55, 2.2, 0.55, 2.2, 1.3, 1.3],
                gap="small",
                vertical_alignment="center",
            )
            with toolbar[0]:
                render_inline_label("X:")
            with toolbar[1]:
                x_column = st.selectbox(
                    "X axis", axis_options,
                    index=get_option_index(axis_options, defaults["x"]),
                    key=f"{key_prefix}_x", label_visibility="collapsed",
                )
            with toolbar[2]:
                render_inline_label("Y:")
            with toolbar[3]:
                y_column = st.selectbox(
                    "Y axis", axis_options,
                    index=get_option_index(axis_options, defaults["y"]),
                    key=f"{key_prefix}_y", label_visibility="collapsed",
                )
            selector_scope = f"{scope}_{index}"
            with toolbar[4]:
                selected_materials = render_filter_dropdown(
                    selector_scope, "Material", materials
                )
            with toolbar[5]:
                selected_velocities = render_filter_dropdown(
                    selector_scope, "Velocity", velocities
                )
        eligible_summary = file_summary[
            file_summary["material"].isin(selected_materials)
            & file_summary["velocity"].isin(selected_velocities)
        ]
        selected_files = render_file_selector(selector_scope, eligible_summary)
        plot_frame = data[data["source_file"].isin(selected_files)].copy()
        plot_frame = build_plot_frame(plot_frame, x_column, y_column)
        if plot_frame.empty:
            st.info("No data matches the current selection.")
            return

        if not allow_processing:
            render_plot_figure(plot_frame, x_column, y_column, y_column)
            return

        job_key = make_job_key(
            scope, index, signature, physical_settings, x_settings, y_settings,
            x_column, y_column, tuple(selected_files),
        )
        render_background_plot(
            job_key, plot_frame, x_column, y_column, x_settings, y_settings
        )

def render_inline_label(text: str) -> None:
    st.markdown(
        f'<div class="inline-control-label">{text}</div>',
        unsafe_allow_html=True,
    )

def render_filter_dropdown(
    index: int,
    label: str,
    values: list[str],
) -> list[str]:
    key_prefix = f"{index}_{label.lower()}"
    for value in values:
        key = f"{key_prefix}_{value}"
        if key not in st.session_state:
            st.session_state[key] = True

    selected_values = [
        value for value in values if st.session_state[f"{key_prefix}_{value}"]
    ]
    control_label = get_filter_label(label, selected_values, values)

    with st.popover(control_label, width="stretch"):
        action_columns = st.columns(2, gap="small")
        if action_columns[0].button(
            "All",
            key=f"{key_prefix}_all",
            width="stretch",
        ):
            set_filter_selection(key_prefix, values, True)
            st.rerun()
        if action_columns[1].button(
            "None",
            key=f"{key_prefix}_none",
            width="stretch",
        ):
            set_filter_selection(key_prefix, values, False)
            st.rerun()

        for value in values:
            st.checkbox(value, key=f"{key_prefix}_{value}")

    return selected_values


def get_filter_label(
    label: str,
    selected_values: list[str],
    all_values: list[str],
) -> str:
    if len(selected_values) == len(all_values):
        return f"{label}: All"
    if not selected_values:
        return f"{label}: None"
    if len(selected_values) == 1:
        return f"{label}: {selected_values[0]}"
    return f"{label}: {len(selected_values)} selected"


def set_filter_selection(
    key_prefix: str,
    values: list[str],
    selected: bool,
) -> None:
    for value in values:
        st.session_state[f"{key_prefix}_{value}"] = selected

def render_processing_controls(
    scope: str,
    source_column: str,
    axis_label: str = "Formula",
) -> FilterSettings:
    formula_key = f"{scope}_filter_formula"
    source_key = f"{scope}_filter_source"
    if formula_key not in st.session_state:
        st.session_state[formula_key] = source_column
    if st.session_state.get(source_key) != source_column:
        try:
            _, existing_steps = parse_filter_formula(st.session_state[formula_key])
        except ValueError:
            existing_steps = ()
        st.session_state[formula_key] = format_filter_formula(
            source_column, existing_steps
        )
        st.session_state[source_key] = source_column

    try:
        parsed_source, current_steps = parse_filter_formula(
            st.session_state[formula_key]
        )
        formula_valid = parsed_source == source_column
    except ValueError:
        current_steps = ()
        formula_valid = False

    if not formula_valid:
        button_label = f"{axis_label}: Invalid"
    elif current_steps:
        suffix = "filter" if len(current_steps) == 1 else "filters"
        button_label = f"{axis_label}: {len(current_steps)} {suffix}"
    else:
        button_label = f"{axis_label}: Raw"

    with st.popover(button_label, width="stretch"):
        formula = st.text_input(
            "Formula",
            key=formula_key,
            help=(
                "Excel-style nested filters. Example: LP(MA(force_g),20). "
                "The innermost value must match the selected Y signal."
            ),
        )
        try:
            parsed_source, workflow = parse_filter_formula(formula)
            formula_valid = parsed_source == source_column
        except ValueError as exc:
            st.error(str(exc))
            parsed_source, workflow = source_column, ()
            formula_valid = False
        else:
            if not formula_valid:
                st.error(f"Formula must start from {source_column}.")
            else:
                st.caption(f"Applied order: {format_filter_workflow(workflow)}")

        filter_name = st.segmented_control(
            "Add filter",
            ["MA", "LP", "HP", "SG", "WH", "Notch"],
            default="MA",
            required=True,
            key=f"{scope}_new_filter",
            width="stretch",
        )
        new_step = render_filter_step_options(scope, str(filter_name))
        actions = st.columns(3, gap="small")
        actions[0].button(
            "Add",
            icon=":material/add:",
            key=f"{scope}_add_filter",
            width="stretch",
            disabled=not formula_valid or new_step is None,
            on_click=update_filter_formula,
            args=(formula_key, source_column, "add", new_step),
        )
        actions[1].button(
            "Remove",
            icon=":material/undo:",
            key=f"{scope}_remove_filter",
            width="stretch",
            disabled=not formula_valid or not workflow,
            on_click=update_filter_formula,
            args=(formula_key, source_column, "remove", None),
        )
        actions[2].button(
            "Reset",
            icon=":material/restart_alt:",
            key=f"{scope}_reset_filter",
            width="stretch",
            disabled=formula_valid and not workflow,
            on_click=update_filter_formula,
            args=(formula_key, source_column, "reset", None),
        )

    return FilterSettings(workflow=workflow if formula_valid else ())


def render_filter_step_options(scope: str, filter_name: str) -> FilterStep | None:
    if filter_name == "MA":
        window = st.number_input(
            "Window (samples)", min_value=1, value=21, step=2,
            key=f"{scope}_ma_window",
        )
        return FilterStep("moving_average", (float(window),))
    if filter_name in {"LP", "HP"}:
        default = 20.0 if filter_name == "LP" else 0.1
        cutoff = st.number_input(
            "Cutoff (Hz)", min_value=0.001, value=default, step=0.1,
            key=f"{scope}_{filter_name.lower()}_cutoff",
        )
        operation = "lowpass" if filter_name == "LP" else "highpass"
        return FilterStep(operation, (float(cutoff),))
    if filter_name == "SG":
        columns = st.columns(2, gap="small")
        window = columns[0].number_input(
            "Window", min_value=3, value=21, step=2,
            key=f"{scope}_sg_window",
        )
        order = columns[1].number_input(
            "Order", min_value=1, value=3, step=1,
            key=f"{scope}_sg_order",
        )
        if int(order) >= int(window):
            st.error("SG order must be smaller than its window.")
            return None
        return FilterStep("savgol", (float(window), float(order)))
    if filter_name == "WH":
        smoothing_lambda = st.number_input(
            "Lambda", min_value=0.0, value=1000.0, step=100.0,
            key=f"{scope}_wh_lambda",
        )
        return FilterStep("whittaker", (float(smoothing_lambda),))

    range_columns = st.columns(2, gap="small")
    minimum = range_columns[0].number_input(
        "Minimum (Hz)", min_value=0.0, value=0.5, step=0.1,
        key=f"{scope}_notch_min",
    )
    maximum = range_columns[1].number_input(
        "Maximum (Hz)", min_value=0.001, value=10.0, step=0.5,
        key=f"{scope}_notch_max",
    )
    detail_columns = st.columns(2, gap="small")
    count = detail_columns[0].number_input(
        "Peaks", min_value=1, max_value=10, value=3, step=1,
        key=f"{scope}_notch_count",
    )
    quality = detail_columns[1].number_input(
        "Q", min_value=1.0, value=30.0, step=1.0,
        key=f"{scope}_notch_q",
    )
    if float(maximum) <= float(minimum):
        st.error("Notch maximum must be greater than its minimum.")
        return None
    return FilterStep(
        "notch",
        (float(minimum), float(maximum), float(count), float(quality)),
    )


def update_filter_formula(
    formula_key: str,
    source_column: str,
    action: str,
    step: FilterStep | None,
) -> None:
    try:
        parsed_source, steps = parse_filter_formula(
            str(st.session_state.get(formula_key, source_column))
        )
    except ValueError:
        parsed_source, steps = source_column, ()
    if parsed_source != source_column:
        parsed_source, steps = source_column, ()
    if action == "add" and step is not None:
        steps = (*steps, step)
    elif action == "remove" and steps:
        steps = steps[:-1]
    elif action == "reset":
        steps = ()
    st.session_state[formula_key] = format_filter_formula(parsed_source, steps)

def render_file_selector(index: int, file_summary: pd.DataFrame) -> list[str]:
    files = file_summary["source_file"].tolist()
    for file_name in files:
        key = f"plot_{index}_file_{file_name}"
        if key not in st.session_state:
            st.session_state[key] = True

    selected_count = sum(
        bool(st.session_state[f"plot_{index}_file_{file_name}"])
        for file_name in files
    )

    with st.expander(f"Data series  {selected_count}/{len(files)}", expanded=False):
        action_columns = st.columns([1, 1, 4], gap="small")
        if action_columns[0].button("All", key=f"plot_{index}_all", width="stretch"):
            set_file_selection(index, files, True)
            st.rerun()
        if action_columns[1].button("None", key=f"plot_{index}_none", width="stretch"):
            set_file_selection(index, files, False)
            st.rerun()

        checkbox_columns = st.columns(3, gap="small")
        selected_files: list[str] = []
        for file_index, row in file_summary.reset_index(drop=True).iterrows():
            file_name = row["source_file"]
            label = get_file_label(row)
            with checkbox_columns[file_index % len(checkbox_columns)]:
                if st.checkbox(
                    label,
                    key=f"plot_{index}_file_{file_name}",
                    help=file_name,
                ):
                    selected_files.append(file_name)

    return selected_files


def set_file_selection(index: int | str, files: list[str], selected: bool) -> None:
    for file_name in files:
        st.session_state[f"plot_{index}_file_{file_name}"] = selected


def get_option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def get_file_label(file_record: pd.Series) -> str:
    return (
        f"{file_record['material']} | "
        f"{file_record['velocity']} | "
        f"{file_record['sample']}"
    )


def build_plot_frame(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    columns = ["time_from_onset_s", x_column, y_column, *METADATA_COLUMNS]
    frame = data.loc[:, list(dict.fromkeys(columns))]
    frame = frame.dropna(subset=["time_from_onset_s", x_column, y_column])
    return frame.sort_values(["source_file", "time_from_onset_s"]).copy()


def downsample_by_trace(plot_frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in plot_frame.groupby("source_file", sort=False, dropna=False):
        step = max(1, math.ceil(len(group) / max_points))
        pieces.append(group.iloc[::step])
    return pd.concat(pieces, ignore_index=True) if pieces else plot_frame


def render_background_plot(
    job_key: str,
    plot_frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_settings: FilterSettings,
    y_settings: FilterSettings,
) -> None:
    if not x_settings.active and not y_settings.active:
        render_plot_figure(plot_frame, x_column, y_column, y_column)
        return

    future = submit_background_job(
        job_key,
        process_axes_frame,
        plot_frame,
        x_column,
        x_settings,
        y_column,
        y_settings,
    )
    if future.done():
        try:
            result: ProcessedFrame = future.result()
        except Exception as exc:  # pragma: no cover
            st.error(f"Processing failed: {exc}")
            render_plot_figure(plot_frame, x_column, y_column, y_column)
            return
        plot_x_column = (
            f"{x_column}__x_processed" if x_settings.active else x_column
        )
        plot_y_column = (
            f"{y_column}__y_processed" if y_settings.active else y_column
        )
        render_plot_figure(
            result.frame,
            plot_x_column,
            plot_y_column,
            y_column,
            x_label=x_column,
        )
        render_processing_notes(result)
        return

    @st.fragment(run_every=0.75)
    def pending_plot() -> None:
        if future.done():
            st.rerun()
        st.caption("Processing in background...")
        render_plot_figure(plot_frame, x_column, y_column, y_column)

    pending_plot()


def render_plot_figure(
    frame: pd.DataFrame,
    x_column: str,
    plot_y_column: str,
    y_label: str,
    x_label: str | None = None,
) -> None:
    figure = make_figure(
        downsample_by_trace(frame, 5_000),
        x_column,
        plot_y_column,
        y_label,
        x_label=x_label,
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())

def render_processing_notes(result: ProcessedFrame) -> None:
    selected_peaks = sorted({
        round(peak, 4)
        for peaks in result.notch_peaks_hz.values()
        for peak in peaks
    })
    if selected_peaks:
        text = ", ".join(f"{peak:g}" for peak in selected_peaks[:10])
        suffix = " ..." if len(selected_peaks) > 10 else ""
        st.caption(f"Auto-notch peaks (Hz): {text}{suffix}")
    if result.warnings:
        with st.expander(f"Processing warnings ({len(result.warnings)})"):
            for warning in result.warnings:
                st.caption(warning)


def make_figure(
    plot_frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    y_label: str,
    x_label: str | None = None,
) -> Any:
    figure = px.line(
        plot_frame, x=x_column, y=y_column, color="source_file",
        line_group="source_file",
        hover_data={
            "material": True, "velocity": True, "sample": True,
            "source_file": False,
        },
        labels={
            x_column: x_label or x_column,
            y_column: y_label,
            "source_file": "data series",
        },
    )
    figure.update_layout(
        height=650, margin={"l": 8, "r": 8, "t": 15, "b": 8},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=False,
        hovermode="closest", font={"color": "#344054", "size": 11},
    )
    style_axes(figure)
    figure.update_traces(line={"width": 1.6})
    return figure


def style_axes(figure: go.Figure) -> None:
    figure.update_xaxes(
        showgrid=True, gridcolor="#e9edf3", zeroline=False, title_standoff=10
    )
    figure.update_yaxes(
        showgrid=True, gridcolor="#e9edf3", zeroline=False, title_standoff=10
    )


def plotly_config() -> dict[str, Any]:
    return {
        "displaylogo": False, "responsive": True, "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


@st.cache_resource
def get_background_runtime() -> tuple[
    ThreadPoolExecutor, dict[str, Future[Any]], threading.Lock
]:
    return ThreadPoolExecutor(max_workers=3), {}, threading.Lock()


def submit_background_job(
    job_key: str,
    function: Callable[..., Any],
    *args: Any,
) -> Future[Any]:
    executor, jobs, lock = get_background_runtime()
    with lock:
        future = jobs.get(job_key)
        if future is None:
            future = executor.submit(function, *args)
            jobs[job_key] = future
        if len(jobs) > 64:
            completed = [
                key for key, item in jobs.items() if item.done() and key != job_key
            ]
            for key in completed[: len(jobs) - 48]:
                jobs.pop(key, None)
    return future


def make_job_key(*parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()

def render_summary_workspace(
    data: pd.DataFrame,
    file_summary: pd.DataFrame,
) -> None:
    materials = sorted(file_summary["material"].dropna().unique().tolist())
    velocities = sorted(file_summary["velocity"].dropna().unique().tolist())
    filter_columns = st.columns(2, gap="small")
    with filter_columns[0]:
        selected_materials = render_filter_dropdown(
            "summary", "Material", materials
        )
    with filter_columns[1]:
        selected_velocities = render_filter_dropdown(
            "summary", "Velocity", velocities
        )

    control_columns = st.columns([2, 2, 1.5, 1], gap="small")
    with control_columns[0]:
        x_column = st.selectbox(
            "X axis", AVAILABLE_COLUMNS,
            index=get_option_index(AVAILABLE_COLUMNS, "hencky_strain"),
            key="summary_x",
        )
    with control_columns[1]:
        y_column = st.selectbox(
            "Y axis", AVAILABLE_COLUMNS,
            index=get_option_index(AVAILABLE_COLUMNS, "net_stress_Pa"),
            key="summary_y",
        )
    with control_columns[2]:
        group_by = st.selectbox(
            "Group by",
            ["Material", "Velocity", "Material + velocity"],
            key="summary_group_by",
        )
    with control_columns[3]:
        with st.popover("Binning", width="stretch"):
            bin_count = st.number_input(
                "Number of X bins", min_value=10, max_value=500,
                value=100, step=10, key="summary_bins",
            )

    filtered = data[
        data["material"].isin(selected_materials)
        & data["velocity"].isin(selected_velocities)
    ].copy()
    summary, peaks = build_summary_tables(
        filtered, x_column, y_column, group_by, int(bin_count)
    )
    if summary.empty:
        st.info("No numeric data matches the summary selection.")
        return

    mean_figure = make_summary_mean_figure(
        summary, x_column, y_column
    )
    peak_figure = px.box(
        peaks, x="group", y="peak", color="group", points="all",
        labels={"group": group_by, "peak": f"Peak {y_column}"},
    )
    peak_figure.update_layout(
        height=520, margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=False,
        font={"color": "#344054", "size": 11},
    )
    style_axes(peak_figure)

    plot_columns = st.columns(2, gap="small")
    with plot_columns[0]:
        st.markdown(
            '<div class="plot-heading">Mean and standard deviation</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(mean_figure, width="stretch", config=plotly_config())
    with plot_columns[1]:
        st.markdown(
            '<div class="plot-heading">Peak by experiment</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(peak_figure, width="stretch", config=plotly_config())


def build_summary_tables(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_by: str,
    bin_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [x_column, y_column, "material", "velocity", "source_file"]
    frame = data.loc[:, list(dict.fromkeys(columns))].dropna(
        subset=[x_column, y_column]
    ).copy()
    if frame.empty or frame[x_column].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    if group_by == "Material":
        frame["group"] = frame["material"].astype(str)
    elif group_by == "Velocity":
        frame["group"] = frame["velocity"].astype(str)
    else:
        frame["group"] = (
            frame["material"].astype(str) + " | " + frame["velocity"].astype(str)
        )

    frame["x_bin"] = pd.cut(
        frame[x_column], bins=bin_count, duplicates="drop"
    )
    summary = (
        frame.groupby(["group", "x_bin"], observed=True)
        .agg(
            x_mean=(x_column, "mean"),
            y_mean=(y_column, "mean"),
            y_std=(y_column, "std"),
            count=(y_column, "size"),
        )
        .reset_index()
    )
    peaks = (
        frame.groupby(["group", "source_file"], observed=True)[y_column]
        .max()
        .rename("peak")
        .reset_index()
    )
    return summary, peaks


def make_summary_mean_figure(
    summary: pd.DataFrame,
    x_label: str,
    y_label: str,
) -> go.Figure:
    figure = go.Figure()
    palette = px.colors.qualitative.Safe
    for index, (group_name, group) in enumerate(summary.groupby("group", sort=True)):
        ordered = group.sort_values("x_mean")
        color = palette[index % len(palette)]
        lower = ordered["y_mean"] - ordered["y_std"].fillna(0.0)
        upper = ordered["y_mean"] + ordered["y_std"].fillna(0.0)
        figure.add_trace(go.Scatter(
            x=ordered["x_mean"], y=lower, mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=ordered["x_mean"], y=upper, mode="lines",
            line={"width": 0}, fill="tonexty", fillcolor=color,
            opacity=0.16, showlegend=False, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=ordered["x_mean"], y=ordered["y_mean"], mode="lines",
            line={"color": color, "width": 2}, name=str(group_name),
        ))
    figure.update_layout(
        height=520, margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        hovermode="closest", font={"color": "#344054", "size": 11},
        legend={"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 9}},
    )
    figure.update_xaxes(title=x_label)
    figure.update_yaxes(title=y_label)
    style_axes(figure)
    return figure

def render_frequency_workspace(
    data: pd.DataFrame,
    file_summary: pd.DataFrame,
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
) -> None:
    materials = sorted(file_summary["material"].dropna().unique().tolist())
    velocities = sorted(file_summary["velocity"].dropna().unique().tolist())
    filter_columns = st.columns(2, gap="small")
    with filter_columns[0]:
        selected_materials = render_filter_dropdown(
            "frequency", "Material", materials
        )
    with filter_columns[1]:
        selected_velocities = render_filter_dropdown(
            "frequency", "Velocity", velocities
        )
    eligible = file_summary[
        file_summary["material"].isin(selected_materials)
        & file_summary["velocity"].isin(selected_velocities)
    ]
    if eligible.empty:
        st.info("No data matches the current frequency filters.")
        return

    controls = st.columns([2.2, 2.2, 1.5, 1.2], gap="small")
    file_options = eligible["source_file"].tolist()
    labels = {
        row["source_file"]: get_file_label(row) for _, row in eligible.iterrows()
    }
    with controls[0]:
        selected_file = st.selectbox(
            "Data series", file_options, format_func=lambda value: labels[value],
            key="frequency_file",
        )
    with controls[1]:
        signal_column = st.selectbox(
            "Signal", AVAILABLE_COLUMNS,
            index=get_option_index(AVAILABLE_COLUMNS, "force_g"),
            key="frequency_signal",
        )
    with controls[2]:
        filter_settings = render_processing_controls("frequency", signal_column)
    with controls[3]:
        peak_settings = render_peak_settings()

    selected_frame = data[data["source_file"] == selected_file].copy()
    job_key = make_job_key(
        "frequency", signature, physical_settings, filter_settings,
        selected_file, signal_column, peak_settings,
    )
    render_background_frequency(
        job_key, selected_frame, signal_column, filter_settings, peak_settings
    )


def render_peak_settings() -> tuple[float, float, int, int]:
    with st.popover("Peak search", width="stretch"):
        peak_min = st.number_input(
            "Minimum frequency (Hz)", min_value=0.0, value=0.1, step=0.1,
            key="frequency_peak_min",
        )
        peak_max = st.number_input(
            "Maximum frequency (Hz)", min_value=0.001, value=10.0, step=0.5,
            key="frequency_peak_max",
        )
        peak_count = st.number_input(
            "Maximum peaks", min_value=1, max_value=50, value=8, step=1,
            key="frequency_peak_count",
        )
        energy_bins = st.number_input(
            "Energy bins", min_value=4, max_value=100, value=20, step=1,
            key="frequency_energy_bins",
        )
    return (
        float(peak_min), max(float(peak_min), float(peak_max)),
        int(peak_count), int(energy_bins),
    )


def render_background_frequency(
    job_key: str,
    frame: pd.DataFrame,
    signal_column: str,
    settings: FilterSettings,
    peak_settings: tuple[float, float, int, int],
) -> None:
    future = submit_background_job(
        job_key, analyze_frequency, frame, signal_column, settings, *peak_settings
    )
    if future.done():
        try:
            result: FrequencyResult = future.result()
        except Exception as exc:  # pragma: no cover
            st.error(f"Frequency analysis failed: {exc}")
            return
        render_frequency_result(result, signal_column)
        return

    @st.fragment(run_every=0.75)
    def pending_frequency() -> None:
        if future.done():
            st.rerun()
        st.info("Frequency analysis is running in the background...")

    pending_frequency()

def render_frequency_result(result: FrequencyResult, signal_column: str) -> None:
    st.caption(
        f"Sampling rate: {result.sample_rate_hz:.3g} Hz | "
        f"Nyquist: {result.sample_rate_hz / 2.0:.3g} Hz"
    )
    if result.notch_peaks_hz:
        selected = ", ".join(f"{value:.4g}" for value in result.notch_peaks_hz)
        st.caption(f"Auto-notch selected (Hz): {selected}")

    fft_figure = go.Figure()
    fft_figure.add_trace(go.Scatter(
        x=result.frequency_hz, y=result.fft_amplitude, mode="lines",
        line={"color": "#2563eb", "width": 1.5}, name="FFT amplitude",
    ))
    if result.peaks_hz.size:
        fft_figure.add_trace(go.Scatter(
            x=result.peaks_hz, y=result.peak_amplitudes, mode="markers",
            marker={"color": "#df3d4f", "size": 7}, name="Peaks",
        ))
    configure_analysis_figure(
        fft_figure, "Frequency (Hz)", f"FFT amplitude ({signal_column})", 365
    )

    psd_figure = go.Figure(go.Scatter(
        x=result.psd_frequency_hz, y=result.psd, mode="lines",
        line={"color": "#07847b", "width": 1.5},
    ))
    configure_analysis_figure(
        psd_figure, "Frequency (Hz)", f"PSD ({signal_column})^2/Hz", 365
    )
    psd_figure.update_yaxes(type="log")

    chart_columns = st.columns(2, gap="small")
    with chart_columns[0]:
        st.plotly_chart(fft_figure, width="stretch", config=plotly_config())
    with chart_columns[1]:
        st.plotly_chart(psd_figure, width="stretch", config=plotly_config())

    centers = (result.energy_left_hz + result.energy_right_hz) / 2.0
    widths = result.energy_right_hz - result.energy_left_hz
    energy_figure = go.Figure(go.Bar(
        x=centers, y=result.energy, width=widths * 0.92,
        marker={"color": "#d39b22"},
        customdata=list(zip(result.energy_left_hz, result.energy_right_hz)),
        hovertemplate=(
            "%{customdata[0]:.3g}-%{customdata[1]:.3g} Hz"
            "<br>Energy: %{y:.4g}<extra></extra>"
        ),
    ))
    configure_analysis_figure(
        energy_figure, "Frequency band (Hz)", "Integrated PSD energy", 310
    )

    lower_columns = st.columns([2, 1], gap="small")
    with lower_columns[0]:
        st.plotly_chart(energy_figure, width="stretch", config=plotly_config())
    with lower_columns[1]:
        st.markdown(
            '<div class="plot-heading">Detected peaks</div>',
            unsafe_allow_html=True,
        )
        peak_table = pd.DataFrame({
            "frequency_Hz": result.peaks_hz,
            "amplitude": result.peak_amplitudes,
        })
        st.dataframe(peak_table, width="stretch", hide_index=True, height=276)

    if result.warnings:
        with st.expander(f"Frequency warnings ({len(result.warnings)})"):
            for warning in result.warnings:
                st.caption(warning)


def configure_analysis_figure(
    figure: go.Figure,
    x_title: str,
    y_title: str,
    height: int,
) -> None:
    figure.update_layout(
        height=height, margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=False,
        hovermode="closest", font={"color": "#344054", "size": 11},
    )
    figure.update_xaxes(title=x_title)
    figure.update_yaxes(title=y_title)
    style_axes(figure)

if __name__ == "__main__":
    main()