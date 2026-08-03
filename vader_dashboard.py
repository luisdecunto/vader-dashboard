"""Standalone VADER Dashboard.

Place this file beside the data folder and run either:
    python vader_dashboard.py
or:
    streamlit run vader_dashboard.py
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    iirnotch,
    savgol_filter,
    sosfiltfilt,
    welch,
)
from scipy.sparse.linalg import spsolve


# =============================================================================
# PHYSICS, RHEOLOGY, AND SIGNAL PROCESSING (EXPERT EDIT ZONE)
# =============================================================================
# Everything from here to the STREAMLIT FRONTEND / GUI marker belongs to the
# scientific layer. Physics contributors should only need to work in this zone.
#
# Extension points:
#   1. Physical inputs: PhysicalSettings + PHYSICS_CONTROL_SPECS.
#   2. Derived/constitutive quantities: CUSTOM_DERIVED_QUANTITIES +
#      add_custom_derived_columns().
#   3. Filters/smoothers: CUSTOM_FILTER_CONTROL_SPECS +
#      CUSTOM_FILTER_EXECUTORS.

MEASURED_PROCESS_COLUMNS = ("force_g", "radial_Hencky_strain")
VARIANT_COLUMN = "processing_variant"
TRACE_COLUMN = "trace_id"
VELOCITY_COLUMN = "velocity_mm_s"


@dataclass(frozen=True)
class PhysicalSettings:
    surface_tension_mN_m: float = 72.0
    capillary_factor: float = 2.0 / np.pi
    force_zero_g: float = 0.0
    min_abs_strain_rate: float = 1e-5
    crop_enabled: bool = True
    crop_strain_column: str = "radial_Hencky_strain"
    crop_threshold: float = 7.0
    crop_min_time_s: float = 0.0001
    force_tail_offset_enabled: bool = True


@dataclass(frozen=True)
class PhysicsControlSpec:
    field_name: str
    label: str
    control: str
    key: str
    group: str
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    number_format: str | None = None
    help: str | None = None
    options: tuple[str, ...] = ()
    format_options_as_columns: bool = False


# Add a PhysicalSettings field and one matching item here. The Physics popover
# will create the control and pass its value into the scientific calculations.
PHYSICS_CONTROL_SPECS = (
    PhysicsControlSpec(
        "surface_tension_mN_m",
        "Surface tension (mN/m)",
        "number",
        "physics_surface_tension",
        "Physics",
        min_value=0.0,
        step=1.0,
    ),
    PhysicsControlSpec(
        "capillary_factor",
        "Capillary factor",
        "number",
        "physics_capillary_factor",
        "Physics",
        min_value=0.0,
        step=0.1,
        help="Surface-tension stress = factor x gamma / diameter.",
    ),
    PhysicsControlSpec(
        "force_zero_g",
        "Force zero (g)",
        "number",
        "physics_force_zero",
        "Physics",
        step=0.1,
        help="force_g is treated as gram-force and converted to newtons.",
    ),
    PhysicsControlSpec(
        "min_abs_strain_rate",
        "Minimum |strain rate| (1/s)",
        "number",
        "physics_min_rate",
        "Physics",
        min_value=1e-9,
        number_format="%.2e",
        help="Viscosity is empty below this magnitude.",
    ),
    PhysicsControlSpec(
        "crop_enabled",
        "Crop experiments",
        "toggle",
        "physics_crop_enabled",
        "Preprocessing",
    ),
    PhysicsControlSpec(
        "crop_strain_column",
        "Crop strain",
        "select",
        "physics_crop_strain_column",
        "Preprocessing",
        options=("radial_Hencky_strain", "vertical_strain"),
        format_options_as_columns=True,
    ),
    PhysicsControlSpec(
        "crop_threshold",
        "Crop threshold",
        "number",
        "physics_crop_threshold",
        "Preprocessing",
    ),
    PhysicsControlSpec(
        "crop_min_time_s",
        "Minimum time (s)",
        "number",
        "physics_crop_min_time",
        "Preprocessing",
        min_value=0.0,
        number_format="%.4f",
    ),
    PhysicsControlSpec(
        "force_tail_offset_enabled",
        "Tail force offset",
        "toggle",
        "physics_force_tail_offset",
        "Preprocessing",
        help=(
            "Subtract the mean force where crop strain is at or above "
            "the threshold."
        ),
    ),
)
PHYSICS_PANEL_CAPTION = (
    "Stress = force / area. Net stress subtracts capillary stress. "
    "Extensional viscosity = net stress / HS strain rate."
)


@dataclass(frozen=True)
class DerivedQuantityDefinition:
    column: str
    display_label: str
    axis_symbol: str
    unit: str
    log_y: bool = False
    menu_label: str | None = None


BUILTIN_DERIVED_QUANTITIES = (
    DerivedQuantityDefinition(
        "area_mm2", "Area", "<i>A</i>", "mm<sup>2</sup>", menu_label="A"
    ),
    DerivedQuantityDefinition(
        "stress_Pa", "Stress", "\u03c3", "Pa", menu_label="\u03c3"
    ),
    DerivedQuantityDefinition(
        "surface_tension_stress_Pa",
        "Surface-tension stress",
        "\u03c3<sub>surf</sub>",
        "Pa",
        menu_label="\u03c3_surf",
    ),
    DerivedQuantityDefinition(
        "net_stress_Pa",
        "Net stress",
        "\u0394\u03c3",
        "Pa",
        menu_label="\u0394\u03c3",
    ),
    DerivedQuantityDefinition(
        "hencky_strain_rate_1_s",
        "HS strain rate",
        "\u03b5\u0307<sub>r</sub>",
        "s<sup>-1</sup>",
        menu_label="\u03b5\u0307\u1d63",
    ),
    DerivedQuantityDefinition(
        "extensional_viscosity_Pa_s",
        "Extensional viscosity",
        "\u03b7<sub>e</sub>",
        "Pa s",
        menu_label="\u03b7\u2091",
        log_y=True,
    ),
)

# Add metadata for colleague-defined constitutive quantities here, then compute
# the matching columns in add_custom_derived_columns() below.
CUSTOM_DERIVED_QUANTITIES: tuple[DerivedQuantityDefinition, ...] = ()
DERIVED_QUANTITY_DEFINITIONS = (
    *BUILTIN_DERIVED_QUANTITIES,
    *CUSTOM_DERIVED_QUANTITIES,
)
DERIVED_COLUMNS = [
    definition.column for definition in DERIVED_QUANTITY_DEFINITIONS
]
LOG_Y_COLUMNS = {
    definition.column
    for definition in DERIVED_QUANTITY_DEFINITIONS
    if definition.log_y
}

COLUMN_DISPLAY_LABELS = {
    "time_from_onset_s": "Time from onset",
    "vertical_distance_L": "Vertical distance",
    "radial_Hencky_strain": "HS strain",
    "vertical_strain": "ε_{z}",
    "D_over_D0": "D / D0",
    "force_g": "Force",
    "diameter_mm": "Diameter",
    VELOCITY_COLUMN: "Velocity",
}
COLUMN_AXIS_SYMBOLS = {
    "time_from_onset_s": "<i>t</i>",
    "vertical_distance_L": "<i>L</i><sub>v</sub>",
    "radial_Hencky_strain": "\u03b5<sub>r</sub>",
    "vertical_strain": "\u03b5<sub>z</sub>",
    "D_over_D0": "<i>D</i>/<i>D</i><sub>0</sub>",
    "force_g": "<i>F</i>",
    "diameter_mm": "<i>D</i>",
    VELOCITY_COLUMN: "<i>v</i>",
}
# Selectbox options are plain text, so Unicode mirrors the plot's math notation.
COLUMN_MENU_LABELS = {
    "time_from_onset_s": "t",
    "vertical_distance_L": "L\u1d65",
    "radial_Hencky_strain": "\u03b5\u1d63",
    "vertical_strain": "\u03b5_{z}",
    "D_over_D0": "D/D\u2080",
    "force_g": "F",
    "diameter_mm": "D",
    VELOCITY_COLUMN: "v",
}
COLUMN_UNITS = {
    "time_from_onset_s": "s",
    "vertical_distance_L": "-",
    "radial_Hencky_strain": "-",
    "vertical_strain": "-",
    "D_over_D0": "-",
    "force_g": "g",
    "diameter_mm": "mm",
    VELOCITY_COLUMN: "mm s<sup>-1</sup>",
}
for _definition in DERIVED_QUANTITY_DEFINITIONS:
    COLUMN_DISPLAY_LABELS[_definition.column] = _definition.display_label
    COLUMN_AXIS_SYMBOLS[_definition.column] = _definition.axis_symbol
    COLUMN_UNITS[_definition.column] = _definition.unit
    COLUMN_MENU_LABELS[_definition.column] = (
        _definition.menu_label or _definition.display_label
    )


def add_custom_derived_columns(
    data: pd.DataFrame,
    settings: PhysicalSettings,
) -> pd.DataFrame:
    """Compute columns declared in CUSTOM_DERIVED_QUANTITIES.

    Keep custom constitutive equations here. Return the same frame with the new
    columns added. The default implementation intentionally changes nothing.
    """
    del settings
    return data

@dataclass(frozen=True)
class FilterStep:
    operation: str
    parameters: tuple[float, ...] = ()

@dataclass(frozen=True)
class FilterSettings:
    lowpass_hz: float | None = None
    highpass_hz: float | None = None
    notch_enabled: bool = False
    notch_min_hz: float = 0.5
    notch_max_hz: float = 20.0
    notch_peak_count: int = 3
    notch_q: float = 30.0
    smoothing: str = "None"
    savgol_window: int = 21
    savgol_order: int = 3
    whittaker_lambda: float = 1_000.0
    workflow: tuple[FilterStep, ...] = ()

    @property
    def active(self) -> bool:
        return any(
            (
                self.lowpass_hz is not None,
                self.highpass_hz is not None,
                self.notch_enabled,
                self.smoothing != "None",
                bool(self.workflow),
            )
        )

@dataclass(frozen=True)
class FormulaVariant:
    expression: str
    label: str
    settings: FilterSettings


@dataclass(frozen=True)
class FilterParameterSpec:
    name: str
    label: str
    key_suffix: str
    default: float
    min_value: float | None = None
    max_value: float | None = None
    step: float = 1.0
    integer: bool = False


@dataclass(frozen=True)
class FilterControlSpec:
    ui_label: str
    operation: str
    formula_name: str
    aliases: tuple[str, ...]
    parameters: tuple[FilterParameterSpec, ...]
    minimum_parameters: int = 0
    omit_default_parameters: bool = True
    workflow_style: str = "call"


BUILTIN_FILTER_CONTROL_SPECS = (
    FilterControlSpec(
        "MA",
        "moving_average",
        "MA",
        ("MA", "MOVINGAVERAGE"),
        (
            FilterParameterSpec(
                "window_samples",
                "Window (samples)",
                "ma_window",
                21.0,
                min_value=1.0,
                step=2.0,
                integer=True,
            ),
        ),
        workflow_style="compact",
    ),
    FilterControlSpec(
        "LP",
        "lowpass",
        "LP",
        ("LP", "LOWPASS"),
        (
            FilterParameterSpec(
                "cutoff_hz",
                "Cutoff (Hz)",
                "lp_cutoff",
                20.0,
                min_value=0.001,
                step=0.1,
            ),
        ),
        minimum_parameters=1,
        omit_default_parameters=False,
        workflow_style="compact",
    ),
    FilterControlSpec(
        "HP",
        "highpass",
        "HP",
        ("HP", "HIGHPASS"),
        (
            FilterParameterSpec(
                "cutoff_hz",
                "Cutoff (Hz)",
                "hp_cutoff",
                0.1,
                min_value=0.001,
                step=0.1,
            ),
        ),
        minimum_parameters=1,
        omit_default_parameters=False,
        workflow_style="compact",
    ),
    FilterControlSpec(
        "SG",
        "savgol",
        "SG",
        ("SG", "SAVGOL"),
        (
            FilterParameterSpec(
                "window_samples",
                "Window",
                "sg_window",
                21.0,
                min_value=3.0,
                step=2.0,
                integer=True,
            ),
            FilterParameterSpec(
                "polynomial_order",
                "Order",
                "sg_order",
                3.0,
                min_value=1.0,
                step=1.0,
                integer=True,
            ),
        ),
    ),
    FilterControlSpec(
        "WH",
        "whittaker",
        "WH",
        ("WH", "WHITTAKER"),
        (
            FilterParameterSpec(
                "smoothing_lambda",
                "Lambda",
                "wh_lambda",
                1_000.0,
                min_value=0.0,
                step=100.0,
            ),
        ),
        workflow_style="compact",
    ),
    FilterControlSpec(
        "Notch",
        "notch",
        "NOTCH",
        ("NT", "NOTCH"),
        (
            FilterParameterSpec(
                "minimum_hz",
                "Minimum (Hz)",
                "notch_min",
                0.5,
                min_value=0.0,
                step=0.1,
            ),
            FilterParameterSpec(
                "maximum_hz",
                "Maximum (Hz)",
                "notch_max",
                10.0,
                min_value=0.001,
                step=0.5,
            ),
            FilterParameterSpec(
                "peak_count",
                "Peaks",
                "notch_count",
                3.0,
                min_value=1.0,
                max_value=10.0,
                step=1.0,
                integer=True,
            ),
            FilterParameterSpec(
                "quality_factor",
                "Q",
                "notch_q",
                30.0,
                min_value=1.0,
                step=1.0,
            ),
        ),
    ),
)

# Add colleague-defined filter metadata here. The formula parser and the
# Streamlit Add filter panel both use this registry.
CUSTOM_FILTER_CONTROL_SPECS: tuple[FilterControlSpec, ...] = ()
FILTER_CONTROL_SPECS = (
    *BUILTIN_FILTER_CONTROL_SPECS,
    *CUSTOM_FILTER_CONTROL_SPECS,
)
FILTER_SPEC_BY_LABEL = {
    specification.ui_label: specification
    for specification in FILTER_CONTROL_SPECS
}
FILTER_SPEC_BY_OPERATION = {
    specification.operation: specification
    for specification in FILTER_CONTROL_SPECS
}
FILTER_SPEC_BY_ALIAS = {
    alias.upper(): specification
    for specification in FILTER_CONTROL_SPECS
    for alias in specification.aliases
}

CustomFilterExecutor = Callable[
    [np.ndarray, tuple[float, ...], float, float, list[str]],
    np.ndarray,
]
# Register custom numerical implementations by operation name. Built-in
# implementations remain in _apply_filter_steps() below.
CUSTOM_FILTER_EXECUTORS: dict[str, CustomFilterExecutor] = {}

def column_display_label(column: str) -> str:
    if is_custom_expression_column(column):
        return custom_expression_text(column)
    return COLUMN_DISPLAY_LABELS.get(column, column)


def column_menu_label(column: str) -> str:
    if is_custom_expression_column(column):
        return f"ƒ {custom_expression_text(column)}"
    return COLUMN_MENU_LABELS.get(column, column)


def normalize_axis_column(column: object) -> str:
    return (
        "radial_Hencky_strain"
        if str(column) == "hencky_strain"
        else str(column)
    )


def formula_source_label(source_column: str) -> str:
    return "HS" if source_column == "radial_Hencky_strain" else source_column


def canonical_formula_source(source_name: str) -> str:
    aliases = {
        "HS": "radial_Hencky_strain",
        "HD": "radial_Hencky_strain",
        "F": "force_g",
    }
    return aliases.get(source_name.upper(), source_name)


def parse_filter_workflow(expression: str) -> tuple[FilterStep, ...]:
    text = re.sub(r"\s+", "", expression).upper()
    if text in {"", "RAW", "NONE"}:
        return ()
    return tuple(_parse_workflow_sequence(text))


def format_filter_workflow(steps: tuple[FilterStep, ...]) -> str:
    labels: list[str] = []
    for step in steps:
        specification = FILTER_SPEC_BY_OPERATION.get(step.operation)
        formula_name = (
            specification.formula_name
            if specification is not None
            else step.operation.upper()
        )
        parameters = ",".join(f"{value:g}" for value in step.parameters)
        if (
            specification is not None
            and specification.workflow_style == "compact"
            and len(step.parameters) == 1
        ):
            labels.append(f"{formula_name}{parameters}")
        else:
            labels.append(f"{formula_name}({parameters})")
    return " > ".join(labels) if labels else "Raw"

def parse_filter_formula(expression: str) -> tuple[str, tuple[FilterStep, ...]]:
    text = re.sub(r"\s+", "", expression)
    if not text:
        raise ValueError("Formula cannot be empty.")
    return _parse_filter_formula_node(text)


def format_filter_formula(
    source_column: str,
    steps: tuple[FilterStep, ...],
) -> str:
    formula = source_column
    for step in steps:
        specification = FILTER_SPEC_BY_OPERATION.get(step.operation)
        formula_name = (
            specification.formula_name
            if specification is not None
            else step.operation.upper()
        )
        defaults = (
            tuple(parameter.default for parameter in specification.parameters)
            if specification is not None
            else ()
        )
        omit_parameters = (
            specification is not None
            and specification.omit_default_parameters
            and step.parameters == defaults
        )
        suffix = ""
        if not omit_parameters:
            suffix = "," + ",".join(
                f"{value:g}" for value in step.parameters
            )
        formula = f"{formula_name}({formula}{suffix})"
    return formula


def parse_filter_formula_list(
    expression_text: str,
    source_column: str,
) -> tuple[FormulaVariant, ...]:
    formulas = split_formula_list(expression_text)
    if not formulas:
        formulas = [formula_source_label(source_column)]

    variants: list[FormulaVariant] = []
    seen: set[str] = set()
    for expression in formulas:
        expanded = expand_filter_formula_shorthand(expression, source_column)
        parsed_source, steps = parse_filter_formula(expanded)
        if parsed_source != source_column:
            raise ValueError(f"Formula must start from {column_display_label(source_column)}.")
        normalized = format_filter_formula(formula_source_label(source_column), steps)
        if normalized in seen:
            continue
        seen.add(normalized)
        label = "Raw" if not steps else normalized
        variants.append(FormulaVariant(normalized, label, FilterSettings(workflow=steps)))
    return tuple(variants)


def split_formula_list(expression_text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(expression_text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character in {";", "\n"} or (character == "," and depth == 0):
            part = expression_text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    final = expression_text[start:].strip()
    if final:
        parts.append(final)
    return parts


def expand_filter_formula_shorthand(expression: str, source_column: str) -> str:
    text = re.sub(r"\s+", "", expression)
    if not text or text.upper() in {"RAW", "NONE"}:
        return source_column
    if source_column in text:
        return text

    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    functions = "|".join(
        re.escape(alias)
        for alias in sorted(FILTER_SPEC_BY_ALIAS, key=len, reverse=True)
    )
    pattern = re.compile(rf"\b({functions})\(({number}(?:,{number})*)\)", re.IGNORECASE)
    while True:
        expanded = pattern.sub(
            lambda match: f"{match.group(1)}({source_column},{match.group(2)})",
            text,
        )
        if expanded == text:
            break
        text = expanded
    return text

def _parse_filter_formula_node(expression: str) -> tuple[str, tuple[FilterStep, ...]]:
    outer = _outer_workflow_call(expression)
    if outer is None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression):
            raise ValueError(f"Invalid source column '{expression}'.")
        return canonical_formula_source(expression), ()

    function_name, arguments_text = outer
    arguments = _split_formula_arguments(arguments_text)
    if not arguments:
        raise ValueError(f"{function_name} requires a source signal.")
    source_column, steps = _parse_filter_formula_node(arguments[0])
    try:
        parameters = tuple(float(value) for value in arguments[1:])
    except ValueError as exc:
        raise ValueError(f"{function_name} parameters must be numbers.") from exc

    specification = FILTER_SPEC_BY_ALIAS.get(function_name.upper())
    if specification is None:
        raise ValueError(f"Unknown filter function '{function_name}'.")

    maximum_parameters = len(specification.parameters)
    if not specification.minimum_parameters <= len(parameters) <= maximum_parameters:
        required = (
            str(specification.minimum_parameters)
            if specification.minimum_parameters == maximum_parameters
            else f"{specification.minimum_parameters}-{maximum_parameters}"
        )
        raise ValueError(
            f"{function_name.upper()} accepts {required} parameter(s)."
        )

    defaults = tuple(
        parameter.default for parameter in specification.parameters
    )
    complete_parameters = parameters + defaults[len(parameters):]
    step = _validated_step(specification.operation, complete_parameters)
    return source_column, (*steps, step)


def _split_formula_arguments(arguments: str) -> list[str]:
    if not arguments:
        return []
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(arguments):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Formula has unmatched parentheses.")
        elif character == "," and depth == 0:
            part = arguments[start:index]
            if not part:
                raise ValueError("Formula contains an empty argument.")
            parts.append(part)
            start = index + 1
    if depth:
        raise ValueError("Formula has unmatched parentheses.")
    final_part = arguments[start:]
    if not final_part:
        raise ValueError("Formula contains an empty argument.")
    parts.append(final_part)
    return parts

def _parse_workflow_sequence(expression: str) -> list[FilterStep]:
    parts = _split_workflow(expression)
    if len(parts) > 1:
        steps: list[FilterStep] = []
        for part in parts:
            steps.extend(_parse_workflow_term(part))
        return steps
    return _parse_workflow_term(expression)


def _split_workflow(expression: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(expression):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Workflow has an unmatched closing parenthesis.")
        elif character in {">", "|"} and depth == 0:
            part = expression[start:index]
            if not part:
                raise ValueError("Workflow contains an empty step.")
            parts.append(part)
            start = index + 1
    if depth:
        raise ValueError("Workflow has an unmatched opening parenthesis.")
    final_part = expression[start:]
    if not final_part:
        raise ValueError("Workflow contains an empty step.")
    parts.append(final_part)
    return parts


def _parse_workflow_term(term: str) -> list[FilterStep]:
    outer = _outer_workflow_call(term)
    if outer is not None:
        head, argument = outer
        parameter_step = _parse_parameter_call(head, argument)
        if parameter_step is not None:
            return [parameter_step]
        inner_steps = _parse_workflow_sequence(argument)
        return [*inner_steps, _parse_compact_step(head)]
    return [_parse_compact_step(term)]


def _outer_workflow_call(term: str) -> tuple[str, str] | None:
    opening = term.find("(")
    if opening <= 0 or not term.endswith(")"):
        return None
    depth = 0
    for index in range(opening, len(term)):
        if term[index] == "(":
            depth += 1
        elif term[index] == ")":
            depth -= 1
            if depth == 0 and index != len(term) - 1:
                return None
    if depth != 0:
        raise ValueError("Workflow has unmatched parentheses.")
    return term[:opening], term[opening + 1 : -1]


def _parse_parameter_call(head: str, argument: str) -> FilterStep | None:
    specification = FILTER_SPEC_BY_ALIAS.get(head.upper())
    if specification is None:
        return None
    try:
        values = tuple(float(value) for value in argument.split(","))
    except ValueError:
        return None
    maximum_parameters = len(specification.parameters)
    if not specification.minimum_parameters <= len(values) <= maximum_parameters:
        return None
    defaults = tuple(
        parameter.default for parameter in specification.parameters
    )
    complete_values = values + defaults[len(values):]
    return _validated_step(specification.operation, complete_values)

def _parse_compact_step(term: str) -> FilterStep:
    patterns = (
        (r"(?:LP|LOWPASS)(\d*\.?\d+)", "lowpass", None),
        (r"(?:HP|HIGHPASS)(\d*\.?\d+)", "highpass", None),
        (r"(?:MA|MOVINGAVERAGE)(\d+)?", "moving_average", (21.0,)),
        (r"(?:SG|SAVGOL)(\d+)?", "savgol", (21.0, 3.0)),
        (r"(?:WH|WHITTAKER)(\d*\.?\d+)?", "whittaker", (1_000.0,)),
    )
    for pattern, operation, defaults in patterns:
        match = re.fullmatch(pattern, term)
        if match is None:
            continue
        if match.groups() and match.group(1) is not None:
            value = float(match.group(1))
            parameters = (value, 3.0) if operation == "savgol" else (value,)
        else:
            parameters = defaults or ()
        return _validated_step(operation, parameters)
    raise ValueError(f"Unknown workflow step '{term}'.")


def _validated_step(operation: str, parameters: tuple[float, ...]) -> FilterStep:
    specification = FILTER_SPEC_BY_OPERATION.get(operation)
    if specification is not None:
        if len(parameters) != len(specification.parameters):
            raise ValueError(
                f"{specification.formula_name} requires "
                f"{len(specification.parameters)} normalized parameter(s)."
            )
        normalized: list[float] = []
        for value, parameter in zip(parameters, specification.parameters):
            if parameter.integer:
                integer_value = int(value)
                if not np.isclose(value, integer_value):
                    raise ValueError(f"{parameter.label} must be an integer.")
                value = float(integer_value)
            if parameter.min_value is not None and value < parameter.min_value:
                raise ValueError(
                    f"{parameter.label} must be at least {parameter.min_value:g}."
                )
            if parameter.max_value is not None and value > parameter.max_value:
                raise ValueError(
                    f"{parameter.label} must be at most {parameter.max_value:g}."
                )
            normalized.append(float(value))
        parameters = tuple(normalized)

    if operation in {"lowpass", "highpass"} and parameters[0] <= 0:
        raise ValueError("Filter cutoffs must be greater than zero.")
    if operation == "moving_average":
        window = int(parameters[0])
        if window < 1:
            raise ValueError("Moving-average windows must be positive integers.")
        parameters = (float(window),)
    elif operation == "savgol":
        window, order = int(parameters[0]), int(parameters[1])
        if window < 3 or order < 1 or order >= window:
            raise ValueError("Savitzky-Golay requires window >= 3 and order < window.")
        parameters = (float(window), float(order))
    elif operation == "whittaker" and parameters[0] < 0:
        raise ValueError("Whittaker lambda cannot be negative.")
    elif operation == "notch":
        minimum, maximum, count, quality = parameters
        if minimum < 0 or maximum <= minimum or count < 1 or quality <= 0:
            raise ValueError("Invalid notch range, count, or Q value.")
        parameters = (minimum, maximum, float(int(count)), quality)
    return FilterStep(operation, parameters)

CUSTOM_EXPRESSION_PREFIX = "__custom_y_expression__:"
CUSTOM_EXPRESSION_COLUMNS = {
    "t": "time_from_onset_s",
    "Lv": "vertical_distance_L",
    "HS": "radial_Hencky_strain",
    "eps_z": "vertical_strain",
    "D": "diameter_mm",
    "F": "force_g",
    "A": "area_mm2",
    "sig": "stress_Pa",
    "sig_surf": "surface_tension_stress_Pa",
    "delta_sig": "net_stress_Pa",
    "HS_rate": "hencky_strain_rate_1_s",
    "visc_e": "extensional_viscosity_Pa_s",
    "v": VELOCITY_COLUMN,
}
CUSTOM_EXPRESSION_TOKENS = (
    *CUSTOM_EXPRESSION_COLUMNS,
    "D0",
    "pi",
    "ST",
)
CUSTOM_EXPRESSION_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "minimum": np.minimum,
    "maximum": np.maximum,
}
_CUSTOM_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
    ast.Mod: lambda left, right: left % right,
}
_CUSTOM_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def is_custom_expression_column(column: object) -> bool:
    return str(column).startswith(CUSTOM_EXPRESSION_PREFIX)


def custom_expression_text(column: object) -> str:
    text = str(column)
    return text[len(CUSTOM_EXPRESSION_PREFIX):] if is_custom_expression_column(text) else text


def validate_custom_expression(expression: str) -> str:
    text = str(expression).strip()
    if not text:
        raise ValueError("The Y formula cannot be empty.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("The Y formula has invalid syntax.") from exc

    def validate(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            validate(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed.")
        elif isinstance(node, ast.Name):
            if node.id not in CUSTOM_EXPRESSION_TOKENS:
                raise ValueError(f"Unknown formula symbol '{node.id}'.")
        elif isinstance(node, ast.BinOp) and type(node.op) in _CUSTOM_BINARY_OPERATORS:
            validate(node.left)
            validate(node.right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _CUSTOM_UNARY_OPERATORS:
            validate(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in CUSTOM_EXPRESSION_FUNCTIONS:
                raise ValueError(f"Unknown formula function '{node.func.id}'.")
            if node.keywords:
                raise ValueError("Formula functions do not accept named arguments.")
            expected_arguments = 2 if node.func.id in {"minimum", "maximum"} else 1
            if len(node.args) != expected_arguments:
                raise ValueError(
                    f"{node.func.id}() requires {expected_arguments} argument(s)."
                )
            for argument in node.args:
                validate(argument)
        else:
            raise ValueError("The Y formula contains an unsupported operation.")

    validate(tree)
    return text


def custom_expression_column(expression: str) -> str:
    return CUSTOM_EXPRESSION_PREFIX + validate_custom_expression(expression)


def evaluate_custom_expression(
    data: pd.DataFrame,
    expression: str,
    settings: PhysicalSettings,
) -> pd.Series:
    text = validate_custom_expression(expression)

    def values(column: str) -> np.ndarray:
        if column not in data.columns:
            raise ValueError(f"Formula input '{column}' is unavailable.")
        return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)

    namespace: dict[str, Any] = {
        alias: values(column)
        for alias, column in CUSTOM_EXPRESSION_COLUMNS.items()
    }
    diameter = namespace["D"]
    diameter_ratio = values("D_over_D0")
    namespace["D0"] = np.divide(
        diameter,
        diameter_ratio,
        out=np.full(diameter.shape, np.nan, dtype=float),
        where=np.isfinite(diameter_ratio) & (diameter_ratio != 0),
    )
    namespace["pi"] = np.pi
    namespace["ST"] = float(settings.surface_tension_mN_m)
    tree = ast.parse(text, mode="eval")

    def calculate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return calculate(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return namespace[node.id]
        if isinstance(node, ast.BinOp):
            return _CUSTOM_BINARY_OPERATORS[type(node.op)](
                calculate(node.left), calculate(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            return _CUSTOM_UNARY_OPERATORS[type(node.op)](calculate(node.operand))
        if isinstance(node, ast.Call):
            function = CUSTOM_EXPRESSION_FUNCTIONS[node.func.id]
            return function(*(calculate(argument) for argument in node.args))
        raise ValueError("The Y formula contains an unsupported operation.")

    with np.errstate(all="ignore"):
        calculated = np.asarray(calculate(tree), dtype=float)
    if calculated.ndim == 0:
        calculated = np.full(len(data), float(calculated), dtype=float)
    else:
        try:
            calculated = np.broadcast_to(calculated, (len(data),)).astype(float, copy=True)
        except ValueError as exc:
            raise ValueError("The Y formula did not produce one value per row.") from exc
    calculated[~np.isfinite(calculated)] = np.nan
    return pd.Series(calculated, index=data.index, name=custom_expression_column(text))


def add_custom_expression_column(
    data: pd.DataFrame,
    column: str,
    settings: PhysicalSettings,
) -> pd.DataFrame:
    if not is_custom_expression_column(column):
        return data
    result = data.copy()
    result[column] = evaluate_custom_expression(
        result, custom_expression_text(column), settings
    )
    return result


@dataclass
class ProcessedFrame:
    frame: pd.DataFrame
    notch_peaks_hz: dict[str, list[float]]
    warnings: list[str]


@dataclass
class PreprocessResult:
    frame: pd.DataFrame
    warnings: list[str]


@dataclass
class PostprocessingAnalysis:
    frame: pd.DataFrame
    windows: dict[str, LinearRangeResult]
    warnings: list[str]
    window_starts: dict[str, tuple[int, float, float]] = field(
        default_factory=dict
    )


@dataclass
class FrequencyResult:
    frequency_hz: np.ndarray
    fft_amplitude: np.ndarray
    psd_frequency_hz: np.ndarray
    psd: np.ndarray
    energy_left_hz: np.ndarray
    energy_right_hz: np.ndarray
    energy: np.ndarray
    peaks_hz: np.ndarray
    peak_amplitudes: np.ndarray
    sample_rate_hz: float
    notch_peaks_hz: list[float]
    warnings: list[str]


@dataclass
class FrequencyBatchResult:
    results: dict[str, FrequencyResult]
    failures: dict[str, str]


def preprocess_experiments(
    data: pd.DataFrame,
    settings: PhysicalSettings,
) -> PreprocessResult:
    pieces: list[pd.DataFrame] = []
    warnings: list[str] = []

    for source_file, group in data.groupby("source_file", sort=False):
        ordered = group.sort_values("time_from_onset_s").copy()
        times = pd.to_numeric(ordered["time_from_onset_s"], errors="coerce")
        strain = pd.to_numeric(
            ordered[settings.crop_strain_column], errors="coerce"
        )

        keep = times > settings.crop_min_time_s
        if settings.crop_enabled:
            keep = keep & strain.notna() & (strain <= settings.crop_threshold)
            if not keep.any():
                warnings.append(
                    f"{source_file}: crop threshold removed every row"
                )

        if settings.force_tail_offset_enabled:
            force = pd.to_numeric(ordered["force_g"], errors="coerce")
            tail_force = force[
                force.notna()
                & strain.notna()
                & (strain >= settings.crop_threshold)
            ]
            if tail_force.empty:
                warnings.append(
                    f"{source_file}: no tail samples reached the crop threshold; "
                    "force offset was not applied"
                )
            else:
                ordered["force_g"] = force - float(tail_force.mean())

        pieces.append(ordered.loc[keep])

    if not pieces:
        return PreprocessResult(data.copy(), ["No experiments were available."])
    return PreprocessResult(
        frame=pd.concat(pieces, ignore_index=True),
        warnings=_unique_strings(warnings),
    )


def add_derived_columns(
    data: pd.DataFrame,
    settings: PhysicalSettings,
) -> pd.DataFrame:
    result = data.copy()
    diameter_m = pd.to_numeric(result["diameter_mm"], errors="coerce") * 1e-3
    valid_diameter = diameter_m.where(diameter_m > 0)

    result["area_mm2"] = np.pi * (result["diameter_mm"] / 2.0) ** 2
    area_m2 = np.pi * (valid_diameter / 2.0) ** 2
    force_n = (
        pd.to_numeric(result["force_g"], errors="coerce") - settings.force_zero_g
    ) * 9.80665e-3
    result["stress_Pa"] = force_n / area_m2

    surface_tension_n_m = settings.surface_tension_mN_m * 1e-3
    result["surface_tension_stress_Pa"] = (
        settings.capillary_factor * surface_tension_n_m / valid_diameter
    )
    result["net_stress_Pa"] = (
        result["stress_Pa"] - result["surface_tension_stress_Pa"]
    )

    result["hencky_strain_rate_1_s"] = np.nan

    for _, group in result.groupby("source_file", sort=False):
        ordered = group.sort_values("time_from_onset_s")
        times = ordered["time_from_onset_s"].to_numpy(dtype=float)
        strain = pd.to_numeric(
            ordered["radial_Hencky_strain"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(times) & np.isfinite(strain)
        if valid.sum() < 3:
            continue

        valid_indices = ordered.index.to_numpy()[valid]
        unique_times, unique_positions = np.unique(times[valid], return_index=True)
        unique_strain = strain[valid][unique_positions]
        if unique_times.size < 3:
            continue

        strain_rate = np.gradient(unique_strain, unique_times, edge_order=2)
        result.loc[valid_indices[unique_positions], "hencky_strain_rate_1_s"] = (
            strain_rate
        )

    rate = result["hencky_strain_rate_1_s"]
    safe_rate = rate.where(rate.abs() >= settings.min_abs_strain_rate)
    result["extensional_viscosity_Pa_s"] = result["net_stress_Pa"] / safe_rate

    result = add_custom_derived_columns(result, settings)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("add_custom_derived_columns() must return a DataFrame.")
    for definition in CUSTOM_DERIVED_QUANTITIES:
        if definition.column not in result.columns:
            result[definition.column] = np.nan

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def estimate_sampling_interval(time_s: np.ndarray) -> float:
    times = np.asarray(time_s, dtype=float)
    times = np.unique(times[np.isfinite(times)])
    if times.size < 2:
        raise ValueError("At least two unique finite time samples are required.")
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    if positive_steps.size == 0:
        raise ValueError("Sampling interval must be positive.")
    interval = float(np.median(positive_steps))
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Sampling interval must be finite and positive.")
    return interval


def process_plot_frame(
    frame: pd.DataFrame,
    y_column: str,
    settings: FilterSettings,
) -> ProcessedFrame:
    processed = frame.copy()
    output_column = f"{y_column}__processed"
    processed[output_column] = np.nan
    notch_peaks: dict[str, list[float]] = {}
    warnings: list[str] = []

    for source_file, group in processed.groupby("source_file", sort=False):
        ordered = group.sort_values("time_from_onset_s")
        filtered_values, selected_peaks, signal_warnings = process_signal(
            ordered["time_from_onset_s"].to_numpy(dtype=float),
            ordered[y_column].to_numpy(dtype=float),
            settings,
        )
        processed.loc[ordered.index, output_column] = filtered_values
        notch_peaks[str(source_file)] = selected_peaks
        warnings.extend(
            f"{source_file}: {message}" for message in signal_warnings
        )

    return ProcessedFrame(
        frame=processed,
        notch_peaks_hz=notch_peaks,
        warnings=_unique_strings(warnings),
    )


def process_axes_frame(
    frame: pd.DataFrame,
    x_column: str,
    x_settings: FilterSettings,
    y_column: str,
    y_settings: FilterSettings,
) -> ProcessedFrame:
    processed = frame.copy()
    notch_peaks: dict[str, list[float]] = {}
    warnings: list[str] = []
    axes = (
        ("x", x_column, x_settings),
        ("y", y_column, y_settings),
    )
    for axis_name, column, settings in axes:
        output_column = f"{column}__{axis_name}_processed"
        processed[output_column] = pd.to_numeric(
            processed[column], errors="coerce"
        )
        if not settings.active:
            continue
        for source_file, group in processed.groupby("source_file", sort=False):
            ordered = group.sort_values("time_from_onset_s")
            filtered_values, selected_peaks, signal_warnings = process_signal(
                ordered["time_from_onset_s"].to_numpy(dtype=float),
                ordered[column].to_numpy(dtype=float),
                settings,
            )
            processed.loc[ordered.index, output_column] = filtered_values
            notch_peaks[f"{axis_name}:{source_file}"] = selected_peaks
            warnings.extend(
                f"{axis_name.upper()} / {source_file}: {message}"
                for message in signal_warnings
            )

    return ProcessedFrame(
        frame=processed,
        notch_peaks_hz=notch_peaks,
        warnings=_unique_strings(warnings),
    )


def process_measured_variant_frame(
    frame: pd.DataFrame,
    force_variant: FormulaVariant,
    strain_variant: FormulaVariant,
    physical_settings: PhysicalSettings,
) -> ProcessedFrame:
    processed = frame.copy()
    notch_peaks: dict[str, list[float]] = {}
    warnings: list[str] = []

    measured_variants = (
        ("force", "force_g", force_variant),
        ("strain", "radial_Hencky_strain", strain_variant),
    )
    for signal_name, column, variant in measured_variants:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        if not variant.settings.active:
            continue
        for source_file, group in processed.groupby("source_file", sort=False):
            ordered = group.sort_values("time_from_onset_s")
            filtered_values, selected_peaks, signal_warnings = process_signal(
                ordered["time_from_onset_s"].to_numpy(dtype=float),
                ordered[column].to_numpy(dtype=float),
                variant.settings,
            )
            processed.loc[ordered.index, column] = filtered_values
            notch_peaks[f"{signal_name}:{source_file}"] = selected_peaks
            warnings.extend(
                f"{signal_name} / {source_file}: {message}"
                for message in signal_warnings
            )

    derived = add_derived_columns(processed, physical_settings)
    derived[VARIANT_COLUMN] = make_variant_label(force_variant, strain_variant)
    derived[TRACE_COLUMN] = (
        derived["source_file"].astype(str) + " | " + derived[VARIANT_COLUMN].astype(str)
    )
    return ProcessedFrame(
        frame=derived,
        notch_peaks_hz=notch_peaks,
        warnings=_unique_strings(warnings),
    )


def process_measured_variant_frames(
    frame: pd.DataFrame,
    force_variants: tuple[FormulaVariant, ...],
    strain_variants: tuple[FormulaVariant, ...],
    physical_settings: PhysicalSettings,
) -> ProcessedFrame:
    force_variants, strain_variants = align_formula_variants(
        force_variants, strain_variants
    )
    frames: list[pd.DataFrame] = []
    notch_peaks: dict[str, list[float]] = {}
    warnings: list[str] = []

    for force_variant, strain_variant in zip(force_variants, strain_variants):
        result = process_measured_variant_frame(
            frame, force_variant, strain_variant, physical_settings
        )
        frames.append(result.frame)
        notch_peaks.update(result.notch_peaks_hz)
        warnings.extend(result.warnings)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ProcessedFrame(
        frame=combined,
        notch_peaks_hz=notch_peaks,
        warnings=_unique_strings(warnings),
    )


def make_formula_variant(
    source_column: str,
    settings: FilterSettings,
) -> FormulaVariant:
    expression = format_filter_formula(
        formula_source_label(source_column), settings.workflow
    )
    return FormulaVariant(
        expression=expression,
        label="Raw" if not settings.active else expression,
        settings=settings,
    )


def analyze_postprocessing_runs(
    frame: pd.DataFrame,
    force_settings: FilterSettings,
    strain_settings: FilterSettings,
    y_column: str,
    y_settings: FilterSettings,
    physical_settings: PhysicalSettings,
    r2_threshold: float,
    min_points: int,
    epsilon_start: float,
) -> PostprocessingAnalysis:
    """Process signals and locate each run's linear strain interval."""
    if not np.isfinite(epsilon_start):
        raise ValueError("The working-window start strain must be finite.")
    processed = process_measured_variant_frame(
        frame,
        make_formula_variant("force_g", force_settings),
        make_formula_variant("radial_Hencky_strain", strain_settings),
        physical_settings,
    )
    analysis_frame = processed.frame
    warnings = list(processed.warnings)
    if y_column not in analysis_frame.columns:
        raise ValueError(f"Unknown postprocessing variable: {y_column}")
    if y_settings.active:
        y_result = process_plot_frame(analysis_frame, y_column, y_settings)
        output_column = f"{y_column}__processed"
        analysis_frame = y_result.frame
        analysis_frame[y_column] = analysis_frame[output_column]
        analysis_frame = analysis_frame.drop(columns=[output_column])
        warnings.extend(y_result.warnings)

    windows: dict[str, LinearRangeResult] = {}
    window_starts: dict[str, tuple[int, float, float]] = {}
    for source_file, group in analysis_frame.groupby(
        "source_file", sort=False
    ):
        ordered = group.sort_values("time_from_onset_s")
        valid = ordered[[
            "time_from_onset_s", "radial_Hencky_strain"
        ]].dropna()
        start_candidates = np.flatnonzero(
            valid["radial_Hencky_strain"].to_numpy(dtype=float) >= epsilon_start
        )
        if not start_candidates.size:
            windows[str(source_file)] = LinearRangeResult(
                None, None, None, None, None
            )
            warnings.append(
                f"{source_file}: no strain samples reached "
                f"εᵣ,0 = {epsilon_start:g}"
            )
            continue

        start_index = int(start_candidates[0])
        candidates = valid.iloc[start_index:]
        start_row = candidates.iloc[0]
        window_starts[str(source_file)] = (
            start_index,
            float(start_row["time_from_onset_s"]),
            float(start_row["radial_Hencky_strain"]),
        )
        try:
            result = find_max_initial_linear_range(
                candidates["time_from_onset_s"].to_numpy(dtype=float),
                candidates["radial_Hencky_strain"].to_numpy(dtype=float),
                r2_threshold=r2_threshold,
                min_points=min_points,
            )
            if result.endpoint_index is not None:
                result = replace(
                    result,
                    endpoint_index=start_index + result.endpoint_index,
                )
            windows[str(source_file)] = result
        except ValueError as exc:
            windows[str(source_file)] = LinearRangeResult(
                None, None, None, None, None
            )
            warnings.append(f"{source_file}: working-window fit failed ({exc})")
    return PostprocessingAnalysis(
        frame=analysis_frame,
        windows=windows,
        warnings=_unique_strings(warnings),
        window_starts=window_starts,
    )

def align_formula_variants(
    force_variants: tuple[FormulaVariant, ...],
    strain_variants: tuple[FormulaVariant, ...],
) -> tuple[tuple[FormulaVariant, ...], tuple[FormulaVariant, ...]]:
    if not force_variants:
        force_variants = parse_filter_formula_list("force_g", "force_g")
    if not strain_variants:
        strain_variants = parse_filter_formula_list(
            "radial_Hencky_strain", "radial_Hencky_strain"
        )
    count = max(len(force_variants), len(strain_variants))
    if len(force_variants) == 1 and count > 1:
        force_variants = force_variants * count
    if len(strain_variants) == 1 and count > 1:
        strain_variants = strain_variants * count
    if len(force_variants) != len(strain_variants):
        raise ValueError(
            "Force and HS-strain formula lists must have the same length, "
            "unless one list has only one formula."
        )
    return force_variants, strain_variants


def make_variant_label(
    force_variant: FormulaVariant,
    strain_variant: FormulaVariant,
) -> str:
    parts: list[str] = []
    if force_variant.settings.active:
        parts.append(f"F: {force_variant.label}")
    if strain_variant.settings.active:
        parts.append(f"eps: {strain_variant.label}")
    return " | ".join(parts) if parts else "Raw"

def process_signal(
    time_s: np.ndarray,
    values: np.ndarray,
    settings: FilterSettings,
) -> tuple[np.ndarray, list[float], list[str]]:
    valid = np.isfinite(time_s) & np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=float)
    if valid.sum() < 5:
        return output, [], ["not enough finite samples"]

    valid_positions = np.flatnonzero(valid)
    times = time_s[valid]
    signal_values = values[valid]
    order = np.argsort(times)
    times = times[order]
    signal_values = signal_values[order]
    sorted_positions = valid_positions[order]

    unique_times, inverse = np.unique(times, return_inverse=True)
    had_duplicates = unique_times.size != times.size
    if unique_times.size < 5:
        return output, [], ["not enough unique time samples"]
    if had_duplicates:
        sums = np.bincount(inverse, weights=signal_values)
        counts = np.bincount(inverse)
        signal_values = sums / counts
        times = unique_times

    try:
        median_step = estimate_sampling_interval(times)
    except ValueError:
        return output, [], ["invalid sampling interval"]

    uniform_time = np.arange(times[0], times[-1] + median_step * 0.5, median_step)
    uniform_values = np.interp(uniform_time, times, signal_values)
    sample_rate = 1.0 / median_step
    nyquist = sample_rate / 2.0
    filtered = uniform_values.copy()
    warnings: list[str] = []

    workflow = settings.workflow or _legacy_filter_steps(settings, warnings)
    filtered, notch_peaks = _apply_filter_steps(
        filtered, workflow, sample_rate, nyquist, warnings
    )
    if had_duplicates:
        output[valid_positions] = np.interp(time_s[valid], uniform_time, filtered)
    else:
        output[sorted_positions] = np.interp(times, uniform_time, filtered)
    return output, notch_peaks, warnings


def analyze_frequency(
    frame: pd.DataFrame,
    y_column: str,
    settings: FilterSettings,
    peak_min_hz: float,
    peak_max_hz: float,
    peak_count: int,
    energy_bins: int,
) -> FrequencyResult:
    ordered = frame.sort_values("time_from_onset_s")
    time_s = ordered["time_from_onset_s"].to_numpy(dtype=float)
    values = ordered[y_column].to_numpy(dtype=float)
    valid = np.isfinite(time_s) & np.isfinite(values)
    if valid.sum() < 8:
        raise ValueError("At least eight finite samples are required.")

    times = time_s[valid]
    signal_values = values[valid]
    unique_times, unique_positions = np.unique(times, return_index=True)
    signal_values = signal_values[unique_positions]
    times = unique_times
    if times.size < 8:
        raise ValueError("At least eight unique finite samples are required.")
    median_step = estimate_sampling_interval(times)
    sample_rate = 1.0 / median_step
    uniform_time = np.arange(times[0], times[-1] + median_step * 0.5, median_step)
    uniform_values = np.interp(uniform_time, times, signal_values)

    processed_values, notch_peaks, warnings = process_signal(
        uniform_time,
        uniform_values,
        settings,
    )
    signal_for_spectrum = processed_values - np.nanmean(processed_values)
    sample_count = signal_for_spectrum.size
    frequency = np.fft.rfftfreq(sample_count, d=median_step)
    fft_amplitude = 2.0 * np.abs(np.fft.rfft(signal_for_spectrum)) / sample_count
    if fft_amplitude.size:
        fft_amplitude[0] *= 0.5

    nperseg = min(256, sample_count)
    psd_frequency, psd = welch(
        signal_for_spectrum,
        fs=sample_rate,
        nperseg=nperseg,
        detrend="linear",
        scaling="density",
    )
    peaks_hz, peak_amplitudes = detect_fft_peaks(
        signal_for_spectrum,
        sample_rate,
        peak_min_hz,
        min(peak_max_hz, sample_rate / 2.0),
        peak_count,
    )

    nyquist = sample_rate / 2.0
    energy_min = min(max(0.0, peak_min_hz), nyquist)
    energy_max = min(max(peak_max_hz, energy_min), nyquist)
    if energy_max <= energy_min:
        energy_min, energy_max = 0.0, nyquist
    bin_edges = np.linspace(energy_min, energy_max, max(2, energy_bins) + 1)
    energy = np.zeros(bin_edges.size - 1)
    if psd_frequency.size > 1:
        frequency_step = float(np.median(np.diff(psd_frequency)))
        in_range = (
            (psd_frequency >= energy_min) & (psd_frequency <= energy_max)
        )
        selected_frequency = psd_frequency[in_range]
        selected_psd = psd[in_range]
        bin_ids = np.clip(
            np.digitize(selected_frequency, bin_edges) - 1,
            0,
            energy.size - 1,
        )
        np.add.at(energy, bin_ids, selected_psd * frequency_step)

    return FrequencyResult(
        frequency_hz=frequency,
        fft_amplitude=fft_amplitude,
        psd_frequency_hz=psd_frequency,
        psd=psd,
        energy_left_hz=bin_edges[:-1],
        energy_right_hz=bin_edges[1:],
        energy=energy,
        peaks_hz=peaks_hz,
        peak_amplitudes=peak_amplitudes,
        sample_rate_hz=sample_rate,
        notch_peaks_hz=notch_peaks,
        warnings=warnings,
    )


def analyze_frequency_runs(
    frame: pd.DataFrame,
    y_column: str,
    settings: FilterSettings,
    peak_min_hz: float,
    peak_max_hz: float,
    peak_count: int,
    energy_bins: int,
) -> FrequencyBatchResult:
    results: dict[str, FrequencyResult] = {}
    failures: dict[str, str] = {}
    for source_file, group in frame.groupby("source_file", sort=False):
        try:
            results[str(source_file)] = analyze_frequency(
                group,
                y_column,
                settings,
                peak_min_hz,
                peak_max_hz,
                peak_count,
                energy_bins,
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures[str(source_file)] = str(exc)
    return FrequencyBatchResult(results=results, failures=failures)


def detect_fft_peaks(
    values: np.ndarray,
    sample_rate_hz: float,
    min_hz: float,
    max_hz: float,
    peak_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(values, dtype=float) - np.nanmean(values)
    sample_count = centered.size
    frequency = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_hz)
    amplitude = 2.0 * np.abs(np.fft.rfft(centered)) / sample_count
    mask = (frequency >= max(0.0, min_hz)) & (frequency <= max_hz)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return np.array([]), np.array([])

    local_amplitude = amplitude[indices]
    minimum_prominence = float(np.max(local_amplitude)) * 0.01
    local_peaks, _ = find_peaks(
        local_amplitude,
        prominence=minimum_prominence,
    )
    peak_indices = indices[local_peaks]
    if peak_indices.size == 0:
        return np.array([]), np.array([])

    ranked = peak_indices[np.argsort(amplitude[peak_indices])[::-1]]
    ranked = ranked[: max(1, int(peak_count))]
    ranked = ranked[np.argsort(frequency[ranked])]
    return frequency[ranked], amplitude[ranked]


def _legacy_filter_steps(
    settings: FilterSettings,
    warnings: list[str],
) -> tuple[FilterStep, ...]:
    steps: list[FilterStep] = []
    invalid_band = (
        settings.highpass_hz is not None
        and settings.lowpass_hz is not None
        and settings.highpass_hz >= settings.lowpass_hz
    )
    if invalid_band:
        warnings.append("high-pass cutoff must be below low-pass cutoff")
    else:
        if settings.highpass_hz is not None:
            steps.append(FilterStep("highpass", (settings.highpass_hz,)))
        if settings.lowpass_hz is not None:
            steps.append(FilterStep("lowpass", (settings.lowpass_hz,)))
    if settings.notch_enabled:
        steps.append(
            FilterStep(
                "notch",
                (
                    settings.notch_min_hz,
                    settings.notch_max_hz,
                    float(settings.notch_peak_count),
                    settings.notch_q,
                ),
            )
        )
    if settings.smoothing == "Savitzky-Golay":
        steps.append(
            FilterStep(
                "savgol",
                (float(settings.savgol_window), float(settings.savgol_order)),
            )
        )
    elif settings.smoothing == "Whittaker":
        steps.append(FilterStep("whittaker", (settings.whittaker_lambda,)))
    return tuple(steps)


def _apply_filter_steps(
    values: np.ndarray,
    steps: tuple[FilterStep, ...],
    sample_rate: float,
    nyquist: float,
    warnings: list[str],
) -> tuple[np.ndarray, list[float]]:
    filtered = np.asarray(values, dtype=float).copy()
    notch_peaks: list[float] = []
    for step in steps:
        parameters = step.parameters
        if step.operation in {"lowpass", "highpass"}:
            filtered = _apply_butterworth(
                filtered,
                parameters[0],
                step.operation,
                sample_rate,
                nyquist,
                warnings,
            )
        elif step.operation == "moving_average":
            filtered = moving_average(filtered, int(parameters[0]))
        elif step.operation == "savgol":
            filtered = _apply_savgol(
                filtered, int(parameters[0]), int(parameters[1]), warnings
            )
        elif step.operation == "whittaker":
            filtered = whittaker_smooth(filtered, parameters[0])
        elif step.operation == "notch":
            minimum, maximum, peak_count, quality = parameters
            selected = detect_fft_peaks(
                filtered,
                sample_rate,
                minimum,
                min(maximum, nyquist * 0.999),
                int(peak_count),
            )[0].tolist()
            notch_peaks.extend(selected)
            for peak_hz in selected:
                try:
                    b, a = iirnotch(peak_hz, quality, fs=sample_rate)
                    filtered = filtfilt(b, a, filtered)
                except ValueError as exc:
                    warnings.append(f"notch at {peak_hz:.3g} Hz skipped ({exc})")
        else:
            executor = CUSTOM_FILTER_EXECUTORS.get(step.operation)
            if executor is None:
                warnings.append(
                    f"unknown workflow step '{step.operation}' skipped"
                )
                continue
            try:
                custom_values = np.asarray(
                    executor(
                        filtered,
                        parameters,
                        sample_rate,
                        nyquist,
                        warnings,
                    ),
                    dtype=float,
                )
            except Exception as exc:
                warnings.append(
                    f"custom filter '{step.operation}' skipped ({exc})"
                )
                continue
            if custom_values.shape != filtered.shape:
                warnings.append(
                    f"custom filter '{step.operation}' returned the wrong shape"
                )
                continue
            filtered = custom_values
    return filtered, notch_peaks


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    signal_values = np.asarray(values, dtype=float)
    if signal_values.size == 0:
        return signal_values.copy()
    effective_window = min(signal_values.size, max(1, int(window)))
    if effective_window == 1:
        return signal_values.copy()
    kernel = np.ones(effective_window, dtype=float)
    totals = np.convolve(signal_values, kernel, mode="same")
    counts = np.convolve(np.ones(signal_values.size), kernel, mode="same")
    return totals / counts

def whittaker_smooth(values: np.ndarray, smoothing_lambda: float) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.size < 3:
        return y.copy()
    difference = sparse.diags(
        [np.ones(y.size - 2), -2.0 * np.ones(y.size - 2), np.ones(y.size - 2)],
        [0, 1, 2],
        shape=(y.size - 2, y.size),
        format="csc",
    )
    system = sparse.eye(y.size, format="csc") + smoothing_lambda * (
        difference.T @ difference
    )
    return np.asarray(spsolve(system, y))


def _apply_butterworth(
    values: np.ndarray,
    cutoff_hz: float,
    filter_type: str,
    sample_rate: float,
    nyquist: float,
    warnings: list[str],
) -> np.ndarray:
    if cutoff_hz <= 0 or cutoff_hz >= nyquist:
        warnings.append(
            f"{filter_type} cutoff {cutoff_hz:g} Hz must be between 0 and "
            f"{nyquist:.3g} Hz"
        )
        return values
    try:
        sos = butter(4, cutoff_hz, btype=filter_type, fs=sample_rate, output="sos")
        return sosfiltfilt(sos, values)
    except ValueError as exc:
        warnings.append(f"{filter_type} skipped ({exc})")
        return values


def _apply_savgol(
    values: np.ndarray,
    requested_window: int,
    polynomial_order: int,
    warnings: list[str],
) -> np.ndarray:
    window = max(3, int(requested_window))
    if window % 2 == 0:
        window += 1
    max_window = values.size if values.size % 2 == 1 else values.size - 1
    window = min(window, max_window)
    if window <= polynomial_order:
        warnings.append("Savitzky-Golay window is too short")
        return values
    return savgol_filter(values, window_length=window, polyorder=polynomial_order)


@dataclass(frozen=True)
class LinearRangeResult:
    """Result of the maximum-initial-linearity search."""

    endpoint_index: int | None
    endpoint_x: float | None
    slope: float | None
    intercept: float | None
    r2: float | None
    residuals: np.ndarray | None = None

    @property
    def valid(self) -> bool:
        return self.endpoint_index is not None


def find_max_initial_linear_range(
    x: np.ndarray,
    y: np.ndarray,
    r2_threshold: float = 0.995,
    constrain_to_first: bool = False,
    min_points: int = 3,
    include_residuals: bool = False,
) -> LinearRangeResult:
    """Find the longest initial interval that satisfies an R-squared limit.

    Prefix sums make every candidate line fit an O(1) operation. After the
    O(N) prefix preparation, the endpoint search evaluates O(log N) candidates.
    The search assumes that the pass/fail criterion is monotonic as the initial
    interval grows. ``constrain_to_first`` fits ``y = y0 + m * (x - x0)``;
    otherwise an ordinary least-squares intercept is used.

    Constant-y intervals are treated as perfectly linear when their residual
    sum of squares is numerically zero, and as R-squared zero otherwise.

    Example
    -------
    >>> x = np.arange(20, dtype=float)
    >>> y = 2.0 * x + 1.0
    >>> y[12:] += 3.0 * (x[12:] - 11.0) ** 2
    >>> result = find_max_initial_linear_range(x, y, r2_threshold=0.999)
    >>> result.valid and result.endpoint_index < len(x) - 1
    True
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if x_values.size != y_values.size:
        raise ValueError("x and y must contain the same number of points.")
    if min_points < 2:
        raise ValueError("min_points must be at least 2.")
    if not np.isfinite(r2_threshold) or r2_threshold > 1.0:
        raise ValueError("r2_threshold must be finite and no greater than 1.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise ValueError("x and y must contain only finite values.")
    if x_values.size > 1 and np.any(np.diff(x_values) <= 0):
        raise ValueError("x must be strictly increasing with no duplicates.")
    if x_values.size < min_points:
        return LinearRangeResult(None, None, None, None, None)

    def prefix(values: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(values, dtype=float)))

    prefix_x = prefix(x_values)
    prefix_y = prefix(y_values)
    prefix_xx = prefix(x_values * x_values)
    prefix_xy = prefix(x_values * y_values)
    prefix_yy = prefix(y_values * y_values)
    delta_x = x_values - x_values[0]
    delta_y = y_values - y_values[0]
    prefix_dxx = prefix(delta_x * delta_x)
    prefix_dxy = prefix(delta_x * delta_y)
    prefix_dyy = prefix(delta_y * delta_y)
    epsilon = np.finfo(float).eps

    def fit(endpoint: int) -> tuple[float, float, float]:
        count = endpoint + 1
        sum_x = prefix_x[count]
        sum_y = prefix_y[count]
        sum_xx = prefix_xx[count]
        sum_xy = prefix_xy[count]
        sum_yy = prefix_yy[count]

        if constrain_to_first:
            denominator = prefix_dxx[count]
            denominator_tolerance = 64.0 * epsilon * max(
                1.0, abs(denominator)
            )
            if denominator <= denominator_tolerance:
                raise ValueError("x values do not span a numerically stable range.")
            slope = prefix_dxy[count] / denominator
            intercept = y_values[0] - slope * x_values[0]
            rss = (
                prefix_dyy[count]
                - 2.0 * slope * prefix_dxy[count]
                + slope * slope * denominator
            )
        else:
            denominator = sum_xx - sum_x * sum_x / count
            denominator_tolerance = 64.0 * epsilon * max(
                1.0, abs(sum_xx), abs(sum_x * sum_x / count)
            )
            if denominator <= denominator_tolerance:
                raise ValueError("x values do not span a numerically stable range.")
            slope = (sum_xy - sum_x * sum_y / count) / denominator
            intercept = (sum_y - slope * sum_x) / count
            rss = (
                sum_yy
                + slope * slope * sum_xx
                + count * intercept * intercept
                - 2.0 * slope * sum_xy
                - 2.0 * intercept * sum_y
                + 2.0 * slope * intercept * sum_x
            )

        rss_scale = max(
            1.0,
            abs(sum_yy),
            abs(slope * slope * sum_xx),
            abs(count * intercept * intercept),
        )
        rss_tolerance = 128.0 * epsilon * rss_scale
        if abs(rss) <= rss_tolerance:
            rss = 0.0
        else:
            rss = max(0.0, float(rss))

        sst = sum_yy - sum_y * sum_y / count
        sst_scale = max(1.0, abs(sum_yy), abs(sum_y * sum_y / count))
        sst_tolerance = 128.0 * epsilon * sst_scale
        if sst <= sst_tolerance:
            r2 = 1.0 if rss <= rss_tolerance else 0.0
        else:
            r2 = 1.0 - rss / sst
        return float(slope), float(intercept), float(r2)

    def finish(
        endpoint: int,
        final_fit: tuple[float, float, float],
    ) -> LinearRangeResult:
        slope, intercept, r2 = final_fit
        residuals = None
        if include_residuals:
            fitted = slope * x_values[: endpoint + 1] + intercept
            residuals = y_values[: endpoint + 1] - fitted
        return LinearRangeResult(
            endpoint_index=endpoint,
            endpoint_x=float(x_values[endpoint]),
            slope=slope,
            intercept=intercept,
            r2=r2,
            residuals=residuals,
        )

    final_index = x_values.size - 1
    full_fit = fit(final_index)
    if full_fit[2] >= r2_threshold:
        return finish(final_index, full_fit)

    first_candidate = min_points - 1
    first_fit = fit(first_candidate)
    if first_fit[2] < r2_threshold:
        return LinearRangeResult(None, None, None, None, None)

    lo = first_candidate
    hi = final_index
    lo_fit = first_fit
    while hi - lo > 1:
        mid = (lo + hi) // 2
        mid_fit = fit(mid)
        if mid_fit[2] >= r2_threshold:
            lo = mid
            lo_fit = mid_fit
        else:
            hi = mid
    return finish(lo, lo_fit)

def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))

# =============================================================================
# STREAMLIT FRONTEND / GUI (APPLICATION LAYER)
# =============================================================================
# Physics contributors should not need to edit anything below this line.

import hashlib
import math
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

STREAMLIT_BOOTSTRAP_ENV = "VADER_STREAMLIT_BOOTSTRAPPED"
def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent
ROOT_DIR = app_root()
DATA_DIR = ROOT_DIR / "data"
FILENAME_SUFFIX = "_PROC1_FAST_processed.csv"
BRAND_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" x="0px" y="0px" viewBox="0 0 100 125" enable-background="new 0 0 100 100" xml:space="preserve">
    <path d="M83.143,43.286V31.714C83.143,17.572,71.57,6,57.429,6H44.571C30.429,6,18.856,17.572,18.856,31.714v11.572L6,76.715  C6,83.143,26.147,96,51,96s45-12.857,45-19.285L83.143,43.286z M31.714,85.554c-1.332,0-2.411-1.08-2.411-2.411  c0-1.333,1.079-2.41,2.411-2.41c1.333,0,2.411,1.077,2.411,2.41C34.125,84.474,33.047,85.554,31.714,85.554 M28.5,51  c0-1.768,0.894-4.353,1.983-5.743c0,0,3.055-3.9,9.267-3.9c3.152,0,5.351,1.449,5.351,1.449c1.477,0.971,2.685,3.212,2.685,4.979  v4.822c0,0.883-1.446,1.607-3.214,1.607H31.714C29.946,54.215,28.5,52.767,28.5,51 M41.356,83.143h-3.214v-6.428l3.214-3.215V83.143  z M49.392,83.143h-4.821V70.286h4.821V83.143z M46.178,67.071V62.25c0-2.664,2.161-4.821,4.822-4.821  c2.662,0,4.821,2.157,4.821,4.821v4.821H46.178z M57.429,83.143h-4.821V70.286h4.821V83.143z M63.857,83.143h-3.214V73.5  l3.214,3.215V83.143z M70.286,85.554c-1.333,0-2.411-1.08-2.411-2.411c0-1.333,1.078-2.41,2.411-2.41c1.331,0,2.41,1.077,2.41,2.41  C72.696,84.474,71.617,85.554,70.286,85.554 M70.286,54.215H57.429c-1.769,0-3.214-0.725-3.214-1.607v-4.822  c0-1.768,1.207-4.009,2.685-4.979c0,0,2.199-1.449,5.351-1.449c6.215,0,9.267,3.9,9.267,3.9C72.608,46.647,73.5,49.232,73.5,51  C73.5,52.767,72.052,54.215,70.286,54.215 M90.153,78.213c-0.19,0.075-0.388,0.108-0.582,0.108c-0.645,0-1.251-0.389-1.496-1.023  L76.386,47.273c-3.091-7.723-6.912-10.738-13.625-10.738c-4.453,0-8.19,2.762-8.224,2.792c-1.984,1.487-5.088,1.48-7.073,0  c-0.033-0.03-3.806-2.792-8.226-2.792c-6.711,0-10.532,3.015-13.628,10.751L13.926,77.298c-0.321,0.827-1.257,1.237-2.081,0.915  c-0.825-0.32-1.236-1.252-0.913-2.082L22.62,46.108c3.585-8.966,8.554-12.787,16.618-12.787c5.515,0,9.968,3.294,10.154,3.431  c0.842,0.634,2.375,0.634,3.215,0c0.187-0.137,4.642-3.431,10.153-3.431c8.065,0,13.033,3.821,16.615,12.774L91.07,76.131  C91.39,76.961,90.982,77.893,90.153,78.213"/>
</svg>
""".strip()

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
RAW_AXIS_COLUMNS = [*REQUIRED_COLUMNS, VELOCITY_COLUMN]
PROCESSED_AXIS_COLUMNS = [*AVAILABLE_COLUMNS, VELOCITY_COLUMN]
METADATA_COLUMNS = [
    "material", "velocity", VELOCITY_COLUMN, "sample", "source_file"
]
RAW_DEFAULT_PLOTS = [
    {"x": "time_from_onset_s", "y": "force_g"},
    {"x": "time_from_onset_s", "y": "diameter_mm"},
]
PROCESSED_DEFAULT_PLOTS = [
    {"x": "time_from_onset_s", "y": "force_g"},
    {"x": "radial_Hencky_strain", "y": "net_stress_Pa"},
]
DATA_PLOT_HEIGHT = 520
POSTPROCESSING_SMALL_PLOT_HEIGHT = 225
POSTPROCESSING_TALL_PLOT_HEIGHT = 487
DEFAULT_STRAIN_FORMULA = "MA(HS,121)"
DEFAULT_POSTPROCESSING_FORCE_FORMULA = "MA(LP(F,20),1000)"

IFF_PALETTE = (
    "#0075CF",
    "#00A6A6",
    "#6F52A2",
    "#E69F00",
    "#2E8B57",
    "#56B4E9",
    "#8C6D31",
    "#6B7280",
    "#C58AF9",
    "#009E73",
)


@dataclass(frozen=True)
class PlotRequest:
    x_column: str
    y_column: str
    selected_files: tuple[str, ...]
    force_variants: tuple[FormulaVariant, ...]
    strain_variants: tuple[FormulaVariant, ...]
    show_raw_overlay: bool
    show_legend: bool
    x_scale: str
    y_scale: str
    selected_materials: tuple[str, ...]
    selected_velocities: tuple[str, ...]
    physical_settings: PhysicalSettings


@dataclass(frozen=True)
class SummaryRequest:
    selected_files: tuple[str, ...]
    x_column: str
    y_column: str
    group_by: str
    bin_count: int
    show_legend: bool
    x_scale: str
    y_scale: str
    selected_materials: tuple[str, ...]
    selected_velocities: tuple[str, ...]
    physical_settings: PhysicalSettings


@dataclass(frozen=True)
class PostprocessingRequest:
    selected_files: tuple[str, ...]
    y_column: str
    force_settings: FilterSettings
    strain_settings: FilterSettings
    y_settings: FilterSettings
    r2_threshold: float
    min_points: int
    epsilon_start: float
    show_legend: bool
    y_scale: str
    show_window: bool
    show_slope: bool
    selected_materials: tuple[str, ...]
    selected_velocities: tuple[str, ...]
    physical_settings: PhysicalSettings

@dataclass(frozen=True)
class FrequencyRequest:
    selected_files: tuple[str, ...]
    signal_column: str
    filter_settings: FilterSettings
    peak_settings: tuple[float, float, int, int, float]
    individual_plot: str
    summary_plot: str
    show_peaks: bool
    show_legend: bool
    individual_x_scale: str
    individual_y_scale: str
    summary_x_scale: str
    summary_y_scale: str
    selected_materials: tuple[str, ...]
    selected_velocities: tuple[str, ...]
    physical_settings: PhysicalSettings


def plot_analysis_key(request: PlotRequest) -> tuple[Any, ...]:
    return (
        request.x_column,
        request.y_column,
        request.selected_files,
        repr(request.force_variants),
        repr(request.strain_variants),
        request.selected_materials,
        request.selected_velocities,
        repr(request.physical_settings),
    )


def frequency_analysis_key(request: FrequencyRequest) -> tuple[Any, ...]:
    return (
        request.selected_files,
        request.signal_column,
        repr(request.filter_settings),
        request.peak_settings,
        request.selected_materials,
        request.selected_velocities,
        repr(request.physical_settings),
    )


def column_axis_title(column: str) -> str:
    if is_custom_expression_column(column):
        return custom_expression_text(column)
    symbol = COLUMN_AXIS_SYMBOLS.get(column, column)
    unit = COLUMN_UNITS.get(column, "-")
    return f"{symbol} [{unit}]"


def parse_velocity_value(value: Any) -> float:
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    return float(match.group(0)) if match else float("nan")


def velocity_sort_key(value: Any) -> tuple[int, float, str]:
    numeric = parse_velocity_value(value)
    if np.isfinite(numeric):
        return (0, numeric, str(value).casefold())
    return (1, float("inf"), str(value).casefold())


def sorted_velocity_values(values: Any) -> list[str]:
    return sorted(
        {str(value) for value in values if pd.notna(value)},
        key=velocity_sort_key,
    )


def normalize_axis_scale(value: str) -> str:
    return "log" if str(value).lower() == "log" else "linear"



def running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False
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
    file_summary = load_file_summary(str(DATA_DIR), signature)
    if file_summary.empty:
        st.info("No CSV files were found in the data folder.")
        st.code(str(DATA_DIR), language="text")
        return

    workspace_scope = next(
        scope for name, _, scope in WORKSPACE_DEFINITIONS if name == workspace
    )
    inactive_scopes = [
        scope
        for _, _, scope in WORKSPACE_DEFINITIONS
        if scope != workspace_scope
    ]
    hidden_workspace_css = "\n".join(
        f".st-key-workspace_slot_{scope} {{ display: none !important; }}"
        for scope in inactive_scopes
    )
    st.markdown(
        f"<style>{hidden_workspace_css}</style>",
        unsafe_allow_html=True,
    )
    workspace_slots = {
        scope: st.container(key=f"workspace_slot_{scope}")
        for _, _, scope in WORKSPACE_DEFINITIONS
    }
    with workspace_slots[workspace_scope]:
        if workspace == WORKSPACE_DATA:
            render_dashboard(
                file_summary, signature, physical_settings,
                "processed", PROCESSED_AXIS_COLUMNS, PROCESSED_DEFAULT_PLOTS,
                allow_processing=True,
            )
        elif workspace == WORKSPACE_FREQUENCY:
            render_frequency_workspace(file_summary, signature, physical_settings)
        elif workspace == WORKSPACE_POSTPROCESSING:
            render_postprocessing_workspace(
                file_summary, signature, physical_settings
            )
        else:
            render_summary_workspace(file_summary, signature, physical_settings)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vader-ink: #172033;
            --vader-muted: #667085;
            --vader-border: #d9dee8;
            --vader-panel: #ffffff;
            --vader-accent: #0075CF;
            --vader-rail: 4rem;
        }
        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-primaryFormSubmit"] {
            color: #ffffff !important;
            background-color: var(--vader-accent) !important;
            border-color: var(--vader-accent) !important;
        }
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButton-primary"]:focus,
        button[data-testid="stBaseButton-primary"]:active,
        button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
        button[data-testid="stBaseButton-primaryFormSubmit"]:focus,
        button[data-testid="stBaseButton-primaryFormSubmit"]:active {
            color: #ffffff !important;
            background-color: #005FA8 !important;
            border-color: #005FA8 !important;
            box-shadow: 0 0 0 1px rgba(0, 117, 207, 0.24) !important;
        }
        input[type="checkbox"] {
            accent-color: var(--vader-accent);
        }
        [data-baseweb="tag"] {
            color: #ffffff !important;
            background-color: var(--vader-accent) !important;
        }
        [data-baseweb="tag"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        label[data-baseweb="checkbox"] input[type="checkbox"]:checked + div,
        label[data-baseweb="checkbox"] input[role="switch"]:checked + div {
            background-color: var(--vader-accent) !important;
            border-color: var(--vader-accent) !important;
        }
        label[data-baseweb="checkbox"] input:checked + div svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        label[data-react-aria-pressable="true"][data-selected="true"]
        > div:first-of-type {
            background-color: var(--vader-accent) !important;
            border-color: var(--vader-accent) !important;
        }
        label[data-react-aria-pressable="true"][data-selected="true"]
        > div:first-of-type svg {
            color: #ffffff !important;
            stroke: #ffffff !important;
        }
        button[aria-expanded="true"] {
            border-color: var(--vader-accent) !important;
            box-shadow: 0 0 0 1px rgba(0, 117, 207, 0.22) !important;
        }
        button[data-testid="stPopoverButton"]:focus,
        button[data-testid="stPopoverButton"]:focus-visible {
            border-color: var(--vader-accent) !important;
            box-shadow: 0 0 0 3px rgba(0, 117, 207, 0.2) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            color: #ffffff !important;
            background-color: var(--vader-accent) !important;
            border-color: var(--vader-accent) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] p {
            color: #ffffff !important;
        }
        div[data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] {
            color: #ffffff !important;
            background-color: var(--vader-accent) !important;
            border-color: var(--vader-accent) !important;
        }
        div[data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] p {
            color: #ffffff !important;
        }        div[class*="_compact_view"] button p {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        div[class*="_compact_view"] button {
            padding-right: 0.35rem;
            padding-left: 0.35rem;
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
        .nav-rail__brand-mark svg {
            display: block;
            width: 2.7rem;
            height: 2.7rem;
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
        .st-key-postprocessing_controls_v5 div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.32rem 0.42rem 0.05rem;
        }
        .st-key-postprocessing_controls_v5 {
            width: 100%;
            max-width: 90rem;
        }
        .st-key-postprocessing_controls_v5 button p,
        .st-key-postprocessing_controls_v5 div[data-testid="stSelectbox"] p {
            font-size: 0.70rem;
        }
        .st-key-postprocessing_controls_v5 .inline-control-label {
            font-size: 0.68rem;
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


WORKSPACE_DATA = "Data"
WORKSPACE_FREQUENCY = "Frequency analysis"
WORKSPACE_POSTPROCESSING = "Postprocessing"
WORKSPACE_SUMMARY = "Summary"
WORKSPACE_DEFINITIONS = (
    (WORKSPACE_DATA, ":material/show_chart:", "processed"),
    (WORKSPACE_FREQUENCY, ":material/graphic_eq:", "frequency"),
    (WORKSPACE_POSTPROCESSING, ":material/calculate:", "postprocessing"),
    (WORKSPACE_SUMMARY, ":material/analytics:", "summary"),
)
LEGACY_WORKSPACE_NAMES = {
    "Filtered / processed": WORKSPACE_DATA,
    "Raw data": WORKSPACE_DATA,
    "Summary plots": WORKSPACE_SUMMARY,
}


def normalize_workspace(workspace: object) -> str:
    candidate = LEGACY_WORKSPACE_NAMES.get(str(workspace), str(workspace))
    valid_names = {definition[0] for definition in WORKSPACE_DEFINITIONS}
    return candidate if candidate in valid_names else WORKSPACE_DATA


def set_workspace(workspace: str) -> None:
    workspace = normalize_workspace(workspace)
    st.session_state["workspace"] = workspace
    st.session_state["_restore_scope"] = next(
        scope
        for name, _, scope in WORKSPACE_DEFINITIONS
        if name == workspace
    )


def render_navigation() -> str:
    current = normalize_workspace(st.session_state.get("workspace", WORKSPACE_DATA))
    st.session_state["workspace"] = current
    with st.container(key="nav_rail"):
        st.markdown(
            f"""
            <div class="nav-rail__brand">
                <span class="nav-rail__brand-mark">
                    {BRAND_ICON_SVG}
                </span>
                <span class="nav-rail__brand-label">VADER Dashboard</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, icon, _ in WORKSPACE_DEFINITIONS:
            st.button(
                label,
                icon=icon,
                key=f"navigate_{label}",
                type="primary" if label == current else "secondary",
                width="stretch",
                help=label,
                on_click=set_workspace,
                args=(label,),
            )
    return current


def render_physical_settings() -> PhysicalSettings:
    defaults = PhysicalSettings()
    values: dict[str, Any] = {}
    with st.container(key="top_physical_settings"):
        with st.popover(
            "Physics",
            icon=":material/tune:",
            width="stretch",
        ):
            current_group: str | None = None
            for specification in PHYSICS_CONTROL_SPECS:
                if (
                    current_group is not None
                    and specification.group != current_group
                ):
                    st.divider()
                current_group = specification.group
                default_value = getattr(defaults, specification.field_name)

                if specification.control == "toggle":
                    value = st.toggle(
                        specification.label,
                        value=bool(default_value),
                        key=specification.key,
                        help=specification.help,
                    )
                elif specification.control == "select":
                    options = list(specification.options)
                    if not options:
                        raise ValueError(
                            f"{specification.field_name} requires select options."
                        )
                    index = (
                        options.index(default_value)
                        if default_value in options
                        else 0
                    )
                    select_arguments: dict[str, Any] = {
                        "label": specification.label,
                        "options": options,
                        "index": index,
                        "key": specification.key,
                        "help": specification.help,
                    }
                    if specification.format_options_as_columns:
                        select_arguments["format_func"] = column_display_label
                    value = st.selectbox(**select_arguments)
                elif specification.control == "number":
                    number_arguments: dict[str, Any] = {
                        "label": specification.label,
                        "value": default_value,
                        "key": specification.key,
                        "help": specification.help,
                    }
                    if specification.min_value is not None:
                        number_arguments["min_value"] = specification.min_value
                    if specification.max_value is not None:
                        number_arguments["max_value"] = specification.max_value
                    if specification.step is not None:
                        number_arguments["step"] = specification.step
                    if specification.number_format is not None:
                        number_arguments["format"] = specification.number_format
                    value = st.number_input(**number_arguments)
                else:
                    raise ValueError(
                        f"Unknown Physics control '{specification.control}'."
                    )
                values[specification.field_name] = value
            st.caption(PHYSICS_PANEL_CAPTION)
    return PhysicalSettings(**values)

def get_data_signature(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    csv_paths = sorted(data_dir.glob("*.csv"))
    return tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
        for path in csv_paths
    )

@st.cache_data(show_spinner=False)
def load_file_summary(
    data_dir_text: str,
    signature: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    del data_dir_text
    records = []
    for path_text, _, _ in signature:
        metadata = parse_filename(Path(path_text))
        records.append(
            {
                "source_file": Path(path_text).name,
                "material": metadata["material"],
                "velocity": metadata["velocity"],
                VELOCITY_COLUMN: parse_velocity_value(metadata["velocity"]),
                "sample": metadata["sample"],
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["source_file", "material", "velocity", VELOCITY_COLUMN, "sample"],
    )


@st.cache_data(show_spinner="Loading selected data")
def load_selected_dataset(
    data_dir_text: str,
    signature: tuple[tuple[str, int, int], ...],
    selected_files: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    del data_dir_text
    selected = set(selected_files)
    frames: list[pd.DataFrame] = []
    issues: list[str] = []

    for path_text, _, _ in signature:
        path = Path(path_text)
        if path.name not in selected:
            continue
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
        frame[VELOCITY_COLUMN] = parse_velocity_value(metadata["velocity"])
        frames.append(frame)

    data = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=REQUIRED_COLUMNS + METADATA_COLUMNS)
    )
    return data, issues

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
        frame[VELOCITY_COLUMN] = parse_velocity_value(metadata["velocity"])

        frames.append(frame)
        file_records.append(
            {
                "source_file": path.name,
                "material": metadata["material"],
                "velocity": metadata["velocity"],
                VELOCITY_COLUMN: parse_velocity_value(metadata["velocity"]),
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
        columns=["source_file", "material", "velocity", VELOCITY_COLUMN, "sample"],
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
    file_summary: pd.DataFrame,
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
    scope: str,
    axis_options: list[str],
    default_plots: list[dict[str, str]],
    allow_processing: bool,
) -> None:
    if st.session_state.get("_restore_scope") == scope:
        for index in range(len(default_plots)):
            restore_plot_controls(scope, index, file_summary, allow_processing)
        st.session_state.pop("_restore_scope", None)

    plot_columns = st.columns(2, gap="small")
    for index, defaults in enumerate(default_plots):
        with plot_columns[index]:
            render_plot_window(
                index, file_summary, defaults, signature,
                physical_settings, scope, axis_options, allow_processing,
            )


def restore_selection_controls(
    selector_scope: str,
    file_summary: pd.DataFrame,
    selected_files: tuple[str, ...],
    selected_materials: tuple[str, ...],
    selected_velocities: tuple[str, ...],
) -> None:
    material_set = set(selected_materials)
    velocity_set = set(selected_velocities)
    file_set = set(selected_files)
    all_materials = sorted(
        file_summary["material"].dropna().astype(str).unique()
    )
    all_velocities = sorted_velocity_values(file_summary["velocity"])
    st.session_state[f"{selector_scope}_data_materials"] = [
        value for value in all_materials if value in material_set
    ]
    st.session_state[f"{selector_scope}_data_velocities"] = [
        value for value in all_velocities if value in velocity_set
    ]
    for material in all_materials:
        st.session_state[f"{selector_scope}_data_material_{material}"] = (
            material in material_set
        )
    for velocity in all_velocities:
        st.session_state[f"{selector_scope}_data_velocity_{velocity}"] = (
            velocity in velocity_set
        )
    for material in file_summary["material"].dropna().astype(str).unique():
        st.session_state[f"{selector_scope}_material_{material}"] = (
            material in material_set
        )
    for velocity in file_summary["velocity"].dropna().astype(str).unique():
        st.session_state[f"{selector_scope}_velocity_{velocity}"] = (
            velocity in velocity_set
        )
    for source_file in file_summary["source_file"].astype(str):
        st.session_state[f"plot_{selector_scope}_file_{source_file}"] = (
            source_file in file_set
        )


def restore_plot_controls(
    scope: str,
    index: int,
    file_summary: pd.DataFrame,
    allow_processing: bool,
) -> None:
    key_prefix = f"{scope}_plot_{index}"
    request = st.session_state.get(f"{key_prefix}_applied_request")
    if request is None or not hasattr(request, "show_legend"):
        return

    st.session_state[f"{key_prefix}_x"] = normalize_axis_column(request.x_column)
    st.session_state[f"{key_prefix}_y"] = normalize_axis_column(request.y_column)
    st.session_state[f"{key_prefix}_show_legend"] = request.show_legend
    st.session_state[f"{key_prefix}_x_scale"] = request.x_scale.title()
    st.session_state[f"{key_prefix}_y_scale"] = request.y_scale.title()
    st.session_state[f"{key_prefix}_scale_for_y"] = request.y_column
    st.session_state[f"{key_prefix}_raw_overlay"] = request.show_raw_overlay
    if allow_processing:
        st.session_state[f"{key_prefix}_force_formula_list"] = "\n".join(
            variant.expression for variant in request.force_variants
        )
        st.session_state[f"{key_prefix}_strain_formula_list"] = "\n".join(
            variant.expression for variant in request.strain_variants
        )
    restore_selection_controls(
        f"{scope}_{index}",
        file_summary,
        request.selected_files,
        request.selected_materials,
        request.selected_velocities,
    )


def render_plot_view_controls(
    key_prefix: str,
    y_column: str,
    allow_raw_overlay: bool = False,
) -> tuple[bool, bool, str, str]:
    show_legend_key = f"{key_prefix}_show_legend"
    raw_overlay_key = f"{key_prefix}_raw_overlay"
    x_scale_key = f"{key_prefix}_x_scale"
    y_scale_key = f"{key_prefix}_y_scale"
    scale_for_y_key = f"{key_prefix}_scale_for_y"

    st.session_state.setdefault(show_legend_key, False)
    st.session_state.setdefault(raw_overlay_key, True)
    st.session_state.setdefault(x_scale_key, "Linear")
    if st.session_state.get(scale_for_y_key) != y_column:
        st.session_state[y_scale_key] = "Log" if uses_log_y(y_column) else "Linear"
        st.session_state[scale_for_y_key] = y_column
    st.session_state.setdefault(y_scale_key, "Linear")

    with st.popover("View", icon=":material/visibility:", width="stretch"):
        if allow_raw_overlay:
            show_raw_overlay = st.toggle(
                "Raw background",
                key=raw_overlay_key,
                help="Show preprocessed raw data behind processed curves.",
            )
        else:
            show_raw_overlay = False
        show_legend = st.toggle("Legend", key=show_legend_key)
        scale_columns = st.columns(2, gap="small")
        with scale_columns[0]:
            x_scale = st.segmented_control(
                "X scale",
                ["Linear", "Log"],
                key=x_scale_key,
                required=True,
                width="stretch",
            )
        with scale_columns[1]:
            y_scale = st.segmented_control(
                "Y scale",
                ["Linear", "Log"],
                key=y_scale_key,
                required=True,
                width="stretch",
            )
    return (
        bool(show_raw_overlay),
        bool(show_legend),
        normalize_axis_scale(str(x_scale)),
        normalize_axis_scale(str(y_scale)),
    )

def _custom_formula_key(key_prefix: str, suffix: str) -> str:
    return f"{key_prefix}_custom_y_{suffix}"


def open_custom_y_formula(key_prefix: str) -> None:
    state_key = f"{key_prefix}_y"
    current = str(st.session_state.get(state_key, ""))
    draft_key = _custom_formula_key(key_prefix, "draft")
    if is_custom_expression_column(current):
        st.session_state[draft_key] = custom_expression_text(current)
    else:
        st.session_state.setdefault(draft_key, "A / D")
    st.session_state[_custom_formula_key(key_prefix, "error")] = ""
    st.session_state["custom_y_dialog_scope"] = key_prefix


def close_custom_y_formula() -> None:
    st.session_state["custom_y_dialog_scope"] = None


def apply_custom_y_formula(key_prefix: str) -> bool:
    draft_key = _custom_formula_key(key_prefix, "draft")
    error_key = _custom_formula_key(key_prefix, "error")
    try:
        column = custom_expression_column(st.session_state.get(draft_key, ""))
    except ValueError as exc:
        st.session_state[error_key] = str(exc)
        return False
    st.session_state[f"{key_prefix}_y"] = column
    st.session_state[error_key] = ""
    st.session_state["custom_y_dialog_scope"] = None
    return True


def build_custom_y_formula(key_prefix: str) -> None:
    left = str(st.session_state[_custom_formula_key(key_prefix, "left")])
    operation = str(st.session_state[_custom_formula_key(key_prefix, "operator")])
    right = str(st.session_state[_custom_formula_key(key_prefix, "right")])
    if right == "Number":
        right = f"{float(st.session_state[_custom_formula_key(key_prefix, 'number')]):g}"
    st.session_state[_custom_formula_key(key_prefix, "draft")] = (
        f"{left} {operation} {right}"
    )


def wrap_custom_y_formula(key_prefix: str) -> None:
    draft_key = _custom_formula_key(key_prefix, "draft")
    function = str(st.session_state[_custom_formula_key(key_prefix, "function")])
    current = str(st.session_state.get(draft_key, "")).strip() or "A / D"
    st.session_state[draft_key] = f"{function}({current})"


@st.dialog("Custom Y formula", width="large")
def render_custom_y_formula_dialog(key_prefix: str) -> None:
    draft_key = _custom_formula_key(key_prefix, "draft")
    st.session_state.setdefault(draft_key, "A / D")
    builder = st.columns(
        [1.0, 0.65, 1.0, 0.72, 0.55],
        gap="small",
        vertical_alignment="bottom",
    )
    with builder[0]:
        st.selectbox(
            "Left",
            CUSTOM_EXPRESSION_TOKENS,
            index=CUSTOM_EXPRESSION_TOKENS.index("A"),
            key=_custom_formula_key(key_prefix, "left"),
        )
    with builder[1]:
        st.selectbox(
            "Operator",
            ["+", "-", "*", "/", "**"],
            index=3,
            key=_custom_formula_key(key_prefix, "operator"),
        )
    with builder[2]:
        st.selectbox(
            "Right",
            (*CUSTOM_EXPRESSION_TOKENS, "Number"),
            index=CUSTOM_EXPRESSION_TOKENS.index("D"),
            key=_custom_formula_key(key_prefix, "right"),
        )
    with builder[3]:
        st.number_input(
            "Number",
            value=1.0,
            format="%.6g",
            key=_custom_formula_key(key_prefix, "number"),
        )
    with builder[4]:
        st.button(
            "Use",
            key=_custom_formula_key(key_prefix, "build"),
            on_click=build_custom_y_formula,
            args=(key_prefix,),
            width="stretch",
        )

    function_row = st.columns(
        [1.0, 0.55, 3.0], gap="small", vertical_alignment="bottom"
    )
    with function_row[0]:
        st.selectbox(
            "Function",
            tuple(CUSTOM_EXPRESSION_FUNCTIONS),
            key=_custom_formula_key(key_prefix, "function"),
        )
    with function_row[1]:
        st.button(
            "Wrap",
            key=_custom_formula_key(key_prefix, "wrap"),
            on_click=wrap_custom_y_formula,
            args=(key_prefix,),
            width="stretch",
        )

    st.text_area(
        "Formula",
        key=draft_key,
        height=90,
        help=(
            "Symbols: t, Lv, HS, eps_z, D, D0, F, A, sig, sig_surf, "
            "delta_sig, HS_rate, visc_e, v, pi, ST. ST uses mN/m."
        ),
    )
    error = str(st.session_state.get(_custom_formula_key(key_prefix, "error"), ""))
    if error:
        st.error(error)
    actions = st.columns([1.0, 1.0, 3.0], gap="small")
    apply_clicked = actions[0].button(
        "Apply",
        type="primary",
        key=_custom_formula_key(key_prefix, "apply"),
        width="stretch",
    )
    cancel_clicked = actions[1].button(
        "Cancel",
        key=_custom_formula_key(key_prefix, "cancel"),
        width="stretch",
    )
    if apply_clicked:
        if apply_custom_y_formula(key_prefix):
            st.rerun()
        st.rerun(scope="fragment")
    if cancel_clicked:
        close_custom_y_formula()
        st.rerun()


def render_axis_selector(
    key_prefix: str,
    axis_name: str,
    axis_options: list[str],
) -> str:
    custom_y_enabled = axis_name.upper() == "Y"
    row = st.columns(
        [0.18, 1, 0.25] if custom_y_enabled else [0.18, 1],
        gap="small",
        vertical_alignment="center",
    )
    state_key = f"{key_prefix}_{axis_name.lower()}"
    if state_key in st.session_state:
        st.session_state[state_key] = normalize_axis_column(
            st.session_state[state_key]
        )
    options = list(axis_options)
    current = str(st.session_state.get(state_key, ""))
    if is_custom_expression_column(current) and current not in options:
        options.append(current)

    with row[0]:
        render_inline_label(f"{axis_name}:")
    with row[1]:
        selected = st.selectbox(
            f"{axis_name} axis",
            options,
            index=None,
            key=state_key,
            label_visibility="collapsed",
            format_func=column_menu_label,
        )
    if custom_y_enabled:
        with row[2]:
            st.button(
                "",
                icon=":material/functions:",
                key=f"{key_prefix}_open_custom_y",
                help="Define a custom Y formula",
                on_click=open_custom_y_formula,
                args=(key_prefix,),
                width="stretch",
            )
        if st.session_state.get("custom_y_dialog_scope") == key_prefix:
            render_custom_y_formula_dialog(key_prefix)
    return str(selected)

def render_plot_window(
    index: int,
    file_summary: pd.DataFrame,
    defaults: dict[str, str],
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
    scope: str,
    axis_options: list[str],
    allow_processing: bool,
) -> None:
    key_prefix = f"{scope}_plot_{index}"
    applied_key = f"{key_prefix}_applied_request"
    with st.container(border=True):
        st.markdown(
            f'<div class="plot-heading">Plot {index + 1}</div>',
            unsafe_allow_html=True,
        )
        force_variants = parse_filter_formula_list("force_g", "force_g")
        strain_variants = parse_filter_formula_list("HS", "radial_Hencky_strain")
        show_raw_overlay = False
        st.session_state.setdefault(f"{key_prefix}_x", defaults["x"])
        st.session_state.setdefault(f"{key_prefix}_y", defaults["y"])

        selector_scope = f"{scope}_{index}"
        materials = sorted(
            file_summary["material"].dropna().astype(str).unique()
        )
        velocities = sorted_velocity_values(file_summary["velocity"])
        group_widths = (
            [1.08, 0.95, 1.08, 1.02]
            if allow_processing
            else [1.08, 0.95, 1.02]
        )
        control_groups = st.columns(
            group_widths,
            gap="small",
            vertical_alignment="top",
        )

        with control_groups[0]:
            x_column = render_axis_selector(
                key_prefix, "X", axis_options
            )
            y_column = render_axis_selector(
                key_prefix, "Y", axis_options
            )

        with control_groups[1]:
            selected_materials = render_filter_dropdown(
                selector_scope, "Material", materials
            )
            selected_velocities = render_filter_dropdown(
                selector_scope, "Velocity", velocities
            )
        eligible = file_summary[
            file_summary["material"].astype(str).isin(selected_materials)
            & file_summary["velocity"].astype(str).isin(selected_velocities)
        ]

        if allow_processing:
            with control_groups[2]:
                strain_variants = render_formula_variant_controls(
                    f"{key_prefix}_strain",
                    "εᵣ",
                    "radial_Hencky_strain",
                    default_formula=DEFAULT_STRAIN_FORMULA,
                )
                force_variants = render_formula_variant_controls(
                    f"{key_prefix}_force", "F", "force_g"
                )
            action_group = control_groups[3]
        else:
            action_group = control_groups[2]

        with action_group:
            selected_files, runs_done = render_data_selector(
                selector_scope, eligible
            )
            action_columns = st.columns(
                [1.7, 0.55],
                gap="small",
                vertical_alignment="center",
            )
            with action_columns[0]:
                with st.container(key=f"{key_prefix}_compact_view"):
                    if allow_processing:
                        (
                            show_raw_overlay,
                            show_legend,
                            x_scale,
                            y_scale,
                        ) = render_plot_view_controls(
                            key_prefix, y_column, allow_raw_overlay=True
                        )
                    else:
                        (
                            _,
                            show_legend,
                            x_scale,
                            y_scale,
                        ) = render_plot_view_controls(key_prefix, y_column)
            with action_columns[1]:
                update_requested = st.button(
                    "",
                    key=f"{key_prefix}_update",
                    type="primary",
                    icon=":material/refresh:",
                    help="Update this plot",
                    width="stretch",
                )
        draft_request = PlotRequest(
            x_column=str(x_column),
            y_column=str(y_column),
            selected_files=tuple(selected_files),
            force_variants=force_variants,
            strain_variants=strain_variants,
            show_raw_overlay=bool(show_raw_overlay),
            show_legend=bool(show_legend),
            x_scale=x_scale,
            y_scale=y_scale,
            selected_materials=tuple(selected_materials),
            selected_velocities=tuple(selected_velocities),
            physical_settings=physical_settings,
        )
        if update_requested or runs_done:
            st.session_state[applied_key] = draft_request

        applied_request = st.session_state.get(applied_key)
        if applied_request is not None and not hasattr(applied_request, "show_legend"):
            applied_request = None
        if applied_request is None:
            render_empty_plot_figure(
                x_column,
                y_column,
                x_scale,
                y_scale,
                key=f"{key_prefix}_empty",
            )
            return
        if plot_analysis_key(draft_request) != plot_analysis_key(applied_request):
            st.caption("Controls changed. Press Update to apply them.")

        applied_request = replace(
            applied_request,
            show_raw_overlay=bool(show_raw_overlay),
            show_legend=bool(show_legend),
            x_scale=x_scale,
            y_scale=y_scale,
        )
        st.session_state[applied_key] = applied_request
        if not applied_request.selected_files:
            render_empty_plot_figure(
                normalize_axis_column(applied_request.x_column),
                normalize_axis_column(applied_request.y_column),
                applied_request.x_scale,
                applied_request.y_scale,
                key=f"{key_prefix}_empty",
            )
            return

        x_column = normalize_axis_column(applied_request.x_column)
        y_column = normalize_axis_column(applied_request.y_column)
        selected_files = list(applied_request.selected_files)
        force_variants = applied_request.force_variants
        strain_variants = applied_request.strain_variants
        show_raw_overlay = applied_request.show_raw_overlay
        show_legend = applied_request.show_legend
        x_scale = applied_request.x_scale
        y_scale = applied_request.y_scale
        applied_physics = applied_request.physical_settings

        raw_data, issues = load_selected_dataset(
            str(DATA_DIR), signature, tuple(selected_files)
        )
        for issue in issues:
            st.warning(issue)
        if raw_data.empty:
            st.info("No compatible rows were loaded for the applied data series.")
            return

        if not allow_processing:
            plot_frame = build_plot_frame(raw_data, x_column, y_column)
            if plot_frame.empty:
                st.info("No data matches the applied selection.")
                return
            render_plot_figure(
                plot_frame,
                x_column,
                y_column,
                y_column,
                show_legend=show_legend,
                x_scale=x_scale,
                y_scale=y_scale,
                download_key=f"{key_prefix}_raw",
            )
            return

        preprocessed = preprocess_dataset(raw_data, applied_physics)
        if preprocessed.warnings:
            with st.expander(f"Preprocessing warnings ({len(preprocessed.warnings)})"):
                for warning in preprocessed.warnings:
                    st.caption(warning)

        raw_frame = derive_dataset(preprocessed.frame, applied_physics)
        try:
            raw_frame = add_custom_expression_column(
                raw_frame, y_column, applied_physics
            )
        except ValueError as exc:
            st.error(f"Custom Y formula failed: {exc}")
            return
        raw_plot_frame = build_plot_frame(raw_frame, x_column, y_column)
        if raw_plot_frame.empty:
            st.info("No data matches the applied selection.")
            return

        try:
            force_variants, strain_variants = align_formula_variants(
                force_variants, strain_variants
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        has_processing = (
            len(force_variants) > 1
            or len(strain_variants) > 1
            or any(variant.settings.active for variant in force_variants)
            or any(variant.settings.active for variant in strain_variants)
        )
        if not has_processing:
            render_plot_figure(
                raw_plot_frame,
                x_column,
                y_column,
                y_column,
                show_legend=show_legend,
                x_scale=x_scale,
                y_scale=y_scale,
                download_key=f"{key_prefix}_preprocessed",
            )
            return

        job_key = make_job_key(
            scope, index, signature, applied_physics,
            force_variants, strain_variants, x_column, y_column,
            tuple(selected_files),
        )
        render_background_processed_plot(
            job_key,
            preprocessed.frame,
            raw_plot_frame,
            x_column,
            y_column,
            force_variants,
            strain_variants,
            applied_physics,
            show_raw_overlay,
            show_legend,
            x_scale,
            y_scale,
            download_key=f"{key_prefix}_processed",
        )

def render_inline_label(text: str) -> None:
    st.markdown(
        f'<div class="inline-control-label">{text}</div>',
        unsafe_allow_html=True,
    )

def render_filter_dropdown(
    index: int | str,
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


def render_formula_variant_controls(
    scope: str,
    label: str,
    source_column: str,
    default_formula: str | None = None,
) -> tuple[FormulaVariant, ...]:
    formula_key = f"{scope}_formula_list"
    editor_key = f"{scope}_edit_variant"
    if formula_key not in st.session_state:
        st.session_state[formula_key] = (
            default_formula or formula_source_label(source_column)
        )

    try:
        variants = parse_filter_formula_list(
            str(st.session_state[formula_key]), source_column
        )
        valid = True
    except ValueError:
        variants = parse_filter_formula_list(source_column, source_column)
        valid = False

    if not valid:
        button_label = f"{label}: Invalid"
    elif len(variants) == 1:
        button_label = f"{label}: {variants[0].label}"
    else:
        button_label = f"{label}: {len(variants)} variants"

    with st.popover(button_label, width="stretch"):
        formula_text = st.text_area(
            "Formula variants",
            key=formula_key,
            height=86,
            help=(
                "One formula per line. Semicolons and top-level commas are also "
                "accepted. Shorthand like LP(20) means LP(signal,20)."
            ),
        )
        try:
            variants = parse_filter_formula_list(formula_text, source_column)
        except ValueError as exc:
            st.error(str(exc))
            return parse_filter_formula_list(
                formula_source_label(source_column), source_column
            )

        variant_indices = list(range(len(variants)))
        if st.session_state.get(editor_key) not in variant_indices:
            st.session_state[editor_key] = 0
        selected_index = st.selectbox(
            "Edit formula",
            variant_indices,
            key=editor_key,
            format_func=lambda index: variants[index].label,
        )
        filter_name = st.segmented_control(
            "Filter",
            [specification.ui_label for specification in FILTER_CONTROL_SPECS],
            default=FILTER_CONTROL_SPECS[0].ui_label,
            required=True,
            key=f"{scope}_new_filter",
            width="stretch",
        )
        new_step = render_filter_step_options(
            f"{scope}_builder", str(filter_name)
        )
        actions = st.columns(
            [1.15, 1.35, 0.62, 0.62, 0.62], gap="small"
        )
        actions[0].button(
            "Apply",
            key=f"{scope}_apply_filter",
            width="stretch",
            disabled=new_step is None,
            on_click=update_formula_variant_list,
            args=(
                formula_key,
                editor_key,
                source_column,
                "apply",
                int(selected_index),
                new_step,
            ),
        )
        actions[1].button(
            "Compare",
            key=f"{scope}_compare_filter",
            width="stretch",
            disabled=new_step is None,
            on_click=update_formula_variant_list,
            args=(
                formula_key,
                editor_key,
                source_column,
                "compare",
                int(selected_index),
                new_step,
            ),
        )
        actions[2].button(
            "",
            icon=":material/undo:",
            help="Remove the outermost filter from this formula",
            key=f"{scope}_undo_filter",
            width="stretch",
            disabled=not variants[int(selected_index)].settings.active,
            on_click=update_formula_variant_list,
            args=(
                formula_key,
                editor_key,
                source_column,
                "undo",
                int(selected_index),
                None,
            ),
        )
        actions[3].button(
            "",
            icon=":material/restart_alt:",
            help="Reset this formula to the raw signal",
            key=f"{scope}_reset_formula",
            width="stretch",
            disabled=not variants[int(selected_index)].settings.active,
            on_click=update_formula_variant_list,
            args=(
                formula_key,
                editor_key,
                source_column,
                "reset",
                int(selected_index),
                None,
            ),
        )
        actions[4].button(
            "",
            icon=":material/delete:",
            help="Delete this formula variant",
            key=f"{scope}_delete_formula",
            width="stretch",
            disabled=len(variants) <= 1,
            on_click=update_formula_variant_list,
            args=(
                formula_key,
                editor_key,
                source_column,
                "delete",
                int(selected_index),
                None,
            ),
        )
    return variants


def update_formula_variant_list(
    formula_key: str,
    editor_key: str,
    source_column: str,
    action: str,
    selected_index: int,
    step: FilterStep | None,
) -> None:
    try:
        variants = list(parse_filter_formula_list(
            str(st.session_state.get(formula_key, source_column)),
            source_column,
        ))
    except ValueError:
        variants = list(parse_filter_formula_list(source_column, source_column))
    selected_index = min(max(selected_index, 0), len(variants) - 1)
    expressions = [variant.expression for variant in variants]
    parsed_source, steps = parse_filter_formula(expressions[selected_index])
    target_expression = expressions[selected_index]

    if action in {"apply", "compare"} and step is not None:
        target_expression = format_filter_formula(
            formula_source_label(parsed_source), (*steps, step)
        )
        if action == "apply":
            expressions[selected_index] = target_expression
        else:
            expressions.insert(selected_index + 1, target_expression)
    elif action == "undo" and steps:
        target_expression = format_filter_formula(
            formula_source_label(parsed_source), steps[:-1]
        )
        expressions[selected_index] = target_expression
    elif action == "reset":
        target_expression = formula_source_label(parsed_source)
        expressions[selected_index] = target_expression
    elif action == "delete" and len(expressions) > 1:
        expressions.pop(selected_index)
        target_expression = expressions[min(selected_index, len(expressions) - 1)]

    normalized = parse_filter_formula_list(
        "\n".join(expressions), source_column
    )
    st.session_state[formula_key] = "\n".join(
        variant.expression for variant in normalized
    )
    st.session_state[editor_key] = next(
        (
            index
            for index, variant in enumerate(normalized)
            if variant.expression == target_expression
        ),
        0,
    )

def render_processing_controls(
    scope: str,
    source_column: str,
    axis_label: str = "Formula",
    default_formula: str | None = None,
    show_axis_label: bool = True,
) -> FilterSettings:
    formula_key = f"{scope}_filter_formula"
    source_key = f"{scope}_filter_source"
    if formula_key not in st.session_state:
        st.session_state[formula_key] = (
            default_formula or formula_source_label(source_column)
        )
    if st.session_state.get(source_key) != source_column:
        try:
            _, existing_steps = parse_filter_formula(st.session_state[formula_key])
        except ValueError:
            existing_steps = ()
        st.session_state[formula_key] = format_filter_formula(
            formula_source_label(source_column), existing_steps
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

    workflow = ()

    if not formula_valid:
        status_label = "Invalid"
    elif current_steps:
        suffix = "filter" if len(current_steps) == 1 else "filters"
        status_label = f"{len(current_steps)} {suffix}"
    else:
        status_label = "Raw"
    button_label = (
        f"{axis_label}: {status_label}" if show_axis_label else status_label
    )

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
                st.error(f"Formula must start from {column_display_label(source_column)}.")
            else:
                st.caption(f"Applied order: {format_filter_workflow(workflow)}")

        filter_name = st.segmented_control(
            "Add filter",
            [specification.ui_label for specification in FILTER_CONTROL_SPECS],
            default=FILTER_CONTROL_SPECS[0].ui_label,
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
    specification = FILTER_SPEC_BY_LABEL.get(filter_name)
    if specification is None:
        st.error(f"Unknown filter '{filter_name}'.")
        return None

    parameters: list[float] = []
    input_columns = (
        st.columns(2, gap="small")
        if len(specification.parameters) > 1
        else None
    )
    for index, parameter in enumerate(specification.parameters):
        target = (
            input_columns[index % len(input_columns)]
            if input_columns is not None
            else st
        )
        if parameter.integer:
            input_arguments: dict[str, Any] = {
                "label": parameter.label,
                "value": int(parameter.default),
                "step": int(parameter.step),
                "key": f"{scope}_{parameter.key_suffix}",
            }
            if parameter.min_value is not None:
                input_arguments["min_value"] = int(parameter.min_value)
            if parameter.max_value is not None:
                input_arguments["max_value"] = int(parameter.max_value)
        else:
            input_arguments = {
                "label": parameter.label,
                "value": float(parameter.default),
                "step": float(parameter.step),
                "key": f"{scope}_{parameter.key_suffix}",
            }
            if parameter.min_value is not None:
                input_arguments["min_value"] = float(parameter.min_value)
            if parameter.max_value is not None:
                input_arguments["max_value"] = float(parameter.max_value)
        parameters.append(float(target.number_input(**input_arguments)))

    try:
        return _validated_step(specification.operation, tuple(parameters))
    except ValueError as exc:
        st.error(str(exc))
        return None

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
    st.session_state[formula_key] = format_filter_formula(formula_source_label(parsed_source), steps)


def numerically_sorted_file_summary(file_summary: pd.DataFrame) -> pd.DataFrame:
    if file_summary.empty:
        return file_summary
    ordered = file_summary.copy()
    ordered["_velocity_sort"] = ordered["velocity"].map(parse_velocity_value)
    ordered["_velocity_text"] = ordered["velocity"].astype(str).str.casefold()
    return ordered.sort_values(
        ["material", "_velocity_sort", "_velocity_text", "sample", "source_file"],
        na_position="last",
    ).drop(columns=["_velocity_sort", "_velocity_text"])


def render_data_selector(
    index: int | str,
    file_summary: pd.DataFrame,
    default_selected: bool = False,
) -> tuple[list[str], bool]:
    ordered_summary = numerically_sorted_file_summary(file_summary)
    files = ordered_summary["source_file"].astype(str).tolist()
    for file_name in files:
        key = f"plot_{index}_file_{file_name}"
        if key not in st.session_state:
            st.session_state[key] = default_selected

    selected_count = sum(
        bool(st.session_state[f"plot_{index}_file_{file_name}"])
        for file_name in files
    )
    popover_key = f"plot_{index}_data_popover"
    done_requested = False
    with st.popover(
        f"Data {selected_count}/{len(files)}",
        icon=":material/dataset:",
        help="Choose data series",
        width="stretch",
        key=popover_key,
        on_change="rerun",
    ):
        with st.form(f"plot_{index}_data_form", border=False):
            if files:
                checkbox_columns = st.columns(2, gap="small")
                for file_index, row in ordered_summary.reset_index(
                    drop=True
                ).iterrows():
                    file_name = str(row["source_file"])
                    label = get_file_label(row)
                    with checkbox_columns[file_index % len(checkbox_columns)]:
                        st.checkbox(
                            label,
                            key=f"plot_{index}_file_{file_name}",
                            help=file_name,
                        )
            else:
                st.caption("No data matches the current filters.")

            action_columns = st.columns(3, gap="small")
            action_columns[0].form_submit_button(
                "All",
                on_click=set_file_selection,
                args=(index, files, True),
                disabled=not files,
                width="stretch",
            )
            action_columns[1].form_submit_button(
                "None",
                on_click=set_file_selection,
                args=(index, files, False),
                disabled=not files,
                width="stretch",
            )
            done_requested = action_columns[2].form_submit_button(
                "Done",
                type="primary",
                on_click=close_data_selector,
                args=(popover_key,),
                width="stretch",
            )

    selected_files = [
        file_name
        for file_name in files
        if st.session_state[f"plot_{index}_file_{file_name}"]
    ]
    return selected_files, bool(done_requested)

def close_data_selector(popover_key: str) -> None:
    st.session_state[popover_key] = False


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
    optional = [
        column for column in (VARIANT_COLUMN, TRACE_COLUMN)
        if column in data.columns
    ]
    columns = ["time_from_onset_s", x_column, y_column, *METADATA_COLUMNS, *optional]
    frame = data.loc[:, list(dict.fromkeys(columns))]
    frame = frame.dropna(subset=["time_from_onset_s", x_column, y_column])
    sort_columns = ["source_file", "time_from_onset_s"]
    if VARIANT_COLUMN in frame.columns:
        sort_columns.insert(0, VARIANT_COLUMN)
    return frame.sort_values(sort_columns).copy()


def downsample_by_trace(plot_frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in plot_frame.groupby("source_file", sort=False, dropna=False):
        step = max(1, math.ceil(len(group) / max_points))
        pieces.append(group.iloc[::step])
    return pd.concat(pieces, ignore_index=True) if pieces else plot_frame



def render_background_processed_plot(
    job_key: str,
    preprocessed_frame: pd.DataFrame,
    raw_plot_frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    force_variants: tuple[FormulaVariant, ...],
    strain_variants: tuple[FormulaVariant, ...],
    physical_settings: PhysicalSettings,
    show_raw_overlay: bool,
    show_legend: bool,
    x_scale: str,
    y_scale: str,
    download_key: str,
) -> None:
    future = submit_background_job(
        job_key,
        process_measured_variant_frames,
        preprocessed_frame,
        force_variants,
        strain_variants,
        physical_settings,
    )
    if future.done():
        try:
            result: ProcessedFrame = future.result()
        except Exception as exc:  # pragma: no cover
            st.error(f"Processing failed: {exc}")
            render_plot_figure(
                raw_plot_frame,
                x_column,
                y_column,
                y_column,
                show_legend=show_legend,
                x_scale=x_scale,
                y_scale=y_scale,
            )
            return
        try:
            processed_frame = add_custom_expression_column(
                result.frame, y_column, physical_settings
            )
        except ValueError as exc:
            st.error(f"Custom Y formula failed: {exc}")
            return
        processed_plot_frame = build_plot_frame(processed_frame, x_column, y_column)
        if processed_plot_frame.empty:
            st.info("Processed data is empty for the current plot selection.")
            return
        render_processed_plot_figure(
            processed_plot_frame,
            x_column,
            y_column,
            y_column,
            raw_plot_frame if show_raw_overlay else None,
            show_legend=show_legend,
            x_scale=x_scale,
            y_scale=y_scale,
            download_key=download_key,
        )
        render_processing_notes(result)
        return

    @st.fragment(run_every=0.75)
    def pending_processed_plot() -> None:
        if normalize_workspace(st.session_state.get("workspace")) != WORKSPACE_DATA:
            st.rerun()
        if future.done():
            st.rerun()
        st.caption("Processing selected data in background...")
        render_plot_figure(
            raw_plot_frame,
            x_column,
            y_column,
            y_column,
            show_legend=show_legend,
            x_scale=x_scale,
            y_scale=y_scale,
        )

    pending_processed_plot()


def prepare_frame_for_axis_scales(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_scale: str,
    y_scale: str,
) -> pd.DataFrame:
    x_log = normalize_axis_scale(x_scale) == "log"
    y_log = normalize_axis_scale(y_scale) == "log"
    if not x_log and not y_log:
        return frame
    prepared = frame.copy()
    if x_log and x_column in prepared.columns:
        x_values = pd.to_numeric(prepared[x_column], errors="coerce")
        prepared[x_column] = x_values.where(x_values > 0)
    if y_log and y_column in prepared.columns:
        y_values = pd.to_numeric(prepared[y_column], errors="coerce")
        prepared[y_column] = y_values.where(y_values > 0)
    return prepared


def render_processed_plot_figure(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    y_label: str,
    raw_frame: pd.DataFrame | None = None,
    x_label: str | None = None,
    show_legend: bool = True,
    x_scale: str = "linear",
    y_scale: str = "linear",
    download_key: str | None = None,
) -> None:
    figure = make_processed_figure(
        frame,
        x_column,
        y_column,
        y_label,
        raw_frame,
        x_label,
        show_legend,
        x_scale,
        y_scale,
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())
    if download_key is not None:
        render_plot_download(
            frame,
            x_column,
            y_column,
            download_key,
            primary_kind="processed",
            raw_frame=raw_frame,
            x_scale=x_scale,
            y_scale=y_scale,
        )


def make_processed_figure(
    plot_frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    y_label: str,
    raw_frame: pd.DataFrame | None = None,
    x_label: str | None = None,
    show_legend: bool = True,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> go.Figure:
    plot_frame = prepare_frame_for_axis_scales(
        plot_frame, x_column, y_column, x_scale, y_scale
    )
    plot_frame = add_series_labels(plot_frame)
    if raw_frame is not None:
        raw_frame = prepare_frame_for_axis_scales(
            raw_frame, x_column, y_column, x_scale, y_scale
        )
    figure = go.Figure()
    if raw_frame is not None and not raw_frame.empty:
        for _, group in raw_frame.groupby("source_file", sort=False, dropna=False):
            ordered = group.sort_values("time_from_onset_s")
            figure.add_trace(
                go.Scatter(
                    x=ordered[x_column],
                    y=ordered[y_column],
                    mode="lines",
                    line={"color": "rgba(90, 96, 110, 0.28)", "width": 1.0},
                    name="Raw",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    source_values = (
        plot_frame["source_file"].astype(str).drop_duplicates().tolist()
    )
    source_colors = {
        source_file: IFF_PALETTE[index % len(IFF_PALETTE)]
        for index, source_file in enumerate(source_values)
    }
    variant_dashes = ("solid", "dash", "dot", "dashdot", "longdash")
    variant_values = (
        plot_frame[VARIANT_COLUMN].dropna().unique().tolist()
        if VARIANT_COLUMN in plot_frame.columns
        else ["Processed"]
    )
    trace_columns = ["source_file"]
    if VARIANT_COLUMN in plot_frame.columns:
        trace_columns.append(VARIANT_COLUMN)
    for variant_index, variant in enumerate(variant_values):
        variant_frame = (
            plot_frame[plot_frame[VARIANT_COLUMN] == variant]
            if VARIANT_COLUMN in plot_frame.columns
            else plot_frame
        )
        dash = variant_dashes[variant_index % len(variant_dashes)]
        for source_file, group in variant_frame.groupby(
            "source_file", sort=False, dropna=False
        ):
            ordered = group.sort_values("time_from_onset_s")
            color = source_colors[str(source_file)]
            series_label = str(ordered["series_label"].iloc[0])
            trace_name = (
                f"{series_label} | {variant}"
                if len(variant_values) > 1
                else series_label
            )
            figure.add_trace(
                go.Scatter(
                    x=ordered[x_column],
                    y=ordered[y_column],
                    mode="lines",
                    line={"color": color, "width": 1.75, "dash": dash},
                    name=trace_name,
                    legendgroup=str(source_file),
                    showlegend=show_legend,
                    customdata=np.stack(
                        [
                            ordered["material"].astype(str),
                            ordered["velocity"].astype(str),
                            ordered["sample"].astype(str),
                            np.full(len(ordered), str(source_file)),
                            np.full(len(ordered), str(variant)),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{customdata[3]}<br>"
                        "%{customdata[4]}<br>"
                        f"{column_display_label(x_label or x_column)}: %{{x:.4g}}<br>"
                        f"{column_display_label(y_label)}: %{{y:.4g}}<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        height=DATA_PLOT_HEIGHT,
        margin={"l": 8, "r": 8, "t": 15, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=show_legend,
        hovermode="closest",
        font={"color": "#344054", "size": 11},
        legend={"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 9}},
    )
    figure.update_xaxes(
        title=column_axis_title(x_label or x_column),
        type=normalize_axis_scale(x_scale),
    )
    figure.update_yaxes(
        title=column_axis_title(y_label),
        type=normalize_axis_scale(y_scale),
    )
    style_axes(figure)
    return figure


def make_empty_plot_figure(
    x_column: str,
    y_column: str,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        height=DATA_PLOT_HEIGHT,
        margin={"l": 8, "r": 8, "t": 15, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        font={"color": "#344054", "size": 11},
    )
    figure.update_xaxes(
        title=column_axis_title(x_column),
        type=normalize_axis_scale(x_scale),
    )
    figure.update_yaxes(
        title=column_axis_title(y_column),
        type=normalize_axis_scale(y_scale),
    )
    style_axes(figure)
    return figure


def render_empty_plot_figure(
    x_column: str,
    y_column: str,
    x_scale: str,
    y_scale: str,
    key: str,
) -> None:
    st.plotly_chart(
        make_empty_plot_figure(x_column, y_column, x_scale, y_scale),
        width="stretch",
        config=plotly_config(),
        key=key,
    )


def render_plot_figure(
    frame: pd.DataFrame,
    x_column: str,
    plot_y_column: str,
    y_label: str,
    x_label: str | None = None,
    show_legend: bool = True,
    x_scale: str = "linear",
    y_scale: str = "linear",
    download_key: str | None = None,
) -> None:
    figure = make_figure(
        downsample_by_trace(frame, 5_000),
        x_column,
        plot_y_column,
        y_label,
        x_label=x_label,
        show_legend=show_legend,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    st.plotly_chart(figure, width="stretch", config=plotly_config())
    if download_key is not None:
        render_plot_download(
            frame,
            x_column,
            plot_y_column,
            download_key,
            primary_kind="raw",
            x_scale=x_scale,
            y_scale=y_scale,
        )


def render_plot_download(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    download_key: str,
    primary_kind: str,
    raw_frame: pd.DataFrame | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> None:
    pieces: list[pd.DataFrame] = []
    for trace_kind, source in (
        (primary_kind, frame),
        ("raw_background", raw_frame),
    ):
        if source is None or source.empty:
            continue
        prepared = prepare_frame_for_axis_scales(
            source, x_column, y_column, x_scale, y_scale
        ).dropna(subset=[x_column, y_column])
        columns = [
            column for column in (
                "source_file", "material", "velocity", VELOCITY_COLUMN, "sample",
                VARIANT_COLUMN, x_column, y_column,
            )
            if column in prepared.columns
        ]
        piece = prepared.loc[:, list(dict.fromkeys(columns))].copy()
        piece.insert(0, "trace_type", trace_kind)
        pieces.append(piece)

    if not pieces:
        return
    export_frame = pd.concat(pieces, ignore_index=True)
    st.download_button(
        "CSV",
        data=export_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"vader_{download_key}.csv",
        mime="text/csv",
        key=f"{download_key}_csv",
        icon=":material/download:",
        help="Download the full-resolution data currently shown in this plot.",
        width="content",
    )

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
    show_legend: bool = True,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> Any:
    plot_frame = prepare_frame_for_axis_scales(
        plot_frame, x_column, y_column, x_scale, y_scale
    )
    plot_frame = add_series_labels(plot_frame)
    figure = px.line(
        plot_frame,
        x=x_column,
        y=y_column,
        color="series_label",
        line_group="source_file",
        color_discrete_sequence=IFF_PALETTE,
        hover_data={
            "material": True,
            "velocity": True,
            "sample": True,
            "source_file": True,
        },
        labels={
            x_column: column_axis_title(x_label or x_column),
            y_column: column_axis_title(y_label),
            "series_label": "Data series",
        },
    )
    figure.update_layout(
        height=DATA_PLOT_HEIGHT,
        margin={"l": 8, "r": 8, "t": 15, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=show_legend,
        hovermode="closest",
        font={"color": "#344054", "size": 11},
        legend={"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 9}},
    )
    style_axes(figure)
    figure.update_xaxes(
        title=column_axis_title(x_label or x_column),
        type=normalize_axis_scale(x_scale),
    )
    figure.update_yaxes(
        title=column_axis_title(y_label),
        type=normalize_axis_scale(y_scale),
    )
    figure.update_traces(line={"width": 1.6})
    return figure

def add_series_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    label_columns = [
        column for column in ("material", "velocity", "sample")
        if column in labeled.columns
    ]
    if label_columns:
        labeled["series_label"] = (
            labeled[label_columns].astype(str).agg(" | ".join, axis=1)
        )
    else:
        labeled["series_label"] = labeled["source_file"].astype(str)
    return labeled


def uses_log_y(y_label: str) -> bool:
    return y_label in LOG_Y_COLUMNS


def prepare_y_for_plot_scale(
    frame: pd.DataFrame,
    y_column: str,
    log_y: bool,
) -> pd.DataFrame:
    if not log_y or y_column not in frame.columns:
        return frame
    prepared = frame.copy()
    values = pd.to_numeric(prepared[y_column], errors="coerce")
    prepared[y_column] = values.where(values > 0)
    return prepared


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
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png", "filename": "VADER_plot", "scale": 2,
        },
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


def restore_postprocessing_controls(file_summary: pd.DataFrame) -> None:
    request = st.session_state.get("postprocessing_applied_request")
    required_fields = (
        "r2_threshold",
        "epsilon_start",
        "y_settings",
        "y_scale",
        "show_window",
        "show_slope",
    )
    if request is None or not all(
        hasattr(request, field_name) for field_name in required_fields
    ):
        return
    st.session_state["postprocessing_y"] = normalize_axis_column(
        request.y_column
    )
    st.session_state["postprocessing_y_formula_filter_formula"] = (
        format_filter_formula(
            formula_source_label(request.y_column),
            request.y_settings.workflow,
        )
    )
    st.session_state["postprocessing_y_formula_filter_source"] = (
        request.y_column
    )
    st.session_state["postprocessing_strain_filter_formula"] = (
        format_filter_formula(
            formula_source_label("radial_Hencky_strain"),
            request.strain_settings.workflow,
        )
    )
    st.session_state["postprocessing_strain_filter_source"] = (
        "radial_Hencky_strain"
    )
    st.session_state["postprocessing_force_filter_formula"] = (
        format_filter_formula(
            formula_source_label("force_g"),
            request.force_settings.workflow,
        )
    )
    st.session_state["postprocessing_force_filter_source"] = "force_g"
    st.session_state["postprocessing_r2_threshold"] = request.r2_threshold
    st.session_state["postprocessing_min_points"] = request.min_points
    st.session_state["postprocessing_epsilon_start"] = request.epsilon_start
    st.session_state["postprocessing_show_legend"] = request.show_legend
    st.session_state["postprocessing_y_scale_control"] = request.y_scale.title()
    st.session_state["postprocessing_scale_for_y"] = request.y_column
    st.session_state["postprocessing_show_window"] = request.show_window
    st.session_state["postprocessing_show_slope"] = request.show_slope
    restore_selection_controls(
        "postprocessing",
        file_summary,
        request.selected_files,
        request.selected_materials,
        request.selected_velocities,
    )


def render_postprocessing_window_controls() -> tuple[float, int, float]:
    st.session_state.setdefault("postprocessing_epsilon_start", 1.0)
    st.session_state.setdefault("postprocessing_r2_threshold", 0.995)
    st.session_state.setdefault("postprocessing_min_points", 3)
    with st.popover(
        "Window",
        icon=":material/fit_screen:",
        width="stretch",
    ):
        epsilon_start = st.number_input(
            "Start εᵣ,₀",
            step=0.1,
            format="%.4g",
            key="postprocessing_epsilon_start",
            help="The first sample at or above this strain starts the window.",
        )
        r2_threshold = st.number_input(
            "Minimum R²",
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.4f",
            key="postprocessing_r2_threshold",
        )
        min_points = st.number_input(
            "Minimum points",
            min_value=2,
            max_value=100_000,
            step=1,
            key="postprocessing_min_points",
        )
    return float(r2_threshold), int(min_points), float(epsilon_start)


def render_postprocessing_view_controls(
    y_column: str,
) -> tuple[bool, str, bool, bool]:
    scale_key = "postprocessing_y_scale_control"
    scale_for_key = "postprocessing_scale_for_y"
    st.session_state.setdefault("postprocessing_show_legend", False)
    st.session_state.setdefault("postprocessing_show_window", True)
    st.session_state.setdefault("postprocessing_show_slope", True)
    if st.session_state.get(scale_for_key) != y_column:
        st.session_state.pop(scale_key, None)
        st.session_state[scale_for_key] = y_column
    default_scale = "Log" if uses_log_y(y_column) else "Linear"
    scale_arguments = (
        {"default": default_scale}
        if scale_key not in st.session_state
        else {}
    )
    with st.container(key="postprocessing_compact_view"):
        with st.popover(
            "View",
            icon=":material/visibility:",
            help="View options",
            width="stretch",
        ):
            show_legend = st.toggle(
                "Legend", key="postprocessing_show_legend"
            )
            show_window = st.toggle(
                "Window", key="postprocessing_show_window"
            )
            show_slope = st.toggle(
                "Slope", key="postprocessing_show_slope"
            )
            y_scale = st.segmented_control(
                "Y axis",
                ["Linear", "Log"],
                key=scale_key,
                width="stretch",
                **scale_arguments,
            )
    return (
        bool(show_legend),
        normalize_axis_scale(str(y_scale or default_scale)),
        bool(show_window),
        bool(show_slope),
    )


def render_postprocessing_plot_shell() -> tuple[Any, Any, Any]:
    left_column, right_column = st.columns(
        [1, 1], gap="small", vertical_alignment="top"
    )
    with left_column:
        st.markdown(
            '<div class="plot-heading">Selected variable vs time</div>',
            unsafe_allow_html=True,
        )
        top_target = st.empty()
        st.markdown(
            '<div class="plot-heading">Hencky strain working window</div>',
            unsafe_allow_html=True,
        )
        strain_target = st.empty()
    with right_column:
        st.markdown(
            '<div class="plot-heading">Selected variable vs strain</div>',
            unsafe_allow_html=True,
        )
        right_target = st.empty()
    return top_target, strain_target, right_target

def valid_postprocessing_request(request: object) -> bool:
    required_fields = (
        "r2_threshold",
        "epsilon_start",
        "y_settings",
        "y_scale",
        "show_window",
        "show_slope",
    )
    return request is not None and all(
        hasattr(request, field_name) for field_name in required_fields
    )


def render_postprocessing_workspace(
    file_summary: pd.DataFrame,
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
) -> None:
    applied_key = "postprocessing_applied_request"
    defaults_version_key = "_postprocessing_defaults_v4"
    if not st.session_state.get(defaults_version_key, False):
        st.session_state["postprocessing_y"] = "extensional_viscosity_Pa_s"
        st.session_state["postprocessing_y_formula_filter_formula"] = (
            formula_source_label("extensional_viscosity_Pa_s")
        )
        st.session_state["postprocessing_y_formula_filter_source"] = (
            "extensional_viscosity_Pa_s"
        )
        st.session_state["postprocessing_strain_filter_formula"] = (
            DEFAULT_STRAIN_FORMULA
        )
        st.session_state["postprocessing_strain_filter_source"] = (
            "radial_Hencky_strain"
        )
        st.session_state["postprocessing_force_filter_formula"] = (
            DEFAULT_POSTPROCESSING_FORCE_FORMULA
        )
        st.session_state["postprocessing_force_filter_source"] = "force_g"
        st.session_state["postprocessing_epsilon_start"] = 1.0
        st.session_state["postprocessing_r2_threshold"] = 0.995
        st.session_state["postprocessing_min_points"] = 3
        st.session_state["postprocessing_show_legend"] = False
        st.session_state["postprocessing_y_scale"] = "Log"
        st.session_state["postprocessing_scale_for_y"] = (
            "extensional_viscosity_Pa_s"
        )
        st.session_state["postprocessing_show_window"] = True
        st.session_state["postprocessing_show_slope"] = True
        st.session_state.pop(applied_key, None)
        st.session_state[defaults_version_key] = True
    if st.session_state.get("_restore_scope") == "postprocessing":
        restore_postprocessing_controls(file_summary)
        st.session_state.pop("_restore_scope", None)

    st.session_state.setdefault(
        "postprocessing_y", "extensional_viscosity_Pa_s"
    )

    materials = sorted(
        file_summary["material"].dropna().astype(str).unique()
    )
    velocities = sorted_velocity_values(file_summary["velocity"])

    with st.container(border=True, key="postprocessing_controls_v5"):
        control_columns = st.columns(
            [1.20, 1.15, 1.25, 1.25, 1.50, 1.20, 1.30, 1.15, 0.42, 0.42],
            gap="small",
            vertical_alignment="center",
        )
        with control_columns[0]:
            selected_materials = render_filter_dropdown(
                "postprocessing", "Material", materials
            )
        with control_columns[1]:
            selected_velocities = render_filter_dropdown(
                "postprocessing", "Velocity", velocities
            )
        eligible = file_summary[
            file_summary["material"].astype(str).isin(selected_materials)
            & file_summary["velocity"].astype(str).isin(selected_velocities)
        ]
        with control_columns[2]:
            selected_files, runs_done = render_data_selector(
                "postprocessing", eligible
            )
        with control_columns[3]:
            y_label, y_control = st.columns(
                [0.18, 0.82], gap=None, vertical_alignment="center"
            )
            with y_label:
                render_inline_label("Y:")
            with y_control:
                y_column = st.selectbox(
                    "Y variable",
                    PROCESSED_AXIS_COLUMNS,
                    index=None,
                    key="postprocessing_y",
                    label_visibility="collapsed",
                    format_func=column_menu_label,
                )
        with control_columns[4]:
            formula_label, formula_control = st.columns(
                [0.34, 0.66], gap=None, vertical_alignment="center"
            )
            with formula_label:
                render_inline_label("Formula:")
            with formula_control:
                y_settings = render_processing_controls(
                    "postprocessing_y_formula",
                    str(y_column),
                    show_axis_label=False,
                )
        with control_columns[5]:
            strain_settings = render_processing_controls(
                "postprocessing_strain",
                "radial_Hencky_strain",
                axis_label="εᵣ",
                default_formula=DEFAULT_STRAIN_FORMULA,
            )
        with control_columns[6]:
            force_settings = render_processing_controls(
                "postprocessing_force",
                "force_g",
                axis_label="F",
                default_formula=DEFAULT_POSTPROCESSING_FORCE_FORMULA,
            )
        with control_columns[7]:
            r2_threshold, min_points, epsilon_start = (
                render_postprocessing_window_controls()
            )
        with control_columns[8]:
            (
                show_legend,
                y_scale,
                show_window,
                show_slope,
            ) = render_postprocessing_view_controls(str(y_column))
        with control_columns[9]:
            update_requested = st.button(
                "",
                key="postprocessing_update",
                type="primary",
                icon=":material/refresh:",
                help="Update postprocessing plots",
                width="stretch",
            )

    plot_targets = render_postprocessing_plot_shell()
    draft_request = PostprocessingRequest(
        selected_files=tuple(selected_files),
        y_column=str(y_column),
        force_settings=force_settings,
        strain_settings=strain_settings,
        y_settings=y_settings,
        r2_threshold=r2_threshold,
        min_points=min_points,
        epsilon_start=epsilon_start,
        show_legend=show_legend,
        y_scale=y_scale,
        show_window=show_window,
        show_slope=show_slope,
        selected_materials=tuple(selected_materials),
        selected_velocities=tuple(selected_velocities),
        physical_settings=physical_settings,
    )
    if update_requested or runs_done:
        st.session_state[applied_key] = draft_request

    applied_request = st.session_state.get(applied_key)
    if not valid_postprocessing_request(applied_request):
        applied_request = None
    if applied_request is None:
        render_postprocessing_plot_layout(
            None,
            str(y_column),
            {},
            show_legend=False,
            y_scale=y_scale,
            show_window=show_window,
            show_slope=show_slope,
            targets=plot_targets,
        )
        return

    immediate_view = replace(
        applied_request,
        show_legend=show_legend,
        y_scale=(
            y_scale
            if applied_request.y_column == str(y_column)
            else applied_request.y_scale
        ),
        show_window=show_window,
        show_slope=show_slope,
    )
    st.session_state[applied_key] = immediate_view
    applied_request = immediate_view
    if repr(draft_request) != repr(applied_request):
        st.caption("Controls changed. Press Update to apply them.")
    if not applied_request.selected_files:
        render_postprocessing_plot_layout(
            None,
            normalize_axis_column(applied_request.y_column),
            {},
            show_legend=False,
            y_scale=applied_request.y_scale,
            show_window=applied_request.show_window,
            show_slope=applied_request.show_slope,
            targets=plot_targets,
        )
        return

    raw_data, issues = load_selected_dataset(
        str(DATA_DIR), signature, applied_request.selected_files
    )
    for issue in issues:
        st.warning(issue)
    if raw_data.empty:
        st.warning("No compatible rows were loaded for the selected data.")
        render_postprocessing_plot_layout(
            None,
            normalize_axis_column(applied_request.y_column),
            {},
            show_legend=False,
            y_scale=applied_request.y_scale,
            show_window=applied_request.show_window,
            show_slope=applied_request.show_slope,
            targets=plot_targets,
        )
        return

    preprocessed = preprocess_dataset(
        raw_data, applied_request.physical_settings
    )
    if preprocessed.warnings:
        with st.expander(
            f"Preprocessing warnings ({len(preprocessed.warnings)})"
        ):
            for warning in preprocessed.warnings:
                st.caption(warning)

    labels = {
        row["source_file"]: get_file_label(row)
        for _, row in file_summary.iterrows()
        if row["source_file"] in applied_request.selected_files
    }
    job_key = make_job_key(
        "postprocessing",
        signature,
        applied_request.selected_files,
        applied_request.y_column,
        applied_request.force_settings,
        applied_request.strain_settings,
        applied_request.y_settings,
        applied_request.r2_threshold,
        applied_request.min_points,
        applied_request.epsilon_start,
        applied_request.physical_settings,
    )
    future = submit_background_job(
        job_key,
        analyze_postprocessing_runs,
        preprocessed.frame,
        applied_request.force_settings,
        applied_request.strain_settings,
        applied_request.y_column,
        applied_request.y_settings,
        applied_request.physical_settings,
        applied_request.r2_threshold,
        applied_request.min_points,
        applied_request.epsilon_start,
    )
    render_background_postprocessing(
        future,
        normalize_axis_column(applied_request.y_column),
        labels,
        applied_request.show_legend,
        applied_request.y_scale,
        applied_request.show_window,
        applied_request.show_slope,
        plot_targets,
    )


def render_background_postprocessing(
    future: Future[Any],
    y_column: str,
    labels: dict[str, str],
    show_legend: bool,
    y_scale: str,
    show_window: bool,
    show_slope: bool,
    targets: tuple[Any, Any, Any],
) -> None:
    if future.done():
        try:
            analysis: PostprocessingAnalysis = future.result()
        except Exception as exc:  # pragma: no cover
            st.error(f"Postprocessing failed: {exc}")
            render_postprocessing_plot_layout(
                None,
                y_column,
                labels,
                show_legend=False,
                y_scale=y_scale,
                show_window=show_window,
                show_slope=show_slope,
                targets=targets,
            )
            return
        if analysis.warnings:
            with st.expander(
                f"Postprocessing warnings ({len(analysis.warnings)})"
            ):
                for warning in analysis.warnings:
                    st.caption(warning)
        render_postprocessing_plot_layout(
            analysis,
            y_column,
            labels,
            show_legend,
            y_scale,
            show_window,
            show_slope,
            targets,
        )
        render_working_window_results(analysis, labels)
        return

    render_postprocessing_plot_layout(
        None,
        y_column,
        labels,
        show_legend=False,
        y_scale=y_scale,
        show_window=show_window,
        show_slope=show_slope,
        targets=targets,
    )

    @st.fragment(run_every=0.75)
    def pending_postprocessing() -> None:
        if (
            normalize_workspace(st.session_state.get("workspace"))
            != WORKSPACE_POSTPROCESSING
        ):
            st.rerun()
        if future.done():
            st.rerun()
        st.caption("Calculating working windows in the background...")

    pending_postprocessing()

def configure_postprocessing_figure(
    figure: go.Figure,
    x_column: str,
    y_column: str,
    height: int,
    show_legend: bool,
    y_scale: str = "linear",
) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 12, "b": 24},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=show_legend,
        hovermode="closest",
        font={"color": "#344054", "size": 11},
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 0,
            "font": {"size": 9},
        },
    )
    figure.update_xaxes(title=column_axis_title(x_column), type="linear")
    figure.update_yaxes(
        title=column_axis_title(y_column),
        type=normalize_axis_scale(y_scale),
    )
    style_axes(figure)
    return figure


def add_windowed_trace(
    figure: go.Figure,
    x_values: np.ndarray,
    y_values: np.ndarray,
    in_window: np.ndarray,
    color: str,
    name: str,
    show_legend: bool,
    show_window: bool,
) -> None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    in_window = np.asarray(in_window, dtype=bool)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    finite_positions = np.flatnonzero(finite)
    inside_positions = np.flatnonzero(finite & in_window)

    def add_trace(
        positions: np.ndarray,
        line_color: str,
        line_width: float,
        legend: bool,
        suffix: str = "",
    ) -> None:
        if not positions.size:
            return
        figure.add_trace(go.Scatter(
            x=x_values[positions],
            y=y_values[positions],
            mode="lines",
            line={"color": line_color, "width": line_width},
            name=name,
            showlegend=legend,
            hovertemplate=(
                f"{name}{suffix}<br>x: %{{x:.5g}}<br>y: %{{y:.5g}}"
                "<extra></extra>"
            ),
        ))

    if not show_window:
        add_trace(finite_positions, color, 1.9, show_legend)
        return
    if not inside_positions.size:
        add_trace(
            finite_positions,
            "rgba(107, 114, 128, 0.34)",
            2.2,
            show_legend,
            " · outside window",
        )
        return

    add_trace(inside_positions, color, 1.9, show_legend)
    first_inside = int(inside_positions[0])
    last_inside = int(inside_positions[-1])
    before = finite_positions[finite_positions < first_inside]
    after = finite_positions[finite_positions > last_inside]
    if before.size:
        before = np.concatenate((before, [first_inside]))
        add_trace(
            before,
            "rgba(107, 114, 128, 0.34)",
            2.2,
            False,
            " · outside window",
        )
    if after.size:
        after = np.concatenate(([last_inside], after))
        add_trace(
            after,
            "rgba(107, 114, 128, 0.34)",
            2.2,
            False,
            " · outside window",
        )


def make_postprocessing_figures(
    analysis: PostprocessingAnalysis | None,
    y_column: str,
    labels: dict[str, str],
    show_legend: bool,
    y_scale: str = "linear",
    show_window: bool = True,
    show_slope: bool = True,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    top_figure = configure_postprocessing_figure(
        go.Figure(),
        "time_from_onset_s",
        y_column,
        POSTPROCESSING_SMALL_PLOT_HEIGHT,
        show_legend,
        y_scale,
    )
    strain_figure = configure_postprocessing_figure(
        go.Figure(),
        "time_from_onset_s",
        "radial_Hencky_strain",
        POSTPROCESSING_SMALL_PLOT_HEIGHT,
        False,
        "linear",
    )
    right_figure = configure_postprocessing_figure(
        go.Figure(),
        "radial_Hencky_strain",
        y_column,
        POSTPROCESSING_TALL_PLOT_HEIGHT,
        False,
        y_scale,
    )
    if analysis is None or analysis.frame.empty:
        return top_figure, strain_figure, right_figure

    source_files = (
        analysis.frame["source_file"].astype(str).drop_duplicates().tolist()
    )
    source_colors = {
        source_file: IFF_PALETTE[index % len(IFF_PALETTE)]
        for index, source_file in enumerate(source_files)
    }
    shade_details: list[
        tuple[float, float, float, float, float, float, float, float]
    ] = []

    for source_file in source_files:
        group = analysis.frame[
            analysis.frame["source_file"].astype(str) == source_file
        ].sort_values("time_from_onset_s")
        series_name = labels.get(source_file, source_file)
        color = source_colors[source_file]
        window = analysis.windows.get(source_file)
        valid_strain = group[[
            "time_from_onset_s", "radial_Hencky_strain"
        ]].dropna()
        if valid_strain.empty:
            continue

        default_start = (
            0,
            float(valid_strain.iloc[0]["time_from_onset_s"]),
            float(valid_strain.iloc[0]["radial_Hencky_strain"]),
        )
        start_index, start_time, start_strain = analysis.window_starts.get(
            source_file, default_start
        )
        endpoint_time: float | None = None
        endpoint_strain: float | None = None
        if (
            window is not None
            and window.valid
            and window.endpoint_index is not None
            and window.endpoint_index < len(valid_strain)
        ):
            endpoint_row = valid_strain.iloc[window.endpoint_index]
            endpoint_time = float(endpoint_row["time_from_onset_s"])
            endpoint_strain = float(endpoint_row["radial_Hencky_strain"])

        time_values = pd.to_numeric(
            group["time_from_onset_s"], errors="coerce"
        ).to_numpy(dtype=float)
        strain_values = pd.to_numeric(
            group["radial_Hencky_strain"], errors="coerce"
        ).to_numpy(dtype=float)
        y_values = pd.to_numeric(
            group[y_column], errors="coerce"
        ).to_numpy(dtype=float)
        in_window = (
            np.isfinite(time_values)
            & (time_values >= start_time)
            & (time_values <= endpoint_time)
            if endpoint_time is not None
            else np.zeros(time_values.shape, dtype=bool)
        )

        add_windowed_trace(
            top_figure,
            time_values,
            y_values,
            in_window,
            color,
            series_name,
            show_legend,
            show_window,
        )
        add_windowed_trace(
            strain_figure,
            time_values,
            strain_values,
            in_window,
            color,
            series_name,
            False,
            show_window,
        )
        add_windowed_trace(
            right_figure,
            strain_values,
            y_values,
            in_window,
            color,
            series_name,
            False,
            show_window,
        )

        finite_times = time_values[np.isfinite(time_values)]
        finite_strain = strain_values[np.isfinite(strain_values)]
        if endpoint_time is None or endpoint_strain is None or window is None:
            continue

        fit_x = valid_strain["time_from_onset_s"].to_numpy(dtype=float)[
            start_index : window.endpoint_index + 1
        ]
        if show_slope and fit_x.size:
            fit_y = window.slope * fit_x + window.intercept
            strain_figure.add_trace(go.Scatter(
                x=fit_x,
                y=fit_y,
                mode="lines",
                line={"color": color, "width": 1.4, "dash": "dash"},
                name=f"{series_name} fit",
                showlegend=False,
                hoverinfo="skip",
            ))
        if show_window:
            strain_figure.add_trace(go.Scatter(
                x=[start_time, endpoint_time],
                y=[start_strain, endpoint_strain],
                mode="markers",
                marker={
                    "color": color,
                    "size": 8,
                    "line": {"color": "#ffffff", "width": 1.5},
                },
                name=f"{series_name} window",
                showlegend=False,
                hovertemplate=(
                    f"{series_name}<br>t: %{{x:.5g}} s<br>"
                    "εᵣ: %{y:.5g}<extra></extra>"
                ),
            ))
            guide_style = {
                "color": color,
                "width": 1.1,
                "dash": "dot",
            }
            for boundary_time, boundary_strain in (
                (start_time, start_strain),
                (endpoint_time, endpoint_strain),
            ):
                strain_figure.add_shape(
                    type="line",
                    x0=boundary_time,
                    x1=boundary_time,
                    y0=0.0,
                    y1=boundary_strain,
                    line=guide_style,
                )
                strain_figure.add_shape(
                    type="line",
                    x0=0.0,
                    x1=boundary_time,
                    y0=boundary_strain,
                    y1=boundary_strain,
                    line=guide_style,
                )

        if finite_times.size and finite_strain.size:
            shade_details.append((
                float(np.min(finite_times)),
                start_time,
                endpoint_time,
                float(np.max(finite_times)),
                float(np.min(finite_strain)),
                start_strain,
                endpoint_strain,
                float(np.max(finite_strain)),
            ))

    if show_window and len(shade_details) == 1:
        (
            minimum_time,
            start_time,
            endpoint_time,
            maximum_time,
            minimum_strain,
            start_strain,
            endpoint_strain,
            maximum_strain,
        ) = shade_details[0]
        for figure in (top_figure, strain_figure):
            if minimum_time < start_time:
                figure.add_vrect(
                    x0=minimum_time,
                    x1=start_time,
                    fillcolor="#6B7280",
                    opacity=0.10,
                    line_width=0,
                    layer="below",
                )
            if endpoint_time < maximum_time:
                figure.add_vrect(
                    x0=endpoint_time,
                    x1=maximum_time,
                    fillcolor="#6B7280",
                    opacity=0.10,
                    line_width=0,
                    layer="below",
                )
        if minimum_strain < start_strain:
            right_figure.add_vrect(
                x0=minimum_strain,
                x1=start_strain,
                fillcolor="#6B7280",
                opacity=0.10,
                line_width=0,
                layer="below",
            )
        if endpoint_strain < maximum_strain:
            right_figure.add_vrect(
                x0=endpoint_strain,
                x1=maximum_strain,
                fillcolor="#6B7280",
                opacity=0.10,
                line_width=0,
                layer="below",
            )
    return top_figure, strain_figure, right_figure


def render_postprocessing_plot_layout(
    analysis: PostprocessingAnalysis | None,
    y_column: str,
    labels: dict[str, str],
    show_legend: bool,
    y_scale: str,
    show_window: bool,
    show_slope: bool,
    targets: tuple[Any, Any, Any],
) -> None:
    top_figure, strain_figure, right_figure = make_postprocessing_figures(
        analysis,
        y_column,
        labels,
        show_legend,
        y_scale,
        show_window,
        show_slope,
    )
    top_target, strain_target, right_target = targets
    top_target.plotly_chart(
        top_figure,
        width="stretch",
        config=plotly_config(),
        key="postprocessing_top_plot",
    )
    strain_target.plotly_chart(
        strain_figure,
        width="stretch",
        config=plotly_config(),
        key="postprocessing_strain_plot",
    )
    right_target.plotly_chart(
        right_figure,
        width="stretch",
        config=plotly_config(),
        key="postprocessing_right_plot",
    )

def render_working_window_results(
    analysis: PostprocessingAnalysis,
    labels: dict[str, str],
) -> None:
    records: list[dict[str, Any]] = []
    for source_file, window in analysis.windows.items():
        start_index, start_time, start_strain = analysis.window_starts.get(
            source_file, (0, None, None)
        )
        endpoint_strain = None
        if window.endpoint_index is not None:
            valid_strain = analysis.frame[
                analysis.frame["source_file"].astype(str) == source_file
            ].sort_values("time_from_onset_s")[[
                "time_from_onset_s", "radial_Hencky_strain"
            ]].dropna()
            if window.endpoint_index < len(valid_strain):
                endpoint_strain = float(
                    valid_strain.iloc[window.endpoint_index][
                        "radial_Hencky_strain"
                    ]
                )
        record: dict[str, Any] = {
            "Data": labels.get(source_file, source_file),
            "Points": (
                window.endpoint_index - start_index + 1
                if window.endpoint_index is not None
                else None
            ),
            "Start time [s]": start_time,
            "Start εᵣ [-]": start_strain,
            "End time [s]": window.endpoint_x,
            "End εᵣ [-]": endpoint_strain,
            "Slope [s⁻¹]": window.slope,
            "Intercept": window.intercept,
            "R²": window.r2,
        }
        records.append(record)
    if not records:
        return
    with st.expander("Working-window results"):
        st.dataframe(
            pd.DataFrame.from_records(records),
            width="stretch",
            hide_index=True,
        )

def restore_summary_controls(file_summary: pd.DataFrame) -> None:
    request = st.session_state.get("summary_applied_request")
    if request is None or not hasattr(request, "show_legend"):
        return
    st.session_state["summary_x"] = normalize_axis_column(request.x_column)
    st.session_state["summary_y"] = normalize_axis_column(request.y_column)
    st.session_state["summary_group_by"] = request.group_by
    st.session_state["summary_bins"] = request.bin_count
    st.session_state["summary_show_legend"] = request.show_legend
    st.session_state["summary_x_scale"] = request.x_scale.title()
    st.session_state["summary_y_scale"] = request.y_scale.title()
    st.session_state["summary_scale_for_y"] = normalize_axis_column(
        request.y_column
    )
    restore_selection_controls(
        "summary",
        file_summary,
        request.selected_files,
        request.selected_materials,
        request.selected_velocities,
    )


def render_summary_workspace(
    file_summary: pd.DataFrame,
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
) -> None:
    applied_key = "summary_applied_request"
    if st.session_state.get("_restore_scope") == "summary":
        restore_summary_controls(file_summary)
        st.session_state.pop("_restore_scope", None)

    materials = sorted(file_summary["material"].dropna().astype(str).unique())
    velocities = sorted_velocity_values(file_summary["velocity"])
    st.session_state.setdefault("summary_x", "radial_Hencky_strain")
    st.session_state.setdefault("summary_y", "net_stress_Pa")
    st.session_state["summary_x"] = normalize_axis_column(
        st.session_state["summary_x"]
    )
    st.session_state["summary_y"] = normalize_axis_column(
        st.session_state["summary_y"]
    )
    st.session_state.setdefault("summary_group_by", "Material")
    st.session_state.setdefault("summary_bins", 100)
    control_columns = st.columns([1.05, 1.05, 1.45, 0.85, 0.8], gap="small")
    with control_columns[0]:
        x_column = st.selectbox(
            "X axis",
            PROCESSED_AXIS_COLUMNS,
            index=None,
            key="summary_x",
            format_func=column_menu_label,
        )
    with control_columns[1]:
        y_column = st.selectbox(
            "Y axis",
            PROCESSED_AXIS_COLUMNS,
            index=None,
            key="summary_y",
            format_func=column_menu_label,
        )
    with control_columns[2]:
        group_by = st.selectbox(
            "Group by",
            ["Material", "Velocity", "Material + velocity"],
            index=None,
            key="summary_group_by",
        )
    with control_columns[3]:
        with st.popover("Binning", width="stretch"):
            bin_count = st.number_input(
                "Number of X bins",
                min_value=10,
                max_value=500,
                value=None,
                step=10,
                key="summary_bins",
            )
    with control_columns[4]:
        _, show_legend, x_scale, y_scale = render_plot_view_controls(
            "summary", str(y_column)
        )

    selection_columns = st.columns(
        [1.0, 1.0, 3.2, 1.0], gap="small", vertical_alignment="top"
    )
    with selection_columns[0]:
        selected_materials = render_filter_dropdown(
            "summary", "Material", materials
        )
    with selection_columns[1]:
        selected_velocities = render_filter_dropdown(
            "summary", "Velocity", velocities
        )
    eligible = file_summary[
        file_summary["material"].astype(str).isin(selected_materials)
        & file_summary["velocity"].astype(str).isin(selected_velocities)
    ]
    with selection_columns[2]:
        selected_files, runs_done = render_data_selector("summary", eligible)
    with selection_columns[3]:
        update_requested = st.button(
            "Update plots",
            key="summary_update",
            type="primary",
            icon=":material/refresh:",
            width="stretch",
        )

    draft_request = SummaryRequest(
        selected_files=tuple(selected_files),
        x_column=str(x_column),
        y_column=str(y_column),
        group_by=str(group_by),
        bin_count=int(bin_count),
        show_legend=bool(show_legend),
        x_scale=x_scale,
        y_scale=y_scale,
        selected_materials=tuple(selected_materials),
        selected_velocities=tuple(selected_velocities),
        physical_settings=physical_settings,
    )
    if update_requested or runs_done:
        st.session_state[applied_key] = draft_request

    applied_request = st.session_state.get(applied_key)
    if applied_request is not None and not hasattr(applied_request, "show_legend"):
        applied_request = None
    if applied_request is None:
        st.info("Choose data series and press Update plots.")
        return
    if repr(draft_request) != repr(applied_request):
        st.caption("Controls changed. Press Update plots to apply them.")
    if not applied_request.selected_files:
        st.info("No data series are applied to the summary plots.")
        return

    raw_data, issues = load_selected_dataset(
        str(DATA_DIR), signature, applied_request.selected_files
    )
    for issue in issues:
        st.warning(issue)
    if raw_data.empty:
        st.info("No compatible rows were loaded for the applied data series.")
        return

    applied_physics = applied_request.physical_settings
    applied_x_column = normalize_axis_column(applied_request.x_column)
    applied_y_column = normalize_axis_column(applied_request.y_column)
    preprocessed = preprocess_dataset(raw_data, applied_physics)
    data = derive_dataset(preprocessed.frame, applied_physics)
    summary, peaks = build_summary_tables(
        data,
        applied_x_column,
        applied_y_column,
        applied_request.group_by,
        applied_request.bin_count,
    )
    if summary.empty:
        st.info("No numeric data matches the applied summary selection.")
        return

    mean_figure = make_summary_mean_figure(
        summary,
        applied_x_column,
        applied_y_column,
        applied_request.show_legend,
        applied_request.x_scale,
        applied_request.y_scale,
    )
    peak_figure = px.box(
        peaks,
        x="group",
        y="peak",
        color="group",
        points="all",
        color_discrete_sequence=IFF_PALETTE,
        labels={
            "group": applied_request.group_by,
            "peak": "Peak " + column_axis_title(applied_y_column),
        },
    )
    peak_figure.update_layout(
        height=520,
        margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=applied_request.show_legend,
        font={"color": "#344054", "size": 11},
    )
    peak_figure.update_yaxes(
        title="Peak " + column_axis_title(applied_request.y_column),
        type=normalize_axis_scale(applied_request.y_scale),
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
    show_legend: bool,
    x_scale: str,
    y_scale: str,
) -> go.Figure:
    log_y = normalize_axis_scale(y_scale) == "log"
    figure = go.Figure()
    for index, (group_name, group) in enumerate(summary.groupby("group", sort=True)):
        ordered = group.sort_values("x_mean")
        color = IFF_PALETTE[index % len(IFF_PALETTE)]
        mean = ordered["y_mean"]
        lower = mean - ordered["y_std"].fillna(0.0)
        upper = mean + ordered["y_std"].fillna(0.0)
        if log_y:
            mean = mean.where(mean > 0)
            lower = lower.where(lower > 0)
            upper = upper.where(upper > 0)
        figure.add_trace(go.Scatter(
            x=ordered["x_mean"],
            y=lower,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=ordered["x_mean"],
            y=upper,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=color,
            opacity=0.16,
            showlegend=False,
            hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=ordered["x_mean"],
            y=mean,
            mode="lines",
            line={"color": color, "width": 2},
            name=str(group_name),
            showlegend=show_legend,
        ))
    figure.update_layout(
        height=520,
        margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=show_legend,
        hovermode="closest",
        font={"color": "#344054", "size": 11},
        legend={"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 9}},
    )
    figure.update_xaxes(
        title=column_axis_title(x_label),
        type=normalize_axis_scale(x_scale),
    )
    figure.update_yaxes(
        title=column_axis_title(y_label),
        type=normalize_axis_scale(y_scale),
    )
    style_axes(figure)
    return figure

def restore_frequency_controls(file_summary: pd.DataFrame) -> None:
    request = st.session_state.get("frequency_applied_request")
    if request is None or not hasattr(request, "show_peaks"):
        return

    st.session_state["frequency_signal"] = request.signal_column
    st.session_state["frequency_filter_formula"] = format_filter_formula(
        formula_source_label(request.signal_column),
        request.filter_settings.workflow,
    )
    st.session_state["frequency_filter_source"] = request.signal_column
    (
        st.session_state["frequency_peak_min"],
        st.session_state["frequency_peak_max"],
        st.session_state["frequency_peak_count"],
        st.session_state["frequency_energy_bins"],
        st.session_state["frequency_histogram_bin_width_hz"],
    ) = request.peak_settings
    st.session_state["frequency_individual_plot"] = request.individual_plot
    st.session_state["frequency_summary_plot"] = request.summary_plot
    st.session_state["frequency_show_peaks"] = request.show_peaks
    st.session_state["frequency_show_legend"] = request.show_legend
    st.session_state["frequency_individual_x_scale"] = (
        request.individual_x_scale.title()
    )
    st.session_state["frequency_individual_y_scale"] = (
        request.individual_y_scale.title()
    )
    st.session_state["frequency_individual_scale_for_plot"] = (
        request.individual_plot
    )
    st.session_state["frequency_summary_x_scale"] = request.summary_x_scale.title()
    st.session_state["frequency_summary_y_scale"] = request.summary_y_scale.title()
    st.session_state["frequency_summary_scale_for_plot"] = request.summary_plot
    restore_selection_controls(
        "frequency",
        file_summary,
        request.selected_files,
        request.selected_materials,
        request.selected_velocities,
    )


def render_frequency_axis_controls(
    key_prefix: str,
    plot_name: str,
) -> tuple[str, str]:
    x_key = f"{key_prefix}_x_scale"
    y_key = f"{key_prefix}_y_scale"
    plot_key = f"{key_prefix}_scale_for_plot"
    st.session_state.setdefault(x_key, "Linear")
    if st.session_state.get(plot_key) != plot_name:
        st.session_state[y_key] = "Log" if plot_name == "PSD" else "Linear"
        st.session_state[plot_key] = plot_name
    st.session_state.setdefault(y_key, "Linear")

    with st.popover("Axes", icon=":material/straighten:", width="stretch"):
        columns = st.columns(2, gap="small")
        with columns[0]:
            x_scale = st.segmented_control(
                "X scale",
                ["Linear", "Log"],
                key=x_key,
                required=True,
                width="stretch",
            )
        with columns[1]:
            y_scale = st.segmented_control(
                "Y scale",
                ["Linear", "Log"],
                key=y_key,
                required=True,
                width="stretch",
            )
    return normalize_axis_scale(str(x_scale)), normalize_axis_scale(str(y_scale))


def render_frequency_individual_view_controls(
    plot_name: str,
) -> tuple[bool, bool, str, str]:
    x_key = "frequency_individual_x_scale"
    y_key = "frequency_individual_y_scale"
    plot_key = "frequency_individual_scale_for_plot"
    st.session_state.setdefault("frequency_show_peaks", True)
    st.session_state.setdefault("frequency_show_legend", False)
    st.session_state.setdefault(x_key, "Linear")
    if st.session_state.get(plot_key) != plot_name:
        st.session_state[y_key] = "Log" if plot_name == "PSD" else "Linear"
        st.session_state[plot_key] = plot_name
    st.session_state.setdefault(y_key, "Linear")

    with st.popover("View", icon=":material/visibility:", width="stretch"):
        toggle_columns = st.columns(2, gap="small")
        with toggle_columns[0]:
            show_peaks = st.toggle("Peaks", key="frequency_show_peaks")
        with toggle_columns[1]:
            show_legend = st.toggle("Legend", key="frequency_show_legend")
        scale_columns = st.columns(2, gap="small")
        with scale_columns[0]:
            x_scale = st.segmented_control(
                "X scale",
                ["Linear", "Log"],
                key=x_key,
                required=True,
                width="stretch",
            )
        with scale_columns[1]:
            y_scale = st.segmented_control(
                "Y scale",
                ["Linear", "Log"],
                key=y_key,
                required=True,
                width="stretch",
            )
    return (
        bool(show_peaks),
        bool(show_legend),
        normalize_axis_scale(str(x_scale)),
        normalize_axis_scale(str(y_scale)),
    )



def render_frequency_workspace(
    file_summary: pd.DataFrame,
    signature: tuple[tuple[str, int, int], ...],
    physical_settings: PhysicalSettings,
) -> None:
    applied_key = "frequency_applied_request"
    defaults_version_key = "_frequency_defaults_v2"
    if not st.session_state.get(defaults_version_key, False):
        reset_keys = {
            applied_key,
            "frequency_signal",
            "frequency_filter_formula",
            "frequency_filter_source",
            "frequency_individual_plot",
            "frequency_summary_plot",
            "frequency_show_legend",
            "frequency_peak_min",
            "frequency_peak_max",
            "frequency_peak_count",
            "frequency_energy_bins",
            "frequency_histogram_bins",
            "frequency_histogram_bin_width_hz",
            "_frequency_runs_defaulted_all_v1",
        }
        for key in list(st.session_state):
            if key in reset_keys or key.startswith(
                ("frequency_material_", "frequency_velocity_", "frequency_file_")
            ):
                st.session_state.pop(key, None)
        st.session_state[defaults_version_key] = True
    if st.session_state.get("_restore_scope") == "frequency":
        restore_frequency_controls(file_summary)
        st.session_state.pop("_restore_scope", None)

    st.session_state.setdefault("frequency_signal", "force_g")
    st.session_state.setdefault("frequency_individual_plot", "FFT")
    st.session_state.setdefault(
        "frequency_summary_plot", "Peak frequency histogram"
    )

    materials = sorted(file_summary["material"].dropna().astype(str).unique())
    velocities = sorted_velocity_values(file_summary["velocity"])
    with st.container(border=True, key="frequency_toolbar_separate_v3"):
        control_columns = st.columns(
            [1.0, 1.0, 0.82, 0.82, 0.92, 1.0, 0.28],
            gap="small",
            vertical_alignment="center",
        )

        with control_columns[0]:
            variable_label, variable_control = st.columns(
                [0.4, 0.6], gap=None, vertical_alignment="center"
            )
            with variable_label:
                render_inline_label("Variable:")
            with variable_control:
                signal_column = st.selectbox(
                    "Signal",
                    AVAILABLE_COLUMNS,
                    index=None,
                    key="frequency_signal",
                    label_visibility="collapsed",
                    format_func=column_menu_label,
                )

        with control_columns[1]:
            formula_label, formula_control = st.columns(
                [0.38, 0.62], gap=None, vertical_alignment="center"
            )
            with formula_label:
                render_inline_label("Formula:")
            with formula_control:
                filter_settings = render_processing_controls(
                    "frequency",
                    str(signal_column),
                    show_axis_label=False,
                )

        with control_columns[2]:
            selected_materials = render_filter_dropdown(
                "frequency", "Material", materials
            )
        with control_columns[3]:
            selected_velocities = render_filter_dropdown(
                "frequency", "Velocity", velocities
            )
        eligible = file_summary[
            file_summary["material"].astype(str).isin(selected_materials)
            & file_summary["velocity"].astype(str).isin(selected_velocities)
        ]
        with control_columns[4]:
            selected_files, runs_done = render_data_selector(
                "frequency", eligible, default_selected=True
            )
        with control_columns[5]:
            peak_settings = render_peak_settings()
        with control_columns[6]:
            update_requested = st.button(
                "",
                key="frequency_update",
                type="primary",
                icon=":material/refresh:",
                width="stretch",
                help="Update frequency-analysis plots",
            )
    frequency_status = st.empty()
    chart_columns = st.columns(2, gap="small")
    with chart_columns[0]:
        with st.container(border=True, key="frequency_individual_panel"):
            title_column, selector_column, view_column = st.columns(
                [0.2, 0.58, 0.22],
                gap="small",
                vertical_alignment="center",
            )
            with title_column:
                render_inline_label("Individual:")
            with selector_column:
                individual_plot = st.selectbox(
                    "Individual plot",
                    ["FFT", "PSD", "Energy by frequency band"],
                    key="frequency_individual_plot",
                    label_visibility="collapsed",
                )
            with view_column:
                (
                    show_peaks,
                    show_legend,
                    individual_x_scale,
                    individual_y_scale,
                ) = render_frequency_individual_view_controls(
                    str(individual_plot)
                )
            individual_chart_target = st.empty()

    with chart_columns[1]:
        with st.container(border=True, key="frequency_summary_panel"):
            title_column, selector_column, axes_column = st.columns(
                [0.2, 0.58, 0.22],
                gap="small",
                vertical_alignment="center",
            )
            with title_column:
                render_inline_label("Summary:")
            with selector_column:
                summary_plot = st.selectbox(
                    "Summary plot",
                    [
                        "Peak frequency histogram",
                        "Peak amplitude histogram",
                        "Dominant frequency by run",
                    ],
                    key="frequency_summary_plot",
                    label_visibility="collapsed",
                )
            with axes_column:
                (
                    summary_x_scale,
                    summary_y_scale,
                ) = render_frequency_axis_controls(
                    "frequency_summary", str(summary_plot)
                )
            summary_chart_target = st.empty()

    render_empty_frequency_axes(
        individual_chart_target,
        summary_chart_target,
        str(signal_column),
        str(individual_plot),
        str(summary_plot),
        individual_x_scale,
        individual_y_scale,
        summary_x_scale,
        summary_y_scale,
    )
    draft_request = FrequencyRequest(
        selected_files=tuple(selected_files),
        signal_column=str(signal_column),
        filter_settings=filter_settings,
        peak_settings=peak_settings,
        individual_plot=str(individual_plot),
        summary_plot=str(summary_plot),
        show_peaks=bool(show_peaks),
        show_legend=bool(show_legend),
        individual_x_scale=individual_x_scale,
        individual_y_scale=individual_y_scale,
        summary_x_scale=summary_x_scale,
        summary_y_scale=summary_y_scale,
        selected_materials=tuple(selected_materials),
        selected_velocities=tuple(selected_velocities),
        physical_settings=physical_settings,
    )
    if update_requested or runs_done:
        st.session_state[applied_key] = draft_request

    applied_request = st.session_state.get(applied_key)
    if applied_request is not None and not hasattr(applied_request, "show_peaks"):
        applied_request = None
    if applied_request is None:
        applied_request = draft_request
        st.session_state[applied_key] = applied_request
    draft_analysis_key = frequency_analysis_key(draft_request)
    applied_analysis_key = frequency_analysis_key(applied_request)
    if draft_analysis_key != applied_analysis_key:
        analysis_field_names = (
            "runs",
            "signal",
            "formula",
            "peak settings",
            "materials",
            "velocities",
            "physics",
        )
        changed_fields = [
            name
            for name, draft_value, applied_value in zip(
                analysis_field_names,
                draft_analysis_key,
                applied_analysis_key,
            )
            if draft_value != applied_value
        ]
        frequency_status.caption(
            f"Changed: {', '.join(changed_fields)}. Press Update to apply."
        )

    applied_request = replace(
        applied_request,
        individual_plot=str(individual_plot),
        summary_plot=str(summary_plot),
        show_peaks=bool(show_peaks),
        show_legend=bool(show_legend),
        individual_x_scale=individual_x_scale,
        individual_y_scale=individual_y_scale,
        summary_x_scale=summary_x_scale,
        summary_y_scale=summary_y_scale,
    )
    st.session_state[applied_key] = applied_request
    if not applied_request.selected_files:
        frequency_status.info("No data series are applied to frequency analysis.")
        return

    raw_data, issues = load_selected_dataset(
        str(DATA_DIR), signature, applied_request.selected_files
    )
    for issue in issues:
        st.warning(issue)
    if raw_data.empty:
        frequency_status.info("No compatible rows were loaded for the applied data series.")
        return

    applied_physics = applied_request.physical_settings
    preprocessed = preprocess_dataset(raw_data, applied_physics)
    if preprocessed.warnings:
        with st.expander(f"Preprocessing warnings ({len(preprocessed.warnings)})"):
            for warning in preprocessed.warnings:
                st.caption(warning)
    data = derive_dataset(preprocessed.frame, applied_physics)
    labels = {
        row["source_file"]: get_file_label(row)
        for _, row in file_summary.iterrows()
        if row["source_file"] in applied_request.selected_files
    }
    job_key = make_job_key(
        "frequency-batch",
        signature,
        frequency_analysis_key(applied_request),
    )
    render_background_frequency(
        job_key,
        data,
        applied_request,
        labels,
        individual_chart_target,
        summary_chart_target,
    )

def render_peak_settings() -> tuple[float, float, int, int, float]:
    defaults = {
        "frequency_peak_min": 0.0,
        "frequency_peak_max": 1000.0,
        "frequency_peak_count": 3,
        "frequency_energy_bins": 20,
        "frequency_histogram_bin_width_hz": 10.0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    with st.popover(
        "Peak settings",
        icon=":material/finance:",
        width="stretch",
    ):
        st.caption("Shared by the individual and summary frequency plots.")
        range_columns = st.columns(2, gap="small")
        peak_min = range_columns[0].number_input(
            "Minimum (Hz)",
            min_value=0.0,
            step=0.1,
            key="frequency_peak_min",
        )
        peak_max = range_columns[1].number_input(
            "Maximum (Hz)",
            min_value=0.001,
            value=None,
            step=0.5,
            key="frequency_peak_max",
        )
        detail_columns = st.columns(3, gap="small")
        peak_count = detail_columns[0].number_input(
            "Peaks / run",
            min_value=1,
            max_value=20,
            value=None,
            step=1,
            key="frequency_peak_count",
        )
        energy_bins = detail_columns[1].number_input(
            "Energy band count",
            min_value=4,
            max_value=100,
            value=None,
            step=1,
            key="frequency_energy_bins",
            help=(
                "Number of equal-width frequency intervals used to integrate "
                "the PSD in the Energy plot."
            ),
        )
        histogram_bin_width_hz = detail_columns[2].number_input(
            "Bin width (Hz)",
            min_value=0.001,
            max_value=1000.0,
            value=None,
            step=1.0,
            key="frequency_histogram_bin_width_hz",
        )
    minimum = float(peak_min)
    maximum = max(minimum + 1e-9, float(peak_max))
    return (
        minimum,
        maximum,
        int(peak_count),
        int(energy_bins),
        float(histogram_bin_width_hz),
    )

def render_background_frequency(
    job_key: str,
    frame: pd.DataFrame,
    request: FrequencyRequest,
    series_labels: dict[str, str],
    individual_chart_target: Any,
    summary_chart_target: Any,
) -> None:
    future = submit_background_job(
        job_key,
        analyze_frequency_runs,
        frame,
        request.signal_column,
        request.filter_settings,
        *request.peak_settings[:4],
    )
    try:
        if future.done():
            result: FrequencyBatchResult = future.result()
        else:
            with st.spinner("Updating frequency analysis..."):
                result = future.result()
    except Exception as exc:  # pragma: no cover
        st.error(f"Frequency analysis failed: {exc}")
        return
    render_frequency_results(
        result,
        request,
        series_labels,
        individual_chart_target,
        summary_chart_target,
    )

def math_axis_title(symbol: str, unit: str) -> str:
    return f"{symbol} [{unit}]"


def spectrum_axis_title(kind: str, signal_column: str) -> str:
    symbol = COLUMN_AXIS_SYMBOLS.get(signal_column, signal_column)
    unit = COLUMN_UNITS.get(signal_column, "-")
    if kind == "FFT":
        return math_axis_title(f"|FFT({symbol})|", unit)
    if kind == "PSD":
        psd_unit = (
            "Hz<sup>-1</sup>"
            if unit == "-"
            else f"{unit}<sup>2</sup> Hz<sup>-1</sup>"
        )
        return math_axis_title(f"PSD({symbol})", psd_unit)
    energy_unit = "-" if unit == "-" else f"{unit}<sup>2</sup>"
    return math_axis_title(f"<i>E</i>({symbol})", energy_unit)

def frequency_individual_axis_titles(
    plot_name: str,
    signal_column: str,
) -> tuple[str, str]:
    spectrum_kind = "Energy" if plot_name == "Energy by frequency band" else plot_name
    return (
        math_axis_title("<i>f</i>", "Hz"),
        spectrum_axis_title(spectrum_kind, signal_column),
    )


def frequency_summary_axis_titles(
    plot_name: str,
    signal_column: str,
) -> tuple[str, str, bool]:
    if plot_name == "Peak amplitude histogram":
        return (
            spectrum_axis_title("FFT", signal_column),
            math_axis_title("<i>N</i><sub>p</sub>", "-"),
            False,
        )
    if plot_name == "Dominant frequency by run":
        return (
            "Data series [-]",
            math_axis_title("<i>f</i><sub>dom</sub>", "Hz"),
            True,
        )
    return (
        math_axis_title("<i>f</i><sub>p</sub>", "Hz"),
        math_axis_title("<i>N</i><sub>p</sub>", "-"),
        False,
    )


def make_empty_frequency_figure(
    x_title: str,
    y_title: str,
    x_scale: str,
    y_scale: str,
    categorical_x: bool = False,
) -> go.Figure:
    figure = go.Figure()
    configure_analysis_figure(
        figure,
        x_title,
        y_title,
        520,
        x_scale,
        y_scale,
    )
    if categorical_x:
        figure.update_xaxes(type="category")
    return figure


def render_empty_frequency_axes(
    individual_target: Any,
    summary_target: Any,
    signal_column: str,
    individual_plot: str,
    summary_plot: str,
    individual_x_scale: str,
    individual_y_scale: str,
    summary_x_scale: str,
    summary_y_scale: str,
) -> None:
    individual_x_title, individual_y_title = frequency_individual_axis_titles(
        individual_plot,
        signal_column,
    )
    summary_x_title, summary_y_title, categorical_x = (
        frequency_summary_axis_titles(summary_plot, signal_column)
    )
    with individual_target.container():
        st.plotly_chart(
            make_empty_frequency_figure(
                individual_x_title,
                individual_y_title,
                individual_x_scale,
                individual_y_scale,
            ),
            width="stretch",
            config=plotly_config(),
            key="frequency_empty_individual",
        )
    with summary_target.container():
        st.plotly_chart(
            make_empty_frequency_figure(
                summary_x_title,
                summary_y_title,
                summary_x_scale,
                summary_y_scale,
                categorical_x,
            ),
            width="stretch",
            config=plotly_config(),
            key="frequency_empty_summary",
        )

def scale_range(minimum: float, maximum: float, scale: str) -> list[float]:
    if normalize_axis_scale(scale) == "log":
        lower = max(float(minimum), np.finfo(float).tiny)
        upper = max(float(maximum), lower * (1.0 + 1e-9))
        return [float(np.log10(lower)), float(np.log10(upper))]
    return [float(minimum), float(maximum)]


def render_frequency_results(
    batch: FrequencyBatchResult,
    request: FrequencyRequest,
    series_labels: dict[str, str],
    individual_chart_target: Any,
    summary_chart_target: Any,
) -> None:
    if not batch.results:
        st.error("Frequency analysis did not produce any valid run.")
        if batch.failures:
            for source_file, message in batch.failures.items():
                st.caption(f"{source_file}: {message}")
        return

    peak_min_hz, peak_max_hz, peak_count, _, histogram_bin_width_hz = (
        request.peak_settings
    )
    sample_rates = [result.sample_rate_hz for result in batch.results.values()]
    st.caption(
        f"Applied: {len(batch.results)} run(s) | Sampling rate: "
        f"{min(sample_rates):.3g}-{max(sample_rates):.3g} Hz | "
        f"Top {peak_count} peak(s) per run"
    )

    individual_figure = go.Figure()
    peak_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    notch_rows: list[str] = []

    for index, (source_file, result) in enumerate(batch.results.items()):
        label = series_labels.get(source_file, source_file)
        color = IFF_PALETTE[index % len(IFF_PALETTE)]
        if request.individual_plot == "FFT":
            individual_figure.add_trace(go.Scatter(
                x=result.frequency_hz,
                y=result.fft_amplitude,
                mode="lines",
                line={"color": color, "width": 1.6},
                name=label,
                legendgroup=source_file,
                showlegend=request.show_legend,
            ))
            if request.show_peaks and result.peaks_hz.size:
                individual_figure.add_trace(go.Scatter(
                    x=result.peaks_hz,
                    y=result.peak_amplitudes,
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 8,
                        "symbol": "circle-open",
                        "line": {"width": 1.5},
                    },
                    name=f"{label} peaks",
                    legendgroup=source_file,
                    showlegend=False,
                    hovertemplate=(
                        f"{label}<br>Frequency: %{{x:.4g}} Hz"
                        "<br>Amplitude: %{y:.4g}<extra></extra>"
                    ),
                ))
        elif request.individual_plot == "PSD":
            individual_figure.add_trace(go.Scatter(
                x=result.psd_frequency_hz,
                y=result.psd,
                mode="lines",
                line={"color": color, "width": 1.6},
                name=label,
                legendgroup=source_file,
                showlegend=request.show_legend,
            ))
            if request.show_peaks and result.peaks_hz.size:
                peak_psd = np.interp(
                    result.peaks_hz,
                    result.psd_frequency_hz,
                    result.psd,
                )
                individual_figure.add_trace(go.Scatter(
                    x=result.peaks_hz,
                    y=peak_psd,
                    mode="markers",
                    marker={
                        "color": color,
                        "size": 8,
                        "symbol": "circle-open",
                        "line": {"width": 1.5},
                    },
                    name=f"{label} peaks",
                    legendgroup=source_file,
                    showlegend=False,
                    hovertemplate=(
                        f"{label}<br>Frequency: %{{x:.4g}} Hz"
                        "<br>PSD: %{y:.4g}<extra></extra>"
                    ),
                ))
        else:
            centers = (result.energy_left_hz + result.energy_right_hz) / 2.0
            individual_figure.add_trace(go.Scatter(
                x=centers,
                y=result.energy,
                mode="lines+markers",
                line={"color": color, "width": 1.5},
                marker={"color": color, "size": 4},
                name=label,
                legendgroup=source_file,
                showlegend=request.show_legend,
            ))

        ranked = np.argsort(result.peak_amplitudes)[::-1]
        for rank, peak_index in enumerate(ranked, start=1):
            peak_rows.append({
                "data_series": label,
                "source_file": source_file,
                "rank": rank,
                "frequency_Hz": float(result.peaks_hz[peak_index]),
                "amplitude": float(result.peak_amplitudes[peak_index]),
            })
        warnings.extend(f"{label}: {warning}" for warning in result.warnings)
        if result.notch_peaks_hz:
            values = ", ".join(f"{value:.4g}" for value in result.notch_peaks_hz)
            notch_rows.append(f"{label}: {values} Hz")

    if request.individual_plot == "FFT":
        individual_x = math_axis_title("<i>f</i>", "Hz")
        individual_y = spectrum_axis_title("FFT", request.signal_column)
    elif request.individual_plot == "PSD":
        individual_x = math_axis_title("<i>f</i>", "Hz")
        individual_y = spectrum_axis_title("PSD", request.signal_column)
    else:
        individual_x = math_axis_title("<i>f</i>", "Hz")
        individual_y = spectrum_axis_title("Energy", request.signal_column)
    configure_analysis_figure(
        individual_figure,
        individual_x,
        individual_y,
        520,
        request.individual_x_scale,
        request.individual_y_scale,
    )
    individual_figure.update_layout(
        showlegend=request.show_legend,
        legend={"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 9}},
    )
    individual_figure.update_xaxes(
        range=scale_range(
            peak_min_hz, peak_max_hz, request.individual_x_scale
        )
    )

    peak_table = pd.DataFrame(peak_rows)
    summary_figure = go.Figure()
    if not peak_table.empty:
        if request.summary_plot == "Peak frequency histogram":

            summary_figure.add_trace(go.Histogram(
                x=peak_table["frequency_Hz"],
                xbins={
                    "start": peak_min_hz,
                    "end": peak_max_hz,
                    "size": histogram_bin_width_hz,
                },
                marker={"color": "#0075CF"},
                hovertemplate=(
                    "Frequency: %{x:.4g} Hz<br>Count: %{y}<extra></extra>"
                ),
            ))
            summary_x = math_axis_title("<i>f</i><sub>p</sub>", "Hz")
            summary_y = math_axis_title("<i>N</i><sub>p</sub>", "-")
        elif request.summary_plot == "Peak amplitude histogram":
            summary_figure.add_trace(go.Histogram(
                x=peak_table["amplitude"],

                marker={"color": "#00A6A6"},
                hovertemplate=(
                    "Amplitude: %{x:.4g}<br>Count: %{y}<extra></extra>"
                ),
            ))
            summary_x = spectrum_axis_title("FFT", request.signal_column)
            summary_y = math_axis_title("<i>N</i><sub>p</sub>", "-")
        else:
            dominant = peak_table.loc[peak_table["rank"] == 1].copy()
            summary_figure.add_trace(go.Bar(
                x=dominant["data_series"],
                y=dominant["frequency_Hz"],
                marker={
                    "color": [
                        IFF_PALETTE[index % len(IFF_PALETTE)]
                        for index in range(len(dominant))
                    ]
                },
                customdata=dominant[["source_file"]],
                hovertemplate=(
                    "%{x}<br>Dominant frequency: %{y:.4g} Hz"
                    "<br>%{customdata[0]}<extra></extra>"
                ),
            ))
            summary_x = "Data series [-]"
            summary_y = math_axis_title("<i>f</i><sub>dom</sub>", "Hz")
    else:
        summary_x = math_axis_title("<i>f</i><sub>p</sub>", "Hz")
        summary_y = math_axis_title("<i>N</i><sub>p</sub>", "-")
        summary_figure.add_annotation(
            text="No peaks detected in the applied range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    configure_analysis_figure(
        summary_figure,
        summary_x,
        summary_y,
        520,
        request.summary_x_scale,
        request.summary_y_scale,
    )
    summary_figure.update_layout(showlegend=False)
    if request.summary_plot == "Peak frequency histogram":
        summary_figure.update_xaxes(
            range=scale_range(
                peak_min_hz, peak_max_hz, request.summary_x_scale
            )
        )
    elif request.summary_plot == "Dominant frequency by run":
        summary_figure.update_xaxes(type="category")
        summary_figure.update_yaxes(
            range=scale_range(
                peak_min_hz, peak_max_hz, request.summary_y_scale
            )
        )

    with individual_chart_target.container():
        st.plotly_chart(
            individual_figure, width="stretch", config=plotly_config()
        )
    with summary_chart_target.container():
        st.plotly_chart(summary_figure, width="stretch", config=plotly_config())

    with st.expander(f"Detected peaks by run ({len(peak_table)})"):
        if peak_table.empty:
            st.info("No peaks were detected in the applied frequency range.")
        else:
            st.dataframe(peak_table, width="stretch", hide_index=True, height=260)
            st.download_button(
                "Peaks CSV",
                data=peak_table.to_csv(index=False).encode("utf-8"),
                file_name="vader_frequency_peaks.csv",
                mime="text/csv",
                key="frequency_peaks_csv",
                icon=":material/download:",
                width="content",
            )

    if notch_rows:
        with st.expander(f"Auto-notch selections ({len(notch_rows)})"):
            for row in notch_rows:
                st.caption(row)
    all_failures = [
        f"{series_labels.get(source_file, source_file)}: {message}"
        for source_file, message in batch.failures.items()
    ]
    all_messages = [*warnings, *all_failures]
    if all_messages:
        with st.expander(f"Frequency warnings ({len(all_messages)})"):
            for message in all_messages:
                st.caption(message)


def configure_analysis_figure(
    figure: go.Figure,
    x_title: str,
    y_title: str,
    height: int,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> None:
    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 18, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        hovermode="closest",
        font={"color": "#344054", "size": 11},
    )
    figure.update_xaxes(
        title=x_title,
        type=normalize_axis_scale(x_scale),
    )
    figure.update_yaxes(
        title=y_title,
        type=normalize_axis_scale(y_scale),
    )
    style_axes(figure)

if __name__ == "__main__":
    if os.environ.get(STREAMLIT_BOOTSTRAP_ENV) == "1" or running_in_streamlit():
        main()
    else:
        os.environ[STREAMLIT_BOOTSTRAP_ENV] = "1"
        from streamlit.web import cli as streamlit_cli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        streamlit_cli.main()