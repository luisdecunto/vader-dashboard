from __future__ import annotations

import re
from dataclasses import dataclass

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


DERIVED_COLUMNS = [
    "area_mm2",
    "stress_Pa",
    "surface_tension_stress_Pa",
    "net_stress_Pa",
    "hencky_strain",
    "hencky_strain_rate_1_s",
    "extensional_viscosity_Pa_s",
]
MEASURED_PROCESS_COLUMNS = ("force_g", "radial_Hencky_strain")
VARIANT_COLUMN = "processing_variant"
TRACE_COLUMN = "trace_id"


@dataclass(frozen=True)
class PhysicalSettings:
    surface_tension_mN_m: float = 72.0
    capillary_factor: float = 2.0
    force_zero_g: float = 0.0
    min_abs_strain_rate: float = 1e-5
    crop_enabled: bool = True
    crop_strain_column: str = "radial_Hencky_strain"
    crop_threshold: float = 7.0
    crop_min_time_s: float = 0.0001
    force_tail_offset_enabled: bool = True


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
def parse_filter_workflow(expression: str) -> tuple[FilterStep, ...]:
    text = re.sub(r"\s+", "", expression).upper()
    if text in {"", "RAW", "NONE"}:
        return ()
    return tuple(_parse_workflow_sequence(text))


def format_filter_workflow(steps: tuple[FilterStep, ...]) -> str:
    labels: list[str] = []
    for step in steps:
        values = step.parameters
        if step.operation == "lowpass":
            labels.append(f"LP{values[0]:g}")
        elif step.operation == "highpass":
            labels.append(f"HP{values[0]:g}")
        elif step.operation == "moving_average":
            labels.append(f"MA{int(values[0])}")
        elif step.operation == "savgol":
            labels.append(f"SG({int(values[0])},{int(values[1])})")
        elif step.operation == "whittaker":
            labels.append(f"WH{values[0]:g}")
        elif step.operation == "notch":
            labels.append("NOTCH(" + ",".join(f"{value:g}" for value in values) + ")")
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
        values = step.parameters
        if step.operation == "moving_average":
            suffix = "" if int(values[0]) == 21 else f",{int(values[0])}"
            formula = f"MA({formula}{suffix})"
        elif step.operation == "lowpass":
            formula = f"LP({formula},{values[0]:g})"
        elif step.operation == "highpass":
            formula = f"HP({formula},{values[0]:g})"
        elif step.operation == "savgol":
            suffix = (
                "" if (int(values[0]), int(values[1])) == (21, 3)
                else f",{int(values[0])},{int(values[1])}"
            )
            formula = f"SG({formula}{suffix})"
        elif step.operation == "whittaker":
            suffix = "" if values[0] == 1_000 else f",{values[0]:g}"
            formula = f"WH({formula}{suffix})"
        elif step.operation == "notch":
            defaults = (0.5, 10.0, 3.0, 30.0)
            suffix = "" if values == defaults else "," + ",".join(
                f"{value:g}" for value in values
            )
            formula = f"NOTCH({formula}{suffix})"
    return formula



def parse_filter_formula_list(
    expression_text: str,
    source_column: str,
) -> tuple[FormulaVariant, ...]:
    formulas = split_formula_list(expression_text)
    if not formulas:
        formulas = [source_column]

    variants: list[FormulaVariant] = []
    seen: set[str] = set()
    for expression in formulas:
        expanded = expand_filter_formula_shorthand(expression, source_column)
        parsed_source, steps = parse_filter_formula(expanded)
        if parsed_source != source_column:
            raise ValueError(f"Formula must start from {source_column}.")
        normalized = format_filter_formula(source_column, steps)
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
    functions = "LP|LOWPASS|HP|HIGHPASS|MA|MOVINGAVERAGE|SG|SAVGOL|WH|WHITTAKER|NT|NOTCH"
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
        return expression, ()

    function_name, arguments_text = outer
    arguments = _split_formula_arguments(arguments_text)
    if not arguments:
        raise ValueError(f"{function_name} requires a source signal.")
    source_column, steps = _parse_filter_formula_node(arguments[0])
    try:
        parameters = tuple(float(value) for value in arguments[1:])
    except ValueError as exc:
        raise ValueError(f"{function_name} parameters must be numbers.") from exc

    operation = {
        "MA": "moving_average", "MOVINGAVERAGE": "moving_average",
        "LP": "lowpass", "LOWPASS": "lowpass",
        "HP": "highpass", "HIGHPASS": "highpass",
        "SG": "savgol", "SAVGOL": "savgol",
        "WH": "whittaker", "WHITTAKER": "whittaker",
        "NT": "notch", "NOTCH": "notch",
    }.get(function_name.upper())
    if operation is None:
        raise ValueError(f"Unknown filter function '{function_name}'.")

    if operation == "moving_average":
        if len(parameters) > 1:
            raise ValueError("MA accepts an optional window.")
        step = _validated_step(operation, parameters or (21.0,))
    elif operation in {"lowpass", "highpass"}:
        if len(parameters) != 1:
            raise ValueError(f"{function_name.upper()} requires one cutoff frequency.")
        step = _validated_step(operation, parameters)
    elif operation == "savgol":
        if len(parameters) > 2:
            raise ValueError("SG accepts an optional window and polynomial order.")
        window = parameters[0] if parameters else 21.0
        order = parameters[1] if len(parameters) == 2 else 3.0
        step = _validated_step(operation, (window, order))
    elif operation == "whittaker":
        if len(parameters) > 1:
            raise ValueError("WH accepts an optional lambda.")
        step = _validated_step(operation, parameters or (1_000.0,))
    else:
        if len(parameters) > 4:
            raise ValueError("NOTCH accepts min, max, count, and Q parameters.")
        minimum = parameters[0] if parameters else 0.5
        maximum = parameters[1] if len(parameters) >= 2 else 10.0
        count = parameters[2] if len(parameters) >= 3 else 3.0
        quality = parameters[3] if len(parameters) == 4 else 30.0
        step = _validated_step(
            operation,
            (minimum, maximum, count, quality),
        )
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
    aliases = {
        "LP": "lowpass", "LOWPASS": "lowpass",
        "HP": "highpass", "HIGHPASS": "highpass",
        "MA": "moving_average", "MOVINGAVERAGE": "moving_average",
        "SG": "savgol", "SAVGOL": "savgol",
        "WH": "whittaker", "WHITTAKER": "whittaker",
        "NT": "notch", "NOTCH": "notch",
    }
    operation = aliases.get(head)
    if operation is None:
        return None
    try:
        values = tuple(float(value) for value in argument.split(","))
    except ValueError:
        return None
    if operation in {"lowpass", "highpass", "moving_average", "whittaker"} and len(values) == 1:
        return _validated_step(operation, values)
    if operation == "savgol" and len(values) in {1, 2}:
        order = values[1] if len(values) == 2 else 3.0
        return _validated_step(operation, (values[0], order))
    if operation == "notch" and 2 <= len(values) <= 4:
        count = values[2] if len(values) >= 3 else 3.0
        quality = values[3] if len(values) == 4 else 30.0
        return _validated_step(operation, (values[0], values[1], count, quality))
    return None


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
    if operation in {"lowpass", "highpass"} and parameters[0] <= 0:
        raise ValueError("Filter cutoffs must be greater than zero.")
    if operation == "moving_average":
        window = int(parameters[0])
        if window < 1 or not np.isclose(parameters[0], window):
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

    result["hencky_strain"] = pd.to_numeric(
        result["radial_Hencky_strain"], errors="coerce"
    )

    result["hencky_strain_rate_1_s"] = np.nan
    for _, group in result.groupby("source_file", sort=False):
        ordered = group.sort_values("time_from_onset_s")
        times = ordered["time_from_onset_s"].to_numpy(dtype=float)
        strain = ordered["hencky_strain"].to_numpy(dtype=float)
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
            "Force and radial-strain formula lists must have the same length, "
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
            warnings.append(f"unknown workflow step '{step.operation}' skipped")
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


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))