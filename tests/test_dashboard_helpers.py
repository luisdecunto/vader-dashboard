import unittest
from dataclasses import fields, replace

import numpy as np
import pandas as pd

from vader_dashboard import (
    DATA_PLOT_HEIGHT,
    DEFAULT_POSTPROCESSING_FORCE_FORMULA,
    DEFAULT_STRAIN_FORMULA,
    LinearRangeResult,
    POSTPROCESSING_SMALL_PLOT_HEIGHT,
    POSTPROCESSING_TALL_PLOT_HEIGHT,
    PostprocessingAnalysis,
    DERIVED_COLUMNS,
    DERIVED_QUANTITY_DEFINITIONS,
    FILTER_CONTROL_SPECS,
    FilterSettings,
    FormulaVariant,
    FrequencyRequest,
    PHYSICS_CONTROL_SPECS,
    PROCESSED_AXIS_COLUMNS,
    PhysicalSettings,
    PlotRequest,
    VELOCITY_COLUMN,
    WORKSPACE_DATA,
    WORKSPACE_DEFINITIONS,
    WORKSPACE_FREQUENCY,
    WORKSPACE_POSTPROCESSING,
    WORKSPACE_SUMMARY,
    column_axis_title,
    column_display_label,
    column_menu_label,
    frequency_analysis_key,
    make_empty_plot_figure,
    make_postprocessing_figures,
    normalize_axis_column,
    normalize_workspace,
    parse_filter_formula,
    parse_filter_formula_list,
    parse_velocity_value,
    plot_analysis_key,
    sorted_velocity_values,
)


class RequestStateTests(unittest.TestCase):
    def test_plot_view_changes_do_not_change_analysis_key(self) -> None:
        settings = FilterSettings()
        request = PlotRequest(
            x_column="time_from_onset_s",
            y_column="force_g",
            selected_files=("run.csv",),
            force_variants=(FormulaVariant("force_g", "Raw", settings),),
            strain_variants=(
                FormulaVariant("radial_Hencky_strain", "Raw", settings),
            ),
            show_raw_overlay=False,
            show_legend=True,
            x_scale="linear",
            y_scale="linear",
            selected_materials=("material",),
            selected_velocities=("5mms",),
            physical_settings=PhysicalSettings(),
        )
        changed_view = replace(
            request,
            show_raw_overlay=True,
            show_legend=False,
            x_scale="log",
            y_scale="log",
        )
        self.assertEqual(plot_analysis_key(request), plot_analysis_key(changed_view))

    def test_frequency_view_changes_do_not_change_analysis_key(self) -> None:
        request = FrequencyRequest(
            selected_files=("run.csv",),
            signal_column="force_g",
            filter_settings=FilterSettings(),
            peak_settings=(0.1, 10.0, 3, 20, 12),
            individual_plot="FFT",
            summary_plot="Peak frequency histogram",
            show_peaks=True,
            show_legend=True,
            individual_x_scale="linear",
            individual_y_scale="linear",
            summary_x_scale="linear",
            summary_y_scale="linear",
            selected_materials=("material",),
            selected_velocities=("5mms",),
            physical_settings=PhysicalSettings(),
        )
        changed_view = replace(
            request,
            individual_plot="PSD",
            summary_plot="Dominant frequency by run",
            show_peaks=False,
            show_legend=False,
            individual_x_scale="log",
            summary_y_scale="log",
        )
        self.assertEqual(
            frequency_analysis_key(request),
            frequency_analysis_key(changed_view),
        )

class ExpertLayerRegistryTests(unittest.TestCase):
    def test_physics_controls_cover_every_setting(self) -> None:
        setting_fields = {field.name for field in fields(PhysicalSettings)}
        control_fields = {
            specification.field_name
            for specification in PHYSICS_CONTROL_SPECS
        }
        self.assertEqual(control_fields, setting_fields)

    def test_derived_columns_and_filter_labels_come_from_registries(self) -> None:
        self.assertEqual(
            DERIVED_COLUMNS,
            [
                definition.column
                for definition in DERIVED_QUANTITY_DEFINITIONS
            ],
        )
        self.assertEqual(
            [specification.ui_label for specification in FILTER_CONTROL_SPECS],
            ["MA", "LP", "HP", "SG", "WH", "Notch"],
        )


class WorkspaceNavigationTests(unittest.TestCase):
    def test_workspaces_follow_analysis_flow(self) -> None:
        self.assertEqual(
            [definition[0] for definition in WORKSPACE_DEFINITIONS],
            [
                WORKSPACE_DATA,
                WORKSPACE_FREQUENCY,
                WORKSPACE_POSTPROCESSING,
                WORKSPACE_SUMMARY,
            ],
        )

    def test_legacy_workspace_names_migrate(self) -> None:
        self.assertEqual(
            normalize_workspace("Filtered / processed"), WORKSPACE_DATA
        )
        self.assertEqual(normalize_workspace("Raw data"), WORKSPACE_DATA)
        self.assertEqual(normalize_workspace("Summary plots"), WORKSPACE_SUMMARY)
        self.assertEqual(normalize_workspace("unknown"), WORKSPACE_DATA)


class DashboardDisplayTests(unittest.TestCase):
    def test_velocities_are_parsed_and_sorted_numerically(self) -> None:
        self.assertEqual(parse_velocity_value("10mms"), 10.0)
        self.assertEqual(
            sorted_velocity_values(["10mms", "5mms", "2mms"]),
            ["2mms", "5mms", "10mms"],
        )

    def test_hs_alias_is_canonicalized_for_processing(self) -> None:
        source, steps = parse_filter_formula("LP(HS,20)")
        self.assertEqual(source, "radial_Hencky_strain")
        self.assertEqual(steps[0].operation, "lowpass")
        legacy_source, _ = parse_filter_formula("LP(HD,20)")
        self.assertEqual(legacy_source, "radial_Hencky_strain")
        variants = parse_filter_formula_list(
            "HS; LP(HS,20)", "radial_Hencky_strain"
        )
        self.assertEqual(
            [variant.expression for variant in variants],
            ["HS", "LP(HS,20)"],
        )

    def test_default_strain_formula_is_121_point_moving_average(self) -> None:
        variants = parse_filter_formula_list(
            DEFAULT_STRAIN_FORMULA, "radial_Hencky_strain"
        )
        self.assertEqual(len(variants), 1)
        workflow = variants[0].settings.workflow
        self.assertEqual(len(workflow), 1)
        self.assertEqual(workflow[0].operation, "moving_average")
        self.assertEqual(workflow[0].parameters, (121.0,))
        source, force_workflow = parse_filter_formula(
            DEFAULT_POSTPROCESSING_FORCE_FORMULA
        )
        self.assertEqual(source, "force_g")
        self.assertEqual(
            [step.operation for step in force_workflow],
            ["lowpass", "moving_average"],
        )

    def test_axis_menus_use_math_notation_without_duplicate_strain(self) -> None:
        self.assertEqual(
            [column_menu_label(column) for column in PROCESSED_AXIS_COLUMNS],
            [
                "t",
                "L\u1d65",
                "\u03b5\u1d63",
                "\u03b5_z",
                "D/D\u2080",
                "F",
                "D",
                "A",
                "\u03c3",
                "\u03c3_surf",
                "\u0394\u03c3",
                "\u03b5\u0307\u1d63",
                "\u03b7\u2091",
                "v",
            ],
        )
        self.assertNotIn("hencky_strain", PROCESSED_AXIS_COLUMNS)
        self.assertEqual(
            normalize_axis_column("hencky_strain"), "radial_Hencky_strain"
        )

    def test_empty_plot_has_axes_but_no_data(self) -> None:
        figure = make_empty_plot_figure(
            "time_from_onset_s",
            "force_g",
            x_scale="linear",
            y_scale="log",
        )
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(figure.layout.height, DATA_PLOT_HEIGHT)
        self.assertEqual(
            figure.layout.xaxis.title.text,
            column_axis_title("time_from_onset_s"),
        )
        self.assertEqual(
            figure.layout.yaxis.title.text,
            column_axis_title("force_g"),
        )
        self.assertEqual(figure.layout.xaxis.type, "linear")
        self.assertEqual(figure.layout.yaxis.type, "log")

    def test_postprocessing_figures_mark_the_working_window(self) -> None:
        time_s = np.arange(10, dtype=float)
        strain = 0.2 * time_s
        frame = pd.DataFrame({
            "source_file": "run.csv",
            "time_from_onset_s": time_s,
            "radial_Hencky_strain": strain,
            "force_g": 3.0 * time_s,
        })
        analysis = PostprocessingAnalysis(
            frame=frame,
            windows={
                "run.csv": LinearRangeResult(
                    endpoint_index=5,
                    endpoint_x=5.0,
                    slope=0.2,
                    intercept=0.0,
                    r2=1.0,
                )
            },
            warnings=[],
        )

        top, strain_plot, right = make_postprocessing_figures(
            analysis,
            "force_g",
            {"run.csv": "Run"},
            show_legend=True,
        )

        self.assertEqual(top.layout.height, POSTPROCESSING_SMALL_PLOT_HEIGHT)
        self.assertEqual(
            strain_plot.layout.height, POSTPROCESSING_SMALL_PLOT_HEIGHT
        )
        self.assertEqual(right.layout.height, POSTPROCESSING_TALL_PLOT_HEIGHT)
        self.assertEqual(len(top.data), 2)
        self.assertEqual(len(strain_plot.data), 4)
        self.assertEqual(len(right.data), 2)
        self.assertEqual(len(top.layout.shapes), 1)
        self.assertEqual(len(strain_plot.layout.shapes), 5)
        self.assertEqual(len(right.layout.shapes), 1)
        self.assertEqual(top.data[1].line.color, "rgba(107, 114, 128, 0.34)")

    def test_plot_labels_use_scientific_notation_and_units(self) -> None:
        self.assertEqual(column_display_label("radial_Hencky_strain"), "HS strain")
        self.assertEqual(column_display_label(VELOCITY_COLUMN), "Velocity")
        self.assertIn("ε<sub>r</sub>", column_axis_title("radial_Hencky_strain"))
        self.assertIn("[-]", column_axis_title("radial_Hencky_strain"))
        self.assertIn("mm s<sup>-1</sup>", column_axis_title(VELOCITY_COLUMN))


if __name__ == "__main__":
    unittest.main()
