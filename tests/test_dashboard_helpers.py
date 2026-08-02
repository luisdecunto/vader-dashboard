import unittest
from dataclasses import fields

from vader_dashboard import (
    DERIVED_COLUMNS,
    DERIVED_QUANTITY_DEFINITIONS,
    FILTER_CONTROL_SPECS,
    PHYSICS_CONTROL_SPECS,
    PROCESSED_AXIS_COLUMNS,
    PhysicalSettings,
    VELOCITY_COLUMN,
    WORKSPACE_DATA,
    WORKSPACE_DEFINITIONS,
    WORKSPACE_FREQUENCY,
    WORKSPACE_POSTPROCESSING,
    WORKSPACE_SUMMARY,
    column_axis_title,
    column_display_label,
    column_menu_label,
    normalize_axis_column,
    normalize_workspace,
    parse_filter_formula,
    parse_filter_formula_list,
    parse_velocity_value,
    sorted_velocity_values,
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

    def test_plot_labels_use_scientific_notation_and_units(self) -> None:
        self.assertEqual(column_display_label("radial_Hencky_strain"), "HS strain")
        self.assertEqual(column_display_label(VELOCITY_COLUMN), "Velocity")
        self.assertIn("ε<sub>r</sub>", column_axis_title("radial_Hencky_strain"))
        self.assertIn("[-]", column_axis_title("radial_Hencky_strain"))
        self.assertIn("mm s<sup>-1</sup>", column_axis_title(VELOCITY_COLUMN))


if __name__ == "__main__":
    unittest.main()
