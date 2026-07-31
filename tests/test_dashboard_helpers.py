import unittest

from vader_dashboard import (
    VELOCITY_COLUMN,
    column_axis_title,
    column_display_label,
    parse_filter_formula,
    parse_filter_formula_list,
    parse_velocity_value,
    sorted_velocity_values,
)


class DashboardDisplayTests(unittest.TestCase):
    def test_velocities_are_parsed_and_sorted_numerically(self) -> None:
        self.assertEqual(parse_velocity_value("10mms"), 10.0)
        self.assertEqual(
            sorted_velocity_values(["10mms", "5mms", "2mms"]),
            ["2mms", "5mms", "10mms"],
        )

    def test_hd_alias_is_canonicalized_for_processing(self) -> None:
        source, steps = parse_filter_formula("LP(HD,20)")
        self.assertEqual(source, "radial_Hencky_strain")
        self.assertEqual(steps[0].operation, "lowpass")
        variants = parse_filter_formula_list(
            "HD; LP(HD,20)", "radial_Hencky_strain"
        )
        self.assertEqual(
            [variant.expression for variant in variants],
            ["HD", "LP(HD,20)"],
        )

    def test_plot_labels_use_scientific_notation_and_units(self) -> None:
        self.assertEqual(column_display_label("radial_Hencky_strain"), "HD strain")
        self.assertEqual(column_display_label(VELOCITY_COLUMN), "Velocity")
        self.assertIn("ε<sub>HD</sub>", column_axis_title("radial_Hencky_strain"))
        self.assertIn("[-]", column_axis_title("radial_Hencky_strain"))
        self.assertIn("mm s<sup>-1</sup>", column_axis_title(VELOCITY_COLUMN))


if __name__ == "__main__":
    unittest.main()
