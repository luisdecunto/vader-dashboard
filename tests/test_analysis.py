import unittest

import numpy as np
import pandas as pd

from vader_dashboard import (
    FilterSettings,
    PhysicalSettings,
    add_derived_columns,
    analyze_frequency,
    analyze_frequency_runs,
    analyze_postprocessing_runs,
    detect_fft_peaks,
    estimate_sampling_interval,
    find_max_initial_linear_range,
    VARIANT_COLUMN,
    format_filter_formula,
    format_filter_workflow,
    parse_filter_formula,
    parse_filter_formula_list,
    parse_filter_workflow,
    preprocess_experiments,
    process_axes_frame,
    process_measured_variant_frames,
    process_signal,
    whittaker_smooth,
)


class DerivedQuantityTests(unittest.TestCase):
    def test_stress_capillary_correction_rate_and_viscosity(self) -> None:
        time_s = np.arange(5.0)
        frame = pd.DataFrame(
            {
                "time_from_onset_s": time_s,
                "diameter_mm": np.full(5, 2.0),
                "force_g": np.full(5, 1.0),
                "radial_Hencky_strain": 0.1 * time_s,
                "vertical_strain": 0.2 * time_s,
                "source_file": "sample.csv",
            }
        )
        result = add_derived_columns(frame, PhysicalSettings())

        area_m2 = np.pi * (0.002 / 2.0) ** 2
        expected_stress = 9.80665e-3 / area_m2
        expected_capillary = (2.0 / np.pi) * 0.072 / 0.002
        expected_net = expected_stress - expected_capillary

        self.assertAlmostEqual(result["area_mm2"].iloc[0], np.pi)
        self.assertAlmostEqual(result["stress_Pa"].iloc[0], expected_stress)
        self.assertAlmostEqual(PhysicalSettings().capillary_factor, 2.0 / np.pi)
        self.assertAlmostEqual(
            result["surface_tension_stress_Pa"].iloc[0], expected_capillary
        )
        self.assertAlmostEqual(result["net_stress_Pa"].iloc[0], expected_net)
        self.assertNotIn("hencky_strain", result.columns)
        np.testing.assert_allclose(result["hencky_strain_rate_1_s"], 0.1)
        np.testing.assert_allclose(
            result["extensional_viscosity_Pa_s"], expected_net / 0.1
        )


class LinearRangeTests(unittest.TestCase):
    def test_initially_linear_then_curved_example(self) -> None:
        x = np.arange(20, dtype=float)
        y = 2.0 * x + 1.0
        y[12:] += 3.0 * (x[12:] - 11.0) ** 2

        result = find_max_initial_linear_range(
            x,
            y,
            r2_threshold=0.999,
            include_residuals=True,
        )

        self.assertEqual(result.endpoint_index, 11)
        self.assertEqual(result.endpoint_x, 11.0)
        self.assertAlmostEqual(result.slope, 2.0)
        self.assertAlmostEqual(result.intercept, 1.0)
        self.assertAlmostEqual(result.r2, 1.0)
        np.testing.assert_allclose(result.residuals, 0.0)

    def test_full_domain_and_constant_signal_pass(self) -> None:
        x = np.linspace(0.0, 5.0, 21)
        linear = find_max_initial_linear_range(
            x, 4.0 * x - 3.0, r2_threshold=0.9999
        )
        constant = find_max_initial_linear_range(
            x,
            np.full_like(x, 7.5),
            r2_threshold=0.9999,
            constrain_to_first=True,
        )

        self.assertEqual(linear.endpoint_index, x.size - 1)
        self.assertEqual(constant.endpoint_index, x.size - 1)
        self.assertEqual(constant.slope, 0.0)
        self.assertEqual(constant.intercept, 7.5)
        self.assertEqual(constant.r2, 1.0)

    def test_no_valid_interval_and_invalid_x_are_handled(self) -> None:
        no_window = find_max_initial_linear_range(
            np.arange(5, dtype=float),
            np.array([0.0, 1.0, 0.0, 2.0, -1.0]),
            r2_threshold=0.999,
            min_points=3,
        )
        self.assertFalse(no_window.valid)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            find_max_initial_linear_range(
                np.array([0.0, 1.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 2.0, 3.0]),
            )

class PreprocessingTests(unittest.TestCase):
    def test_crop_and_tail_force_offset_drop_rows_above_threshold(self) -> None:
        frame = pd.DataFrame(
            {
                "time_from_onset_s": np.arange(6.0),
                "radial_Hencky_strain": [0, 2, 4, 6, 8, 9],
                "force_g": [10, 11, 12, 13, 20, 22],
                "source_file": "sample.csv",
            }
        )
        settings = PhysicalSettings(
            crop_threshold=7.0,
            crop_min_time_s=0.1,
        )
        result = preprocess_experiments(frame, settings)

        np.testing.assert_allclose(
            result.frame["time_from_onset_s"], [1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(result.frame["force_g"], [-10.0, -9.0, -8.0])
        self.assertFalse(result.warnings)

    def test_sampling_interval_uses_median_positive_step(self) -> None:
        time_s = np.array([0.0, 0.1, 0.2, 0.3, 2.3, np.nan])
        self.assertAlmostEqual(estimate_sampling_interval(time_s), 0.1)


class PostprocessingAnalysisTests(unittest.TestCase):
    def test_working_window_starts_at_requested_strain(self) -> None:
        time_s = np.arange(10, dtype=float)
        frame = pd.DataFrame({
            "source_file": "run.csv",
            "time_from_onset_s": time_s,
            "radial_Hencky_strain": 0.5 * time_s,
            "force_g": 2.0 * time_s,
            "diameter_mm": np.full(time_s.size, 2.0),
        })

        result = analyze_postprocessing_runs(
            frame,
            FilterSettings(),
            FilterSettings(),
            "force_g",
            FilterSettings(),
            PhysicalSettings(),
            r2_threshold=0.999,
            min_points=3,
            epsilon_start=1.0,
        )

        self.assertEqual(result.window_starts["run.csv"][0], 2)
        self.assertAlmostEqual(result.window_starts["run.csv"][1], 2.0)
        self.assertAlmostEqual(result.window_starts["run.csv"][2], 1.0)
        self.assertEqual(result.windows["run.csv"].endpoint_index, 9)

class SignalProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_rate = 100.0
        self.time_s = np.arange(0.0, 10.0, 1.0 / self.sample_rate)

    def test_lowpass_suppresses_high_frequency_component(self) -> None:
        values = np.sin(2 * np.pi * self.time_s) + 0.5 * np.sin(
            2 * np.pi * 20.0 * self.time_s
        )
        filtered, _, warnings = process_signal(
            self.time_s,
            values,
            FilterSettings(lowpass_hz=5.0),
        )
        frequency = np.fft.rfftfreq(values.size, 1.0 / self.sample_rate)
        before = np.abs(np.fft.rfft(values))
        after = np.abs(np.fft.rfft(filtered))
        index_20_hz = int(np.argmin(np.abs(frequency - 20.0)))

        self.assertFalse(warnings)
        self.assertLess(after[index_20_hz], before[index_20_hz] * 0.05)

    def test_highpass_suppresses_low_frequency_component(self) -> None:
        values = np.sin(2 * np.pi * 0.5 * self.time_s) + np.sin(
            2 * np.pi * 10.0 * self.time_s
        )
        filtered, _, warnings = process_signal(
            self.time_s, values, FilterSettings(highpass_hz=5.0)
        )
        frequency = np.fft.rfftfreq(values.size, 1.0 / self.sample_rate)
        before = np.abs(np.fft.rfft(values))
        after = np.abs(np.fft.rfft(filtered))
        index_low = int(np.argmin(np.abs(frequency - 0.5)))
        self.assertFalse(warnings)
        self.assertLess(after[index_low], before[index_low] * 0.05)

    def test_auto_notch_selects_and_suppresses_peak(self) -> None:
        values = np.sin(2 * np.pi * 2.0 * self.time_s) + 0.8 * np.sin(
            2 * np.pi * 8.0 * self.time_s
        )
        filtered, peaks, warnings = process_signal(
            self.time_s,
            values,
            FilterSettings(
                notch_enabled=True,
                notch_min_hz=7.0,
                notch_max_hz=9.0,
                notch_peak_count=1,
            ),
        )
        frequency = np.fft.rfftfreq(values.size, 1.0 / self.sample_rate)
        index_notch = int(np.argmin(np.abs(frequency - 8.0)))
        before = np.abs(np.fft.rfft(values))[index_notch]
        after = np.abs(np.fft.rfft(filtered))[index_notch]
        self.assertFalse(warnings)
        self.assertAlmostEqual(peaks[0], 8.0, places=6)
        self.assertLess(after, before * 0.2)

    def test_savgol_smoothing_reduces_noise(self) -> None:
        rng = np.random.default_rng(7)
        clean = np.sin(2 * np.pi * self.time_s)
        noisy = clean + rng.normal(0.0, 0.3, clean.size)
        smoothed, _, warnings = process_signal(
            self.time_s,
            noisy,
            FilterSettings(
                smoothing="Savitzky-Golay",
                savgol_window=21,
                savgol_order=3,
            ),
        )
        self.assertFalse(warnings)
        self.assertLess(np.mean((smoothed - clean) ** 2), np.mean((noisy - clean) ** 2))

    def test_moving_average_reduces_noise(self) -> None:
        rng = np.random.default_rng(19)
        clean = np.sin(2 * np.pi * self.time_s)
        noisy = clean + rng.normal(0.0, 0.35, clean.size)
        smoothed, _, warnings = process_signal(
            self.time_s,
            noisy,
            FilterSettings(workflow=parse_filter_workflow("MA21")),
        )
        self.assertFalse(warnings)
        self.assertLess(np.mean((smoothed - clean) ** 2), np.mean((noisy - clean) ** 2))

    def test_nested_workflow_matches_linear_order(self) -> None:
        nested = parse_filter_workflow("MA(LP20)")
        linear = parse_filter_workflow("LP20 > MA21")
        self.assertEqual(nested, linear)
        self.assertEqual(format_filter_workflow(nested), "LP20 > MA21")

        values = np.sin(2 * np.pi * 2.0 * self.time_s) + 0.4 * np.sin(
            2 * np.pi * 30.0 * self.time_s
        )
        nested_values, _, _ = process_signal(
            self.time_s, values, FilterSettings(workflow=nested)
        )
        linear_values, _, _ = process_signal(
            self.time_s, values, FilterSettings(workflow=linear)
        )
        np.testing.assert_allclose(nested_values, linear_values)

    def test_excel_style_filter_formula(self) -> None:
        source, steps = parse_filter_formula("LP(MA(force_g),20)")
        self.assertEqual(source, "force_g")
        self.assertEqual(
            [step.operation for step in steps],
            ["moving_average", "lowpass"],
        )
        self.assertEqual(
            format_filter_formula(source, steps),
            "LP(MA(force_g),20)",
        )
    def test_x_and_y_formulas_are_processed_independently(self) -> None:
        frame = pd.DataFrame(
            {
                "time_from_onset_s": self.time_s,
                "x_signal": np.sin(2 * np.pi * 2.0 * self.time_s),
                "y_signal": np.cos(2 * np.pi * 4.0 * self.time_s),
                "source_file": "sample.csv",
            }
        )
        x_settings = FilterSettings(workflow=parse_filter_workflow("MA5"))
        y_settings = FilterSettings(workflow=parse_filter_workflow("MA17"))
        result = process_axes_frame(
            frame, "x_signal", x_settings, "y_signal", y_settings
        )
        expected_x, _, _ = process_signal(
            self.time_s, frame["x_signal"].to_numpy(), x_settings
        )
        expected_y, _, _ = process_signal(
            self.time_s, frame["y_signal"].to_numpy(), y_settings
        )
        np.testing.assert_allclose(result.frame["x_signal__x_processed"], expected_x)
        np.testing.assert_allclose(result.frame["y_signal__y_processed"], expected_y)
    def test_invalid_workflow_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_filter_workflow("MA0")
        with self.assertRaises(ValueError):
            parse_filter_workflow("LP20 >")
    def test_peak_detection_respects_frequency_range(self) -> None:
        values = np.sin(2 * np.pi * 3.0 * self.time_s) + 0.4 * np.sin(
            2 * np.pi * 8.0 * self.time_s
        )
        peaks, _ = detect_fft_peaks(values, self.sample_rate, 2.0, 4.0, 3)
        self.assertEqual(peaks.size, 1)
        self.assertAlmostEqual(peaks[0], 3.0, places=6)

    def test_whittaker_smoother_reduces_curvature(self) -> None:
        rng = np.random.default_rng(42)
        noisy = np.sin(self.time_s) + rng.normal(0.0, 0.2, self.time_s.size)
        smoothed = whittaker_smooth(noisy, 1000.0)
        self.assertEqual(smoothed.shape, noisy.shape)
        self.assertLess(
            np.var(np.diff(smoothed, n=2)),
            np.var(np.diff(noisy, n=2)),
        )

    def test_formula_list_accepts_shorthand_variants(self) -> None:
        variants = parse_filter_formula_list(
            "LP(20), LP(50); MA(LP(20),1000)", "force_g"
        )

        self.assertEqual(
            [variant.expression for variant in variants],
            ["LP(force_g,20)", "LP(force_g,50)", "MA(LP(force_g,20),1000)"],
        )

    def test_measured_processing_precedes_derived_quantities(self) -> None:
        frame = pd.DataFrame(
            {
                "time_from_onset_s": np.arange(7.0),
                "diameter_mm": np.full(7, 2.0),
                "force_g": [0, 0, 9, 0, 0, 0, 0],
                "radial_Hencky_strain": np.linspace(0.0, 0.6, 7),
                "vertical_strain": np.linspace(0.0, 1.2, 7),
                "source_file": "sample.csv",
            }
        )
        force_variants = parse_filter_formula_list("MA(force_g,3)", "force_g")
        strain_variants = parse_filter_formula_list(
            "radial_Hencky_strain", "radial_Hencky_strain"
        )

        result = process_measured_variant_frames(
            frame, force_variants, strain_variants, PhysicalSettings()
        ).frame

        self.assertEqual(result[VARIANT_COLUMN].iloc[0], "F: MA(force_g,3)")
        self.assertLess(result["force_g"].max(), 9.0)
        area_m2 = np.pi * (0.002 / 2.0) ** 2
        np.testing.assert_allclose(
            result["stress_Pa"], result["force_g"] * 9.80665e-3 / area_m2
        )
    def test_frequency_analysis_returns_fft_psd_energy_and_peaks(self) -> None:
        values = np.sin(2 * np.pi * 3.0 * self.time_s)
        frame = pd.DataFrame(
            {"time_from_onset_s": self.time_s, "force_g": values}
        )
        result = analyze_frequency(
            frame,
            "force_g",
            FilterSettings(),
            1.0,
            5.0,
            5,
            12,
        )
        self.assertEqual(result.energy.size, 12)
        self.assertGreater(result.psd.size, 0)
        self.assertTrue(np.any(np.isclose(result.peaks_hz, 3.0)))

    def test_frequency_analysis_handles_multiple_runs_independently(self) -> None:
        sample_rate = 100.0
        time_s = np.arange(0.0, 4.0, 1.0 / sample_rate)
        frame = pd.concat(
            [
                pd.DataFrame({
                    "time_from_onset_s": time_s,
                    "force_g": np.sin(2.0 * np.pi * frequency * time_s),
                    "source_file": source_file,
                })
                for source_file, frequency in (
                    ("run_a.csv", 3.0),
                    ("run_b.csv", 7.0),
                )
            ],
            ignore_index=True,
        )

        batch = analyze_frequency_runs(
            frame, "force_g", FilterSettings(), 1.0, 10.0, 3, 12
        )

        self.assertFalse(batch.failures)
        self.assertEqual(set(batch.results), {"run_a.csv", "run_b.csv"})
        self.assertTrue(
            np.any(np.isclose(batch.results["run_a.csv"].peaks_hz, 3.0))
        )
        self.assertTrue(
            np.any(np.isclose(batch.results["run_b.csv"].peaks_hz, 7.0))
        )


if __name__ == "__main__":
    unittest.main()