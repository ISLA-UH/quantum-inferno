import unittest

import numpy as np

import quantum_inferno.utilities.calculations as calc


class CalculationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.timestamps = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])  # this equates to 10s intervals = 0.1Hz
        cls.timeseries = np.array([2, 3, 4, 2, 3, 4, 2, 3, 4])

    def test_integrate_with_cumtrapz_timestamps_s(self):
        result = calc.integrate_with_cumtrapz_timestamps_s(self.timestamps, self.timeseries)
        self.assertEqual(len(result), len(self.timeseries))
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 240)

    def test_integrate_with_cumtrapz_sample_rate_hz(self):
        result = calc.integrate_with_cumtrapz_sample_rate_hz(0.1, self.timeseries)
        self.assertEqual(len(result), len(self.timeseries))
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 240)

    def test_derivative_with_gradient_timestamps_s(self):
        result = calc.derivative_with_gradient_timestamps_s(self.timestamps, self.timeseries)
        self.assertEqual(len(result), len(self.timeseries))
        self.assertEqual(result[0], 0.1)
        self.assertEqual(result[3], -0.05)
        self.assertEqual(result[-1], 0.1)

    def test_derivative_with_gradient_sample_rate_hz(self):
        result = calc.derivative_with_gradient_sample_rate_hz(0.1, self.timeseries)
        self.assertEqual(len(result), len(self.timeseries))
        self.assertEqual(result[0], 0.1)
        self.assertEqual(result[3], -0.05)
        self.assertEqual(result[-1], 0.1)

    def test_get_fill_from_filling_method(self):
        result = calc.get_fill_from_filling_method(self.timeseries, "zero")
        self.assertEqual(result, 0)
        result = calc.get_fill_from_filling_method(self.timeseries, "nan")
        self.assertTrue(np.isnan(result))  # np.nan does not equal itself
        result = calc.get_fill_from_filling_method(self.timeseries, "mean")
        self.assertEqual(result, np.mean(self.timeseries))
        result = calc.get_fill_from_filling_method(self.timeseries, "median")
        self.assertEqual(result, np.median(self.timeseries))
        result = calc.get_fill_from_filling_method(self.timeseries, "min")
        self.assertEqual(result, np.min(self.timeseries))
        result = calc.get_fill_from_filling_method(self.timeseries, "max")
        self.assertEqual(result, np.max(self.timeseries))
        result = calc.get_fill_from_filling_method(self.timeseries, "tail")
        self.assertEqual(result, self.timeseries[-1])
        result = calc.get_fill_from_filling_method(self.timeseries, "head")
        self.assertEqual(result, self.timeseries[0])

    def test_get_fill_from_filling_method_invalid(self):
        with self.assertRaises(ValueError):
            calc.get_fill_from_filling_method(np.array([[1], [2]]), "zero")

    def test_append_fill_start(self):
        result = calc.append_fill(self.timeseries, 0, "start")
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], self.timeseries[-1])

    def test_append_fill_end(self):
        result = calc.append_fill(self.timeseries, 0, "end")
        self.assertEqual(len(result), len(self.timeseries) + 1)
        self.assertEqual(result[0], self.timeseries[0])
        self.assertEqual(result[-1], 0)

    def test_derivative_with_difference_timestamps_s(self):
        result = calc.derivative_with_difference_timestamps_s(self.timestamps, self.timeseries)
        self.assertEqual(len(result), len(self.timeseries))
        self.assertEqual(result[0], 0.1)
        self.assertEqual(result[-1], 0.0)

    def test_derivative_with_difference_sample_rate_hz(self):
        result = calc.derivative_with_difference_sample_rate_hz(0.1, self.timeseries)
        self.assertEqual(len(result), len(self.timeseries))
        self.assertEqual(result[0], 0.1)
        self.assertEqual(result[-1], 0.0)

    def test_round_value(self):
        result = calc.round_value(1.5, "floor")
        self.assertEqual(result, 1)
        result = calc.round_value(1.5, "ceil")
        self.assertEqual(result, 2)
        result = calc.round_value(1.5, "round")
        self.assertEqual(result, 2)

    def test_get_num_points(self):
        result = calc.get_num_points(10.0, 10.0, "round", "points")
        self.assertEqual(result, 100)
        result = calc.get_num_points(10.0, 10.0, "round", "log2")
        self.assertEqual(result, 7)
        result = calc.get_num_points(5.0, 2.0, "round", "pow2")
        self.assertEqual(result, 1024)
