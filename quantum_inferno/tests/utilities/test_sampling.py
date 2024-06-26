import unittest

import numpy as np

import quantum_inferno.utilities.sampling as samp


class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.timeseries = np.array([10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40, 30, 20])

    def test_subsample(self):
        result, new_rate = samp.subsample(self.timeseries, 1.0, 3, "nth")
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], 10)
        self.assertEqual(result[-1], 30)
        self.assertAlmostEqual(new_rate, 0.33, 2)

    def test_subsample_avg(self):
        result, new_rate = samp.subsample(self.timeseries, 1.0, 3, "average")
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 20)
        self.assertEqual(result[-1], 50)
        self.assertAlmostEqual(new_rate, 0.33, 2)

    def test_subsample_med(self):
        result, new_rate = samp.subsample(self.timeseries, 1.0, 3, "median")
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 20)
        self.assertEqual(result[-1], 50)
        self.assertAlmostEqual(new_rate, 0.33, 2)

    def test_subsample_max(self):
        result, new_rate = samp.subsample(self.timeseries, 1.0, 3, "max")
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 30)
        self.assertEqual(result[-1], 60)
        self.assertAlmostEqual(new_rate, 0.33, 2)

    def test_subsample_min(self):
        result, new_rate = samp.subsample(self.timeseries, 1.0, 3, "min")
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 10)
        self.assertEqual(result[-1], 40)
        self.assertAlmostEqual(new_rate, 0.33, 2)

    def test_resample_uneven_timestamps(self):
        timestamps = np.array([10, 15, 22, 31, 41, 46, 53, 60, 75, 79, 89, 92, 93, 100])
        result, new_rate = samp.resample_uneven_timeseries(self.timeseries, timestamps, 0.2)
        self.assertEqual(len(result), 18)
        self.assertEqual(result[0], 10)
        self.assertAlmostEqual(result[-1], 27.14, 2)
        self.assertAlmostEqual(result[4], 38.89, 2)
        self.assertEqual(new_rate, 0.2)

    def test_resample_uneven_timestamps_assumed(self):
        timestamps = np.array([10, 15, 22, 31, 41, 46, 53, 60, 75, 79, 89, 92, 93, 100])
        result, new_rate = samp.resample_uneven_timeseries(self.timeseries, timestamps)
        self.assertEqual(len(result), 13)
        self.assertEqual(result[0], 10)
        self.assertAlmostEqual(result[-1], 29.89, 2)
        self.assertAlmostEqual(result[4], 46.69, 2)
        self.assertAlmostEqual(new_rate, 0.14, 2)

    def test_resample_with_sample_rate(self):
        result, new_rate = samp.resample_with_sample_rate(self.timeseries, 1.0, 0.5)
        self.assertEqual(len(result), 7)
        self.assertAlmostEqual(result[0], 12.47, 2)
        self.assertAlmostEqual(result[-1], 30.32, 2)
        self.assertAlmostEqual(new_rate, 0.5, 2)

    def test_subsample_2d(self):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        result = samp.subsample_2d(array, 2, "nth")
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[-1][-1], 11)

    def test_subsample_2d_avg(self):
        array = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21]])
        result = samp.subsample_2d(array, 3, "average")
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[0][0], 2)
        self.assertEqual(result[-1][-1], 19)

    def test_subsample_2d_med(self):
        array = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21]])
        result = samp.subsample_2d(array, 3, "median")
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[0][0], 2)
        self.assertEqual(result[-1][-1], 19)

    def test_subsample_2d_max(self):
        array = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21]])
        result = samp.subsample_2d(array, 3, "max")
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[0][0], 3)
        self.assertEqual(result[-1][-1], 20)

    def test_subsample_2d_min(self):
        array = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21]])
        result = samp.subsample_2d(array, 3, "min")
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[-1][-1], 18)

    def test_decimate_timeseries(self):
        result = samp.decimate_timeseries(np.concatenate([self.timeseries, self.timeseries]), 3)
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result[0], 9.80, 2)
        self.assertAlmostEqual(result[-1], 19.35, 2)

    def test_decimate_timeseries_2(self):
        result = samp.decimate_timeseries(np.concatenate([self.timeseries, self.timeseries]), 2)
        self.assertEqual(len(result), 14)
        self.assertAlmostEqual(result[0], 9.94, 2)
        self.assertAlmostEqual(result[-1], 29.28, 2)

    def test_decimate_collection(self):
        repeated_timeseries = np.concatenate([self.timeseries, self.timeseries])
        result = samp.decimate_timeseries_collection(np.array([repeated_timeseries]), 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 10)
        self.assertAlmostEqual(result[0][0], 9.80, 2)
        self.assertAlmostEqual(result[0][-1], 19.35, 2)
