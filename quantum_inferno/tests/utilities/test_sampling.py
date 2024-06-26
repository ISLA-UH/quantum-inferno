import unittest

import numpy as np

import quantum_inferno.utilities.sampling as samp


class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.timeseries = np.array([10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40])

    def test_subsample(self):
        result, new_rate = samp.subsample(self.timeseries, 1., 3, samp.SubsampleMethod.NTH)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 10)
        self.assertEqual(result[-1], 60)
        self.assertAlmostEqual(new_rate, .33, 2)

    def test_subsample_avg(self):
        result, new_rate = samp.subsample(self.timeseries, 1., 3, samp.SubsampleMethod.AVG)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 20)
        self.assertEqual(result[-1], 50)
        self.assertAlmostEqual(new_rate, .33, 2)

    def test_subsample_med(self):
        result, new_rate = samp.subsample(self.timeseries, 1., 3, samp.SubsampleMethod.MED)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 20)
        self.assertEqual(result[-1], 50)
        self.assertAlmostEqual(new_rate, .33, 2)

    def test_subsample_max(self):
        result, new_rate = samp.subsample(self.timeseries, 1., 3, samp.SubsampleMethod.MAX)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 30)
        self.assertEqual(result[-1], 60)
        self.assertAlmostEqual(new_rate, .33, 2)

    def test_subsample_min(self):
        result, new_rate = samp.subsample(self.timeseries, 1., 3, samp.SubsampleMethod.MIN)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 10)
        self.assertEqual(result[-1], 40)
        self.assertAlmostEqual(new_rate, .33, 2)

    def test_resample_uneven_timestamps(self):
        timestamps = np.array([10, 15, 22, 31, 41, 46, 53, 60, 75, 79, 89, 100])
        result, new_rate = samp.resample_uneven_timeseries(self.timeseries, timestamps, .2)
        self.assertEqual(len(result), 18)
        self.assertEqual(result[0], 10)
        self.assertAlmostEqual(result[-1], 44.55, 2)
        self.assertAlmostEqual(result[4], 38.89, 2)
        self.assertEqual(new_rate, .2)

    def test_resample_uneven_timestamps_assumed(self):
        timestamps = np.array([10, 15, 22, 31, 41, 46, 53, 60, 75, 79, 89, 100])
        result, new_rate = samp.resample_uneven_timeseries(self.timeseries, timestamps)
        self.assertEqual(len(result), 11)
        self.assertEqual(result[0], 10)
        self.assertAlmostEqual(result[-1], 47.44, 2)
        self.assertAlmostEqual(result[4], 53.45, 2)
        self.assertAlmostEqual(new_rate, .12, 2)

    def test_resample_with_sample_rate(self):
        result, new_rate = samp.resample_with_sample_rate(self.timeseries, 1., .5)
        self.assertEqual(len(result), 6)
        self.assertAlmostEqual(result[0], 16.79, 2)
        self.assertAlmostEqual(result[-1], 51.67, 2)
        self.assertAlmostEqual(new_rate, .5, 2)

    def test_subsample_2d(self):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        result = samp.subsample_2d(array, 2, samp.SubsampleMethod.NTH)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[-1][-1], 11)

    def test_subsample_2d_avg(self):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        result = samp.subsample_2d(array, 3, samp.SubsampleMethod.AVG)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[0][0], 1.5)
        self.assertEqual(result[-1][-1], 10.5)

    def test_decimate_timeseries(self):
        result = samp.decimate_timeseries(np.array([10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40,
                                                    10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40,
                                                    10, 20, 30, 40]), 3)
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result[0], 10.07, 2)
        self.assertAlmostEqual(result[-1], 38.82, 2)

    def test_decimate_timeseries_2(self):
        result = samp.decimate_timeseries(np.array([10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40,
                                                    10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40,
                                                    10, 20, 30, 40]), 2)
        self.assertEqual(len(result), 14)
        self.assertAlmostEqual(result[0], 9.99, 2)
        self.assertAlmostEqual(result[-1], 25.74, 2)

    def test_decimate_collection(self):
        result = samp.decimate_timeseries_collection(np.array([[10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40,
                                                               10, 20, 30, 40, 50, 60, 70, 80, 70, 60, 50, 40,
                                                               10, 20, 30, 40]]), 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 10)
        self.assertAlmostEqual(result[0][0], 10.07, 2)
        self.assertAlmostEqual(result[0][-1], 38.82, 2)
