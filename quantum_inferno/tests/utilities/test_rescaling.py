import unittest

from quantum_inferno.utilities import rescaling


class MyTestCase(unittest.TestCase):
    def test_to_log2_with_epsilon(self):
        result = rescaling.to_log2_with_epsilon(100.)
        self.assertAlmostEqual(result, 6.64, 2)

    def test_is_power_of_two(self):
        result = rescaling.is_power_of_two(8)
        self.assertTrue(result)
        result = rescaling.is_power_of_two(9)
        self.assertFalse(result)

    def test_to_decibel_with_epsilon(self):
        result = rescaling.to_decibel_with_epsilon(100., 1.0, rescaling.DataScaleType.AMP)
        self.assertEqual(result, 40.)
        result = rescaling.to_decibel_with_epsilon(100., 1.0, rescaling.DataScaleType.POW)
        self.assertEqual(result, 20.)
