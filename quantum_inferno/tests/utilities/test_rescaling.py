import unittest

from quantum_inferno.utilities import rescaling


class MyTestCase(unittest.TestCase):
    def test_to_log2_with_epsilon(self):
        result = rescaling.to_log2_with_epsilon(100.0)
        self.assertAlmostEqual(result, 6.64, 2)
        result = rescaling.to_log2_with_epsilon(-100.0)
        self.assertAlmostEqual(result, 6.64, 2)
        values = [100.0, -100.0]
        result = rescaling.to_log2_with_epsilon(values)
        self.assertAlmostEqual(result[0], 6.64, 2)
        self.assertAlmostEqual(result[1], 6.64, 2)

    def test_is_power_of_two(self):
        result = rescaling.is_power_of_two(8)
        self.assertTrue(result)
        result = rescaling.is_power_of_two(9)
        self.assertFalse(result)

    def test_to_decibel_with_epsilon(self):
        result = rescaling.to_decibel_with_epsilon(100.0, 1.0)
        self.assertEqual(result, 40.0)
        result = rescaling.to_decibel_with_epsilon(-100.0, 1.0)
        self.assertEqual(result, 40.0)
        result = rescaling.to_decibel_with_epsilon(100.0, 1.0, "amplitude")
        self.assertEqual(result, 40.0)
        result = rescaling.to_decibel_with_epsilon(100.0, 1.0, "power")
        self.assertEqual(result, 20.0)
        result = rescaling.to_decibel_with_epsilon(100.0, 100.0)
        self.assertEqual(result, 0)
        values = [100.0, -100.0]
        result = rescaling.to_decibel_with_epsilon(values, 1.0)
        self.assertEqual(result[0], 40.0)
        self.assertEqual(result[1], 40.0)
