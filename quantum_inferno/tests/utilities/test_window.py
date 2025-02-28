import unittest

import numpy as np

import quantum_inferno.utilities.window as window


class TestWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ary = np.array([5, 20, 30, 40, 45, 50, 60, 70, 85])

    def test_get_tukey(self):
        result = window.get_tukey(self.ary)
        self.assertEqual(len(result), 9)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(result[0], 0, 2)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(result[-1], 0, 2)

    def test_taper_power_correction(self):
        result = window.taper_power_correction(self.ary)
        self.assertAlmostEqual(result, 23175.)
