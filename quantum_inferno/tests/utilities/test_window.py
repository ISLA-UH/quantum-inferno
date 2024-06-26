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
        self.assertAlmostEqual(result[0], 0, 2)
        self.assertAlmostEqual(result[-1], 0, 2)

    def test_get_tukey_alpha_0(self):
        result = window.get_tukey(self.ary, 0)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 1, 2)
        self.assertAlmostEqual(result[-1], 1, 2)

    def test_get_tukey_alpha_1(self):
        result = window.get_tukey(self.ary, 1)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 0, 2)
        self.assertAlmostEqual(result[-1], 0, 2)

    def test_get_tukey_alpha_2(self):
        result = window.get_tukey(self.ary, 2)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 0, 2)
        self.assertAlmostEqual(result[-1], 0, 2)

    def test_get_tukey_by_buffer_num(self):
        result = window.get_tukey_by_buffer_num(self.ary, 2)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 0, 2)
        self.assertAlmostEqual(result[-1], 0, 2)

    def test_get_tukey_by_buffer_num_alpha_0(self):
        result = window.get_tukey_by_buffer_num(self.ary, 2, 0)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 1, 2)
        self.assertAlmostEqual(result[-1], 1, 2)

    def test_get_tukey_by_buffer_num_alpha_1(self):
        result = window.get_tukey_by_buffer_num(self.ary, 2, 1)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 0, 2)
        self.assertAlmostEqual(result[-1], 0, 2)

    def test_get_tukey_by_buffer_s(self):
        result = window.get_tukey_by_buffer_s(self.ary, 4, 1)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 0, 2)
        self.assertAlmostEqual(result[-1], 0, 2)

    def test_get_tukey_by_buffer_s_alpha_0(self):
        result = window.get_tukey_by_buffer_s(self.ary, 4, 1, 0)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 1, 2)
        self.assertAlmostEqual(result[-1], 1, 2)

    def test_get_tukey_by_buffer_s_alpha_1(self):
        result = window.get_tukey_by_buffer_s(self.ary, 4, 1, 1)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result[0], 0, 2)
        self.assertAlmostEqual(result[-1], 0, 2)
