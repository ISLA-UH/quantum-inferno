import unittest

import numpy as np

import quantum_inferno.synth.doppler as dplr


class TestDoppler(unittest.TestCase):
    def test_time_duration(self):
        timestamps = np.array([10, 20, 30, 40, 100])
        result = dplr.time_duration(timestamps)
        self.assertEqual(result, 90)
