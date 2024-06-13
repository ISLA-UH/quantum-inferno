import unittest

import numpy as np

import quantum_inferno.plot_templates.plot_templates as pt


class SanitizeTimestampsTest(unittest.TestCase):
    def test_sanitize_timestamps(self):
        timestamps = np.array([10, 20, 30, 40, 50])
        new_timestamps = pt.sanitize_timestamps(timestamps)
        self.assertEqual(new_timestamps[0], 0)
