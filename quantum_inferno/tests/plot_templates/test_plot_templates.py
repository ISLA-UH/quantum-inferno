import unittest

import numpy as np

import quantum_inferno.plot_templates.plot_templates as pt


class SanitizeTimestampsTest(unittest.TestCase):
    def test_sanitize_timestamps(self):
        timestamps = np.array([10, 20, 30, 40, 50])
        new_timestamps = pt.sanitize_timestamps(timestamps)
        self.assertEqual(new_timestamps[0], 0)


class GetTimeLabelTest(unittest.TestCase):
    def test_get_time_label(self):
        start_time_epoch = 1704067200  # 2024 Jan 1, 00:00
        time_units = "s"
        time_label = pt.get_time_label(start_time_epoch, time_units, 0.)
        self.assertEqual(time_label, "Time (s) from UTC 2024-01-01 00:00:00")


class MeshTimeFrequencyEdgesTest(unittest.TestCase):
    def test_mesh_time_freq_edge(self):
        frequency = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        timestamps = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        min_freq = 0
        max_freq = 9

        t_edge, f_edge, f_min, f_max = pt.mesh_time_frequency_edges(frequency, timestamps, min_freq, max_freq)

        self.assertEqual(len(t_edge), len(timestamps) + 1)
        self.assertEqual(t_edge[0], -5.)
        self.assertEqual(len(f_edge), len(frequency) + 1)
        self.assertEqual(f_edge[0], -.5)
        self.assertEqual(f_min, -.5)
        self.assertEqual(f_max, 9.)
