import unittest

import numpy as np

from quantum_inferno.synth.benchmark_signals import well_tempered_tone
from quantum_inferno.utilities import picker


class TestPicker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.signal, cls.timestamps, cls.time, cls.sample_rate, cls.freq_center, cls.resolution = \
            well_tempered_tone()

    def test_find_sample_rate_hz_from_timestamps(self):
        timestamps = np.array([10, 20, 30, 40, 50])
        sample_rate = picker.find_sample_rate_hz_from_timestamps(timestamps, 'ms')
        self.assertEqual(sample_rate, 100.)

    def test_scale_signal_by_extraction_type_max(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.signal, picker.ExtractionType.SIGMAX)
        self.assertEqual(len(scaled_signal), 8192)
        self.assertEqual(scaled_signal[0], 1.)
        self.assertAlmostEqual(scaled_signal[-1], 0.89, 2)

    def test_scale_signal_by_extraction_type_min(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.signal, picker.ExtractionType.SIGMIN)
        self.assertEqual(len(scaled_signal), 8192)
        self.assertEqual(scaled_signal[0], 1.)
        self.assertAlmostEqual(scaled_signal[-1], 0.89, 2)

    def test_scale_signal_by_extraction_type_abs(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.signal, picker.ExtractionType.SIGABS)
        self.assertEqual(len(scaled_signal), 8192)
        self.assertEqual(scaled_signal[0], 1.)
        self.assertAlmostEqual(scaled_signal[-1], 0.89, 2)

    def test_scale_signal_by_extraction_type_bit(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.signal, picker.ExtractionType.SIGBIT)
        self.assertEqual(len(scaled_signal), 8192)
        self.assertAlmostEqual(scaled_signal[0], 3.2e-16, 2)
        self.assertAlmostEqual(scaled_signal[-1], -0.16, 2)

    def test_apply_bandpass(self):
        result = picker.apply_bandpass(self.signal, (100, 300), self.sample_rate)
        self.assertEqual(len(result), 8192)
        self.assertAlmostEqual(result[0], -3.73e-5, 2)
        self.assertAlmostEqual(result[-1], 1.41e-5, 2)

    def test_find_peaks_by_extraction_type_with_bandpass(self):
        result = picker.find_peaks_by_extraction_type_with_bandpass(self.signal, (100, 300), self.sample_rate)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 8190)

    def test_find_peaks_by_extraction_type(self):
        result = picker.find_peaks_by_extraction_type(self.signal, picker.ExtractionType.SIGMAX)
        self.assertEqual(len(result), 607)
        self.assertEqual(result[0], 13)
        self.assertEqual(result[1], 27)

    def test_find_peaks_with_bits(self):
        result = picker.find_peaks_with_bits(self.signal, self.sample_rate)
        self.assertEqual(len(result), 95)
        self.assertEqual(result[0], 27)
        self.assertEqual(result[1], 175)

    def test_extract_signal_index_with_buffer(self):
        result = picker.extract_signal_index_with_buffer(self.sample_rate, 4000, 2, 2)
        self.assertEqual(result[0], 2400)
        self.assertEqual(result[-1], 5600)

    def test_extract_signal_with_buffer_seconds(self):
        result = picker.extract_signal_with_buffer_seconds(self.signal, self.sample_rate, 4000, 2, 2)
        self.assertEqual(len(result), 3200)
        self.assertAlmostEqual(result[0], 0.71, 2)
        self.assertAlmostEqual(result[-1], -0.95, 2)

    def test_find_peaks_to_comb_function(self):
        result = picker.find_peaks_to_comb_function(self.signal, np.ndarray([100, 200]))
        self.assertEqual(len(result), 8192)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[100], 1)
