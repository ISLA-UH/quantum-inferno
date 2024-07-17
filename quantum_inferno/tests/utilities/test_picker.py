import unittest

import numpy as np

from quantum_inferno.synth.benchmark_signals import well_tempered_tone
from quantum_inferno.utilities import picker


class TestPicker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.signal, cls.timestamps, _, cls.sample_rate, cls.freq_center, cls.resolution = well_tempered_tone()
        cls.two_peak_signal = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1])

    def test_find_sample_rate_hz_from_timestamps(self):
        timestamps = np.array([10, 20, 30, 40, 50])
        sample_rate = picker.find_sample_rate_hz_from_timestamps(timestamps, "ms")
        self.assertEqual(sample_rate, 100.0)

    def test_scale_signal_by_extraction_type_max(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.two_peak_signal, "sigmax")
        self.assertEqual(len(scaled_signal), len(self.two_peak_signal))
        self.assertEqual(scaled_signal[5], 1.0)
        self.assertEqual(scaled_signal[15], -1.0)

    def test_scale_signal_by_extraction_type_min(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.two_peak_signal, "sigmin")
        self.assertEqual(len(scaled_signal), len(self.two_peak_signal))
        self.assertEqual(scaled_signal[5], -1.0)
        self.assertEqual(scaled_signal[15], 1.0)

    def test_scale_signal_by_extraction_type_abs(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.two_peak_signal, "sigabs")
        self.assertEqual(len(scaled_signal), len(self.two_peak_signal))
        self.assertEqual(scaled_signal[5], 1.0)
        self.assertEqual(scaled_signal[15], -1.0)

    def test_scale_signal_by_extraction_type_log2(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.two_peak_signal, "log2")
        self.assertEqual(len(scaled_signal), len(self.two_peak_signal))
        self.assertAlmostEqual(scaled_signal[5], 2.32, 2)
        self.assertAlmostEqual(scaled_signal[15], 2.32, 2)

    def test_scale_signal_by_extraction_type_log2max(self):
        scaled_signal = picker.scale_signal_by_extraction_type(self.two_peak_signal, "log2max")
        self.assertEqual(len(scaled_signal), len(self.two_peak_signal))
        self.assertEqual(scaled_signal[5], 1.0)
        self.assertEqual(scaled_signal[15], 1.0)

    def test_apply_bandpass(self):
        result = picker.apply_bandpass(self.signal, (100, 200), self.sample_rate)
        self.assertEqual(len(result), len(self.signal))
        self.assertAlmostEqual(result[100], 0.0, 2)

    def test_apply_bandpass_with_bad_bandpass(self):
        self.assertRaises(ValueError, picker.apply_bandpass, self.signal, (300, 100), self.sample_rate)
        self.assertRaises(ValueError, picker.apply_bandpass, self.signal, (100, 100), self.sample_rate)
        self.assertRaises(ValueError, picker.apply_bandpass, self.signal, (1, self.sample_rate), self.sample_rate)
        self.assertRaises(ValueError, picker.apply_bandpass, self.signal, (0, 1), self.sample_rate)

    def test_find_peaks_by_extraction_type_with_bandpass(self):
        result = picker.find_peaks_by_extraction_type_with_bandpass(self.signal, (100, 300), self.sample_rate)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[1], 8190)

    def test_find_peaks_by_extraction_type(self):
        result = picker.find_peaks_by_extraction_type(self.two_peak_signal, "sigmax")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 5)

    def test_find_peaks_with_bits(self):
        result = picker.find_peaks_with_bits(self.signal, self.sample_rate)
        self.assertEqual(len(result), 65)
        self.assertEqual(result[0], 27)
        self.assertEqual(result[1], 128)

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
        result = picker.find_peaks_to_comb_function(self.signal, [100, 200])
        self.assertEqual(len(result), 8192)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[100], 1)
        self.assertEqual(result[200], 1)

        result = picker.find_peaks_to_comb_function(self.signal, np.array([100, 200]))
        self.assertEqual(len(result), 8192)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[100], 1)
        self.assertEqual(result[200], 1)

        result = picker.find_peaks_to_comb_function(self.signal, 200)
        self.assertEqual(len(result), 8192)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[200], 1)
