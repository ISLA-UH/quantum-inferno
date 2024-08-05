import unittest
from quantum_inferno.synth.benchmark_signals import well_tempered_tone

from quantum_inferno.utilities import short_time_fft
from quantum_inferno.utilities import frequency_scales


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.start_hz = 1
        cls.end_hz = 5
        cls.tukey_alpha = 0.25
        cls.signal, cls.timestamps, cls.fft_nd, cls.sample_rate, cls.freq_center, cls.resolution = well_tempered_tone()
        cls.stft_obj = short_time_fft.get_stft_object_tukey(
            sample_rate_hz=cls.sample_rate,
            tukey_alpha=cls.tukey_alpha,
            segment_length=cls.fft_nd,
            overlap_length=cls.fft_nd // 2,
            scaling="magnitude",
        )
        cls.band_order = 3
        cls.base = 10 ** 0.3
        cls.reference_frequency = 1.0

    def test_get_linear_frequency_bins_range(self):
        result_default = frequency_scales.get_linear_frequency_bins_range(self.sample_rate, self.fft_nd)
        self.assertTrue((result_default == self.stft_obj.f).all())
        result_range = frequency_scales.get_linear_frequency_bins_range(
            self.sample_rate, self.fft_nd, self.start_hz, self.end_hz
        )
        self.assertTrue((result_range == [1.5625, 3.125, 4.6875]).all())

    def test_get_shorttime_fft_frequency_bins(self):
        result = frequency_scales.get_shorttime_fft_frequency_bins(self.sample_rate, self.fft_nd)
        self.assertTrue((result == self.stft_obj.f).all())

    def test_get_band_numbers(self):
        result = frequency_scales.get_band_numbers(
            self.sample_rate, self.band_order, None, None, self.base, self.reference_frequency
        )
        self.assertEqual(len(result), 28)
        result_ranged = frequency_scales.get_band_numbers(
            self.sample_rate, self.band_order, self.start_hz, self.end_hz, self.base, self.reference_frequency
        )
        self.assertEqual(len(result_ranged), 8)

    def test_get_log_central_frequency_bins_range(self):
        result_default = frequency_scales.get_log_central_frequency_bins_range(
            self.sample_rate, self.band_order, None, None, self.base, self.reference_frequency
        )
        self.assertEqual(len(result_default), 28)
        self.assertEqual(result_default[0], 1.0)
        self.assertAlmostEqual(result_default[-1], 501.187233627)
        result_range = frequency_scales.get_log_central_frequency_bins_range(
            self.sample_rate, self.band_order, self.start_hz, self.end_hz, self.base, self.reference_frequency
        )
        self.assertEqual(result_range[0], 1.0)
        self.assertAlmostEqual(result_range[-1], 5.011872336)

    def test_get_log_edge_frequencies(self):
        result_default = frequency_scales.get_log_edge_frequencies(
            self.sample_rate, self.band_order, None, None, self.base, self.reference_frequency
        )
        self.assertEqual(len(result_default), 29)
        self.assertAlmostEqual(result_default[0], 0.891250938)
        self.assertAlmostEqual(result_default[-1], 562.3413251903)
        result_range = frequency_scales.get_log_edge_frequencies(
            self.sample_rate, self.band_order, self.start_hz, self.end_hz, self.base, self.reference_frequency
        )
        self.assertEqual(len(result_range), 9)
        self.assertAlmostEqual(result_range[0], 0.891250938)
        self.assertAlmostEqual(result_range[-1], 5.623413252)
