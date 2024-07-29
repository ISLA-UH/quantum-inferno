import unittest

import numpy as np
from scipy.signal import ShortTimeFFT

from quantum_inferno.synth.benchmark_signals import well_tempered_tone
from quantum_inferno.utilities import short_time_fft


class TestPicker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tukey_alpha = 0.25
        cls.signal, cls.timestamps, cls.fft_nd, cls.sample_rate, cls.freq_center, cls.resolution = well_tempered_tone()
        cls.two_peak_signal = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -4, -3, -2, -1])

    def test_get_stft_object_tukey(self):
        stft_obj = short_time_fft.get_stft_object_tukey(
            sample_rate_hz=self.sample_rate,
            tukey_alpha=self.tukey_alpha,
            segment_length=self.fft_nd,
            overlap_length=self.fft_nd // 2,
            scaling="magnitude",
        )
        self.assertIsInstance(stft_obj, ShortTimeFFT)
        self.assertTrue(stft_obj.invertible)
        self.assertEqual(stft_obj.scaling, "magnitude")
        self.assertEqual(stft_obj.fft_mode, "onesided")
        self.assertEqual(stft_obj.m_num, self.fft_nd)  # check window length
        self.assertEqual(stft_obj.mfft, self.fft_nd)  # check window length with mfft
        self.assertEqual(stft_obj.hop, self.fft_nd // 2)  # check hop length
        self.assertEqual(stft_obj.delta_f, self.resolution)

    def test_stft_tukey(self):
        frequencies, times, stfts = short_time_fft.stft_tukey(
            timeseries=self.signal,
            sample_rate_hz=self.sample_rate,
            tukey_alpha=self.tukey_alpha,
            segment_length=self.fft_nd,
            overlap_length=self.fft_nd // 2,
            scaling="magnitude",
        )
        self.assertEqual(times.shape[0], len(self.timestamps) // (self.fft_nd // 2) + 1)
        self.assertEqual(frequencies.shape, (self.fft_nd // 2 + 1,))
        self.assertEqual(stfts.shape, (self.fft_nd // 2 + 1, len(self.timestamps) // (self.fft_nd // 2) + 1))

    def test_istft_tukey(self):
        frequencies, times, stfts = short_time_fft.stft_tukey(
            timeseries=self.signal,
            sample_rate_hz=self.sample_rate,
            tukey_alpha=self.tukey_alpha,
            segment_length=self.fft_nd,
            overlap_length=self.fft_nd // 2,
            scaling="magnitude",
        )
        reconstructed_time, reconstructed_signal = short_time_fft.istft_tukey(
            stft_to_invert=stfts,
            sample_rate_hz=self.sample_rate,
            tukey_alpha=self.tukey_alpha,
            segment_length=self.fft_nd,
            overlap_length=self.fft_nd // 2,
            scaling="magnitude",
        )
        self.assertEqual(len(reconstructed_signal), len(self.signal))
        self.assertTrue(np.allclose(self.signal, reconstructed_signal, atol=1e-14))
        self.assertTrue(np.allclose(self.timestamps, reconstructed_time, atol=1e-14))

    def test_spectrogram_tukey(self):
        frequencies, times, spectrogram = short_time_fft.spectrogram_tukey(
            timeseries=self.signal,
            sample_rate_hz=self.sample_rate,
            tukey_alpha=self.tukey_alpha,
            segment_length=self.fft_nd,
            overlap_length=self.fft_nd // 2,
            scaling="magnitude",
        )
        self.assertEqual(times.shape[0], len(self.timestamps) // (self.fft_nd // 2) + 1)
        self.assertEqual(frequencies.shape, (self.fft_nd // 2 + 1,))
        self.assertEqual(spectrogram.shape, (self.fft_nd // 2 + 1, len(self.timestamps) // (self.fft_nd // 2) + 1))
