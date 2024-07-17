import unittest

import numpy as np

from quantum_inferno.synth.benchmark_signals import well_tempered_tone
from quantum_inferno.utilities import short_time_fft
from scipy.signal import ShortTimeFFT


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
