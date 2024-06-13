# import unittest
# import numpy as np
# import quantum_inferno.scales_dyadic as sd
#
#
# class MyTestCase(unittest.TestCase):
#     def test_scales_dyadic(self):
#         scale_order0 = 6.
#         frequency_sample_hz0 = 100.
#         time_display_points_float: float = sd.DEFAULT_TIME_DISPLAY_S*frequency_sample_hz0
#         time_display_points_pow2: int = int(2**np.ceil(np.log2(time_display_points_float)))
#         physical_frequency_hz = \
#             sd.log_frequency_hz_from_fft_points(frequency_sample_hz0, time_display_points_pow2,
#                                                 scale_order0, sd.Slice.F1HZ, sd.Slice.G3)
#
#         self.assertEqual(6000., time_display_points_float)
#         self.assertEqual(8192, time_display_points_pow2)
#         self.assertEqual(13.0, np.log2(time_display_points_pow2))
#         self.assertEqual(0.1778279410038923, physical_frequency_hz[0])
#         self.assertEqual(39.810717055349706, physical_frequency_hz[-1])
#         self.assertEqual(48, len(physical_frequency_hz))
