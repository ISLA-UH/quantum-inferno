"""
scales_dyadic.py
Physical to cyber conversion with preferred quasi-dyadic orders.
Compute the complete set of Nth order bands from Nyquist to the averaging frequency.
Compute the standard deviation of the Gaussian envelope for the averaging frequency.
Example: base10, base2, dB, bits

"""

import numpy as np
from typing import Tuple, Union

""" Smallest number for 64-, 32-, and 16-bit floats. Deploy to avoid division by zero or log zero singularities"""
EPSILON64 = np.finfo(np.float64).eps
EPSILON32 = np.finfo(np.float32).eps
EPSILON16 = np.finfo(np.float16).eps

# Convert microseconds to seconds
MICROS_TO_S = 1E-6

# 3*pi/4
M_OVER_N = 0.75 * np.pi

"""
Standardized scales
"""


class Slice:
    """
    Constants for slice calculations, supersedes inferno/slice
    """
    # Preferred Orders
    ORD1 = 1.  # Octaves; repeats nearly every three decades (1mHz, 1Hz, 1kHz)
    ORD3 = 3.  # 1/3 octaves; reduced temporal uncertainty for sharp transients and good for decade cycles
    ORD6 = 6.  # 1/6 octaves, good compromise for time-frequency resolition
    ORD12 = 12.  # Musical tones, continuous waves, long duration sweeps
    ORD24 = 24.  # High resolution; long duration widow, 24 bands per octave
    ORD48 = 48.  # Ultra-high spectral resolution for blowing minds and interstellar communications
    # Constant Q Base
    G2 = 2.  # Base two for perfect octaves and fractional octaves
    G3 = 10. ** 0.3  # Reconciles base2 and base10
    # Time
    T_PLANCK = 5.4E-44  # 2.**(-144)   Planck time in seconds
    T0S = 1E-42  # Universal Scale in S
    T1S = 1.    # 1 second
    T100S = 100.  # 1 hectosecond, IMS low band edge
    T1000S = 1000.  # 1 kiloseconds = 1 mHz
    T1M = 60.  # 1 minute in seconds
    T1H = T1M*60.  # 1 hour in seconds
    T1D = T1H*24.   # 1 day in seconds
    TU = 2.**58  # Estimated age of the known universe in seconds
    # Frequency
    F1HZ = 1.  # 1 Hz
    F1KHZ = 1_000.  # 1 kHz
    F0HZ = 1.E42  # 1/Universal Scale
    FU = 2.**-58  # 1/Estimated age of the known universe in s

    # NOMINAL SAMPLE RATES (REDVOX API M, 2022)
    FS1HZ = 1.        # SOH
    FS10HZ = 10.      # iOS Barometer
    FS30HZ = 30.      # Android barometer
    FS80HZ = 80.      # Infrasound
    FS200HZ = 200.    # Android Magnetometer, Fast
    FS400HZ = 400.    # Android Accelerometer and Gyroscope, Fast
    FS800HZ = 800.    # Infrasound and Low Audio
    FS8KHZ = 8_000.    # Speech Audio
    FS16KHZ = 16_000.  # ML Audio
    FS48KHZ = 48_000.  # Audio to Ultrasound


# DEFAULT CONSTANTS FOR TIME_FREQUENCY CANVAS
default_scale_base = Slice.G3
default_scale_order = Slice.ORD3
default_ref_frequency_hz = Slice.F1HZ

default_scale_order_min: float = 0.75  # Garces (2022)
default_fft_pow2_points_max: int = 2**15  # Computational FFT limit, tune to computing system
default_fft_pow2_points_min: int = 2**8  # For a tolerable display
default_mesh_pow2_pixels: int = 2**19  # Total of pixels per mesh, tune to plotting engine
default_time_display_s: float = 60.  # Physical time to display; sets display truncation


def scale_order_check(scale_order: float = default_scale_order):
    """
    Ensure no negative, complex, or unreasonably small orders are passed; override to 1/3 octave band
    Standard orders are scale_order = [1, 3, 6, 12, 24]. Order must be >= 0.75 or reverts to N=3
    :param scale_order: Band order,
    """
    # I'm confident there are better admissibility tests
    scale_order = np.abs(scale_order)  # Should be real, positive float
    if scale_order < default_scale_order_min:
        print('N<0.75 specified, overriding using N = ', default_scale_order)
        scale_order = default_scale_order
    return scale_order


def scale_multiplier(scale_order: float = default_scale_order):
    """
    Scale multiplier for scale bands of order N > 0.75
    :param scale_order: scale order
    :return:
    """
    scale_order_check(scale_order)
    return M_OVER_N*scale_order


def cycles_from_order(scale_order: float) -> float:
    """
    Compute the number of cycles M for a specified band order N
    N is the quantization parameter for the constant Q wavelet filters

    :param scale_order: Band order, must be > 0.75 or reverts to N=3
    :return: cycles_M, number of cycled per normalized angular frequency
    """
    scale_order_check(scale_order)
    cycles_per_scale = scale_multiplier(scale_order)
    return cycles_per_scale


def order_from_cycles(cycles_per_scale: float) -> float:
    """
    Compute the number of cycles M for a specified band order N
    N is the quantization parameter for the constant Q wavelet filters

    :param cycles_per_scale: Should be greater than or equal than one
    :return: cycles_M, number of cycled per normalized angular frequency
    """
    if np.abs(cycles_per_scale) < 1:
        cycles_per_scale = 1.
    return cycles_per_scale/M_OVER_N


def base_multiplier(scale_order: float = default_scale_order,
                    scale_base: float = default_scale_base):
    """
    Dyadic (log2) foundation for arbitrary base
    :param scale_order:
    :param scale_base:
    :return:
    """

    scale_order_check(scale_order)
    return scale_order/np.log2(scale_base)


def scale_from_frequency_hz(scale_order: float,
                            scale_frequency_center_hz: Union[np.ndarray, float],
                            frequency_sample_rate_hz: float) -> \
        Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Nondimensional scale and angular frequency for canonical Gabor/Morlet wavelet
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: scale_atom, scaled angular frequency
    """
    scale_angular_frequency = 2. * np.pi * scale_frequency_center_hz/frequency_sample_rate_hz
    scale_atom = cycles_from_order(scale_order)/scale_angular_frequency
    return scale_atom, scale_angular_frequency


def gaussian_sigma_from_frequency(frequency_sample_hz: float,
                                  frequency_hz: np.ndarray,
                                  scale_order: float = default_scale_order):
    """
    Standard deviation (sigma) of Gaussian envelope from frequency (Garces, 2023)
    Use 3/8 = 0.375
    :param frequency_hz: physical frequency in Hz
    :param frequency_sample_hz: sensor sample rate in Hz
    :param scale_order: Scale order
    :return: standard deviation from physical frequency in Hz
    """
    return 0.375*scale_order*frequency_sample_hz/frequency_hz


def scale_bands_from_ave(frequency_sample_hz: float,
                         frequency_ave_hz: float,
                         scale_order: float = default_scale_base,
                         scale_ref_hz: float = default_ref_frequency_hz,
                         scale_base: float = default_scale_base):
    """

    :param frequency_sample_hz:
    :param frequency_ave_hz:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return: [band_nyq, band_ave, band_max, log2_ave_life_dyad]
    """

    # Framework constants
    scale_mult = scale_multiplier(scale_order)
    order_over_log2base = base_multiplier(scale_order, scale_base)

    # Dyadic transforms
    log2_scale_mult = np.log2(scale_mult)

    # Dependent on reference and averaging frequency
    log2_ave_physical = np.log2(scale_ref_hz/frequency_ave_hz)

    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz/scale_ref_hz)
    log2_ave_cyber = np.log2(frequency_sample_hz/frequency_ave_hz)

    # Closest largest power of two to averaging frequency
    log2_ave_life_dyad = np.ceil(log2_scale_mult + log2_ave_cyber)

    # Shortest period, Nyquist limit
    band_nyq = int(np.ceil(order_over_log2base*(1-log2_ref)))
    # Closest period band
    band_ave = int(np.round(order_over_log2base*log2_ave_physical))
    # Longest period band
    band_max = int(np.floor(order_over_log2base*(log2_ave_life_dyad - log2_scale_mult - log2_ref)))

    return [band_nyq, band_ave, band_max, log2_ave_life_dyad]


def scale_band_from_frequency(frequency_input_hz: float,
                              scale_order: float = default_scale_base,
                              scale_ref_hz: float = default_ref_frequency_hz,
                              scale_base: float = default_scale_base):
    """
    For any one mid frequencies; not meant to be vectorized, only for comparing expectations.
    DOES NOT DEPEND ON SAMPLE RATE
    :param frequency_input_hz:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return: [band_number, band_frequency_hz]
    """

    # TODO: Meant for single input values away from max and min bands
    # Framework constants
    order_over_log2base = base_multiplier(scale_order, scale_base)
    # Dependent on reference frequency
    log2_in_physical = np.log2(scale_ref_hz/frequency_input_hz)
    # Closest period band, TODO: check for duplicates if vectorized
    band_number = int(np.round(order_over_log2base*log2_in_physical))
    band_frequency_hz = scale_ref_hz*scale_base**(-band_number/scale_order)

    return [band_number, band_frequency_hz]


def scale_bands_from_pow2(frequency_sample_hz: float,
                          log2_ave_life_dyad: int,
                          scale_order: float = default_scale_base,
                          scale_ref_hz: float = default_ref_frequency_hz,
                          scale_base: float = default_scale_base):
    """

    :param frequency_sample_hz:
    :param log2_ave_life_dyad:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return: [band_nyq, band_max]
    """

    # Framework constants
    scale_mult = scale_multiplier(scale_order)
    order_over_log2base = base_multiplier(scale_order, scale_base)

    # Dyadic transforms
    log2_scale_mult = np.log2(scale_mult)

    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz/scale_ref_hz)

    # Shortest period, Nyquist limit
    band_nyq = int(np.ceil(order_over_log2base*(1-log2_ref)))
    # Longest period band
    band_max = int(np.floor(order_over_log2base*(log2_ave_life_dyad - log2_scale_mult - log2_ref)))

    return [band_nyq, band_max]


def period_from_bands(band_min: int,
                      band_max: int,
                      scale_order: float = default_scale_base,
                      scale_ref_hz: float = default_ref_frequency_hz,
                      scale_base: float = default_scale_base):

    bands = np.arange(band_min, band_max+1)
    # Increasing order
    period = scale_base**(bands/scale_order)/scale_ref_hz
    return period


def frequency_from_bands(band_min: int,
                         band_max: int,
                         scale_order: float = default_scale_base,
                         scale_ref_hz: float = default_ref_frequency_hz,
                         scale_base: float = default_scale_base):

    bands = np.arange(band_min, band_max+1)
    # Flip so it increases
    frequency = np.flip(scale_ref_hz*scale_base**(-bands/scale_order))
    return frequency


def log_frequency_hz_from_fft_points(
        frequency_sample_hz: float,
        fft_points: int,
        scale_order: float = default_scale_base,
        scale_ref_hz: float = default_ref_frequency_hz,
        scale_base: float = default_scale_base
) -> np.ndarray:

    # TODO: Make function to round to to nearest power of two and perform all-around error checking for pow2
    # See log 2 functions below
    log2_ave_life_dyad = int(np.ceil(np.log2(fft_points)))
    # Framework constants
    scale_mult = scale_multiplier(scale_order)
    order_over_log2base = base_multiplier(scale_order, scale_base)

    # Dyadic transforms
    log2_scale_mult = np.log2(scale_mult)

    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz/scale_ref_hz)

    # Shortest period, Nyquist limit
    band_nyq = int(np.ceil(order_over_log2base*(1-log2_ref)))
    # Shortest period, nominal 8/10 of Nyquist
    band_aa = int(np.ceil(order_over_log2base*(np.log2(2.5) - log2_ref)))
    # Longest period band
    band_max = int(np.floor(order_over_log2base*(log2_ave_life_dyad - log2_scale_mult - log2_ref)))

    # Stopped before Nyquist, before max
    # bands_old = np.arange(band_nyq+1, band_max+1)
    # Stopped at 0.8 of Nyquist
    bands = np.arange(band_aa, band_max+1)
    # Flip so frequency increases up to one band below Nyquist
    frequency_logarithmic_hz = np.flip(scale_ref_hz*scale_base**(-bands/scale_order))
    return frequency_logarithmic_hz


def lin_frequency_hz_from_fft_points(
        frequency_sample_hz: float,
        fft_points: int,
        scale_order: float = default_scale_base,
        scale_ref_hz: float = default_ref_frequency_hz,
        scale_base: float = default_scale_base
) -> np.ndarray:
    """

    :param frequency_sample_hz:
    :param fft_points:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return:
    """

    frequency_log_hz = log_frequency_hz_from_fft_points(frequency_sample_hz, fft_points,
                                                        scale_order, scale_ref_hz, scale_base)
    frequency_lin_min = frequency_log_hz[0]  # First band
    frequency_lin_max = frequency_log_hz[-1]  # Last band
    frequency_linear_hz = np.linspace(start=frequency_lin_min, stop=frequency_lin_max, num=len(frequency_log_hz))
    return frequency_linear_hz


# TODO: Turn into a test function
if __name__ == "__main__":
    # Framework specs
    scale_order0 = 6.
    scale_base0 = Slice.G3
    scale_ref0 = Slice.F1HZ

    # Sensor specific
    frequency_sample_hz0 = 100.
    print('\nNominal Sample Rate, Hz:', frequency_sample_hz0)

    fft_points_max = default_fft_pow2_points_max
    fft_points_min = default_fft_pow2_points_min
    max_mesh_pixels = default_mesh_pow2_pixels
    # Mesh time display limit
    time_display_s = default_time_display_s
    # TODO: Minimum number of points in display

    time_display_points_float: float = time_display_s*frequency_sample_hz0
    time_display_points_pow2: int = int(2**np.ceil(np.log2(time_display_points_float)))
    print('Display time, s:', time_display_s)
    print('Display points:', time_display_points_float)
    print('Display points pow2:', time_display_points_pow2)
    print('Display pow2 duration:', time_display_points_pow2/frequency_sample_hz0)

    # Works for 60s display window - display window becomes the max averaging lifetime
    # We have a computational averaging lifetime (~2^16) and a max averaging lifetime (display window)
    if time_display_points_pow2 > fft_points_max:
        fft_points0 = fft_points_max
    else:
        fft_points0 = time_display_points_pow2

    # Do not drop below stft_fft_points min
    if fft_points0 < fft_points_min:
        # 256 points is minimal
        fft_points0 = fft_points_min

    print('MAX FFT: Order, Base, Reference, log2(FFT points)')
    print(scale_order0, scale_base0, scale_ref0, np.log2(fft_points0))

    physical_frequency_hz = \
        log_frequency_hz_from_fft_points(frequency_sample_hz0, fft_points0,
                                         scale_order0, scale_ref0, scale_base0)
    number_of_frequencies = len(physical_frequency_hz)
    reduction_factor_float = number_of_frequencies*fft_points0/max_mesh_pixels

    if reduction_factor_float > 1:
        reduction_factor_pow2 = 2**np.ceil(np.log2(reduction_factor_float))
    else:
        reduction_factor_pow2 = 1.

    print('Physical frequency, Min, Max:', physical_frequency_hz[0], physical_frequency_hz[-1])
    print('Averaging window duration:', fft_points0/frequency_sample_hz0)
    print('Number of bands:', number_of_frequencies)
    print('Mesh reduction factor:', reduction_factor_pow2)




