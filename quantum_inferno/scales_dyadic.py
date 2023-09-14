"""
scales_dyadic.py
Physical to cyber conversion with preferred quasi-dyadic orders.
Compute the complete set of Nth order bands from Nyquist to the averaging frequency.
Compute the standard deviation of the Gaussian envelope for the averaging frequency.
Example: base10, base2, dB, bits

"""

import numpy as np
from typing import Tuple, Union

""" 
Smallest number > 0 for 64-, 32-, and 16-bit floats.  
Use to avoid division by zero or log zero singularities
"""
EPSILON64 = np.finfo(np.float64).eps
EPSILON32 = np.finfo(np.float32).eps
EPSILON16 = np.finfo(np.float16).eps

# Scale multiplier for scale bands of order N > 0.75
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
DEFAULT_SCALE_BASE = Slice.G3
DEFAULT_SCALE_ORDER = Slice.ORD3
DEFAULT_REF_FREQUENCY_HZ = Slice.F1HZ

DEFAULT_SCALE_ORDER_MIN: float = 0.75  # Garces (2022)
DEFAULT_FFT_POW2_POINTS_MAX: int = 2 ** 15  # Computational FFT limit, tune to computing system
DEFAULT_FFT_POW2_POINTS_MIN: int = 2 ** 8  # For a tolerable display
DEFAULT_MESH_POW2_PIXELS: int = 2**19  # Total of pixels per mesh, tune to plotting engine
DEFAULT_TIME_DISPLAY_S: float = 60.  # Physical time to display; sets display truncation


def scale_order_check(scale_order: float = DEFAULT_SCALE_ORDER):
    """
    Ensure no negative, complex, or unreasonably small orders are passed; override to 1/3 octave band
    Standard orders are scale_order = [1, 3, 6, 12, 24]. Order must be >= 0.75 or reverts to N=3

    :param scale_order: Band order,
    """
    # TODO: Refine
    # I'm confident there are better admissibility tests
    scale_order = np.abs(scale_order)  # Should be real, positive float
    if scale_order < DEFAULT_SCALE_ORDER_MIN:
        print(f'** Warning from scales_dyadic.scale_order_check:\n'
              f'N < {DEFAULT_SCALE_ORDER_MIN} specified, overriding using N = {DEFAULT_SCALE_ORDER_MIN}')
        scale_order = DEFAULT_SCALE_ORDER_MIN
    return scale_order


def scale_multiplier(scale_order: float = DEFAULT_SCALE_ORDER):
    """
    Scale multiplier for scale bands of order N > 0.75
    :param scale_order: scale order
    :return:
    """
    return M_OVER_N * scale_order_check(scale_order)


def cycles_from_order(scale_order: float) -> float:
    """
    Compute the number of cycles M for a specified band order N.
    N is the quantization parameter for the constant Q wavelet filters

    :param scale_order: Band order, must be > 0.75 or reverts to N=0.75
    :return: cycles_M, number of cycled per normalized angular frequency
    """
    return scale_multiplier(scale_order)


def order_from_cycles(cycles_per_scale: float) -> float:
    """
    Compute the number of cycles M for a specified band order N
    where N is the quantization parameter for the constant Q wavelet filters

    :param cycles_per_scale: Should be greater than or equal than one
    :return: cycles_M, number of cycled per normalized angular frequency
    """
    # A single cycle is the min req
    if np.abs(cycles_per_scale) < 1:
        cycles_per_scale = 1.
    return scale_order_check(cycles_per_scale / M_OVER_N)


def base_multiplier(scale_order: float = DEFAULT_SCALE_ORDER,
                    scale_base: float = DEFAULT_SCALE_BASE):
    """
    Dyadic (log2) foundation for arbitrary base

    :param scale_order:
    :param scale_base:
    :return:
    """
    return scale_order_check(scale_order) / np.log2(scale_base)


def scale_from_frequency_hz(scale_order: float,
                            scale_frequency_center_hz: Union[np.ndarray, float],
                            frequency_sample_rate_hz: float) -> \
        Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Non-dimensional scale and angular frequency for canonical Gabor/Morlet wavelet

    :param scale_order:
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: scale_atom, scaled angular frequency
    """
    scale_angular_frequency = 2. * np.pi * scale_frequency_center_hz / frequency_sample_rate_hz
    scale_atom = cycles_from_order(scale_order) / scale_angular_frequency
    return scale_atom, scale_angular_frequency


def gaussian_sigma_from_frequency(frequency_sample_hz: float,
                                  frequency_hz: np.ndarray,
                                  scale_order: float = DEFAULT_SCALE_ORDER):
    """
    Standard deviation (sigma) of Gaussian envelope from frequency (Garces, 2023)
    Use 3/8 = 0.375

    :param frequency_hz: physical frequency in Hz
    :param frequency_sample_hz: sensor sample rate in Hz
    :param scale_order: Scale order
    :return: standard deviation from physical frequency in Hz
    """
    return 0.375 * scale_order * frequency_sample_hz / frequency_hz


def scale_bands_from_ave(frequency_sample_hz: float,
                         frequency_ave_hz: float,
                         scale_order: float = DEFAULT_SCALE_BASE,
                         scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
                         scale_base: float = DEFAULT_SCALE_BASE):
    """

    :param frequency_sample_hz:
    :param frequency_ave_hz:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return: Shortest period (Nyquist limit), Closest period band, Longest period band,
             Closest largest power of two to averaging frequency
    """
    # Framework constants
    order_over_log2base = base_multiplier(scale_order, scale_base)
    # Dyadic transforms
    log2_scale_mult = np.log2(scale_multiplier(scale_order))
    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz / scale_ref_hz)
    # Closest largest power of two to averaging frequency
    log2_ave_life_dyad = np.ceil(log2_scale_mult + np.log2(frequency_sample_hz / frequency_ave_hz))
    return [int(np.ceil(order_over_log2base*(1 - log2_ref))),
            int(np.round(order_over_log2base * np.log2(scale_ref_hz / frequency_ave_hz))),
            int(np.floor(order_over_log2base*(log2_ave_life_dyad - log2_scale_mult - log2_ref))),
            log2_ave_life_dyad]


def scale_band_from_frequency(frequency_input_hz: float,
                              scale_order: float = DEFAULT_SCALE_BASE,
                              scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
                              scale_base: float = DEFAULT_SCALE_BASE):
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
    # Dependent on reference frequency
    log2_in_physical = np.log2(scale_ref_hz/frequency_input_hz)
    # Closest period band, TODO: check for duplicates if vectorized
    band_number = int(np.round(base_multiplier(scale_order, scale_base) * log2_in_physical))
    band_frequency_hz = scale_ref_hz*scale_base**(-band_number/scale_order)

    return [band_number, band_frequency_hz]


def scale_bands_from_pow2(frequency_sample_hz: float,
                          log2_ave_life_dyad: int,
                          scale_order: float = DEFAULT_SCALE_BASE,
                          scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
                          scale_base: float = DEFAULT_SCALE_BASE):
    """
    :param frequency_sample_hz:
    :param log2_ave_life_dyad:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return: Shortest period (Nyquist limit) and Longest period band
    """
    # Framework constants
    order_over_log2base = base_multiplier(scale_order, scale_base)
    # Dyadic transforms
    log2_scale_mult = np.log2(scale_multiplier(scale_order))
    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz/scale_ref_hz)
    return [int(np.ceil(order_over_log2base * (1 - log2_ref))),
            int(np.floor(order_over_log2base * (log2_ave_life_dyad - log2_scale_mult - log2_ref)))]


def period_from_bands(band_min: int,
                      band_max: int,
                      scale_order: float = DEFAULT_SCALE_BASE,
                      scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
                      scale_base: float = DEFAULT_SCALE_BASE):
    """

    :param band_min:
    :param band_max:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return:
    """
    # Increasing order
    return scale_base**(np.arange(band_min, band_max + 1) / scale_order) / scale_ref_hz


def frequency_from_bands(band_min: int,
                         band_max: int,
                         scale_order: float = DEFAULT_SCALE_BASE,
                         scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
                         scale_base: float = DEFAULT_SCALE_BASE):
    """

    :param band_min:
    :param band_max:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return:
    """
    # Flip the result so it increases
    return np.flip(scale_ref_hz*scale_base**(-np.arange(band_min, band_max + 1) / scale_order))


def log_frequency_hz_from_fft_points(
        frequency_sample_hz: float,
        fft_points: int,
        scale_order: float = DEFAULT_SCALE_BASE,
        scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
        scale_base: float = DEFAULT_SCALE_BASE
) -> np.ndarray:
    """

    :param frequency_sample_hz:
    :param fft_points:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return:
    """
    # TODO: Make function to round to to nearest power of two and perform all-around error checking for pow2
    # See log 2 functions below
    log2_ave_life_dyad = int(np.ceil(np.log2(fft_points)))
    # Framework constants
    scale_mult = scale_multiplier(scale_order)
    order_over_log2base = base_multiplier(scale_order, scale_base)

    # Dyadic transforms
    log2_scale_mult = np.log2(scale_mult)

    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz / scale_ref_hz)

    # Shortest period, Nyquist limit
    # band_nyq = int(np.ceil(order_over_log2base*(1-log2_ref)))
    # Shortest period, nominal 8/10 of Nyquist
    band_aa = int(np.ceil(order_over_log2base * (np.log2(2.5) - log2_ref)))
    # Longest period band
    band_max = int(np.floor(order_over_log2base * (log2_ave_life_dyad - log2_scale_mult - log2_ref)))

    # Stopped before Nyquist, before max
    # bands_old = np.arange(band_nyq+1, band_max+1)
    # Stopped at 0.8 of Nyquist
    bands = np.arange(band_aa, band_max + 1)
    # Flip so frequency increases up to one band below Nyquist
    frequency_logarithmic_hz = np.flip(scale_ref_hz * scale_base**(-bands / scale_order))
    return frequency_logarithmic_hz


def lin_frequency_hz_from_fft_points(
        frequency_sample_hz: float,
        fft_points: int,
        scale_order: float = DEFAULT_SCALE_BASE,
        scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
        scale_base: float = DEFAULT_SCALE_BASE
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
    frequency_linear_hz = np.linspace(start=frequency_log_hz[0], stop=frequency_log_hz[-1],
                                      num=len(frequency_log_hz))
    return frequency_linear_hz
