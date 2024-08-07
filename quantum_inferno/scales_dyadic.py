"""
This module contains constants and functions that help with
physical to cyber conversion with preferred quasi-dyadic orders.
"""

import sys
from typing import List, Tuple, Union

import numpy as np

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


def get_epsilon() -> float:
    """
    Return epsilon for float64, float32, float16 by detecting current interpreter's max size
    """
    if sys.maxsize > 2 ** 32:
        return EPSILON64
    elif sys.maxsize > 2 ** 16:
        return EPSILON32
    else:
        return EPSILON16


class Slice:
    """
    Constants for slice calculations, supersedes inferno/slice
    """
    # Preferred Orders
    ORD1 = 1.0  # Octaves; repeats nearly every three decades (1mHz, 1Hz, 1kHz)
    ORD3 = 3.0  # 1/3 octaves; reduced temporal uncertainty for sharp transients and good for decade cycles
    ORD6 = 6.0  # 1/6 octaves, good compromise for time-frequency resolution
    ORD12 = 12.0  # Musical tones, continuous waves, long duration sweeps
    ORD24 = 24.0  # High resolution; long duration window, 24 bands per octave
    ORD48 = 48.0  # Ultra-high spectral resolution for blowing minds and interstellar communications
    # Constant Q Base
    G2 = 2.0  # Base two for perfect octaves and fractional octaves
    G3 = 10.0 ** 0.3  # Reconciles base2 and base10
    # Time
    T_PLANCK = 5.4e-44  # 2.**(-144)   Planck time in seconds
    T0S = 1e-42  # Universal Scale in S
    T1S = 1.0  # 1 second
    T100S = 100.0  # 1 hectosecond, IMS low band edge
    T1000S = 1000.0  # 1 kiloseconds = 1 mHz
    T1M = 60.0  # 1 minute in seconds
    T1H = T1M * 60.0  # 1 hour in seconds
    T1D = T1H * 24.0  # 1 day in seconds
    TU = 2.0 ** 58  # Estimated age of the known universe in seconds
    # Frequency
    F1HZ = 1.0  # 1 Hz
    F1KHZ = 1_000.0  # 1 kHz
    F0HZ = 1.0e42  # 1/Universal Scale
    FU = 2.0 ** -58  # 1/Estimated age of the known universe in s

    # NOMINAL SAMPLE RATES (REDVOX API M, 2022)
    FS1HZ = 1.0  # SOH
    FS10HZ = 10.0  # iOS Barometer
    FS30HZ = 30.0  # Android barometer
    FS80HZ = 80.0  # Infrasound
    FS200HZ = 200.0  # Android Magnetometer, Fast
    FS400HZ = 400.0  # Android Accelerometer and Gyroscope, Fast
    FS800HZ = 800.0  # Infrasound and Low Audio
    FS8KHZ = 8_000.0  # Speech Audio
    FS16KHZ = 16_000.0  # ML Audio
    FS48KHZ = 48_000.0  # Audio to Ultrasound


# DEFAULT CONSTANTS FOR TIME_FREQUENCY CANVAS
DEFAULT_SCALE_BASE = Slice.G3
DEFAULT_SCALE_ORDER = Slice.ORD3
DEFAULT_REF_FREQUENCY_HZ = Slice.F1HZ

# compute maximum size for FFT calculations
__SIZE_MAX = get_epsilon()
if __SIZE_MAX == EPSILON64:
    __MAX_FFT_LIMIT_BY_SYSTEM = 63
elif __SIZE_MAX == EPSILON32:
    __MAX_FFT_LIMIT_BY_SYSTEM = 31
else:
    __MAX_FFT_LIMIT_BY_SYSTEM = 15

DEFAULT_SCALE_ORDER_MIN: float = 0.75  # Garces (2022)
DEFAULT_FFT_POW2_POINTS_MAX: int = 2 ** __MAX_FFT_LIMIT_BY_SYSTEM  # Computational FFT limit, tuned to computing system
DEFAULT_FFT_POW2_POINTS_MIN: int = 2 ** 8  # For a tolerable display
DEFAULT_MESH_POW2_PIXELS: int = 2 ** 19  # Total of pixels per mesh, tune to plotting engine
DEFAULT_TIME_DISPLAY_S: float = 60.0  # Physical time to display; sets display truncation
VALID_SCALE_ORDERS: List[float] = [0.75, 1, 1.5, 3, 6, 12, 24, 48]  # list of valid scale orders


def scale_order_check(scale_order: float = DEFAULT_SCALE_ORDER, show_warning: bool = True) -> float:
    """
    Ensure no negative, complex, or unreasonably small orders are passed; override to 1/3 octave band
    Standard orders are one of: 1, 3, 6, 12, 24. If order < 0.75 it reverts to order = 3

    :param scale_order: Band order, preferably one of: 1, 3, 6, 12, 24.  Must be > 0.75 or reverts to N=0.75
    :param show_warning: if True, prints warning of invalid scale_order.  Default True
    :return: sanitized scale order
    """
    scale_order = np.abs(scale_order)  # Force to be a real, positive float
    if scale_order < DEFAULT_SCALE_ORDER_MIN:
        if show_warning:
            print(
                f"** Warning from scales_dyadic.scale_order_check:\n"
                f"N < {DEFAULT_SCALE_ORDER_MIN} specified, overriding using N = {DEFAULT_SCALE_ORDER_MIN}"
            )
        scale_order = DEFAULT_SCALE_ORDER_MIN
    return scale_order


def scale_multiplier(scale_order: float = DEFAULT_SCALE_ORDER) -> float:
    """
    :param scale_order: Band order, preferably one of: 1, 3, 6, 12, 24.  Must be > 0.75 or reverts to N=0.75
    :return: Scale multiplier for scale bands of order N > 0.75
    """
    return M_OVER_N * scale_order_check(scale_order)


def cycles_from_order(scale_order: float) -> float:
    """
    Compute the number of cycles M for a specified band order N.
    N is the quantization parameter for the constant Q wavelet filters

    :param scale_order: Band order, preferably one of: 1, 3, 6, 12, 24.  Must be > 0.75 or reverts to N=0.75
    :return: number of cycled per normalized angular frequency
    """
    return scale_multiplier(scale_order)


def order_from_cycles(cycles_per_scale: float) -> float:
    """
    Compute the number of cycles M for a specified band order N
    where N is the quantization parameter for the constant Q wavelet filters

    :param cycles_per_scale: Should be greater than or equal than one
    :return: number of cycled per normalized angular frequency
    """
    # A single cycle is the min req
    if np.abs(cycles_per_scale) < 1:
        cycles_per_scale = 1.0
    return scale_order_check(cycles_per_scale / M_OVER_N)


def base_multiplier(scale_order: float = DEFAULT_SCALE_ORDER, scale_base: float = DEFAULT_SCALE_BASE) -> float:
    """
    :param scale_order: Band order, preferably one of: 1, 3, 6, 12, 24.  Must be > 0.75 or reverts to N=0.75
    :param scale_base: scale base
    :return: Dyadic (log2) foundation for arbitrary base
    """
    return scale_order_check(scale_order) / np.log2(scale_base)


def scale_from_frequency_hz(
    scale_order: float, scale_frequency_center_hz: Union[np.ndarray, float], frequency_sample_rate_hz: float
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Non-dimensional scale and angular frequency for canonical Gabor/Morlet wavelet

    :param scale_order: Band order, preferably one of: 1, 3, 6, 12, 24.  Must be > 0.75 or reverts to N=0.75
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: scale_atom, scaled angular frequency
    """
    scale_angular_frequency = 2.0 * np.pi * scale_frequency_center_hz / frequency_sample_rate_hz
    scale_atom = cycles_from_order(scale_order) / scale_angular_frequency
    return scale_atom, scale_angular_frequency


def band_frequency_low_high(
    frequency_order_input: float,
    frequency_base_input: float,
    frequency_ref_input: float,
    frequency_low_input: float,
    frequency_high_input: float,
    frequency_sample_rate_input: float,
) -> Tuple[float, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Standard Logarithmic Interval Time Parameters: ALWAYS USE HZ

    :param frequency_order_input: Nth order
    :param frequency_base_input: G2 or G3
    :param frequency_ref_input: reference frequency
    :param frequency_low_input: the lowest frequency of interest
    :param frequency_high_input: highest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :return: scale_order, scale_base, scale_band_number, reference frequency value, Algebraic center of frequencies,
             Geometric center of frequencies, frequency_start, frequency_end
    """
    scale_ref_input = 1 / frequency_ref_input
    scale_nyquist_input = 2 / frequency_sample_rate_input
    scale_low_input = 1 / frequency_high_input
    if scale_low_input < scale_nyquist_input:
        scale_low_input = scale_nyquist_input
    scale_high_input = 1 / frequency_low_input

    (
        scale_order,
        scale_base,
        scale_band_number,
        scale_ref,
        scale_center_algebraic,
        scale_center_geometric,
        scale_start,
        scale_end,
    ) = band_intervals_periods(
        frequency_order_input, frequency_base_input, scale_ref_input, scale_low_input, scale_high_input
    )
    frequency_ref = 1 / scale_ref
    frequency_center_geometric = 1 / scale_center_geometric
    frequency_end = 1 / scale_start
    frequency_start = 1 / scale_end
    frequency_center_algebraic = (frequency_end + frequency_start) / 2.0

    # Inherit the order, base, and frequency band number (negative of period band number because of the inversion)
    return (
        scale_order,
        scale_base,
        -scale_band_number,
        frequency_ref,
        frequency_center_algebraic,
        frequency_center_geometric,
        frequency_start,
        frequency_end,
    )


def band_intervals_periods(
    scale_order_input: float,
    scale_base_input: float,
    scale_ref_input: float,
    scale_low_input: float,
    scale_high_input: float,
    show_warnings: bool = True,
) -> Tuple[float, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate Standard Logarithmic Interval Scale Parameters using time scales in seconds
    If scales are provided as frequency, previous computations convert to time.
    Designed to take bappsband to just below Nyquist, within a band edge.
    ALWAYS CONVERT TO SECONDS
    Last updated: 20200905

    :param scale_order_input: Band order N, for ISO3 use N = 1.0 or 3.0 or 6.0 or 12.0 or 24.0
    :param scale_base_input: reference base G; i.e. G3 = 10.**0.3 or G2 = 2.0
    :param scale_ref_input: time reference: in seconds
    :param scale_low_input: Lowest scale. If Nyquist scale, 2 * sample interval in seconds.
    :param scale_high_input: Highest scale of interest in seconds
    :param show_warnings: if True, show any warnings encountered.  Default True
    :return: scale_order, scale_base, scale_band_number, scale_ref, scale_center_algebraic,
            scale_center_geometric, scale_start, scale_end
    """
    # Initiate error handling, all inputs should be numeric, positive, and real
    # If not real and positive, make them so
    [scale_ref, scale_low, scale_high, scale_base, scale_order] = np.absolute(
        [scale_ref_input, scale_low_input, scale_high_input, scale_base_input, scale_order_input]
    )

    # Check for compliance with ISO3 and/or ANSI S1.11 and for scale_order = 1, 3, 6, 12, and 24
    if scale_base == Slice.G3 or scale_base == Slice.G2:
        pass
    elif scale_base < 1.0:
        if show_warnings:
            print("\nWARNING: Base must be greater than unity. Overriding to G = 2")
        scale_base = Slice.G2
    elif show_warnings:
        print("\nWARNING: Base is not ISO3 or ANSI S1.11 compliant")
        print(f"Continuing With Non-standard base = {scale_base}...")

    # Check for compliance with ISO3 for scale_order = 1, 3, 6, 12, and 24
    # and the two 'special' orders 0.75 and 1.5
    if scale_order in VALID_SCALE_ORDERS:
        pass
    elif scale_order < 0.75:
        if show_warnings:
            print("Order must be greater than 0.75. Overriding to Order 1")
        scale_order = 1
    elif show_warnings:
        print(f"\nWARNING: Recommend Orders {VALID_SCALE_ORDERS}")
        print(f"Continuing With Non-standard Order = {scale_order}...")

    # Compute scale edge and width parameters
    scale_edge = scale_base ** (1.0 / (2.0 * scale_order))
    scale_width = scale_edge - 1.0 / scale_edge

    if scale_low < Slice.T0S:
        scale_low = Slice.T0S / scale_edge
    if scale_high < scale_low:
        if show_warnings:
            print("\nWARNING: Upper scale must be larger than the lowest scale")
            print("Overriding to min = max/G\n")
        scale_low = scale_high / scale_base
    if scale_high == scale_low:
        if show_warnings:
            print("\nWARNING: Upper scale = lowest scale, returning closest band edges")
        scale_high *= scale_edge
        scale_low /= scale_edge

    # Max and min bands are computed relative to the center scale
    n_max = np.round(scale_order * np.log(scale_high / scale_ref) / np.log(scale_base))
    n_min = np.floor(scale_order * np.log(scale_low / scale_ref) / np.log(scale_base))

    # Evaluate min, ensure it stays below Nyquist period
    scale_center_n_min = scale_ref * np.power(scale_base, n_min / scale_order)
    if (scale_center_n_min < scale_low) or (scale_center_n_min / scale_edge < scale_low - get_epsilon()):
        n_min += 1

    # Check for band number anomalies
    if n_max < n_min:
        if show_warnings:
            print("\nSPECMOD: Insufficient bandwidth for Nth band specification")
            print(f"Minimum scaled bandwidth (scale_high - scale_low)/scale_center = {scale_width}")
            print("Correct scale High/Low input parameters")
            print("Apply one order")
        n_max = np.floor(np.log10(scale_high) / np.log10(scale_base))
        n_min = n_max - scale_order

    # Band number array for Nth octave
    scale_band_number = np.arange(n_min, n_max + 1)

    # Compute exact, evenly (log) distributed, constant Q,
    # Nth octave center and band edge frequencies
    scale_band_exponent = scale_band_number / scale_order

    scale_center_geometric = scale_ref * np.power(scale_base * np.ones(scale_band_number.shape), scale_band_exponent)
    scale_start = scale_center_geometric / scale_edge
    scale_end = scale_center_geometric * scale_edge
    # The spectrum is centered on the algebraic center scale
    scale_center_algebraic = (scale_start + scale_end) / 2.0

    return (
        scale_order,
        scale_base,
        scale_band_number,
        scale_ref,
        scale_center_algebraic,
        scale_center_geometric,
        scale_start,
        scale_end,
    )


def log_frequency_hz_from_fft_points(
    frequency_sample_hz: float,
    fft_points: int,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
) -> np.ndarray:
    """
    :param frequency_sample_hz: sample rate of frequency in Hz
    :param fft_points: number of fft points
    :param scale_order: Band order, preferably one of: 1, 3, 6, 12, 24.  Must be > 0.75 or reverts to N=0.75
    :param scale_ref_hz: reference frequency in Hz
    :param scale_base: scale base
    :return: array of scaled values
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
    # Shortest period, nominal 8/10 of Nyquist
    band_aa = int(np.ceil(order_over_log2base * (np.log2(2.5) - log2_ref)))
    # Longest period band
    band_max = int(np.floor(order_over_log2base * (log2_ave_life_dyad - log2_scale_mult - log2_ref)))

    # Stopped before Nyquist, before max
    # Stopped at 0.8 of Nyquist
    bands = np.arange(band_aa, band_max + 1)
    # Flip so frequency increases up to one band below Nyquist
    return np.flip(scale_ref_hz * scale_base ** (-bands / scale_order))
