"""
scales_dyadic.py
Physical to cyber conversion with preferred quasi-dyadic orders.
Compute the complete set of Nth order bands from Nyquist to the averaging frequency.
Compute the standard deviation of the Gaussian envelope for the averaging frequency.
Example: base10, base2, dB, bits

"""

import numpy as np
from typing import Tuple, Union
import sys

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


# return epsilon for float64, float32, float16 by detecting system max size
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
    ORD6 = 6.0  # 1/6 octaves, good compromise for time-frequency resolition
    ORD12 = 12.0  # Musical tones, continuous waves, long duration sweeps
    ORD24 = 24.0  # High resolution; long duration widow, 24 bands per octave
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

DEFAULT_SCALE_ORDER_MIN: float = 0.75  # Garces (2022)
DEFAULT_FFT_POW2_POINTS_MAX: int = 2 ** 15  # Computational FFT limit, tune to computing system
DEFAULT_FFT_POW2_POINTS_MIN: int = 2 ** 8  # For a tolerable display
DEFAULT_MESH_POW2_PIXELS: int = 2 ** 19  # Total of pixels per mesh, tune to plotting engine
DEFAULT_TIME_DISPLAY_S: float = 60.0  # Physical time to display; sets display truncation


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
        print(
            f"** Warning from scales_dyadic.scale_order_check:\n"
            f"N < {DEFAULT_SCALE_ORDER_MIN} specified, overriding using N = {DEFAULT_SCALE_ORDER_MIN}"
        )
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
        cycles_per_scale = 1.0
    return scale_order_check(cycles_per_scale / M_OVER_N)


def base_multiplier(scale_order: float = DEFAULT_SCALE_ORDER, scale_base: float = DEFAULT_SCALE_BASE):
    """
    Dyadic (log2) foundation for arbitrary base

    :param scale_order:
    :param scale_base:
    :return:
    """
    return scale_order_check(scale_order) / np.log2(scale_base)


def scale_from_frequency_hz(
    scale_order: float, scale_frequency_center_hz: Union[np.ndarray, float], frequency_sample_rate_hz: float
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Non-dimensional scale and angular frequency for canonical Gabor/Morlet wavelet

    :param scale_order:
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: scale_atom, scaled angular frequency
    """
    scale_angular_frequency = 2.0 * np.pi * scale_frequency_center_hz / frequency_sample_rate_hz
    scale_atom = cycles_from_order(scale_order) / scale_angular_frequency
    return scale_atom, scale_angular_frequency


# band_Frequency_low_high from libquantum -> used in synthetics
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
    :param frequency_low_input: lowest frequency of interest
    :param frequency_high_input: highest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :return:scale_order (Band order N > 1, defaults to 1.),
        scale_base (positive reference Base G > 1, defaults to G3),
        scale_band_number (Band number n),
        frequency_ref (reference frequency value),
        frequency_center_algebraic (Algebraic center of frequencies),
        frequency_center_geometric (Geometric center of frequencies),
        frequency_start (first frequency),
        frequency_end (last frequency)
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


# band_intervals_periods from libquantum -> used in band_frequency_low_high
def band_intervals_periods(
    scale_order_input: float,
    scale_base_input: float,
    scale_ref_input: float,
    scale_low_input: float,
    scale_high_input: float,
) -> Tuple[float, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Standard Logarithmic Interval Scale Parameters using time scales in seconds
    If scales are provided as frequency, previous computations convert to time.
    Designed to take bappsband to just below Nyquist, within a band edge.
    ALWAYS CONVERT TO SECONDS
    Last updated: 20200905

    Parameters
    ----------
    scale_order_input: float
        Band order N, for ISO3 use N = 1.0 or 3.0 or 6.0 or 12.0 or 24.0
    scale_base_input: float
        reference base G; i.e. G3 = 10.**0.3 or G2 = 2.0
    scale_ref_input: float
        time reference: in seconds
    scale_low_input: float
        Lowest scale. If Nyquist scale, 2 * sample interval in seconds.
    scale_high_input: float
        Highest scale of interest in seconds

    Returns
    -------
    scale_order: float
        Band order N > 1, defaults to 1.
    scale_base: float
        positive reference Base G > 1, defaults to G3
    scale_ref: float
        positive reference scale
    scale_band_number: numpy ndarray
        Band number n
    scale_center_algebraic: numpy ndarray
        Algebraic center band scale
    scale_center_geometric: numpy ndarray
        Geometric center band scale
    scale_start: numpy ndarray
        Lower band edge scale
    scale_end: numpy ndarray
        Upper band edge scale
    """
    # Initiate error handling, all inputs should be numeric, positive, and real
    # Need error check for string inputs
    # If not real and positive, make them so
    [scale_ref, scale_low, scale_high, scale_base, scale_order] = np.absolute(
        [scale_ref_input, scale_low_input, scale_high_input, scale_base_input, scale_order_input]
    )

    # print('\nConstructing standard band intervals')
    # Check for compliance with ISO3 and/or ANSI S1.11 and for scale_order = 1, 3, 6, 12, and 24
    if scale_base == Slice.G3:
        pass
    elif scale_base == Slice.G2:
        pass
    elif scale_base < 1.0:
        print("\nWARNING: Base must be greater than unity. Overriding to G = 2")
        scale_base = Slice.G2
    else:
        print("\nWARNING: Base is not ISO3 or ANSI S1.11 compliant")
        print(("Continuing With Non-standard base = " + repr(scale_base) + " ..."))

    # Check for compliance with ISO3 for scale_order = 1, 3, 6, 12, and 24
    # and the two 'special' orders 0.75 and 1.5
    valid_scale_orders = (0.75, 1, 1.5, 3, 6, 12, 24, 48)
    if scale_order in valid_scale_orders:
        pass
    elif scale_order < 0.75:
        print("Order must be greater than 0.75. Overriding to Order 1")
        scale_order = 1
    else:
        print(("\nWARNING: Recommend Orders " + str(valid_scale_orders)))
        print(("Continuing With Non-standard Order = " + repr(scale_order) + " ..."))

    # Compute scale edge and width parameters
    scale_edge = scale_base ** (1.0 / (2.0 * scale_order))
    scale_width = scale_edge - 1.0 / scale_edge

    if scale_low < Slice.T0S:
        scale_low = Slice.T0S / scale_edge
    if scale_high < scale_low:
        print("\nWARNING: Upper scale must be larger than the lowest scale")
        print("Overriding to min = max/G  \n")
        scale_low = scale_high / scale_base
    if scale_high == scale_low:
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
        print("\nSPECMOD: Insufficient bandwidth for Nth band specification")
        print(("Minimum scaled bandwidth (scale_high - scale_low)/scale_center = " + repr(scale_width) + "\n"))
        print("Correct scale High/Low input parameters \n")
        print("Apply one order  \n")
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


def gaussian_sigma_from_frequency(
    frequency_sample_hz: float, frequency_hz: np.ndarray, scale_order: float = DEFAULT_SCALE_ORDER
):
    """
    Standard deviation (sigma) of Gaussian envelope from frequency (Garces, 2023)
    Use 3/8 = 0.375

    :param frequency_hz: physical frequency in Hz
    :param frequency_sample_hz: sensor sample rate in Hz
    :param scale_order: Scale order
    :return: standard deviation from physical frequency in Hz
    """
    return 0.375 * scale_order * frequency_sample_hz / frequency_hz


def scale_bands_from_ave(
    frequency_sample_hz: float,
    frequency_ave_hz: float,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
):
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
    return [
        int(np.ceil(order_over_log2base * (1 - log2_ref))),
        int(np.round(order_over_log2base * np.log2(scale_ref_hz / frequency_ave_hz))),
        int(np.floor(order_over_log2base * (log2_ave_life_dyad - log2_scale_mult - log2_ref))),
        log2_ave_life_dyad,
    ]


def scale_band_from_frequency(
    frequency_input_hz: float,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
):
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
    log2_in_physical = np.log2(scale_ref_hz / frequency_input_hz)
    # Closest period band, TODO: check for duplicates if vectorized
    band_number = int(np.round(base_multiplier(scale_order, scale_base) * log2_in_physical))
    band_frequency_hz = scale_ref_hz * scale_base ** (-band_number / scale_order)

    return [band_number, band_frequency_hz]


def scale_bands_from_pow2(
    frequency_sample_hz: float,
    log2_ave_life_dyad: int,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
):
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
    log2_ref = np.log2(frequency_sample_hz / scale_ref_hz)
    return [
        int(np.ceil(order_over_log2base * (1 - log2_ref))),
        int(np.floor(order_over_log2base * (log2_ave_life_dyad - log2_scale_mult - log2_ref))),
    ]


def period_from_bands(
    band_min: int,
    band_max: int,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
):
    """

    :param band_min:
    :param band_max:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return:
    """
    # Increasing order
    return scale_base ** (np.arange(band_min, band_max + 1) / scale_order) / scale_ref_hz


def frequency_from_bands(
    band_min: int,
    band_max: int,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
):
    """

    :param band_min:
    :param band_max:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return:
    """
    # Flip the result so it increases
    return np.flip(scale_ref_hz * scale_base ** (-np.arange(band_min, band_max + 1) / scale_order))


def log_frequency_hz_from_fft_points(
    frequency_sample_hz: float,
    fft_points: int,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
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
    frequency_logarithmic_hz = np.flip(scale_ref_hz * scale_base ** (-bands / scale_order))
    return frequency_logarithmic_hz


def lin_frequency_hz_from_fft_points(
    frequency_sample_hz: float,
    fft_points: int,
    scale_order: float = DEFAULT_SCALE_BASE,
    scale_ref_hz: float = DEFAULT_REF_FREQUENCY_HZ,
    scale_base: float = DEFAULT_SCALE_BASE,
) -> np.ndarray:
    """

    :param frequency_sample_hz:
    :param fft_points:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return:
    """
    frequency_log_hz = log_frequency_hz_from_fft_points(
        frequency_sample_hz, fft_points, scale_order, scale_ref_hz, scale_base
    )
    frequency_linear_hz = np.linspace(start=frequency_log_hz[0], stop=frequency_log_hz[-1], num=len(frequency_log_hz))
    return frequency_linear_hz
