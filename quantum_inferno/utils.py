"""
This module contains general utilities that can work with values containing nans.
TODO: Break up into smaller modules
TODO: Add tests
"""

import os
from typing import Tuple
import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from quantum_inferno.scales_dyadic import get_epsilon


""" Directory check & make """

# Now in exporting
def checkmake_dir(dir_name: str):
    """
    Check if the dir_name exists; if not, make it
    :param dir_name: directory name to check for
    """
    existing_dir: bool = os.path.isdir(dir_name)
    if not existing_dir:
        os.makedirs(dir_name)


""" Logical power of two flag """

# now in rescaling
def is_power_of_two(n: int) -> bool:
    """
    :param n: value to check
    :return True if n is positive and a power of 2, False otherwise
    """
    return n > 0 and not (n & (n - 1))


""" Time/Sample Duration Utils """
# add to calculations


def duration_points(sample_rate_hz: float, time_s: float) -> Tuple[int, int, int]:
    """
    Compute number of points

    :param sample_rate_hz: sample rate in Hz
    :param time_s: duration, period, or scale of time in seconds
    :return: number of points, floor and ceiling of log2 of number of points
    """
    points_float: float = sample_rate_hz * time_s
    return int(points_float), int(np.floor(np.log2(points_float))), int(np.ceil(np.log2(points_float)))


def duration_ceil(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute ceiling of the number of points, and convert to seconds

    :param sample_rate_hz: sample rate in Hz
    :param time_s: duration, period, or scale of time in seconds
    :return: ceil of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_ceil_log2: int = int(np.ceil(np.log2(sample_rate_hz * time_s)))
    points_ceil_pow2: int = 2 ** points_ceil_log2
    return points_ceil_log2, points_ceil_pow2, float(points_ceil_pow2 / sample_rate_hz)


def duration_floor(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute floor of the number of points, and convert to seconds

    :param sample_rate_hz: sample rate in Hz
    :param time_s: duration, period, or scale of time in seconds
    :return: floor of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_floor_log2: int = int(np.floor(np.log2(sample_rate_hz * time_s)))
    points_floor_pow2: int = 2 ** points_floor_log2
    return points_floor_log2, points_floor_pow2, float(points_floor_pow2 / sample_rate_hz)


""" Sampling Utils """

# added to windows
def taper_tukey(sig_wf_or_time: np.ndarray, fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window
    todo: is fraction cosine a scale and what are its limits. Reconcile with ref to alpha in rest of code
    :param sig_wf_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    return signal.windows.tukey(M=np.size(sig_wf_or_time), alpha=fraction_cosine, sym=True)


# added to calculations
# Integrals and derivatives
def integrate_cumtrapz(timestamps_s: np.ndarray, sensor_wf: np.ndarray, initial_value: float = 0) -> np.ndarray:
    """
    Cumulative trapezoid integration using scipy.integrate.cumulative_trapezoid
    Initiated by Kei 2021 06, work in progress. See blast_derivative_integral for validation.

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input.  Default 0
    :return: integrated data with the same length as the input
    """
    return cumulative_trapezoid(x=timestamps_s, y=sensor_wf, initial=initial_value)


def derivative_gradient(timestamps_s: np.ndarray, sensor_wf: np.ndarray) -> np.ndarray:
    """
    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input
    """
    return np.gradient(sensor_wf, timestamps_s)


def derivative_diff(timestamps_s: np.ndarray, sensor_wf: np.ndarray) -> np.ndarray:
    """
    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input. Hold/repeat last value
    """
    derivative_data0 = np.diff(sensor_wf) / np.diff(timestamps_s)
    derivative_data = np.append(derivative_data0, derivative_data0[-1])

    return derivative_data


# in scales_dyadic
def log2epsilon(x: np.ndarray) -> np.ndarray:
    """
    log 2 of the absolute value of linear amplitude, with EPSILON to avoid singularities

    :param x: time series or fft - not power
    :return: ndarray
    """
    y = np.log2(np.abs(x) + get_epsilon())
    return y
