"""
Methods for mathematical operations.
(Can't be named "math" because it's a built-in module)
"""

from enum import Enum

from scipy.integrate import cumulative_trapezoid
import numpy as np


class FillLoc(Enum):
    """
    Enumeration for fill locations
    """
    START = "start"
    END = "end"


class FillType(Enum):
    """
    Enumeration for fill types
    """
    ZERO = "zero"
    NAN = "nan"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    TAIL = "tail"
    HEAD = "head"


class RoundingType(Enum):
    """
    Enumeration for rounding types
    """
    FLOOR = "floor"
    CEIL = "ceil"
    ROUND = "round"


class OutputType(Enum):
    """
    Enumeration for output types
    """
    BITS = "bits"
    POINTS = "points"


# cumulative trapezoidal integration
def integrate_with_cumtrapz_timestamps_s(
    timestamps_s: np.ndarray, timeseries: np.ndarray, initial_value: float = 0
) -> np.ndarray:
    """
    Cumulative trapezoid integration using scipy.integrate.cumulative_trapezoid

    :param timestamps_s: timestamps in seconds
    :param timeseries: sensor waveform
    :param initial_value: initial value of the integral
    :return: integrated waveform
    """
    return cumulative_trapezoid(y=timeseries, x=timestamps_s, initial=initial_value)


def integrate_with_cumtrapz_sample_rate_hz(
    sample_rate_hz: float, timeseries: np.ndarray, initial_value: float = 0
) -> np.ndarray:
    """
    Cumulative trapezoid integration using scipy.integrate.cumulative_trapezoid

    :param sample_rate_hz: sample rate in Hz
    :param timeseries: sensor waveform
    :param initial_value: initial value of the integral
    :return: integrated waveform
    """
    return cumulative_trapezoid(y=timeseries, dx=1 / sample_rate_hz, initial=initial_value)


def derivative_with_gradient_timestamps_s(timestamps_s: np.ndarray, timeseries: np.ndarray) -> np.ndarray:
    """
    Derivative using numpy.gradient

    :param timestamps_s: timestamps in seconds
    :param timeseries: sensor waveform
    :return: derivative waveform
    """
    return np.gradient(timeseries, timestamps_s)


def derivative_with_gradient_sample_rate_hz(sample_rate_hz: float, timeseries: np.ndarray) -> np.ndarray:
    """
    Derivative using numpy.gradient

    :param sample_rate_hz: sample rate in Hz
    :param timeseries: sensor waveform
    :return: derivative waveform
    """
    return np.gradient(timeseries, 1 / sample_rate_hz)


def get_fill_from_filling_method(array_1d: np.ndarray, fill_type: FillType) -> float:
    """
    Returns the fill value based on the fill type

    :param array_1d: 1D array with data to be filled
    :param fill_type: The fill type
    :return: The fill value
    """
    if np.shape(array_1d) != ():  # check if array_1d is a 1D array
        raise ValueError(f"array_1d has shape {np.shape(array_1d)} but should be a 1D array")

    elif fill_type == FillType.ZERO:
        return 0
    elif fill_type == FillType.NAN:
        return np.nan
    elif fill_type == FillType.MEAN:
        return np.mean(array_1d)  # check in place to insure a float is returned
    elif fill_type == FillType.MEDIAN:
        return np.median(array_1d)  # check in place to insure a float is returned
    elif fill_type == FillType.MIN:
        return np.min(array_1d)
    elif fill_type == FillType.MAX:
        return np.max(array_1d)
    elif fill_type == FillType.TAIL:
        return array_1d[-1]
    elif fill_type == FillType.HEAD:
        return array_1d[0]
    else:
        raise ValueError("Invalid fill type")


def append_fill(array_1d: np.ndarray, fill_value: float, fill_loc: FillLoc) -> np.ndarray:
    """
    Append fill value to the array based on the fill location

    :param array_1d: 1D array with data
    :param fill_value: fill value
    :param fill_loc: fill location
    :return: array with fill value appended
    """
    if fill_loc == FillLoc.START:
        return np.insert(array_1d, 0, fill_value)
    elif fill_loc == FillLoc.END:
        return np.append(array_1d, fill_value)
    else:
        raise ValueError("Invalid fill location")


def derivative_with_difference_timestamps_s(
    timestamps_s: np.ndarray, timeseries: np.ndarray, fill: FillType = FillType.ZERO, fill_loc: FillLoc = FillLoc.END
) -> np.ndarray:
    """
    Derivative using numpy.diff with fill options to return the same length as the input

    :param timestamps_s: timestamps in seconds
    :param timeseries: sensor waveform
    :param fill: fill type
    :param fill_loc: fill location
    :return: derivative waveform with the same length as the input
    """
    derivative = np.diff(timeseries) / np.diff(timestamps_s)
    fill_value = get_fill_from_filling_method(derivative, fill)
    return append_fill(derivative, fill_value, fill_loc)


def derivative_with_difference_sample_rate_hz(
    sample_rate_hz: float, timeseries: np.ndarray, fill: FillType = FillType.ZERO, fill_loc: FillLoc = FillLoc.END
) -> np.ndarray:
    """
    Derivative using numpy.diff with fill options to return the same length as the input

    :param sample_rate_hz: sample rate in Hz
    :param timeseries: sensor waveform
    :param fill: fill type
    :param fill_loc: fill location
    :return: derivative waveform with the same length as the input
    """
    derivative = np.diff(timeseries) * sample_rate_hz
    fill_value = get_fill_from_filling_method(derivative, fill)
    return append_fill(derivative, fill_value, fill_loc)


# return round based on the rounding method
def round_value(value: float, rounding: RoundingType) -> int:
    """
    Round value based on the rounding method for positive or negative floats
    For rounding type ROUND, if the decimals is halfway between two integers, it will round to the nearest even integer

    :param value: value to be rounded
    :param rounding: rounding type
    :return: rounded value
    """
    if rounding == RoundingType.FLOOR:
        return int(np.floor(value))
    elif rounding == RoundingType.CEIL:
        return int(np.ceil(value))
    elif rounding == RoundingType.ROUND:
        return int(np.round(value))
    else:
        raise ValueError("Invalid rounding type")


# get number of points in a waveform based on the sample rate and duration and round based on the rounding method
def get_num_points(sample_rate_hz: float, duration_s: float, round_type: RoundingType, unit: OutputType) -> int:
    """
    TODO: Make defaults for RoundingType, OutputType, add POW2 option/function. BITS is maybe not the best descriptor?
    Get number of points in a waveform based on the sample rate and duration and round based on the rounding method
    For rounding type ROUND, if the decimals is halfway between two integers, it will round to the nearest even integer

    :param sample_rate_hz: sample rate in Hz
    :param duration_s: duration in seconds
    :param round_type: rounding type
    :param unit: output type (POINTS or BITS)
    :return: number of points
    """
    if unit == OutputType.POINTS:
        return round_value(sample_rate_hz * duration_s, round_type)
    elif unit == OutputType.BITS:
        return round_value(np.log2(sample_rate_hz * duration_s), round_type)
    else:
        raise ValueError("Invalid output type")
