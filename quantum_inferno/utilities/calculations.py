"""
Methods for mathematical operations.
(Can't be named "math" because it's a built-in module)
"""

from scipy.integrate import cumulative_trapezoid
import numpy as np

FILL_LOCATIONS = ["start", "end"]
FILL_TYPES = ["zero", "nan", "mean", "median", "min", "max", "tail", "head"]
ROUNDING_TYPES = ["floor", "ceil", "round", "ceil_power_of_two", "floor_power_of_two"]
OUTPUT_TYPES = ["log2", "points", "pow2"]


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


def get_fill_from_filling_method(array_1d: np.ndarray, fill_type: str) -> float:
    """
    Returns the fill value based on the fill type

    Refer to FILL_TYPES for available options

    :param array_1d: 1D array with data to be filled
    :param fill_type: The fill type
    :return: The fill value
    """
    if len(np.shape(array_1d)) != 1:  # check if array_1d is a 1D array
        raise ValueError(f"array_1d has shape {np.shape(array_1d)} but should be a 1D array")

    if fill_type not in FILL_TYPES:
        raise ValueError(f"Invalid fill type {fill_type}, must be one of {FILL_TYPES}")
    elif fill_type == "zero":
        return 0
    elif fill_type == "nan":
        return np.nan
    elif fill_type == "mean":
        return np.mean(array_1d)  # check in place to insure a float is returned
    elif fill_type == "median":
        return np.median(array_1d)  # check in place to insure a float is returned
    elif fill_type == "min":
        return np.min(array_1d)
    elif fill_type == "max":
        return np.max(array_1d)
    elif fill_type == "tail":
        return array_1d[-1]
    elif fill_type == "head":
        return array_1d[0]


def append_fill(array_1d: np.ndarray, fill_value: float, fill_loc: str) -> np.ndarray:
    """
    Append fill value to the array based on the fill location

    Refer to FILL_LOCATIONS for available options

    :param array_1d: 1D array with data
    :param fill_value: fill value
    :param fill_loc: fill location
    :return: array with fill value appended
    """
    if fill_loc not in FILL_LOCATIONS:
        raise ValueError(f"Invalid fill location {fill_loc}, must be one of {FILL_LOCATIONS}")
    if fill_loc == "start":
        return np.insert(array_1d, 0, fill_value)
    elif fill_loc == "end":
        return np.append(array_1d, fill_value)


def derivative_with_difference_timestamps_s(
    timestamps_s: np.ndarray, timeseries: np.ndarray, fill_type: str = "zero", fill_loc: str = "end"
) -> np.ndarray:
    """
    Derivative using numpy.diff with fill options to return the same length as the input.

    Refer to FILL_TYPES for available options

    Refer to FILL_LOCATIONS for available options

    :param timestamps_s: timestamps in seconds
    :param timeseries: sensor waveform
    :param fill_type: fill type.  Default "zero"
    :param fill_loc: fill location.  Default "end"
    :return: derivative waveform with the same length as the input
    """
    derivative = np.diff(timeseries) / np.diff(timestamps_s)
    fill_value = get_fill_from_filling_method(derivative, fill_type)
    return append_fill(derivative, fill_value, fill_loc)


def derivative_with_difference_sample_rate_hz(
    sample_rate_hz: float, timeseries: np.ndarray, fill_type: str = "zero", fill_loc: str = "end"
) -> np.ndarray:
    """
    Derivative using numpy.diff with fill options to return the same length as the input

    Refer to FILL_TYPES for available options

    Refer to FILL_LOCATIONS for available options

    :param sample_rate_hz: sample rate in Hz
    :param timeseries: sensor waveform
    :param fill_type: fill type.  Default "zero"
    :param fill_loc: fill location.  Default "end"
    :return: derivative waveform with the same length as the input
    """
    derivative = np.diff(timeseries) * sample_rate_hz
    fill_value = get_fill_from_filling_method(derivative, fill_type)
    return append_fill(derivative, fill_value, fill_loc)


def round_value(value: float, rounding_type: str = "round") -> int:
    """
    Round value based on the rounding method for positive or negative floats.
    For rounding type "round", if the decimal is halfway between two integers, it will round to the nearest even integer

    Refer to ROUNDING_TYPES for available options

    :param value: value to be rounded
    :param rounding_type: method of rounding.  Default "round"
    :return: rounded value
    """
    if rounding_type not in ROUNDING_TYPES:
        raise ValueError(f"Invalid rounding type {rounding_type}, must be one of {ROUNDING_TYPES}")
    elif rounding_type == "floor":
        return int(np.floor(value))
    elif rounding_type == "ceil":
        return int(np.ceil(value))
    elif rounding_type == "round":
        return int(np.round(value))
    elif rounding_type == "ceil_power_of_two":
        return 2 ** int(np.ceil(np.log2(value)))
    elif rounding_type == "floor_power_of_two":
        return 2 ** int(np.floor(np.log2(value)))
    else:
        raise ValueError("Invalid rounding type")


def get_num_points(sample_rate_hz: float, duration_s: float, rounding_type: str, output_unit: str) -> int:
    """
    Get number of points in a waveform based on the sample rate and duration and round based on the rounding method.
    For rounding type "round", if the decimal is halfway between two integers, it will round to the nearest even integer

    :param sample_rate_hz: sample rate in Hz
    :param duration_s: duration in seconds
    :param rounding_type: rounding type
    :param output_unit: output unit (points, log2, or pow2)
    :return: number of points
    """
    if output_unit not in OUTPUT_TYPES:
        raise ValueError(f"Invalid output unit {output_unit}, must be one of {OUTPUT_TYPES}")
    elif output_unit == "points":
        return round_value(sample_rate_hz * duration_s, rounding_type)
    elif output_unit == "log2":
        return round_value(np.log2(sample_rate_hz * duration_s), rounding_type)
    elif output_unit == "pow2":
        return round_value(2 ** (sample_rate_hz * duration_s), rounding_type)
