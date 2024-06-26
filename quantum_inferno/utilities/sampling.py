"""
This module contains methods used in resampling signals
"""
from typing import Tuple
from enum import Enum

import numpy as np
from scipy.signal import resample, decimate


class SubsampleMethod(Enum):
    AVG = "average"  # use the average of the subset of samples
    MED = "median"  # use the median of the subset of samples
    MAX = "max"  # use the maximum of the subset of samples
    MIN = "min"  # use the minimum of the subset of samples
    NTH = "nth"  # use every nth sample


def subsample(
    timeseries: np.ndarray, sample_rate_hz: float, subsample_factor: int, method: SubsampleMethod = SubsampleMethod.NTH
) -> Tuple[np.ndarray, float]:
    """
    Subsample a time series by given method (default is every nth sample).
    Truncates the signal if the length is not divisible by the subsample factor, except for the nth method.
    if subsample_factor is less than 2, return the original signal.

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the signal
    :param subsample_factor: factor to subsample by (returns every nth sample)
    :param method: method to use for subsampling
    :return: subsampled signal and new sample rate
    """
    if subsample_factor < 2:
        print(f"Warning: subsample factor is less than 2, returning the original signal")
        return timeseries, sample_rate_hz

    new_sample_rate = sample_rate_hz / subsample_factor

    if method != SubsampleMethod.NTH and len(timeseries) % subsample_factor != 0:
        print("here")
        print(len(timeseries))
        timeseries = timeseries[: -(len(timeseries) % subsample_factor)]
        print(len(timeseries))

    if method == SubsampleMethod.NTH:
        return timeseries[::subsample_factor], new_sample_rate
    elif method == SubsampleMethod.AVG:
        return np.mean(timeseries.reshape(-1, subsample_factor), axis=1), new_sample_rate
    elif method == SubsampleMethod.MED:
        return np.median(timeseries.reshape(-1, subsample_factor), axis=1), new_sample_rate
    elif method == SubsampleMethod.MAX:
        return np.max(timeseries.reshape(-1, subsample_factor), axis=1), new_sample_rate
    elif method == SubsampleMethod.MIN:
        return np.min(timeseries.reshape(-1, subsample_factor), axis=1), new_sample_rate


def resample_uneven_timeseries(
    timeseries: np.ndarray, timestamps_s: np.ndarray, new_sample_rate_hz: float or None = None
) -> Tuple[np.ndarray, float]:
    """
    Resample uneven time series using linear interpolation.
    If new_sample_rate_hz is None, the new sample rate is the average sample rate of the input signal.

    :param timeseries: input signal
    :param timestamps_s: timestamps of the input signal in seconds
    :param new_sample_rate_hz: the sample rate to resample to in Hz (default is None)
    :return: resampled signal and new sample rate
    """
    if new_sample_rate_hz is None:
        new_sample_rate_hz = 1 / np.mean(np.diff(timestamps_s))
    new_timestamps = np.arange(timestamps_s[0], timestamps_s[-1], 1 / new_sample_rate_hz)
    return np.interp(new_timestamps, timestamps_s, timeseries), new_sample_rate_hz


def resample_with_sample_rate(
    timeseries: np.ndarray, sample_rate_hz: float, new_sample_rate_hz: float
) -> Tuple[np.ndarray, float]:
    """
    Resample a time series to a new sample rate using scipy.signal.resample.

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the input signal
    :param new_sample_rate_hz: the sample rate to resample to in Hz
    :return: resampled signal and new sample rate
    """
    new_length = int(len(timeseries) * new_sample_rate_hz / sample_rate_hz)
    return resample(timeseries, new_length), new_sample_rate_hz


# subsample a 2d array along the second axis
def subsample_2d(array: np.ndarray, subsample_factor: int, method: SubsampleMethod = SubsampleMethod.NTH) -> np.ndarray:
    """
    Subsample a 2D array along the second axis.
    Truncates the signal if the length is not divisible by the subsample factor.
    if subsample_factor is less than 2, return the original signal.

    :param array: input 2D array
    :param subsample_factor: factor to subsample
    :param method: method to use for subsampling (default is every nth sample)
    :return: subsampled 2D array
    """
    if subsample_factor < 2:
        print(f"Warning: subsample factor is less than 2, returning the original signal")
        return array

    if method != SubsampleMethod.NTH:
        remainder = array.shape[1] % subsample_factor
        if remainder != 0:
            array = array[:, : -remainder]

    if method == SubsampleMethod.NTH:
        return array[:, ::subsample_factor]
    # todo: this doesn't work.  either do subsampling per subarray or just not do it
    elif method == SubsampleMethod.AVG:
        x = array.reshape((array.shape[0], -1))
        return np.mean(array.reshape((array.shape[0], -1)), axis=1)
    elif method == SubsampleMethod.MED:
        return np.median(array.reshape(array.shape[0], -1), axis=1)
    elif method == SubsampleMethod.MAX:
        return np.max(array.reshape(array.shape[0], -1), axis=1)
    elif method == SubsampleMethod.MIN:
        return np.min(array.reshape(array.shape[0], -1), axis=1)


# decimate a 1d array
def decimate_timeseries(timeseries: np.ndarray, decimation_factor: int) -> np.ndarray:
    """
    Decimate a time series by a given factor using scipy.signal.decimate.
    Timeseries must be 28 samples or longer.

    :param timeseries: input signal
    :param decimation_factor: factor to decimate by
    :return: decimated signal
    """
    return decimate(timeseries, decimation_factor, zero_phase=True)


# decimate a collection of timeseries with the same sample rate at once
def decimate_timeseries_collection(timeseries_collection: np.ndarray, decimation_factor: int) -> np.ndarray:
    """
    Decimate a collection of time series with the same sample rate at once using scipy.signal.decimate.
    Each timeseries must be 28 samples or longer.

    :param timeseries_collection: input signal
    :param decimation_factor: factor to decimate by
    :return: decimated signal
    """
    return decimate(timeseries_collection, decimation_factor, axis=1, zero_phase=True)
