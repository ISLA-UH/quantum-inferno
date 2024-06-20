"""
A set of functions to pick key portions of a signal.
"""

from enum import Enum
from typing import Tuple

import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt

from quantum_inferno.utilities.date_time import convert_time_unit
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon


class ExtractionType(Enum):
    SIGMAX: str = "sigmax"  # Extract signal using the largest positive value
    SIGMIN: str = "sigmin"  # Extract signal using the largest negative value
    SIGABS: str = "sigabs"  # Extract signal using the largest absolute value
    SIGBIT: str = "sigbit"  # Extract signal using the largest absolute value in bits


class ScalingType(Enum):
    AMPS: str = "amps"  # Signal is in amplitudes
    BITS: str = "bits"  # Signal is in bits


def find_sample_rate_hz_from_timestamps(timestamps: np.ndarray, time_unit: str = "s") -> float:
    """
    Find the sample rate from timestamps in a given time unit (picoseconds to years defined in convert_time_unit())

    :param timestamps: input timestamps
    :param time_unit: time unit of the timestamps (default to "s")
    :return: sample rate in Hz
    """
    timestamps_seconds = convert_time_unit(timestamps, time_unit, "s")
    return 1.0 / np.mean(np.diff(timestamps_seconds))


def scale_signal_by_extraction_type(signal: np.ndarray, extraction_type: ExtractionType) -> np.ndarray:
    """
    Normalize the signal based on the extraction type

    :param signal: input signal
    :param extraction_type: extraction type
    :return: normalized signal
    """
    if extraction_type == ExtractionType.SIGMAX:
        return signal / np.nanmax(signal)
    elif extraction_type == ExtractionType.SIGMIN:
        return signal / np.nanmax(-signal)
    elif extraction_type == ExtractionType.SIGABS:
        return signal / np.nanmax(np.abs(signal))
    elif extraction_type == ExtractionType.SIGBIT:
        return to_log2_with_epsilon(signal)


def apply_bandpass(
    timeseries: np.ndarray, filter_band: Tuple[float, float], sample_rate_hz: float, filter_order: int = 7
) -> np.ndarray:
    """
    Apply a bandpass filter to the timeseries data

    :param timeseries: input signal
    :param filter_band: bandpass filter band
    :param sample_rate_hz: sample rate of the signal
    :param filter_order: order of the filter
    :return: filtered signal
    """
    if filter_band[0] < 0 or filter_band[1] > sample_rate_hz / 2:
        raise ValueError(f"Invalid bandpass filter band, {filter_band}, for sample rate {sample_rate_hz}")
    if filter_band[0] >= filter_band[1]:
        raise ValueError(
            f"Invalid bandpass filter band, {filter_band}, the lower bound must be less than the upper bound"
        )
    sos = butter(filter_order, filter_band, fs=sample_rate_hz, btype="band", output="sos")
    return sosfiltfilt(sos, timeseries)


def find_peaks_by_extraction_type_with_bandpass(
    timeseries: np.ndarray,
    filter_band: Tuple[float, float],
    sample_rate_hz: float,
    filter_order: int = 7,
    extraction_type: ExtractionType = ExtractionType.SIGMAX,
    height: float or None = 0.7,
    *args,
) -> np.ndarray:
    """
    Find peaks in the timeseries data using a normalized bandpass filter

    :param timeseries: input signal
    :param filter_band: bandpass filter band
    :param sample_rate_hz: sample rate of the signal
    :param filter_order: order of the filter (default 7)
    :param extraction_type: extraction type (default SIGMAX)
    :param height: minimum height for the peaks (default 0.7)
    :param args: additional arguments for scipy's find_peaks
    :return: location of peaks in the timeseries data
    """
    filtered_timeseries = apply_bandpass(timeseries, filter_band, sample_rate_hz, filter_order)
    scaled_filtered_timeseries = scale_signal_by_extraction_type(filtered_timeseries, extraction_type)

    return signal.find_peaks(scaled_filtered_timeseries, height=height, *args)[0]


def find_peaks_by_extraction_type(
    timeseries: np.ndarray, extraction_type: ExtractionType = ExtractionType.SIGMAX, height: float or None = 0.7, *args
) -> np.ndarray:
    """
    Find peaks in the timeseries data by extraction type

    :param timeseries: input signal
    :param extraction_type: extraction type (default SIGMAX)
    :param height: minimum height for the peaks (default 0.7)
    :param args: additional arguments for scipy's find_peaks
    :return: location of peaks in the timeseries data
    """
    scaled_timeseries = scale_signal_by_extraction_type(timeseries, extraction_type)

    return signal.find_peaks(scaled_timeseries, height=height, *args)[0]


def find_peaks_with_bits(
    timeseries: np.ndarray,
    sample_rate_hz: float,
    scaling_type: ScalingType = ScalingType.AMPS,
    threshold_bits: int or None = 1,
    time_distance_seconds: float or None = 0.1,
    *args,
) -> np.ndarray:
    """
    Find peaks in the timeseries data with a threshold in bits (originally picker_signal_finder)
    :param timeseries: time series
    :param sample_rate_hz: sample rate of the signal
    :param scaling_type: scaling type of the signal (default AMPS)
    :param threshold_bits: threshold in bits (default 1)
    :param time_distance_seconds: minimum time distance between peaks in seconds (default 0.1)
    :param args: additional arguments for scipy's find_peaks
    :return: location of peaks in the timeseries data
    """
    timeseries_in_bits = to_log2_with_epsilon(timeseries)

    if scaling_type == ScalingType.BITS:
        height = np.max(timeseries_in_bits) - threshold_bits
    else:
        height = np.max(timeseries) - 2 ** threshold_bits

    return signal.find_peaks(
        timeseries_in_bits, height=height, distance=int(time_distance_seconds * sample_rate_hz), *args
    )[0]


def extract_signal_index_with_buffer(
    sample_rate_hz: float, peak: int, intro_buffer_s: float, outro_buffer_s: float
) -> Tuple[int, int]:
    """
    Extract start and end index of the extracted signal with a buffer around the peak

    :param sample_rate_hz: sample rate of the signal
    :param peak: peak location
    :param intro_buffer_s: intro buffer in seconds
    :param outro_buffer_s: outro buffer in seconds
    :return: start and end index of the extracted signal
    """
    if intro_buffer_s < 0 or outro_buffer_s < 0:
        raise ValueError(f"Negative intro_buffer_s or outro_buffer_s, {intro_buffer_s}, {outro_buffer_s}")
    return peak - int(intro_buffer_s * sample_rate_hz), peak + int(outro_buffer_s * sample_rate_hz)


def extract_signal_with_buffer_seconds(
    timeseries: np.ndarray, sample_rate_hz, peak: int, intro_buffer_s: float, outro_buffer_s: float
) -> np.ndarray:
    """
    Extract a signal with a buffer in seconds around the peak

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the signal
    :param peak: peak location
    :param intro_buffer_s: intro buffer in seconds
    :param outro_buffer_s: outro buffer in seconds
    :return: extracted signal
    """
    intro_index, outro_index = extract_signal_index_with_buffer(sample_rate_hz, peak, intro_buffer_s, outro_buffer_s)

    if intro_index < 0:
        print(f"Warning: intro buffer exceeds the signal length, intro_index: {intro_index}")
        intro_index = 0
    if outro_index > len(timeseries):
        print(f"Warning: outro buffer exceeds the signal length, outro_index: {outro_index}")
        outro_index = len(timeseries)

    return timeseries[intro_index:outro_index]


def find_peaks_to_comb_function(timeseries: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """
    Returns a comb function of the same length as the timeseries with 1s at the peak locations and 0s elsewhere
    :param timeseries: input signal
    :param peaks: peak locations
    :return: a comb function with the peak locations
    """
    comb_function = np.zeros(len(timeseries))
    comb_function[peaks] = 1
    return comb_function
