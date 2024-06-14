"""
A set of functions to pick key portions of a signal.
"""

from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from quantum_inferno.scales_dyadic import get_epsilon
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon
from scipy.signal import butter, sosfiltfilt


class ExtractionType(Enum):
    SIGMAX: str = "sigmax"  # Extract signal using the largest positive value
    SIGMIN: str = "sigmin"  # Extract signal using the largest negative value
    SIGABS: str = "sigabs"  # Extract signal using the largest absolute value
    SIGBIT: str = "sigbit"  # Extract signal using the largest absolute value in bits


class ScalingType(Enum):
    BITS: str = "bits"  # data is in bits
    AMPS: str = "amplitude"  # data is in amplitude


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


def find_peaks_with_bandpass(
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
    """
    filtered_timeseries = apply_bandpass(timeseries, filter_band, sample_rate_hz, filter_order)
    scaled_filtered_timeseries = scale_signal_by_extraction_type(filtered_timeseries, extraction_type)

    return signal.find_peaks(scaled_filtered_timeseries, height=height, *args)[0]


# # test find_peaks on a sine function
# time_series_buffer = np.linspace(0, 1, 1000)
# time_series = np.sin(2 * np.pi * 2 * time_series_buffer)
# print(signal.find_peaks(time_series, height=0.7)[0])
#
# plt.Figure()
# plt.plot(time_series_buffer, time_series)
# plt.show()
