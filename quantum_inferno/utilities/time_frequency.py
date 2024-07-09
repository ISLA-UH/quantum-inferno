"""
Methods for calculating frequency and time-frequency representations of signals.
Try to match all the defaults...
"""

import numpy as np
from scipy import signal
from typing import Tuple

from quantum_inferno.utilities.calculations import round_value


# return the Short-Time Fourier Transform (STFT) object with default parameters
def get_stft_object_tukey(
    sample_rate_hz: float, tukey_alpha: float, segment_length: int, overlap_length: int
) -> signal.ShortTimeFFT:
    """
    Return the Short-Time Fourier Transform (STFT) object with a Tukey window using ShortTimeFFT class
    Calculates the number of fft points based on the segment length using ceil_power_of_two rounding method

    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :return: ShortTimeFFT object
    """
    # checks
    if segment_length < overlap_length:
        print(
            f"overlap length {overlap_length} must be smaller than segment length {segment_length}"
            " using half of the segment length as the overlap length"
        )
        overlap_length = segment_length // 2

    # calculate the values to be used in the ShortTimeFFT object
    tukey_window = signal.windows.tukey(segment_length, alpha=tukey_alpha)
    fft_points = round_value(segment_length, "ceil_power_of_two")
    hop_length = segment_length - overlap_length

    # create the ShortTimeFFT object
    stft_obj = signal.ShortTimeFFT(
        win=tukey_window, hop=hop_length, fs=sample_rate_hz, mfft=fft_points, fft_mode="onesided", scale_to="magnitude"
    )

    return stft_obj


def stft_tukey(
    timeseries: np.ndarray, sample_rate_hz: float or int, tukey_alpha: float, segment_length: int, overlap_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Short-Time Fourier Transform (STFT) of a signal with a Tukey window using ShortTimeFFT class
    Returns the time, frequency, and magnitude of the STFT similar to legacy scipy.signal.stft

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :return: time, frequency, and magnitude of the STFT
    """

    # create the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length)

    # calculate the STFT with detrending
    stft_magnitude = stft_obj.stft_detrend(x=timeseries, detr="constant")

    # calculate the time and frequency bins
    time_bins = np.arange(start=0, stop=stft_obj.delta_t * np.shape(stft_magnitude)[1], step=stft_obj.delta_t)
    frequency_bins = stft_obj.f

    return frequency_bins, time_bins, stft_magnitude


# get inverse Short-Time Fourier Transform (iSTFT) with default parameters
def istft_tukey(
    stft_magnitude: np.ndarray,
    sample_rate_hz: float or int,
    tukey_alpha: float,
    segment_length: int,
    overlap_length: int,
) -> np.ndarray:
    """
    Calculate the inverse Short-Time Fourier Transform (iSTFT) of a signal with a Tukey window using ShortTimeFFT class

    :param stft_magnitude: magnitude of the STFT
    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :return: iSTFT of the signal
    """
    # create the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length)
    return stft_obj.istft(stft_magnitude)
