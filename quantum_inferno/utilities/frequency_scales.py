"""
Utilities for calculating frequencies for both linear and logarithmic scales.
"""

import numpy as np
import scipy.signal as signal


def get_linear_frequency_bins_range(
    sample_rate_hz: float, segment_length: int, start_hz: float = None, end_hz: float = None
) -> np.ndarray:
    """
    Get the frequency bins with given sample rate and segment length that matches output from scipy.signal.spectrogram.
    Default starts at 0 Hz and ends at Nyquist frequency.

    :param sample_rate_hz: sample rate of the signal
    :param segment_length: length of the segment
    :param start_hz: start frequency in Hz
    :param end_hz: end frequency in Hz
    :return: frequency bins
    """
    # Default values
    if start_hz is None:
        start_hz = 0
    if end_hz is None:
        end_hz = sample_rate_hz / 2

    # Check values
    if start_hz < 0:
        print(f"Waring: start_hz ({start_hz}) is less than 0, setting to 0")
        start_hz = 0
    if end_hz > sample_rate_hz / 2:
        print(f"Waring: end_hz ({end_hz}) is greater than Nyquist frequency, setting to Nyquist frequency")
        end_hz = sample_rate_hz / 2
    if start_hz > end_hz:
        print(f"Waring: start_hz ({start_hz}) is greater than end_hz ({end_hz}), setting to 0 and Nyquist frequency")
        start_hz = 0
        end_hz = sample_rate_hz / 2
    if segment_length < 0:
        ValueError(f"sample_rate_hz ({sample_rate_hz}) is less than 0")
    if sample_rate_hz < 0:
        raise ValueError(f"sample_rate_hz ({sample_rate_hz}) is less than 0")
    if segment_length > sample_rate_hz:
        print(
            f"Warning: segment_length ({segment_length}) is greater than sample_rate_hz ({sample_rate_hz})"
            f", setting to sample_rate_hz"
        )
        segment_length = sample_rate_hz

    frequency_step = sample_rate_hz / segment_length
    return np.arange(start=start_hz, stop=end_hz + frequency_step, step=frequency_step)


def get_shorttime_fft_frequency_bins(sample_rate_hz: float, segment_length: int) -> np.ndarray:
    """
    Get the frequency bins with given sample rate and segment length that matches output from ShortTimeFFT.
    Starts at 0 Hz and ends at Nyquist frequency.

    :param sample_rate_hz: sample rate of the signal
    :param segment_length: length of the segment
    :return: frequency bins
    """
    return get_linear_frequency_bins_range(sample_rate_hz, segment_length)


def get_log_central_frequency_bins_range(
    sample_rate_hz: float, band_order: float, start_hz: float = None, end_hz: float = None
) -> np.ndarray:
    """
    Get the central frequency bins with given sample rate, band order, start and end frequency.
    Default starts at 0 Hz and ends at Nyquist frequency.
    :param sample_rate_hz: sample rate of the signal
    :param band_order: band order
    :param start_hz: start frequency in Hz
    :param end_hz: end frequency in Hz
    :return: central frequency bins
    """
    # Default values
    if start_hz is None:
        start_hz = 0
    if end_hz is None:
        end_hz = sample_rate_hz / 2

    # Check values
    if start_hz < 0:
        print(f"Waring: start_hz ({start_hz}) is less than 0, setting to 0")
        start_hz = 0
    if end_hz > sample_rate_hz / 2:
        print(f"Waring: end_hz ({end_hz}) is greater than Nyquist frequency, setting to Nyquist frequency")
        end_hz = sample_rate_hz / 2
    if start_hz > end_hz:
        print(f"Waring: start_hz ({start_hz}) is greater than end_hz ({end_hz}), setting to 0 and Nyquist frequency")
        start_hz = 0
        end_hz = sample_rate_hz / 2
    if sample_rate_hz < 0:
        raise ValueError(f"sample_rate_hz ({sample_rate_hz}) is less than 0")
    if band_order < 0:
        raise ValueError(f"band_order ({band_order}) is less than 0")

    # Calculate central frequency bins
