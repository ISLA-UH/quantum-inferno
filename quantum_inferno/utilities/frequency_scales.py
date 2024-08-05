"""
Utilities for calculating frequencies for both linear and logarithmic scales.
"""
from typing import Optional
import numpy as np


def get_linear_frequency_bins_range(
    sample_rate_hz: float, segment_length: int, start_hz: Optional[float] = None, end_hz: Optional[float] = None
) -> np.ndarray:
    """
    Get the frequency bins with given sample rate and segment length that matches output from scipy.signal.spectrogram.
    Default starts at 0 Hz and ends at Nyquist frequency.

    :param sample_rate_hz: sample rate of the signal
    :param segment_length: length of the segment
    :param start_hz: Optional start frequency in Hz.  If None, value is set to 0.  Default None
    :param end_hz: Optional end frequency in Hz.  If None, value is set to sample_rate_hz / 2.  Default None
    :return: frequency bins
    """
    # Default values
    if start_hz is None:
        start_hz = 0
    if end_hz is None:
        end_hz = sample_rate_hz / 2

    # Check values
    if start_hz < 0:
        print(f"Warning: start_hz ({start_hz}) is less than 0, setting to 0")
        start_hz = 0
    if end_hz > sample_rate_hz / 2:
        print(f"Warning: end_hz ({end_hz}) is greater than Nyquist frequency, setting to Nyquist frequency")
        end_hz = sample_rate_hz / 2
    if start_hz > end_hz:
        print(f"Warning: start_hz ({start_hz}) is greater than end_hz ({end_hz}), setting to 0 and Nyquist frequency")
        start_hz = 0
        end_hz = sample_rate_hz / 2
    if segment_length < 0:
        raise ValueError(f"segment_length ({segment_length}) is less than 0")
    if sample_rate_hz < 0:
        raise ValueError(f"sample_rate_hz ({sample_rate_hz}) is less than 0")
    if segment_length > sample_rate_hz:
        print(
            f"Warning: segment_length ({segment_length}) is greater than sample_rate_hz ({sample_rate_hz})"
            f", setting to sample_rate_hz"
        )
        segment_length = sample_rate_hz

    frequency_step = sample_rate_hz / segment_length
    full_range = np.arange(start=0, stop=sample_rate_hz / 2 + frequency_step, step=frequency_step)
    return full_range[(full_range >= start_hz) & (full_range <= end_hz)]


def get_shorttime_fft_frequency_bins(sample_rate_hz: float, segment_length: int) -> np.ndarray:
    """
    Get the frequency bins with given sample rate and segment length that matches output from ShortTimeFFT.
    Starts at 0 Hz and ends at Nyquist frequency.

    :param sample_rate_hz: sample rate of the signal
    :param segment_length: length of the segment
    :return: frequency bins
    """
    return get_linear_frequency_bins_range(sample_rate_hz, segment_length)


def get_band_numbers(
    sample_rate_hz: float,
    band_order: float,
    start_hz: float = None,
    end_hz: float = None,
    base: float = 10 ** 0.3,
    reference_frequency: float = 1,
) -> np.ndarray:
    """
    Get the band numbers with given sample rate, band order, start and end frequency.
    Default gets band order starting at 1 Hz and ends at Nyquist frequency.

    :param sample_rate_hz: sample rate of the signal
    :param band_order: band order
    :param start_hz: start frequency in Hz
    :param end_hz: end frequency in Hz
    :param base: base for the logarithmic scale (default 10 ** 0.3)
    :param reference_frequency: reference frequency for the logarithmic scale (default 1)
    :return: band numbers
    """
    # Default values
    if start_hz is None:
        start_hz = 1
    if end_hz is None:
        end_hz = sample_rate_hz / 2

    # Check values
    if sample_rate_hz < 0:
        raise ValueError(f"sample_rate_hz ({sample_rate_hz}) is less than 0")
    if band_order < 0:
        raise ValueError(f"band_order ({band_order}) is less than 0")
    if start_hz < 0:
        print(f"Warning: start_hz ({start_hz}) is less than or equal 0, setting to 1")
        start_hz = 1
    if end_hz > sample_rate_hz / 2:
        print(f"Warning: end_hz ({end_hz}) is greater than Nyquist frequency, setting to Nyquist frequency")
        end_hz = sample_rate_hz / 2
    if start_hz > end_hz:
        print(f"Warning: start_hz ({start_hz}) is greater than end_hz ({end_hz}), setting to 1 and Nyquist frequency")
        start_hz = 1
        end_hz = sample_rate_hz / 2

    # Calculate band numbers
    j_min = np.floor(band_order * np.log(start_hz / reference_frequency) / np.log(base))
    j_max = np.ceil(band_order * np.log(end_hz / reference_frequency) / np.log(base))
    return np.arange(j_min, j_max + 1)


def get_log_central_frequency_bins_range(
    sample_rate_hz: float,
    band_order: float,
    start_hz: float = None,
    end_hz: float = None,
    base: float = 10 ** 0.3,
    reference_frequency: float = 1,
) -> np.ndarray:
    """
    Get the central frequency bins with given sample rate, band order, start and end frequency.
    Default starts at 1 Hz and ends at Nyquist frequency.

    :param sample_rate_hz: sample rate of the signal
    :param band_order: band order
    :param start_hz: start frequency in Hz
    :param end_hz: end frequency in Hz
    :param base: base for the logarithmic scale (default 10 ** 0.3)
    :param reference_frequency: reference frequency for the logarithmic scale (default 1)
    :return: central frequency bins
    """
    # Get band numbers
    band_numbers = get_band_numbers(sample_rate_hz, band_order, start_hz, end_hz, base, reference_frequency)

    # Calculate central frequencies
    central_frequencies = reference_frequency * base ** (band_numbers / band_order)

    return central_frequencies


def get_log_edge_frequencies(
    sample_rate_hz: float,
    band_order: float,
    start_hz: float = None,
    end_hz: float = None,
    base: float = 10 ** 0.3,
    reference_frequency: float = 1,
) -> np.ndarray:
    """
    Get the edge frequencies with given sample rate, band order, start and end frequency.
    Default starts at 1 Hz and ends at Nyquist frequency.

    :param sample_rate_hz: sample rate of the signal
    :param band_order: band order
    :param start_hz: start frequency in Hz
    :param end_hz: end frequency in Hz
    :param base: base for the logarithmic scale (default 10 ** 0.3)
    :param reference_frequency: reference frequency for the logarithmic scale (default 1)
    :return: edge frequencies
    """
    # Get band numbers
    band_numbers = get_band_numbers(sample_rate_hz, band_order, start_hz, end_hz, base, reference_frequency)

    # Calculate edge frequencies
    edge_frequencies = reference_frequency * base ** ((band_numbers - 0.5) / band_order)
    edge_frequencies = np.append(
        edge_frequencies, reference_frequency * base ** ((band_numbers + 0.5) / band_order)[-1]
    )

    return edge_frequencies
