"""
Methods for calculating frequency and time-frequency representations of signals.
Try to match all the defaults...
"""

from typing import Tuple
import numpy as np
from scipy import signal

from quantum_inferno.utilities.calculations import round_value

# Create dictionaries for the types to avoid having to use Literal when running the functions
scaling_type = ["magnitude", "psd", None]
padding_type = ["zeros", "edge", "even", "odd"]


# return the Short-Time Fourier Transform (STFT) object with default parameters
def get_stft_object_tukey(
    sample_rate_hz: float, tukey_alpha: float, segment_length: int, overlap_length: int, scaling: str = "magnitude"
) -> signal.ShortTimeFFT:
    """
    Return the Short-Time Fourier Transform (STFT) object with a Tukey window using ShortTimeFFT class
    Calculates the number of fft points based on the segment length using ceil_power_of_two rounding method

    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :param scaling: scaling of the STFT (default is magnitude, other options are 'psd' and None)
    :return: ShortTimeFFT object
    """
    # checks
    if segment_length < overlap_length:
        print(
            f"overlap length {overlap_length} must be smaller than segment length {segment_length}"
            " using half of the segment length as the overlap length"
        )
        overlap_length = segment_length // 2

    if tukey_alpha < 0 or tukey_alpha > 1:
        print(f"Warning: Tukey alpha {tukey_alpha} must be between 0 and 1, using 0.25 as the default value")
        tukey_alpha = 0.25

    if scaling not in scaling_type:
        print(f"Warning: scaling {scaling} must be one of {scaling_type}, using 'magnitude' as the default value")
        scaling = "magnitude"

    # calculate the values to be used in the ShortTimeFFT object
    tukey_window = signal.windows.tukey(segment_length, alpha=tukey_alpha)
    fft_points = round_value(segment_length, "ceil_power_of_two")
    hop_length = segment_length - overlap_length

    # create the ShortTimeFFT object
    stft_obj = signal.ShortTimeFFT(
        win=tukey_window, hop=hop_length, fs=sample_rate_hz, mfft=fft_points, fft_mode="onesided", scale_to=scaling
    )

    return stft_obj


def stft_tukey(
    timeseries: np.ndarray,
    sample_rate_hz: float or int,
    tukey_alpha: float,
    segment_length: int,
    overlap_length: int,
    scaling: str = "magnitude",
    padding: str = "zeros",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Short-Time Fourier Transform (STFT) of a signal with a Tukey window using ShortTimeFFT class
    Returns the time, frequency, and magnitude of the STFT similar to legacy scipy.signal.stft

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :param scaling: scaling of the STFT (default is None, other options are 'magnitude' and 'psd)
    :param padding: padding method for the STFT (default is 'zeros', other options are 'edge', 'even', and 'odd')
    :return: time, frequency, and magnitude of the STFT
    """
    # check if padding is valid
    if padding not in padding_type:
        print(f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value")
        padding = "zeros"

    # create the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length, scaling)

    # calculate the STFT with detrending
    stft_magnitude = stft_obj.stft_detrend(x=timeseries, detr="constant", padding=padding)

    # calculate the time and frequency bins
    time_bins = np.arange(start=0, stop=stft_obj.delta_t * np.shape(stft_magnitude)[1], step=stft_obj.delta_t)
    frequency_bins = stft_obj.f

    return frequency_bins, time_bins, stft_magnitude


# get inverse Short-Time Fourier Transform (iSTFT) with default parameters
def istft_tukey(
    stft_to_invert: np.ndarray,
    sample_rate_hz: float or int,
    tukey_alpha: float,
    segment_length: int,
    overlap_length: int,
    scaling: str = "magnitude",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the inverse Short-Time Fourier Transform (iSTFT) of a signal with a Tukey window using ShortTimeFFT class

    :param stft_to_invert: The STFT to be inverted
    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :param scaling: scaling of the STFT (default is None, other options are 'magnitude' and 'psd)
    :return: timestamps and iSTFT of the signal
    """
    # create the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length, scaling)

    # The index of the last window where only half of the window contains the signal
    last_window_index = int((np.shape(stft_to_invert)[1] - 1) * stft_obj.hop)

    # return timestamps for the iSTFT that includes the full signal
    timestamps = np.arange(start=0, stop=last_window_index / sample_rate_hz, step=1 / sample_rate_hz)

    return timestamps, stft_obj.istft(stft_to_invert, k1=last_window_index)


# get the spectrogram with default parameters
def spectrogram_tukey(
    timeseries: np.ndarray,
    sample_rate_hz: float or int,
    tukey_alpha: float,
    segment_length: int,
    overlap_length: int,
    scaling: str = "magnitude",
    padding: str = "zeros",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Spectrogram of a signal with a Tukey window using ShortTimeFFT class
    Returns the time, frequency, and spectrogram similar to legacy scipy.signal.spectrogram

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :param scaling: scaling of the spectrogram (default is 'magnitude', other options are 'psd' and None)
    :param padding: padding of the signal (default is 'zeros', other options are 'periodic' and 'constant')
    :return: time, frequency, and magnitude of the STFT
    """
    # check if padding is valid
    if padding not in padding_type:
        print(f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value")
        padding = "zeros"

    # Make the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length, scaling)

    # Calculate the spectrogram
    spectrogram = stft_obj.spectrogram(x=timeseries, padding=padding)

    # calculate the time and frequency bins
    time_bins = np.arange(start=0, stop=stft_obj.delta_t * np.shape(spectrogram)[1], step=stft_obj.delta_t)
    frequency_bins = stft_obj.f

    return frequency_bins, time_bins, spectrogram
