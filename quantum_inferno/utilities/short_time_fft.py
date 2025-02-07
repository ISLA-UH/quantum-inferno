"""
Methods for calculating frequency and time-frequency representations of signals.
Try to match all the defaults...
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal

from quantum_inferno import qi_debugger
from quantum_inferno.scales_dyadic import cycles_from_order
from quantum_inferno.utilities.calculations import get_num_points, round_value
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

# Create dictionaries for the types to avoid having to use Literal when running the functions
scaling_type = ["magnitude", "psd", None]
padding_type = ["zeros", "edge", "even", "odd"]


# todo: allow any number of points for the fft window, or just use nearest pow 2?
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
    :param scaling: scaling of the STFT (default is "magnitude", other options are "psd" and None)
    :return: ShortTimeFFT object
    """
    # checks
    if segment_length < overlap_length:
        qi_debugger.add_message(
            f"overlap length {overlap_length} must be smaller than segment length {segment_length}"
            " using half of the segment length as the overlap length"
        )
        # print(
        #     f"overlap length {overlap_length} must be smaller than segment length {segment_length}"
        #     " using half of the segment length as the overlap length"
        # )
        overlap_length = segment_length // 2

    if tukey_alpha < 0 or tukey_alpha > 1:
        qi_debugger.add_message(
            f"Warning: Tukey alpha {tukey_alpha} must be between 0 and 1, using 0.25 as the default value"
        )
        # print(f"Warning: Tukey alpha {tukey_alpha} must be between 0 and 1, using 0.25 as the default value")
        tukey_alpha = 0.25

    if scaling not in scaling_type:
        qi_debugger.add_message(
            f"Warning: scaling {scaling} must be one of {scaling_type}, using 'magnitude' as the default value"
        )
        # print(f"Warning: scaling {scaling} must be one of {scaling_type}, using 'magnitude' as the default value")
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


def get_stft_tukey_mag(
        timeseries:np.ndarray,
        sample_rate_hz: Union[float, int],
        tukey_alpha: float,
        segment_length: int,
        overlap_length: Optional[int] = None,
        scaling: str = "magnitude",
        padding: str = "zeros"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Short-Time Fourier Transform (STFT) of a signal with a Tukey window using ShortTimeFFT class
    Returns the frequency, time bins, and magnitude of the detrended STFT similar to legacy scipy.signal.stft
    Note: If you want the STFT object, use get_stft_object_tukey

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap, if not supplied, it is half of the segment_length
    :param scaling: scaling of the STFT (default is None, other options are 'magnitude' and 'psd)
    :param padding: padding method for the STFT (default is 'zeros', other options are 'edge', 'even', and 'odd')
    :return: frequency, time bins, and magnitude of the detrended STFT
    """
    # check if padding is valid
    if padding not in padding_type:
        qi_debugger.add_message(
            f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value"
        )
        # print(f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value")
        padding = "zeros"

    # todo: check segment length and overlap length compared to timeseries length?
    if overlap_length is None:
        overlap_length = int(segment_length / 2)

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
    sample_rate_hz: Union[float, int],
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
    :param scaling: scaling of the STFT (default is None, other options are 'magnitude' and 'psd')
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
    sample_rate_hz: Union[float, int],
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
    :param padding: padding of the signal (default is 'zeros', other options are 'edge', 'even', and 'odd')
    :return: time, frequency, and magnitude of the STFT
    """
    # check if padding is valid
    if padding not in padding_type:
        qi_debugger.add_message(
            f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value"
        )
        # print(f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value")
        padding = "zeros"

    # Make the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length, scaling)

    # Calculate the spectrogram
    spectrogram = stft_obj.spectrogram(x=timeseries, padding=padding)

    # calculate the time and frequency bins
    time_bins = np.arange(start=0, stop=stft_obj.delta_t * np.shape(spectrogram)[1], step=stft_obj.delta_t)
    frequency_bins = stft_obj.f

    return frequency_bins, time_bins, spectrogram


def stft_from_sig(
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        band_order_nth: float,
        center_frequency_hz: Optional[float] = None,
        octaves_below_center: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stft from signal

    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param band_order_nth: Nth order of constant Q bands
    :param center_frequency_hz: optional center frequency of the signal in Hz.  Default 3/20 of Nyquist
    :param octaves_below_center: number of octaves below center frequency to set the averaging frequency.  Default 4
    :return: numpy arrays of: STFT, STFT_bits, time_stft_s, frequency_stft_hz
    """
    if center_frequency_hz is None:
        center_frequency_hz = frequency_sample_rate_hz * 0.075  # 3/20th of Nyquist
    frequency_averaging_hz = center_frequency_hz / octaves_below_center
    duration_fft_s = cycles_from_order(band_order_nth) / frequency_averaging_hz
    ave_points_ceil_log2 = get_num_points(
        sample_rate_hz=frequency_sample_rate_hz,
        duration_s=duration_fft_s,
        rounding_type="ceil",
        output_unit="log2",
    )
    time_fft_nd: int = 2 ** ave_points_ceil_log2
    if len(sig_wf) < time_fft_nd:
        raise ValueError(
            f"Signal length: {len(sig_wf)} is less than time_fft_nd: {time_fft_nd}"
        )
    stft_scaling = 2 * np.sqrt(np.pi) / time_fft_nd

    frequency_stft_hz, time_stft_s, stft_complex = get_stft_tukey_mag(
        timeseries=sig_wf,
        sample_rate_hz=frequency_sample_rate_hz,
        segment_length=time_fft_nd,
        overlap_length=int(time_fft_nd / 2),
        tukey_alpha=1.0,
    )
    stft_complex *= stft_scaling
    stft_bits = to_log2_with_epsilon(stft_complex)

    return stft_complex, stft_bits, time_stft_s, frequency_stft_hz


def stft_complex_pow2(
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        segment_points: int,
        overlap_points: int = None,
        alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplest, with 50% overlap and built-in defaults.  Uses nfft length of nearest power of two of segment_points.

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param segment_points: number of points in a segment
    :param overlap_points: number of points in overlap, if not given, equal to half the segment_points
    :param alpha: Tukey window alpha
    :return: frequency_stft_hz, time_stft_s, stft_complex
    """
    if overlap_points is None:
        overlap_points = int(segment_points / 2)
    return get_stft_tukey_mag(sig_wf, frequency_sample_rate_hz, alpha, segment_points, overlap_points)


def gtx_complex_pow2(
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        segment_points: int,
        gaussian_sigma: int = None,
        overlap_points: int = None,
        fft_points: int = None,
        padding: str = "zeros",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian taper with 50% overlap and built-in defaults.  Uses nfft length of nearest power of two for segment_points.

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param segment_points: number of points in a segment
    :param gaussian_sigma: gaussian window variance.  Default 1/4 of segment_points
    :param overlap_points: number of points in overlap.  Default half of segment_points
    :param fft_points: number of points in fft.  Default nearest greatest power of 2 of segment_points
    :param padding: padding method for the STFT (default is 'zeros', other options are 'edge', 'even', and 'odd')
    :return: frequency_stft_hz, time_stft_s, stft_complex
    """
    # check if padding is valid
    if padding not in padding_type:
        qi_debugger.add_message(
            f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value"
        )
        # print(f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value")
        padding = "zeros"
    if overlap_points is None:
        overlap_points = int(segment_points / 2)
    if gaussian_sigma is None:
        gaussian_sigma = int(segment_points / 4)
    # calculate the values to be used in the ShortTimeFFT object
    gaussian_window = signal.windows.gaussian(segment_points, std=gaussian_sigma)
    if fft_points is None:
        fft_points = round_value(segment_points, "ceil_power_of_two")
    hop_length = segment_points - overlap_points

    # create the ShortTimeFFT object
    stft_obj = signal.ShortTimeFFT(
        win=gaussian_window, hop=hop_length, fs=frequency_sample_rate_hz, mfft=fft_points, fft_mode="onesided",
        scale_to="magnitude"
    )

    # calculate the STFT with detrending
    stft_magnitude = np.abs(stft_obj.stft_detrend(x=sig_wf, detr="constant", padding=padding))

    # calculate the time and frequency bins
    time_bins = np.arange(start=0, stop=stft_obj.delta_t * np.shape(stft_magnitude)[1], step=stft_obj.delta_t)
    frequency_bins = stft_obj.f

    return frequency_bins, time_bins, stft_magnitude
