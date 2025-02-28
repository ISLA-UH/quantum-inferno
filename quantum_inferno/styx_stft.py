"""
Methods for calculating frequency and time-frequency representations of signals.
Historical note: Scipy added signal.ShortTimeFFT in version 1.12.0
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal

from quantum_inferno import qi_debugger
from quantum_inferno.scales_dyadic import cycles_from_order
from quantum_inferno.utilities.calculations import get_num_points, round_value
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon
from quantum_inferno.utilities.window import taper_power_correction

# STFT window types allowed
stft_window_type = ["tukey", "gaussian"]
# Welch averaging allowed
welch_average = ["mean", "median"]
# Create dictionaries for the types to avoid having to use Literal when running the functions
scaling_type = ["magnitude", "psd", None]
padding_type = ["zeros", "edge", "even", "odd"]


def get_stft_object(
        window_type: str,
        window_args: Union[List, Tuple],
        sample_rate_hz: float,
        segment_length: int,
        overlap_length: int,
        scaling: Optional[str] = "magnitude",
        fft_points: Optional[int] = None
) -> signal.ShortTimeFFT:
    """
    Return an STFT object with the given parameters, using the specified window.  Designed for future expansion.
    Allowed window types are: "tukey", "gaussian".
    Window arguments must be given as a list or tuple.

    * Tukey window requires alpha value as the window_args.  Example: [.25]
    * Gaussian window requires sigma value as the window_args.  Example: [4]

    :param window_type: type of window to use.  Refer to list above for valid types.  Invalid types default to "tukey"
    :param window_args: arguments for the window function.  See above for details.
    :param sample_rate_hz: sample rate in hz
    :param segment_length: length of window segment
    :param overlap_length: length of overlap segment
    :param scaling: Optional scaling type of window.  Defaults to "magnitude".  Other options are "psd" and None
    :param fft_points: Optional number of points in the fft.  If None, uses nearest power of two of segment_length.
    :return: STFT object
    """
    if scaling not in scaling_type:
        qi_debugger.add_message(
            f"Warning: scaling {scaling} must be one of {scaling_type}, using 'magnitude' as the default value"
        )
        scaling = "magnitude"
    if segment_length < overlap_length:
        qi_debugger.add_message(
            f"Warning: overlap length {overlap_length} must be smaller than segment length {segment_length}"
            " using half of the segment length as the overlap length"
        )
        overlap_length = segment_length // 2

    # calculate the values to be used in the ShortTimeFFT object
    if window_type.lower() == "gaussian":
        if len(window_args) != 1 or window_args[0] is None:
            gaussian_sigma = segment_length // 4
            qi_debugger.add_message(
                f"Warning: Gaussian window requires one argument, using {gaussian_sigma} as the default value"
            )
        else:
            gaussian_sigma = window_args[0]
        window = signal.windows.gaussian(segment_length, std=gaussian_sigma)
    # this catches anything that's not "gaussian".  Uses tukey window.
    else:
        tukey_alpha = 0.25
        if len(window_args) != 1:
            qi_debugger.add_message(
                f"Warning: Tukey window requires one argument, using {tukey_alpha} as the default value"
            )
        elif window_args[0] < 0 or window_args[0] > 1 or window_args[0] is None:
            qi_debugger.add_message(
                f"Warning: Tukey alpha {window_args[0]} must be between 0 and 1, using 0.25 as the default value"
            )
        else:
            # reset tukey alpha to given value
            tukey_alpha = window_args[0]
        window = signal.windows.tukey(segment_length, alpha=tukey_alpha)

    # Compute the number of fft points
    if fft_points is None:
        fft_points = round_value(segment_length, "ceil_power_of_two")
    hop_length = segment_length - overlap_length

    # create the ShortTimeFFT object
    # noinspection PyTypeChecker
    return signal.ShortTimeFFT(
        win=window, hop=hop_length, fs=sample_rate_hz, mfft=fft_points, fft_mode="onesided", scale_to=scaling
    )


def get_stft_object_tukey(
    sample_rate_hz: float, tukey_alpha: float, segment_length: int, overlap_length: int,
        scaling: Optional[str] = "magnitude", fft_points: Optional[int] = None
) -> signal.ShortTimeFFT:
    """
    Return the Short-Time Fourier Transform (STFT) object with a Tukey window using ShortTimeFFT class.
    If fft_points not given, calculates the number of fft points based on the segment length using ceil_power_of_two
    rounding method

    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :param scaling: Optional scaling of the STFT.  Default is "magnitude", other options are "psd" and None
    :param fft_points: Optional number of points in the fft.  If None, uses nearest power of two of segment_length.
                        Default None
    :return: ShortTimeFFT object
    """
    return get_stft_object("tukey", [tukey_alpha], sample_rate_hz, segment_length, overlap_length, scaling, fft_points)


def get_stft_object_gaussian(
        sample_rate_hz: float, gaussian_sigma: float, segment_length: int, overlap_length: int,
        scaling: Optional[str] = "magnitude", fft_points: Optional[int] = None
) -> signal.ShortTimeFFT:
    """
    Return the Short-Time Fourier Transform (STFT) object with a Gaussian window using ShortTimeFFT class.
    If fft_points not given, calculates the number of fft points based on the segment length using ceil_power_of_two
    rounding method

    :param sample_rate_hz: sample rate of the signal
    :param gaussian_sigma: shape parameter of the Gaussian window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :param scaling: Optional scaling of the STFT.  Default is "magnitude", other options are "psd" and None
    :param fft_points: Optional number of points in the fft.  If None, uses nearest power of two of segment_length.
                        Default None
    :return: ShortTimeFFT object
    """
    return get_stft_object("gaussian", [gaussian_sigma], sample_rate_hz, segment_length, overlap_length, scaling,
                           fft_points)


def get_freq_time_bins(stft_obj: signal.ShortTimeFFT, stop_scalar: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param stft_obj: calculated stft object
    :param stop_scalar: length of the time bins
    :return: frequency and time bin ndarrays
    """
    time_bins = np.arange(start=0, stop=stft_obj.delta_t * stop_scalar, step=stft_obj.delta_t)
    frequency_bins = stft_obj.f

    return frequency_bins, time_bins


def get_stft_tukey(
        timeseries:np.ndarray,
        sample_rate_hz: Union[float, int],
        tukey_alpha: float,
        segment_length: int,
        overlap_length: int,
        scaling: Optional[str] = "magnitude",
        padding: str = "zeros",
        fft_points: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Short-Time Fourier Transform (STFT) of a signal with a Tukey window using ShortTimeFFT class
    Returns the frequency, time bins, and magnitude of the detrended STFT similar to legacy scipy.signal.stft
    Note: If you want the STFT object, use get_stft_object_tukey()

    :param timeseries: input signal
    :param sample_rate_hz: sample rate of the signal
    :param tukey_alpha: shape parameter of the Tukey window
    :param segment_length: length of the segment
    :param overlap_length: length of the overlap
    :param scaling: Optional scaling of the STFT.  Default is "magnitude", other options are "psd" and None
    :param padding: Padding method for the STFT.  Default is "zeros", other options are "edge", "even", and "odd"
    :param fft_points: Optional number of points in the fft.  If None, uses nearest power of two of segment_length.
                        Default None
    :return: frequency, time bins, and magnitude of the detrended STFT
    """
    # check if padding is valid
    if padding not in padding_type:
        qi_debugger.add_message(
            f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value"
        )
        padding = "zeros"

    # create the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length, scaling, fft_points)

    # TODO: test correction factor
    # Compute window correction factor
    window_correction_factor = taper_power_correction(stft_obj.win)

    # calculate the STFT with detrending
    # noinspection PyTypeChecker
    stft_magnitude = stft_obj.stft_detrend(x=timeseries, detr="constant", padding=padding)

    # calculate the time and frequency bins
    frequency_bins, time_bins = get_freq_time_bins(stft_obj, np.shape(stft_magnitude)[1])

    return frequency_bins, time_bins, stft_magnitude


# get inverse Short-Time Fourier Transform (iSTFT), must match the get_stft_tukey() input parameters
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
    :param scaling: Optional scaling of the STFT.  Default is "magnitude", other options are "psd" and None
    :return: timestamps and iSTFT of the signal
    """
    # create the ShortTimeFFT object
    stft_obj = get_stft_object_tukey(sample_rate_hz, tukey_alpha, segment_length, overlap_length, scaling)

    # The index of the last window where only half of the window contains the signal
    last_window_index = int((np.shape(stft_to_invert)[1] - 1) * stft_obj.hop)

    # return timestamps for the iSTFT that includes the full signal
    timestamps = np.arange(start=0, stop=last_window_index / sample_rate_hz, step=1 / sample_rate_hz)

    return timestamps, stft_obj.istft(stft_to_invert, k1=last_window_index)


def get_stft_gaussian(
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        gaussian_sigma: int,
        segment_points: int,
        overlap_points: int,
        padding: str = "zeros",
        fft_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the Short-Time Fourier Transform (STFT) of a signal with a Gaussian window using ShortTimeFFT class
    Returns the frequency, time bins, and magnitude of the detrended STFT similar to legacy scipy.signal.stft
    Note: If you want the STFT object, use get_stft_object_gaussian()

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param gaussian_sigma: gaussian window variance
    :param segment_points: number of points in a segment
    :param overlap_points: number of points in overlap
    :param padding: Padding method for the STFT.  Default is "zeros", other options are "edge", "even", and "odd"
    :param fft_points: Optional number of points in fft.  If None, defaults to nearest greatest power of 2 of
                        segment_points
    :return: frequency_stft_hz, time_stft_s, stft_complex
    """
    # check if padding is valid
    if padding not in padding_type:
        qi_debugger.add_message(
            f"Warning: padding {padding} must be one of {padding_type}, using 'zeros' as the default value"
        )
        padding = "zeros"

    # create the ShortTimeFFT object
    stft_obj = get_stft_object_gaussian(frequency_sample_rate_hz, gaussian_sigma, segment_points, overlap_points,
                                        "magnitude", fft_points)

    # calculate the STFT with detrending
    # noinspection PyTypeChecker
    stft_magnitude = np.abs(stft_obj.stft_detrend(x=sig_wf, detr="constant", padding=padding))

    # calculate the time and frequency bins
    frequency_bins, time_bins = get_freq_time_bins(stft_obj, np.shape(stft_magnitude)[1])

    return frequency_bins, time_bins, stft_magnitude


def stft_complex_pow2(
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        segment_points: int,
        overlap_points: int = None,
        alpha: float = 0.25,
        fft_points: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    With built-in defaults.  Uses nfft length of nearest power of two of segment_points.

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param segment_points: number of points in a segment
    :param overlap_points: number of points in overlap, if not given, equal to half the segment_points
    :param alpha: Tukey window alpha.  Default 0.25
    :param fft_points: Optional number of points in the fft.  If None, uses nearest power of two of segment_points.
                        Default None
    :return: frequency_stft_hz, time_stft_s, stft_complex
    """
    if overlap_points is None:
        overlap_points = int(segment_points / 2)
    return get_stft_tukey(sig_wf, frequency_sample_rate_hz, alpha, segment_points, overlap_points,
                          fft_points=fft_points)


def gtx_complex_pow2(
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        segment_points: int,
        overlap_points: Optional[int] = None,
        gaussian_sigma: Optional[int] = None,
        fft_points: Optional[int] = None,
        padding: str = "zeros",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian taper with built-in defaults.  Uses nfft length of nearest power of two for segment_points.

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param segment_points: number of points in a segment
    :param gaussian_sigma: gaussian window variance.  Default 1/4 of segment_points
    :param overlap_points: number of points in overlap.  Default half of segment_points
    :param fft_points: number of points in fft.  Default nearest greatest power of 2 of segment_points
    :param padding: Padding method for the STFT.  Default is "zeros", other options are "edge", "even", and "odd"
    :return: frequency_stft_hz, time_stft_s, stft_complex
    """
    if overlap_points is None:
        overlap_points = int(segment_points / 2)
    if gaussian_sigma is None:
        gaussian_sigma = int(segment_points / 4)
    return get_stft_gaussian(sig_wf, frequency_sample_rate_hz, gaussian_sigma, segment_points, overlap_points, padding,
                             fft_points)


def welch_from_stft(
        stft_complex: np.ndarray,
        average: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Welch's method of the STFT
    :param stft_complex: complex STFT array
    :param average: type of averaging.  Default is "mean", other option is "median"
    :return: welch_power
    """
    if average not in welch_average:
        qi_debugger.add_message(
            f"Warning: average {average} must be one of {welch_average}, using 'mean' as the default value"
        )
        average = "mean"

    if average == "mean":
        return np.mean(np.abs(stft_complex)**2, axis=1)
    else:
        return np.median(np.abs(stft_complex)**2, axis=1)


# TODO: Modify this function to use the ShortTimeFFT object
# it kinda does since it invokes get_stft_tukey()
def stft_from_order(
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

    frequency_stft_hz, time_stft_s, stft_complex = get_stft_tukey(
        timeseries=sig_wf,
        sample_rate_hz=frequency_sample_rate_hz,
        segment_length=time_fft_nd,
        overlap_length=int(time_fft_nd / 2),
        tukey_alpha=1.0,
    )
    stft_complex *= stft_scaling
    stft_bits = to_log2_with_epsilon(stft_complex)

    return stft_complex, stft_bits, time_stft_s, frequency_stft_hz
