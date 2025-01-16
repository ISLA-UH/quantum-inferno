"""
This module calculates spectra: STFT, FFT
"""
from typing import Tuple, Union

import numpy as np
import scipy.signal as signal

from quantum_inferno.scales_dyadic import cycles_from_order
from quantum_inferno.utilities.calculations import get_num_points
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon


def stft_from_sig(
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        band_order_nth: float,
        center_frequency_hz: float = None,
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

    frequency_stft_hz, time_stft_s, stft_complex = stft_complex_pow2(
        sig_wf=sig_wf,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        segment_points=time_fft_nd,
        alpha=1.0,
    )
    stft_complex *= stft_scaling
    stft_bits = to_log2_with_epsilon(stft_complex)

    return stft_complex, stft_bits, time_stft_s, frequency_stft_hz


def butter_bandpass(
    sig_wf: np.ndarray,
    frequency_sample_rate_hz: float,
    frequency_cut_low_hz,
    frequency_cut_high_hz,
    filter_order: int = 4,
    tukey_alpha: float = 0.5,
) -> np.ndarray:
    """
    Butterworth bandpass filter

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param frequency_cut_low_hz: frequency low value
    :param frequency_cut_high_hz: frequency high value
    :param filter_order: filter order
    :param tukey_alpha: Tukey window alpha
    :return: filtered signal waveform as numpy array
    """
    nyquist = 0.5 * frequency_sample_rate_hz
    edge_low = frequency_cut_low_hz / nyquist
    edge_high = frequency_cut_high_hz / nyquist
    if edge_high >= 1:
        print(
            f"Warning: Frequency cutoff {frequency_cut_high_hz} greater than Nyquist {nyquist} Hz, using half Nyquist"
        )
        edge_high = 0.5  # Half of nyquist
    [b, a] = signal.butter(N=filter_order, Wn=[edge_low, edge_high], btype="bandpass")
    sig_taper = np.copy(sig_wf)
    sig_taper *= signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
    return signal.filtfilt(b, a, sig_taper)


def butter_highpass(
    sig_wf: np.ndarray,
    frequency_sample_rate_hz: float,
    frequency_cut_low_hz: Union[float, int],
    filter_order: int = 4,
    tukey_alpha: float = 0.5,
) -> np.ndarray:
    """
    Butterworth bandpass filter

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param frequency_cut_low_hz: frequency low value
    :param filter_order: filter order
    :param tukey_alpha: Tukey window alpha
    :return: filtered signal waveform as numpy array
    """
    edge_low = frequency_cut_low_hz / (0.5 * frequency_sample_rate_hz)

    if edge_low >= 1:
        raise ValueError(
            f"Frequency cutoff {frequency_cut_low_hz} is greater than Nyquist {0.5*frequency_sample_rate_hz}"
        )

    [b, a] = signal.butter(N=filter_order, Wn=[edge_low], btype="highpass")
    sig_taper = np.copy(sig_wf)
    sig_taper *= signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
    return signal.filtfilt(b, a, sig_taper)


def butter_lowpass(
    sig_wf: np.ndarray,
    frequency_sample_rate_hz: float,
    frequency_cut_high_hz: Union[float, int],
    filter_order: int = 4,
    tukey_alpha: float = 0.5,
) -> np.ndarray:
    """
    Butterworth bandpass filter

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param frequency_cut_high_hz: frequency low value
    :param filter_order: filter order
    :param tukey_alpha: Tukey window alpha
    :return: filtered signal waveform as numpy array
    """
    edge_high = frequency_cut_high_hz / (0.5 * frequency_sample_rate_hz)

    if edge_high >= 1:
        raise ValueError(
            f"Frequency cutoff {frequency_cut_high_hz} is greater than Nyquist {0.5*frequency_sample_rate_hz}"
        )
    [b, a] = signal.butter(N=filter_order, Wn=[edge_high], btype="lowpass")
    sig_taper = np.copy(sig_wf)
    sig_taper *= signal.windows.tukey(M=len(sig_taper), alpha=tukey_alpha)
    return signal.filtfilt(b, a, sig_taper)


def stft_complex_pow2(
    sig_wf: np.ndarray,
    frequency_sample_rate_hz: float,
    segment_points: int,
    overlap_points: int = None,
    nfft_points: int = None,
    alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplest, with 50% overlap and built-in defaults

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param segment_points: number of points in a segment
    :param overlap_points: number of points in overlap
    :param nfft_points: length of FFT
    :param alpha: Tukey window alpha
    :return: frequency_stft_hz, time_stft_s, stft_complex
    """
    if nfft_points is None:
        nfft_points = int(2 ** np.ceil(np.log2(segment_points)))
    if overlap_points is None:
        overlap_points = int(segment_points / 2)
    return signal.stft(
        x=sig_wf,
        fs=frequency_sample_rate_hz,
        window=("tukey", alpha),
        nperseg=segment_points,
        noverlap=overlap_points,
        nfft=nfft_points,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        boundary="zeros",
        padded=True,
    )


def gtx_complex_pow2(
    sig_wf: np.ndarray,
    frequency_sample_rate_hz: float,
    segment_points: int,
    gaussian_sigma: int = None,
    overlap_points: int = None,
    nfft_points: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian taper with 50% overlap and built-in defaults

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param segment_points: number of points in a segment
    :param gaussian_sigma: gaussian window variance
    :param overlap_points: number of points in overlap
    :param nfft_points: length of FFT
    :return: frequency_stft_hz, time_stft_s, stft_complex
    """
    if nfft_points is None:
        nfft_points = int(2 ** np.ceil(np.log2(segment_points)))
    if overlap_points is None:
        overlap_points = int(segment_points / 2)
    if gaussian_sigma is None:
        gaussian_sigma = int(segment_points / 4)
    return signal.stft(
        x=sig_wf,
        fs=frequency_sample_rate_hz,
        window=("gaussian", gaussian_sigma),
        nperseg=segment_points,
        noverlap=overlap_points,
        nfft=nfft_points,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        boundary="zeros",
        padded=True,
    )


def welch_power_pow2(
    sig_wf: np.ndarray,
    frequency_sample_rate_hz: float,
    segment_points: int,
    nfft_points: int = None,
    overlap_points: int = None,
    alpha: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplest, with 50% overlap and built-in defaults

    :param sig_wf: signal waveform as numpy array
    :param frequency_sample_rate_hz: frequency sample rate in Hz
    :param segment_points: number of points in a segment
    :param overlap_points: number of points in overlap
    :param nfft_points: length of FFT
    :param alpha: Tukey window alpha
    :return: frequency_welch_hz, welch_power
    """
    if nfft_points is None:
        nfft_points = int(2 ** np.ceil(np.log2(segment_points)))
    if overlap_points is None:
        overlap_points = int(segment_points / 2)
    # Compute the Welch PSD; averaged spectrum over sliding windows
    return signal.welch(
        x=sig_wf,
        fs=frequency_sample_rate_hz,
        window=("tukey", alpha),
        nperseg=segment_points,
        noverlap=overlap_points,
        nfft=nfft_points,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        scaling="spectrum",
        average="mean",
    )
