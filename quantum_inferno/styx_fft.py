"""
This module calculates spectra: STFT, FFT
"""
from typing import Tuple, Union

import numpy as np
import scipy.signal as signal

from quantum_inferno import scales_dyadic as scales


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
