"""
This module contains functions to construct quantized, standardized information packets using binary metrics.
No-chirp/sweep (index_shift=0, variable removed), simplified for the base stockwell transform.
Based on Garces (2023).
"""
from typing import Tuple, Union

import numpy as np
import scipy.signal as signal

from quantum_inferno import scales_dyadic as scales


def wavelet_variance_theory(amp: float, time_s: np.ndarray, scale: float, omega: float) -> Tuple[float, float]:
    """
    Theoretical variance of a Gabor wavelet, with real and imaginary components

    :param amp: wavelet amplitude
    :param time_s: time support vector
    :param scale: wavelet scale
    :param omega: angular frequency
    :return: nominal wavelet real and imaginary variance
    """
    base_var = amp**2 / len(time_s) * 0.5 * np.sqrt(np.pi) * scale
    return base_var / (1 + np.exp(-(scale * omega)**2)), base_var / (1 - np.exp(-(scale * omega)**2))


def wavelet_amplitude(scale_atom: Union[np.ndarray, float]) -> \
        Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    :param scale_atom: atom/logon scale
    :return: amp_canonical, amp_unit_spectrum
    """
    # amp_canonical = return unit integrated power and spectral energy. Good for math, ref William et al. 1991.
    # amp_unit_spectrum = return unit peak spectrum; for practical implementation.
    # Programmers: Although tempting, do not simplify - this follows the original math and is a touchstone.
    amp_canonical = (np.pi * scale_atom ** 2) ** (-1/4)
    amp_unit_spectrum = (4 * np.pi * scale_atom ** 2) ** (-1/4) * amp_canonical
    return amp_canonical, amp_unit_spectrum


def amplitude_convert_norm_to_spect(scale_atom: Union[np.ndarray, float]) -> \
        Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    :param scale_atom: atom/logon scale
    :return: amp_canonical, amp_unit_spectrum
    """
    # amp_canonical = return unit integrated power and spectral energy. Good for math, ref William et al. 1991.
    # amp_unit_spectrum = return unit peak spectrum; for practical implementation.
    amp_canonical = (np.pi * scale_atom ** 2) ** (-1/4)
    amp_unit_spectrum = (4 * np.pi * scale_atom**2) ** (-1/4) * amp_canonical
    amp_norm2spect = amp_unit_spectrum/amp_canonical

    return amp_norm2spect


def wavelet_time(time_s: np.ndarray, offset_time_s: float, frequency_sample_rate_hz: float) -> np.ndarray:
    """
    :param time_s: array with time
    :param offset_time_s: offset time in seconds
    :param frequency_sample_rate_hz: sample rate in Hz
    :return: numpy array with scaled time-shifted time
    """
    return frequency_sample_rate_hz * (time_s - offset_time_s)


def wavelet_complex(
        band_order_nth: float,
        time_s: np.ndarray,
        offset_time_s: float,
        scale_frequency_center_hz: Union[np.ndarray, float],
        frequency_sample_rate_hz: float
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float], Union[np.ndarray, float],
           Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Quantized atom for specified band_order_Nth and arbitrary time duration.
    Returns a frequency x time dimension wavelet vector

    :param band_order_nth: Nth order of constant Q bands
    :param time_s: time in seconds, duration should be greater than or equal to M/fc
    :param offset_time_s: offset time in seconds, should be between min and max of time_s
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate on Hz
    :return: waveform_complex, time_shifted_s, scale_angular_frequency, scale, omega, amp_canonical, amp_unit_spectrum
    """
    # Center and nondimensionalize time
    xtime_shifted = wavelet_time(time_s, offset_time_s, frequency_sample_rate_hz)

    # Nondimensional chirp parameters
    scale_atom, scale_angular_frequency = \
        scales.scale_from_frequency_hz(band_order_nth, scale_frequency_center_hz, frequency_sample_rate_hz)

    if np.isscalar(scale_atom):
        # Single frequency input
        xtime = xtime_shifted
        scale = scale_atom
        omega = scale_angular_frequency
    else:
        # Convert scale, frequency and time vectors to [frequency x time] matrices
        xtime = np.tile(xtime_shifted, (len(scale_atom), 1))
        scale = np.tile(scale_atom, (len(xtime_shifted), 1)).T
        omega = np.tile(scale_angular_frequency, (len(xtime_shifted), 1)).T

    # Base wavelet with unit absolute amplitude.
    # Note centered imaginary wavelet (sine) does not reach unity because of Gaussian envelope.
    wavelet_gabor = np.exp(-0.5*(xtime/scale)**2) * np.exp(1j*omega*xtime)
    amp_canonical, amp_unit_spectrum = wavelet_amplitude(scale)

    return wavelet_gabor, xtime_shifted, scale_angular_frequency, scale, omega, amp_canonical, amp_unit_spectrum


def wavelet_centered_4cwt(
        band_order_nth: float,
        duration_points: int,
        scale_frequency_center_hz: Union[np.ndarray, float],
        frequency_sample_rate_hz: float,
        dictionary_type: str = "norm"
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float], Union[np.ndarray, float], Union[np.ndarray, float]]:
    """
    Gabor atoms for CWT computation centered on the duration of signal

    Options for dictionary_type: "norm" (Canonical unit-norm), "spect" (unit spectrum), or "unit" (unit modulus)

    :param duration_points: number of points in the signal
    :param band_order_nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate is Hz
    :param dictionary_type: determines amplification value.  Default "norm"
    :return: waveform_complex, time_shifted_s, scale, omega, amp
    """
    time_s = np.arange(duration_points) / frequency_sample_rate_hz

    wavelet_gabor, xtime_shifted, scale_angular_frequency, scale, omega, amp_canonical, amp_unit_spectrum = \
        wavelet_complex(band_order_nth, time_s, time_s[-1]/2., scale_frequency_center_hz, frequency_sample_rate_hz)

    if dictionary_type == "spect":
        amp = amp_unit_spectrum
    elif dictionary_type == "unit":
        amp = 1. if np.isscalar(scale) else np.ones(scale.shape)
    else:
        amp = amp_canonical

    return amp * wavelet_gabor, xtime_shifted / frequency_sample_rate_hz, scale, omega, amp


def cwt_complex_any_scale_pow2(
        band_order_nth: float,
        sig_wf: np.ndarray,
        frequency_sample_rate_hz: float,
        cwt_type: str = "fft",
        dictionary_type: str = "norm"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate CWT

    :param band_order_nth: Nth order of constant Q bands
    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate in Hz
    :param cwt_type: one of "fft", or "morlet2". Default is "fft"
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
    :return: frequency_cwt_hz, time_cwt_s, cwt
    """
    wavelet_points = len(sig_wf)
    time_cwt_s = np.arange(wavelet_points) / frequency_sample_rate_hz
    cycles_m = scales.cycles_from_order(scale_order=band_order_nth)

    frequency_cwt_hz = scales.log_frequency_hz_from_fft_points(
        frequency_sample_hz=frequency_sample_rate_hz,
        fft_points=len(sig_wf),
        scale_order=band_order_nth)

    cw_complex, _, _, _, amp = \
        wavelet_centered_4cwt(band_order_nth=band_order_nth,
                              duration_points=wavelet_points,
                              scale_frequency_center_hz=frequency_cwt_hz,
                              frequency_sample_rate_hz=frequency_sample_rate_hz,
                              dictionary_type=dictionary_type)

    if cwt_type == "morlet2":
        scale_atom, _ = \
            scales.scale_from_frequency_hz(scale_order=band_order_nth,
                                           frequency_sample_rate_hz=frequency_sample_rate_hz,
                                           scale_frequency_center_hz=frequency_cwt_hz)
        cwt = signal.cwt(data=sig_wf, wavelet=signal.morlet2,
                         widths=scale_atom,
                         w=cycles_m,
                         dtype=np.complex128)
        if dictionary_type == 'spect':
            # Convert to 2d matrix
            cwt *= np.tile(amplitude_convert_norm_to_spect(scale_atom=scale_atom), (wavelet_points, 1)).T

    else:
        # Convolution using the fft method
        cwt = signal.fftconvolve(np.tile(sig_wf, (len(frequency_cwt_hz), 1)),
                                 np.conj(np.fliplr(cw_complex)), mode='same', axes=-1)

    return frequency_cwt_hz, time_cwt_s, cwt
