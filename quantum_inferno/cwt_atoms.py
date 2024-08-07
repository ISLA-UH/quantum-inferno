"""
This module contains functions to construct quantized, standardized information packets using binary metrics.
"""

import numpy as np
import scipy.signal as signal
from quantum_inferno import scales_dyadic as scales
from typing import Tuple, Union
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon


def chirp_complex(
    band_order_nth: float,
    time_s: np.ndarray,
    offset_time_s: float,
    scale_frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    index_shift: float = 0,
    scale_base: float = scales.Slice.G2,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Quantum chirp for specified band_order_Nth and arbitrary time duration
    Unscaled, to be used by both Dictionary 1 and Dictionary 2

    :param band_order_nth: Nth order of constant Q bands
    :param time_s: time in seconds, duration should be greater than or equal to M/fc
    :param offset_time_s: offset time in seconds, should be between min and max of time_s
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate on Hz
    :param index_shift: Redshift = -1, Blueshift = +1, None=0
    :param scale_base: G2 or G3
    :return: waveform_complex, time_shifted_s, normal_scaling, spectrum_scaling
    """
    xtime_shifted = chirp_time(time_s, offset_time_s, frequency_sample_rate_hz)
    time_shifted_s = xtime_shifted / frequency_sample_rate_hz

    # Fundamental chirp parameters
    cycles_m, _, gamma = chirp_mqg_from_n(band_order_nth, index_shift, scale_base)
    scale_atom = chirp_scale(cycles_m, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
    normal_scaling, spectrum_scaling = chirp_amplitude(scale_atom, gamma, index_shift)

    wavelet_gauss = np.exp(-p_complex * xtime_shifted ** 2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * cycles_m * xtime_shifted / scale_atom)

    return wavelet_gabor, time_shifted_s, normal_scaling, spectrum_scaling


def chirp_spectrum(
    frequency_hz: np.ndarray,
    offset_time_s: float,
    band_order_nth: float,
    frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    index_shift: float = 0,
    scale_base: float = scales.Slice.G2,
) -> Tuple[Union[complex, float, np.ndarray], np.ndarray]:
    """
    Spectrum of quantum wavelet for specified band_order_Nth and arbitrary time duration

    :param frequency_hz: frequency range below Nyquist
    :param offset_time_s: time of wavelet centroid
    :param band_order_nth: Nth order of constant Q bands
    :param frequency_center_hz: band center frequency in Hz
    :param frequency_sample_rate_hz: sample rate on Hz
    :param index_shift: index of shift. Default is 0.0
    :param scale_base: positive reference Base G > 1. Default is G2
    :return: Fourier transform of the Gabor atom and shifted frequency in hz
    """
    cycles_m, _, gamma = chirp_mqg_from_n(band_order_nth, index_shift, scale_base)
    scale_atom = chirp_scale(cycles_m, frequency_center_hz, frequency_sample_rate_hz)
    p_complex = chirp_p_complex(scale_atom, gamma, index_shift)

    angular_frequency_center = 2 * np.pi * frequency_center_hz / frequency_sample_rate_hz
    angular_frequency = 2 * np.pi * frequency_hz / frequency_sample_rate_hz
    offset_phase = 2 * np.pi * frequency_hz * offset_time_s
    angular_frequency_shifted = angular_frequency - angular_frequency_center
    frequency_shifted_hz = angular_frequency_shifted * frequency_sample_rate_hz / (2 * np.pi)

    spectrum_amplitude = np.sqrt(p_complex / np.abs(p_complex))
    gauss_arg = 1.0 / (4 * p_complex)
    spectrum_gauss = np.exp(-gauss_arg * (angular_frequency_shifted ** 2))
    # Phase shift from time offset
    spectrum_gabor = spectrum_amplitude * spectrum_gauss * np.exp(-1j * offset_phase)

    return spectrum_gabor, frequency_shifted_hz


def chirp_spectrum_centered(
    band_order_nth: float,
    scale_frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    index_shift: float = 0,
    scale_base: float = scales.Slice.G2,
) -> Tuple[Union[complex, float, np.ndarray], np.ndarray]:
    """
    Spectrum of quantum wavelet for specified band_order_Nth and arbitrary time duration

    :param band_order_nth: Nth order of constant Q-bands
    :param scale_frequency_center_hz: band center frequency in Hz
    :param frequency_sample_rate_hz: sample rate on Hz
    :param index_shift: index of shift. Default is 0.0
    :param scale_base: positive reference Base G > 1. Default is G2
    :return: Fourier transform of the Gabor atom and shifted frequency in hz
    """
    cycles_m, _, gamma = chirp_mqg_from_n(band_order_nth, index_shift, scale_base)
    scale_atom = chirp_scale(cycles_m, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
    angular_frequency_shifted = np.arange(-np.pi, np.pi, np.pi / 2 ** 7)
    frequency_shifted_hz = angular_frequency_shifted * frequency_sample_rate_hz / (2 * np.pi)

    spectrum_amplitude = np.sqrt(p_complex / np.abs(p_complex))
    spectrum_gauss = np.exp(-(angular_frequency_shifted ** 2) / (4 * p_complex))

    return spectrum_amplitude * spectrum_gauss, frequency_shifted_hz


def chirp_mqg_from_n(
    band_order_nth: float, index_shift: float = 0, scale_base: float = scales.Slice.G2
) -> Tuple[float, float, float]:
    """
    Compute the quality factor Q and multiplier M for a specified band order N.
    N is THE quantization parameter for the binary constant Q wavelet filters.

    :param band_order_nth: Band order, must be > 0.75 or reverts to N=3
    :param index_shift: index of shift. Default is 0.
    :param scale_base: positive reference Base G > 1. Default is G2
    :return: cycles M, quality factor Q, gamma
    """
    if band_order_nth < 0.7:
        band_order_nth = 3.0
        print(f"N < 0.7 specified, using N = {band_order_nth}")
    order_bandedge = scale_base ** (1.0 / 2.0 / band_order_nth)  # kN in Garces 2013
    order_scaled_bandwidth = order_bandedge - 1.0 / order_bandedge
    quality_factor_q = 1.0 / order_scaled_bandwidth  # Exact for Nth octave bands
    # Gamma is M/(2Q)
    gamma = np.sqrt(np.log(2)) * (1 - np.log(2) * (index_shift / np.pi) ** 2) ** (-0.5)
    cycles_m = 2 * quality_factor_q * gamma  # Exact, from 1/2 power points

    return cycles_m, quality_factor_q, gamma


def chirp_scale(
    cycles_m: float, scale_frequency_center_hz: Union[np.ndarray, float], frequency_sample_rate_hz: float
) -> float:
    """
    Non-dimensional scale for canonical Morlet wavelet

    :param cycles_m: number of cycles per band period
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :return: scale atom
    """
    return cycles_m * frequency_sample_rate_hz / scale_frequency_center_hz / (2.0 * np.pi)


def chirp_scale_from_order(
    band_order_nth: float,
    scale_frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    index_shift: float = 0,
    scale_base: float = scales.Slice.G2,
) -> float:
    """
    Non-dimensional scale for canonical Morlet wavelet

    :param band_order_nth: Band order, must be > 0.75 or reverts to N=3
    :param scale_frequency_center_hz: scale frequency in hz
    :param frequency_sample_rate_hz: sample rate in hz
    :param index_shift: index fo shift. Default is 0.0
    :param scale_base: positive reference Base G > 1. Default is G2
    :return: scale atom
    """
    cycles_m, _, _ = chirp_mqg_from_n(band_order_nth, index_shift, scale_base)
    return chirp_scale(cycles_m, frequency_sample_rate_hz, scale_frequency_center_hz)


def chirp_uncertainty(
    scale_atom: float, frequency_sample_rate_hz: float, gamma: float, index_shift: float
) -> Tuple[float, float, float]:
    """
    Calculate the uncertainty of chirp

    :param scale_atom: from chirp_scale or chirp_scale_from_order
    :param frequency_sample_rate_hz: sample rate in hz
    :param gamma: from index_shift, M/(2Q)
    :param index_shift: index of shift
    :return: time std in seconds, frequency std in Hz, angular frequency std in Hz
    """
    time_std_s = scale_atom / np.sqrt(2) / frequency_sample_rate_hz
    angular_frequency_std = np.sqrt(1 + (index_shift * gamma) ** 2) / scale_atom / np.sqrt(2)
    angular_frequency_std_hz = frequency_sample_rate_hz * angular_frequency_std
    frequency_std_hz = angular_frequency_std_hz / 2 / np.pi

    return time_std_s, frequency_std_hz, angular_frequency_std_hz


def chirp_p_complex(scale_atom: float, gamma: float, index_shift: float) -> complex:
    """
    Fundamental chirp variable

    :param scale_atom: from chirp_scale or chirp_scale_from_order
    :param gamma: from index_shift, M/(2Q)
    :param index_shift: index of shift
    :return: p_complex
    """
    return (1 - 1j * index_shift * gamma / np.pi) / (2 * scale_atom ** 2)


def chirp_amplitude(scale_atom: float, gamma: float, index_shift: float) -> Tuple[float, float]:
    """
    Find chirp amplitude

    :param scale_atom: from chirp_scale or chirp_scale_from_order
    :param gamma: from index_shift, M/(2Q)
    :param index_shift: index of shift
    :return: normal_scaling, spectrum_scaling
    """
    p_complex = chirp_p_complex(scale_atom, gamma, index_shift)
    normal_scaling = 1 / np.pi ** 0.25 * 1 / np.sqrt(scale_atom)
    spectrum_scaling = np.sqrt(np.abs(p_complex) / np.pi)
    return normal_scaling, spectrum_scaling


def chirp_time(time_s: np.ndarray, offset_time_s: float, frequency_sample_rate_hz: float) -> np.ndarray:
    """
    Scaled time-shifted time

    :param time_s: array with time
    :param offset_time_s: offset time in seconds
    :param frequency_sample_rate_hz: sample rate in Hz
    :return: numpy array with time-shifted time
    """
    return frequency_sample_rate_hz * (time_s - offset_time_s)


def chirp_scales_from_duration(
    band_order_nth: float, sig_duration_s: float, index_shift: float = 0.0, scale_base: float = scales.Slice.G2
) -> Tuple[float, float]:
    """
    Calculate scale factor for time and frequency from chirp duration

    :param band_order_nth: Band order
    :param sig_duration_s: signal duration in seconds
    :param index_shift: index fo shift. Default is 0.0
    :param scale_base: positive reference Base G > 1. Default is G2
    :return: time in seconds and frequency in Hz scale factors
    """
    cycles_m, _, _ = chirp_mqg_from_n(band_order_nth, index_shift, scale_base)
    scale_time_s = sig_duration_s / cycles_m
    scale_frequency_hz = 1 / scale_time_s
    return scale_time_s, scale_frequency_hz


def chirp_frequency_bands(
    scale_order_input: float,
    frequency_low_input: float,
    frequency_sample_rate_input: float,
    frequency_high_input: float,
    index_shift: float = 0,
    frequency_ref: float = scales.Slice.F1HZ,
    scale_base: float = scales.Slice.G2,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate frequency bands for chirp

    :param scale_order_input: Nth order specification
    :param frequency_low_input: lowest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :param frequency_high_input: highest frequency of interest
    :param index_shift: index of shift
    :param frequency_ref: reference frequency
    :param scale_base: positive reference Base G > 1. Default is G2
    :return: Nth order, cycles M, quality factor Q, gamma, geometric center of frequencies, start frequency,
        end frequency
    """
    (
        order_Nth,
        scale_base,
        _,
        frequency_ref,
        _,
        frequency_center_geometric,
        frequency_start,
        frequency_end,
    ) = scales.band_frequency_low_high(
        frequency_order_input=scale_order_input,
        frequency_base_input=scale_base,
        frequency_ref_input=frequency_ref,
        frequency_low_input=frequency_low_input,
        frequency_high_input=frequency_high_input,
        frequency_sample_rate_input=frequency_sample_rate_input,
    )
    cycles_m, quality_q, gamma = chirp_mqg_from_n(order_Nth, index_shift, scale_base)

    return order_Nth, cycles_m, quality_q, gamma, frequency_center_geometric, frequency_start, frequency_end


def chirp_centered_4cwt(
    band_order_nth: float,
    sig_or_time: np.ndarray,
    scale_frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    index_shift: float = 0,
    scale_base: float = scales.Slice.G2,
    dictionary_type: str = "norm",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gabor atoms for CWT computation centered on the duration of signal

    :param sig_or_time: time or time series, wavelet matches this duration
    :param band_order_nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate is Hz
    :param index_shift: index of shift
    :param scale_base: G2 or G3
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect")
    :return: waveform_complex, time_shifted_s
    """
    duration_points = len(sig_or_time)
    time_s = np.arange(duration_points) / frequency_sample_rate_hz
    offset_time_s = time_s[-1] / 2.0

    wavelet_gabor, time_centered_s, normal_scaling, spectrum_scaling = chirp_complex(
        band_order_nth,
        time_s,
        offset_time_s,
        scale_frequency_center_hz,
        frequency_sample_rate_hz,
        index_shift,
        scale_base,
    )

    wavelet_chirp = (normal_scaling if dictionary_type == "norm" else spectrum_scaling) * wavelet_gabor

    return wavelet_chirp, time_centered_s


def cwt_chirp_complex(
    band_order_nth: float,
    sig_wf: np.ndarray,
    frequency_low_hz: float,
    frequency_sample_rate_hz: float,
    frequency_high_hz: float = scales.Slice.F0HZ,
    cwt_type: str = "fft",
    index_shift: float = 0,
    frequency_ref: float = scales.Slice.F1HZ,
    scale_base: float = scales.Slice.G2,
    dictionary_type: str = "norm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate CWT for chirp

    :param band_order_nth: Nth order of constant Q bands
    :param sig_wf: array with input signal
    :param frequency_low_hz: lowest frequency in Hz
    :param frequency_sample_rate_hz: sample rate in Hz
    :param frequency_high_hz: highest frequency in Hz
    :param cwt_type: one of "conv", "fft", or "morlet2". Default is "fft"
           Address ghost folding in "fft", compared to "conv"
    :param index_shift: index of shift. Default is 0.0
    :param frequency_ref: reference frequency in Hz. Default is F1HZ
    :param scale_base: G2 or G3. Default is G2
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
    :return: cwt, cwt_bits, time_s, frequency_cwt_hz
    """
    wavelet_points = len(sig_wf)
    time_s = np.arange(wavelet_points) / frequency_sample_rate_hz

    if cwt_type == "morlet2":
        index_shift = 0

    # Planck frequency is absolute upper limit
    if frequency_high_hz > frequency_sample_rate_hz / 2.0:
        frequency_high_hz = frequency_sample_rate_hz / 2.0

    (
        order_Nth,
        cycles_M,
        _,
        _,
        frequency_cwt_hz_flipped,
        frequency_start_flipped,
        frequency_end_flipped,
    ) = chirp_frequency_bands(
        scale_order_input=band_order_nth,
        frequency_low_input=frequency_low_hz,
        frequency_sample_rate_input=frequency_sample_rate_hz,
        frequency_high_input=frequency_high_hz,
        index_shift=index_shift,
        frequency_ref=frequency_ref,
        scale_base=scale_base,
    )

    scale_points = len(frequency_cwt_hz_flipped)

    if cwt_type == "morlet2":
        scale_atom = chirp_scale(cycles_M, frequency_cwt_hz_flipped, frequency_sample_rate_hz)
        cwt_flipped = signal.cwt(
            data=sig_wf, wavelet=signal.morlet2, widths=scale_atom, w=cycles_M, dtype=np.complex128
        )
    elif cwt_type == "fft":
        sig_fft = np.fft.fft(sig_wf)
        cwt_flipped = np.empty((scale_points, wavelet_points), dtype=np.complex128)
        for ii in range(scale_points):
            atom, _ = chirp_centered_4cwt(
                band_order_nth=order_Nth,
                sig_or_time=sig_wf,
                scale_frequency_center_hz=frequency_cwt_hz_flipped[ii],
                frequency_sample_rate_hz=frequency_sample_rate_hz,
                index_shift=index_shift,
                scale_base=scale_base,
                dictionary_type=dictionary_type,
            )
            atom_fft = np.fft.fft(atom)
            cwt_raw = np.fft.ifft(sig_fft * np.conj(atom_fft))
            cwt_flipped[ii, :] = np.append(cwt_raw[wavelet_points // 2:], cwt_raw[0: wavelet_points // 2])

    elif cwt_type == "conv":
        cwt_flipped = np.empty((scale_points, wavelet_points), dtype=np.complex128)
        for ii in range(scale_points):
            atom, _ = chirp_centered_4cwt(
                band_order_nth=order_Nth,
                sig_or_time=sig_wf,
                scale_frequency_center_hz=frequency_cwt_hz_flipped[ii],
                frequency_sample_rate_hz=frequency_sample_rate_hz,
                index_shift=index_shift,
                scale_base=scale_base,
                dictionary_type=dictionary_type,
            )
            cwt_flipped[ii, :] = signal.convolve(sig_wf, np.conj(atom)[::-1], mode="same")
    else:
        raise ValueError(f"Incorrect cwt_type: {cwt_type} specified in cwt_chirp_complex")

    # Time scales are increasing, which is the opposite of what is expected for the frequency. Flip.
    frequency_cwt_hz = np.flip(frequency_cwt_hz_flipped)
    cwt = np.flipud(cwt_flipped)
    cwt_bits = to_log2_with_epsilon(cwt)

    return cwt, cwt_bits, time_s, frequency_cwt_hz


def cwt_chirp_from_sig(
    sig_wf: np.ndarray,
    frequency_sample_rate_hz: float,
    band_order_nth: float = 3,
    cwt_type: str = "fft",
    index_shift: float = 0,
    frequency_ref: float = scales.Slice.F1HZ,
    scale_base: float = scales.Slice.G2,
    dictionary_type: str = "norm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate CWT for chirp

    :param sig_wf: array with input signal
    :param frequency_sample_rate_hz: sample rate in Hz
    :param band_order_nth: Nth order of constant Q bands
    :param cwt_type: one of "conv", "fft", or "morlet2". Default is "fft"
    :param index_shift: index of shift. Default is 0.0
    :param frequency_ref: reference frequency in Hz. Default is F1HZ
    :param scale_base: G2 or G3. Default is G2
    :param dictionary_type: Canonical unit-norm ("norm") or unit spectrum ("spect"). Default is "norm"
    :return: cwt, cwt_bits, time_s, frequency_cwt_hz
    """
    duration_s = len(sig_wf) / frequency_sample_rate_hz
    _, min_frequency_hz = chirp_scales_from_duration(
        band_order_nth=band_order_nth, sig_duration_s=duration_s, index_shift=index_shift, scale_base=scale_base
    )

    return cwt_chirp_complex(
        band_order_nth=band_order_nth,
        sig_wf=sig_wf,
        frequency_low_hz=min_frequency_hz,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        frequency_high_hz=frequency_sample_rate_hz / 2.0,
        cwt_type=cwt_type,
        index_shift=index_shift,
        frequency_ref=frequency_ref,
        scale_base=scale_base,
        dictionary_type=dictionary_type,
    )
