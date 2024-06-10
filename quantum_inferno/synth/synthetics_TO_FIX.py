"""
This module constructs synthetic signals
"""

import numpy as np
import scipy.signal as signal
from typing import Optional, Tuple, Union

# from quantum_inferno import scales_dyadic, utils, atoms_FOR_CWT
from quantum_inferno import scales_dyadic, utils, atoms_FOR_CWT


def gabor_loose_grain(
    band_order_nth: float,
    number_points: int,
    scale_frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    index_shift: float = 0,
    frequency_base_input: float = scales_dyadic.Slice.G2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loose grain with tight Tukey wrap to ensure zero at edges

    :param band_order_nth: Nth order of constant Q bands
    :param number_points: Number of points in the signal
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param index_shift: index of shift for the Gabor chirp, default of zero
    :param frequency_base_input: G2 or G3. Default is G2
    :return: numpy array with Tukey grain
    """
    # Fundamental chirp parameters
    cycles_m, quality_factor_q, gamma = atoms_FOR_CWT.chirp_MQG_from_N(
        band_order_nth, index_shift, frequency_base_input
    )
    scale_atom = atoms_FOR_CWT.chirp_scale(cycles_m, scale_frequency_center_hz, frequency_sample_rate_hz)

    # # Time from nominal duration
    # grain_duration_s = cycles_M/scale_frequency_center_hz
    # Time from number of points
    time_s = np.arange(number_points) / frequency_sample_rate_hz

    xtime_shifted = atoms_FOR_CWT.chirp_time(time_s, np.max(time_s) / 2.0, frequency_sample_rate_hz)
    wavelet_gauss = np.exp(-atoms_FOR_CWT.chirp_p_complex(scale_atom, gamma, index_shift) * xtime_shifted ** 2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * cycles_m * xtime_shifted / scale_atom)

    return np.copy(wavelet_gabor) * utils.taper_tukey(wavelet_gabor, 0.1), time_s, scale_atom


def gabor_tight_grain(
    band_order_nth: float,
    scale_frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    index_shift: float = 0,
    frequency_base_input: float = scales_dyadic.Slice.G2,
) -> np.ndarray:
    """
    Gabor grain with tight Tukey wrap to ensure zero at edges

    :param band_order_nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param index_shift: index of shift
    :param frequency_base_input: G2 or G3. Default is G2
    :return: numpy array with Tukey grain
    """

    # Fundamental chirp parameters
    cycles_m, quality_factor_q, gamma = atoms_FOR_CWT.chirp_MQG_from_N(
        band_order_nth, index_shift, frequency_base_input
    )
    scale_atom = atoms_FOR_CWT.chirp_scale(cycles_m, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = atoms_FOR_CWT.chirp_p_complex(scale_atom, gamma, index_shift)

    # Time from nominal duration
    grain_duration_s = cycles_m / scale_frequency_center_hz
    time_s = np.arange(int(np.round(grain_duration_s * frequency_sample_rate_hz))) / frequency_sample_rate_hz

    xtime_shifted = atoms_FOR_CWT.chirp_time(time_s, np.max(time_s) / 2.0, frequency_sample_rate_hz)
    wavelet_gabor = np.exp(-p_complex * xtime_shifted ** 2) * np.exp(1j * cycles_m * xtime_shifted / scale_atom)

    return np.copy(wavelet_gabor) * utils.taper_tukey(wavelet_gabor, 0.1)


def tukey_tight_grain(
    band_order_nth: float,
    scale_frequency_center_hz: float,
    frequency_sample_rate_hz: float,
    fraction_cosine: float = 0.5,
    index_shift: float = 0,
    frequency_base_input: float = scales_dyadic.Slice.G2,
) -> np.ndarray:
    """
    Tukey grain with same support as Gabor atom

    :param band_order_nth: Nth order of constant Q bands
    :param scale_frequency_center_hz: center frequency fc in Hz
    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param fraction_cosine: fraction of the window inside the cosine tapered window,
        shared between the head and tail.  Default 0.5
    :param index_shift: index of shift.  Default 0
    :param frequency_base_input: G2 or G3. Default G2
    :return: numpy array with Tukey grain
    """

    # Fundamental chirp parameters
    cycles_m, quality_factor_q, gamma = atoms_FOR_CWT.chirp_MQG_from_N(
        band_order_nth, index_shift, frequency_base_input
    )
    scale_atom = atoms_FOR_CWT.chirp_scale(cycles_m, scale_frequency_center_hz, frequency_sample_rate_hz)
    p_complex = atoms_FOR_CWT.chirp_p_complex(scale_atom, gamma, index_shift)

    # Time from nominal duration
    grain_duration_s = cycles_m / scale_frequency_center_hz
    time_s = np.arange(int(np.round(grain_duration_s * frequency_sample_rate_hz))) / frequency_sample_rate_hz

    xtime_shifted = atoms_FOR_CWT.chirp_time(time_s, np.max(time_s) / 2.0, frequency_sample_rate_hz)
    # Pull out phase component from gaussian envelope
    wavelet_gabor = np.exp(1j * cycles_m * xtime_shifted / scale_atom + 1j * np.imag(-p_complex * xtime_shifted ** 2))

    return np.copy(wavelet_gabor) * utils.taper_tukey(wavelet_gabor, fraction_cosine)


def gabor_grain_frequencies(
    frequency_order_input: float,
    frequency_low_input: float,
    frequency_high_input: float,
    frequency_sample_rate_input: float,
    frequency_base_input: float = scales_dyadic.Slice.G2,
    frequency_ref_input: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frequencies for g-chirps

    :param frequency_order_input: Nth order
    :param frequency_low_input: lowest frequency of interest
    :param frequency_high_input: highest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :param frequency_base_input: G2 or G3. Default is G2
    :param frequency_ref_input: reference frequency. Default is 1.0
    :return: three numpy arrays with center frequency, start frequency and end frequency
    """
    (
        scale_order,
        scale_base,
        _,
        frequency_ref,
        frequency_center_algebraic,
        frequency_center,
        frequency_start,
        frequency_end,
    ) = scales_dyadic.band_frequency_low_high(
        frequency_order_input,
        frequency_base_input,
        frequency_ref_input,
        frequency_low_input,
        frequency_high_input,
        frequency_sample_rate_input,
    )

    return frequency_center, frequency_start, frequency_end


def chirp_rdvxm_noise_16bit(
    duration_points: int = 2 ** 12,
    sample_rate_hz: float = 80.0,
    noise_std_loss_bits: float = 4.0,
    frequency_center_hz: Optional[float] = None,
):
    """
    Construct chirp with linear frequency sweep, white noise added, anti-aliased filter applied

    :param duration_points: number of points, length of signal. Default is 2 ** 12
    :param sample_rate_hz: sample rate in Hz. Default is 80.0
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 4.0
    :param frequency_center_hz: center frequency fc in Hz. Optional
    :return: numpy ndarray with anti-aliased chirp with white noise
    """
    if not frequency_center_hz:
        frequency_center_hz = 8.0 / (duration_points / sample_rate_hz)
    frequency_start_hz = 0.5 * frequency_center_hz
    frequency_end_hz = sample_rate_hz / 4.0

    sig_time_s = np.arange(int(duration_points)) / sample_rate_hz
    chirp_wf = signal.chirp(
        sig_time_s, frequency_start_hz, sig_time_s[-1], frequency_end_hz, method="linear", phi=0, vertex_zero=True
    )
    chirp_wf *= taper_tukey(chirp_wf, 0.25)
    chirp_white = chirp_wf + white_noise_fbits(sig=chirp_wf, std_bit_loss=noise_std_loss_bits)
    chirp_white_aa = antialias_half_nyquist(chirp_white)
    chirp_white_aa.astype(np.float16)

    return chirp_white_aa


def sawtooth_rdvxm_noise_16bit(
    duration_points: int = 2 ** 12,
    sample_rate_hz: float = 80.0,
    noise_std_loss_bits: float = 4.0,
    frequency_center_hz: Optional[float] = None,
) -> np.ndarray:
    """
    Construct an anti-aliased sawtooth waveform with white noise

    :param duration_points: number of points, length of signal. Default is 2 ** 12
    :param sample_rate_hz: sample rate in Hz. Default is 80.0
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 4.0
    :param frequency_center_hz: center frequency fc in Hz. Optional
    :return: numpy ndarray with anti-aliased sawtooth signal with white noise
    """
    frequency_center_hz = frequency_center_hz if frequency_center_hz else 8.0 / (duration_points / sample_rate_hz)

    sig_time_s = np.arange(int(duration_points)) / sample_rate_hz
    saw_wf = signal.sawtooth((2 * np.pi * frequency_center_hz) * sig_time_s, width=0)
    saw_wf *= taper_tukey(saw_wf, 0.25)
    saw_white = saw_wf + white_noise_fbits(sig=saw_wf, std_bit_loss=noise_std_loss_bits)
    saw_white_aa = antialias_half_nyquist(saw_white)
    saw_white_aa.astype(np.float16)

    return saw_white_aa


def sawtooth_doppler_noise_16bit(phase_radians: np.ndarray, noise_std_loss_bits: float = 4.0) -> np.ndarray:
    """
    Construct an anti-aliased sawtooth waveform with white noise

    :param phase_radians: time-varying phase in radians
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 4.0
    :return: numpy ndarray with anti-aliased sawtooth signal with white noise
    """
    saw_wf = signal.sawtooth(phase_radians, width=0)
    saw_wf *= taper_tukey(saw_wf, 0.25)
    saw_white = saw_wf + white_noise_fbits(sig=saw_wf, std_bit_loss=noise_std_loss_bits)
    saw_white_aa = antialias_half_nyquist(saw_white)
    saw_white_aa.astype(np.float16)

    return saw_white_aa


def chirp_linear_in_noise(
    snr_bits: float,
    sample_rate_hz: float,
    duration_s: float,
    frequency_start_hz: float,
    frequency_end_hz: float,
    intro_s: Union[int, float],
    outro_s: Union[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct chirp with linear frequency sweep, white noise added.

    :param snr_bits: number of bits below signal standard deviation
    :param sample_rate_hz: sample rate in Hz
    :param duration_s: duration of chirp in seconds
    :param frequency_start_hz: start frequency in Hz
    :param frequency_end_hz: end frequency in Hz
    :param intro_s: number of seconds before chirp
    :param outro_s: number of seconds after chirp
    :return: numpy ndarray with waveform, numpy ndarray with time in seconds
    """
    sig_time_s = np.arange(int(sample_rate_hz * duration_s)) / sample_rate_hz
    chirp_wf = signal.chirp(
        sig_time_s, frequency_start_hz, sig_time_s[-1], frequency_end_hz, method="linear", phi=0, vertex_zero=True
    )
    chirp_wf *= taper_tukey(chirp_wf, 0.25)
    sig_wf = np.concatenate(
        (np.zeros(int(intro_s * sample_rate_hz)), chirp_wf, np.zeros(int(outro_s * sample_rate_hz)))
    )
    synth_wf = sig_wf + white_noise_fbits(sig=sig_wf, std_bit_loss=snr_bits)
    return synth_wf, np.arange(len(synth_wf)) / sample_rate_hz


def white_noise_fbits(sig: np.ndarray, std_bit_loss: float) -> np.ndarray:
    """
    Compute white noise with zero mean and standard deviation that is snr_bits below the input signal

    :param sig: input signal, detrended
    :param std_bit_loss: number of bits below signal standard deviation
    :return: gaussian noise with zero mean
    """
    # This is in power, or variance.  White noise, zero mean
    return np.random.normal(0, np.std(sig) / 2.0 ** std_bit_loss, size=sig.size)


def taper_tukey(sig_or_time: np.ndarray, fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window

    :param sig_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    return signal.windows.tukey(M=np.size(sig_or_time), alpha=fraction_cosine, sym=True)


def antialias_half_nyquist(synth: np.ndarray, filter_order: int = 4) -> np.ndarray:
    """
    Antialiasing filter with -3dB at 1/4 of sample rate, 1/2 of Nyquist

    :param filter_order: filter order sets decay rate
    :param synth: array with signal data
    :return: numpy array with anti-aliased signal
    """
    # Antialiasing filter with -3dB at 1/4 of sample rate, 1/2 of Nyquist
    # Signal frequencies are scaled by Nyquist
    edge_high = 0.5
    [b, a] = signal.butter(filter_order, edge_high, btype="lowpass")
    return signal.filtfilt(b, a, np.copy(synth))


def frequency_algebraic_nth(frequency_geometric: np.ndarray, band_order_nth: float) -> np.ndarray:
    """
    Compute algebraic frequencies in band order

    :param frequency_geometric: geometric frequencies
    :param band_order_nth:  Nth order of constant Q bands
    :return:
    """
    return frequency_geometric * (np.sqrt(1 + 1 / (8 * band_order_nth ** 2)))
