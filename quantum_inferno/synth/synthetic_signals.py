"""
This module constructs synthetic signals
"""

import numpy as np
import scipy.signal as signal
from typing import Optional, Tuple, Union
from quantum_inferno import scales_dyadic


# TODO: ADD Gabor grain, see also cwt and dyadic scales

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
    :return: numpy arrays of center, start and end frequencies
    """
    (
        _,
        _,
        _,
        _,
        _,
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


def chirp_noise_16bit(
    duration_points: int = 2 ** 12,
    sample_rate_hz: float = 80.0,
    noise_std_loss_bits: float = 4.0,
    frequency_center_hz: Optional[float] = None,
) -> np.ndarray:
    """
    Construct chirp with linear frequency sweep, white noise added, anti-aliased filter applied

    :param duration_points: number of points, length of signal. Default is 2 ** 12
    :param sample_rate_hz: sample rate in Hz. Default is 80.0
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 4.0
    :param frequency_center_hz: Optional center frequency fc in Hz.  Default None
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

    return chirp_white_aa.astype(np.float16)


def sawtooth_noise_16bit(
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
    :param frequency_center_hz: Optional center frequency fc in Hz.  Default None
    :return: numpy ndarray with anti-aliased sawtooth signal with white noise
    """
    frequency_center_hz = frequency_center_hz if frequency_center_hz else 8.0 / (duration_points / sample_rate_hz)

    sig_time_s = np.arange(int(duration_points)) / sample_rate_hz
    saw_wf = signal.sawtooth((2 * np.pi * frequency_center_hz) * sig_time_s, width=0)
    saw_wf *= taper_tukey(saw_wf, 0.25)
    saw_white = saw_wf + white_noise_fbits(sig=saw_wf, std_bit_loss=noise_std_loss_bits)
    saw_white_aa = antialias_half_nyquist(saw_white)

    return saw_white_aa.astype(np.float16)


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

    :param sig: detrended input signal
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

    :param synth: array with signal data
    :param filter_order: sets decay rate
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
    :param band_order_nth: Nth order of constant Q bands
    :return: algebraic frequencies from band order
    """
    return frequency_geometric * (np.sqrt(1 + 1 / (8 * band_order_nth ** 2)))
