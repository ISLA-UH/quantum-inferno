"""
This module provides benchmark synthetic functions

"""
from typing import Tuple

import numpy as np
import scipy.signal as signal

from quantum_inferno.utilities.window import get_tukey
from quantum_inferno.synth import synthetic_signals

DEFAULT_TIME_SAMPLE_INTERVAL = 1e-3
DEFAULT_TIME_DURATION = 1.0
DEFAULT_OVERSAMPLE_SCALE = 2


""" Signal conditioning  """


def signal_gate(wf: np.ndarray, t: np.ndarray, tmin: float, tmax: float, fraction_cosine: float = 0) -> np.ndarray:
    """
    Time gate and apply Tukey window, rectangular is the default

    :param wf: waveform
    :param t: timestamp array
    :param tmin: lower time limit
    :param tmax: upper time limit
    :param fraction_cosine: 0 = rectangular, 1 = Hann
    :return: waveform
    """
    index_exclude = np.logical_or(t < tmin, t > tmax)
    index_include = np.logical_and(t >= tmin, t <= tmax)
    wf[index_exclude] = 0.0
    wf[index_include] *= signal.windows.tukey(M=index_include.sum(), alpha=fraction_cosine)
    return wf


def oversample_time(time_duration: float, time_sample_interval: float, oversample_scale: float) -> np.ndarray:
    """
    Return an oversampled time by a factor oversample_scale

    :param time_duration: duration of waveform
    :param time_sample_interval: length of an interval
    :param oversample_scale: oversample waveform, then decimate by scale
    :return: oversampled timestamps
    """
    oversample_interval = time_sample_interval / oversample_scale
    number_points = int(time_duration / oversample_interval)
    time_all = np.arange(number_points) * oversample_interval
    return time_all


""" Reference Signatures, starting with the quantized Gabor chirp"""


def quantum_chirp(
    omega: float,
    order: float = 12.0,
    gamma: float = 0.0,
    gauss: bool = True,
    oversample_scale: int = DEFAULT_OVERSAMPLE_SCALE,
) -> Tuple[np.ndarray, int]:
    """
    Constructs a tone or a sweep with a gaussian window option and a duration of 2^n points

    :param omega: center frequency < pi. Resets to pi/4 if >= 1 (Nyquist).
    :param order: fractional octave band, sets the Q and duration. Default of 12 for chromatic scale.
    :param gamma: sweep index, could be positive or negative. Default of zero yields the atom.
    :param gauss: Apply the Gauss envelope, True as default. False constructs constant amplitude CW or sweep
    :param oversample_scale: oversample waveform, then decimate by scale
    :return: waveform, scale factor
    """
    if omega >= 0.8 * np.pi:
        print("Omega >= 0.8*pi (AA*Nyquist), reset to pi * 2**(-1/N")
        omega = np.pi * 2 ** (-1 / order)

    # Gabor atom specifications
    scale_multiplier = 3.0 / 4.0 * np.pi * order

    # Atom scale
    scale = scale_multiplier / omega

    # Chirp index gamma, blueshift
    chirp_scale = scale * np.sqrt(1 + gamma ** 2)

    # scale multiplier Mc
    window_support_points = 2.0 * np.pi * chirp_scale
    # scale up
    window_support_pow2 = 2 ** int((np.ceil(np.log2(window_support_points))))

    # Oversample by a factor of two:
    window_support_pow2_oversample = oversample_scale * window_support_pow2

    time0 = np.arange(window_support_pow2_oversample)
    time = time0 - time0[-1] / 2

    chirp_phase = omega * time + 0.5 * gamma * (time / chirp_scale) ** 2
    if gauss:
        chirp_wf_oversample = np.exp(-0.5 * (time / chirp_scale) ** 2 + 1j * chirp_phase)
    else:
        chirp_wf_oversample = np.exp(1j * chirp_phase)

    # Downsample to anti-alias
    chirp_wf = signal.decimate(x=np.real(chirp_wf_oversample), q=oversample_scale) + 1j * signal.decimate(
        x=np.imag(chirp_wf_oversample), q=oversample_scale
    )

    return chirp_wf, window_support_pow2


def synth_00(
    frequency_0: float = 100.0,
    frequency_1: float = 200.0,
    frequency_2: float = 400.0,
    time_start_2: float = 0.25,
    time_stop_2: float = 0.4,
    time_sample_interval: float = DEFAULT_TIME_SAMPLE_INTERVAL,
    time_duration: float = DEFAULT_TIME_DURATION,
    oversample_scale: int = DEFAULT_OVERSAMPLE_SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate three sine waves, oversample and decimate to AA
    Always work with non-dimensionalized units (number of points, Nyquist, etc.)

    :param frequency_0: first frequency
    :param frequency_1: second frequency
    :param frequency_2: third frequency
    :param time_start_2: start time of third frequency
    :param time_stop_2: end time of third frequency
    :param time_sample_interval: length of an interval
    :param time_duration: length of waveform
    :param oversample_scale: oversample synthetic, then decimate by scale
    :return: synth waveform and timestamps
    """
    # Oversample, then decimate
    time_all = oversample_time(time_duration, time_sample_interval, oversample_scale)

    # Construct sine waves with unit amplitude [rms * sqrt(2)]
    sin_0 = np.sin(2.0 * np.pi * frequency_0 * time_all)
    signal_gate(wf=sin_0, t=time_all, tmin=0, tmax=0.5)
    sin_1 = np.sin(2.0 * np.pi * frequency_1 * time_all)
    signal_gate(wf=sin_1, t=time_all, tmin=0.5, tmax=1.0)
    sin_2 = np.sin(2.0 * np.pi * frequency_2 * time_all)
    signal_gate(wf=sin_2, t=time_all, tmin=time_start_2, tmax=time_stop_2)

    # Superpose gated sinusoids
    superpose = sin_0 + sin_1 + sin_2
    signal_gate(wf=superpose, t=time_all, tmin=0.0, tmax=1.0, fraction_cosine=0.05)

    # Decimate by same oversample scale, essentially an AA filter
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


def synth_01(
    a: float = 100.0,
    b: float = 20.0,
    f: float = 5.0,
    time_sample_interval: float = DEFAULT_TIME_SAMPLE_INTERVAL,
    time_duration: float = DEFAULT_TIME_DURATION,
    oversample_scale: int = DEFAULT_OVERSAMPLE_SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Example Synthetic 1

    :param a: first value
    :param b: second value
    :param f: third value
    :param time_sample_interval: length of an interval
    :param time_duration: length of waveform
    :param oversample_scale: oversample synthetic, then decimate by scale
    :return: synth waveform and timestamps
    """
    # Oversample, then decimate, essentially an AA filter
    time_all = oversample_time(time_duration, time_sample_interval, oversample_scale)
    superpose = np.cos(a * np.pi * time_all - b * np.pi * time_all * time_all) + np.cos(
        4.0 * np.pi * np.sin(np.pi * f * time_all) + np.pi * 80.0 * time_all
    )
    # Taper
    signal_gate(wf=superpose, t=time_all, tmin=0.0, tmax=1.0, fraction_cosine=0.05)
    # Decimate by same oversample scale
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


def synth_02(
    t1: float = 0.3,
    t2: float = 0.7,
    t3: float = 0.5,
    f1: float = 45.0,
    f2: float = 75.0,
    f3: float = 15.0,
    time_sample_interval: float = DEFAULT_TIME_SAMPLE_INTERVAL,
    time_duration: float = DEFAULT_TIME_DURATION,
    oversample_scale: int = DEFAULT_OVERSAMPLE_SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Example Synthetic 2

    :param t1: first time
    :param t2: second time
    :param t3: third time
    :param f1: first frequency
    :param f2: second frequency
    :param f3: third frequency
    :param time_sample_interval: length of an interval
    :param time_duration: length of waveform
    :param oversample_scale: oversample synthetic, then decimate by scale
    :return: synth waveform and timestamps
    """
    t = oversample_time(time_duration, time_sample_interval, oversample_scale)

    pulse1 = np.exp(-35.0 * np.pi * (t - t1) ** 2) * np.cos(np.pi * f1 * t)
    pulse2 = np.exp(-35.0 * np.pi * (t - t2) ** 2) * np.cos(np.pi * f1 * t)
    pulse3 = np.exp(-55.0 * np.pi * (t - t3) ** 2) * np.cos(np.pi * f2 * t)
    pulse4 = np.exp(-45.0 * np.pi * (t - t3) ** 2) * np.cos(np.pi * f3 * t)

    superpose = pulse1 + pulse2 + pulse3 + pulse4

    # Decimate by same oversample scale, essentially an AA filter
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


def synth_03(
    a: float = 30.0,
    b: float = 40.0,
    c: float = 150.0,
    time_sample_interval: float = DEFAULT_TIME_SAMPLE_INTERVAL,
    time_duration: float = DEFAULT_TIME_DURATION,
    oversample_scale: int = DEFAULT_OVERSAMPLE_SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Example Synthetic 3

    :param a: first value
    :param b: second value
    :param c: third value
    :param time_sample_interval: length of an interval
    :param time_duration: length of waveform
    :param oversample_scale: oversample synthetic, then decimate by scale
    :return: synth waveform and timestamps
    """
    # Oversample, then decimate
    time_all = oversample_time(time_duration, time_sample_interval, oversample_scale)
    superpose = np.cos(20.0 * np.pi * np.log(a * time_all + 1.0)) + np.cos(
        b * np.pi * time_all + c * np.pi * (time_all ** 2)
    )
    signal_gate(wf=superpose, t=time_all, tmin=0.0, tmax=1.0, fraction_cosine=0.05)

    # Decimate by same oversample scale, essentially an AA filter
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


""" Fancy test tone with time-domain specifications"""


def well_tempered_tone(
    frequency_sample_rate_hz: float = 800.0,
    frequency_center_hz: float = 60.0,
    time_duration_s: float = 10.24,
    time_fft_s: float = 0.64,
    use_fft_frequency: bool = True,
    add_noise_taper_aa: bool = False,
    output_desc: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, float, float, float]:
    """
    Return a tone of unit amplitude and fixed frequency with a constant sample rate

    :param frequency_sample_rate_hz: sample rate of frequency in Hz
    :param frequency_center_hz: center frequency in Hz
    :param time_duration_s: duration in seconds
    :param time_fft_s: length of segments. Default 1 second
    :param use_fft_frequency: use FFT frequency.  Default True
    :param add_noise_taper_aa: add noise taper.  Default False
    :param output_desc: if True, output description of waveform.  Default False
    :return: waveform, timestamps, fft duration, sample rate, center frequency, frequency resolution
    """
    # The segments determine the spectral resolution
    frequency_resolution_hz = 1.0 / time_fft_s

    # The FFT efficiency is based on powers of 2; it is always possible to pad with zeros.
    # Set the record duration, make a power of 2. Note that int rounds down
    time_duration_nd = 2 ** (int(np.log2(time_duration_s * frequency_sample_rate_hz)))
    # Set the fft duration, make a power of 2
    time_fft_nd = 2 ** (int(np.log2(time_fft_s * frequency_sample_rate_hz)))

    # Warn user if the time duration or fft duration is not a power of 2 and show the new values
    # The values are rounded down to the nearest power of 2 in the previous step.
    if time_duration_nd != time_duration_s * frequency_sample_rate_hz:
        print(
            f"Warning: The time duration {time_duration_s} s with given sample rate doesn't produce data points "
            f"that are power of two, adjusting time duration to {time_duration_nd} s"
        )
    if time_fft_nd != time_fft_s * frequency_sample_rate_hz:
        print(
            f"Warning: fft duration {time_fft_s} s with given sample rate doesn't produce data points "
            f"that are power of two, adjusting fft duration to {time_fft_nd} s"
        )

    # The fft frequencies are set by the duration of the fft
    # In this example we only need the positive frequencies
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1 / frequency_sample_rate_hz)
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz - frequency_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    frequency_resolution_fft_hz = frequency_sample_rate_hz / time_fft_nd

    # Dimensionless time (samples)
    time_nd = np.arange(time_duration_nd)
    time_s = time_nd / frequency_sample_rate_hz

    # Convert to dimensionless time and frequency, which is typically used in mathematical formulas.
    # Scale by the sample rate.
    if use_fft_frequency:
        f_c = frequency_center_fft_hz / frequency_sample_rate_hz
        # Construct synthetic tone with 2^n points and max FFT amplitude at exact fft frequency
        # mic_sig = np.cos(2.0 * np.pi * frequency_center_fft * time_nd)
    else:
        # Dimensionless center frequency
        f_c = frequency_center_hz / frequency_sample_rate_hz
        # Compare to synthetic tone with 2^n points and max FFT amplitude NOT at exact fft frequency
        # It does NOT return unit amplitude (but it's close)
    mic_sig = np.cos(2.0 * np.pi * f_c * time_nd)

    if add_noise_taper_aa:
        # Add noise
        mic_sig += synthetic_signals.white_noise_fbits(sig=mic_sig, std_bit_loss=8.0)
        # Taper before AA
        mic_sig *= get_tukey(array=mic_sig, alpha=0.1)
        # Antialias (AA)
        synthetic_signals.antialias_half_nyquist(mic_sig)

    if output_desc:
        print("WELL TEMPERED TONE SYNTHETIC")
        print("Nyquist frequency:", frequency_sample_rate_hz / 2)
        print("Nominal signal frequency, hz:", frequency_center_hz)
        print("FFT signal frequency, hz:", frequency_center_fft_hz)
        print("Nominal spectral resolution, hz", frequency_resolution_hz)
        print("FFT spectral resolution, hz", frequency_resolution_fft_hz)
        print("Number of signal points:", time_duration_nd)
        print("log2(points):", np.log2(time_duration_nd))
        print("Number of FFT points:", time_fft_nd)
        print("log2(FFT points):", np.log2(time_fft_nd))

    return mic_sig, time_s, time_fft_nd, frequency_sample_rate_hz, frequency_center_fft_hz, frequency_resolution_fft_hz
