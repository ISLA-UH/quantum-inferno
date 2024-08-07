""""
Standardized Stockwell transform (stx) with optimization parameters
After Moukadem et al., 2022, A new optimized Stockwell transform applied on synthetic and real non-stationary signals
Rederivation in preparation, Garces et al. 2022; last updated 9 May 2022
"""
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from quantum_inferno import scales_dyadic as scales
from typing import Tuple
from quantum_inferno.utilities.rescaling import is_power_of_two


def sig_pad_up_to_pow2(sig_wf: np.ndarray, n_fft: int, verbosity: bool = False):
    """
    Zero-pad signal to the higher 2**n points for FFT

    :param sig_wf: waveform to add zero padding
    :param n_fft: length of fft
    :param verbosity: if True, outputs additional debugging statements
    :return: waveform with zero padding, length of fft to nearest greater power of 2, number of points added
    """
    # Flatten to 2D and memorize original shape
    n_times = sig_wf.shape[-1]

    if verbosity:
        print(f"length of fft: {n_fft}, waveform shape: {n_times}")
    if n_fft < n_times:
        raise ValueError(f"n_fft cannot be smaller than signal size. Got {n_fft} < {n_times}.")

    if n_fft is None or (not is_power_of_two(n_fft) and n_times > n_fft):
        # Compute next power of 2
        n_fft = 2 ** int(np.ceil(np.log2(n_times)))

    if n_times < n_fft:
        if verbosity:
            print(
                f'The input signal is shorter ({sig_wf.shape[-1]}) than "n_fft" ({n_fft}). '
                "Applying zero padding."
            )
        zero_pad: int = n_fft - n_times
        sig_wf = np.concatenate((sig_wf, np.zeros(sig_wf.shape[:-1] + zero_pad, sig_wf.dtype)), axis=-1)
    else:
        zero_pad: int = 0

    return sig_wf, n_fft, zero_pad


# Taken from the latest libquantum version, adjusted for use with quantum_inferno
def tfr_stx_fft(
    sig_wf: np.ndarray,
    time_sample_interval: float,
    scale_order_input: float = 8.0,
    n_fft_in: int = None,
    frequency_min: float = None,
    frequency_max: float = None,
    frequency_step: float = None,
    factor_q: float = 0.0,
    power_p: float = 0.0,
    power_r: float = 1.0,
    is_geometric: bool = False,
    is_inferno: bool = False,
    scale_base_input: float = scales.Slice.G3,
    scale_ref_input: float = scales.Slice.T1S,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stockwell transform, fft implementation.
    Optimized version has more free variables in sigma_scaling; testing in progress

    :param sig_wf: input waveform. If not 2**n points, it will zero pad up
    :param time_sample_interval: sample interval, inverse of sample rate
    :param scale_order_input: fractional octave band order; 12 is the musical standard
    :param n_fft_in: requested nfft, should be greater or equal to the signal length. Method zero pads up.
    :param frequency_min: lowest stx frequency of interest
    :param frequency_max: highest stx frequency of interest
    :param frequency_step: stx frequency of interest if linearly sampled
    :param factor_q: sigma_scaling adjustment, under evaluation
    :param power_p: sigma_scaling adjustment, under evaluation
    :param power_r: sigma_scaling adjustment. under evaluation
    :param is_geometric: are frequencies geometrically spaced? If so, overrides frequency_step
    :param is_inferno: are frequencies geometrically spaced and standardized as in inferno?
    :param scale_base_input: scale base; default is base 10, base 2 octaves is scales.Slice.G2
    :param scale_ref_input: scale reference time; default is 1 s
    :return: tfr_stx, psd_stx, frequency_stx_hz, frequency_stx_fft, windows_fft
    """
    frequency_sample_rate: float = 1 / time_sample_interval
    cycles_m: float = 12.0 / 5.0 * scale_order_input
    lin_fft_decimate: float = 2.0

    # Compute the nearest higher power of two number of points for the fft
    sig_wf_pow2, n_fft_pow2, zero_pad = sig_pad_up_to_pow2(sig_wf, n_fft_in)

    # For reduction back to the near original signal after taking the ifft. Should match input sig.
    n_fft_out = n_fft_pow2 - zero_pad

    # Transformations are on zero padded signals from here onwards
    # Take FFT and concatenate. A leaner version could let the fft do the padding.
    sig_fft = fft(sig_wf_pow2)
    sig_fft_cat = np.concatenate([sig_fft, sig_fft], axis=-1)

    frequency_fft = fftfreq(n_fft_pow2, time_sample_interval)  # in units of 1/sample interval
    omega_fft = 2 * np.pi * frequency_fft / frequency_sample_rate  # scaled angular frequency

    window_longest_time = n_fft_pow2 / frequency_sample_rate
    frequency_min_nth = cycles_m / window_longest_time

    # Initialize stx frequencies
    if frequency_min is None:
        frequency_min = frequency_min_nth
    if frequency_max is None:
        frequency_max = frequency_sample_rate / 2.0

    # Computing nearest frequency later on anyway, and then using that to compute the fft.
    start_f_idx = np.abs(frequency_fft - frequency_min).argmin()
    stop_f_idx = np.abs(frequency_fft - frequency_max).argmin()
    f_start = frequency_fft[start_f_idx]
    f_stop = frequency_fft[stop_f_idx]

    # Linear scale
    if frequency_step is None:
        # Reduce the fft resolution by a factor of lin_fft_decimate
        frequency_step = (frequency_max - frequency_min) * lin_fft_decimate / len(frequency_fft)
        frequency_stx = np.arange(f_start, f_stop, frequency_step)
    else:
        frequency_stx = np.arange(f_start, f_stop, frequency_step)

    # if geometric (log) frequency
    if is_geometric is True:
        # if standardized to ISO3
        if is_inferno is True:
            (
                order_Nth,
                scale_base,
                scale_band_number,
                frequency_ref,
                frequency_center_algebraic,
                frequency_center_geometric,
                frequency_start,
                frequency_end,
            ) = scales.band_frequency_low_high(
                frequency_order_input=scale_order_input,
                frequency_low_input=f_start,
                frequency_high_input=f_stop,
                frequency_sample_rate_input=frequency_sample_rate,
                frequency_base_input=scale_base_input,
                frequency_ref_input=scale_ref_input,
            )
            frequency_stx = frequency_center_geometric
        else:
            num_octaves = np.log2(f_stop / f_start)
            num_bands = int(num_octaves * scale_order_input)
            frequency_stx = np.logspace(np.log2(f_start), np.log2(f_stop), num=num_bands, base=scale_base_input)

    # Construct shifting frequency indexes
    frequency_stx_fft = np.empty(len(frequency_stx))

    # Construct time domain and fft of window
    windows_fft = np.empty((len(frequency_stx), n_fft_pow2), dtype=np.complex128)
    # tfr_stx_pow2 = np.empty(1, n_fft_pow2, dtype=np.complex128)

    tfr_stx = np.empty((len(frequency_stx), n_fft_out), dtype=np.complex128)
    psd_stx = np.empty((len(frequency_stx), n_fft_out))

    # Minimized for loops and operations
    for isx, fsx in enumerate(frequency_stx):
        stx_index = np.abs(frequency_fft - fsx).argmin()
        frequency_stx_fft[isx] = frequency_fft[stx_index]
        # eq non-dimensional angular fft frequency
        omega_sx = 2 * np.pi * frequency_stx_fft[isx] / frequency_sample_rate
        if omega_sx == 0.0:
            windows_fft[isx] = np.ones(len(n_fft_pow2))
        else:
            # Sigma is the standard deviation of the Gaussian
            sigma_scaling = (1 + factor_q * (omega_sx ** power_p)) * (omega_sx ** (1 - power_r))
            sigma = cycles_m / omega_sx
            sigma *= sigma_scaling
            windows_fft[isx] = np.exp(-0.5 * (sigma ** 2.0) * (omega_fft ** 2.0))

        # This is the main event
        tfr_stx_pow2 = ifft(sig_fft_cat[stx_index: stx_index + n_fft_pow2] * windows_fft[isx])
        if zero_pad > 0:
            tfr_stx[isx, :] = tfr_stx_pow2[:-zero_pad:1]
        else:
            tfr_stx[isx, :] = tfr_stx_pow2
        # Power
        tfr_abs = np.abs(tfr_stx[isx, :]) ** 2
        # Add EPSILON for zero handling
        psd_stx[isx, :] = tfr_abs + scales.get_epsilon()

    return tfr_stx, psd_stx, frequency_stx, frequency_stx_fft, windows_fft


def stx_complex_any_scale_pow2(
    band_order_nth: float, sig_wf: np.ndarray, frequency_sample_rate_hz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    With some assumptions and simplifications, and with some vectorization

    :param band_order_nth: Fractional octave band - revisit
    :param sig_wf: input signal with 2^M points
    :param frequency_sample_rate_hz: sample rate in Hz
    :return: frequency_stx_hz, time_stx_s, tfr_stx
    """
    n_fft_pow2 = len(sig_wf)
    frequency_stx_hz = scales.log_frequency_hz_from_fft_points(
        frequency_sample_hz=frequency_sample_rate_hz, fft_points=n_fft_pow2, scale_order=band_order_nth
    )

    scale_points = len(frequency_stx_hz)
    # Take FFT and concatenate. A leaner version could let the fft do the padding.
    sig_fft = fft(sig_wf)
    sig_fft_cat = np.concatenate([sig_fft, sig_fft], axis=-1)

    frequency_fft = fftfreq(n_fft_pow2, 1 / frequency_sample_rate_hz)  # in units of 1/sample interval
    omega_fft = 2 * np.pi * frequency_fft / frequency_sample_rate_hz  # scaled angular frequency
    omega_stx = 2 * np.pi * frequency_stx_hz / frequency_sample_rate_hz  # non-dimensional angular stx frequency
    sigma_stx = scales.cycles_from_order(scale_order=band_order_nth) / omega_stx

    # Construct 2d matrices
    # sig_fft_cat_2d = np.tile(sig_fft_cat, (scale_points, 1))
    windows_fft_2d = np.exp(
        -0.5 * (np.tile(sigma_stx, (n_fft_pow2, 1)).T ** 2.0) * (np.tile(omega_fft, (scale_points, 1)) ** 2.0)
    )

    # Construct shifting frequency indexes
    tfr_stx = np.empty((scale_points, n_fft_pow2), dtype=np.complex128)

    # Minimal iteration
    for isx, fsx in enumerate(frequency_stx_hz):
        # This is the main event
        stx_index = np.abs(frequency_fft - fsx).argmin()
        tfr_stx[isx, :] = ifft(sig_fft_cat[stx_index: stx_index + n_fft_pow2] * windows_fft_2d[isx, :])

    return frequency_stx_hz, np.arange(n_fft_pow2) / frequency_sample_rate_hz, tfr_stx
