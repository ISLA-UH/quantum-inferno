"""
This module facilitates the rapid construction of the Waxler and Assink (WK) pulse synthetic and its spectrum.
References:
- Waxler, R. and J. Assink (2019). Propagation Modeling Through Realistic Atmospheres and Benchmarking,
Chapter 15 in Infrasound Monitoring for Atmospheric Studies,
Second Edition, Springer, Switzerland, DOI 10.1007/978-3-319-75140_5, p. 509-549.
- Garces, M. A. (2024). Spectral Entropy (in progress)

"""
from typing import Optional, Tuple, Union

import numpy as np

from quantum_inferno.synth.synthetic_signals import white_noise_fbits, antialias_half_nyquist
from quantum_inferno.scales_dyadic import get_epsilon

def wk_blast_period_center(time_center_s: np.ndarray, pseudo_period_s: float) -> np.ndarray:
    """
    WK pulse

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy array with GT blast pulse
    """
    # With the +1, tau is the first zero crossing time.
    time_pos_s = pseudo_period_s / 4.0
    tau = time_center_s / time_pos_s + 1.0
    # Initialize GT
    p_gt = np.zeros(tau.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.0))  # ONLY positive pulse
    sigint_g17 = np.where((1.0 < tau) & (tau <= 1 + np.sqrt(6.0)))  # GT balanced pulse
    p_gt[sigint1] = 1.0 - tau[sigint1]
    p_gt[sigint_g17] = 1.0 / 6.0 * (1.0 - tau[sigint_g17]) * (1.0 + np.sqrt(6) - tau[sigint_g17]) ** 2.0

    return p_gt


def wk_blast_center_fast(
    frequency_peak_hz: float = 6.3, sample_rate_hz: float = 100.0, noise_std_loss_bits: float = 16.
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast computation of GT pulse with noise

    :param frequency_peak_hz: peak frequency, nominal 6.3 Hz for 1 tonne TNT
    :param sample_rate_hz: sample rate, nominal 100 Hz
    :param noise_std_loss_bits: noise loss relative to signal variance
    :return: centered time in seconds, GT pulse with white noise
    """
    # 16 cycles for 6th octave (M = 14)
    duration_points = int(16 / frequency_peak_hz * sample_rate_hz)
    time_center_s = np.arange(duration_points) / sample_rate_hz
    time_center_s -= time_center_s[-1] / 2.0
    sig_gt = wk_blast_period_center(time_center_s, 1 / frequency_peak_hz)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    return time_center_s, antialias_half_nyquist(sig_gt + sig_noise)


def wk_blast_center_noise(
    duration_s: float = 16., frequency_peak_hz: float = 6.3,
    sample_rate_hz: float = 100., noise_std_loss_bits: float = 16.
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast computation of GT pulse with noise for a specified duration in seconds

    :param duration_s: signal duration in seconds.  Default 16
    :param frequency_peak_hz: peak frequency.  Default 6.3 Hz for 1 tonne TNT
    :param sample_rate_hz: sample rate.  Default 100 Hz
    :param noise_std_loss_bits: noise loss relative to signal variance.  Default 16
    :return: centered time in seconds, GT pulse with white noise
    """
    time_center_s = np.arange(int(duration_s * sample_rate_hz)) / sample_rate_hz
    time_center_s -= time_center_s[-1] / 2.0
    sig_gt = wk_blast_period_center(time_center_s, 1 / frequency_peak_hz)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    return time_center_s, antialias_half_nyquist(sig_gt + sig_noise)


def wk_blast_center_noise_uneven(
    sensor_epoch_s: np.array, noise_std_loss_bits: float = 2., frequency_center_hz: Optional[float] = None
) -> np.ndarray:
    """
    Construct the WX explosion pulse for even or uneven sensor time
    in Gaussian noise with SNR in bits re signal STD.
    This is a very flexible variation.

    :param sensor_epoch_s: array with timestamps for signal in epoch seconds
    :param noise_std_loss_bits: number of bits below signal standard deviation. Default is 2
    :param frequency_center_hz: Optional center frequency in Hz
    :return: numpy array with anti-aliased GT explosion pulse with Gaussian noise
    """
    time_duration_s = sensor_epoch_s[-1] - sensor_epoch_s[0]

    pseudo_period_s = 1 / frequency_center_hz if frequency_center_hz else time_duration_s / 4.0

    # Convert to seconds
    time_center_s = sensor_epoch_s - sensor_epoch_s[0] - time_duration_s / 2.0
    sig_gt = wk_blast_period_center(time_center_s, pseudo_period_s)
    sig_noise = white_noise_fbits(np.copy(sig_gt), noise_std_loss_bits)

    return antialias_half_nyquist(sig_gt + sig_noise)


def wk_blast_ft(frequency_peak_hz: float, frequency_hz: Union[float, np.ndarray]) -> Union[float, complex, np.ndarray]:
    """
    Fourier transform of the GT blast pulse

    :param frequency_peak_hz: peak frequency in Hz
    :param frequency_hz: frequency in Hz, float or np.ndarray
    :return: Fourier transform of the GT blast pulse
    """
    w_scaled = 0.5 * np.pi * frequency_hz / frequency_peak_hz
    ft_g17_positive = (1.0 - 1j * w_scaled - np.exp(-1j * w_scaled)) / w_scaled ** 2.0
    ft_g17_negative = (
        np.exp(-1j * w_scaled * (1 + np.sqrt(6.0)))
        / (3.0 * w_scaled ** 4.0)
        * (
            1j * w_scaled * np.sqrt(6.0)
            + 3.0
            + np.exp(1j * w_scaled * np.sqrt(6.0)) * (3.0 * w_scaled ** 2.0 + 1j * w_scaled * 2.0 * np.sqrt(6.0) - 3.0)
        )
    )
    return (ft_g17_positive + ft_g17_negative) * np.pi / (2 * np.pi * frequency_peak_hz)


def wk_blast_spectral_density(
    frequency_peak_hz: float, frequency_hz: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], float]:
    """
    Spectral density of the GT blast pulse

    :param frequency_peak_hz: peak frequency in Hz
    :param frequency_hz: frequency in Hz, float or np.ndarray
    :return: spectral_density, spectral_density_peak
    """
    fourier_tx = wk_blast_ft(frequency_peak_hz, frequency_hz)
    spectral_density = 2 * np.abs(fourier_tx * np.conj(fourier_tx))
    return spectral_density, np.max(spectral_density)
