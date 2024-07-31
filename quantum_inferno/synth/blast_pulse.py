"""
This module facilitates the rapid construction of the GT blast pulse synthetic,
its integral and derivatives, and its spectrum
References:
- Garcés, M. A. (2019). Explosion Source Models,
Chapter in Infrasound Monitoring for Atmospheric Studies,
Second Edition, Springer, Switzerland, DOI 10.1007/978-3-319-75140_5, p. 273-345.
- Schnurr, J. M., K. Kim, M. A. Garcés, A. Rodgers (2020).
Improved parametric models for explosion pressure signals derived from large datasets,
Seism. Res. Lett.
- Kim, K, A. R. Rodgers, M. A. Garces, and S. C. Myers (2021).
Empirical Acoustic Source Model for Chemical Explosions in Air.
Bulletin of the Seismological Society of America
"""
from typing import Optional, Tuple, Union

import numpy as np

from quantum_inferno.synth.synthetic_signals import white_noise_fbits, antialias_half_nyquist
from quantum_inferno.scales_dyadic import get_epsilon


def gt_blast_period_center(time_center_s: np.ndarray, pseudo_period_s: float) -> np.ndarray:
    """
    GT blast pulse

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


def gt_hilbert_blast_period_center(time_center_s: np.ndarray, pseudo_period_s: float) -> np.ndarray:
    """
    Hilbert transform of the GT blast pulse

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy array with Hilbert transform of the GT blast pulse
    """
    # With the +1, tau is the first zero crossing time.
    time_pos_s = pseudo_period_s / 4.0
    tau = time_center_s / time_pos_s + 1.0
    a = 1 + np.sqrt(6)
    # Initialize GT
    p_gt_h = np.zeros(tau.size)  # Hilbert of Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.0))  # ONLY positive pulse
    sigint2 = np.where((1.0 < tau) & (tau <= 1 + np.sqrt(6.0)))  # GT balanced pulse
    tau1 = tau[sigint1]
    tau2 = tau[sigint2]

    p_gt_h[sigint1] = 1.0 + (1 - tau1) * np.log(tau1 + get_epsilon()) - (1 - tau1) * np.log(1 - tau1 + get_epsilon())
    p_gt_h21 = (a - 1) / 6.0 * (a * (2 * a + 5) - 1 + 6 * tau2 ** 2 - 3 * tau2 * (1 + 3 * a))
    p_gt_h22 = (tau2 - 1) * (a - tau2) ** 2 * (np.log(a - tau2 + get_epsilon()) - np.log(tau2 - 1 + get_epsilon()))
    p_gt_h[sigint2] = 1.0 / 6.0 * (p_gt_h21 + p_gt_h22)
    p_gt_h /= np.pi

    return p_gt_h


def gt_blast_center_fast(
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
    sig_gt = gt_blast_period_center(time_center_s, 1 / frequency_peak_hz)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    return time_center_s, antialias_half_nyquist(sig_gt + sig_noise)


def gt_blast_center_noise(
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
    sig_gt = gt_blast_period_center(time_center_s, 1 / frequency_peak_hz)
    sig_noise = white_noise_fbits(sig_gt, noise_std_loss_bits)
    return time_center_s, antialias_half_nyquist(sig_gt + sig_noise)


def gt_blast_center_noise_uneven(
    sensor_epoch_s: np.array, noise_std_loss_bits: float = 2., frequency_center_hz: Optional[float] = None
) -> np.ndarray:
    """
    Construct the GT explosion pulse of Garces (2019) for even or uneven sensor time
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
    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_noise = white_noise_fbits(np.copy(sig_gt), noise_std_loss_bits)

    return antialias_half_nyquist(sig_gt + sig_noise)


# Integrals and derivatives, with delta function estimate and discontinuity boundary conditions
def gt_blast_derivative_period_center(time_center_s: np.ndarray, pseudo_period_s: float) -> np.ndarray:
    """
    Derivative of the GT blast with delta function approximation

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy ndarray with derivative of the GT blast with delta function approximation
    """
    # Garces (2019) ground truth GT blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
    time_pos_s = pseudo_period_s / 4.0
    tau = time_center_s / time_pos_s + 1.0
    # Initialize GT
    p_gtd = np.zeros(tau.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.0))  # ONLY positive pulse
    sigint_g17 = np.where((1.0 < tau) & (tau <= 1 + np.sqrt(6.0)))  # GT balanced pulse
    p_gtd[sigint1] = -1.0
    p_gtd[sigint_g17] = -1.0 / 6.0 * (3.0 + np.sqrt(6) - 3 * tau[sigint_g17]) * (1.0 + np.sqrt(6) - tau[sigint_g17])

    return p_gtd


def gt_blast_integral_period_center(time_center_s: np.ndarray, pseudo_period_s: float) -> np.ndarray:
    """
    Integral of the GT blast with initial condition at zero crossing

    :param time_center_s: array with time
    :param pseudo_period_s: period in seconds
    :return: numpy ndarray with integral of GT blast pulse
    """
    # Garces (2019) ground truth GT blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
    time_pos_s = pseudo_period_s / 4.0
    tau = time_center_s / time_pos_s + 1.0
    # Initialize GT
    p_gti = np.zeros(tau.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.0))  # ONLY positive pulse
    sigint_g17 = np.where((1.0 < tau) & (tau <= 1 + np.sqrt(6.0)))  # GT balanced pulse
    p_gti[sigint1] = (1.0 - tau[sigint1] / 2.0) * tau[sigint1]

    p_gti[sigint_g17] = (
        -tau[sigint_g17]
        / 72.0
        * (
            3 * tau[sigint_g17] ** 3
            - 4 * (3 + 2 * np.sqrt(6)) * tau[sigint_g17] ** 2
            + 6 * (9 + 4 * np.sqrt(6)) * tau[sigint_g17]
            - 12 * (7 + 2 * np.sqrt(6))
        )
    )

    integration_constant = p_gti[sigint1][-1] - p_gti[sigint_g17][0]
    p_gti[sigint_g17] += integration_constant

    return p_gti


def gt_blast_center_integral_and_derivative(
    frequency_peak_hz: float, sample_rate_hz: float
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integral and derivative of GT pulse relative to tau (NOT time_s)

    :param frequency_peak_hz: peak frequency in Hz
    :param sample_rate_hz: sample rate in Hz
    :return: tau center, numpy ndarray with GT blast pulse, numpy ndarray with integral of GT blast pulse, numpy array
        with derivative of GT blast pulse
    """
    pseudo_period_s = 1 / frequency_peak_hz
    time_center_s = np.arange(int(2 / frequency_peak_hz * sample_rate_hz)) / sample_rate_hz
    time_center_s -= time_center_s[-1] / 2.0
    tau_center = time_center_s / (pseudo_period_s / 4.0)

    sig_gt = gt_blast_period_center(time_center_s, pseudo_period_s)
    sig_gt_i = gt_blast_integral_period_center(time_center_s, pseudo_period_s)
    sig_gt_d = gt_blast_derivative_period_center(time_center_s, pseudo_period_s)
    sig_gt_d[np.argmax(sig_gt) - 1] = np.max(np.diff(sig_gt)) / np.mean(np.diff(tau_center))

    return tau_center, sig_gt, sig_gt_i, sig_gt_d


def gt_blast_ft(frequency_peak_hz: float, frequency_hz: Union[float, np.ndarray]) -> Union[float, complex, np.ndarray]:
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


def gt_blast_spectral_density(
    frequency_peak_hz: float, frequency_hz: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], float]:
    """
    Spectral density of the GT blast pulse

    :param frequency_peak_hz: peak frequency in Hz
    :param frequency_hz: frequency in Hz, float or np.ndarray
    :return: spectral_density, spectral_density_peak
    """
    fourier_tx = gt_blast_ft(frequency_peak_hz, frequency_hz)
    spectral_density = 2 * np.abs(fourier_tx * np.conj(fourier_tx))
    return spectral_density, np.max(spectral_density)
