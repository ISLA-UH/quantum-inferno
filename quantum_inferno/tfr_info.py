"""
This module returns information and entropy from a TFR Power array
"""
from typing import Tuple

import numpy as np
import scipy.fft as sfft

import quantum_inferno.scales_dyadic as scales
from quantum_inferno.utilities.matrix import d1tile_x_d0d1, d0tile_x_d0d1


def log2_ceil(x: float, epsilon: float = scales.EPSILON64) -> float:
    """
    Compute ceiling of log2 of a positive input argument.
    Corrects for negative, complex or zero inputs by taking the absolute value and adding EPSILON

    :param x: input, converts to positive real
    :param epsilon: override zero, negative, and imaginary values
    :return: ceiling of log2
    """
    return np.ceil(np.log2(np.abs(x) + epsilon))


def log2_round(x: float, epsilon: float = scales.EPSILON64) -> float:
    """
    Compute rounded value of log2 of a positive input argument.
    Corrects for negative, complex or zero inputs by taking the absolute value and adding EPSILON

    :param x: input, converts to positive real
    :param epsilon: override zero, negative, and imaginary values
    :return: rounded to nearest integer value of log2
    """
    return float(np.round(np.log2(np.abs(x) + epsilon)))


def log2_floor(x: float, epsilon: float = scales.EPSILON64) -> float:
    """
    Compute floor of log2 of a positive input argument.
    Corrects for negative, complex or zero inputs by taking the absolute value and adding EPSILON

    :param x: input, converts to positive real
    :param epsilon: override zero, negative, and imaginary values
    :return: floor of log2
    """
    return np.floor(np.log2(np.abs(x) + epsilon))


def mat_max_idx(a: np.ndarray) -> Tuple[np.ndarray]:
    """
    :param a: matrix to find maximum for
    :return: The indexes of the max of a matrix
    """
    return np.unravel_index(a.argmax(), a.shape)


def mat_min_idx(a: np.ndarray) -> Tuple[np.ndarray]:
    """
    :param a: matrix to find minimum for
    :return: The indexes of the min of a matrix
    """
    return np.unravel_index(a.argmin(), a.shape)


def scale_log2_64(in_array: np.ndarray) -> np.ndarray:
    """
    :param in_array: input array
    :return: log2 of array values plus EPSILON64
    """
    return np.log2(in_array + scales.EPSILON64)


def scale_power_bits(power: np.ndarray) -> np.ndarray:
    """
    :param power: power from time-frequency representation
    :return: scaled power bits minus the maximum value
    """
    power_bits = scale_log2_64(power)
    return power_bits - np.max(power_bits)


def power_dynamics_scaled_bits(tfr_power: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Essential scales for power

    :param tfr_power: power from time-frequency representation
    :return: Log2 scaled power bits, dynamic time range, dynamic frequency range
    """
    tfr_power_bits = scale_power_bits(tfr_power)  # Log2 scaled power bits
    # Dynamic range per time step re max
    tfr_power_per_time_bits = scale_power_bits(np.sum(tfr_power, axis=0))
    # Dynamic range per frequency band re max
    tfr_power_per_freq_bits = scale_power_bits(np.sum(tfr_power, axis=1))
    return tfr_power_bits, tfr_power_per_time_bits, tfr_power_per_freq_bits


def get_info_and_entropy_32(marginal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param marginal: array of values to start with
    :return: info, entropy and ref_entropy of the array, using EPSILON32
    """
    info = -np.log2(marginal + scales.EPSILON32)
    return info, marginal * info, np.log2(len(marginal)) / len(marginal)


class Shannon:
    """
    class for Shannon information of a signal

    Attributes:
        marginal: np.ndarray, waveform

        info: np.ndarray, info of waveform

        entropy: np.ndarray, entropy of waveform

        ref_entropy: np.ndarray, reference entropy of waveform

        isnr: np.ndarray, isnr of waveform

        esnr: np.ndarray, esnr of waveform
    """

    def __init__(self, marginal: np.ndarray):
        """
        :param marginal: waveform
        """
        self.marginal: np.ndarray = marginal
        self.info: np.ndarray
        self.entropy: np.ndarray
        self.ref_entropy: np.ndarray
        # Average entropy for P ~ 1 / NFFT
        self.info, self.entropy, self.ref_entropy = get_info_and_entropy_32(self.marginal)
        self.isnr: np.ndarray = np.log2(len(self.info)) - self.info
        self.esnr: np.ndarray = self.entropy / self.ref_entropy


class ShannonTDR(Shannon):
    """
    class for Shannon TDR information, refer to Shannon for base class

    Attributes:
        sig: np.ndarray, data to process
    """
    def __init__(self, sig_in_real: np.ndarray):
        """
        :param sig_in_real: data to process
        """
        self.sig: np.ndarray = sig_in_real / np.sqrt(np.sum(sig_in_real ** 2))
        super().__init__(self.sig ** 2)

    def print_total_ref_entropy(self):
        print("Ref entropy, time:", self.ref_entropy)

    def print_total_entropy(self):
        print("Total Entropy, time:", np.sum(self.entropy))

    def print_total_marginal(self):
        print("Sum of time marginal:", np.sum(self.marginal))


class ShannonFFT(Shannon):
    """
    class for Shannon FFT information, refer to Shannon for base class

    Attributes:
        sig: np.ndarray, data to process

        angle_rads: np.ndarray, angle in radians

        frequency: np.ndarray, frequencies of the data
    """
    def __init__(self, sig_in_real: np.ndarray):
        """
        :param sig_in_real: data to process
        """
        self.sig: np.ndarray = sfft.rfft(x=sig_in_real)
        self.angle_rads: np.ndarray = np.unwrap(np.angle(self.sig))
        self.frequency: np.ndarray = np.arange(len(self.angle_rads)) / len(self.angle_rads) / 2.0
        fft_sq = np.abs(self.sig) ** 2
        super().__init__(fft_sq / np.sum(fft_sq))

    def print_total_ref_entropy(self):
        print("Ref entropy, frequency:", self.ref_entropy)

    def print_total_entropy(self):
        print("Total Entropy, frequency:", np.sum(self.entropy))

    def print_total_marginal(self):
        print("Sum of frequency marginal:", np.sum(self.marginal))


def shannon_tdr_fft(sig_in_real: np.ndarray) -> Tuple[ShannonTDR, ShannonFFT]:
    """
    Shannon information and entropy

    :param sig_in_real: data to process
    :return: ShannonTDR and ShannonFFT of the data
    """
    return ShannonTDR(sig_in_real), ShannonFFT(sig_in_real)


class ShannonStft:
    """
    Class for Shannon stft

    Attributes:
        info: np.ndarray, info of waveform

        shannon_bits: np.ndarray, shannon bits of waveform

        ref_bits: np.ndarray, ref bits of waveform

        isnr: np.ndarray, isnr of waveform

        esnr: np.ndarray, esnr of waveform
    """

    def __init__(self, tfr_pow_pdf: np.ndarray, deg_free: int):
        """
        :param tfr_pow_pdf: tfr power matrix
        :param deg_free: degrees of freedom
        """
        self.info: np.ndarray = -scale_log2_64(tfr_pow_pdf)
        self.shannon_bits: np.ndarray = tfr_pow_pdf * self.info
        self.ref_bits: float = np.log2(deg_free) / deg_free
        self.isnr: np.ndarray = np.log2(deg_free) - self.info
        self.esnr: np.ndarray = self.shannon_bits / self.ref_bits


def shannon_stft_from_tfr_power(tfr_power: np.ndarray) -> ShannonStft:
    """
    :param tfr_power: tfr power matrix
    :return: ShannonStft of the data
    """
    return ShannonStft(tfr_power / np.sum(tfr_power), tfr_power.shape[0] * tfr_power.shape[1])


class ShannonStftPerTime(ShannonStft):
    """
    Class for Shannon stft per time, refer to ShannonStft for base class
    """
    def __init__(self, tfr_power: np.ndarray):
        """
        :param tfr_power: tfr power matrix
        """
        tfr_power_per_time_pdf = d1tile_x_d0d1(d1=1 / np.sum(tfr_power, axis=0) + scales.EPSILON64, d0d1=tfr_power)
        super().__init__(tfr_power_per_time_pdf, tfr_power.shape[0])


class ShannonStftPerFreq(ShannonStft):
    """
    Class for Shannon stft per frequency, refer to ShannonStft for base class
    """
    def __init__(self, tfr_power: np.ndarray):
        """
        :param tfr_power: tfr power matrix
        """
        tfr_power_per_freq_pdf = d0tile_x_d0d1(d0=1 / np.sum(tfr_power, axis=1) + scales.EPSILON64, d0d1=tfr_power)
        super().__init__(tfr_power_per_freq_pdf, tfr_power.shape[1])
