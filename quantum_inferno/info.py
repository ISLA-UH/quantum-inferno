"""Module to return information and entropy from TFR Power"""

import numpy as np
from scipy import signal
import libquantum_ops.scales_dyadic as scales
from libquantum_ops import utils
import scipy.fft


# TODO: Use 'em or lose 'em
def log2_ceil(x: float, epsilon: float = scales.EPSILON64) -> float:
    """
    Compute ceiling of log2 of a positive input argument.
    Corrects for negative, complex or zero inputs by
    taking the absolute value and adding EPSILON
    :param x: input, converts to positive real
    :param epsilon: override zero, negative, and imaginary values
    :return: ceiling of log2
    """
    real_x = np.abs(x) + epsilon
    return np.ceil(np.log2(real_x))


def log2_round(x: float, epsilon: float = scales.EPSILON64) -> float:
    """
    Compute ceiling of log2 of a positive input argument.
    Corrects for negative, complex or zero inputs by
    taking the absolute value and adding EPSILON
    :param x: input, converts to positive real
    :param epsilon: override zero, negative, and imaginary values
    :return: ceiling of log2
    """
    real_x = np.abs(x) + epsilon
    return np.round(np.log2(real_x))


def log2_floor(x: float, epsilon: float = scales.EPSILON64) -> float:
    """
    Compute ceiling of log2 of a positive input argument.
    Corrects for negative, complex or zero inputs by
    taking the absolute value and adding EPSILON
    :param x: input, converts to positive real
    :param epsilon: override zero, negative, and imaginary values
    :return: ceiling of log2
    """
    real_x = np.abs(x) + epsilon
    return np.floor(np.log2(real_x))


def mat_max_idx(a: np.ndarray):
    """Find the indexes of the max of a matrix"""
    return np.unravel_index(a.argmax(), a.shape)


def mat_min_idx(a: np.ndarray):
    """Find the indexes of the min of a matrix"""
    return np.unravel_index(a.argmin(), a.shape)


def power_dynamics_scaled_bits(tfr_power: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Essential scales for power
    :param tfr_power: power from time-frequency representation
    :return:
    """
    tfr_power_bits = np.log2(tfr_power + scales.EPSILON64)
    tfr_power_bits -= np.max(tfr_power_bits)
    # Dynamic range per time step re max
    tfr_power_per_time = np.sum(tfr_power, axis=0)
    tfr_power_per_time_bits = np.log2(tfr_power_per_time + scales.EPSILON64)
    tfr_power_per_time_bits -= np.max(tfr_power_per_time_bits)
    # Dynamic range per frequency band re max
    tfr_power_per_freq = np.sum(tfr_power, axis=1)
    tfr_power_per_freq_bits = np.log2(tfr_power_per_freq + scales.EPSILON64)
    tfr_power_per_freq_bits -= np.max(tfr_power_per_freq_bits)

    return [tfr_power_bits, tfr_power_per_freq_bits, tfr_power_per_time_bits]


def shannon(tfr_power):
    """
    Shannon information and entropy
    :param tfr_power:
    :return:
    """
    atom_bits = 3
    num_freq = tfr_power.shape[0]
    num_time = tfr_power.shape[1]
    num_dof = num_freq*num_time  # Degrees of freedom
    ref_shannon_bits = np.log2(num_dof)/num_dof
    # ref_shannon_bits = (np.log2(num_freq)+3)/num_dof

    # Normalized power pdf, information, and entropy over whole distribution
    tfr_power_pdf = tfr_power/np.sum(tfr_power)
    # Shannon information; NOT ADDITIVE, but already normalized
    tfr_info = -np.log2(tfr_power_pdf + scales.EPSILON64)
    tfr_isnr = np.log2(num_dof) - tfr_info
    # Shannon entropy per bin
    # This is the expected value of the information
    tfr_shannon_bits = tfr_power_pdf * tfr_info
    tfr_shannon_total_bits = np.sum(tfr_shannon_bits)
    # Relative to ref_bits
    tfr_esnr = tfr_shannon_bits/ref_shannon_bits

    # # tfr_esnr = num_dof*tfr_shannon_bits/tfr_shannon_total_bits
    # tfr_esnr = (tfr_shannon_bits - ref_shannon_bits)*num_dof
    # tfr_esnr = tfr_shannon_bits/ref_shannon_bits/atom_bits
    # tfr_esnr = (np.log2(num_dof) - tfr_shannon_bits)*num_dof
    # # Relative entropy dsnr
    # tfr_dsnr_uni = ref_shannon_bits-tfr_shannon_bits
    # # Total relative entropy
    # tfr_DSNR_total = num_dof*ref_shannon_bits-tfr_shannon_total_bits

    return [tfr_info, tfr_shannon_bits, tfr_isnr, tfr_esnr]


def shannon_esnrt_per_time(tfr_power):
    """
    Normalized power pdf, information, and entropy per time step
    :param tfr_power:
    :return:
    """

    num_freq = tfr_power.shape[0]
    num_dof = num_freq  # Degrees of freedom
    ref_shannon_bits = np.log2(num_dof)/num_dof

    tfr_power_per_time = np.sum(tfr_power, axis=0) + scales.EPSILON64
    tfr_power_per_time_pdf = utils.d1tile_x_d0d1(d1=1/tfr_power_per_time, d0d1=tfr_power)
    tfr_info_per_time = -np.log2(tfr_power_per_time_pdf + scales.EPSILON64)
    tfr_isnr_per_time = np.log2(num_dof) - tfr_info_per_time

    tfr_shannon_per_time_bits = tfr_power_per_time_pdf * tfr_info_per_time
    # tfr_shannon_sum_per_time_bits = np.sum(tfr_shannon_per_time_bits, axis=0)
    # tfr_esnr_per_time = \
    #     utils.d1tile_x_d0d1(d1=tfr_power.shape[0]/tfr_shannon_sum_per_time_bits, d0d1=tfr_shannon_per_time_bits)
    # Relative to ref_bits
    # tfr_esnr_per_time = (tfr_shannon_per_time_bits - ref_shannon_bits)*num_dof
    tfr_esnr_per_time = tfr_shannon_per_time_bits/ref_shannon_bits

    return [tfr_info_per_time, tfr_shannon_per_time_bits, tfr_isnr_per_time, tfr_esnr_per_time]


def shannon_esnrf_per_freq(tfr_power):
    """
    Normalized power pdf, information, and entropy per frequency step
    :param tfr_power:
    :return:
    """

    atom_bits = 3
    num_time = tfr_power.shape[1]
    num_dof = num_time  # Degrees of freedom
    ref_shannon_bits = np.log2(num_dof)/num_dof
    # ref_shannon_bits = 3/num_dof

    tfr_power_per_freq = np.sum(tfr_power, axis=1) + scales.EPSILON64
    tfr_power_per_freq_pdf = utils.d0tile_x_d0d1(d0=1/tfr_power_per_freq, d0d1=tfr_power)
    tfr_info_per_freq = -np.log2(tfr_power_per_freq_pdf + scales.EPSILON64)
    # TODO: taper the np.log2(num_dof), otherwise blows up at edges
    # TODO: Verify, standardize with dsp taper
    tukey_power_envelope = signal.windows.tukey(M=num_time, alpha=0.2, sym=True)**2
    power_taper_tile = utils.just_tile_d1(d1_array1d_in=tukey_power_envelope, d0d1_shape=tfr_power.shape)
    tfr_isnr_per_freq = np.log2(num_dof*power_taper_tile + scales.EPSILON64) - tfr_info_per_freq

    # # Debug
    # print('\nlog2DOF:', np.log2(num_dof))
    # print(num_dof_tapered.shape)
    # print(tfr_power.shape)
    # print('tfr_power_per_freq:', tfr_power_per_freq)
    # print('tfr_isnr_per_freq[:, -100]:', tfr_isnr_per_freq[:, -100])
    # print('tfr_isnr_per_freq[:, -1]:', tfr_isnr_per_freq[:, -1])
    # exit()
    # TODO: Reconcile with dsp taper
    tfr_shannon_per_freq_bits = tfr_power_per_freq_pdf * power_taper_tile * tfr_info_per_freq
    # tfr_shannon_sum_per_freq_bits = np.sum(tfr_shannon_per_freq_bits, axis=1)
    # tfr_esnr_per_freq = \
    #     utils.d0tile_x_d0d1(d0=tfr_power.shape[1]/tfr_shannon_sum_per_freq_bits, d0d1=tfr_shannon_per_freq_bits)
    # Relative to ref_bits
    # tfr_esnr_per_freq = (tfr_shannon_per_freq_bits - ref_shannon_bits)*num_dof
    # tfr_esnr_per_freq = tfr_shannon_per_freq_bits/ref_shannon_bits
    tfr_esnr_per_freq = tfr_shannon_per_freq_bits/ref_shannon_bits  # /atom_bits

    return [tfr_info_per_freq, tfr_shannon_per_freq_bits, tfr_isnr_per_freq, tfr_esnr_per_freq]


def shannon_tdr_fft(sig_in_real, verbose: bool = True):
    """
    Shannon information and entropy
    :param sig_in_real:
    :param verbose: print to screen
    :return:
    """

    atom_bits = 3
    num_dof = len(sig_in_real)
    ref_shannon_bits = np.log2(num_dof)/num_dof

    # Time-domain representation
    sig_sq_total = np.sum(sig_in_real**2)
    tdr_sig = sig_in_real/np.sqrt(sig_sq_total)
    # Time marginals
    tdr_marginal = tdr_sig**2
    tdr_info = -np.log2(tdr_marginal + scales.EPSILON32)
    tdr_entropy = tdr_marginal*tdr_info
    # tdr_marginal_total = np.sum(tdr_marginal)
    # tdr_entropy_total = np.sum(tdr_entropy)

    # FFT information and entropy of real input signal
    fft_sig = scipy.fft.rfft(x=sig_in_real)
    fft_angle_rads = np.unwrap(np.angle(fft_sig))
    # RFFT only goes up to Nyquist TODO: use rfftfreq, presently assuming sample rate of unity
    fft_frequency = np.arange(len(fft_angle_rads))/len(fft_angle_rads)/2.
    # Frequency marginals
    fft_sq = np.abs(fft_sig)**2
    fft_sq_total = np.sum(fft_sq)
    fft_marginal = fft_sq/fft_sq_total
    fft_info = -np.log2(fft_marginal + scales.EPSILON32)
    fft_entropy = fft_marginal*fft_info
    # fft_marginal_total = np.sum(fft_marginal)
    # fft_entropy_total = np.sum(fft_entropy)

    # Average entropy for P ~ 1/NFFT
    ref_tdr_entropy = np.log2(len(tdr_marginal))/len(tdr_marginal)
    ref_fft_entropy = np.log2(len(fft_marginal))/len(fft_marginal)

    # THE CRUX
    tdr_isnr = np.log2(len(tdr_info)) - tdr_info
    tdr_esnr = tdr_entropy/ref_tdr_entropy
    fft_isnr = np.log2(len(fft_info)) - fft_info
    fft_esnr = fft_entropy/ref_fft_entropy

    if verbose:
        print('Ref entropy, time:', ref_tdr_entropy)
        print('Ref entropy, frequency:', ref_fft_entropy)
        print('Total Entropy, time:', np.sum(tdr_entropy))
        print('Total Entropy, frequency:', 2*np.sum(fft_entropy))
        print('Sum of time marginal:', np.sum(tdr_marginal))
        print('Sum of frequency marginal:', np.sum(fft_marginal))

    return [tdr_sig, tdr_marginal, tdr_info, tdr_entropy, tdr_isnr, tdr_esnr,
            fft_sig, fft_marginal, fft_frequency, fft_angle_rads, fft_info, fft_entropy, fft_isnr, fft_esnr]


def shannon_tdr(sig_in_real, verbose: bool = True):
    """
    Shannon information and entropy
    :param sig_in_real:
    :param verbose: print to screen
    :return:
    """

    # Time-domain representation
    sig_sq_total = np.sum(sig_in_real**2)
    tdr_sig = sig_in_real/np.sqrt(sig_sq_total)
    # Time marginals
    # TODO: rel relative to q=1/M; log2(p/q), after Rosso et al. 2001
    tdr_marginal = tdr_sig**2
    tdr_info = -np.log2(tdr_marginal + scales.EPSILON32)
    tdr_info_rel = np.log2(tdr_marginal*len(tdr_marginal) + scales.EPSILON32)
    tdr_entropy = tdr_marginal*tdr_info
    tdr_entropy_total = np.sum(tdr_entropy)
    # Average entropy for P ~ 1/NFFT
    ref_tdr_entropy = np.log2(len(tdr_marginal))/len(tdr_marginal)
    ref_tdr_rel_entropy = 1/(2*np.log(2))*(np.max(tdr_marginal)-1/len(tdr_marginal))**2

    # THE CRUX
    tdr_isnr = np.log2(len(tdr_info)) - tdr_info
    # Stable
    tdr_esnr = tdr_entropy/ref_tdr_entropy
    # TODO: Experiment
    # tdr_rel_info = np.log2(tdr_marginal*len(tdr_marginal) + scales.EPSILON32)
    # May be promising, see Lin (1999)
    # tdr_rel_info = np.log2(tdr_marginal/(0.5*tdr_marginal + 0.5/len(tdr_marginal)) + scales.EPSILON32)
    # tdr_rel_entropy = tdr_marginal*tdr_rel_info
    # Doesn't go to zero
    # tdr_rel_entropy = (tdr_marginal-1/len(tdr_marginal))*tdr_rel_info
    # # Formal rel entropy
    # tdr_esnr = tdr_rel_entropy
    # Promising
    # tdr_esnr = tdr_rel_entropy/ref_tdr_rel_entropy

    if verbose:
        print('Ref entropy, time:', ref_tdr_entropy)
        print('Total Entropy, time:', np.sum(tdr_entropy))
        print('Sum of time marginal:', np.sum(tdr_marginal))

    return [tdr_sig, tdr_marginal, tdr_info, tdr_entropy, tdr_isnr, tdr_esnr]


def shannon_fft(fft_sig, verbose: bool = True):
    """
    Shannon information and entropy
    :param fft_sig:
    :param verbose: print to screen
    :return:
    """

    # FFT information and entropy of real input signal
    # fft_sig = scipy.fft.rfft(x=sig_in_real)
    fft_angle_rads = np.unwrap(np.angle(fft_sig))
    # # RFFT only goes up to Nyquist
    # fft_frequency = np.arange(len(fft_angle_rads))/len(fft_angle_rads)/2.
    # Frequency marginals
    fft_sq = np.abs(fft_sig)**2
    fft_sq_total = np.sum(fft_sq)
    fft_marginal = fft_sq/fft_sq_total
    fft_info = -np.log2(fft_marginal + scales.EPSILON32)
    fft_info_rel = np.log2(fft_marginal*len(fft_marginal) + scales.EPSILON32)
    fft_entropy = fft_marginal*fft_info
    # fft_marginal_total = np.sum(fft_marginal)
    fft_entropy_total = np.sum(fft_entropy)

    # Average entropy for P ~ 1/NFFT
    ref_fft_entropy = np.log2(len(fft_marginal)) / len(fft_marginal)
    # ref_fft_rel_entropy = 1/(2*np.log(2))*(np.max(fft_marginal)-1/len(fft_marginal))**2

    # THE CRUX
    fft_isnr = np.log2(len(fft_info)) - fft_info
    # Stable
    fft_esnr = fft_entropy/ref_fft_entropy
    # TODO: Experiments
    # fft_esnr = fft_marginal*fft_info_rel/ref_fft_entropy
    # fft_esnr = (fft_entropy - np.log2(len(fft_marginal)))
    # fft_rel_info = np.log2(fft_marginal*len(fft_marginal) + scales.EPSILON32)
    # fft_rel_entropy = fft_marginal*fft_rel_info
    # Formal def
    # fft_esnr = fft_rel_entropy
    # # Not bad
    # fft_esnr = np.log2(len(fft_marginal))*fft_marginal*fft_rel_info
    # Promising
    # fft_esnr = fft_entropy_total * fft_rel_entropy
    # Large
    # fft_esnr = fft_rel_entropy/ref_fft_rel_entropy
    # This should work
    # fft_esnr = np.log2(len(fft_marginal))/len(fft_marginal) - fft_entropy

    if verbose:
        print('Ref entropy, frequency:', ref_fft_entropy)
        print('Total Entropy, frequency:', 2*np.sum(fft_entropy))
        print('Sum of frequency marginal:', np.sum(fft_marginal))

    return [fft_marginal, fft_angle_rads, fft_info, fft_entropy, fft_isnr, fft_esnr]


def shannon_stft(tfr_power):
    """
    Shannon information and entropy
    :param tfr_power:
    :return:
    """

    atom_bits = 3
    num_freq = tfr_power.shape[0]
    num_time = tfr_power.shape[1]
    # Previous
    num_dof = num_freq*num_time    # Degrees of freedom
    ref_shannon_bits = np.log2(num_dof)/num_dof
    # TODO: Works OK
    # num_dof = num_time*(1-stft_overlap_fractional)    # Degrees of freedom
    # num_dof = num_time    # Degrees of freedom
    # ref_shannon_bits = np.log2(num_dof)/num_dof

    # Normalized power pdf, information, and entropy over whole distribution
    tfr_power_pdf = tfr_power/np.sum(tfr_power)
    # Shannon information; NOT ADDITIVE, but already normalized
    tfr_info = -np.log2(tfr_power_pdf + scales.EPSILON64)
    tfr_isnr = np.log2(num_dof) - tfr_info
    # Shannon entropy per bin
    # This is the expected value of the information
    tfr_shannon_bits = tfr_power_pdf * tfr_info
    # Relative to ref_bits
    tfr_esnr = tfr_shannon_bits/ref_shannon_bits

    # print('NFFT:', nfft)
    # print('Ref shannon bits')
    # print(ref_shannon_bits)
    # print('min max tfr_shannon_bits')
    # print(np.min(tfr_shannon_bits), np.max(tfr_shannon_bits))

    return [tfr_info, tfr_shannon_bits, tfr_isnr, tfr_esnr]


def shannon_stft_esnrt_per_time(tfr_power):
    """
    Normalized power pdf, information, and entropy per time step
    :param tfr_power:
    :return:
    """

    num_time = tfr_power.shape[1]
    num_freq = tfr_power.shape[0]

    # # TODO: Clean up!
    # num_dof = num_time    # Degrees of freedom
    num_dof = num_freq
    ref_shannon_bits = np.log2(num_dof)/num_dof

    tfr_power_per_time = np.sum(tfr_power, axis=0) + scales.EPSILON64
    tfr_power_per_time_pdf = utils.d1tile_x_d0d1(d1=1/tfr_power_per_time, d0d1=tfr_power)
    tfr_info_per_time = -np.log2(tfr_power_per_time_pdf + scales.EPSILON64)
    tfr_isnr_per_time = np.log2(num_freq) - tfr_info_per_time

    tfr_shannon_per_time_bits = tfr_power_per_time_pdf * tfr_info_per_time
    # Relative to ref_bits
    tfr_esnr_per_time = tfr_shannon_per_time_bits/ref_shannon_bits

    print('Ref tfr_shannon_per_time_bits')
    print(ref_shannon_bits)
    print('min max tfr_shannon_per_time_bits')
    print(np.min(tfr_shannon_per_time_bits), np.max(tfr_shannon_per_time_bits))

    return [tfr_info_per_time, tfr_shannon_per_time_bits, tfr_isnr_per_time, tfr_esnr_per_time]


def shannon_stft_esnrf_per_freq(tfr_power):
    """
    Normalized power pdf, information, and entropy per frequency step
    :param tfr_power:
    :return:
    """

    atom_bits = 3
    num_time = tfr_power.shape[1]
    num_freq = tfr_power.shape[0]
    num_dof = num_time    # Degrees of freedom
    ref_shannon_bits = np.log2(num_dof)/num_dof

    tfr_power_per_freq = np.sum(tfr_power, axis=1) + scales.EPSILON64
    # Threshold the power to prevent it from blowing up near Nyquist
    # min_inv = np.min(inv_tfr_power_per_freq)
    # idx = np.argwhere(inv_tfr_power_per_freq > min_inv*2**32)
    # inv_tfr_power_per_freq[idx] = np.zeros(idx.shape)

    tfr_power_per_freq_pdf = utils.d0tile_x_d0d1(d0=1/tfr_power_per_freq, d0d1=tfr_power)
    tfr_info_per_freq = -np.log2(tfr_power_per_freq_pdf + scales.EPSILON64)
    tfr_isnr_per_freq = np.log2(num_dof) - tfr_info_per_freq

    tfr_shannon_per_freq_bits = tfr_power_per_freq_pdf * tfr_info_per_freq

    # Relative to ref_bits
    tfr_esnr_per_freq = tfr_shannon_per_freq_bits/ref_shannon_bits

    print('Ref tfr_esnr_per_freq bits')
    print(ref_shannon_bits)
    print('min max tfr_esnr_per_freq')
    print(np.min(tfr_esnr_per_freq), np.max(tfr_esnr_per_freq))

    return [tfr_info_per_freq, tfr_shannon_per_freq_bits, tfr_isnr_per_freq, tfr_esnr_per_freq]

# def shannon_per_freq_order(tfr_power, order: float = 3):
#     """
#     Normalized power pdf, information, and entropy per frequency step
#     :param order:
#     :param tfr_power:
#     :return:
#     """
#
#     # TODO: Pass the number of wavelets per band
#     num_freq = tfr_power.shape[0]
#     scaling = 2**(np.arange(num_freq)/order)/2
#
#     num_time = tfr_power.shape[1]
#     num_dof = num_time  # Degrees of freedom
#     ref_shannon_bits = np.log2(num_dof)/num_dof
#
#     tfr_power_per_freq = np.sum(tfr_power, axis=1)
#     tfr_power_per_freq_pdf = utils.d0tile_x_d0d1(d0=1/tfr_power_per_freq, d0d1=tfr_power)
#     tfr_info_per_freq = -np.log2(tfr_power_per_freq_pdf + scales.EPSILON64)
#     tfr_isnr_per_freq = np.log2(tfr_power.shape[1]) - tfr_info_per_freq
#
#     tfr_shannon_per_freq_bits = tfr_power_per_freq_pdf * tfr_info_per_freq
#     # tfr_shannon_sum_per_freq_bits = np.sum(tfr_shannon_per_freq_bits, axis=1)
#     # tfr_esnr_per_freq = \
#     #     utils.d0tile_x_d0d1(d0=tfr_power.shape[1]/tfr_shannon_sum_per_freq_bits, d0d1=tfr_shannon_per_freq_bits)
#     # tfr_esnr_per_freq = tfr_shannon_per_freq_bits/ref_shannon_bits
#     tfr_esnr_per_freq = \
#         utils.d0tile_x_d0d1(d0=1/ref_shannon_bits/scaling, d0d1=tfr_shannon_per_freq_bits)
#
#     return [tfr_info_per_freq, tfr_shannon_per_freq_bits, tfr_isnr_per_freq, tfr_esnr_per_freq]