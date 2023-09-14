"""
This module contains methods used in resampling signals
"""
import numpy as np
from scipy import interpolate, signal


def resampled_mean_no_hop(sig_wf: np.array,
                          sig_time: np.array,
                          resampling_factor: int):
    """
    Resamples a time series.  If the resampling factor is less than 1, returns the original series.

    Assume len(sig_time) and len(resampling_factor) is a power of two.
    todo: what to do if not

    :param sig_wf: audio waveform
    :param sig_time: audio timestamps in seconds, same dimensions as sig
    :param resampling_factor: down sampling factor.  if less than 1, return the original series.
    :return: resampled signal waveform, resampled signal timestamps
    """
    if resampling_factor <= 1:
        print('No Resampling/Averaging!')
        return sig_wf, sig_time
    # Variance over the signal waveform, all bands
    points_resampled: int = int(np.round((len(sig_wf)/resampling_factor)))
    new_wf_resampled = np.reshape(a=sig_wf, newshape=(points_resampled, resampling_factor))

    return [np.mean(new_wf_resampled, axis=1), sig_time[0::resampling_factor]]


# todo: doc string return type is wrong, consider object for return type
def resampled_power_per_band_no_hop(sig_wf: np.array,
                                    sig_time: np.array,
                                    power_tfr: np.array,
                                    resampling_factor: int):
    """
    Resamples a time series and the wavelet tfr time dimension
    Assume len(sig_time) is a power of two
    Assume len(resampling_factor) is a power of two

    :param sig_wf: audio waveform
    :param sig_time: audio timestamps in seconds, same dimensions as sig
    :param power_tfr: time-frequency representation with same number of columns (time) as sig
    :param resampling_factor: downsampling factor

    :return: rms_sig_wf, rms_sig_time_s
    """
    var_sig = (sig_wf - np.mean(sig_wf))**2

    if resampling_factor <= 1:
        print('No Resampling/Averaging!')
        exit()
    # Variance over the signal waveform, all bands
    # Number of rows (frequency)
    number_bands = power_tfr.shape[0]
    points_resampled: int = int(np.round((len(sig_wf)/resampling_factor)))
    var_wf_resampled = np.reshape(a=var_sig, newshape=(points_resampled, resampling_factor))

    var_sig_wf_mean = np.mean(var_wf_resampled, axis=1)
    var_sig_wf_max = np.max(var_wf_resampled, axis=1)
    var_sig_wf_min = np.min(var_wf_resampled, axis=1)

    # Reshape TFR
    var_tfr_resampled = np.reshape(a=power_tfr, newshape=(number_bands, points_resampled, resampling_factor))
    print(f'var_tfr_resampled.shape: {var_tfr_resampled.shape}')
    var_tfr_mean = np.mean(var_tfr_resampled, axis=2)
    var_tfr_max = np.max(var_tfr_resampled, axis=2)
    var_tfr_min = np.min(var_tfr_resampled, axis=2)

    # Resampled signal time
    var_sig_time_s = sig_time[0::resampling_factor]

    return [var_sig_time_s, var_sig_wf_mean, var_sig_wf_max, var_sig_wf_min, var_tfr_mean, var_tfr_max, var_tfr_min]


def resample_uneven_signal(sig_wf: np.ndarray,
                           sig_epoch_s: np.ndarray,
                           sample_rate_new_hz: float = None):
    """
    Uses interpolation to resample an unevenly sampled signal

    :param sig_wf:
    :param sig_epoch_s:
    :param sample_rate_new_hz:
    :return: resampled signal and timestamps
    """
    if sample_rate_new_hz is None:
        # Round up
        sample_rate_new_hz = np.ceil(1 / np.mean(np.diff(sig_epoch_s)))
    sig_new_epoch_s = np.arange(sig_epoch_s[0], sig_epoch_s[-1], 1 / sample_rate_new_hz)
    f = interpolate.interp1d(sig_epoch_s, sig_wf)
    return f(sig_new_epoch_s), sig_new_epoch_s


def upsample_fourier(sig_wf: np.ndarray,
                     sig_sample_rate_hz: float,
                     new_sample_rate_hz: float = 8000.) -> np.ndarray:
    """
    Up-sample the Fourier way.

    :param sig_wf: input signal waveform, reasonably well preprocessed
    :param sig_sample_rate_hz: signal sample rate
    :param new_sample_rate_hz: resampling sample rate in Hz, default 8000
    :return: resampled signal
    """
    return signal.resample(x=sig_wf, num=int(len(sig_wf) * new_sample_rate_hz / sig_sample_rate_hz))
