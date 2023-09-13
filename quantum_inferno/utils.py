"""
This module contains general utilities that can work with values containing nans.
TODO: Break up into smaller modules
"""

import os
from enum import Enum
from typing import Tuple, Union
import numpy as np
from scipy import interpolate, signal
from scipy.integrate import cumulative_trapezoid
from quantum_inferno.scales_dyadic import EPSILON64 as EPSILON


""" Directory check & make """


def checkmake_dir(dir_name: str):
    """
    Check if the dir_name exists; if not, make it
    :param dir_name: directory name to check for
    """
    existing_dir: bool = os.path.isdir(dir_name)
    if not existing_dir:
        os.makedirs(dir_name)


""" Logical power of two flag """


def is_power_of_two(n: int) -> bool:
    """
    :param n: value to check
    :return True if n is positive and a power of 2, False otherwise
    """
    return n > 0 and not (n & (n - 1))


""" Time/Sample Duration Utils """


def duration_points(sample_rate_hz: float, time_s: float) -> Tuple[int, int, int]:
    """
    Compute number of points

    :param sample_rate_hz: sample rate in Hz
    :param time_s: duration, period, or scale of time in seconds
    :return: number of points, floor and ceiling of log2 of number of points
    """
    points_float: float = sample_rate_hz * time_s
    return int(points_float), int(np.floor(np.log2(points_float))), int(np.ceil(np.log2(points_float)))


def duration_ceil(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute ceiling of the number of points, and convert to seconds

    :param sample_rate_hz: sample rate in Hz
    :param time_s: duration, period, or scale of time in seconds
    :return: ceil of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_ceil_log2: int = int(np.ceil(np.log2(sample_rate_hz * time_s)))
    points_ceil_pow2: int = 2**points_ceil_log2
    return points_ceil_log2, points_ceil_pow2, float(points_ceil_pow2 / sample_rate_hz)


def duration_floor(sample_rate_hz: float, time_s: float) -> Tuple[int, int, float]:
    """
    Compute floor of the number of points, and convert to seconds

    :param sample_rate_hz: sample rate in Hz
    :param time_s: duration, period, or scale of time in seconds
    :return: floor of log 2 of number of points, power of two number of points, corresponding time in s
    """
    points_floor_log2: int = int(np.floor(np.log2(sample_rate_hz * time_s)))
    points_floor_pow2: int = 2**points_floor_log2
    return points_floor_log2, points_floor_pow2, float(points_floor_pow2 / sample_rate_hz)


""" Sampling Utils """


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


def taper_tukey(sig_wf_or_time: np.ndarray,
                fraction_cosine: float) -> np.ndarray:
    """
    Constructs a symmetric Tukey window with the same dimensions as a time or signal numpy array.
    fraction_cosine = 0 is a rectangular window, 1 is a Hann window
    todo: is fraction cosine a scale and what are its limits

    :param sig_wf_or_time: input signal or time
    :param fraction_cosine: fraction of the window inside the cosine tapered window, shared between the head and tail
    :return: tukey taper window amplitude
    """
    return signal.windows.tukey(M=np.size(sig_wf_or_time), alpha=fraction_cosine, sym=True)


# todo: this is in synthetics.py line 300
# Integrals and derivatives
def integrate_cumtrapz(timestamps_s: np.ndarray,
                       sensor_wf: np.ndarray,
                       initial_value: float = 0) -> np.ndarray:
    """
    Cumulative trapezoid integration using scipy.integrate.cumulative_trapezoid
    Initiated by Kei 2021 06, work in progress. See blast_derivative_integral for validation.

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input (default is 0)
    :return: integrated data with the same length as the input
    """
    return cumulative_trapezoid(x=timestamps_s, y=sensor_wf, initial=initial_value)


def derivative_gradient(timestamps_s: np.ndarray,
                        sensor_wf: np.ndarray) -> np.ndarray:
    """
    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input
    """
    return np.gradient(sensor_wf, timestamps_s)


def derivative_diff(timestamps_s: np.ndarray,
                    sensor_wf: np.ndarray) -> np.ndarray:
    """
    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input. Hold/repeat last value
    """
    derivative_data0 = np.diff(sensor_wf)/np.diff(timestamps_s)
    derivative_data = np.append(derivative_data0, derivative_data0[-1])

    return derivative_data


class ExtractionType(Enum):
    """
    Enumeration of valid extraction types.

    ARGMAX = max of the absolute value of the signal

    SIGMAX = fancier signal picker, from POSITIVE max

    BITMAX = fancier signal picker, from ABSOLUTE max
    """
    ARGMAX: str = "argmax"
    SIGMAX: str = "sigmax"
    BITMAX: str = "bitmax"


def sig_extract(
        sig: np.ndarray,
        time_epoch_s: np.ndarray,
        intro_s: float,
        outro_s: float,
        pick_bits_below_max: float = 1.,
        pick_time_interval_s: float = 1.,
        extract_type: ExtractionType = ExtractionType.ARGMAX
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract signal and time relative to reference index

    :param sig: input signal
    :param time_epoch_s: epoch time of signal in seconds
    :param intro_s: time before pick
    :param outro_s: time after pick
    :param pick_bits_below_max: the pick threshold in bits below max
    :param pick_time_interval_s: pick time interval between adjacent max point
    :param extract_type: Type of extraction, see ExtractionType class.  Default to ARGMAX.
    :return: extracted signal np.ndarray, extracted signal timestamps np.ndarray, and pick time
    """
    if extract_type == ExtractionType.SIGMAX:
        # First pick
        pick_func = picker_signal_max_index(sig_sample_rate_hz=1./np.mean(np.diff(time_epoch_s)),
                                            sig=sig, bits_pick=pick_bits_below_max,
                                            time_interval_s=pick_time_interval_s)[0]
        pick_fun2 = picker_signal_finder(sig=sig, sig_sample_rate_hz=1./np.mean(np.diff(time_epoch_s)),
                                         bits_pick=pick_bits_below_max, mode="max",
                                         time_interval_s=pick_time_interval_s)[0]
        assert(pick_func == pick_fun2)
    elif extract_type == ExtractionType.BITMAX:
        # First pick
        pick_func = picker_signal_bit_index(sig_sample_rate_hz=1./np.mean(np.diff(time_epoch_s)),
                                            sig=sig, bits_pick=pick_bits_below_max,
                                            time_interval_s=pick_time_interval_s)[0]
        pick_fun2 = picker_signal_finder(sig=sig, sig_sample_rate_hz=1./np.mean(np.diff(time_epoch_s)),
                                         bits_pick=pick_bits_below_max, mode="bit",
                                         time_interval_s=pick_time_interval_s)[0]
        assert(pick_func == pick_fun2)
    else:
        if extract_type != ExtractionType.ARGMAX:
            print('Unexpected extraction type to sig_extract, return max')
        # Max pick
        pick_func = np.argmax(np.abs(sig))

    pick_time_epoch_s = time_epoch_s[pick_func]
    intro_index = np.argmin(np.abs(time_epoch_s - (pick_time_epoch_s - intro_s)))
    outro_index = np.argmin(np.abs(time_epoch_s - (pick_time_epoch_s + outro_s)))

    return sig[intro_index: outro_index], time_epoch_s[intro_index: outro_index], pick_time_epoch_s


def sig_frame(sig: np.ndarray,
              time_epoch_s: np.ndarray,
              epoch_s_start: float,
              epoch_s_stop: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame one-component signal within start and stop epoch times

    :param sig: input signal
    :param time_epoch_s: input epoch time in seconds
    :param epoch_s_start: start epoch time
    :param epoch_s_stop: stop epoch time
    :return: truncated time series and timestamps
    """
    intro_index = np.argmin(np.abs(time_epoch_s - epoch_s_start))
    outro_index = np.argmin(np.abs(time_epoch_s - epoch_s_stop))

    return sig[intro_index: outro_index], time_epoch_s[intro_index: outro_index]


def sig3c_frame(sig3c: np.ndarray,
                time_epoch_s: np.ndarray,
                epoch_s_start: float,
                epoch_s_stop: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame three-component signal within start and stop epoch times

    :param sig3c: input signal with three components
    :param time_epoch_s: input epoch time in seconds
    :param epoch_s_start: start epoch time
    :param epoch_s_stop: stop epoch time
    :return: truncated time series and timestamps
    """
    intro_index = np.argmin(np.abs(time_epoch_s - epoch_s_start))
    outro_index = np.argmin(np.abs(time_epoch_s - epoch_s_stop))

    return sig3c[:, intro_index: outro_index], time_epoch_s[intro_index: outro_index]


def dbepsilon(x: np.ndarray) -> np.ndarray:
    """
    :param x: time series
    :return: the absolute value of a time series as dB
    """
    return 10 * np.log10(np.abs(x ** 2) + EPSILON)


def dbepsilon_max(x: np.ndarray) -> float:
    """
    :param x: time series
    :return: max of the absolute value of a time series to dB
    """
    return 10 * np.log10(np.max(np.abs(x ** 2)) + EPSILON)


def log2epsilon(x: np.ndarray) -> np.ndarray:
    """
    :param x: time series or fft - not power
    :return: log 2 of the absolute value of linear amplitude, with EPSILON to avoid singularities
    """
    return np.log2(np.abs(x) + EPSILON)


def log2epsilon_max(x: np.ndarray) -> float:
    """
    :param x: time series or fft - not power
    :return: max of the log 2 of absolute value of linear amplitude, with EPSILON to avoid singularities
    """
    return np.max(np.log2(np.abs(x) + EPSILON))


"""
Picker modules
"""


def picker_signal_finder(sig: np.array,
                         sig_sample_rate_hz: float,
                         bits_pick: float,
                         time_interval_s: float,
                         mode: str = "max") -> np.array:
    """
    computes picker index for positive (option "max") or absolute (option "bit") max of a signal.

    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection threshold in bits loss
    :param time_interval_s: min time interval between events
    :param mode: "max" or "bit", if not either, uses max.  Default "max"
    :return: picker index for the signal
    """
    height_min = log2epsilon_max(sig) - bits_pick if mode == "bit" else np.max(sig) - 2 ** bits_pick
    time_index_pick, _ = signal.find_peaks(log2epsilon(sig) if mode == "bit" else sig,
                                           height=height_min,
                                           distance=int(time_interval_s * sig_sample_rate_hz))
    return time_index_pick


def picker_signal_max_index(sig: np.array,
                            sig_sample_rate_hz: float,
                            bits_pick: float,
                            time_interval_s: float) -> np.array:
    """
    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection threshold in bits loss
    :param time_interval_s: min time interval between events
    :return: picker index for the POSITIVE max of a signal
    """
    # Compute the distance
    distance_points = int(time_interval_s * sig_sample_rate_hz)
    height_min = np.max(sig) - 2 ** bits_pick
    time_index_pick, _ = signal.find_peaks(sig, height=height_min, distance=distance_points)

    return time_index_pick


def picker_signal_bit_index(sig: np.array,
                            sig_sample_rate_hz: float,
                            bits_pick: float,
                            time_interval_s: float) -> np.ndarray:
    """
    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection threshold in db loss
    :param time_interval_s: min time interval between events
    :return: picker index from the ABSOLUTE max in bits
    """
    # Compute the distance
    distance_points = int(time_interval_s * sig_sample_rate_hz)
    height_min = log2epsilon_max(sig) - bits_pick
    index_pick, _ = signal.find_peaks(log2epsilon(sig), height=height_min, distance=distance_points)

    return index_pick


def picker_comb(sig_pick: np.ndarray,
                index_pick: np.ndarray) -> np.ndarray:
    """
    Constructs a comb function from the picks

    :param sig_pick: 1D record corresponding to the picks
    :param index_pick: indexes for the picks
    :return: comb function with unit amplitude
    """
    comb = np.zeros(sig_pick.shape)
    comb[index_pick] = np.ones(index_pick.shape)
    return comb


"""
Matrix transformations
"""


def columns_ops(sxx: np.ndarray, mode: str = "sum") -> np.ndarray:
    """
    Perform the operation specified over all columns in a 1D or 2D array.
    Operations allowed are: "sum" or "mean".

    :param sxx: input vector or matrix
    :param mode: "sum" or "mean".  fails if not either option.  Default "sum"
    :return: ndarray with appropriate operation.
    """
    if not isinstance(sxx, np.ndarray):
        raise TypeError('Input must be array.')
    elif len(sxx) == 0:
        raise ValueError('Cannot compute on empty array.')

    if mode == "sum":
        func = np.sum
    elif mode == "mean":
        func = np.mean
    else:
        raise ValueError(f"Unknown array operation {mode} requested.")

    if np.isnan(sxx).any() or np.isinf(sxx).any():
        sxx = np.nan_to_num(sxx)

    if len(sxx.shape) == 1:
        opd = func(sxx, axis=0)
    elif len(sxx.shape) == 2:
        opd = func(sxx, axis=1)
    else:
        raise TypeError(f'Cannot handle an array of shape {sxx.shape}.')

    return opd


def sum_columns(sxx: np.ndarray) -> np.ndarray:
    """
    Sum over all the columns in a 1D or 2D array

    :param sxx: input vector or matrix
    :return: ndarray with sum
    """
    return columns_ops(sxx, "sum")


def mean_columns(sxx: np.ndarray) -> np.ndarray:
    """
    Compute the mean of the columns in a 1D or 2D array

    :param sxx: input vector or matrix
    :return: ndarray with mean
    """
    return columns_ops(sxx, "mean")


def just_tile_d0(d0_array1d_in: Union[float, np.ndarray], d0d1_shape: tuple) -> np.ndarray:
    """
    Constructs tiled array from 1D array to the shape specified by d0d1_shape

    :param d0_array1d_in: 1D scalar or vector with dimension d0
    :param d0d1_shape: Tuple with output array shape, first dimension must match d0
    :return: ndarray
    """
    # TODO: Revisit - too many silly options
    if len(d0d1_shape) == 1:
        # Scalar to array, special case
        tiled_matrix = np.tile(d0_array1d_in, (d0d1_shape[0]))
    elif len(d0d1_shape) == 2 and d0d1_shape[0] == len(d0_array1d_in):
        tiled_matrix = np.tile(d0_array1d_in, (d0d1_shape[1], 1)).T
    else:
        raise TypeError(f'Cannot handle an array of shape {d0_array1d_in.shape}.')

    return tiled_matrix


def just_tile_d1(d1_array1d_in: Union[float, np.ndarray], d0d1_shape: tuple) -> np.ndarray:
    """
    Create array of repeated values with dimensions that match those of energy array
    Useful to multiply time-dependent values to frequency-time matrices

    :param d1_array1d_in: 1D input vector, nominally row time multipliers
    :param d0d1_shape: 2D array, second dimension should be that same as d1
    :return: array with matching values
    """
    if len(d0d1_shape) == 1:
        tiled_matrix = np.tile(d1_array1d_in, (d0d1_shape[0]))
    elif len(d0d1_shape) == 2 and d0d1_shape[1] == len(d1_array1d_in):
        tiled_matrix = np.tile(d1_array1d_in, (d0d1_shape[0], 1))
    else:
        raise TypeError(f'Cannot handle an array of shape {d1_array1d_in.shape}.')

    return tiled_matrix


def sum_tile(sxx: np.ndarray) -> np.ndarray:
    """
    Compute the sum of the columns in a 1D or 2D array and then re-tile to the original size

    :param sxx: input vector or matrix
    :return: ndarray of sum
    """
    sum_c = sum_columns(sxx)

    # create array of repeated values of PSD with dimensions that match those of energy array
    if len(sxx.shape) == 1:
        sum_c_matrix = np.tile(sum_c, (sxx.shape[0]))
    elif len(sxx.shape) == 2:
        sum_c_matrix = np.tile(sum_c, (sxx.shape[1], 1)).T
    else:
        raise TypeError(f'Cannot handle an array of shape {sxx.shape}.')

    return sum_c_matrix


def mean_tile(sxx: np.ndarray, shape_out: np.ndarray) -> np.ndarray:
    """
    Compute the mean of the columns in a 1D or 2D array and then re-tile to the original size

    :param sxx: input vector or matrix
    :param shape_out: shape of output vector or matrix
    :return: ndarray of mean
    """
    sum_c = mean_columns(sxx)

    # create array of repeated values of PSD with dimensions that match those of energy array
    if len(shape_out) == 1:
        sum_c_matrix = np.tile(sum_c, (shape_out[0]))
    elif len(shape_out) == 2:
        sum_c_matrix = np.tile(sum_c, (shape_out[1], 1)).T
    else:
        raise TypeError(f'Cannot handle an array of shape {sxx.shape}.')

    return sum_c_matrix


def d0tile_x_d0d1(d0: Union[float, np.ndarray], d0d1: np.ndarray) -> np.ndarray:
    """
    Create array of repeated values with dimensions that match those of energy array
    Useful to multiply frequency-dependent values to frequency-time matrices

    :param d0: 1D input vector, nominally column frequency/scale multipliers
    :param d0d1: 2D array, first dimension should be that same as d1
    :return: array with matching values
    """
    # TODO: Add error catch, and firm up use case
    shape_out = d0d1.shape

    if len(shape_out) == 1:
        d0_matrix = np.tile(d0, (shape_out[0]))
    elif len(shape_out) == 2:
        d0_matrix = np.tile(d0, (shape_out[1], 1)).T
    else:
        raise TypeError(f'Cannot handle an array of shape {d0.shape}.')

    if d0_matrix.shape == d0d1.shape:
        d0_x_d0d1 = d0_matrix * d0d1
    else:
        raise TypeError(f'Cannot handle an array of shape {d0.shape}.')
    return d0_x_d0d1


def d1tile_x_d0d1(d1: Union[float, np.ndarray],
                  d0d1: np.ndarray) -> np.ndarray:
    """
    Create array of repeated values with dimensions that match those of energy array
    Useful to multiply time-dependent values to frequency-time matrices

    :param d1: 1D input vector, nominally row time multipliers
    :param d0d1: 2D array, second dimension should be that same as d1
    :return: array with matching values
    """
    shape_out = d0d1.shape

    if len(shape_out) == 1:
        d1_matrix = np.tile(d1, (shape_out[0]))
    elif len(shape_out) == 2:
        # TODO: Test
        d1_matrix = np.tile(d1, (shape_out[0], 1))
    else:
        raise TypeError(f'Cannot handle an array of shape {d1.shape}.')

    if d1_matrix.shape == d0d1.shape:
        d1_x_d0d1 = d1_matrix * d0d1
    else:
        raise TypeError(f'Cannot handle an array of shape {d1.shape}.')
    return d1_x_d0d1


def decimate_array(sig_wf: np.array, downsampling_factor: int) -> np.ndarray:
    """
    Decimate data and timestamps for an individual station
    All signals MUST have the same sample rate
    todo: what if they don't

    :param sig_wf: signal waveform
    :param downsampling_factor: the down-sampling factor
    :return: decimated data as numpy array
    """
    return signal.decimate(x=sig_wf, q=downsampling_factor, axis=1, zero_phase=True)
