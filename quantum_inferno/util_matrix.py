"""
This module contains methods to update matrices and vectors
"""
from typing import Union
import numpy as np
from scipy import signal


def columns_ops(sxx: np.ndarray, mode: str = "sum") -> np.ndarray:
    """
    Perform the operation specified over all columns in a 1D or 2D array.
    Operations allowed are: "sum" or "mean".

    :param sxx: input vector or matrix
    :param mode: "sum" or "mean".  fails if not either option.  Default "sum"
    :return: ndarray with appropriate operation.
    """
    if not isinstance(sxx, np.ndarray):
        raise TypeError("Input must be array.")
    elif len(sxx) == 0:
        raise ValueError("Cannot compute on empty array.")

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
        raise TypeError(f"Cannot handle an array of shape {sxx.shape}.")

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
        raise TypeError(f"Cannot handle an array of shape {d0_array1d_in.shape}.")

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
        raise TypeError(f"Cannot handle an array of shape {d1_array1d_in.shape}.")

    return tiled_matrix


# can be used with other methods to tile the output to the original size
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
        raise TypeError(f"Cannot handle an array of shape {sxx.shape}.")

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
        raise TypeError(f"Cannot handle an array of shape {sxx.shape}.")

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
        raise TypeError(f"Cannot handle an array of shape {d0.shape}.")

    if d0_matrix.shape == d0d1.shape:
        d0_x_d0d1 = d0_matrix * d0d1
    else:
        raise TypeError(f"Cannot handle an array of shape {d0.shape}.")
    return d0_x_d0d1


def d1tile_x_d0d1(d1: Union[float, np.ndarray], d0d1: np.ndarray) -> np.ndarray:
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
        raise TypeError(f"Cannot handle an array of shape {d1.shape}.")

    if d1_matrix.shape == d0d1.shape:
        d1_x_d0d1 = d1_matrix * d0d1
    else:
        raise TypeError(f"Cannot handle an array of shape {d1.shape}.")
    return d1_x_d0d1


# added to sampling.py
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
