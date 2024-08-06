"""
Utilities for matrix operations.
"""

import numpy as np
from enum import Enum


class MatrixAxis(Enum):
    ROW = "row"  # d1
    COLUMN = "column"  # d0


def array_from_number(number: float or int, shape: tuple) -> np.ndarray:
    """
    Create an array with repeated values from a given float.

    :param number: float or int value to repeat
    :param shape: shape of the array
    :return: 1D or 2D array
    """
    return np.full(shape, number)


def n_tile_array(array: np.ndarray, n: int, axis: MatrixAxis) -> np.ndarray:
    """
    Tile a 1D array n times by repeating values in a column wise or row wise direction.
    If row wise, the output array will have n rows and the same number of columns as the input array.
    If column wise, the output array will have n columns and the same number of rows as the input array.

    :param array: 1D array
    :param n: number of times to tile
    :param axis: direction to tile
    :return: 2D array with repeated values
    """
    if n < 1:
        print("Warning: n must be greater than 1. Returning original array.")
        return array

    if axis == MatrixAxis.ROW:
        return np.tile(array, (n, 1))
    elif axis == MatrixAxis.COLUMN:
        return np.reshape(np.tile(array, (1, n)), (-1, n), order="F")
    else:
        raise ValueError("Invalid direction. Must be either ROW or COLUMN.")


def tile_array_to_shape(array: np.ndarray, shape: tuple, axis: MatrixAxis = None) -> np.ndarray:
    """
    Tile an array with a given shape by repeating values in a column wise or row wise direction to match.
    If axis is not specified, the direction will be determined by the shape.
    If row wise, the output array will have n rows and the same number of columns as the input array.
    If column wise, the output array will have n columns and the same number of rows as the input array.

    :param array: 1D array
    :param shape: shape of the output array
    :param axis: direction to tile
    :return: 2D array with repeated values
    """
    if shape[0] == 1 or shape[1] == 1:
        print("Warning: shape must be greater than 1. Returning original array.")
        return array

    if axis is None:
        if shape[0] == shape[1] and shape[0] == array.shape[0]:
            print("If shape is square, defaulting to row direction.")
            axis = MatrixAxis.ROW
        elif shape[0] == array.shape[0]:
            axis = MatrixAxis.ROW
        elif shape[1] == array.shape[0]:
            axis = MatrixAxis.COLUMN
        elif array.ndim == 1:
            print("Input array is 1D, Defaulting to row direction.")
            axis = MatrixAxis.ROW
        else:
            raise ValueError("Invalid shape. Must be a multiple of the input array.")

    if axis == MatrixAxis.ROW and shape[0] == array.shape[0]:
        print("row")
        return np.tile(array, (shape[1], 1))
    elif axis == MatrixAxis.COLUMN and shape[1] == array.shape[0]:
        print("column")
        return np.reshape(np.tile(array, (1, shape[0])), (-1, shape[0]), order="F")
    else:
        raise ValueError("Invalid direction or shape. Must be either ROW or COLUMN and be a multiple of input array.")


# TODO: work on simplifying these 3 functions (Legacy code included in styx functions to be cleaned up)
def d0tile_x_d0d1(d0: float or np.ndarray, d0d1: np.ndarray) -> np.ndarray:
    """
    Create array of repeated values with dimensions that match those of energy array
    Useful to multiply frequency-dependent values to frequency-time matrices

    :param d0: 1D input vector, nominally column frequency/scale multipliers
    :param d0d1: 2D array, first dimension should be that same as d1
    :return: array with matching values
    """
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


def d1tile_x_d0d1(d1: float or np.ndarray, d0d1: np.ndarray) -> np.ndarray:
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
        d1_matrix = np.tile(d1, (shape_out[0], 1))
    else:
        raise TypeError(f"Cannot handle an array of shape {d1.shape}.")

    if d1_matrix.shape == d0d1.shape:
        d1_x_d0d1 = d1_matrix * d0d1
    else:
        raise TypeError(f"Cannot handle an array of shape {d1.shape}.")
    return d1_x_d0d1


def just_tile_d1(d1_array1d_in: float or np.ndarray, d0d1_shape: tuple) -> np.ndarray:
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
