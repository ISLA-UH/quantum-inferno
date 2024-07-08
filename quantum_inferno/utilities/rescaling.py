"""
A set of functions to rescale data.
"""
from typing import Union
import numpy as np
from quantum_inferno.scales_dyadic import get_epsilon


DATA_SCALE_TYPE = ["amplitude", "power"]


def to_log2_with_epsilon(x: Union[np.ndarray, float, list]) -> Union[np.ndarray, float]:
    """
    Convert the absolute value of the data to log2 with epsilon added to avoid log(0) and log(<0) errors.

    :param x: data or value to rescale
    :return: rescaled data or value
    """
    return np.log2(np.abs(x) + get_epsilon())


def is_power_of_two(n: int) -> bool:
    """
    :param n: value to check
    :return True if n is positive and a power of 2, False otherwise
    """
    return n > 0 and not (n & (n - 1))


def to_decibel_with_epsilon(
    x: Union[np.ndarray, float, list], reference: float = 1.0, input_scaling: str = "amplitude"
) -> Union[np.ndarray, float]:
    """
    Convert data to decibels with epsilon added to avoid log(0) errors.

    :param x: data or value to rescale
    :param reference: reference value for the decibel scaling (default is None)
    :param input_scaling: the scaling type of the data (default is amplitude)
    :return: rescaled data or value as decibels
    """
    if input_scaling not in DATA_SCALE_TYPE:
        print("Invalid input scaling type.  Defaulting to amplitude.")
        input_scaling = "amplitude"
    scale_val = 10 if input_scaling == "power" else 20

    if reference == 0:
        raise ValueError("Reference value cannot be zero.")
    elif reference == 1:
        return scale_val * np.log10(np.abs(x) + get_epsilon())
    else:
        return scale_val * np.log10(np.abs(x) + get_epsilon()) - scale_val * np.log10(reference + get_epsilon())
