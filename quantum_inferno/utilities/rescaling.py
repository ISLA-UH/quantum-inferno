"""
A set of functions to rescale data.

"""
from typing import Iterable, Union
import numpy as np

from quantum_inferno import qi_debugger
from quantum_inferno.scales_dyadic import get_epsilon


DATA_SCALE_TYPE = ["amplitude", "power"]


def remove_nanmean(in_data: np.ndarray) -> np.ndarray:
    """
    Remove the mean of the input data.

    :param in_data: data to remove mean
    :return: data with mean removed
    """
    return in_data - np.nanmean(in_data)


def set_vals_to(in_data: np.ndarray, target_indices: Union[Iterable, int], new_val: float) -> np.ndarray:
    """
    Set all values in the specified indices of the input data to a new value.
    Raises an error if any are encountered.

    :param in_data: data to set values
    :param target_indices: indices to change
    :param new_val: new value to set
    :return: data with all values set to new value
    """
    repl_data = np.copy(in_data)
    np.put(repl_data, target_indices, new_val)
    return repl_data


def remove_dc_offset_and_nans(in_data: np.ndarray) -> np.ndarray:
    """
    Remove the DC offset and set NaNs to 0 from the input data.

    :param in_data: data to update
    :return: data with DC offset removed and NaNs replaced with 0
    """
    return set_vals_to(remove_nanmean(in_data), np.argwhere(np.isnan(in_data)), 0.)


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
        qi_debugger.add_message("Invalid input scaling type.  Defaulting to amplitude.")
        # print("Invalid input scaling type.  Defaulting to amplitude.")
        input_scaling = "amplitude"
    scale_val = 10 if input_scaling == "power" else 20

    if reference == 0:
        raise ValueError("Reference value cannot be zero.")
    elif reference == 1:
        return scale_val * np.log10(np.abs(x) + get_epsilon())
    else:
        return scale_val * np.log10(np.abs(x) + get_epsilon()) - scale_val * np.log10(reference + get_epsilon())
