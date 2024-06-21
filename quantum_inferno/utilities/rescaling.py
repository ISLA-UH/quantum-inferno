"""
A set of functions to rescale data.
"""

from enum import Enum
import numpy as np
from quantum_inferno.scales_dyadic import get_epsilon


class DataScaleType(Enum):
    AMP: str = "amplitude"  # Provided data is in amplitude (waveform, etc.)
    POW: str = "power"  # Provided data is in power (psd, etc.)


def to_log2_with_epsilon(x: np.ndarray or float) -> np.ndarray or float:
    """
    :param x: data or value to rescale
    :return: rescaled data or value
    """
    return np.log2(x + get_epsilon())


def is_power_of_two(n: int) -> bool:
    """
    :param n: value to check
    :return True if n is positive and a power of 2, False otherwise
    """
    return n > 0 and not (n & (n - 1))


# TODO: would reference option to be min max etc be better rather than setting reference manually?
def to_decibel_with_epsilon(
    x: np.ndarray or float, reference: float = 1.0, scaling: DataScaleType = DataScaleType.AMP
) -> np.ndarray or float:
    """
    Convert data to decibels with epsilon added to avoid log(0) errors.
    :param x: data or value to rescale
    :param reference: reference value for the decibel scaling (default is None)
    :param scaling: the scaling type of the data (default is amplitude)
    :return:
    """
    if scaling == DataScaleType.POW:
        return 10 * np.log10(np.abs(x / reference) + get_epsilon())
    else:
        return 20 * np.log10(np.abs(x / reference) + get_epsilon())
