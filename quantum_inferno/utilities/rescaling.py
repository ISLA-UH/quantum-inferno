"""
A set of functions to rescale data.
"""

from enum import Enum
from typing import Tuple
import numpy as np
from scipy import signal
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


# TODO: Function to find the maximum of the data may be redundant
def log2_with_epsilon_max(x: np.ndarray) -> float:
    """
    :param x: data to find the maximum of in log2 space
    :return: maximum of the data in log2 space
    """
    return np.max(to_log2_with_epsilon(x))


# TODO: would reference option to be min max etc be better?
def to_decibel_with_epsilon(
    x: np.ndarray or float, reference: float = 1.0, scaling: DataScaleType = DataScaleType.AMP
) -> np.ndarray or float:
    """
    Convert data to decibels with epsilon added to avoid log(0) errors.
    :param x: data or value to rescale
    :param reference: reference value for the decibel scaling (default is None)
    :param scaling: the type of the data (default is amplitude)
    :return:
    """
    if scaling == DataScaleType.POW:
        return 10 * np.log10(np.abs(x / reference) + get_epsilon())
    else:
        return 20 * np.log10(np.abs(x / reference) + get_epsilon())


# TODO: Function to find the maximum of the data may be redundant
def decibel_with_epsilon_max(x: np.ndarray) -> float:
    """
    :param x: data to find the maximum of in dB space
    :return: maximum of the data in dB space
    """
    return np.max(to_decibel_with_epsilon(x))
