"""
A set of functions to rescale data.
"""

from enum import Enum
from typing import Tuple
import numpy as np
from scipy import signal
from quantum_inferno.scales_dyadic import get_epsilon as EPSILON


def to_log2_with_epsilon(x: np.ndarray or float) -> np.ndarray or float:
    """
    :param x: data or value to rescale
    :return: rescaled data or value
    """
    return np.log2(x + EPSILON)


# TODO: Function to find the maximum of the data may be redundant
def log2_with_epsilon_max(x: np.ndarray) -> float:
    """
    :param x: data to find the maximum of in log2 space
    :return: maximum of the data in log2 space
    """
    return np.max(to_log2_with_epsilon(x))


# TODO: would it be better to have options to set the reference for dB? Also option to take power or amplitude?
def to_decibel_with_epsilon(x: np.ndarray or float) -> np.ndarray or float:
    """
    :param x: amplitude data to rescale
    :return: rescaled data in dB
    """
    return 20 * np.log10(np.abs(x) + EPSILON)


def decibel_with_epsilon_max(x: np.ndarray) -> float:
    """
    :param x: data to find the maximum of in dB space
    :return: maximum of the data in dB space
    """
    return np.max(to_decibel_with_epsilon(x))
