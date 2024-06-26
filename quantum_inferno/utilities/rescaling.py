"""
A set of functions to rescale data.
"""
from typing import Union

from enum import Enum
import numpy as np

from quantum_inferno.scales_dyadic import get_epsilon


class DataScaleType(Enum):
    AMP: str = "amplitude"  # Provided data is in amplitude (waveform, etc.)
    POW: str = "power"  # Provided data is in power (psd, etc.)


# todo: apparently you can encounter invalid values while executing this function.  refer to test_picker test functions
# such as test_scale_signal_by_extraction_type_bit() for an example
def to_log2_with_epsilon(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
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
    x: Union[np.ndarray, float], reference: float = 1.0, scaling: DataScaleType = DataScaleType.AMP
) -> Union[np.ndarray, float]:
    """
    Convert data to decibels with epsilon added to avoid log(0) errors.

    :param x: data or value to rescale
    :param reference: reference value for the decibel scaling (default is None)
    :param scaling: the scaling type of the data (default is amplitude)
    :return: rescaled data or value as decibels
    """
    scale_val = 10 if scaling == DataScaleType.POW else 20
    return scale_val * np.log10(np.abs(x / reference) + get_epsilon())
