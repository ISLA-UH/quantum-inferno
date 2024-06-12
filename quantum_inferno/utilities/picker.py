"""
A set of functions to pick key portions of a signal.
"""

from enum import Enum
from typing import Tuple
import numpy as np
from scipy import signal
from quantum_inferno.scales_dyadic import get_epsilon as EPSILON
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon


class ExtractionType(Enum):
    ARGMAX: str = "argmax"  # Extract signal using the largest absolute value
    SIGMAX: str = "sigmax"  # fancier signal picker, from POSITIVE max #TODO: reword
    BITMAX: str = "bitmax"  # fancier signal picker, from ABSOLUTE max #TODO: reword


class ScalingType(Enum):
    BITS: str = "bits"  # data is in bits
    AMPS: str = "amplitude"  # data is in amplitude


# TODO: Figure what is wanted for find peaks other than a wrapper for find_peaks
def find_peaks_in_bits(
    waveform_data: np.ndarray,
    waveform_sample_rate_hz: float,
    bit_height_from_max: float,
    time_difference: float,
    waveform_scaling: ScalingType = ScalingType.AMPS,
) -> np.ndarray:
    """
    Finds peak of waveform in bits
    :param waveform_data:
    :param waveform_sample_rate_hz:
    :param bit_height_from_max:
    :param time_difference:
    :param waveform_scaling:
    :return:
    """
