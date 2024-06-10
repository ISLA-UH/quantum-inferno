"""
A set of functions to pick key portions of a signal.
"""

from enum import Enum
from typing import Tuple
import numpy as np
from scipy import signal
from quantum_inferno.scales_dyadic import get_epsilon as EPSILON


class ExtractionType(Enum):
    ARGMAX: str = "argmax"  # Extract signal using the largest absolute value
    SIGMAX: str = "sigmax"  # fancier signal picker, from POSITIVE max #TODO: reword
    BITMAX: str = "bitmax"  # fancier signal picker, from ABSOLUTE max #TODO: reword
