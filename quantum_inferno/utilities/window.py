"""
Methods for creating windows.
"""

import numpy as np
from scipy.signal import windows

from quantum_inferno import qi_debugger

def taper_power_correction(taper_window: np.ndarray) -> any:
    """
    Correct the spectral power.
    :param taper_window: taper for the amplitude correction
    """
    return np.sum(np.abs(taper_window) ** 2)


def get_tukey(array: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a symmetric Tukey window (AKA: tapered cosine window) with same shape as input array.
    Note: alpha of 0 is a rectangular window, 1 is a Hann window.
    :param array: input array to get shape from
    :param alpha: fraction of the window inside the cosine tapered window, shared between the head and tail
    """
    return windows.tukey(M=np.size(array), alpha=alpha, sym=True)

