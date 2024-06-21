"""
Methods for creating windows.
"""

import numpy as np
from scipy.signal import windows


def get_tukey(array: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a symmetric Tukey window (AKA: tapered cosine window) with same shape as input array.
    Note: alpha of 0 is a rectangular window, 1 is a Hann window.
    :param array: input array to get shape from
    :param alpha: fraction of the window inside the cosine tapered window, shared between the head and tail
    """
    return windows.tukey(M=np.size(array), alpha=alpha, sym=True)


def get_tukey_by_buffer_num(array: np.ndarray, taper_num: int, alpha: float = 0.5) -> np.ndarray:
    """
    Create window with the same shape as input array, tukey tapered on the head and tail with ones in the middle.

    Note: alpha of 0 is a rectangular window, 1 is a Hann window.
    :param array: input array to get shape from
    :param taper_num: number of points to taper on each side
    :param alpha: fraction of the window inside the cosine tapered window, shared between the head and tail
    """
    if len(array) < taper_num * 2:
        print(f"Warning: array length {len(array)} is less than taper_num {taper_num * 2}. Using full array length.")
        return get_tukey(array, alpha=alpha)

    tukey = windows.tukey(taper_num * 2, alpha=alpha, sym=True)
    middle_ones = np.ones(len(array) - taper_num * 2)
    return np.concatenate((tukey[:taper_num], middle_ones, tukey[taper_num:]))


def get_tukey_by_buffer_s(array: np.ndarray, taper_s: float, sample_rate_hz: float, alpha: float = 0.5) -> np.ndarray:
    """
    Create window with the same shape as input array, tukey tapered on the head and tail with ones in the middle.

    Note: alpha of 0 is a rectangular window, 1 is a Hann window.
    :param array: input array to get shape from
    :param taper_s: duration of taper in seconds
    :param sample_rate_hz: sample rate in Hz
    :param alpha: fraction of the window inside the cosine tapered window, shared between the head and tail
    """
    taper_num = int(taper_s * sample_rate_hz)
    return get_tukey_by_buffer_num(array, taper_num, alpha=alpha)
