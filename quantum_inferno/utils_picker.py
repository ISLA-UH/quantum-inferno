"""
This module contains methods used to select key portions of signals
"""
from enum import Enum
from typing import Tuple
import numpy as np
from scipy import signal
from quantum_inferno.scales_dyadic import EPSILON64 as EPSILON


class ExtractionType(Enum):
    """
    Enumeration of valid extraction types.

    ARGMAX = max of the absolute value of the signal

    SIGMAX = fancier signal picker, from POSITIVE max

    BITMAX = fancier signal picker, from ABSOLUTE max
    """

    ARGMAX: str = "argmax"
    SIGMAX: str = "sigmax"
    BITMAX: str = "bitmax"


def sig_extract(
    sig: np.ndarray,
    time_epoch_s: np.ndarray,
    intro_s: float,
    outro_s: float,
    pick_bits_below_max: float = 1.0,
    pick_time_interval_s: float = 1.0,
    extract_type: ExtractionType = ExtractionType.ARGMAX,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract signal and time relative to reference index

    :param sig: input signal
    :param time_epoch_s: epoch time of signal in seconds
    :param intro_s: time before pick
    :param outro_s: time after pick
    :param pick_bits_below_max: the pick threshold in bits below max
    :param pick_time_interval_s: pick time interval between adjacent max point
    :param extract_type: Type of extraction, see ExtractionType class.  Default to ARGMAX.
    :return: extracted signal np.ndarray, extracted signal timestamps np.ndarray, and pick time
    """
    if extract_type == ExtractionType.SIGMAX:
        # First pick
        pick_func = picker_signal_max_index(
            sig_sample_rate_hz=1.0 / np.mean(np.diff(time_epoch_s)),
            sig=sig,
            bits_pick=pick_bits_below_max,
            time_interval_s=pick_time_interval_s,
        )[0]
        pick_fun2 = picker_signal_finder(
            sig=sig,
            sig_sample_rate_hz=1.0 / np.mean(np.diff(time_epoch_s)),
            bits_pick=pick_bits_below_max,
            mode="max",
            time_interval_s=pick_time_interval_s,
        )[0]
        assert pick_func == pick_fun2
    elif extract_type == ExtractionType.BITMAX:
        # First pick
        pick_func = picker_signal_bit_index(
            sig_sample_rate_hz=1.0 / np.mean(np.diff(time_epoch_s)),
            sig=sig,
            bits_pick=pick_bits_below_max,
            time_interval_s=pick_time_interval_s,
        )[0]
        pick_fun2 = picker_signal_finder(
            sig=sig,
            sig_sample_rate_hz=1.0 / np.mean(np.diff(time_epoch_s)),
            bits_pick=pick_bits_below_max,
            mode="bit",
            time_interval_s=pick_time_interval_s,
        )[0]
        assert pick_func == pick_fun2
    else:
        if extract_type != ExtractionType.ARGMAX:
            print("Unexpected extraction type to sig_extract, return max")
        # Max pick
        pick_func = np.argmax(np.abs(sig))

    pick_time_epoch_s = time_epoch_s[pick_func]
    intro_index = np.argmin(np.abs(time_epoch_s - (pick_time_epoch_s - intro_s)))
    outro_index = np.argmin(np.abs(time_epoch_s - (pick_time_epoch_s + outro_s)))

    return sig[intro_index:outro_index], time_epoch_s[intro_index:outro_index], pick_time_epoch_s


def sig_frame(
    sig: np.ndarray, time_epoch_s: np.ndarray, epoch_s_start: float, epoch_s_stop: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame one-component signal within start and stop epoch times

    :param sig: input signal
    :param time_epoch_s: input epoch time in seconds
    :param epoch_s_start: start epoch time
    :param epoch_s_stop: stop epoch time
    :return: truncated time series and timestamps
    """
    intro_index = np.argmin(np.abs(time_epoch_s - epoch_s_start))
    outro_index = np.argmin(np.abs(time_epoch_s - epoch_s_stop))

    return sig[intro_index:outro_index], time_epoch_s[intro_index:outro_index]


def sig3c_frame(
    sig3c: np.ndarray, time_epoch_s: np.ndarray, epoch_s_start: float, epoch_s_stop: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame three-component signal within start and stop epoch times

    :param sig3c: input signal with three components
    :param time_epoch_s: input epoch time in seconds
    :param epoch_s_start: start epoch time
    :param epoch_s_stop: stop epoch time
    :return: truncated time series and timestamps
    """
    intro_index = np.argmin(np.abs(time_epoch_s - epoch_s_start))
    outro_index = np.argmin(np.abs(time_epoch_s - epoch_s_stop))

    return sig3c[:, intro_index:outro_index], time_epoch_s[intro_index:outro_index]


def dbepsilon(x: np.ndarray) -> np.ndarray:
    """
    :param x: time series
    :return: the absolute value of a time series as dB
    """
    return 10 * np.log10(np.abs(x ** 2) + EPSILON)


def dbepsilon_max(x: np.ndarray) -> float:
    """
    :param x: time series
    :return: max of the absolute value of a time series to dB
    """
    return 10 * np.log10(np.max(np.abs(x ** 2)) + EPSILON)


def log2epsilon(x: np.ndarray) -> np.ndarray:
    """
    :param x: time series or fft - not power
    :return: log 2 of the absolute value of linear amplitude, with EPSILON to avoid singularities
    """
    return np.log2(np.abs(x) + EPSILON)


def log2epsilon_max(x: np.ndarray) -> float:
    """
    :param x: time series or fft - not power
    :return: max of the log 2 of absolute value of linear amplitude, with EPSILON to avoid singularities
    """
    return np.max(np.log2(np.abs(x) + EPSILON))


def picker_signal_finder(
    sig: np.array, sig_sample_rate_hz: float, bits_pick: float, time_interval_s: float, mode: str = "max"
) -> np.array:
    """
    computes picker index for positive (option "max") or absolute (option "bit") max of a signal.

    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection threshold in bits loss
    :param time_interval_s: min time interval between events
    :param mode: "max" or "bit", if not either, uses max.  Default "max"
    :return: picker index for the signal
    """
    height_min = log2epsilon_max(sig) - bits_pick if mode == "bit" else np.max(sig) - 2 ** bits_pick
    time_index_pick, _ = signal.find_peaks(
        log2epsilon(sig) if mode == "bit" else sig,
        height=height_min,
        distance=int(time_interval_s * sig_sample_rate_hz),
    )
    return time_index_pick


# TODO: Both picker_signal_max_index and picker_signal_bit_index is already contained in picker_signal_finder
def picker_signal_max_index(
    sig: np.array, sig_sample_rate_hz: float, bits_pick: float, time_interval_s: float
) -> np.array:
    """
    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection threshold in bits loss
    :param time_interval_s: min time interval between events
    :return: picker index for the POSITIVE max of a signal
    """
    # Compute the distance
    distance_points = int(time_interval_s * sig_sample_rate_hz)
    height_min = np.max(sig) - 2 ** bits_pick
    time_index_pick, _ = signal.find_peaks(sig, height=height_min, distance=distance_points)

    return time_index_pick


def picker_signal_bit_index(
    sig: np.array, sig_sample_rate_hz: float, bits_pick: float, time_interval_s: float
) -> np.ndarray:
    """
    :param sig: array of waveform data
    :param sig_sample_rate_hz: float sample rate of reference sensor
    :param bits_pick: detection threshold in db loss
    :param time_interval_s: min time interval between events
    :return: picker index from the ABSOLUTE max in bits
    """
    # Compute the distance
    distance_points = int(time_interval_s * sig_sample_rate_hz)
    height_min = log2epsilon_max(sig) - bits_pick
    index_pick, _ = signal.find_peaks(log2epsilon(sig), height=height_min, distance=distance_points)

    return index_pick


def picker_comb(sig_pick: np.ndarray, index_pick: np.ndarray) -> np.ndarray:
    """
    Constructs a comb function from the picks

    :param sig_pick: 1D record corresponding to the picks
    :param index_pick: indexes for the picks
    :return: comb function with unit amplitude
    """
    comb = np.zeros(sig_pick.shape)
    comb[index_pick] = np.ones(index_pick.shape)
    return comb
