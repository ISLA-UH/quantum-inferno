"""
This module contains quick plotting routines
"""
import numpy as np
import matplotlib.pyplot as plt

from quantum_inferno.scales_dyadic import get_epsilon


def plot_tdr_sig(sig_wfm: np.ndarray, sig_time: np.ndarray, signal_time_base: str = "seconds") -> plt.Figure:
    """
    Waveform

    :param sig_wfm:
    :param sig_time:
    :param signal_time_base:
    :return: figure of waveform for plotting
    """
    fig = plt.figure()
    plt.plot(sig_time, sig_wfm)
    plt.title("Input waveform")
    plt.xlabel("Time, " + signal_time_base)
    return fig


def plot_tdr_rms(
        sig_wfm: np.ndarray,
        sig_time: np.ndarray,
        sig_rms_wf: np.ndarray,
        sig_rms_time: np.ndarray,
        signal_time_base: str = "seconds"
) -> plt.Figure:
    """
    Waveform

    :param sig_wfm:
    :param sig_time:
    :param sig_rms_wf:
    :param sig_rms_time:
    :param signal_time_base:
    :return: figure of waveform for plotting
    """
    fig = plt.figure()
    plt.plot(sig_time, sig_wfm)
    plt.plot(sig_rms_time, sig_rms_wf)
    plt.title("Input waveform and RMS")
    plt.xlabel("Time, " + signal_time_base)
    return fig


def plot_tfr_lin(
        tfr_power: np.ndarray,
        tfr_frequency: np.ndarray,
        tfr_time: np.ndarray,
        title_str: str = "TFR, power",
        signal_time_base: str = "seconds"
) -> plt.Figure:
    """
    TFR in linear power

    :param tfr_power:
    :param tfr_frequency:
    :param tfr_time:
    :param title_str:
    :param signal_time_base:
    :return: figure of waveform for plotting
    """
    fig = plt.figure()
    plt.pcolormesh(tfr_time, tfr_frequency, tfr_power, cmap="RdBu_r")
    plt.title(title_str)
    plt.ylabel("Frequency, samples per " + signal_time_base)
    plt.xlabel("Time, " + signal_time_base)
    return fig


def plot_tfr_bits(
        tfr_power: np.ndarray,
        tfr_frequency: np.ndarray,
        tfr_time: np.ndarray,
        bits_min: float = -8,
        bits_max: float = 0,
        title_str: str = "TFR, top bits",
        y_scale: str = None,
        tfr_x_str: str = "Time, seconds",
        tfr_y_str: str = "Frequency, hz",
        tfr_y_flip: bool = False,
) -> plt.Figure:
    """
    TFR in bits

    :param tfr_power:
    :param tfr_frequency:
    :param tfr_time:
    :param bits_max:
    :param bits_min:
    :param y_scale:
    :param title_str:
    :param tfr_x_str:
    :param tfr_y_str:
    :param tfr_y_flip:
    :return: figure of waveform for plotting
    """
    tfr_bits = 0.5 * np.log2(tfr_power / np.max(tfr_power))

    fig = plt.figure()
    plt.pcolormesh(tfr_time, tfr_frequency, tfr_bits, cmap="RdBu_r", vmin=bits_min, vmax=bits_max, shading="nearest")
    plt.yscale("linear" if y_scale is None else "log")

    if tfr_y_flip:
        plt.ylim(np.max(tfr_frequency), np.min(tfr_frequency))
    plt.title(title_str)
    plt.ylabel(tfr_y_str)
    plt.xlabel(tfr_x_str)

    return fig


def plot_st_window_tdr_lin(window: np.ndarray, freq_sx: np.ndarray, time_fft: np.ndarray) -> plt.Figure:
    """
    plot something

    :param window:
    :param freq_sx:
    :param time_fft:
    :return: figure of waveform for plotting
    """
    fig = plt.figure(figsize=(8, 8))
    for j, freq in enumerate(freq_sx):
        plt.plot(time_fft, np.abs(window[j, :]), label=freq)
    plt.legend()
    plt.title("TDR window, linear")
    return fig


def plot_st_window_tfr_bits(window: np.ndarray, frequency_sx: np.ndarray, frequency_fft: np.ndarray) -> plt.Figure:
    """
    plot something

    :param window:
    :param frequency_sx:
    :param frequency_fft:
    :return: figure of waveform for plotting
    """
    fig = plt.figure(figsize=(8, 8))
    for j, freq in enumerate(frequency_sx):
        plt.plot(frequency_fft, np.log2(np.abs(window[j, :]) + get_epsilon()), label=freq)
    plt.legend()
    plt.title("TFR window, bits")
    return fig


def plot_st_window_tfr_lin(window: np.ndarray, frequency_sx: np.ndarray, frequency_fft: np.ndarray) -> plt.Figure:
    """
    plot something

    :param window:
    :param frequency_sx:
    :param frequency_fft:
    :return: figure of waveform for plotting
    """
    fig = plt.figure(figsize=(8, 8))
    for j, freq in enumerate(frequency_sx):
        plt.plot(frequency_fft, np.abs(window[j, :]), label=freq)
    plt.legend()
    plt.title("TFR window, lin")
    return fig
