"""
libquantum example: nwave_tdr_highpass
Constructs and processes N wave
"""
import numpy as np
import matplotlib.pyplot as plt
from quantum_inferno import styx_fft
from quantum_inferno.synth import synthetics_TO_FIX
import scipy.signal as signal


print(__doc__)
EVENT_NAME = "nwave_tfr"
station_id_str = "synth"
# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 1


def n_wave_period_center(
    intro_s: float, n_wave_duration_s: float, outro_s: float, sample_rate_hz: float
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    N-Wave is simple

    :param intro_s: intro in s
    :param n_wave_duration_s: N wave duration in seconds
    :param outro_s: outro in s
    :param sample_rate_hz: sample rate in Hz
    :return: numpy array with GT blast pulse
    """

    total_duration_s = intro_s + n_wave_duration_s + outro_s
    time_points = int(sample_rate_hz * total_duration_s)
    time_s = np.arange(time_points) / sample_rate_hz
    tau = (time_s - intro_s) / n_wave_duration_s
    # Initialize
    p_n_wave = np.zeros(time_points)
    # Initialize time ranges
    sigint1 = np.where((intro_s <= time_s) & (time_s <= time_s[-1] - outro_s))  # nwave
    p_n_wave[sigint1] = 1.0 - 2 * (time_s[sigint1] - intro_s) / n_wave_duration_s

    return p_n_wave, time_s, tau


if __name__ == "__main__":
    """
    Blast test
    """

    order_number_input = 6
    EVENT_NAME = "N-Wave Test"
    station_id_str = "Synth"

    frequency_sample_rate_hz = 8000.0
    pseudo_period_main_s = 0.2

    # Duration set by the number of cycles
    mult = 8.0
    decimate = 10
    intro = mult * pseudo_period_main_s
    sig_wf_raw, sig_time_s_raw, sig_tau = n_wave_period_center(
        intro_s=intro,
        n_wave_duration_s=pseudo_period_main_s,
        outro_s=mult * pseudo_period_main_s,
        sample_rate_hz=frequency_sample_rate_hz,
    )
    sig_wf_aa = synthetics_TO_FIX.antialias_half_nyquist(synth=np.copy(sig_wf_raw))
    sig_wf_dec = signal.decimate(np.copy(sig_wf_aa), decimate)
    sig_time_s = sig_time_s_raw[::10]

    sig_wf_lp00 = styx_fft.butter_lowpass(
        sig_wf=sig_wf_dec,
        frequency_sample_rate_hz=frequency_sample_rate_hz / decimate,
        frequency_cut_high_hz=320.0,
        filter_order=4,
        tukey_alpha=0,
    )

    sig_wf_bp1 = styx_fft.butter_bandpass(
        sig_wf=sig_wf_dec,
        frequency_sample_rate_hz=frequency_sample_rate_hz / decimate,
        frequency_cut_low_hz=20,
        frequency_cut_high_hz=320,
        filter_order=4,
        tukey_alpha=0.0,
    )

    plt.plot(sig_time_s_raw - intro, sig_wf_raw, label="Nominal 0.2s N wave @ 8 kHz")
    plt.plot(sig_time_s - intro, sig_wf_lp00, "--", label="Lowpassed, fc=320 Hz")
    plt.plot(sig_time_s - intro, sig_wf_bp1, label="Bandpassed, fc=20-320 Hz")
    plt.title("Nominal, lowpassed, and bandpassed filtered N wave")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized")
    plt.xlim(-0.2, 0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
