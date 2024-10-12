"""
Quantum inferno example: s01_cw_power_plots.py
Tutorial on the quantum-inferno continuous waveform and power plotting functions and their parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import quantum_inferno.plot_templates.plot_base as ptb
import quantum_inferno.utilities.short_time_fft as stft
from quantum_inferno.plot_templates.plot_templates import plot_cw_and_power
from quantum_inferno.synth import benchmark_signals

print(__doc__)


def calculate_t_domain_example_values():
    # The values calculated here to be plotted are adapted from the examples, refer to s02_tone_stft_vs_spectrogram.py
    frequency_tone_hz = 60
    [
        mic_sig,
        time_s,
        time_fft_nd,
        frequency_sample_rate_hz,
        frequency_center_fft_hz,
        frequency_resolution_fft_hz,
    ] = benchmark_signals.well_tempered_tone(
        frequency_center_hz=frequency_tone_hz,
        frequency_sample_rate_hz=800,
        time_duration_s=16,
        time_fft_s=1,
        use_fft_frequency=True,
        add_noise_taper_aa=True,
    )
    return frequency_resolution_fft_hz, frequency_sample_rate_hz, mic_sig, time_s, time_fft_nd


def calculate_power_example_values(mic_sig, time_fft_nd, frequency_sample_rate_hz, mic_sig_variance, alpha=0.25):
    frequency_welch_hz, psd_welch_power = signal.welch(
        x=mic_sig,
        fs=frequency_sample_rate_hz,
        window=("tukey", alpha),
        nperseg=time_fft_nd,
        noverlap=time_fft_nd // 2,
        nfft=time_fft_nd,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        scaling="spectrum",
        average="mean",
    )
    frequency_spect_hz, time_spect_s, spec_mag = stft.spectrogram_tukey(
        timeseries=mic_sig,
        sample_rate_hz=frequency_sample_rate_hz,
        tukey_alpha=alpha,
        segment_length=time_fft_nd,
        overlap_length=time_fft_nd // 2,  # 50% overlap
        scaling="magnitude",
        padding="zeros",
    )
    # Since one-sided, multiply by 2 to get the full power
    spec_power = 2 * spec_mag
    # Compute the spectrogram with the stft option
    frequency_stft_hz, time_stft_s, stft_complex = stft.stft_tukey(
        timeseries=mic_sig,
        sample_rate_hz=frequency_sample_rate_hz,
        tukey_alpha=alpha,
        segment_length=time_fft_nd,
        overlap_length=time_fft_nd // 2,  # 50% overlap
        scaling="magnitude",
        padding="zeros",
    )
    # Since one-sided, multiply by 2 to get the full power
    stft_power = 2 * np.abs(stft_complex) ** 2
    welch_over_var = psd_welch_power / mic_sig_variance
    spect_over_var = np.average(spec_power, axis=1) / mic_sig_variance
    stft_over_var = np.average(stft_power, axis=1) / mic_sig_variance
    return frequency_welch_hz, welch_over_var, frequency_spect_hz, spect_over_var, frequency_stft_hz, stft_over_var


def main():
    print("\nWe'll start by generating the times series and power arrays from s02_tone_stft_vs_spectrogram.py.")
    # Calculate some values to be plotted (see s02_tone_stft_vs_spectrogram.py for details on the values themselves)
    freq_resolution_fft_hz, freq_sample_rate_hz, mic_sig, time_s, time_fft_nd = calculate_t_domain_example_values()
    mic_sig_var = np.var(mic_sig)
    f_welch_hz, welch_over_var, f_spect_hz, spect_over_var, f_stft_hz, stft_over_var = calculate_power_example_values(
        mic_sig, time_fft_nd, freq_sample_rate_hz, mic_sig_var)
    # # # EXAMPLE: DEFAULT PARAMETERS # # #
    print("\nLike in the previous example, we'll be using Panel objects--this time CwPanel and PowerPanel. We'll start")
    print("by building the Panel objects with only the required variables, then explore parameters.")
    print("First, let's give CwPanel time and signal arrays.")
    # create CwPanel object with only required parameters
    cw_panel = ptb.CwPanel(mic_sig, time_s)
    print("Next, we'll build the PowerPanel object, which requires a list of PowerPanelData objects, each of which")
    print("requires a signal (power) array, frequency array, linestyle, linewidth, and label.")
    # create PowerPanel object with only required parameters
    power_panel = ptb.PowerPanel(
        [
            ptb.PowerPanelData(welch_over_var, f_welch_hz, "-", 1, "Welch"),
            ptb.PowerPanelData(spect_over_var, f_spect_hz, "-", 2, "Spect"),
            ptb.PowerPanelData(stft_over_var, f_stft_hz, "--", 1, "STFT"),
        ],
    )
    print("We'll now plot the figure with only the required parameters. Close the figure to continue.")
    fig_required = plot_cw_and_power(cw_panel, power_panel)
    plt.show()
    print("\nThere are less optional parameters for the CW and power plots than for the waveform plots, but we'll")
    print("still go over them. For the CW panel, the optional parameters are y_units, x_units, and title.")
    # set optional parameters for CwPanel and PowerPanel objects
    cw_panel.title = "Synthetic CW with 25% taper"
    cw_panel.y_units = "Normalized Amplitude"
    cw_panel.x_units = "seconds"
    print("For the Power panel, the optional parameters are y_units, x_units, and title.")
    power_panel.y_units = "Power/VAR(signal)"
    power_panel.x_units = "Hz"
    power_panel.title = "Welch, Spect, and STFT Power, f = 60 Hz"
    print("We'll now plot the figure with the optional parameters. Close the figure to continue.")
    fig_optional = plot_cw_and_power(cw_panel, power_panel)
    plt.show()
    print("\nThis concludes the tutorial on the quantum-inferno CW and power plotting template.")


if __name__ == "__main__":
    main()
