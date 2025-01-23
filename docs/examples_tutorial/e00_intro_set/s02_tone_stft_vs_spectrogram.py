"""
Quantum inferno example: s02_tone_stft_vs_spectrogram.py
Compute and display spectrogram on simple tone with a taper window.
There is an independent Tukey taper (w/ alpha) on each Welch and Spectrogram subwindow.
Contract over the columns and compare to Welch power spectral density (PSD) to verify amplitudes.
Case study:
Sinusoid input with unit amplitude
Validate:
Welch power averaged over the signal duration is 1/2

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import quantum_inferno.plot_templates.plot_base as ptb
import quantum_inferno.utilities.short_time_fft as stft
from quantum_inferno.plot_templates.plot_templates import plot_cw_and_power, plot_mesh_wf_vert
from quantum_inferno.synth import benchmark_signals
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

print(__doc__)

if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows. Added taper, noise, and AA to signal.
    Added STFT Tukey window alpha > 0.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    # Construct a tone of fixed frequency with a constant sample rate
    # In this example, noise, taper, and antialiasing filter are all added.
    # In the first example (FFT), the nominal signal duration was 1s.
    # In this example the nominal signal duration is 16s, with averaging (fft) window duration of 1s.
    # Compare to synthetic tone with 2^n points and max FFT amplitude at exact and NOT exact fft frequency
    # If NOT exact fft frequency does not return unit amplitude (but it's close)
    frequency_tone_hz = 60
    EVENT_NAME = f"{frequency_tone_hz} Hz Tone Test,"
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

    # alpha: Shape parameter of the Welch and STFT Tukey window, representing the fraction of the window inside the
    #        cosine tapered region.
    # If zero, the Tukey window is equivalent to a rectangular window.
    # If one, the Tukey window is equivalent to a Hann window.
    alpha = 0.25  # 25% Tukey (Cosine) window

    # Computed Variance; divides by the number of points
    mic_sig_var = np.var(mic_sig)
    mic_sig_var_nominal = 1 / 2.0

    # sig_s = 10.23875, time_spect_s = 9.92, time_stft_s = 10.24
    # Compute the Welch PSD; averaged spectrum over sliding windows
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

    # Shift the time_spect_s to start at the first time point since it returns the center of the window
    time_spect_s = time_spect_s - time_spect_s[0]

    # Compute the spectrogram with the stft option
    stft_obj = stft.get_stft_object_tukey(
        sample_rate_hz=frequency_sample_rate_hz,
        tukey_alpha=alpha,
        segment_length=time_fft_nd,
        overlap_length=time_fft_nd // 2,  # 50% overlap
        scaling="magnitude",
    )
    stft_complex = stft_obj.stft(mic_sig, padding="zeros")

    stft_magnitude = np.abs(stft_obj.stft_detrend(x=mic_sig, detr="constant", padding="zeros"))
    # calculate the time and frequency bins
    time_stft_s = np.arange(start=0, stop=stft_obj.delta_t * np.shape(stft_magnitude)[1], step=stft_obj.delta_t)
    frequency_stft_hz = stft_obj.f

    # Since one-sided, multiply by 2 to get the full power
    stft_power = 2 * np.abs(stft_complex) ** 2

    # Compute the ratio of the PSD to the variance
    # Average over the columns for the spectrogram and STFT
    welch_over_var = psd_welch_power / mic_sig_var
    spect_over_var = np.average(spec_power, axis=1) / mic_sig_var
    stft_over_var = np.average(stft_power, axis=1) / mic_sig_var

    # Express in log2(power) with epsilon
    mic_spect_bits = to_log2_with_epsilon(spec_power)
    mic_stft_bits = to_log2_with_epsilon(stft_power)
    print(f"Max spect: {np.max(spec_power)}")
    print(f"Max stft: {np.max(stft_power)}")
    print(f"Max spect bits: {np.max(mic_spect_bits)}")
    print(f"Max stft bits: {np.max(mic_stft_bits)}")

    # Compute the inverse stft (istft)
    sig_time_istft, sig_wf_istft = stft.istft_tukey(
        stft_to_invert=stft_complex,
        sample_rate_hz=frequency_sample_rate_hz,
        tukey_alpha=alpha,
        segment_length=time_fft_nd,
        overlap_length=time_fft_nd // 2,  # 50% overlap
        scaling="magnitude",
    )

    print("\n*** SUMMARY: STFT Time-Frequency Representation (TFR) estimates for a constant-frequency tone  ***")
    print("The signal.ShortTimeFFT's spectrogram and stft with scaling=magnitude are comparable.")
    print("The spectrogram returns power, whereas the stft returns invertible, complex Fourier coefficients.")
    print("The Welch spectrum is reproduced by averaging the stft over the time dimension.")
    print(
        "** NOTE: EXACT RECONSTRUCTION NOT EXPECTED WITH TAPER AND OTHER DEVIATIONS FROM IDEAL. PLAY WITH alpha. "
        "ACCEPT AND QUANTIFY COMPROMISE **"
    )

    cw_panel = ptb.CwPanel(
        mic_sig, time_s, y_units="Norm", x_units="s", title=f"Synthetic CW with {alpha*100:.2f}% taper"
    )
    power_panel = ptb.PowerPanel(
        [
            ptb.PowerPanelData(welch_over_var, frequency_welch_hz, "-", 1, "Welch"),
            ptb.PowerPanelData(spect_over_var, frequency_spect_hz, "-", 2, "Spect"),
            ptb.PowerPanelData(stft_over_var, frequency_stft_hz, "--", 1, "STFT"),
        ],
        y_units="Power/VAR(signal)",
        x_units="Hz",
        title=f"Welch, Spect, and STFT Power, f = {frequency_center_fft_hz:.3f} Hz",
    )
    fig = plot_cw_and_power(cw_panel, power_panel)

    # Plot the inverse stft (full recovery)
    fig2, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(11, 6), sharex="all")
    ax1.plot(sig_time_istft, sig_wf_istft)
    ax1.set_title("Inverse CW from ISTFT")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax1.set_xlim([-0.25, 10.5])
    ax2.plot(sig_time_istft, (mic_sig - sig_wf_istft) ** 2)
    ax2.set_title("(original-inverse ISTFT)$^2$")
    ax2.set_xlabel("Time, s")
    ax2.set_ylabel("Norm")

    # Select plot frequencies
    fmin = 2 * frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist
    wf_base = ptb.WaveformPlotBase(station_id="log$_2\\frac{1}{2}=-1$", figure_title=f"Spectrogram for {EVENT_NAME}")
    wf_panel = ptb.WaveformPanel(mic_sig, time_s)
    mesh_base = ptb.MeshBase(time_spect_s, frequency_spect_hz, frequency_hz_ymin=fmin, frequency_hz_ymax=fmax)
    mesh_panel = ptb.MeshPanel(mic_spect_bits, colormap_scaling="range", cbar_units="log$_2$(Power)")
    spect = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

    wf_base.figure_title = f"STFT for {EVENT_NAME}"
    mesh_base.time = time_stft_s
    mesh_base.frequency = frequency_stft_hz
    mesh_panel.tfr = mic_stft_bits
    stft = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

    plt.show()
