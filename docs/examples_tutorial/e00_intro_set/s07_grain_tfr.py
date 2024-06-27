"""
Quantum inferno example: s07_grain_tfr
Time frequency representation (TFR) of a Gabor atom, or sound grain
TODO: Select better scaling units to assess grain performance

"""
import numpy as np
import matplotlib.pyplot as plt
from quantum_inferno import styx_stx, styx_cwt, styx_fft
import quantum_inferno.plot_templates.plot_cyberspectral as pltq
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

print(__doc__)

if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    # alpha: Shape parameter of the Welch and STFT Tukey window, representing the fraction of the window
    # inside the cosine tapered region.
    # If zero, the Tukey window is equivalent to a rectangular window.
    # If one, the Tukey window is equivalent to a Hann window.
    alpha = 0.25

    # Specifying the grain parameters requires some forethought
    frequency_center_hz = 100
    frequency_sample_rate_hz = 800
    order_number_input = 12

    EVENT_NAME = "Gabor atom centered at " + str(frequency_center_hz) + " Hz"
    station_id_str = ", Dyadic Order: " + str(order_number_input)

    # TODO: ADD Averaging frequency for fft_nd
    time_nd = 2 ** 11
    time_fft_nd = 2 ** 7

    # The CWX and STX will be evaluated from the number of points in FFT of the signal
    frequency_cwt_pos_hz = np.fft.rfftfreq(time_nd, d=1 / frequency_sample_rate_hz)
    # Want to evaluate the CWX and STX at the NFFT frequencies of the sliding-window Welch/STFT spectra
    frequency_stft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1 / frequency_sample_rate_hz)

    # TODO: REVISIT THIS - CWT FREQUENCY IS COMPUTED from the number of points of the signal
    # CWT
    cwt_fft_index = np.argmin(np.abs(frequency_cwt_pos_hz - frequency_center_hz))
    frequency_center_cwt_hz = frequency_cwt_pos_hz[cwt_fft_index]
    frequency_resolution_cwt_hz = frequency_sample_rate_hz / time_nd
    # STFT
    stft_fft_index = np.argmin(np.abs(frequency_stft_pos_hz - frequency_center_hz))
    frequency_center_stft_hz = frequency_stft_pos_hz[stft_fft_index]
    frequency_resolution_stft_hz = frequency_sample_rate_hz / time_fft_nd

    # Compare:
    print("These two should coincide for a fair comparison")
    print("Center CWT FFT frequency, Hz:", frequency_center_cwt_hz)
    print("Center STFT FFT frequency, Hz:", frequency_center_stft_hz)

    # exit()
    # TODO: Note oversampling on CWT leads to overestimation of energy!!
    frequency_cwt_fft_hz = frequency_stft_pos_hz[2:]
    # frequency_inferno_hz = frequency_cwt_pos_hz[1:]

    # TODO: SPELL OUT NORMALIZATION
    mic_sig_complex, time_s, scale, omega, amp = styx_cwt.wavelet_centered_4cwt(
        band_order_nth=order_number_input,
        duration_points=time_nd,
        scale_frequency_center_hz=frequency_center_stft_hz,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        dictionary_type="norm",
    )

    time_s -= time_s[0]
    mic_sig_real = np.real(mic_sig_complex)
    mic_sig_imag = np.imag(mic_sig_complex)

    # Computed Variance; divides by the number of points
    mic_sig_real_var = np.var(mic_sig_real)
    mic_sig_imag_var = np.var(mic_sig_imag)

    # Theoretical variance TODO: construct function
    mic_sig_real_var_nominal = (
        amp ** 2 / len(time_s) * 0.5 * np.sqrt(np.pi) * scale * (1 + np.exp(-((scale * omega) ** 2)))
    )
    mic_sig_imag_var_nominal = (
        amp ** 2 / len(time_s) * 0.5 * np.sqrt(np.pi) * scale * (1 - np.exp(-((scale * omega) ** 2)))
    )

    # Mathematical integral ~ computed Variance * Number of Samples. The dictionary type = "norm" returns 1/2.
    mic_sig_real_integral = np.var(mic_sig_real) * len(mic_sig_real)
    mic_sig_imag_integral = np.var(mic_sig_imag) * len(mic_sig_real)

    print("\nAtom Variance")
    print("mic_sig_real_variance:", mic_sig_real_var)
    print("real_variance_nominal:", mic_sig_real_var_nominal)
    print("mic_sig_imag_variance:", mic_sig_imag_var)
    print("imag_variance_nominal:", mic_sig_imag_var_nominal)

    # Choose the real component as the test signal
    mic_sig = np.copy(mic_sig_real)
    mic_sig_var = mic_sig_real_var
    mic_sig_var_nominal = mic_sig_real_var_nominal
    print("\nChoose real part as signal:")
    print("var/nominal var:", mic_sig_var / mic_sig_var_nominal)

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, psd_welch_power = styx_fft.welch_power_pow2(
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        nfft_points=time_fft_nd,
        segment_points=time_fft_nd,  # nfft must be greater than or equal to nperseg.
    )

    # STFT
    frequency_stft_hz, time_stft_s, stft_complex = styx_fft.stft_complex_pow2(
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        nfft_points=time_fft_nd,
        segment_points=time_fft_nd,  # nfft must be greater than or equal to nperseg.
    )

    stft_power = 2 * np.abs(stft_complex) ** 2

    # CWT
    frequency_cwt_hz, time_cwt_s, cwt_complex = styx_cwt.cwt_complex_any_scale_pow2(
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        band_order_nth=order_number_input,
        dictionary_type="spect",
    )

    cwt_power = 2 * np.abs(cwt_complex) ** 2

    # Stockwell transform
    frequency_stx_hz, time_stx_s, stx_complex = styx_stx.stx_complex_any_scale_pow2(
        band_order_nth=order_number_input, sig_wf=mic_sig, frequency_sample_rate_hz=frequency_sample_rate_hz
    )

    stx_power = 2 * np.abs(stx_complex) ** 2

    # Scale power by variance
    welch_over_var = psd_welch_power / mic_sig_var
    stft_over_var = np.average(stft_power, axis=1) / mic_sig_var
    cwt_over_var = np.average(cwt_power, axis=1) / mic_sig_var
    stx_over_var = np.average(stx_power, axis=1) / mic_sig_var

    # Express variance-scaled TFR in Log2
    mic_stft_bits = to_log2_with_epsilon(stft_power/mic_sig_var)
    mic_cwt_bits = to_log2_with_epsilon(cwt_power/mic_sig_var)
    mic_stx_bits = to_log2_with_epsilon(stx_power/mic_sig_var)

    print("\nSum variance-scaled power spectral density (PSD)")
    print("Welch PSD, Scaled:", np.sum(welch_over_var))
    print("STFT PSD, Scaled:", np.sum(stft_over_var))
    print("CWT PSD, Scaled:", np.sum(cwt_over_var))
    print("STX PSD, Scaled:", np.sum(stx_over_var))

    print("\nMax variance-scaled spectral power")
    print("1/sqrt(2):", 1 / np.sqrt(2))
    print("Max Scaled Welch PSD", np.max(welch_over_var))
    print("Max Scaled STFT PSD:", np.max(stft_over_var))
    print("Max Scaled CWT PSD:", np.max(cwt_over_var))
    print("Max Scaled STX PSD:", np.max(stx_over_var))

    print("\nMax variance-scaled TFR power")
    print("Max Scaled CWT Power:", np.max(cwt_power/mic_sig_var))
    print("Max Log2 Scaled STFT:", np.max(mic_stft_bits))
    print("Max Log2 Scaled CWT:", np.max(mic_cwt_bits))
    print("Max Log2 Scaled STX:", np.max(mic_stx_bits))

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic CW, with taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax2.semilogx(frequency_welch_hz, welch_over_var, label="Welch")
    ax2.semilogx(frequency_stft_hz, stft_over_var, ".-", label="STFT")
    ax2.semilogx(frequency_cwt_hz, cwt_over_var, "-.", label="CWT")
    ax2.semilogx(frequency_stx_hz, stx_over_var, "--", label="STX")
    ax2.set_title("Spectral Power, f = " + str(round(frequency_center_stft_hz * 100) / 100) + " Hz")
    ax2.set_xlabel("Frequency, hz")
    ax2.set_ylabel("Power / Var(sig)")
    ax2.grid(True)
    ax2.legend()

    # Select plot frequencies
    fmin = 2 * frequency_resolution_stft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist

    pltq.plot_wf_mesh_vert(
        station_id="",
        wf_panel_a_sig=mic_sig,
        wf_panel_a_time=time_s,
        mesh_time=time_stft_s,
        mesh_frequency=frequency_stft_hz,
        mesh_panel_b_tfr=mic_stft_bits,
        mesh_panel_b_colormap_scaling="range",
        wf_panel_a_units="Norm",
        mesh_panel_b_cbar_units="bits",
        start_time_epoch=0,
        figure_title="stft for " + EVENT_NAME,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    pltq.plot_wf_mesh_vert(
        station_id=station_id_str,
        wf_panel_a_sig=mic_sig,
        wf_panel_a_time=time_s,
        mesh_time=time_cwt_s,
        mesh_frequency=frequency_cwt_hz,
        mesh_panel_b_tfr=mic_cwt_bits,
        mesh_panel_b_colormap_scaling="range",
        wf_panel_a_units="Norm",
        mesh_panel_b_cbar_units="bits",
        start_time_epoch=0,
        figure_title="CWT for " + EVENT_NAME,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    pltq.plot_wf_mesh_vert(
        station_id=station_id_str,
        wf_panel_a_sig=mic_sig,
        wf_panel_a_time=time_s,
        mesh_time=time_s,
        mesh_frequency=frequency_stx_hz,
        mesh_panel_b_tfr=mic_stx_bits,
        mesh_panel_b_colormap_scaling="range",
        wf_panel_a_units="Norm",
        mesh_panel_b_cbar_units="bits",
        start_time_epoch=0,
        figure_title="STX for " + EVENT_NAME,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    plt.show()
