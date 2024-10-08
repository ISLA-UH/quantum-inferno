"""
Quantum inferno example: s05_atom_tfr
Time frequency representation (TFR) of a Gabor atom, otherwise referred to as logon, or sound grain
TODO: Refine scaling units to assess grain performance

"""
import numpy as np
import matplotlib.pyplot as plt

from quantum_inferno import styx_stx, styx_cwt, styx_fft
import quantum_inferno.plot_templates.plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_mesh_wf_vert
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

    EVENT_NAME = f"Gabor atom centered at {frequency_center_hz} Hz"
    ORDER_NUM = f"Dyadic Order: {order_number_input}"

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
    print(f"Center CWT FFT frequency: {frequency_resolution_cwt_hz} Hz")
    print(f"Center STFT FFT frequency: {frequency_resolution_stft_hz} Hz")

    # exit()
    # TODO: Note oversampling on CWT leads to overestimation of energy!!
    frequency_cwt_fft_hz = frequency_stft_pos_hz[2:]
    # frequency_inferno_hz = frequency_cwt_pos_hz[1:]

    # Can choose unit spectrum or canonical normalized
    mic_sig_complex, time_s, scale, omega, amp = styx_cwt.wavelet_centered_4cwt(
        band_order_nth=order_number_input,
        duration_points=time_nd,
        scale_frequency_center_hz=frequency_center_stft_hz,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        dictionary_type="spect",
    )

    time_s -= time_s[0]
    mic_sig_real = np.real(mic_sig_complex)
    mic_sig_imag = np.imag(mic_sig_complex)

    # Computed Variance; divides by the number of points
    mic_sig_real_var = np.var(mic_sig_real)
    mic_sig_imag_var = np.var(mic_sig_imag)

    # Theoretical variance
    mic_sig_real_var_nominal, mic_sig_imag_var_nominal = styx_cwt.wavelet_variance_theory(amp, time_s, scale, omega)

    print("\nGabor atom Variance")
    print(f"mic_sig_real_variance: {mic_sig_real_var}")
    print(f"real_variance_nominal: {mic_sig_real_var_nominal}")
    print(f"mic_sig_imag_variance: {mic_sig_imag_var}")
    print(f"imag_variance_nominal: {mic_sig_imag_var_nominal}")

    # Choose the real component as the test signal
    mic_sig = np.copy(mic_sig_real)
    mic_sig_var = mic_sig_real_var
    mic_sig_var_nominal = mic_sig_real_var_nominal
    print("\nChoose real part as signal:")
    print(f"Computed variance/theoretical variance: {mic_sig_var / mic_sig_var_nominal}")

    # Scale by RMS; power is scaled by variance
    # TODO: Reconcile factor of 2
    mic_sig /= 2*np.sqrt(mic_sig_var)

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
    welch_over_var = psd_welch_power
    stft_over_var = np.average(stft_power, axis=1)
    cwt_over_var = np.average(cwt_power, axis=1)
    stx_over_var = np.average(stx_power, axis=1)

    # Express variance-scaled TFR in Log2
    mic_stft_bits = to_log2_with_epsilon(stft_power)
    mic_cwt_bits = to_log2_with_epsilon(cwt_power)
    mic_stx_bits = to_log2_with_epsilon(stx_power)

    print("\nSum variance-scaled power spectral density (PSD)")
    print(f"Welch PSD, Scaled: {np.sum(welch_over_var)}")
    print(f"STFT PSD, Scaled: {np.sum(stft_over_var)}")
    print(f"CWT PSD, Scaled: {np.sum(cwt_over_var)}")
    print(f"STX PSD, Scaled: {np.sum(stx_over_var)}")

    print("\nMax variance-scaled spectral power")
    print(f"1/[4 sqrt(2)]: {1 / (4 * np.sqrt(2))}")
    print(f"Max Scaled Welch PSD: {np.max(welch_over_var)}")
    print(f"Max Scaled STFT PSD: {np.max(stft_over_var)}")
    print(f"Max Scaled CWT PSD: {np.max(cwt_over_var)}")
    print(f"Max Scaled STX PSD: {np.max(stx_over_var)}")

    print("\nMax variance-scaled TFR power")
    print(f"Max Scaled CWT Power: {np.max(cwt_power)}")
    print(f"Max Log2 Scaled STFT: {np.max(mic_stft_bits)}")
    print(f"Max Log2 Scaled CWT: {np.max(mic_cwt_bits)}")
    print(f"Max Log2 Scaled STX: {np.max(mic_stx_bits)}")

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic CW with taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("sig")
    ax2.semilogx(frequency_welch_hz, welch_over_var, label="Welch")
    ax2.semilogx(frequency_stft_hz, stft_over_var, ".-", label="STFT")
    ax2.semilogx(frequency_cwt_hz, cwt_over_var, "-.", label="CWT")
    ax2.semilogx(frequency_stx_hz, stx_over_var, "--", label="STX")
    ax2.set_title(f"Spectral Power, f = {frequency_center_stft_hz:.3f} Hz")
    ax2.set_xlabel("Frequency, hz")
    ax2.set_ylabel("Power / Var(sig)")
    ax2.grid(True)
    ax2.legend()

    # Select plot frequencies
    fmin = 2 * frequency_resolution_stft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist

    # Plot the STFT
    wf_base = ptb.WaveformPlotBase("", f"STFT for {EVENT_NAME}, {ORDER_NUM}")
    wf_panel = ptb.WaveformPanel(mic_sig, time_s)
    mesh_base = ptb.MeshBase(time_stft_s, frequency_stft_hz, frequency_hz_ymin=fmin, frequency_hz_ymax=fmax)
    mesh_panel = ptb.MeshPanel(mic_stft_bits, colormap_scaling="range", cbar_units="log$_2$(Power)")
    stft = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

    # Plot the CWT
    wf_base.figure_title = f"CWT for {EVENT_NAME}, {ORDER_NUM}"
    mesh_base.time = time_cwt_s
    mesh_base.frequency = frequency_cwt_hz
    mesh_panel.tfr = mic_cwt_bits
    cwt = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

    # Plot the STX
    wf_base.figure_title = f"STX for {EVENT_NAME}, {ORDER_NUM}"
    mesh_base.time = time_s
    mesh_base.frequency = frequency_stx_hz
    mesh_panel.tfr = mic_stx_bits
    stx = plot_mesh_wf_vert(mesh_base, mesh_panel, wf_base, wf_panel)

    plt.show()
