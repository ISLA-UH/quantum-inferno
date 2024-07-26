"""
Quantum inferno example: s06_tone_tfr.py
Compute and compare Welch, STFT, CWT, and STWV on simple tone with a taper window.
TODO: Turn signal welch and stft into functions with hard-coded defaults

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from quantum_inferno import styx_stx, styx_cwt
import quantum_inferno.plot_templates.plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_wf_mesh_vert
from quantum_inferno.synth import benchmark_signals
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

print(__doc__)

if __name__ == "__main__":
    # Set tone parameters. Play with order number to learn things about the methods
    frequency_tone_hz = 60
    order_number_input = 12
    EVENT_NAME = f"{frequency_tone_hz} Hz Tone Test"
    ORDER_NUM = f"Dyadic Order {order_number_input}"

    # alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the tapered region.
    # If zero, the Tukey window is equivalent to a rectangular window.
    # If one, the Tukey window is equivalent to a Hann window.
    alpha = 0.5

    """
    Compute the spectrogram over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """
    # Construct a tone of fixed frequency with a constant sample rate as in previous
    # The nominal signal duration is 16s, with nominal averaging (fft) window duration of 1s.
    [
        mic_sig,
        time_s,
        time_fft_nd,
        frequency_sample_rate_hz,
        frequency_center_fft_hz,
        frequency_resolution_fft_hz,
    ] = benchmark_signals.well_tempered_tone(frequency_center_hz=frequency_tone_hz, add_noise_taper_aa=True)

    # # Computed and nominal values
    # mic_sig_rms = np.std(mic_sig)
    # mic_sig_rms_nominal = 1 / np.sqrt(2)

    # Computed Variance; divides by the number of points
    mic_sig_var = np.var(mic_sig)
    mic_sig_var_nominal = 1 / 2.0

    # TODO: Construct functions in styx.fft or styx.stft with defaults that match dwt, cwt, stx, and wv defaults
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

    # Compute the spectrogram with the stft option
    frequency_stft_hz, time_stft_s, stft_complex = signal.stft(
        x=mic_sig,
        fs=frequency_sample_rate_hz,
        window=("tukey", alpha),
        nperseg=time_fft_nd,
        noverlap=time_fft_nd // 2,
        nfft=time_fft_nd,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        boundary="zeros",
        padded=True,
    )

    stft_power = 2 * np.abs(stft_complex) ** 2

    # Compute the CWT
    [frequency_cwt_hz, time_cwt_s, cwt_complex] = styx_cwt.cwt_complex_any_scale_pow2(
        band_order_nth=order_number_input,
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        cwt_type="fft",
        dictionary_type="spect",
    )

    cwt_power = 2 * np.abs(cwt_complex) ** 2

    # Compute Stockwell transform (STX)
    frequency_stx_hz, time_stx_s, stx_complex = styx_stx.stx_complex_any_scale_pow2(
        band_order_nth=order_number_input,
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz
    )

    stx_power = 2 * np.abs(stx_complex) ** 2

    # print("STX Frequency:", frequency_stx_fft_hz)
    # TODO: Reconcile STX, CWT frequency with STFT; could be its own exercise
    # Compute the spectral power over signal variance; should be close to unity at tone frequency
    fft_var_norm_welch = np.abs(psd_welch_power) / mic_sig_var
    fft_var_norm_stft = np.average(stft_power, axis=1) / mic_sig_var
    fft_var_norm_cwt = np.average(cwt_power, axis=1) / mic_sig_var
    fft_var_norm_stx = np.average(stx_power, axis=1) / mic_sig_var

    # Express in log2(power)
    mic_stft_bits = to_log2_with_epsilon(stft_power)
    mic_cwt_bits = to_log2_with_epsilon(cwt_power)
    mic_stx_bits = to_log2_with_epsilon(stx_power)

    print(f"Max stft bits: {np.max(mic_stft_bits)}")
    print(f"Max cwt bits: {np.max(mic_cwt_bits)}")
    print(f"Max stx bits: {np.max(mic_stx_bits)}")

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic CW with taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax2.semilogx(frequency_welch_hz, fft_var_norm_welch, label="Welch")
    ax2.semilogx(frequency_stft_hz, fft_var_norm_stft, ".-", label="STFT")
    ax2.semilogx(frequency_cwt_hz, fft_var_norm_cwt, "-.", label="CWT")
    ax2.semilogx(frequency_stx_hz, fft_var_norm_stx, "--", label="STX")

    ax2.set_title(f"Welch and Spect FFT (RMS), f = {frequency_center_fft_hz:.3f} Hz")
    ax2.set_xlabel("Frequency, hz")
    ax2.set_ylabel("VAR(|FFT|) / VAR(SIG)")
    ax2.grid(True)
    ax2.legend()

    # Show the STFT, CWT, and STX
    # Select plot frequencies
    fmin = 2 * frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist

    # Plot the STFT
    wf_base = ptb.WaveformBase("", f"STFT for {EVENT_NAME}")
    wf_panel = ptb.WaveformPanel(mic_sig, time_s)
    mesh_base = ptb.MeshBase(time_stft_s, frequency_stft_hz, frequency_hz_ymin=fmin, frequency_hz_ymax=fmax)
    mesh_panel = ptb.MeshPanel(mic_stft_bits, colormap_scaling="range", cbar_units="log$_2$(Power)")
    stft = plot_wf_mesh_vert(wf_base, wf_panel, mesh_base, mesh_panel)

    # Plot the CWT
    wf_base.figure_title = f"CWT for {EVENT_NAME}, {ORDER_NUM}"
    mesh_base.time = time_cwt_s
    mesh_base.frequency = frequency_cwt_hz
    mesh_panel.tfr = mic_cwt_bits
    cwt = plot_wf_mesh_vert(wf_base, wf_panel, mesh_base, mesh_panel)

    # Plot the STX
    wf_base.figure_title = f"STX for {EVENT_NAME}, {ORDER_NUM}"
    mesh_base.time = time_s
    mesh_base.frequency = frequency_stx_hz
    mesh_panel.tfr = mic_stx_bits
    stx = plot_wf_mesh_vert(wf_base, wf_panel, mesh_base, mesh_panel)

    plt.show()
