"""
Quantum inferno example: s08_blast_tfr
Use GT blast synthetic to compare time-frequency representations of explosion transients.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from quantum_inferno.synth import blast_pulse as kaboom
from quantum_inferno.styx_cwt import cwt_complex_any_scale_pow2
from quantum_inferno.styx_stx import tfr_stx_fft
import quantum_inferno.plot_templates.plot_cyberspectral as pltq
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon
from quantum_inferno.utilities.window import get_tukey
from quantum_inferno.utilities.short_time_fft import stft_tukey


print(__doc__)


if __name__ == "__main__":
    """
    Blast test
    """

    # alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
    # If zero, the Tukey window is equivalent to a rectangular window.
    # If one, the Tukey window is equivalent to a Hann window.
    alpha = 1
    order_number_input = 6
    EVENT_NAME = "Blast Test"
    station_id_str = "Synth"

    frequency_sample_rate_hz = 800.0
    # Target frequency
    frequency_center_hz = 60
    pseudo_period_main_s = 1 / frequency_center_hz

    # Duration set by the number of cycles
    window_cycles = 4 * 32
    window_duration_s = window_cycles * pseudo_period_main_s
    time_points = 2 ** int(np.log2(window_duration_s * frequency_sample_rate_hz))  # Use 2^n
    time_fft_nd = int(time_points / 2 ** 3)
    sig_duration_s = time_points / frequency_sample_rate_hz
    std_bit_loss = 1.0

    time_s, mic_sig = kaboom.gt_blast_center_noise(
        duration_s=sig_duration_s,
        frequency_peak_hz=frequency_center_hz,
        sample_rate_hz=frequency_sample_rate_hz,
        noise_std_loss_bits=std_bit_loss,
    )

    time_s -= time_s[0]
    mic_sig *= get_tukey(array=time_s, alpha=0.1)  # Add taper
    mic_sig /= np.max(mic_sig)  # Max unit amplitude

    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1 / frequency_sample_rate_hz)
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz - frequency_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    frequency_resolution_fft_hz = frequency_sample_rate_hz / time_fft_nd

    # TODO: Workflow is the same for all these examples - create template!!
    # TODO: Stabilize units, do the math for both wavelet, GT, and Waxler pulse
    # TODO: Use variance instead of RMS
    # Computed and nominal values
    mic_sig_rms = np.std(mic_sig)
    # TODO: Get RMS
    mic_sig_rms_nominal = 1 / np.sqrt(2)

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
    frequency_stft_hz, time_stft_s, stft_complex = stft_tukey(
        timeseries=mic_sig,
        sample_rate_hz=frequency_sample_rate_hz,
        tukey_alpha=alpha,
        segment_length=time_fft_nd,
        overlap_length=time_fft_nd // 2,  # 50% overlap
        scaling="magnitude",
        padding="zeros",
    )

    stft_power = 2 * np.abs(stft_complex) ** 2

    # Compute complex wavelet transform (cwt)
    [frequency_cwt_hz, time_cwt_s, cwt_complex] = cwt_complex_any_scale_pow2(
        band_order_nth=order_number_input,
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        cwt_type="fft",
        dictionary_type="spect",
    )

    cwt_power = np.abs(cwt_complex) ** 2

    # Compute Stockwell transform
    # TODO: Return time with zero origin
    [stx_complex, _, frequency_stx_hz, frequency_stx_fft_hz, W] = tfr_stx_fft(
        sig_wf=mic_sig,
        time_sample_interval=1 / frequency_sample_rate_hz,
        frequency_min=frequency_resolution_fft_hz,
        frequency_max=frequency_sample_rate_hz / 2,
        scale_order_input=order_number_input,
        scale_ref_input=1 / frequency_center_fft_hz,
        n_fft_in=1024,  # NEED n_fft_in to function
        is_geometric=True,
        is_inferno=False,
    )

    stx_power = 2 * np.abs(stx_complex) ** 2

    # print("STX Frequency:", frequency_stx_fft_hz)
    # TODO: Reconcile STX frequency with STFT
    # Compute the 'equivalent' fft rms amplitude
    fft_rms_welch = np.sqrt(np.abs(psd_welch_power)) / mic_sig_rms
    fft_rms_stft = np.sqrt(np.average(stft_power, axis=1)) / mic_sig_rms
    fft_rms_cwt = np.sqrt(np.average(cwt_power, axis=1)) / mic_sig_rms
    fft_rms_stx = np.sqrt(np.average(stx_power, axis=1)) / mic_sig_rms

    # Express in bits; revisit
    # TODO: What units shall we use? Evaluate CWT and Stockwell first

    mic_stft_bits = to_log2_with_epsilon(np.sqrt(stft_power))
    mic_cwt_bits = to_log2_with_epsilon(np.sqrt(cwt_power))
    mic_stx_bits = to_log2_with_epsilon(np.sqrt(stx_power))

    print("Max stft bits:", np.max(mic_stft_bits))
    print("Max cwt bits:", np.max(mic_cwt_bits))
    print("Max stx bits:", np.max(mic_stx_bits))

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic CW, with taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax2.semilogx(frequency_welch_hz, fft_rms_welch, label="Welch")
    ax2.semilogx(frequency_stft_hz, fft_rms_stft, ".-", label="STFT")
    ax2.semilogx(frequency_cwt_hz, fft_rms_cwt, "-.", label="CWT")
    ax2.semilogx(frequency_stx_hz, fft_rms_stx, "--", label="STX")

    ax2.set_title("Welch and Spect FFT (RMS), f = " + str(round(frequency_center_fft_hz * 100) / 100) + " Hz")
    ax2.set_xlabel("Frequency, hz")
    ax2.set_ylabel("FFT RMS / std(sig)")
    ax2.grid(True)
    ax2.legend()

    # plt.show()
    # exit()
    # Select plot frequencies
    fmin = 2 * frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist

    pltq.plot_wf_mesh_vert(
        station_id="00",
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
        station_id="00",
        wf_panel_a_sig=mic_sig,
        wf_panel_a_time=time_s,
        mesh_time=time_cwt_s,
        mesh_frequency=frequency_cwt_hz,
        mesh_panel_b_tfr=mic_cwt_bits,
        mesh_panel_b_colormap_scaling="range",
        wf_panel_a_units="Norm",
        mesh_panel_b_cbar_units="bits",
        start_time_epoch=0,
        figure_title="cwt for " + EVENT_NAME,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    pltq.plot_wf_mesh_vert(
        station_id="00",
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
