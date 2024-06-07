"""
libquantum example: s06_stft_tone_stft_atom_styx
Compute stft spectrogram with libquantum
TODO: Turn into functions with hard-coded defaults
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from quantum_inferno.synth import benchmark_signals
import quantum_inferno.plot_templates.plot_time_frequency_reps_black as pltq
from quantum_inferno import utils, scales_dyadic
from quantum_inferno.styx_cwt import cwt_complex_any_scale_pow2
from quantum_inferno.styx_stx import tfr_stx_fft

print(__doc__)
EVENT_NAME = "tone test"
station_id_str = "synth"
# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 1

# Set tone parameters
frequency_tone_hz = 60
order_number_input = 3

if __name__ == "__main__":
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

    # TODO: CLEAN UP!! Construct functions in s01 Welch
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

    # TODO: CLEAN UP!! This is a mess
    # Compute complex wavelet transform (cwt)
    frequencies_cwt_hz = scales_dyadic.log_frequency_hz_from_fft_points(
        frequency_sample_hz=frequency_sample_rate_hz,
        fft_points=len(mic_sig),
        scale_order=order_number_input)

    # TODO: BUILD frequency_cwt_hz INTO METHOD
    [frequency_cwt_hz, time_cwt_s, cwt_complex] = cwt_complex_any_scale_pow2(
        band_order_nth=order_number_input,
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        frequency_cwt_hz=frequencies_cwt_hz,
        cwt_type="fft",
        dictionary_type="spect"
    )

    cwt_power = 2 * np.abs(cwt_complex) ** 2

    # Compute Stockwell transform
    [stx_complex, _, frequency_stx_hz, frequency_stx_fft_hz, W] = tfr_stx_fft(
        sig_wf=mic_sig,
        n_fft_in=8192,  # needs to have value or error pops up in sig_pad_up_to_pow2(sig_wf, n_fft_in)
        time_sample_interval=1 / frequency_sample_rate_hz,
        frequency_min=frequency_resolution_fft_hz,
        frequency_max=frequency_sample_rate_hz / 2,
        scale_order_input=order_number_input,
        scale_ref_input=1 / 59.66796875,
        is_geometric=True,
        is_inferno=False,
    )

    stx_power = 2 * np.abs(stx_complex) ** 2

    # print("STX Frequency:", frequency_stx_fft_hz)
    # TODO: Reconcile STX frequency with STFT
    # Compute the 'equivalent' fft rms amplitude
    fft_var_norm_welch = np.abs(psd_welch_power) / mic_sig_var
    fft_var_norm_stft = np.average(stft_power, axis=1) / mic_sig_var
    fft_var_norm_cwt = np.average(cwt_power, axis=1) / mic_sig_var
    fft_var_norm_stx = np.average(stx_power, axis=1) / mic_sig_var

    # Express in bits; revisit
    # TODO: What units shall we use? Evaluate CWT and Stockwell first

    mic_stft_bits = utils.log2epsilon(np.sqrt(stft_power))
    mic_cwt_bits = utils.log2epsilon(np.sqrt(cwt_power))
    mic_stx_bits = utils.log2epsilon(np.sqrt(stx_power))

    print("Max stft bits:", np.max(mic_stft_bits))
    print("Max cwt bits:", np.max(mic_cwt_bits))
    print("Max stx bits:", np.max(mic_stx_bits))

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic CW, with taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax2.semilogx(frequency_welch_hz, fft_var_norm_welch, label="Welch")
    ax2.semilogx(frequency_stft_hz, fft_var_norm_stft, ".-", label="STFT")
    ax2.semilogx(frequency_cwt_hz, fft_var_norm_cwt, "-.", label="CWT")
    ax2.semilogx(frequency_stx_hz, fft_var_norm_stx, "--", label="STX")

    ax2.set_title("Welch and Spect FFT (RMS), f = " + str(round(frequency_center_fft_hz * 100) / 100) + " Hz")
    ax2.set_xlabel("Frequency, hz")
    ax2.set_ylabel("VAR|FFT| / VAR(SIG)")
    ax2.grid(True)
    ax2.legend()

    # plt.show()
    # exit()
    # Select plot frequencies
    fmin = 2 * frequency_resolution_fft_hz
    fmax = frequency_sample_rate_hz / 2  # Nyquist

    # TODO: Change redvox_id to station_id
    pltq.plot_wf_mesh_vert(
        redvox_id="00",
        wf_panel_a_sig=mic_sig,
        wf_panel_a_time=time_s,
        mesh_time=time_stft_s,
        mesh_frequency=frequency_stft_hz,
        mesh_panel_b_tfr=mic_stft_bits,
        mesh_panel_b_colormap_scaling="range",
        wf_panel_a_units="Norm",
        mesh_panel_b_cbar_units="bits",
        start_time_epoch=0,
        figure_title="STFT for " + EVENT_NAME,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    pltq.plot_wf_mesh_vert(
        redvox_id="00",
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

    pltq.plot_wf_mesh_vert(
        redvox_id="00",
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

    plt.show()