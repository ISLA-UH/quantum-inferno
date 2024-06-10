"""
Inferno example s01_tone_spectral_canvas.
Define the cyberspectral canvas from a knowledge of the signal center frequency and passband.
Compute a periodogram and a spectrogram of a tone over sliding windows.
The Welch method is equivalent to averaging the spectrogram over the columns.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_inferno import styx_fft, scales_dyadic, utils
import quantum_inferno.plot_templates.plot_cyberspectral as pltq

print(__doc__)


if __name__ == "__main__":
    """
    Compute a spectrogram of a Gabor wavelet (logon, grain) over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    EVENT_NAME = "Example e01"
    station_id_str = "e01"

    # Specify a Gaussian wavelet as a prototype band-limited transient signal
    # with a well-defined center frequency
    frequency_center_hz = 5
    # A very tight wavelet would have a single cycle. Due to the inevitability of window tapering,
    # one should generally include more than one cycle in the analysis window.
    logon_number_of_cycles = 3
    # The order scales with the number of cycles in a wavelet
    logon_order = scales_dyadic.order_from_cycles(logon_number_of_cycles)

    print(
        f"Wavelets containing {str(logon_number_of_cycles)} "
        f"cycles of the center frequency would have order {str(logon_order)}"
    )
    print(
        "Recommend analysis using only standardized orders 1, 3, 6, 12, 24 "
        "tuned to the signal (transients to continuous)"
    )

    # Since the transient is well centered in frequency, we can define the passband.

    # Let's set the Nyquist four octaves above center. This is the upper limit of the spectral canvas.
    octaves_above_center = 4
    frequency_nyquist_hz = frequency_center_hz * octaves_above_center
    # Which sets the sample rate.
    frequency_sample_rate_hz = 2 * frequency_nyquist_hz

    # Just as Nyquist is the cutoff of a lowpass filter, the averaging frequency can be considered
    # the cutoff of the analysis highpass filter.
    # Let's set the averaging frequency to be four octaves below center.
    octaves_below_center = 3
    frequency_averaging_hz = frequency_center_hz / octaves_below_center
    # The FFT duration is set by the averaging period and the number of cycles
    duration_fft_s = logon_number_of_cycles / frequency_averaging_hz

    # The spectral canvas passband is set
    print(
        f"The spectral analysis is constrained to {str(frequency_averaging_hz)} " f" - {str(frequency_nyquist_hz)} Hz"
    )

    # Spectral cyberspace is defined is sample points and scaled frequencies
    ave_points_ceil_log2, ave_points_ceil_pow2, ave_time_ceil_pow2_s = utils.duration_ceil(
        sample_rate_hz=frequency_sample_rate_hz, time_s=duration_fft_s
    )
    time_fft_nd = 2 ** ave_points_ceil_log2
    # Scale the total number of points to the averaging window
    time_nd = time_fft_nd * 4

    # The CWX and STX will be evaluated from the number of points in FFT of the signal
    frequency_cwt_pos_hz = np.fft.rfftfreq(time_nd, d=1 / frequency_sample_rate_hz)
    # Want to evaluate the CWX and STX at the NFFT frequencies of the sliding-window Welch/STFT spectra
    frequency_stft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1 / frequency_sample_rate_hz)

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

    # Construct test wavelet
    time_s = np.arange(time_nd) / frequency_sample_rate_hz
    mic_sig_complex = np.exp(1j * 2 * np.pi * frequency_center_hz * time_s)
    mic_sig_real = np.real(mic_sig_complex)
    mic_sig_imag = np.imag(mic_sig_complex)

    # Computed Variance; divides by the number of points
    mic_sig_real_var = np.var(mic_sig_real)
    mic_sig_imag_var = np.var(mic_sig_imag)

    # Theoretical variance TODO: construct function
    mic_sig_real_var_nominal = 1 / 2
    mic_sig_imag_var_nominal = 1 / 2

    # Mathematical integral ~ computed Variance * Number of Samples. The dictionary type = "norm" returns 1/2.
    mic_sig_real_integral = np.var(mic_sig_real) * len(mic_sig_real)
    mic_sig_imag_integral = np.var(mic_sig_imag) * len(mic_sig_real)

    print("\nAtom Variance")
    print("mic_sig_real_variance:", mic_sig_real_var)
    print("real_variance_nominal:", mic_sig_real_var_nominal)
    print("mic_sig_imag_variance:", mic_sig_imag_var)
    print("imag_variance_nominal:", mic_sig_imag_var_nominal)

    # # Choose the real component as the test signal
    # mic_sig = np.copy(mic_sig_real)
    # # In principle, could highpass/bandpass
    #
    # mic_sig_var = mic_sig_real_var
    # mic_sig_var_nominal = mic_sig_real_var_nominal
    # print('\nChoose real part as signal:')
    # print('var/nominal var:', mic_sig_var/mic_sig_var_nominal)

    # Choose the imaginary component as the test signal
    mic_sig = np.copy(-mic_sig_imag)
    # In principle, could highpass/bandpass

    mic_sig_var = mic_sig_imag_var
    mic_sig_var_nominal = mic_sig_imag_var_nominal
    print("\nChoose imaginary part as signal:")
    print("var/nominal var:", mic_sig_var / mic_sig_var_nominal)

    fractional_overlap = 0.95
    overlap_pts = np.round(fractional_overlap * time_fft_nd)
    tukey_alpha = 1
    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, psd_welch_power = styx_fft.welch_power_pow2(
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        segment_points=time_fft_nd,
        overlap_points=overlap_pts,
        alpha=tukey_alpha,
    )

    # STFT with Tukey window
    frequency_stft_hz, time_stft_s, stft_complex = styx_fft.stft_complex_pow2(
        sig_wf=mic_sig,
        frequency_sample_rate_hz=frequency_sample_rate_hz,
        segment_points=time_fft_nd,
        overlap_points=overlap_pts,
        alpha=tukey_alpha,
    )

    # # STFT with Gaussian window
    # frequency_stft_hz, time_stft_s, stft_complex = \
    #     styx_fft.gtx_complex_pow2(sig_wf=mic_sig,
    #                               frequency_sample_rate_hz=frequency_sample_rate_hz,
    #                               segment_points=time_fft_nd,
    #                               overlap_points=overlap_pts)

    stft_power = 2 * np.abs(stft_complex) ** 2

    # Scale power by variance
    welch_over_var = psd_welch_power / mic_sig_var
    stft_over_var = np.average(stft_power, axis=1) / mic_sig_var

    print("\nSum scaled spectral power")
    print("Sum Welch:", np.sum(welch_over_var))
    print("Sum STFT:", np.sum(stft_over_var))

    plt.style.use("dark_background")
    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic Gabor wavelet with taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax2.semilogx(frequency_welch_hz, welch_over_var, label="Welch")
    ax2.semilogx(frequency_stft_hz, stft_over_var, ".-", label="STFT")

    ax2.set_title("Welch and Spect FFT (RMS), f = " + str(round(frequency_center_stft_hz * 100) / 100) + " Hz")
    ax2.set_xlabel("Frequency, hz")
    ax2.set_ylabel("Power / var(sig)")
    ax2.grid(True)
    ax2.legend()

    # plt.show()
    # exit()
    # plt.figure()
    # plt.plot(cwt_information_bits_per_sample)

    # Select plot frequencies
    fmin = frequency_averaging_hz
    fmax = frequency_nyquist_hz
    # +EPSILON16 reduces numerical noise artifacts

    plt.style.use("dark_background")
    pltq.plot_wf_mesh_vert(
        station_id=station_id_str,
        wf_panel_a_sig=mic_sig,
        wf_panel_a_time=time_s,
        mesh_time=time_stft_s,
        mesh_frequency=frequency_stft_hz,
        mesh_panel_b_tfr=np.log2(stft_power + scales_dyadic.EPSILON16),
        mesh_panel_b_colormap_scaling="auto",
        frequency_scaling="linear",
        wf_panel_a_units="Norm",
        mesh_panel_b_cbar_units="bits",
        start_time_epoch=0,
        figure_title="STFT",
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
        mesh_colormap="inferno",
        waveform_color="yellow",
        mesh_panel_b_ytick_style="plain",
    )

    plt.show()
