"""
libquantum example: s00_grain_tfr

"""
import numpy as np
import matplotlib.pyplot as plt
from quantum_inferno import styx_fft, styx_cwt, scales_dyadic
import quantum_inferno.plot_templates.plot_cyberspectral as pltq
print(__doc__)


if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    EVENT_NAME = 'grain test'
    station_id_str = 'synth'

    # Specifying the grain parameters requires some forethought
    frequency_center_hz = 5
    frequency_sample_rate_hz = 80
    order_number_input = 6

    # TODO: ADD Averaging frequency for fft_nd
    time_nd = 2**9
    time_fft_nd = 2**6

    # The CWX and STX will be evaluated from the number of points in FFT of the signal
    frequency_cwt_pos_hz = np.fft.rfftfreq(time_nd, d=1/frequency_sample_rate_hz)
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
    print('These two should coincide for a fair comparison')
    print('Center CWT FFT frequency, Hz:', frequency_center_cwt_hz)
    print('Center STFT FFT frequency, Hz:', frequency_center_stft_hz)

    # exit()
    # TODO: Note oversampling on CWT leads to overestimation of energy!!
    frequency_cwt_fft_hz = frequency_stft_pos_hz[2:]
    # frequency_inferno_hz = frequency_cwt_pos_hz[1:]

    mic_sig_complex, time_s, scale, omega, amp = \
        styx_cwt.wavelet_centered_4cwt(band_order_Nth=order_number_input,
                                       duration_points=time_nd,
                                       scale_frequency_center_hz=frequency_center_stft_hz,
                                       frequency_sample_rate_hz=frequency_sample_rate_hz,
                                       dictionary_type="norm")
    mic_sig_real = np.real(mic_sig_complex)
    mic_sig_imag = np.imag(mic_sig_complex)

    # Computed Variance; divides by the number of points
    mic_sig_real_var = np.var(mic_sig_real)
    mic_sig_imag_var = np.var(mic_sig_imag)

    # Theoretical variance TODO: construct function
    mic_sig_real_var_nominal = amp**2/len(time_s) * 0.5*np.sqrt(np.pi)*scale * \
                               (1 + np.exp(-(scale*omega)**2))
    mic_sig_imag_var_nominal = amp**2/len(time_s) * 0.5*np.sqrt(np.pi)*scale * \
                               (1 - np.exp(-(scale*omega)**2))

    # Mathematical integral ~ computed Variance * Number of Samples. The dictionary type = "norm" returns 1/2.
    mic_sig_real_integral = np.var(mic_sig_real)*len(mic_sig_real)
    mic_sig_imag_integral = np.var(mic_sig_imag)*len(mic_sig_real)

    print('\nAtom Variance')
    print('mic_sig_real_variance:', mic_sig_real_var)
    print('real_variance_nominal:', mic_sig_real_var_nominal)
    print('mic_sig_imag_variance:', mic_sig_imag_var)
    print('imag_variance_nominal:', mic_sig_imag_var_nominal)

    # Choose the real component as the test signal
    mic_sig = np.copy(mic_sig_real)
    mic_sig_var = mic_sig_real_var
    mic_sig_var_nominal = mic_sig_real_var_nominal
    print('\nChoose real part as signal:')
    print('var/nominal var:', mic_sig_var/mic_sig_var_nominal)

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, psd_welch_power = \
        styx_fft.welch_power_pow2(sig_wf=mic_sig,
                                  frequency_sample_rate_hz=frequency_sample_rate_hz,
                                  segment_points=time_fft_nd)

    # Information overload methods
    welch_power, welch_power_per_band, welch_power_per_sample, welch_power_total, welch_power_scaled, \
    welch_information_bits, welch_information_bits_per_band, welch_information_bits_per_sample, \
    welch_information_bits_total, welch_information_scaled = styx_fft.power_and_information_shannon_welch(psd_welch_power)
    
    # # STFT with 25% Tukey window
    # frequency_stft_hz, time_stft_s, stft_complex = \
    #     styx_fft.stft_complex_pow2(sig_wf=mic_sig,
    #                                frequency_sample_rate_hz=frequency_sample_rate_hz,
    #                                segment_points=time_fft_nd)

    # STFT with Gaussian window
    frequency_stft_hz, time_stft_s, stft_complex = \
        styx_fft.gtx_complex_pow2(sig_wf=mic_sig,
                                  frequency_sample_rate_hz=frequency_sample_rate_hz,
                                  segment_points=time_fft_nd)

    # Information overload methods
    stft_power, stft_power_per_band, stft_power_per_sample, stft_power_total, stft_power_scaled, \
    stft_information_bits, stft_information_bits_per_band, stft_information_bits_per_sample, \
    stft_information_bits_total, stft_information_scaled = styx_fft.power_and_information_shannon_stft(stft_complex)

    # Scale power by variance
    welch_over_var = psd_welch_power / mic_sig_var
    stft_over_var = np.average(stft_power, axis=1) / mic_sig_var

    print('\nSum scaled spectral power')
    print('Sum Welch:', np.sum(welch_over_var))
    print('Sum STFT:', np.sum(stft_over_var))

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title('Synthetic Gabor wavelet with taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_welch_hz, welch_over_var, label='Welch')
    ax2.semilogx(frequency_stft_hz, stft_over_var, '.-', label="STFT")

    ax2.set_title('Welch and Spect FFT (RMS), f = ' + str(round(frequency_center_stft_hz * 100) / 100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('Power / var(sig)')
    ax2.grid(True)
    ax2.legend()

    # plt.show()
    # exit()
    # plt.figure()
    # plt.plot(cwt_information_bits_per_sample)

    # Select plot frequencies
    fmin = frequency_resolution_stft_hz
    fmax = frequency_sample_rate_hz/2  # Nyquist

    pltq.plot_wf_mesh_vert(station_id='00',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_stft_s,
                           mesh_frequency=frequency_stft_hz,
                           mesh_panel_b_tfr=np.log2(stft_power + scales_dyadic.EPSILON64),
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="stft for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    plt.show()

