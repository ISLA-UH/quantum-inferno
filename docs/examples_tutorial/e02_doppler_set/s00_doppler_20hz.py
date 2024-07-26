"""
Inferno example s00_doppler_20hz.
Doppler example: source and image from reflecting boundary

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from quantum_inferno import scales_dyadic
from quantum_inferno.plot_templates import plot_base as ptb
from quantum_inferno.plot_templates.plot_templates import plot_wf_mesh_vert
import quantum_inferno.synth.doppler as doppler
import quantum_inferno.synth.synthetics_NEW as synth

print(__doc__)


if __name__ == "__main__":
    # Define space 3D
    space_dimensions = 3
    # Signal speed in meters per second (mps), maps space to time.
    # Could be an input variable, but need to add Lorentz correction for light.
    # Source and receiver position
    distance_x_m = 1000.
    source_y_m = 10.
    source_z_m = 150.
    receiver_height_m = 1.
    # Convert to 3-element arrays to be converted to XYZ column matrices. Make a def
    source_position_vector_initial_xyz_m = np.array([-distance_x_m, source_y_m, source_z_m])
    source_position_vector_final_xyz_m = np.array([distance_x_m, source_y_m, source_z_m])
    # Total trajectory distance is known
    source_trajectory_m = doppler.range_scalar(source_position_vector_initial_xyz_m, source_position_vector_final_xyz_m)

    # Test with static receiver, move rx after verifying
    receiver_position_vector_initial_xyz_m = np.array([0., 0., receiver_height_m])
    receiver_position_vector_final_xyz_m = np.array([0., 0., receiver_height_m])
    # Total trajectory distance is known
    receiver_trajectory_m = doppler.range_scalar(receiver_position_vector_initial_xyz_m,
                                                 receiver_position_vector_final_xyz_m)

    # Source and receiver velocity
    signal_speed_mps = 340.
    source_speed_mps = 1 * 68.
    receiver_speed_mps = 0.

    source_mach_number = source_speed_mps / signal_speed_mps
    receiver_mach_number = receiver_speed_mps / signal_speed_mps
    print('Source speed:', source_speed_mps)
    print('Source Mach number:', source_mach_number)

    # HAVE ENOUGH INFORMATION TO SET TIME: FORWARDS VS INVERSE PROBLEM
    # FORWARDS PROBLEM
    tau_segment_s = source_trajectory_m / source_speed_mps
    tau_sample_rate_hz = 1000
    tau_number_samples = int(tau_segment_s * tau_sample_rate_hz)
    tau_source_s = np.arange(tau_number_samples) / tau_sample_rate_hz

    time_receiver_s, range_time_m, omega_over_omega_center = doppler.doppler_forward(
        tau_source_s,
        signal_speed_mps, source_speed_mps, receiver_speed_mps,
        space_dimensions,
        source_position_vector_initial_xyz_m, source_position_vector_final_xyz_m,
        receiver_position_vector_initial_xyz_m, receiver_position_vector_final_xyz_m)

    # print(source_position_vector_initial_xyz_m)
    image_time_receiver_s, image_range_time_m, image_omega_over_omega_center = doppler.image_doppler_forward(
        tau_source_s,
        signal_speed_mps, source_speed_mps, receiver_speed_mps,
        space_dimensions,
        source_position_vector_initial_xyz_m, source_position_vector_final_xyz_m,
        receiver_position_vector_initial_xyz_m, receiver_position_vector_final_xyz_m)
    # print(source_position_vector_initial_xyz_m)

    # Phase forwards-propagated to the source
    center_frequency_hz = 20.
    phase_radians_forward = 2. * np.pi * center_frequency_hz * tau_source_s

    # In the forward problem, the phase is going to be the same for the source and the image
    # Than have to reconcile time to add image. Alternatively, only superpose for the inverse problem.
    sig_source = np.sin(phase_radians_forward)
    # TODO: Deal with r=0 exception (if r<1)
    sig_receiver = sig_source / range_time_m
    image_sig_receiver = sig_source / image_range_time_m

    # Time of closest approach (CA) is tau/2
    index_ca = int(tau_number_samples / 2.)
    tau_ca_s = tau_source_s[index_ca]
    time_ca_s = time_receiver_s[index_ca]
    image_time_ca_s = image_time_receiver_s[index_ca]

    time_duration_source_s = doppler.time_duration(tau_source_s)
    time_duration_receiver_s = doppler.time_duration(time_receiver_s)
    image_time_duration_receiver_s = doppler.time_duration(image_time_receiver_s)

    # Compare duration
    print('Source time duration:', time_duration_source_s)
    print('Observer time of first arrival, last arrival, and signal duration:\n',
          time_receiver_s[0], time_receiver_s[-1], time_duration_receiver_s)
    print('Image time of first arrival, last arrival, and signal duration:',
          image_time_receiver_s[0], image_time_receiver_s[-1], image_time_duration_receiver_s)
    print('Duration Ratio, direct:', time_duration_receiver_s / time_duration_source_s)
    print('Duration Ratio, image:', image_time_duration_receiver_s / time_duration_source_s)

    # Use smallest overlaping interval. All segments must be calculated separately for source and image.

    """INVERSE PROBLEM
    Where the observed time is evenly sampled."""
    # MUST BE CAREFUL WITH INITIAL CONDITIONS! Not a reciprocal problem.
    # This needs must be explored
    inv_time_receiver_s_start = np.min(time_receiver_s)
    inv_time_receiver_s_end = np.max(time_receiver_s)
    inv_time_sample_rate_hz = 1. * tau_sample_rate_hz
    duration_inv_time_receiver_s = inv_time_receiver_s_end - inv_time_receiver_s_start
    inv_time_number_samples = int(duration_inv_time_receiver_s * inv_time_sample_rate_hz)

    inv_time_receiver_s = inv_time_receiver_s_start + np.arange(inv_time_number_samples) / inv_time_sample_rate_hz

    # This resets the boundary conditions. Only know receiver location at this time.

    inv_tau_source_s, inv_range_tau_m, inv_omega_over_omega_center = doppler.doppler_inverse(
        inv_time_receiver_s,
        signal_speed_mps, source_speed_mps, receiver_speed_mps,
        space_dimensions,
        source_position_vector_initial_xyz_m, source_position_vector_final_xyz_m,
        receiver_position_vector_initial_xyz_m, receiver_position_vector_final_xyz_m)
    print(source_position_vector_initial_xyz_m)
    # PROBLEM: same name, so not reverting - error

    image_inv_tau_source_s, image_inv_range_tau_m, image_inv_omega_over_omega_center = doppler.image_doppler_inverse(
        inv_time_receiver_s,
        signal_speed_mps, source_speed_mps, receiver_speed_mps,
        space_dimensions,
        source_position_vector_initial_xyz_m, source_position_vector_final_xyz_m,
        receiver_position_vector_initial_xyz_m, receiver_position_vector_final_xyz_m)
    print(source_position_vector_initial_xyz_m)

    inv_min_range_index = np.argmin(inv_range_tau_m)
    inv_time_min_range_s = inv_time_receiver_s[inv_min_range_index]

    # In the inverse problem, the receiver time is evenly sampled for both the direct and reflected (image) source.
    # Phase forwards-propagated to the source
    # TODO:DEF
    inv_phase_radians = 2. * np.pi * center_frequency_hz * inv_tau_source_s
    image_inv_phase_radians = 2. * np.pi * center_frequency_hz * image_inv_tau_source_s

    inv_sig_source = np.sin(inv_phase_radians)
    image_inv_sig_source = np.sin(image_inv_phase_radians)
    # TODO: Deal with r=0 exception (if r<1)
    inv_sig_receiver = inv_sig_source/inv_range_tau_m
    image_inv_sig_receiver = image_inv_sig_source / image_inv_range_tau_m
    inv_sig = inv_sig_receiver + image_inv_sig_receiver

    # Spectral analysis parameters
    title = 'Synthetic sound, direct and reflected wave, 1m receiver height'

    sig_wf = synth.sawtooth_doppler_noise_16bit(inv_phase_radians)/inv_range_tau_m + \
        synth.sawtooth_doppler_noise_16bit(image_inv_phase_radians)/image_inv_range_tau_m
    sig_wf /= np.abs(np.max(sig_wf))

    sample_frequency_hz = 1 * inv_time_sample_rate_hz
    number_points_period_int = int(sample_frequency_hz / center_frequency_hz)
    number_periods_nfft_window = 16
    nfft_int = int(number_periods_nfft_window * number_points_period_int)
    nfft_pow2 = 2**int(np.round(np.log2((number_periods_nfft_window * number_points_period_int))))
    # nfft = nfft_pow2  # TODO - reconcile frequency axes
    nfft = nfft_int
    print('\nNFFT number of center periods: ', number_periods_nfft_window)
    print('NFFT exact and nearest power of 2: ', nfft_int, nfft_pow2)
    print('NFFT duration, s: ', nfft_int / sample_frequency_hz, nfft_pow2 / sample_frequency_hz)
    print('Spectral resolution, Hz: ', sample_frequency_hz / float(nfft_int), sample_frequency_hz / float(nfft_pow2))

    # Spectrogram/Welch window specs
    nfft_tukey_percent = 100  # How much of the window is a cosine
    nfft_percent_overlap = 95

    # Convert to spect specs
    noverlap = int(nfft_percent_overlap / 100. * nfft)
    nfft_tukey_decimal = nfft_tukey_percent / 100.
    # Compute spectrogram
    f_center, t_center, synth_Sxx = \
        spectrogram(sig_wf, fs=sample_frequency_hz, window=('tukey', nfft_tukey_decimal),
                    nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend='constant',
                    return_onesided=True, scaling='spectrum', axis=-1, mode='psd')
    # Compute dBs and remove zero frequency for log scaling option
    synth_Sxx_db = 10 * np.log10(np.abs(synth_Sxx[1:, :]))
    # Normalize to max spectrum. Can normalize to center frequency.
    synth_Sxx_db -= np.max(synth_Sxx_db)

    # ### PLOT ###
    # plt.figure(1, figsize=(8, 7))
    # plt.subplot(211), plt.plot(tau_source_s-tau_ca_s, range_time_m)
    # plt.title('Range vs tau, m')
    # plt.grid(True)
    # plt.subplot(212), plt.plot(time_receiver_s-time_ca_s, range_time_m)
    # plt.title('Range vs t, m')
    # plt.xlabel('Time re CA, s')
    # plt.grid(True)

    plt.figure(1, figsize=(8, 7))
    plt.subplot(211), plt.plot(tau_source_s, range_time_m)
    plt.title('Range vs tau, m')
    plt.grid(True)
    plt.subplot(212), plt.plot(time_receiver_s, range_time_m)
    plt.title('Range vs t, m')
    plt.xlabel('Time, s')
    plt.grid(True)

    plt.figure(2, figsize=(8, 4))
    plt.plot(np.diff(time_receiver_s)/np.diff(tau_source_s))
    plt.plot(1/omega_over_omega_center, '.-')
    plt.title('Sample interval ratio, observer time over source time')
    plt.ylabel('dt/dtau')
    plt.xlabel('Sample number')
    plt.grid(True)

    plt.figure(3, figsize=(8, 9))
    plt.subplot(311), plt.plot(tau_source_s, sig_source)
    plt.title('Forwards problem: Source, tau')
    plt.subplot(312), plt.plot(time_receiver_s, sig_receiver)
    plt.title('Direct to Receiver vs time')
    plt.subplot(313), plt.plot(image_time_receiver_s, image_sig_receiver)
    plt.title('Source and image')
    plt.xlabel('Time, s')

    plt.figure(4, figsize=(8, 9))
    plt.subplot(311), plt.plot(inv_tau_source_s, inv_sig_source)
    plt.title('Inverse Problem: Source vs tau')
    plt.subplot(312), plt.plot(inv_time_receiver_s, inv_sig_receiver)
    plt.title('Direct to Receiver vs time')
    plt.subplot(313), plt.plot(inv_time_receiver_s, image_inv_sig_receiver)
    plt.title('Reflection to Receiver vs time')
    plt.xlabel('Time, s')

    # Shift relative to time of closest approach
    plot_inv_time_s = inv_time_receiver_s - inv_time_min_range_s
    plot_inv_spect_time_s = \
        t_center + inv_time_receiver_s[0] - inv_time_min_range_s + 0.5 * nfft / inv_time_sample_rate_hz

    # Plot against Tca
    plt.figure(5, figsize=(8, 6))
    plt.subplot(211), plt.plot(plot_inv_time_s, inv_sig_receiver + image_inv_sig_receiver)
    plt.title('Inverse Problem: ' + title)
    plt.subplot(212), plt.plot(plot_inv_time_s, inv_range_tau_m)
    plt.title('Range vs time')
    plt.xlabel('Time relative to closest approach, s')

    ###################
    # Plot spectrograms
    synth_type = title

    time_wf_s = plot_inv_time_s
    time_spect_s = plot_inv_spect_time_s

    figure_number = 3
    # time_s_xmin = t_center[0]
    # time_s_xmax = t_center[-1]
    time_s_xmax = -plot_inv_spect_time_s[0]
    time_s_xmin = plot_inv_spect_time_s[0]
    # time_s_xmax = time_sensor_s[-1]
    frequency_hz_ymin = 5.
    frequency_hz_ymax = sample_frequency_hz / 2.

    frequency_yscale = 'linear'
    # frequency_yscale = 'log'
    dB_range = 70
    dB_colormax = np.max(synth_Sxx_db)
    dB_colormin = dB_colormax - dB_range

    plt.style.use('dark_background')

    wf_base = ptb.WaveformBase("synthetic", "STFT of Doppler Shift", waveform_color="yellow")
    wf_panel = ptb.WaveformPanel(sig_wf, time_wf_s)
    mesh_base = ptb.MeshBase(t_center, f_center, frequency_scaling=frequency_yscale,
                             frequency_hz_ymin=frequency_hz_ymin, frequency_hz_ymax=frequency_hz_ymax,
                             colormap="inferno")
    mesh_panel = ptb.MeshPanel(np.log2(synth_Sxx + scales_dyadic.EPSILON32),
                               colormap_scaling="auto", ytick_style="plain")
    stft = plot_wf_mesh_vert(wf_base, wf_panel, mesh_base, mesh_panel)

    plt.show()
