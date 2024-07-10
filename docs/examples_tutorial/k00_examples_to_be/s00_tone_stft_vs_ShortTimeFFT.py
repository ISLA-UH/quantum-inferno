"""
An example comparing the legacy stft function and the ShortTimeFFT class in the scipy.signal module.

The example signal is a 60 Hz tone with a 10.24s duration, 800 Hz sample rate, split into 0.64s segments.
A Tukey taper with 25% alpha is applied to each STFT window along with a "constant" de-trend.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from quantum_inferno.synth import benchmark_signals
import quantum_inferno.plot_templates.plot_cyberspectral as pltq
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

from quantum_inferno.utilities.short_time_fft import stft_tukey, istft_tukey, get_stft_object_tukey

print(__doc__)

if __name__ == "__main__":
    # Create a tone of fixed frequency
    tone_frequency_hz = 60
    sample_rate_hz = 800
    duration_s = 10.24
    segment_duration_s = 0.64

    [
        signal_timeseries,
        signal_times_s,
        signal_number_of_fft_points,
        signal_sample_rate_hz,
        signal_frequency_center_fft_hz,
        signal_frequency_resolution_fft_hz,
    ] = benchmark_signals.well_tempered_tone(
        frequency_center_hz=tone_frequency_hz,
        frequency_sample_rate_hz=sample_rate_hz,
        time_duration_s=duration_s,
        time_fft_s=segment_duration_s,
        use_fft_frequency=True,
        add_noise_taper_aa=True,
    )

    signal_variance = np.var(signal_timeseries)
    signal_variance_nominal = 1 / 2.0

    # Compute the STFT of the signal using the scipy.signal.stft function
    tukey_alpha = 0.25  # 25% Tukey (Cosine) window [if zero, rectangular window; if one, Hann window]

    stft_frequencies, stft_times, stft_magnitudes = signal.stft(
        x=signal_timeseries,
        fs=signal_sample_rate_hz,
        window=("tukey", tukey_alpha),
        nperseg=signal_number_of_fft_points,
        noverlap=signal_number_of_fft_points // 2,  # 50% overlap
        nfft=signal_number_of_fft_points,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        boundary="zeros",
        padded=True,
    )

    # Compute the STFT of the signal using the short_time_fft.stft_tukey function which uses scipy.signal.ShortTimeFFT
    ShortTimeFFT_frequencies, ShortTimeFFT_times, ShortTimeFFT_magnitudes = stft_tukey(
        timeseries=signal_timeseries,
        sample_rate_hz=signal_sample_rate_hz,
        tukey_alpha=tukey_alpha,
        segment_length=signal_number_of_fft_points,
        overlap_length=signal_number_of_fft_points // 2,  # 50% overlap
    )

    # Since one-sided, multiply by 2 to get the full power
    stft_power = 2 * np.abs(stft_magnitudes) ** 2
    ShortTimeFFT_power = 2 * np.abs(ShortTimeFFT_magnitudes) ** 2

    # Compute the ratio of the PSD to the variance by averaging over the columns for the STFT
    stft_over_var = np.average(stft_power, axis=1) / signal_variance
    ShortTimeFFT_over_var = np.average(ShortTimeFFT_power, axis=1) / signal_variance

    # Conver to bits [log2(power)] with epsilon to avoid log(0)
    stft_bits = to_log2_with_epsilon(stft_power)
    ShortTimeFFT_bits = to_log2_with_epsilon(ShortTimeFFT_power)

    # Check some values for the STFT to make sure they are the same
    print("### scipy.signal.stft vs scipy.signal.ShortTimeFFT ###")
    print(f"The shape of the STFT matches the ShortTimeFFT: {np.shape(stft_power) == np.shape(ShortTimeFFT_power)}")
    print(f"The expected maximum power of the STFT is 0.5")
    print(f"The computed maximum power with scipy.signal.stft is {np.max(stft_power)}")
    print(f"The computed maximum power with scipy.signal.ShortTimeFFT is {np.max(ShortTimeFFT_power)}")
    print(
        f"The STFTs are the same within taper: {np.allclose(stft_magnitudes[3:-3], ShortTimeFFT_magnitudes[3:-3], atol=1e-4)}"
    )
    print(
        f"The average difference between the two STFT powers within taper: {np.mean(np.abs(stft_power[3:-3] - ShortTimeFFT_power[3:-3]))}"
    )

    # Convert the stft and ShortTimeFFT back to a time series
    istft_stft_time_s, istft_stft_timeseries = signal.istft(
        Zxx=stft_magnitudes,
        fs=signal_sample_rate_hz,
        window=("tukey", tukey_alpha),
        nperseg=signal_number_of_fft_points,
        noverlap=signal_number_of_fft_points // 2,
        nfft=signal_number_of_fft_points,
        input_onesided=True,
        boundary=True,
        time_axis=-1,
        freq_axis=-2,
    )

    # TODO: Fix the istft_tukey function to match the scipy.signal.istft function [currently creates extra]
    istft_ShortTimeFFT_timeseries = istft_tukey(
        stft_magnitude=ShortTimeFFT_magnitudes,
        sample_rate_hz=signal_sample_rate_hz,
        tukey_alpha=tukey_alpha,
        segment_length=signal_number_of_fft_points,
        overlap_length=signal_number_of_fft_points // 2,
    )

    # check if same thing happens when getting the object
    short_time_fft_object = get_stft_object_tukey(
        sample_rate_hz=signal_sample_rate_hz,
        tukey_alpha=tukey_alpha,
        segment_length=signal_number_of_fft_points,
        overlap_length=signal_number_of_fft_points // 2,
    )

    check_stft = short_time_fft_object.stft_detrend(signal_timeseries, "constant")
    check_istft = short_time_fft_object.istft(check_stft)
    print(len(check_istft) / signal_sample_rate_hz)

    # Plot
    event_name = str(tone_frequency_hz) + " Hz Tone Test"

    print(
        len(signal_timeseries) / signal_sample_rate_hz,
        len(istft_stft_timeseries) / signal_sample_rate_hz,
        len(istft_ShortTimeFFT_timeseries) / signal_sample_rate_hz,
    )
    print(np.shape(stft_power), np.shape(ShortTimeFFT_power))

    # Select plot frequencies
    fmin = 2 * signal_frequency_resolution_fft_hz
    fmax = signal_sample_rate_hz / 2  # Nyquist

    pltq.plot_wf_mesh_vert(
        station_id=", Log2(1/2)=-1",
        wf_panel_a_sig=signal_timeseries - istft_stft_timeseries,
        wf_panel_a_time=signal_times_s,
        mesh_time=stft_times,
        mesh_frequency=stft_frequencies,
        mesh_panel_b_tfr=stft_bits,
        mesh_panel_b_colormap_scaling="range",
        wf_panel_a_units="istft difference to original",
        mesh_panel_b_cbar_units="Log2(Power)",
        start_time_epoch=0,
        figure_title="scipy.signal.stft for " + event_name,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    pltq.plot_wf_mesh_vert(
        station_id=", Log2(1/2)=-1",
        wf_panel_a_sig=signal_timeseries - istft_ShortTimeFFT_timeseries[: len(signal_timeseries)],
        wf_panel_a_time=signal_times_s,
        mesh_time=ShortTimeFFT_times,
        mesh_frequency=ShortTimeFFT_frequencies,
        mesh_panel_b_tfr=ShortTimeFFT_bits,
        mesh_panel_b_colormap_scaling="range",
        wf_panel_a_units="istft difference to original",
        mesh_panel_b_cbar_units="Log2(Power)",
        start_time_epoch=0,
        figure_title="scipy.signal.ShortTimeFFT for " + event_name,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    pltq.plot_wf_mesh_vert(
        station_id="-9 dB = 0.126, 9 dB = 7.94",
        wf_panel_a_sig=istft_stft_timeseries - istft_ShortTimeFFT_timeseries[: len(signal_timeseries)],
        wf_panel_a_time=istft_stft_time_s,
        mesh_time=stft_times,
        mesh_frequency=stft_frequencies,
        mesh_panel_b_tfr=(10 * np.log10(ShortTimeFFT_power / stft_power)),
        mesh_panel_b_colormap_scaling="else",
        mesh_colormap="seismic",
        wf_panel_a_units="istft difference",
        mesh_panel_b_cbar_units="dB",
        mesh_panel_b_color_min=-9,
        mesh_panel_b_color_max=9,
        mesh_panel_b_color_range=18,
        start_time_epoch=0,
        figure_title="(10*LOG10(signal.ShortTimeFFT / signal.stft)) for " + event_name,
        frequency_hz_ymin=fmin,
        frequency_hz_ymax=fmax,
    )

    plt.show()
