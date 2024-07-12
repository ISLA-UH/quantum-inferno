"""
An example comparing the legacy stft function and the ShortTimeFFT class in the scipy.signal module.

The example signal is a 60 Hz tone with a 10.24s duration, 800 Hz sample rate, split into 0.64s segments.
A Tukey taper with 25% alpha is applied to each STFT window along with a "constant" de-trend.

The number of points of the full signal is 10.24 * 800 = 8192.
The window length is 0.64 * 800 = 512 points.
The overlap length is 256 points (50% overlap).

The number of time bins for the STFT is 33 (8192 / 256 + 1), where the additional bin is due last window overlap.
It may be beneficial to remove the last time bin, as it is not a full window.
Additionally, for scipy.signal.istft, the function assumes the last window is a not full window.
However, the ShortTimeFFT class assumes the last window is a full window and will return a longer time series.
The method used in this examples, istft_tukey, will return a time series of the same length as the input signal.

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
    tukey_alpha = 0.25  # 25% Tukey (Cosine) window [if zero, rectangular window; if one, Hann window]

    # Compute the welch PSD of the signal using the scipy.signal.welch function for comparison
    frequency_welch_hz, psd_welch_power = signal.welch(
        x=signal_timeseries,
        fs=signal_sample_rate_hz,
        window=("tukey", tukey_alpha),
        nperseg=signal_number_of_fft_points,
        noverlap=signal_number_of_fft_points // 2,
        nfft=signal_number_of_fft_points,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        scaling="spectrum",
        average="mean",
    )

    # Compute the STFT of the signal using the scipy.signal.stft function
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
        scaling="magnitude",
    )

    # Since one-sided, multiply by 2 to get the full power
    stft_power = 2 * np.abs(stft_magnitudes) ** 2
    ShortTimeFFT_power = 2 * np.abs(ShortTimeFFT_magnitudes) ** 2

    # Compute the ratio of the PSD to the variance by averaging over the columns for the STFT
    welch_over_variance = psd_welch_power / signal_variance
    stft_over_variance = np.average(stft_power, axis=1) / signal_variance
    ShortTimeFFT_over_variance = np.average(ShortTimeFFT_power, axis=1) / signal_variance

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
        f"The STFTs (Power) are the same within taper (tolerance: 1e-3): "
        f"{np.allclose(stft_power[3:-3], ShortTimeFFT_power[3:-3], atol=1e-3)}"
    )
    print(
        f"The average difference between the two STFT powers within taper: "
        f"{np.mean(np.abs(stft_power[3:-3] - ShortTimeFFT_power[3:-3]))}"
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

    istft_ShortTimeFFT_time_s, istft_ShortTimeFFT_timeseries = istft_tukey(
        stft_to_invert=ShortTimeFFT_magnitudes,
        sample_rate_hz=signal_sample_rate_hz,
        tukey_alpha=tukey_alpha,
        segment_length=signal_number_of_fft_points,
        overlap_length=signal_number_of_fft_points // 2,
        scaling="magnitude",
    )

    # check if same thing happens when getting the object
    print("\n### Check the ShortTimeFFT object STFT and ISTFT and the signal length discrepancy ###")
    short_time_fft_object = get_stft_object_tukey(
        sample_rate_hz=signal_sample_rate_hz,
        tukey_alpha=tukey_alpha,
        segment_length=signal_number_of_fft_points,
        overlap_length=signal_number_of_fft_points // 2,
    )
    check_stft = short_time_fft_object.stft_detrend(signal_timeseries, "constant")
    check_istft = short_time_fft_object.istft(check_stft)

    print(f"Signal length: {len(signal_timeseries)}, duration: {len(signal_timeseries) / signal_sample_rate_hz}")
    print(
        f"Signal length divided by segment length: {int(len(signal_timeseries) / (signal_number_of_fft_points // 2))} "
        f"\nSTFT time bins length: {len(check_stft[0])}"
    )
    print(f"Reconstructed signal length: {len(check_istft)}, difference: {len(check_istft) - len(signal_timeseries)}")

    # Plot
    event_name = str(tone_frequency_hz) + " Hz Tone Test"
    fmin = 2 * signal_frequency_resolution_fft_hz
    fmax = signal_sample_rate_hz / 2  # Nyquist

    plt.figure(figsize=(8, 6))
    plt.title(f"Power / Variance Comparison, f = {signal_frequency_center_fft_hz:.3f} Hz")
    plt.plot(frequency_welch_hz, welch_over_variance, label="Welch PSD / Variance", alpha=0.8)
    plt.plot(stft_frequencies, stft_over_variance, ".-", label="STFT PSD / Variance", alpha=0.8)
    plt.plot(ShortTimeFFT_frequencies, ShortTimeFFT_over_variance, "--", label="ShortTimeFFT PSD / Variance", alpha=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD / Variance")
    plt.xscale("log")
    plt.xlim([0.8, signal_sample_rate_hz / 2])
    plt.grid()
    plt.legend()

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
        wf_panel_a_sig=signal_timeseries - istft_ShortTimeFFT_timeseries,
        wf_panel_a_time=istft_ShortTimeFFT_time_s,
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
        wf_panel_a_sig=istft_stft_timeseries - istft_ShortTimeFFT_timeseries,
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
