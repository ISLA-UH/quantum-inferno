"""
An example computing STFTs with different scaling using the ShortTimeFFT class in the scipy.signal module.

The example signal is a 60 Hz tone with a 10.24s duration, 800 Hz sample rate, split into 0.64s segments.
A Tukey taper with 25% alpha is applied to each STFT window along with a "constant" de-trend.

The number of points of the full signal is 10.24 * 800 = 8192.
The window length is 0.64 * 800 = 512 points.
The overlap length is 256 points (50% overlap).

The expected nominal variance of the signal is 1/2 = 0.5
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from quantum_inferno.synth import benchmark_signals
import quantum_inferno.plot_templates.plot_cyberspectral as pltq
from quantum_inferno.utilities.rescaling import to_log2_with_epsilon

from quantum_inferno.utilities.short_time_fft import stft_tukey, istft_tukey, get_stft_object_tukey

print(__doc__)
# TODO: ADD SPECTROGRAM COMPARISON


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
    print(f"Calculated signal variance: {signal_variance:.3f} (nominal: {signal_variance_nominal:.3f})")
    tukey_alpha = 0.25  # 25% Tukey (Cosine) window [if zero, rectangular window; if one, Hann window]

    # Plot a snapshot (first two seconds) of the signal
    plt.figure(figsize=(10, 6))
    plt.plot(signal_times_s[: int(800 * 2)], signal_timeseries[: int(800 * 2)])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Snapshot of the 60 Hz tone signal (first two seconds)")
    plt.grid(True)

    # Compute the welch PSD of the signal using the scipy.signal.welch function for comparison
    welch_frequency_hz, welch_psd = signal.welch(
        x=signal_timeseries,
        fs=signal_sample_rate_hz,
        window=("tukey", tukey_alpha),
        nperseg=signal_number_of_fft_points,
        noverlap=signal_number_of_fft_points // 2,
        nfft=signal_number_of_fft_points,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        scaling="density",
        average="mean",
    )

    # Compute the welch spectrum of the signal using the scipy.signal.welch function for comparison
    _, welch_spectrum = signal.welch(
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

    # Compute the STFT magnitudes of the signal using the ShortTimeFFT class
    ShortTimeFFT_frequencies, ShortTimeFFT_times, ShortTimeFFT_magnitudes = stft_tukey(
        timeseries=signal_timeseries,
        sample_rate_hz=signal_sample_rate_hz,
        tukey_alpha=tukey_alpha,
        segment_length=signal_number_of_fft_points,
        overlap_length=signal_number_of_fft_points // 2,  # 50% overlap
        scaling="magnitude",
    )

    # Compute the STFT power spectral density of the signal using the ShortTimeFFT class
    _, _, ShortTimeFFT_psd = stft_tukey(
        timeseries=signal_timeseries,
        sample_rate_hz=signal_sample_rate_hz,
        tukey_alpha=tukey_alpha,
        segment_length=signal_number_of_fft_points,
        overlap_length=signal_number_of_fft_points // 2,  # 50% overlap
        scaling="psd",
    )

    # Plot the Welch PSD and the ShortTimeFFT PSD
    plt.figure(figsize=(10, 6))
    plt.plot(
        welch_frequency_hz,
        signal_frequency_resolution_fft_hz * welch_psd / signal_variance,
        ".--",
        label="Welch, PSD",
        alpha=0.75,
    )
    plt.plot(
        ShortTimeFFT_frequencies,
        signal_frequency_resolution_fft_hz * 2 * np.mean(np.abs(ShortTimeFFT_psd) ** 2, axis=1) / signal_variance,
        ".-",
        label="ShortTimeFFT, PSD",
        alpha=0.75,
    )
    plt.xlim([signal_frequency_center_fft_hz - 10, signal_frequency_center_fft_hz + 10])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("df $\\cdot$ PSD/VAR(signal)")
    plt.title("PSD Scaling: Welch vs ShortTimeFFT.stft")
    plt.legend()
    plt.grid(True)

    # Plot the Welch spectrum and the ShortTimeFFT magnitudes
    plt.figure(figsize=(10, 6))
    plt.plot(welch_frequency_hz, welch_spectrum / signal_variance, ".-", label="Welch, Spectrum", alpha=0.75)
    plt.plot(
        ShortTimeFFT_frequencies,
        np.mean(2 * np.abs(ShortTimeFFT_magnitudes) ** 2, axis=1) / signal_variance,
        ".--",
        label="ShortTimeFFT, Magnitudes",
        alpha=0.75,
    )
    plt.xlim([signal_frequency_center_fft_hz - 10, signal_frequency_center_fft_hz + 10])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spectrum/VAR(signal)")
    plt.title("Spectrum Scaling: Welch vs ShortTimeFFT.stft")
    plt.legend()
    plt.grid(True)

plt.show()
