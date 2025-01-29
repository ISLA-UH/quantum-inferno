"""
Quantum inferno example: s01_tone_fft_welch.py
Compute Welch power spectral density (PSD) on simple tone to verify amplitudes
Case study: Real-valued cosine input with unit amplitude
Validate: Nominal Welch power averaged over the signal duration is 1/2
See also: https://docs.scipy.org/doc/scipy-1.15.1/tutorial/signal.html#tutorial-spectralanalysis

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from quantum_inferno.synth import benchmark_signals

print(__doc__)


if __name__ == "__main__":
    """
    Average the Fast Fourier Transform (FFT) over sliding windows using the Welch method
    Inspect and verify amplitude corrections for different scalings (density, spectrum)
    """

    # Construct a tone of fixed frequency with a constant sample rate
    # For this example, no noise, taper, or antialiasing filters are added.
    # In the first example (FFT), the nominal signal duration was 1s.
    # In this example the nominal signal duration is 16s, with averaging (fft) window duration of 1s.
    frequency_tone_hz = 60
    [
        mic_sig,
        time_s,
        time_fft_nd,
        frequency_sample_rate_hz,
        frequency_center_fft_hz,
        frequency_resolution_fft_hz,
    ] = benchmark_signals.well_tempered_tone(
        frequency_center_hz=frequency_tone_hz,
        frequency_sample_rate_hz=800,
        time_duration_s=16,
        time_fft_s=1,
        use_fft_frequency=True,
        add_noise_taper_aa=False,
        output_desc=False
    )

    # alpha: Shape parameter of the Welch Tukey window, representing the fraction of the window inside the cosine
    #        tapered region.
    # If zero [0], the Tukey window is equivalent to a rectangular window.
    # If one [1], the Tukey window is equivalent to a Hann window.
    alpha = 0

    # Compute Variance: divides by the number of points
    mic_sig_var = np.var(mic_sig)
    mic_sig_var_nominal = 1 / 2.0
    print("mic_sig_var/mic_sig_var_nominal", mic_sig_var/mic_sig_var_nominal)

    # Compute the Welch PSD: averaged spectrum over sliding windows
    # Use the spectrum scaling and explicit defaults for nfft and noverlap
    frequency_welch_hz, power_welch_spectrum = signal.welch(
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

    # The density scaling option is the spectrum divided by the spectral resolution.
    # The density option also depends more on the window type, can verify by varying alpha.
    _, power_welch_density = signal.welch(
        x=mic_sig,
        fs=frequency_sample_rate_hz,
        window=("tukey", alpha),
        nperseg=time_fft_nd,
        noverlap=time_fft_nd // 2,
        nfft=time_fft_nd,
        detrend="constant",
        return_onesided=True,
        axis=-1,
        scaling="density",
        average="mean",
    )

    print("Welch returns only the positive frequencies")
    print("len(Pxx):", len(frequency_welch_hz))

    # The spectrum option returns the var at the peak, which for a tone will have a var of 1/2
    fft_welch_over_var = power_welch_spectrum / mic_sig_var
    fft_welch_psd_times_df_over_var = frequency_resolution_fft_hz * power_welch_density / mic_sig_var

    print("\n*** SUMMARY: Welch spectral power estimates for a constant-frequency tone  ***")
    print("The Welch spectral estimate averages the FFT over overlapping windows.")
    print(
        "For the Welch spectrum scaling, "
        "the variance of an untapered unit amplitude tone with no DC offset is P**2 = 1/2 = var(signal)"
    )
    print("The Welch density scaling is divided by the spectral resolution")
    print("** IMPORTANT NOTE: THE WELCH METHOD IMPLEMENTS FACTOR OF 2 POWER CORRECTION FOR NEGATIVE FREQUENCIES **")

    # Show the waveform and the averaged FFT over the whole record:
    fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic CW, no taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax2.semilogx(frequency_welch_hz, fft_welch_over_var, label="Welch spectrum/var")
    ax2.semilogx(frequency_welch_hz, fft_welch_psd_times_df_over_var, ".--", label="df $\\cdot$ Welch density/var")
    ax2.set_title(f"Welch PSD/VAR(signal), f = {frequency_center_fft_hz:.3f} Hz")
    ax2.set_xlabel("Frequency, Hz")
    ax2.set_ylabel("Power/VAR(signal)")
    ax2.grid(True)
    ax2.legend()

    # Show the waveform and the averaged FFT over the whole record:
    fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title("Synthetic CW, no taper")
    ax1.set_xlabel("Time, s")
    ax1.set_ylabel("Norm")
    ax2.semilogx(frequency_welch_hz, power_welch_spectrum, label="Welch spectrum")
    ax2.semilogx(frequency_welch_hz, frequency_resolution_fft_hz * power_welch_density, ".--",
                 label="df $\\cdot$ Welch density")
    ax2.set_title(f"Welch PSD, f = {frequency_center_fft_hz:.3f} Hz")
    ax2.set_xlabel("Frequency, Hz")
    ax2.set_ylabel("Power")
    ax2.grid(True)
    ax2.legend()

    plt.show()
