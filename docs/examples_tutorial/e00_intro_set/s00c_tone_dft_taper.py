"""Quantum inferno example: s00b_tone_dft_taper.py

Introduction to Time-Frequency Representations (TFRs).
Compute Discrete (DFT) and Fast Fourier Transform (FFT) on a
simple tone to verify peak and average signal power with a taper
window applied to the input signal. See ``tone_dft_intro.py`` for
a similar example without tapering.

Case study: sinusoid input with unit amplitude. Zero padding makes
the FFT length a power of two.

Validate nominal DFT and FFT power estimates with taper
correction and basic variance metrics.

See also:
https://docs.scipy.org/doc/scipy/tutorial/fft.html
https://docs.scipy.org/doc/scipy-1.15.1/tutorial/signal.html#tutorial-spectralanalysis
"""
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import get_window

print(__doc__)

# List of taper windows to apply to the input signal in this example
# The default, implicit taper is a rectangular window.
window_types: list[str] = [
    'boxcar', 'gaussian', 'hann', 'tukey', 'blackman'
]

if __name__ == "__main__":
    # Simple constant-amplitude sinusoid example (no taper by default).
    # Construct a tone at a fixed frequency with a given sample rate.
    frequency_sample_rate_hz = 800.
    frequency_design_center_hz = 60.
    sig_amplitude = 1.  # Nominal amplitude in some physical units
    sig_duration_s = 60.  # Nominal input signal duration in seconds
    # Compute DFT number of digital points. Note int() rounds down.
    dft_samples: int = int(sig_duration_s * frequency_sample_rate_hz)
    # Dimensionless time (cyber time) from input record cyber duration.
    time_cyber = np.arange(dft_samples)
    # Standard deviation for the Gaussian window
    gauss_std = (dft_samples - 1) / np.pi / 2

    def get_scipy_window(
        win_name: str, num_samples: int, fftbins: bool = True
    ) -> Any:
        # Select a taper window from the list of available windows.
        # The Gaussian window requires a standard deviation argument.
        if win_name == 'gaussian':
            return get_window((win_name, gauss_std), num_samples,
                              fftbins=fftbins)

        return get_window(win_name, num_samples, fftbins=fftbins)

    # Dimensionless center frequency:
    frequency_center = frequency_design_center_hz / frequency_sample_rate_hz

    # Construct synthetic tone at the design center frequency.
    # No anti-alias preconditioning is applied because the center
    # frequency is well below Nyquist for this example.
    sig_cosine = (
        sig_amplitude
        * np.cos(2 * np.pi * frequency_center * time_cyber)
    )
    sig_var = np.var(sig_cosine)
    sig_var = np.var(sig_cosine)

    """Design the FFT performance by specifying a target spectral
    resolution in physical space. This corresponds to the linear
    frequency equivalent of specifying dyadic scale order.
    """
    # The DFT spectral resolution is set by the record duration
    frequency_resolution_dft = 1.0 / dft_samples
    frequency_resolution_dft_hz = (
        frequency_sample_rate_hz * frequency_resolution_dft
    )

    # The DFT does not need to be a power of two; FFTs are used for speed.
    # Specify the desired spectral resolution to design the FFT length.
    fft_design_resolution_hz = 0.25 * frequency_resolution_dft_hz
    # The spectral resolution sets the nominal record duration for the FFT.
    fft_design_time_s = 1.0 / fft_design_resolution_hz

    # FFTs are most efficient at power-of-two lengths. Truncate or pad
    # DFT input to meet this requirement where appropriate.
    if fft_design_resolution_hz > frequency_resolution_dft_hz:
        print('*** STOP: FFT design resolution > DFT resolution. Exiting. ***')
        exit()

    # Finer spectral resolution: increase the number of points and pad with zeros.
    fft_samples = 2 ** (
        int(np.ceil(np.log2(fft_design_time_s * frequency_sample_rate_hz)))
    )

    # The dyadic spectral resolution is modified accordingly
    frequency_resolution_fft = 1.0 / fft_samples
    frequency_resolution_fft_hz = (
        frequency_sample_rate_hz * frequency_resolution_fft
    )

    # DFT estimation
    # Use only positive frequencies
    frequency_dft_pos_hz = scipy.fft.rfftfreq(
        dft_samples, d=1 / frequency_sample_rate_hz
    )
    # Find the closest frequency to the design frequency. Cast to int.
    dft_index = int(
        np.argmin(np.abs(frequency_dft_pos_hz - frequency_design_center_hz))
    )
    frequency_center_dft_hz = frequency_dft_pos_hz[dft_index]
    # For display
    frequency_fft_over_df = scipy.fft.rfftfreq(fft_samples, d=1 / dft_samples)
    # The new FFT duration (power of two) defines the spectral resolution.
    frequency_fft_pos_hz = scipy.fft.rfftfreq(
        fft_samples, d=1 / frequency_sample_rate_hz
    )
    fft_peak_index = int(
        np.argmin(np.abs(frequency_fft_pos_hz - frequency_design_center_hz))
    )
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_peak_index]
    # Convert to dimensionless frequency
    frequency_center_fft = frequency_center_fft_hz / frequency_sample_rate_hz

    print('DFT samples:', dft_samples)
    print('log2(DFT samples):', np.log2(dft_samples))
    print('FFT samples:', fft_samples)
    print('log2(FFT samples):', np.log2(fft_samples))


    """
    Study taper windows to reduce spectral leakage.
    """
    # Create a figure to show the taper windows
    fg1, ax1 = plt.subplots(
        len(window_types), 1, sharex='all', sharey='all', figsize=(6., 4.)
    )
    for c_, (w_name_, ax_) in enumerate(zip(window_types, ax1)):
        # Select a taper window from the list of available windows
        win_taper = get_scipy_window(w_name_, dft_samples, fftbins=False)
        ax_.plot(win_taper, f'C{c_}-', label=w_name_)
        ax_.text(0.1, 0.5, w_name_, color=f'C{c_}', verticalalignment='bottom',
                 horizontalalignment='left', bbox={'color': 'white', 'pad': 0})
        ax_.grid()
    ax1[0].set_title("Example Taper Windows")
    fg1.tight_layout(h_pad=0.4)

    # Create a figure to show the spectral leakage of the tapered signal
    fg0, axx = plt.subplots(
        len(window_types), 1, sharex='all', sharey='all', figsize=(6., 4.)
    )
    for c_, (w_name_, ax_) in enumerate(zip(window_types, axx)):
        # Select a taper window from the list of available windows
        win_taper = get_scipy_window(w_name_, dft_samples, fftbins=False)

        # Compute the FFT of the padded taper
        W_ = scipy.fft.rfft(
            win_taper / np.abs(np.sum(win_taper)), n=fft_samples
        )
        W_dB = 20 * np.log10(np.maximum(abs(W_), 1e-250))
        ax_.plot(frequency_fft_over_df, W_dB, f'C{c_}-', label=w_name_)
        ax_.text(0.1, -50, w_name_, color=f'C{c_}', verticalalignment='bottom',
                 horizontalalignment='left', bbox={'color': 'white', 'pad': 0})
        ax_.set_yticks([-20, -60])
        # ax_.grid(axis='x')
        ax_.grid()
    axx[0].set_title("Spectral Leakage of Example Windows")
    fg0.supylabel(
        "Normalized Magnitude (dB)",
        x=0.04,
        y=0.5,
        fontsize='medium',
    )
    axx[-1].set(xlabel=r"Normalized frequency $f/\Delta f$ in bins",
                xlim=(0, 9), ylim=(-75, 3))
    fg0.tight_layout(h_pad=0.4)

    # Apply tapers to signal
    fg2, axx = plt.subplots(
        len(window_types), 1, sharex='all', sharey='all', figsize=(6., 4.)
    )
    for c_, (w_name_, ax_) in enumerate(zip(window_types, axx)):
        # Select a taper window from the list of available windows
        win_taper = get_scipy_window(w_name_, dft_samples, fftbins=False)

        # Apply the selected taper window to the input signal
        sig_cosine_tapered = win_taper * sig_cosine
        # Compute the variance of the taper window
        sig_tapered_var = np.var(sig_cosine_tapered)

        win_mean = np.abs(np.sum(win_taper)) / dft_samples
        win_var = np.sum(win_taper ** 2) / dft_samples
        win_rms = np.sqrt(win_var)
        spectral_amplitude_correction_factor = 1.0 / win_mean
        spectral_energy_correction_factor = 1.0 / win_rms

        print('\n*** Taper window:', w_name_)
        print(' Input signal variance:', sig_var)
        print(' Tapered signal variance:', sig_tapered_var)
        print(' Taper window mean amp:', win_mean)
        print(' Taper window rms:', win_rms)
        print(' Taper amplitude correction factor (ACF = 1/mean):')
        print('   ', spectral_amplitude_correction_factor)
        print(' Taper energy correction factor (ECF = 1/rms):')
        print('   ', spectral_energy_correction_factor)
        # Compute the fft of the padded or truncated tapered signal
        fft_sig_pos = scipy.fft.rfft(sig_cosine_tapered, n=fft_samples)
        fft_square = np.abs(fft_sig_pos) ** 2
        fft_power = 2. * frequency_resolution_fft * fft_square / dft_samples
        fft_power_corrected = np.maximum(
            fft_power * spectral_energy_correction_factor ** 2,
            1e-250,
        )
        W_dB = 10 * np.log10(fft_power_corrected)
        ax_.plot(frequency_fft_pos_hz, W_dB, f'C{c_}-', label=w_name_)
        ax_.text(
            frequency_design_center_hz,
            -160,
            w_name_,
            color=f'C{c_}',
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox={'color': 'white', 'pad': 0},
        )
        ax_.set_yticks([-140, -80, -6])
        # ax_.grid(axis='x')
        ax_.grid()
        print('FFT peak power at', frequency_center_fft_hz, 'Hz:')
        print(' ', fft_power_corrected[fft_peak_index])
        print(' ', W_dB[fft_peak_index], 'dB')
        print('Expected power (amplitude^2/2):', (sig_amplitude ** 2) / 2)
    axx[0].set_title("Spectral Leakage of Example Windows")
    fg2.supylabel(r"$10\,\log_{10}<Power>$ in dB",
                  x=0.04, y=0.5, fontsize='medium')
    axx[-1].set(xlabel=r"Frequency, Hz",
                xlim=(58, 62), ylim=(-140, -0))
    fg2.tight_layout(h_pad=0.4)

    plt.show()
    exit()
