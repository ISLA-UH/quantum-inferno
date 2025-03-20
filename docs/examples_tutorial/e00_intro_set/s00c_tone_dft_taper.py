"""
Quantum inferno example: s00b_tone_dft_taper.py
Introduction to Time-Frequency Representations (TFRs).
Compute Discrete (DFT) and Fast Fourier Transform (FFT) on simple tone to verify peak and average signal power,
with a taper window applied to the input signal.
See tone_dft_intro.py for a similar example without tapering.
Case study:
Sinusoid input with unit amplitude. Zero padding to make the fft a power of 2.
Validate:
Nominal DFT and FFT power estimates with taper correction. Validate variance metrics.
See also: https://docs.scipy.org/doc/scipy/tutorial/fft.html
https://docs.scipy.org/doc/scipy-1.15.1/tutorial/signal.html#tutorial-spectralanalysis

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import get_window, windows
from quantum_inferno.utilities.window import taper_power_correction

print(__doc__)

# List of taper windows to apply to the input signal in this example
# The default, implicit taper is a rectangular window.
window_types: list = ['boxcar', 'gaussian', 'hann', 'tukey', 'blackman']

if __name__ == "__main__":
    # This first example is a constant-amplitude sinusoidal signal of fixed duration, no taper window.
    # Construct a tone of fixed frequency with a constant sample rate in physical units - Hertz (Hz).
    frequency_sample_rate_hz = 800.
    frequency_design_center_hz = 60.
    sig_amplitude = 1.  # Nominal amplitude in some physical units
    sig_duration_s = 60.  # Nominal input signal duration in physical (analog) seconds. Accurate to 1/sample_rate.
    # Compute DFT number of digital points. Note int() rounds down.
    dft_samples:int = int(sig_duration_s * frequency_sample_rate_hz)
    # Dimensionless time (cyber time) from input record cyber duration.
    time_cyber = np.arange(dft_samples)
    # Standard deviation for the Gaussian window
    gauss_std = (dft_samples - 1) / np.pi / 2
    # Dimensionless center frequency:
    frequency_center = frequency_design_center_hz / frequency_sample_rate_hz

    # Construct synthetic tone at design center frequency using the dimensionless specifications.
    # Note there is no anti-alias preconditioning as test center frequency is well below Nyquist.
    sig_cosine = sig_amplitude * np.cos(2 * np.pi * frequency_center * time_cyber)
    sig_var = np.var(sig_cosine)

    """ 
    Design the fft performance by specifying its target spectral resolution in physical space
    This is the linear frequency equivalent of specifying the order of log frequency dyadic scales.
    
    """
    # The DFT spectral resolution is set by the record duration
    frequency_resolution_dft = 1. / dft_samples
    frequency_resolution_dft_hz = frequency_sample_rate_hz*frequency_resolution_dft

    # The dft does not need to be a power of 2. FFT makes it so.
    # This example specifies the desired spectral resolution to compute the FFT duration.
    fft_design_resolution_hz = 0.25*frequency_resolution_dft_hz
    # The spectral resolution sets the nominal record duration for the FFT.
    fft_design_time_s = 1. / fft_design_resolution_hz

    # The FFT efficiency is based on powers of 2. The DFT input can be truncated or zero padded.
    # Make the FFT number of points (duration) a power of 2 (dyadic) based on design spectral resolution.
    if fft_design_resolution_hz > frequency_resolution_dft_hz:
        print('*** STOP: FFT design resolution is greater than DFT resolution. Not in this example ***')
        exit()
    else:
        # Finer spectral resolution, increase the number of points and zero pad
        fft_samples = 2 ** (int(np.ceil(np.log2(fft_design_time_s * frequency_sample_rate_hz))))

    # The dyadic spectral resolution is modified accordingly
    frequency_resolution_fft = 1. / fft_samples
    frequency_resolution_fft_hz = frequency_sample_rate_hz*frequency_resolution_fft

    # DFT estimation
    # Use only positive frequencies
    frequency_dft_pos_hz = scipy.fft.rfftfreq(dft_samples, d= 1 / frequency_sample_rate_hz)
    # Find the closest frequency to the design frequency.
    dft_index:int = np.argmin(np.abs(frequency_dft_pos_hz - frequency_design_center_hz))
    frequency_center_dft_hz = frequency_dft_pos_hz[dft_index]
    # For display
    frequency_fft_over_df = scipy.fft.rfftfreq(fft_samples, d= 1 / dft_samples)
    # The new FFT duration (power of two) defines the actual spectral resolution.
    frequency_fft_pos_hz = scipy.fft.rfftfreq(fft_samples, d= 1 / frequency_sample_rate_hz)
    # Find the closest frequency to the design frequency.
    fft_index: int = np.argmin(np.abs(frequency_fft_pos_hz - frequency_design_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    # Convert to dimensionless frequency
    frequency_center_fft = frequency_center_fft_hz/frequency_sample_rate_hz

    print('DFT samples:', dft_samples)
    print('log2(DFT samples):', np.log2(dft_samples))
    print('FFT samples:', fft_samples)
    print('log2(FFT samples):', np.log2(fft_samples))


    """
    Study taper windows to reduce spectral leakage.
    
    """
    # Create a figure to show the taper windows
    fg1, ax1 = plt.subplots(len(window_types), 1, sharex='all', sharey='all', figsize=(6., 4.))
    for c_, (w_name_, ax_) in enumerate(zip(window_types, ax1)):
        # Select a taper window from the list of available windows
        if w_name_ == 'gaussian':
            # Gaussian window requires a standard deviation parameter
            win_taper = get_window((w_name_, gauss_std), dft_samples, fftbins=False)
        else:
            win_taper = get_window(w_name_, dft_samples, fftbins=False)
        ax_.plot(win_taper, f'C{c_}-', label=w_name_)
        ax_.text(0.1, 0.5, w_name_, color=f'C{c_}', verticalalignment='bottom',
                 horizontalalignment='left', bbox={'color': 'white', 'pad': 0})
        ax_.grid()
    ax1[0].set_title("Example Taper Windows")
    fg1.tight_layout(h_pad=0.4)

    # Create a figure to show the spectral leakage of the tapered signal
    fg0, axx = plt.subplots(len(window_types), 1, sharex='all', sharey='all', figsize=(6., 4.))
    for c_, (w_name_, ax_) in enumerate(zip(window_types, axx)):
        # Select a taper window from the list of available windows
        if w_name_ == 'gaussian':
            win_taper = get_window((w_name_, gauss_std), dft_samples, fftbins=False)
        else:
            win_taper = get_window(w_name_, dft_samples, fftbins=False)

        # Compute the fft of the padded taper
        W_ = scipy.fft.rfft(win_taper / np.abs(np.sum(win_taper)), n=fft_samples)
        W_dB = 20*np.log10(np.maximum(abs(W_), 1e-250))
        ax_.plot(frequency_fft_over_df, W_dB, f'C{c_}-', label=w_name_)
        ax_.text(0.1, -50, w_name_, color=f'C{c_}', verticalalignment='bottom',
                 horizontalalignment='left', bbox={'color': 'white', 'pad': 0})
        ax_.set_yticks([-20, -60])
        # ax_.grid(axis='x')
        ax_.grid()
    axx[0].set_title("Spectral Leakage of Example Windows")
    fg0.supylabel(r"Normalized Magnitude $20\,\log_{10}|W(f)/c^\operatorname{amp}|$ in dB",
                  x=0.04, y=0.5, fontsize='medium')
    axx[-1].set(xlabel=r"Normalized frequency $f/\Delta f$ in bins",
                xlim=(0, 9), ylim=(-75, 3))
    fg0.tight_layout(h_pad=0.4)

    # Apply tapers to signal
    fg2, axx = plt.subplots(len(window_types), 1, sharex='all', sharey='all', figsize=(6., 4.))
    for c_, (w_name_, ax_) in enumerate(zip(window_types, axx)):
        # Select a taper window from the list of available windows
        if w_name_ == 'gaussian':
            win_taper = get_window((w_name_, gauss_std), dft_samples, fftbins=False)
        else:
            win_taper = get_window(w_name_, dft_samples, fftbins=False)

        # Apply the selected taper window to the input signal
        sig_cosine_tapered = win_taper* sig_cosine
        # Compute the variance of the taper window
        sig_tapered_var = np.var(sig_cosine_tapered)

        win_mean = np.abs(np.sum(win_taper)) / dft_samples
        win_var = np.sum(win_taper**2) / dft_samples
        win_rms = np.sqrt(win_var)
        spectral_amplitude_correction_factor = 1 / win_mean
        spectral_energy_correction_factor = 1 / win_rms

        print('\n *** Taper window: ', w_name_)
        print(' Input signal variance: ', sig_var)
        print(' Tapered signal variance: ', sig_tapered_var)
        print(' Taper window mean amp: ', win_mean)
        print(' Taper window rms: ', win_rms)
        print(' Taper amplitude correction factor (ACF=1/mean): ', spectral_amplitude_correction_factor)
        print(' Taper energy correction factor (ECF=1/rms): ', spectral_energy_correction_factor)
        # Compute the fft of the padded or truncated tapered signal
        fft_sig_pos = scipy.fft.rfft(sig_cosine_tapered, n=fft_samples)
        fft_square = np.abs(fft_sig_pos)**2
        fft_power = 2. * frequency_resolution_fft * fft_square/dft_samples

        W_dB = 10*np.log10(np.maximum(fft_power*spectral_energy_correction_factor**2, 1e-250))
        ax_.plot(frequency_fft_pos_hz, W_dB, f'C{c_}-', label=w_name_)
        ax_.text(frequency_design_center_hz, -160, w_name_, color=f'C{c_}', verticalalignment='bottom',
                 horizontalalignment='left', bbox={'color': 'white', 'pad': 0})
        ax_.set_yticks([-140, -80, -6])
        # ax_.grid(axis='x')
        ax_.grid()
    axx[0].set_title("Spectral Leakage of Example Windows")
    fg2.supylabel(r"$10\,\log_{10}<Power>$ in dB",
                  x=0.04, y=0.5, fontsize='medium')
    axx[-1].set(xlabel=r"Frequency, Hz",
                xlim=(58, 62), ylim=(-140, -0))
    fg2.tight_layout(h_pad=0.4)

    plt.show()
    exit()


    """
    DFT, FFT, and useful metrics
    
    """
    # Compute the Real DFT of the input record
    dft_sig_pos = np.fft.rfft(sig_cosine, n=dft_samples)
    dft_sig_tapered_pos = np.fft.rfft(sig_cosine_tapered, n=dft_samples)
    # Compute the Real FFT of the zero-padded or truncated record
    fft_sig_pos = np.fft.rfft(sig_cosine, n=fft_samples)
    fft_sig_tapered_pos = np.fft.rfft(sig_cosine_tapered, n=fft_samples)

    print('SIGNAL DESIGN SPECIFICATIONS: BASIC COSINE TONE')
    print(' Signal amplitude, a:', sig_amplitude)
    print(' Signal duration, s:', sig_duration_s)
    print(' Signal variance, a**2:', sig_var)
    print(' Nyquist frequency:', frequency_sample_rate_hz/2)
    print(' Nominal signal frequency, hz:', frequency_design_center_hz)
    print(' DFT signal frequency, hz:', frequency_center_dft_hz)
    print(' FFT signal frequency, hz:', frequency_center_fft_hz)
    print(' DFT spectral resolution, hz', frequency_resolution_dft_hz)
    print(' FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print(' DFT duration, points:', dft_samples)
    print(' FFT duration, points:', fft_samples)
    print(' log2(DFT points):', np.log2(dft_samples))
    print(' log2(FFT points):', np.log2(fft_samples))
    print('Real DFT/FFT returns only the positive frequencies')
    print(' Length of RDFT:', len(dft_sig_pos))
    print(' Length of RFFT:', len(fft_sig_pos))

    # By scaling by number of points, the RFFT returns the Fourier coefficient of the positive
    # frequency averaged over the signal duration.
    # The positive frequency contributes only half the amplitude and power
    # Estimate amplitude and power metrics using only positive frequencies

    dft_abs = np.abs(dft_sig_pos)
    dft_square = dft_abs**2
    dft_abs_pos_over_N = dft_abs / dft_samples
    dft_power = 2 * frequency_resolution_dft * dft_square / dft_samples

    fft_abs = np.abs(fft_sig_pos)
    fft_square = fft_abs**2
    if fft_samples < dft_samples:
        fft_abs_pos_over_N = fft_abs / fft_samples
        fft_power = 2. * frequency_resolution_fft * fft_square / fft_samples
    else:
        fft_abs_pos_over_N = fft_abs / dft_samples
        fft_power = 2. * frequency_resolution_fft * fft_square / dft_samples

    print('\nDFT SUMMARY METRICS')
    print('|RDFT(dft_fc)/N|:', dft_abs_pos_over_N[dft_index])
    print('DFT Averaged Power:', np.sum(dft_power))

    print('FFT SUMMARY METRICS')
    print('|RFFT(fft_fc)/N|:', fft_abs_pos_over_N[fft_index])
    print('DFT Averaged Power', np.sum(fft_power))

    print('\n*** SUMMARY: FFT of a constant frequency tone with unit peak amplitude ***')
    print('Primary aim: verify metrics for DFT and FFT with simple tone')
    print('Positive frequency FFT amplitude is 1/2, negative frequency FFT amplitude is 1/2')
    print('Power averaged over the signal duration is P**2/N = 2 |RFFT/N|**2 = 1/2')
    print('Variance is 1/2, RMS amplitude is 1/sqrt(2)')
    print('Averaged power spectral density returns input signal variance')
    print('** IMPORTANT NOTE: EXACT RECONSTRUCTION ONLY OCCURS AT FFT FREQUENCY **\n')

    # Show the waveform, DFT, and FFT
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8, 5))
    ax1.plot(time_cyber / frequency_sample_rate_hz, sig_cosine)
    ax1.set_title('Synthetic tone, no taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.loglog(frequency_fft_pos_hz, fft_power, '-.', label='fft')
    ax2.loglog(frequency_dft_pos_hz, dft_power, label='dft')
    ax2.set_title(f"FFT Power, f = {frequency_center_fft_hz:.3f} Hz")
    ax2.set_xlabel('Frequency, Hz')
    ax2.set_ylabel("$2\\cdot\\mid\\frac{RFFT}{N}\\mid ^2$")
    ax2.legend()
    ax2.grid(True)

    plt.show()
