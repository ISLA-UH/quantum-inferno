"""
Quantum inferno example: s00b_tone_dft_intro.py
Introduction to Time-Frequency Representations (TFRs).
Compute Discrete (DFT) and Fast Fourier Transform (FFT) on simple tone to verify peak and average signal power.
The foundation of efficient TFR computation is the FFT.
When N = number of points is a power of 2, computation scales as N log N instead of N**2
Case study:
Sinusoid input with unit amplitude
Validate:
Nominal DFT and FFT power estimates. Validate variance metrics.
See also: https://docs.scipy.org/doc/scipy/tutorial/fft.html
https://docs.scipy.org/doc/scipy-1.15.1/tutorial/signal.html#tutorial-spectralanalysis

"""
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

if __name__ == "__main__":
    # This first example is a constant-amplitude sinusoidal signal of fixed duration, no taper window.
    # Construct a tone of fixed frequency with a constant sample rate in physical units - Hertz (Hz).
    frequency_sample_rate_hz = 800.
    frequency_design_center_hz = 60.
    sig_amplitude = 1.  # Nominal amplitude in some physical units
    sig_duration_s = 2.  # Nominal input signal duration in physical (analog) seconds. Accurate to 1/sample_rate.
    # Compute DFT number of digital points. Note int() rounds down.
    time_dft_nd = int(sig_duration_s * frequency_sample_rate_hz)
    # Dimensionless time (cyber time) from input record cyber duration.
    time_cyber = np.arange(time_dft_nd)
    # Dimensionless center frequency:
    frequency_center = frequency_design_center_hz / frequency_sample_rate_hz

    # Construct synthetic tone at design center frequency using the dimensionless specifications.
    # Note there is no taper (e.g. using default rectangular window) or anti-alias preconditioning.
    sig_cosine = sig_amplitude * np.cos(2 * np.pi * frequency_center * time_cyber)
    sig_variance = np.var(sig_cosine)

    # The DFT spectral resolution is set by the record duration
    frequency_resolution_dft = 1./time_dft_nd
    frequency_resolution_dft_hz = frequency_sample_rate_hz*frequency_resolution_dft

    """ 
    Design the fft performance by specifying its target spectral resolution in physical space
    This is the linear frequency equivalent of specifying the order of log frequency dyadic scales.
    
    """
    # The dft does not need to be a power of 2. FFT makes it so.
    # This example specifies the desired spectral resolution to compute the FFT duration.
    fft_design_resolution_hz = 0.5
    # The spectral resolution sets the nominal record duration for the FFT.
    fft_design_time_s = 1. / fft_design_resolution_hz

    # The FFT efficiency is based on powers of 2. The DFT input can be truncated or zero padded.
    # Make the FFT number of points (duration) a power of 2 (dyadic) based on design spectral resolution.
    if fft_design_resolution_hz > frequency_resolution_dft_hz:
        # Coarser spectral resolution, decrease the number of points and truncate
        time_fft_nd = 2**(int(np.floor(np.log2(fft_design_time_s * frequency_sample_rate_hz))))
    else:
        # Finer spectral resolution, increase the number of points and zero pad
        time_fft_nd = 2**(int(np.ceil(np.log2(fft_design_time_s * frequency_sample_rate_hz))))

    # The dyadic spectral resolution is modified accordingly
    frequency_resolution_fft = 1./time_fft_nd
    frequency_resolution_fft_hz = frequency_sample_rate_hz*frequency_resolution_fft

    # DFT estimation
    # Use only positive frequencies
    frequency_dft_pos_hz = np.fft.rfftfreq(time_dft_nd, d=1/frequency_sample_rate_hz)
    # Find the closest frequency to the design frequency.
    dft_index = np.argmin(np.abs(frequency_dft_pos_hz - frequency_design_center_hz))
    frequency_center_dft_hz = frequency_dft_pos_hz[dft_index]

    # The new FFT duration (power of two) defines the actual spectral resolution.
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1/frequency_sample_rate_hz)
    # Find the closest frequency to the design frequency.
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz - frequency_design_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    # Convert to dimensionless frequency
    frequency_center_fft = frequency_center_fft_hz/frequency_sample_rate_hz

    """
    DFT, FFT, and useful metrics
    
    """
    # Compute the Real DFT of the input record
    dft_sig_pos = np.fft.rfft(sig_cosine, n=time_dft_nd)
    # Compute the Real FFT of the zero-padded or truncated record
    fft_sig_pos = np.fft.rfft(sig_cosine, n=time_fft_nd)

    print('SIGNAL DESIGN SPECIFICATIONS: BASIC COSINE TONE')
    print(' Signal amplitude, a:', sig_amplitude)
    print(' Signal duration, s:', sig_duration_s)
    print(' Signal variance, a**2:', sig_variance)
    print(' Nyquist frequency:', frequency_sample_rate_hz/2)
    print(' Nominal signal frequency, hz:', frequency_design_center_hz)
    print(' DFT signal frequency, hz:', frequency_center_dft_hz)
    print(' FFT signal frequency, hz:', frequency_center_fft_hz)
    print(' DFT spectral resolution, hz', frequency_resolution_dft_hz)
    print(' FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print(' DFT duration, points:', time_dft_nd)
    print(' FFT duration, points:', time_fft_nd)
    print(' log2(DFT points):', np.log2(time_dft_nd))
    print(' log2(FFT points):', np.log2(time_fft_nd))
    print('Real DFT/FFT returns only the positive frequencies')
    print(' Length of RDFT:', len(dft_sig_pos))
    print(' Length of RFFT:', len(fft_sig_pos))

    # By scaling by number of points, the RFFT returns the Fourier coefficient of the positive
    # frequency averaged over the signal duration.
    # The positive frequency contributes only half the amplitude and power
    # Estimate amplitude and power metrics using only positive frequencies

    dft_abs = np.abs(dft_sig_pos)
    dft_square = dft_abs**2
    dft_abs_pos_over_N = dft_abs/time_dft_nd
    dft_power = 2 * frequency_resolution_dft * dft_square/time_dft_nd

    fft_abs = np.abs(fft_sig_pos)
    fft_square = fft_abs**2
    if time_fft_nd < time_dft_nd:
        fft_abs_pos_over_N = fft_abs/time_fft_nd
        fft_power = 2. * frequency_resolution_fft * fft_square/time_fft_nd
    else:
        fft_abs_pos_over_N = fft_abs/time_dft_nd
        fft_power = 2. * frequency_resolution_fft * fft_square/time_dft_nd

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
