"""
Quantum inferno example: s00_tone_dft_intro.py
Introduction to Time-Frequency Representations (TFRs).
Compute Discrete (DFT) and Fast Fourier Transform (FFT) on simple tone to verify averaged signal power.
The foundation  of efficient TFR computation is the FFT.
When N = number of points is a power of 2, computation scales as N log N instead of N**2
Case study:
Sinusoid input with unit amplitude
Validate:
Nominal DFT and FFT power averaged over the signal duration is 1/2
See also: https://docs.scipy.org/doc/scipy-1.15.1/tutorial/signal.html#tutorial-spectralanalysis

"""
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

if __name__ == "__main__":
    # This first example is a sinusoidal signal within a record of fixed duration and no taper window.
    # Construct a tone of fixed frequency with a constant sample rate
    # <o - Later examples use a functional version of the synthetic: leave the steps here! - o>

    frequency_sample_rate_hz = 800.
    frequency_design_center_hz = 16.
    # The dft does not need to be a power of 2. However, the fft will make it so.
    time_dft_s = 1.  # Nominal value
    dft_design_resolution_hz = 1./time_dft_s  # Nominal value
    # Scale the target fft resolution.
    # Finer (<1) or coarser (>=1) spectral resolution. Consider finer rez.
    fft_resolution_scale = 1./4.
    # Design spectral resolution
    fft_design_resolution_hz = fft_resolution_scale*dft_design_resolution_hz
    time_fft_s = 1./fft_design_resolution_hz

    # The FFT efficiency is based on powers of 2. Inspect padding with zeros.
    # Compute DFT number of points. Note int() rounds down.
    time_dft_nd = int(time_dft_s*frequency_sample_rate_hz)
    # Make FFT number of points a power of 2 based on design spectral resolution.
    if fft_resolution_scale >= 1:
        # Coarser spectral resolution by decreasing the number of points.
        time_fft_nd = 2**(int(np.floor(np.log2(time_fft_s*frequency_sample_rate_hz))))
    else:
        # Finer spectral resolution by increasing the number of points.
        time_fft_nd = 2**(int(np.ceil(np.log2(time_fft_s*frequency_sample_rate_hz))))

    # DFT estimation
    # In this example we only need the positive frequencies.
    frequency_dft_pos_hz = np.fft.rfftfreq(time_dft_nd, d=1/frequency_sample_rate_hz)
    # Find the closest frequency to the design frequency.
    dft_index = np.argmin(np.abs(frequency_dft_pos_hz - frequency_design_center_hz))
    frequency_center_dft_hz = frequency_dft_pos_hz[dft_index]
    frequency_resolution_dft_hz = frequency_sample_rate_hz/time_dft_nd

    # The new FFT duration (power of two) defines the actual spectral resolution.
    # The fft frequencies are set by the duration of the fft.
    # In this example we only need the positive frequencies.
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1/frequency_sample_rate_hz)
    # Find the closest frequency to the design frequency.
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz - frequency_design_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    frequency_resolution_fft_hz = frequency_sample_rate_hz/time_fft_nd

    # Convert to dimensionless time and frequency, which is typically used in mathematical formulas.
    # Scale by the sample rate.
    # Dimensionless center frequency:
    frequency_center = frequency_design_center_hz / frequency_sample_rate_hz
    frequency_center_fft = frequency_center_fft_hz/frequency_sample_rate_hz
    # Dimensionless time (samples) from input record duration
    time_nd = np.arange(time_dft_nd)

    # Construct synthetic tone at design center frequency
    # Note there is no taper (e.g. using default rectangular window).
    mic_sig = np.cos(2*np.pi*frequency_center*time_nd)

    # Compute the Real DFT of the input record
    dft_sig_pos = np.fft.rfft(mic_sig, n=time_dft_nd)
    # Compute the Real FFT of the zero-padded record
    fft_sig_pos = np.fft.rfft(mic_sig, n=time_fft_nd)


    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Nominal signal frequency, hz:', frequency_design_center_hz)
    print('DFT signal frequency, hz:', frequency_center_dft_hz)
    print('FFT signal frequency, hz:', frequency_center_fft_hz)
    print('Nominal dft spectral resolution, hz', dft_design_resolution_hz)
    print('DFT spectral resolution, hz', frequency_resolution_dft_hz)
    print('FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print('Number of DFT points:', time_dft_nd)
    print('Number of FFT points:', time_fft_nd)
    print('log2(DFT points):', np.log2(time_dft_nd))
    print('log2(FFT points):', np.log2(time_fft_nd))


    print('RFFT returns only the positive frequencies')
    print('len(RDFT):', len(dft_sig_pos))
    print('len(RFFT):', len(fft_sig_pos))

    # By scaling by number of points, the RFFT returns the Fourier coefficient of the positive
    # frequency averaged over the signal duration.
    # The positive frequency contributes only half the amplitude.
    # The negative frequency contributes the other half.
    dft_abs_pos_over_N = np.abs(dft_sig_pos)/len(mic_sig)
    dft_abs_power = 2*dft_abs_pos_over_N**2
    fft_abs_pos_over_N = np.abs(fft_sig_pos)/len(mic_sig)
    fft_abs_power = 2*fft_abs_pos_over_N**2

    print('|RDFT(dft_fc)/N|:', dft_abs_pos_over_N[dft_index])
    print('2*|RDFT(dft_fc)/N|**2:', dft_abs_power[dft_index])
    print('|RFFT(fft_fc)/N|:', fft_abs_pos_over_N[fft_index])
    print('2*|RFFT(fft_fc)/N|**2:', fft_abs_power[fft_index])

    print('\n*** SUMMARY: FFT of a constant frequency tone with unit peak amplitude ***')
    print('Positive frequency FFT amplitude is 1/2, negative frequency FFT amplitude is 1/2')
    print('Power averaged over the signal duration is P**2 = 2 |RFFT/N|**2 = 1/2')
    print('RMS amplitude is sqrt(P**2) = 1/sqrt(2)')
    print('** IMPORTANT NOTE: EXACT RECONSTRUCTION ONLY OCCURS AT FFT FREQUENCY **')

    # Show the waveform and its FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8, 5))
    ax1.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    ax1.set_title('Synthetic tone, no taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_fft_pos_hz, fft_abs_power)
    ax2.set_title(f"FFT Power, f = {frequency_center_fft_hz:.3f} Hz")
    ax2.set_xlabel('Frequency, Hz')
    ax2.set_ylabel("$2\\cdot\\mid\\frac{RFFT}{N}\\mid ^2$")
    ax2.grid(True)

    plt.show()
