"""
Quantum inferno example: s00_tone_fft_intro.py
Introduction to Time-Frequency Representations (TFRs).
Compute Fast Fourier Transform (FFT) on simple tones to verify amplitudes
The foundation  of efficient TFR computation is the FFT.
For N = number of points, computation scales as N log N instead of N**2
Case study:
Sinusoid input with unit amplitude
Validate:
Nominal FFT power averaged over the signal duration is 1/2

"""
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

if __name__ == "__main__":
    # In practice, our quest begins with a signal within a record of fixed duration.
    # Construct a tone of fixed frequency with a constant sample rate
    # <o - Later examples use a functional version of the synthetic: leave the steps here! - o>

    frequency_sample_rate_hz = 800.
    frequency_center_hz = 60.
    time_duration_s = 1  # Nominal value
    # Use the whole record to compute the fft
    time_fft_s = time_duration_s
    # It determines the target spectral resolution
    frequency_resolution_hz = 1/time_fft_s

    # The FFT efficiency is based on powers of 2; it is always possible to pad with zeros.
    # Set the record duration, make a power of 2. Note that int rounds down
    time_duration_nd = 2**(int(np.log2(time_duration_s*frequency_sample_rate_hz)))
    # Set the fft duration, make a power of 2
    time_fft_nd = 2**(int(np.log2(time_fft_s*frequency_sample_rate_hz)))

    # The new duration (power of two) defines the new spectral resolution.
    # The fft frequencies are set by the duration of the fft.
    # In this example we only need the positive frequencies.
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1/frequency_sample_rate_hz)
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz-frequency_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    frequency_resolution_fft_hz = frequency_sample_rate_hz/time_fft_nd

    # Convert to dimensionless time and frequency, which is typically used in mathematical formulas.
    # Scale by the sample rate.
    # Dimensionless center frequency:
    frequency_center = frequency_center_hz/frequency_sample_rate_hz
    frequency_center_fft = frequency_center_fft_hz/frequency_sample_rate_hz
    # Dimensionless time (samples):
    time_nd = np.arange(time_duration_nd)

    # Construct synthetic tone with 2^n points and max FFT amplitude at EXACT fft frequency
    mic_sig = np.cos(2*np.pi*frequency_center_fft*time_nd)
    # # Compare to synthetic tone with 2^n points and max FFT amplitude NOT at exact fft frequency
    # # It does NOT return unit amplitude
    # mic_sig = np.cos(2*np.pi*frequency_center*time_nd)

    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Nominal signal frequency, hz:', frequency_center_hz)
    print('FFT signal frequency, hz:', frequency_center_fft_hz)
    print('Nominal spectral resolution, hz', frequency_resolution_hz)
    print('FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print('Number of FFT points:', time_duration_nd)
    print('log2(FFT points):', np.log2(time_duration_nd))

    # Compute the RFFT of the whole record
    fft_sig_pos = np.fft.rfft(mic_sig)
    # Compare to the FFT
    fft_sig = np.fft.fft(mic_sig)
    print('RFFT returns only the positive frequencies')
    print('len(FFT):', len(fft_sig))
    print('len(RFFT):', len(fft_sig_pos))
    print('RFFT[fc]:', fft_sig_pos[fft_index])

    # By scaling by number of points, the RFFT returns the Fourier coefficient of the positive
    # frequency averaged over the signal duration.
    # The positive frequency contributes only half the amplitude.
    # The negative frequency contributes the other half.
    fft_abs_pos_over_N = np.abs(fft_sig_pos)/len(mic_sig)
    fft_abs_power = 2*fft_abs_pos_over_N**2

    print('|RFFT(fc)/N|:', fft_abs_pos_over_N[fft_index])
    print('2*|RFFT(fc)/N|**2:', fft_abs_power[fft_index])

    print('\n*** SUMMARY: FFT of a constant frequency tone with unit peak amplitude ***')
    print('Positive frequency FFT amplitude is 1/2, negative frequency FFT amplitude is 1/2')
    print('Power averaged over the signal duration is P**2 = 2 |RFFT|**2 = 1/2')
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
