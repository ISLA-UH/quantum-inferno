"""
    * One-sided exponential amplitude modulation (See :ref:`amexpos`)
    * Constant frequency modulation (See :ref:`fmconst`)
    * -5 dB complex gaussian noise (See :ref:`noisecg` and :ref:`sigmerge`)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming
from tftb.generators import amexpos, fmconst, sigmerge, noisecg
from tftb.processing.cohen import Spectrogram

# https://tftb.readthedocs.io/en/latest/quickstart/intro_examples_1.html

sig_points: int = int(2**8)

# Generate a noisy transient signal, odd construction
transsig = amexpos(64, kind='unilateral') * fmconst(64)[0]
signal = np.hstack((np.zeros((100,)), transsig, np.zeros((92,))))
signal = sigmerge(signal, noisecg(256), -5)

fig, ax = plt.subplots(2, 1)
ax1, ax2 = ax
ax1.plot(np.real(signal))
ax1.grid()
ax1.set_title('Noisy Transient Signal')
ax1.set_xlabel('Time')
ax1.set_xlim((0, 256))
ax1.set_ylim((np.real(signal).max(), np.real(signal.min())))

# Energy spectrum of the signal
dsp = np.fft.fftshift(np.abs(np.fft.fft(signal)) ** 2)
fwindow = hamming(65)  # TODO: Odd spec, 1/2 of the window

ax2.plot(np.arange(-128, 128, dtype=float) / 256, dsp)
ax2.set_title('Energy spectrum of noisy transient signal')
ax2.set_xlabel('Normalized frequency')
ax2.grid()
ax2.set_xlim(-0.5, 0.5)
plt.subplots_adjust(hspace=0.5)
spec = Spectrogram(signal, n_fbins=128, fwindow=fwindow)
spec.run()
spec.plot(kind="contour", threshold=0.1, show_tf=False)

plt.show()
