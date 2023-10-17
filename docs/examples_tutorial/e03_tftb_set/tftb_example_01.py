"""
todo: Describe me
https://tftb.readthedocs.io/en/latest/apiref/tftb.generators.html
"""

import numpy as np
import matplotlib.pyplot as plt
from tftb.processing.cohen import WignerVilleDistribution
from tftb.generators import amgauss, fmlin, sigmerge, noisecg
# https://tftb.readthedocs.io/en/latest/quickstart/intro_examples_1.html


def plot_xy(x, y: np.ndarray, title_str: str):
    plt.figure()
    plt.plot(x, np.real(y))
    plt.grid()
    plt.title(title_str)


sig_points: int = int(2**7)

# Gauss chirp
sig_1 = amgauss(sig_points, 50.0, 40.0) * fmlin(sig_points, 0.05, 0.3, 50)[0]
sig_1_t = np.arange(sig_points, dtype=float)
plot_xy(sig_1_t, np.real(sig_1), "Linear Frequency Modulation")

# Some other sweep
fmin, fmax = 0.0, 0.5
sig_2, _ = fmlin(sig_points, fmin, fmax)
plot_xy(sig_1_t, np.real(sig_2), "Sig 2")

sig_2_spect = np.fft.fftshift(np.abs(np.fft.fft(sig_2) ** 2))
sig_2_spect_f = (np.arange(sig_points, dtype=float) - float(sig_points) / 2) / 128.0
plot_xy(sig_2_spect_f, sig_2_spect, "Sig 2 spectrum")
# Wigner-Ville distribution
wvd = WignerVilleDistribution(sig_2)
wvd.run()
wvd.plot(kind='contour', extent=[0, sig_points, fmin, fmax])

sig_3 = sigmerge(sig_2, noisecg(sig_points), 0)
plot_xy(sig_1_t, np.real(sig_3), "Sig 3 = Sig 2 + Noise")
sig_3_spect = np.fft.fftshift(np.abs(np.fft.fft(sig_3)) ** 2)
plot_xy(sig_2_spect_f, sig_3_spect, "Sig 3 spectrum")

# Wigner-Ville distribution
wvd = WignerVilleDistribution(sig_3)
wvd.run()
wvd.plot(kind='contour', extent=[0, sig_points, fmin, fmax])

plt.show()
