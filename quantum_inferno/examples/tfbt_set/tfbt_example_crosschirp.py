from tftb.generators import amgauss, fmlin
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tftb.processing.cohen import WignerVilleDistribution
from tftb.generators import fmlin, sigmerge, noisecg
# https://tftb.readthedocs.io/en/latest/quickstart/intro_examples_1.html


def plot_xy(x, y: np.ndarray, title_str: str):
    plt.figure()
    plt.plot(x, np.real(y))
    plt.grid()
    plt.title(title_str)


if __name__ == "__main__":
    print('This takes some time to run ... please be patient')
    # crossed sweeps
    file_path = "/Users/mgarces/Documents/DATA_2022/Wigner_chirp/chirps.wav"
    print('**** The Warning is from having unrecognized metadata on the .wav header ****')
    fs, sig_2 = wavfile.read(file_path)
    sig_points = len(sig_2)
    sig_t = np.arange(sig_points, dtype=float)
    plot_xy(sig_t, np.real(sig_2), "Sig 2")

    sig_2_spect = np.fft.fftshift(np.abs(np.fft.fft(sig_2) ** 2))
    sig_2_spect_f = (np.arange(sig_points, dtype=float) - float(sig_points) / 2) / 128.0
    plot_xy(sig_2_spect_f, sig_2_spect, "Sig 2 spectrum")
    # Wigner-Ville distribution
    wvd = WignerVilleDistribution(sig_2)
    wvd.run()
    wvd.plot(kind='contour')

    sig_3 = sigmerge(sig_2, noisecg(sig_points), 0)
    plot_xy(sig_t, np.real(sig_3), "Sig 3 = Sig 2 + Noise")
    sig_3_spect = np.fft.fftshift(np.abs(np.fft.fft(sig_3)) ** 2)
    plot_xy(sig_2_spect_f, sig_3_spect, "Sig 3 spectrum")

    # Wigner-Ville distribution
    wvd = WignerVilleDistribution(sig_3)
    wvd.run()
    wvd.plot(kind='contour')

    plt.show()
