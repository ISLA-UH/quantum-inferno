#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2015 jaidev <jaidev@newton>
#
# Distributed under terms of the MIT license.

"""
===============================================
Time and Frequency Localization Characteristics
===============================================

Generate a signal that has localized characteristics in both time and frequency
and compute the following estimates:

    * time center
    * time duration
    * frequency center
    * frequency spreading

Example 2.1 from the tutorial.
"""

from tftb.generators import fmlin, amgauss, fmconst
from tftb.processing import loctime, locfreq, plotifl, inst_freq, group_delay
import numpy as np
from numpy import real
import matplotlib.pyplot as plt

# generate signal
signal = fmlin(256)[0] * amgauss(256)
plt.figure()
plt.subplot(211), plt.plot(np.real(signal))
plt.xlim(0, 256)
plt.xlabel('Time')
plt.ylabel('Real part')
plt.title('Signal')
plt.grid()
fsig = np.fft.fftshift(np.abs(np.fft.fft(signal)) ** 2)
plt.subplot(212), plt.plot(np.linspace(0, 0.5, 256), fsig)
plt.xlabel('Normalized frequency')
plt.ylabel('Squared modulus')
plt.title('Spectrum')
plt.grid()
plt.subplots_adjust(hspace=0.5)

# NEXT: Gabor Wavelet
x = amgauss(128) * fmconst(128)[0]
plt.figure()
plt.plot(real(x))
plt.grid()
plt.xlim(0, 128)
plt.title("Gaussian amplitude modulation")

tm, T = loctime(x)
print("Time Center: {}".format(tm))
print("Time Duration: {}".format(T))
num, B = locfreq(x)
TB = T*B
print("Frequency Center: {}".format(num))
print("Frequency Bandwidth: {}".format(B))
print("Duration*Bandwidth: {}".format(TB))

# Hilbert transform
sig, _ = fmlin(256)
plt.figure()
plt.plot(real(sig))
plt.title("Sig")

# Group delay, same as instantaneous f in spectrum
fnorm = np.linspace(0, .5, 10)
gd = group_delay(sig, fnorm)
plt.figure()
plt.plot(gd, fnorm)
plt.grid(True)
plt.xlim(0, 256)
plt.xlabel('Time')
plt.ylabel('Normalized Frequency')
plt.title('Group Delay Estimation')

# Instantaneous frequency
time_samples = np.arange(3, 257)
ifr = inst_freq(sig)[0]
plotifl(time_samples, ifr)
