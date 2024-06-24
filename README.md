# quantum-inferno
## Quantized Information Entropy, Nth Octave

### Caveat Emptor: Early Release Dev Version

### Description
Computes standardized time-frequency representations (TFRs) for power, information, and entropy, 
built on the Gabor wavelets with minimal time-frequency uncertainty with 
logarithmic constant-Q base 2 (binary) scales and frequency bands of quantized order N.

All algorithms are based on FFTs for computational efficiency. 
The short-term Fourier transform (STFT) is included as the baseline TFR.
Algorithms for the Continuous Wavelet Transform (CWT), Discrete Wavelet Transform (DWT), 
and Stockwell Transform (STX) are provided.

Refer to the open access publications:

- [Garcés, M.A. Quantized Information in Spectral Cyberspace. Entropy 2023, 25, 419](https://doi.org/10.3390/e25030419)

- [Garcés, M.A. Quantized Constant-Q Gabor Atoms for 
Sparse Binary Representations of Cyber-Physical Signatures. Entropy 2020, 22, 936](https://doi.org/10.3390/e22090936)

- [Garcés, M.A. On Infrasound Standards, Part 1 Time, Frequency, and Energy Scaling. 
Inframatics 2013, 2, 13-35](https://doi.org/10.4236/inframatics.2013.22002)
 
Recommended background reading in chronological order:
- Gabor, D. Theory of Communication, Part 3. Electr. Eng. 1946, 93, 445–457.
- Shannon, C.E. The Mathematical Theory of Communication; University of Illinois Press: Urbana, IL, USA, 1998; [1949 first ed].
- Harris, F. J. On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform, Proceedings of the IEEE, 1978, 66 (1), 51-83.
- Cohen, L. Time-Frequency Analysis, Prentice-Hall, NI 07458, 1995.
- Stockwell, R. G., Mansina, L, and R. P. Lowe. Localization of the Complex Spectrum: The S Transform. Signal Processing, IEEE Transactions, 1996, 44 no. 4, 998-1001.
- Mallat, S. A Wavelet Tour of Signal Processing: The Sparse Way, 3rd ed.; Academic Press: Cambridge, MA, USA, 2009 [1998 first ed].


### Installation
```shell script
pip install quantum-inferno
```

More details will be placed in the [Installation guide](https://github.com/ISLA-UH/quantum-inferno/blob/main/docs/installation.md).

### Examples
Full examples can be found in the [examples documentation](https://github.com/ISLA-UH/quantum-inferno/blob/main/docs/examples.md#examples-using-quantum-inferno).

### API Documentation
Check the [API Documentation](https://ISLA-UH.github.io/quantum-inferno).

### Resources

- Found an issue? Submit a [bug report](https://github.com/ISLA-UH/quantum-inferno/issues).
- [MIT License](https://github.com/ISLA-UH/quantum-inferno/blob/main/LICENSE)
