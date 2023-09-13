"""
Plot 1D entropy in the time or frequency domain
"""

from typing import Optional
import numpy as np
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from quantum_inferno import info


def plot_fft_compare(sig_in,
                     fs_hz: float,
                     duration_no_pad_points: int,
                     welch_points: int,
                     overlap_points: int,
                     fmin_hz: float = 2E-5,
                     fmax_hz: float = 1E-2,
                     station_str: Optional[str] = None,
                     do_show: bool = True,
                     do_save: bool = False,
                     save_path: str = None):
    """
    Comparisons for time and fft isnr and esnr
    :param sig_in:
    :param fs_hz:
    :param duration_no_pad_points:
    :param welch_points:
    :param overlap_points:
    :param fmin_hz: Plotting min frequency limit
    :param fmax_hz: Plotting max frequency limit
    :param station_str: Station ID
    :param do_show:
    :param do_save:
    :param save_path:
    :return:
    """

    time_idx = duration_no_pad_points
    # TODO: Clean up, this is silly
    # TODO: Build the distribution only from the portion of the spectrum that was used
    sig_in_no_pad = sig_in[0:time_idx]
    time = np.arange(len(sig_in_no_pad))/fs_hz/3600

    [tdr_sig, _, _, _, tdr_isnr, tdr_esnr] = \
        info.shannon_tdr(sig_in_real=sig_in_no_pad)

    rfft_sig1 = rfft(x=sig_in_no_pad)
    rfft_freq1 = rfftfreq(n=len(sig_in_no_pad), d=1/fs_hz)
    [fft_marginal1, fft_angle_rads1, _, _, fft_isnr1, fft_esnr1] = \
        info.shannon_fft(fft_sig=rfft_sig1)

    rfft_sig2 = rfft(x=sig_in)
    rfft_freq2 = rfftfreq(n=len(sig_in), d=1/fs_hz)
    [fft_marginal2, fft_angle_rads2, _, _, fft_isnr2, fft_esnr2] = \
        info.shannon_fft(fft_sig=rfft_sig2)

    # Use zero padding
    rfft_freq3, welch_power = welch(x=sig_in_no_pad, fs=fs_hz,
                                    nperseg=welch_points, noverlap=overlap_points,
                                    nfft=len(sig_in))
    [fft_marginal3, _, _, _, fft_isnr3, fft_esnr3] = \
        info.shannon_fft(fft_sig=np.sqrt(welch_power))

    # set the font size
    fontsz = 'x-large'
    # Initialize subplots
    fig, ax = plt.subplots(3, 2)
    fig.set_size_inches(16, 8)

    # Axis numbering line
    ref_line = mlines.Line2D([], [])

    ax[2, 0].plot(time, tdr_sig)
    ax[2, 0].set_ylabel('Norm Signal', fontsize=fontsz)
    ax[2, 0].set_xlabel('Time, hours', fontsize=fontsz)
    ax[2, 0].set_xlim(time[0], time[time_idx-1])
    ax[2, 0].tick_params(labelsize='x-large')
    ax[2, 0].grid(True)
    if station_str is None:
        label_a = '(a)'
    else:
        label_a = '(a)' + station_str

    ax[2, 0].legend(handles=[ref_line], labels=[label_a], loc=2,
                    frameon=False, handlelength=0, handletextpad=-0.5, fontsize='x-large')

    ax[1, 0].plot(time, tdr_isnr, '.')
    ax[1, 0].set_xticklabels([])
    ax[1, 0].set_ylim(-8, 8)
    ax[1, 0].set_xlim(time[0], time[time_idx-1])
    ax[1, 0].tick_params(labelsize='x-large')
    ax[1, 0].set_ylabel('Information SNR', fontsize=fontsz)
    ax[1, 0].grid(True)
    ax[1, 0].legend(handles=[ref_line], labels=['(b)'], loc=2,
                    frameon=False, handlelength=0, handletextpad=-0.5, fontsize='x-large')

    ax[0, 0].plot(time, tdr_esnr, '.')
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_ylabel('Entropy SNR', fontsize=fontsz)
    ax[0, 0].set_xlim(time[0], time[time_idx-1])
    ax[0, 0].tick_params(labelsize='x-large')
    ax[0, 0].grid(True)
    ax[0, 0].legend(handles=[ref_line], labels=['(c)'], loc=2,
                    frameon=False, handlelength=0, handletextpad=-0.5, fontsize='x-large')

    # Phase can be a bit boring
    # ax[2, 1].semilogx(rfft_freq1, fft_angle_rads1, '.', base=10, label='FFT')
    # ax[2, 1].semilogx(rfft_freq2, fft_angle_rads2, '.', base=10, label='FFT Padded')
    # ax[2, 1].set_xlim(fmin_hz, fmax)
    # # ax[2, 1].set_ylim(angle_min, angle_max+10)
    # ax[2, 1].set_ylim(-500, 50)
    # ax[2, 1].set_ylabel('Phase, rad', fontsize=fontsz)
    # ax[2, 1].set_xlabel('Frequency, Hz', fontsize=fontsz)
    # ax[2, 1].tick_params(labelsize='x-large')
    # ax[2, 1].grid(True)
    # panel_number = ax[2, 1].legend(handles=[ref_line], labels=['(d)'], loc=2,
    #                                frameon=False, handlelength=0, handletextpad=-0.5, fontsize='x-large')
    # ax[2, 1].legend(loc=1)
    # ax[2, 1].add_artist(panel_number)

    ax[2, 1].semilogx(rfft_freq1, fft_marginal1, '--', base=10, label='FFT')
    ax[2, 1].semilogx(rfft_freq2, fft_marginal2, '.', base=10, label='FFT Padded')
    ax[2, 1].semilogx(rfft_freq3, fft_marginal3, '.', base=10, label='Welch Padded')
    ax[2, 1].set_xlim(fmin_hz, fmax_hz)
    ax[2, 1].set_ylabel('Marginals', fontsize=fontsz)
    ax[2, 1].set_xlabel('Frequency, Hz', fontsize=fontsz)
    ax[2, 1].tick_params(labelsize='x-large')
    ax[2, 1].grid(True)
    panel_number = ax[2, 1].legend(handles=[ref_line], labels=['(d)'], loc=2,
                                   frameon=False, handlelength=0, handletextpad=-0.5, fontsize='x-large')
    ax[2, 1].legend(loc=1)
    ax[2, 1].add_artist(panel_number)

    ax[1, 1].semilogx(rfft_freq1, fft_isnr1, '--', base=10, label='FFT')
    ax[1, 1].semilogx(rfft_freq2, fft_isnr2, '.', base=10, label='FFT Padded')
    ax[1, 1].semilogx(rfft_freq3, fft_isnr3, '.', base=10, label='Welch Padded')
    ax[1, 1].set_xticklabels([])
    ax[1, 1].set_ylabel('Information SNR', fontsize=fontsz)
    ax[1, 1].set_ylim(-4, 12)
    ax[1, 1].set_xlim(fmin_hz, fmax_hz)
    ax[1, 1].tick_params(labelsize='x-large')
    ax[1, 1].grid(True)
    panel_number = ax[1, 1].legend(handles=[ref_line], labels=['(e)'], loc=2,
                                   frameon=False, handlelength=0, handletextpad=-0.5, fontsize='x-large')
    ax[1, 1].legend(loc=1)
    ax[1, 1].add_artist(panel_number)

    ax[0, 1].semilogx(rfft_freq1, fft_esnr1, '.', base=10, label='FFT')
    ax[0, 1].semilogx(rfft_freq2, fft_esnr2, '.', base=10, label='FFT Padded')
    ax[0, 1].semilogx(rfft_freq3, fft_esnr3, '.', base=10, label='Welch Padded')
    # NOTE: Below returns isnr because log2(plog2(p)) = log2(p) + log2(log2(p)) ~ log2p
    # ax[0, 1].semilogx(rfft_freq1, np.log2(fft_esnr1), '--', base=10, label='FFT')
    # ax[0, 1].semilogx(rfft_freq2, np.log2(fft_esnr2), '.', base=10, label='FFT Padded')
    # ax[0, 1].semilogx(rfft_freq3, np.log2(fft_esnr3), '.', base=10, label='Welch Padded')
    # ax[0, 1].set_ylim(-4, 12)
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_ylabel('Entropy SNR', fontsize=fontsz)
    # ax[0, 1].set_title(title_label)
    ax[0, 1].set_xlim(fmin_hz, fmax_hz)
    ax[0, 1].tick_params(labelsize='x-large')
    ax[0, 1].grid(True)
    panel_number = ax[0, 1].legend(handles=[ref_line], labels=['(f)'], loc=2,
                                   frameon=False, handlelength=0, handletextpad=-0.5, fontsize='x-large')
    ax[0, 1].legend(loc=1)
    ax[0, 1].add_artist(panel_number)
    plt.tight_layout()

    if do_save:
        if save_path is None:
            print('Save path must be specified! Exiting program.')
            exit()
        fig.savefig(save_path)

    if do_show:
        plt.show()
