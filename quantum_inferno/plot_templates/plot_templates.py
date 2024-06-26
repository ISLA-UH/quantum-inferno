"""
Base templates for plots:
* 3 waveforms
* 1 waveform, 2 mesh
* 1 waveform, 1 mesh
"""
import math
from typing import cast, List, Literal, Optional, Tuple

from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable, AxesDivider
import numpy as np

import quantum_inferno.utilities.date_time as dt
from quantum_inferno.plot_templates import plot_base as plt_base


def sanitize_timestamps(time_input: np.ndarray, start_epoch: Optional[float] = None) -> np.ndarray:
    """
    Sanitize timestamps

    :param time_input: array with timestamps
    :param start_epoch: optional start time to sanitize timestamps with.  Default None (use the first timestamp)
    :return: timestamps re-calculated from given epoch_start or first timestamp
    """
    return time_input - (time_input[0] if start_epoch is None else start_epoch)


def get_time_label(
        start_time_epoch: float,
        units_time: str,
        utc_offset_h: float = 0.
) -> str:
    """
    :param start_time_epoch: start time in seconds since epoch UTC
    :param units_time: units of time
    :param utc_offset_h: hours offset from UTC.  Default 0 (UTC time)
    :return: label for time units on a chart
    """
    label: str = f"Time ({units_time})"
    if start_time_epoch != 0:
        start_datetime_epoch = dt.get_datetime_from_timestamp_to_utc(start_time_epoch, utc_offset_h)
        label += f' from UTC {start_datetime_epoch.strftime("%Y-%m-%d %H:%M:%S")}'
    return label


def mesh_time_frequency_edges(
        frequency: np.ndarray,
        time: np.ndarray,
        frequency_ymin: float,
        frequency_ymax: float,
        frequency_scaling: str = "linear"
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Find time and frequency edges for plotting.  Raises an error if data is invalid.

    :param frequency: frequencies
    :param time: timestamps of the data
    :param frequency_ymin: minimum frequency for y-axis
    :param frequency_ymax: maximum frequency for y-axis
    :param frequency_scaling: "log" or "linear". Default is "linear"
    :return: time and frequency edges, frequency min and max
    """
    if frequency_ymin > frequency_ymax:
        raise ValueError("Highest frequency must be greater than lowest frequency")
    if not np.all(frequency[:-1] <= frequency[1:]):
        raise ValueError("Frequency must be increasing, flip it")
    if not np.all(time[:-1] <= time[1:]):
        raise ValueError("Time must be increasing, flip it")

    t_half_bin: float = np.abs(time[2] - time[1]) / 2.
    t_edge: np.ndarray = np.append(time[0] - t_half_bin, time + t_half_bin)

    if frequency_scaling == "log":
        k_edge: float = np.sqrt(frequency[-1] / frequency[-2])
        f_edge: np.ndarray = np.append(frequency / k_edge, k_edge * frequency[-1])
    else:
        f_half_bin: float = (frequency[2] - frequency[1]) / 2.
        f_edge: np.ndarray = np.append(frequency[0] - f_half_bin, frequency + f_half_bin)

    if frequency_ymin < f_edge[1]:
        frequency_ymin = f_edge[0]
    elif frequency_ymin <= 0 and frequency_scaling == "log":
        frequency_ymin = f_edge[1]
    if frequency_ymax > f_edge[-1]:
        frequency_ymax = f_edge[-1]

    if not isinstance(frequency_ymin, float):
        frequency_ymin = float(frequency_ymin)
    if not isinstance(frequency_ymax, float):
        frequency_ymax = float(frequency_ymax)

    return t_edge, f_edge, frequency_ymin, frequency_ymax


def plot_wf_3_vert(
        wf_base: plt_base.WaveformBase,
        wf_panel_a: plt_base.WaveformPanel,
        wf_panel_b: plt_base.WaveformPanel,
        wf_panel_c: plt_base.WaveformPanel,
        sanitize_times: bool = True
) -> plt.Figure:
    """
    plot 3 waveforms

    :param wf_base: base params for plotting waveforms
    :param wf_panel_a: first waveform to plot
    :param wf_panel_b: second waveform to plot
    :param wf_panel_c: third waveform to plot
    :param sanitize_times: if True, sanitize the timestamps.  Default True
    :return: figure to plot
    """
    time_label: str = get_time_label(wf_base.start_time_epoch, wf_base.units_time)

    epoch_start = wf_panel_a.time[0] if wf_base.start_time_epoch == 0 and sanitize_times else wf_base.start_time_epoch
    wf_panel_c_time_zero = sanitize_timestamps(wf_panel_c.time, epoch_start)
    wf_panel_b_time_zero = sanitize_timestamps(wf_panel_b.time, epoch_start)
    wf_panel_a_time_zero = sanitize_timestamps(wf_panel_a.time, epoch_start)

    # Catch cases where there may not be any data
    if wf_panel_a_time_zero[0] == wf_panel_a_time_zero[-1] and wf_panel_b_time_zero[0] == wf_panel_b_time_zero[-1]\
            and wf_panel_c_time_zero[0] == wf_panel_c_time_zero[-1]:
        print("No data to plot for " + wf_base.figure_title)
        return plt.figure()

    if wf_panel_a_time_zero[0] == wf_panel_b_time_zero[0] == wf_panel_c_time_zero[0]:
        time_xmin = wf_panel_a_time_zero[0]
    else:
        time_xmin = np.min([wf_panel_a_time_zero[0], wf_panel_b_time_zero[0], wf_panel_c_time_zero[0]])

    if wf_panel_a_time_zero[-1] == wf_panel_b_time_zero[-1] == wf_panel_c_time_zero[-1]:
        time_xmax = wf_panel_a_time_zero[-1]
    else:
        time_xmax = np.max([wf_panel_a_time_zero[-1], wf_panel_b_time_zero[-1], wf_panel_c_time_zero[-1]])

    # Figure starts here
    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
        plt.subplots(3, 1,
                     figsize=(wf_base.params_tfr.figure_size_x,
                              wf_base.params_tfr.figure_size_y),
                     sharex=True)
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    fig_panel_c: plt.Axes = axes[0]
    axes_iter = 0

    for pnl in ["c", "b", "a"]:
        fig_panel: plt.Axes = axes[axes_iter]
        axes_iter += 1
        # python eval() function allows us to use variables to name other variables
        wf_panel_zero = eval(f"wf_panel_{pnl}_time_zero")
        wf_panel = eval(f"wf_panel_{pnl}")
        fig_panel.plot(wf_panel_zero, wf_panel.sig)
        if wf_base.label_panel_show:
            fig_panel.text(0.01, 0.95, wf_panel.label, transform=fig_panel.transAxes,
                           fontsize=wf_base.params_tfr.text_size,
                           fontweight=wf_base.labels_fontweight, va='top')
        fig_panel.set_ylabel(wf_panel.units, size=wf_base.params_tfr.text_size)
        fig_panel.set_xlim(time_xmin, time_xmax)
        fig_panel.tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize='large')
        fig_panel.grid(True)
        fig_panel.tick_params(axis='y', labelsize='large')
        fig_panel.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
        fig_panel.yaxis.get_offset_text().set_x(-0.034)

    if wf_base.figure_title_show:
        fig_panel_c.set_title(f"{wf_base.figure_title} at Station {wf_base.station_id}")

    fig.text(.5, .01, time_label, ha='center', size=wf_base.params_tfr.text_size)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=.1, hspace=0.13)

    return fig


def plot_wf_mesh_2_vert(
        wf_base: plt_base.WaveformBase,
        wf_panel: plt_base.WaveformPanel,
        mesh_base: plt_base.MeshBase,
        mesh_panel_b: plt_base.MeshPanel,
        mesh_panel_c: plt_base.MeshPanel,
        sanitize_times: bool = True
) -> plt.Figure:
    """
    plot 1 waveform and 2 meshes

    :param wf_base: base params for plotting waveform
    :param wf_panel: the waveform to plot
    :param mesh_base: base params for plotting mesh
    :param mesh_panel_b: first mesh to plot
    :param mesh_panel_c: second mest to plot
    :param sanitize_times: if True, sanitize the timestamps.  Default True
    :return: figure to plot
    """
    # Time zeroing and scrubbing, if needed
    time_label = get_time_label(wf_base.start_time_epoch, wf_base.units_time)
    epoch_start = wf_panel.time[0] if wf_base.start_time_epoch == 0 and sanitize_times else wf_base.start_time_epoch
    wf_panel_a_elapsed_time = sanitize_timestamps(wf_panel.time, epoch_start)

    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # frequency and time must be increasing!
    t_edge, f_edge, frequency_fix_ymin, frequency_fix_ymax = \
        mesh_time_frequency_edges(frequency=mesh_base.frequency, time=mesh_base.time,
                                  frequency_ymin=mesh_base.frequency_hz_ymin,
                                  frequency_ymax=mesh_base.frequency_hz_ymax,
                                  frequency_scaling=mesh_base.frequency_scaling)

    # Figure starts here
    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
        plt.subplots(3, 1,
                     figsize=(wf_base.params_tfr.figure_size_x,
                              wf_base.params_tfr.figure_size_y),
                     sharex=True)
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    fig_mesh_panel_c: plt.Axes = axes[0]
    fig_mesh_panel_b: plt.Axes = axes[1]
    fig_wf_panel_a: plt.Axes = axes[2]

    # Top panel mesh --------------------------
    # Display preference
    wf_panel_a_time_xmin: int = wf_panel_a_elapsed_time[0]
    wf_panel_a_time_xmax: int = t_edge[-1]

    # Override, default is autoscaling to min and max values
    if not mesh_panel_b.is_auto_color_min_max():
        print("Mesh 1 color scaling with user inputs")

    if not mesh_panel_c.is_auto_color_min_max():
        print("Mesh 0 color scaling with user inputs")

    # Setup color map ticks
    all_cbar_ticks_lens: List[int] = [
        len(str(math.ceil(mesh_panel_c.color_min))),
        len(str(math.floor(mesh_panel_c.color_max))),
        len(str(math.ceil(mesh_panel_b.color_min))),
        len(str(math.floor(mesh_panel_b.color_max)))
    ]
    max_cbar_tick_len: int = sorted(all_cbar_ticks_lens)[-1]
    cbar_tick_fmt: str = f"%-{max_cbar_tick_len}s"

    if mesh_base.shading in ["auto", "gouraud"]:
        mesh_x = mesh_base.time
        mesh_y = mesh_base.frequency
        shading = cast(Literal, mesh_base.shading)
    else:
        mesh_x = t_edge
        mesh_y = f_edge
        shading = None

    for pnl in ["c", "b"]:
        # python eval() function allows us to use variables to name other variables
        fig_panel = eval(f"fig_mesh_panel_{pnl}")
        mesh_panel = eval(f"mesh_panel_{pnl}")
        pcolormesh = fig_panel.pcolormesh(mesh_x,
                                          mesh_y,
                                          mesh_panel.tfr,
                                          vmin=mesh_panel.color_min,
                                          vmax=mesh_panel.color_max,
                                          cmap=mesh_base.colormap,
                                          shading=shading,
                                          snap=True)
        mesh_panel_div: AxesDivider = make_axes_locatable(fig_panel)
        mesh_panel_cax: plt.Axes = mesh_panel_div.append_axes("right", size="1%", pad="0.5%")
        mesh_panel_cbar: Colorbar = fig.colorbar(pcolormesh, cax=mesh_panel_cax,
                                                 ticks=[math.ceil(mesh_panel.color_min),
                                                        math.floor(mesh_panel.color_max)],
                                                 format=cbar_tick_fmt)
        mesh_panel_cbar.set_label(mesh_panel.cbar_units, rotation=270,
                                  size=wf_base.params_tfr.text_size)
        mesh_panel_cax.tick_params(labelsize='large')
        fig_panel.set_ylabel(mesh_base.units_frequency, size=wf_base.params_tfr.text_size)
        fig_panel.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
        fig_panel.set_ylim(frequency_fix_ymin, frequency_fix_ymax)
        fig_panel.set_yscale(mesh_base.frequency_scaling)
        fig_panel.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        fig_panel.tick_params(axis='y', labelsize='large')

    # set specific panel values
    if wf_base.figure_title_show:
        fig_mesh_panel_c.set_title(f"{wf_base.figure_title} ({wf_base.station_id})")
    fig_mesh_panel_b.margins(x=0)

    # Waveform panel
    fig_wf_panel_a.plot(wf_panel_a_elapsed_time, wf_panel.sig)
    fig_wf_panel_a.set_ylabel(wf_panel.units, size=wf_base.params_tfr.text_size)
    fig_wf_panel_a.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
    fig_wf_panel_a.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize='large')
    fig_wf_panel_a.grid(True)
    fig_wf_panel_a.tick_params(axis='y', labelsize='large')
    fig_wf_panel_a.ticklabel_format(style="plain", scilimits=(0, 0), axis="y")
    fig_wf_panel_a.yaxis.get_offset_text().set_x(-0.034)

    wf_panel_a_div: AxesDivider = make_axes_locatable(fig_wf_panel_a)
    wf_panel_a_cax: plt.Axes = wf_panel_a_div.append_axes("right", size="1%", pad="0.5%")
    wf_panel_a_cax.axis("off")

    fig.text(.5, .01, time_label, ha='center', size=wf_base.params_tfr.text_size)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=.1, hspace=0.13)

    return fig


def plot_wf_mesh_vt(
        wf_base: plt_base.WaveformBase,
        wf_panel: plt_base.WaveformPanel,
        mesh_base: plt_base.MeshBase,
        mesh_panel: plt_base.MeshPanel,
        sanitize_times: bool = True
) -> plt.Figure:
    """
    plot a waveform and a mesh

    :param wf_base: base params for plotting waveform
    :param wf_panel: the waveform to plot
    :param mesh_base: base params for plotting mesh
    :param mesh_panel: the mesh to plot
    :param sanitize_times: if True, sanitize the timestamps.  Default True
    :return: figure to plot
    """
    # Time zeroing and scrubbing, if needed
    time_label = get_time_label(wf_base.start_time_epoch, wf_base.units_time)
    epoch_start = wf_panel.time[0] if wf_base.start_time_epoch == 0 and sanitize_times else wf_base.start_time_epoch
    wf_pnl_a_elapsed_time = sanitize_timestamps(wf_panel.time, epoch_start)

    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # frequency and time must be increasing!
    t_edge, f_edge, frequency_fix_ymin, frequency_fix_ymax = \
        mesh_time_frequency_edges(frequency=mesh_base.frequency, time=mesh_base.time,
                                  frequency_ymin=mesh_base.frequency_hz_ymin,
                                  frequency_ymax=mesh_base.frequency_hz_ymax,
                                  frequency_scaling=mesh_base.frequency_scaling)

    # Figure starts here
    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
        plt.subplots(2, 1,
                     figsize=(wf_base.params_tfr.figure_size_x,
                              wf_base.params_tfr.figure_size_y),
                     sharex=True)
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    fig_mesh_panel_b: plt.Axes = axes[0]
    fig_wf_panel_a: plt.Axes = axes[1]

    # Top panel mesh --------------------------
    # Display preference
    wf_panel_a_time_xmin: int = wf_pnl_a_elapsed_time[0]
    wf_panel_a_time_xmax: int = t_edge[-1]

    if not mesh_panel.is_auto_color_min_max():
        print("Mesh color scaling with user's inputs")

    # Setup color map ticks
    all_cbar_ticks_lens: List[int] = [
        len(str(math.ceil(mesh_panel.color_min))),
        len(str(math.floor(mesh_panel.color_max)))]
    max_cbar_tick_len: int = sorted(all_cbar_ticks_lens)[-1]
    cbar_tick_fmt: str = f"%-{max_cbar_tick_len}s"

    if mesh_base.shading in ["auto", "gouraud"]:
        mesh_x = mesh_base.time
        mesh_y = mesh_base.frequency
        shading = cast(Literal, mesh_base.shading)
    else:
        mesh_x = t_edge
        mesh_y = f_edge
        shading = None

    pcolormesh_top: QuadMesh = fig_mesh_panel_b.pcolormesh(mesh_x,
                                                           mesh_y,
                                                           mesh_panel.tfr,
                                                           vmin=mesh_panel.color_min,
                                                           vmax=mesh_panel.color_max,
                                                           cmap=mesh_base.colormap,
                                                           shading=shading,
                                                           snap=True)

    mesh_panel_b_div: AxesDivider = make_axes_locatable(fig_mesh_panel_b)
    mesh_panel_b_cax: plt.Axes = mesh_panel_b_div.append_axes("right", size="1%", pad="0.5%")
    mesh_panel_b_cbar: Colorbar = fig.colorbar(pcolormesh_top, cax=mesh_panel_b_cax,
                                               ticks=[math.ceil(mesh_panel.color_min),
                                                      math.floor(mesh_panel.color_max)],
                                               format=cbar_tick_fmt)
    mesh_panel_b_cbar.set_label(mesh_panel.cbar_units, rotation=270,
                                size=wf_base.params_tfr.text_size)
    mesh_panel_b_cax.tick_params(labelsize='large')
    if wf_base.figure_title_show:
        fig_mesh_panel_b.set_title(f"{wf_base.figure_title} at Station {wf_base.station_id}")
    fig_mesh_panel_b.set_ylabel(mesh_base.units_frequency, size=wf_base.params_tfr.text_size)
    fig_mesh_panel_b.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
    fig_mesh_panel_b.set_ylim(frequency_fix_ymin, frequency_fix_ymax)
    fig_mesh_panel_b.set_yscale(mesh_base.frequency_scaling)
    fig_mesh_panel_b.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    fig_mesh_panel_b.tick_params(axis='y', labelsize='large')
    if mesh_base.frequency_scaling == "linear":
        # Only works for linear range
        fig_mesh_panel_b.ticklabel_format(style=mesh_panel.ytick_style, scilimits=(0, 0), axis="y")

    # Waveform panel
    fig_wf_panel_a.plot(wf_pnl_a_elapsed_time, wf_panel.sig, color=wf_base.waveform_color)
    fig_wf_panel_a.set_ylabel(wf_panel.units, size=wf_base.params_tfr.text_size)
    fig_wf_panel_a.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
    fig_wf_panel_a.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize='large')
    fig_wf_panel_a.grid(True)
    fig_wf_panel_a.tick_params(axis='y', labelsize='large')
    ytick_style = wf_panel.ytick_style
    if wf_panel.yscaling == 'auto':
        fig_wf_panel_a.set_ylim(np.min(wf_panel.sig), np.max(wf_panel.sig))
        ytick_style = "plain"
    elif wf_panel.yscaling == 'symmetric':
        fig_wf_panel_a.set_ylim(-np.max(np.abs(wf_panel.sig)), np.max(np.abs(wf_panel.sig)))
    elif wf_panel.yscaling == 'positive':
        fig_wf_panel_a.set_ylim(0, np.max(np.abs(wf_panel.sig)))
    else:
        fig_wf_panel_a.set_ylim(plt_base.DEFAULT_YLIM_MIN, plt_base.DEFAULT_YLIM_MAX)
    fig_wf_panel_a.ticklabel_format(style=ytick_style, scilimits=(0, 0), axis="y")
    fig_wf_panel_a.yaxis.get_offset_text().set_x(-0.034)
    wf_panel_a_div: AxesDivider = make_axes_locatable(fig_wf_panel_a)
    wf_panel_a_cax: plt.Axes = wf_panel_a_div.append_axes("right", size="1%", pad="0.5%")
    wf_panel_a_cax.axis("off")

    fig.text(.5, .01, time_label, ha='center', size=wf_base.params_tfr.text_size)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=.1, hspace=0.13)

    return fig
