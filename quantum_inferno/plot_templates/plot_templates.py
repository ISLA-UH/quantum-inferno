"""
Base templates for plots:
* 3 waveforms
* 1 waveform, 2 mesh
* 1 waveform, 1 mesh
* CW and Power
"""
import math
from typing import List, Optional, Tuple, Union

from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable, AxesDivider
import numpy as np

import quantum_inferno.utilities.date_time as dt
from quantum_inferno.plot_templates import plot_base as plt_base


def adjust_figure_height(
        figure_size_y: int,
        n_rows: int,
        n_rows_standard: int = 2,
        hspace: float = 0.13
) -> Tuple[float, float, float]:
    """
    Adjust the figure height based on the number of rows to preserve standard panel aspect ratios

    :param figure_size_y: figure height
    :param n_rows: number of rows in figure
    :param n_rows_standard: number of rows in the figure for which height is not adjusted.  Default 2
    :param hspace: height space between panels, fraction of average panel height.  Default 0.13
    :return: adjusted figure height, space param for title, space param for x label
    """
    # space needed for the time label = 10% of the base figure height
    n_px_x_label: float = figure_size_y * 0.1
    # space needed for the title = 6% of the base figure height
    n_px_title: float = figure_size_y * 0.06
    n_px_panel: float = (figure_size_y - n_px_x_label - n_px_title) / ((1. + hspace) * n_rows_standard - hspace)
    n_px_hspace = hspace * n_px_panel
    adjusted_figure_size_y: float = n_px_panel * n_rows + n_px_hspace * (n_rows - 1) + n_px_x_label + n_px_title
    frac_title = 1 - n_px_title / adjusted_figure_size_y
    frac_x_label = n_px_x_label / adjusted_figure_size_y
    return adjusted_figure_size_y, frac_title, frac_x_label


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


def get_colormesh(axes: plt.Axes, time: np.ndarray, freq: np.ndarray, shading: Union[str, None],
                  mesh_base: plt_base.MeshBase, mesh_panel: plt_base.MeshPanel) -> QuadMesh:
    """
    :param axes: matplotlib axes to invoke the pcolormesh function
    :param time: timestamp array
    :param freq: frequency array
    :param shading: shading value
    :param mesh_base: MeshBase to get info from
    :param mesh_panel: MeshPanel to get info from
    :return: Quadmesh for plotting
    """
    return axes.pcolormesh(time, freq, mesh_panel.tfr, vmin=mesh_panel.color_min, vmax=mesh_panel.color_max,
                           cmap=mesh_base.colormap, shading=shading, snap=True)


def setup_plot(ax: plt.Axes, ylabel_units: str, text_size: int, is_waveform: bool,
               is_bottom: bool, ytick_style: str = "plain"):
    """
    set up a plot with common values
    :param ax: pyplot Axes to set values for
    :param ylabel_units: label for y-axis units
    :param text_size: size of text
    :param is_waveform: if True, the Axes is plotting a waveform, otherwise it's a mesh
    :param is_bottom: if True, the Axes being plotted is the bottom subplot
    :param ytick_style: tick style for waveform y-axis.  Does nothing if is_waveform is False.  Default "plain"
    """
    ax.set_ylabel(ylabel_units, size=text_size)
    ax.tick_params(axis="x", which="both", bottom=is_bottom, labelbottom=is_bottom, labelsize=text_size)
    ax.tick_params(axis="y", labelsize=text_size)
    if is_waveform:
        ax.grid(True)
        ax.ticklabel_format(style=ytick_style, scilimits=(0, 0), axis="y")
        ax.yaxis.get_offset_text().set_x(-0.034)


def get_panel_labels(n: int):
    """
    Get panel labels for a figure with n panels
    :param n: number of panels
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [f"({letters[i]})" for i in range(n)]


def plot_n_mesh_wf_vert(
        mesh_base: plt_base.MeshBase,
        panels: List[plt_base.MeshPanel],
        wf_base: plt_base.WaveformPlotBase,
        wf_panel: plt_base.WaveformPanel,
        sanitize_times: bool = True,
        use_default_size: bool = True
) -> plt.Figure:
    """
    Plot 1 or more mesh panels above the base waveform panel in a vertical layout.

    :param mesh_base: base values for mesh plots
    :param panels: list of mesh panels to display, in order of display
    :param wf_base: base values for plotting waveforms
    :param wf_panel: WaveformPanel required for figure
    :param sanitize_times: if True, sanitize timestamps.  Default True
    :param use_default_size: if True, use the default size for the plots, otherwise size dynamically.  Default True
    :return: figure to display
    """
    num_panels: int = len(panels) + 1
    time_label: str = get_time_label(wf_base.start_time_epoch, wf_base.units_time)
    # if we need to sanitize times and have a start epoch of 0, use the first timestamp, otherwise use plot base start
    epoch_start = wf_panel.time[0] if wf_base.start_time_epoch == 0 and sanitize_times else wf_base.start_time_epoch
    fig_params = wf_base.params_tfr

    # Time is in the center of the window, frequency is in the fft coefficient center.
    # frequency and time must be increasing!
    t_edge, f_edge, frequency_fix_ymin, frequency_fix_ymax = \
        mesh_time_frequency_edges(frequency=mesh_base.frequency, time=mesh_base.time,
                                  frequency_ymin=mesh_base.frequency_hz_ymin,
                                  frequency_ymax=mesh_base.frequency_hz_ymax,
                                  frequency_scaling=mesh_base.frequency_scaling)

    wf_panel_n_time_zero = sanitize_timestamps(wf_panel.time, epoch_start)
    time_xmin = wf_panel_n_time_zero[0]
    time_xmax = t_edge[-1]

    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    mesh_x, mesh_y, shading = mesh_base.get_colormesh_params()
    if shading is None:
        mesh_x = t_edge
        mesh_y = f_edge

    # format colorbar ticks
    all_cbar_ticks_lens: List[int] = []
    for p in panels:
        all_cbar_ticks_lens.append(max(len(str(math.ceil(p.color_min))), len(str(math.floor(p.color_max)))))
    max_cbar_tick_len: int = sorted(all_cbar_ticks_lens)[-1]
    cbar_tick_fmt: str = f"%-{max_cbar_tick_len}s"

    hspace = 0.13
    if use_default_size:
        title_space = .94
        xlabel_space = .1
        adj_fig_height = fig_params.figure_size_y
    else:
        adj_fig_height, title_space, xlabel_space = adjust_figure_height(fig_params.figure_size_y, num_panels)

    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(fig_params.figure_size_x, adj_fig_height),
        sharex=True,
    )

    panel_index = 0
    # main plotting loop
    for p in panels:
        if isinstance(p, plt_base.MeshPanel):
            p.set_color_min_max()
            if not p.is_auto_color_min_max():
                print(f"Mesh panel {panel_index} color scaling with user inputs")
            setup_plot(axes[panel_index], mesh_base.units_frequency, fig_params.text_size, False, False)
            ax_div: AxesDivider = make_axes_locatable(axes[panel_index])
            mesh_panel_cax: plt.Axes = ax_div.append_axes("right", size="1%", pad="0.5%")
            mesh_panel_cbar: Colorbar = fig.colorbar(
                get_colormesh(axes[panel_index], mesh_x, mesh_y, shading, mesh_base, p),
                cax=mesh_panel_cax,
                ticks=[math.ceil(p.color_min), math.floor(p.color_max)],
                format=cbar_tick_fmt)
            mesh_panel_cbar.set_label(p.cbar_units, rotation=270, size=fig_params.text_size)
            mesh_panel_cax.tick_params(labelsize=fig_params.text_size)
            axes[panel_index].set_ylim(frequency_fix_ymin, frequency_fix_ymax)
            axes[panel_index].set_yscale(mesh_base.frequency_scaling)
            if mesh_base.frequency_scaling == "linear":
                # Only works for linear range
                axes[panel_index].ticklabel_format(style=p.ytick_style, scilimits=(0, 0), axis="y")
        if panel_index != 0 and panel_index != num_panels - 1:
            axes[panel_index].margins(x=0)
        panel_index += 1

    axes[-1].plot(wf_panel_n_time_zero, wf_panel.sig, color=wf_base.waveform_color)
    axes[-1].set_xlim(time_xmin, time_xmax)
    wf_panel.set_y_lims(axes[-1])
    setup_plot(axes[-1], wf_panel.units, fig_params.text_size, True, True, wf_panel.ytick_style)
    ax_div: AxesDivider = make_axes_locatable(axes[-1])
    wf_panel_a_cax: plt.Axes = ax_div.append_axes("right", size="1%", pad="0.5%")
    wf_panel_a_cax.axis("off")

    if wf_base.figure_title_show:
        title = f"{wf_base.figure_title}"
        if wf_base.station_id:
            title += f" at Station {wf_base.station_id}"
        axes[0].set_title(title, fontsize=fig_params.text_size)
    if wf_base.label_panel_show:
        panel_labels = get_panel_labels(n=len(axes))
        for i in range(len(panels)):
            axes[i].text(0.01, 0.95, panel_labels[i], transform=axes[i].transAxes,
                         fontsize=fig_params.text_size, fontweight=wf_base.labels_fontweight, va="top",
                         color=panels[i].panel_label_color)
        axes[-1].text(0.01, 0.95, panel_labels[-1], transform=axes[-1].transAxes,
                      fontsize=fig_params.text_size, fontweight=wf_base.labels_fontweight, va="top",
                      color=wf_panel.panel_label_color)
    fig.text(.5, .01, time_label, ha="center", size=fig_params.text_size)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=xlabel_space, top=title_space, hspace=hspace)

    return fig


def plot_mesh_wf_vert(
        mesh_base: plt_base.MeshBase,
        mesh_panel: plt_base.MeshPanel,
        wf_base: plt_base.WaveformPlotBase,
        wf_panel: plt_base.WaveformPanel,
        sanitize_times: bool = True,
        use_default_size: bool = True
) -> plt.Figure:
    """
    Specifically plot one mesh and one waveform, vertically

    :param mesh_base: base values for mesh plots
    :param mesh_panel: mesh panel to display
    :param wf_base: base values for plotting waveforms
    :param wf_panel: waveform to display
    :param sanitize_times: if True, sanitize timestamps.  Default True
    :param use_default_size: if True, use the default size for the plots, otherwise size dynamically.  Default True
    :return: Figure to plot
    """
    return plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel, sanitize_times, use_default_size)


def plot_wf_3_vert(
        wf_base: plt_base.WaveformPlotBase,
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
    fig, axes = plt.subplots(3, 1,
                             figsize=(wf_base.params_tfr.figure_size_x,
                                      wf_base.params_tfr.figure_size_y),
                             sharex=True)
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
                           fontweight=wf_base.labels_fontweight, va="top")
        is_bottom = True if pnl == "a" else False
        setup_plot(fig_panel, wf_panel.units, wf_base.params_tfr.text_size, True, is_bottom, "sci")
        fig_panel.set_xlim(time_xmin, time_xmax)

    if wf_base.figure_title_show:
        fig_panel_c.set_title(f"{wf_base.figure_title} at Station {wf_base.station_id}")

    fig.text(.5, .01, time_label, ha="center", size=wf_base.params_tfr.text_size)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=.1, hspace=0.13)

    return fig


def setup_cw_power_plot(ax: plt.Axes, y_units: str, x_units: str, text_size: int):
    """
    Set the x and y labels and set tick params for the given Axes

    :param ax: the Axes object to update
    :param y_units: label for y-axis
    :param x_units: label for x-axis
    :param text_size: size of text
    """
    ax.set_ylabel(y_units, size=text_size)
    ax.set_xlabel(f"Time ({x_units})", size=text_size)
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
    ax.tick_params(axis="y", which="both", left=True, labelleft=True, labelsize="large")
    ax.grid(True)


def plot_cw_and_power(
        cw_panel: plt_base.CwPanel,
        power_panel: plt_base.PowerPanel,
        cw_plot_base: plt_base.CwPowerPlotBase = plt_base.CwPowerPlotBase()
) -> plt.Figure:
    """
    Template for CW and power plots

    :param cw_panel: CW panel to plot
    :param power_panel: Power panel to plot
    :param cw_plot_base: base parameters for plotting.  Default to default CwPowerPlotBase values
    :return: Figure to plot
    """
    # Catch cases where there may not be any data
    if cw_panel.is_no_data():
        print("No data to plot.")
        return plt.Figure()

    # Figure starts here
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(cw_plot_base.params_tfr.figure_size_x, cw_plot_base.params_tfr.figure_size_y),
    )
    fig_cw_panel: plt.Axes = ax[0]
    fig_power_panel: plt.Axes = ax[1]

    if cw_plot_base.figure_title_show:
        fig_cw_panel.set_title(cw_panel.title, size=cw_plot_base.params_tfr.text_size)
        fig_power_panel.set_title(power_panel.title, size=cw_plot_base.params_tfr.text_size)

    fig_cw_panel.plot(cw_panel.time, cw_panel.sig)
    setup_cw_power_plot(fig_cw_panel, cw_panel.y_units, cw_panel.x_units, cw_plot_base.params_tfr.text_size)

    for i in power_panel.panel_data:
        fig_power_panel.semilogx(i.freq, i.sig, ls=i.linestyle, lw=i.linewidth, label=i.sig_label)
    setup_cw_power_plot(fig_power_panel, power_panel.y_units, power_panel.x_units, cw_plot_base.params_tfr.text_size)
    fig_power_panel.legend()

    fig.tight_layout()
    fig.subplots_adjust()
    return fig
