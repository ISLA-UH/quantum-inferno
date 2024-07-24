"""
This module contains updated and streamlined versions of the quantum inferno plot templates.
"""
from typing import List, Tuple, Union
import math
import numpy as np
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable, AxesDivider

from quantum_inferno.plot_templates import plot_base as plt_base
from quantum_inferno.plot_templates import figure_attributes as fa
from quantum_inferno.plot_templates import plot_templates as plt_tpl
from quantum_inferno.plot_templates.plot_cyberspectral import mesh_time_frequency_edges


def adjust_figure_height(figure_size_y: int, n_rows: int, n_rows_standard: int = 2) -> int:
    """
    Adjust the figure height based on the number of rows to preserve standard panel aspect ratios

    :param figure_size_y: figure size y
    :param n_rows: number of rows
    :param n_rows_standard: default 3 todo: what is this
    :return: adjusted figure size y
    """
    # TODO: currently very rough estimate; factor in label sizes and whitespace
    # todo: return type ambiguous, forcing int as default
    return int(figure_size_y * n_rows / n_rows_standard)


def plot_nwf_nmesh_vert(panels_dict: dict,
                        plot_base: plt_base.PlotBase,
                        mesh_base: plt_base.MeshBase,
                        wf_base: plt_base.WaveformBase,
                        sanitize_times: bool = True,):
    """
    Plot n vertical panels - mesh (top panels) and time series waveforms (bottom panels)
    :param panels_dict: dictionary containing panel order (key), panel type ("panel_type" value), and the panel object
                        ("panel" value, either a MeshPanel or a WaveformPanel)
    :param plot_base: base params for plotting figure
    :param mesh_base: base params for plotting mesh panel(s)
    :param wf_base: base params for plotting wf panel(s)
    :param sanitize_times: if True (default), sanitize timestamps
    :return: figure to plot
    """
    n: int = len(panels_dict)
    time_label: str = plt_tpl.get_time_label(plot_base.start_time_epoch, plot_base.units_time)
    if plot_base.start_time_epoch == 0 and sanitize_times:
        epoch_start = panels_dict[n - 1]["panel"].time[0]
    else:
        epoch_start = plot_base.start_time_epoch
    fig_params = plot_base.params_tfr

    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # frequency and time must be increasing!
    t_edge, f_edge, frequency_fix_ymin, frequency_fix_ymax = \
        mesh_time_frequency_edges(frequency=mesh_base.frequency, time=mesh_base.time,
                                  frequency_ymin=mesh_base.frequency_hz_ymin,
                                  frequency_ymax=mesh_base.frequency_hz_ymax,
                                  frequency_scaling=mesh_base.frequency_scaling)

    wf_panel_n_time_zero = plt_tpl.sanitize_timestamps(panels_dict[n - 1]["panel"].time, epoch_start)
    time_xmin = wf_panel_n_time_zero[0]
    time_xmax = t_edge[-1]

    if mesh_base.shading in ["auto", "gouraud"]:
        mesh_x = mesh_base.time
        mesh_y = mesh_base.frequency
        shading = mesh_base.get_shading_as_literal()
    else:
        mesh_x = t_edge
        mesh_y = f_edge
        shading = None

    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = plt.subplots(
        n,
        1,
        figsize=(fig_params.figure_size_x, adjust_figure_height(fig_params.figure_size_y, n)),
        sharex=True,
    )
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]

    # format colorbar ticks
    all_cbar_ticks_lens: List[int] = []
    for i in range(len(panels_dict)):
        if panels_dict[i]["panel_type"] == "mesh":
            mesh_panel = panels_dict[i]["panel"]
            all_cbar_ticks_lens.append(
                max(len(str(math.ceil(mesh_panel.color_min))), len(str(math.floor(mesh_panel.color_max)))))
    max_cbar_tick_len: int = sorted(all_cbar_ticks_lens)[-1]
    cbar_tick_fmt: str = f"%-{max_cbar_tick_len}s"

    # main plotting loop
    for i in range(len(panels_dict)):
        if panels_dict[i]["panel_type"] == "mesh":
            mesh_panel = panels_dict[i]["panel"]
            print(mesh_panel.color_min, mesh_panel.color_max)
            mesh_panel.set_color_min_max()
            print(mesh_panel.color_min, mesh_panel.color_max)
            if not mesh_panel.is_auto_color_min_max():
                print(f"Mesh panel {i} color scaling with user inputs")
            pcolormesh_i = axes[i].pcolormesh(mesh_x,
                                              mesh_y,
                                              mesh_panel.tfr,
                                              vmin=mesh_panel.color_min,
                                              vmax=mesh_panel.color_max,
                                              cmap=mesh_base.colormap,
                                              shading=shading,
                                              snap=True)
            mesh_panel_div: AxesDivider = make_axes_locatable(axes[i])
            mesh_panel_cax: plt.Axes = mesh_panel_div.append_axes("right", size="1%", pad="0.5%")
            mesh_panel_cbar: Colorbar = fig.colorbar(
                pcolormesh_i,
                cax=mesh_panel_cax,
                ticks=[math.ceil(mesh_panel.color_min), math.floor(mesh_panel.color_max)],
                format=cbar_tick_fmt)
            mesh_panel_cbar.set_label(mesh_panel.cbar_units, rotation=270, size=fig_params.text_size)
            mesh_panel_cax.tick_params(labelsize="large")
            axes[i].set_ylabel(mesh_base.units_frequency, size=fig_params.text_size)
            axes[i].set_xlim(time_xmin, time_xmax)
            axes[i].set_ylim(frequency_fix_ymin, frequency_fix_ymax)
            axes[i].set_yscale(mesh_base.frequency_scaling)
            axes[i].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            axes[i].tick_params(axis="y", labelsize="large")
        elif panels_dict[i]["panel_type"] == "wf":
            wf_panel = panels_dict[i]["panel"]
            axes[i].plot(wf_panel_n_time_zero, wf_panel.sig)
            axes[i].set_ylabel(wf_panel.units, size=fig_params.text_size)
            axes[i].set_xlim(time_xmin, time_xmax)
            axes[i].tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
            axes[i].grid(True)
            axes[i].tick_params(axis="y", labelsize="large")
            if wf_panel.yscaling == "auto":
                axes[i].set_ylim(np.min(wf_panel.sig), np.max(wf_panel.sig))
                axes[i].ticklabel_format(style="plain", scilimits=(0, 0), axis="y")
            elif wf_panel.yscaling == "symmetric":
                axes[i].set_ylim(-np.max(np.abs(wf_panel.sig)), np.max(np.abs(wf_panel.sig)))
            elif wf_panel.yscaling == "positive":
                axes[i].set_ylim(0, np.max(np.abs(wf_panel.sig)))
            else:
                axes[i].set_ylim(-10, 10)
            axes[i].ticklabel_format(style="plain", scilimits=(0, 0), axis="y")
            axes[i].yaxis.get_offset_text().set_x(-0.034)
            wf_panel_a_div: AxesDivider = make_axes_locatable(axes[i])
            wf_panel_a_cax: plt.Axes = wf_panel_a_div.append_axes("right", size="1%", pad="0.5%")
            wf_panel_a_cax.axis("off")
        if i != 0 and i != n - 1:
            axes[i].margins(x=0)

    if plot_base.figure_title_show:
        axes[0].set_title(plot_base.figure_title)
    fig.text(.5, .01, time_label, ha='center', size=fig_params.text_size)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=.1, hspace=0.13)

    return fig


def plot_wf_mesh_vert_example(
        station_id: str,
        wf_panel_a_sig: np.ndarray,
        wf_panel_a_time: np.ndarray,
        mesh_time: np.ndarray,
        mesh_frequency: np.ndarray,
        mesh_panel_b_tfr: np.ndarray,
        params_tfr=plt_base.AudioParams(fa.AspectRatioType(3)),
        frequency_scaling: str = "log",
        mesh_shading: str = "auto",
        wf_panel_a_yscaling: str = "auto",
        wf_panel_a_ytick_style: str = "plain",
        mesh_panel_b_ytick_style: str = "sci",
        mesh_panel_b_colormap_scaling: str = "auto",
        mesh_panel_b_color_max: float = 15,
        mesh_panel_b_color_range: float = 15,
        mesh_panel_b_color_min: float = 0,
        start_time_epoch: float = 0,
        frequency_hz_ymin: float = None,
        frequency_hz_ymax: float = None,
        mesh_colormap: str = None,
        waveform_color: str = None,
        units_time: str = "s",
        units_frequency: str = "Hz",
        wf_panel_a_units: str = "Norm",
        mesh_panel_b_cbar_units: str = "bits",
        figure_title: str = "Time-Frequency Representation",
        figure_title_show: bool = True,
):
    """
    Plot 2 vertical panels - mesh (top panel) and signal waveform (bottom panel)

    :param wf_panel_a_yscaling: 'auto', 'symmetric', 'positive'
    :param station_id: name of station
    :param wf_panel_a_sig: array with signal waveform for bottom panel
    :param wf_panel_a_time: array with signal timestamps for bottom panel
    :param mesh_time: array with mesh time
    :param mesh_frequency: array with mesh frequencies
    :param mesh_panel_b_tfr: array with mesh tfr data for mesh plot (top panel)
    :param params_tfr: parameters for tfr. Check AudioParams().
    :param frequency_scaling: "log" or "linear". Default is "log"
    :param mesh_shading: type of mesh shading, one of "auto", "gouraud" or "else". Default is "auto"
    :param mesh_panel_b_colormap_scaling: color scaling for mesh plot (top panel). One of: "auto", "range" or "else"
        (use inputs given in mesh_panel_b_color_max, mesh_panel_b_color_range, mesh_panel_b_color_min).
        Default is "auto"
    :param mesh_panel_b_color_max: maximum value for color scaling for mesh plot (top panel). Default is 15.0
    :param mesh_panel_b_color_range:range between maximum and minimum values in color scaling for scatter plot
        (top panel). Default is 15.0
    :param mesh_panel_b_color_min: minimum value for color scaling for mesh plot (top panel). Default is 0.0
    :param start_time_epoch: start time in epoch UTC. Default is 0.0
    :param frequency_hz_ymin: minimum frequency for y-axis
    :param frequency_hz_ymax: maximum frequency for y-axis
    :param waveform_color: color of waveform for bottom panel. Default is "midnightblue"
    :param mesh_colormap: a Matplotlib Colormap instance or registered colormap name. If None, inherits style sheet spec
    :param units_time: units of time. Default is "s"
    :param units_frequency: units of frequency. Default is "Hz"
    :param wf_panel_a_units: units of waveform plot (bottom panel). Default is "Norm"
    :param wf_panel_a_ytick_style: 'plain' or 'sci'. Default is "plain"
    :param mesh_panel_b_ytick_style: 'plain' or 'sci'. Default is "sci"
    :param mesh_panel_b_cbar_units: units of colorbar for mesh plot (top panel). Default is "bits"
    :param figure_title: title of figure. Default is "Time-Frequency Representation"
    :param figure_title_show: show title if True. Default is True
    :return: plot
    """
    plot_base = plt_base.PlotBase(station_id=station_id,
                                  figure_title=figure_title,
                                  figure_title_show=figure_title_show,
                                  start_time_epoch=start_time_epoch,
                                  params_tfr=params_tfr,
                                  units_time=units_time
                                  )
    wf_base = plt_base.WaveformBase(station_id=station_id,
                                    label_panel_show=False,
                                    labels_fontweight=None,
                                    waveform_color=waveform_color,
                                    figure_title=figure_title,
                                    figure_title_show=figure_title_show,)
    mesh_base = plt_base.MeshBase(time=mesh_time, frequency=mesh_frequency,
                                  frequency_scaling=frequency_scaling, shading=mesh_shading,
                                  frequency_hz_ymin=frequency_hz_ymin,
                                  frequency_hz_ymax=frequency_hz_ymax,
                                  colormap=mesh_colormap,
                                  units_frequency=units_frequency)
    # build panels
    wf_panel = plt_base.WaveformPanel(sig=wf_panel_a_sig, time=wf_panel_a_time, units=wf_panel_a_units, label="(wf)",
                                      yscaling=wf_panel_a_yscaling, ytick_style=wf_panel_a_ytick_style)
    mesh_panel = plt_base.MeshPanel(tfr=mesh_panel_b_tfr,
                                    colormap_scaling=mesh_panel_b_colormap_scaling,
                                    color_max=mesh_panel_b_color_max,
                                    color_range=mesh_panel_b_color_range, color_min=mesh_panel_b_color_min,
                                    cbar_units=mesh_panel_b_cbar_units, ytick_style=mesh_panel_b_ytick_style)
    fig = plot_nwf_nmesh_vert(
        {0: {"panel_type": "mesh", "panel": mesh_panel}, 1: {"panel_type": "wf", "panel": wf_panel}},
        plot_base=plot_base,
        sanitize_times=True,
        mesh_base=mesh_base,
        wf_base=wf_base)

    return fig


def plot_cw_and_power(
        cw_panel_sig: np.ndarray,
        power_panel_sigs: List[np.ndarray],
        cw_panel_time: np.ndarray,
        power_panel_freqs: List[np.ndarray],
        power_panel_ls: List[str] = None,
        power_panel_lw: List[int] = None,
        power_panel_sig_labels: List[str] = None,
        cw_panel_units: str = "Norm",
        power_panel_y_units: str = "Power/Var(signal)",
        power_panel_x_units: str = "Frequency, Hz",
        params_tfr=fa.FigureParameters(fa.AspectRatioType(3)),
        units_time: str = "s",
        cw_panel_title: str = "CW",
        power_panel_title: str = "Power",
        figure_title_show: bool = True,
) -> Union[plt.Figure, None]:
    """
    Template for CW and power plots from intro set examples
    :param cw_panel_sig: CW signal waveform
    :param power_panel_sigs: list of power signal arrays
    :param cw_panel_time: CW time series
    :param power_panel_freqs: list of power frequency arrays
    :param power_panel_ls: list of line styles for power signals
    :param power_panel_lw: list of line widths for power signals
    :param power_panel_sig_labels: list of labels for power signals
    :param cw_panel_units: units of the CW signal waveform
    :param power_panel_y_units: y-axis units of the power signals
    :param power_panel_x_units: x-axis units of the power signals
    :param params_tfr: parameters for tfr; see AudioParams() in figure_attributes.py
    :param units_time: x-axis units for the CW panel
    :param cw_panel_title: title of the CW panel
    :param power_panel_title: title of the power panel
    :param figure_title_show: show panel titles if True; default is True
    :return: the figure
    """
    # Catch cases where there may not be any data
    time_xmin = cw_panel_time[0]
    time_xmax = cw_panel_time[-1]
    if time_xmin == time_xmax:
        print("No data to plot.")
        return

    # Figure starts here
    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = plt.subplots(
        1,
        2,
        figsize=(params_tfr.figure_size_x, params_tfr.figure_size_y),
    )
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    cw_panel: plt.Axes = axes[0]
    power_panel: plt.Axes = axes[1]

    if figure_title_show:
        cw_panel.set_title(cw_panel_title, size=params_tfr.text_size)
        power_panel.set_title(power_panel_title, size=params_tfr.text_size)

    cw_panel.plot(cw_panel_time, cw_panel_sig)
    cw_panel.set_ylabel(cw_panel_units, size=params_tfr.text_size)
    cw_panel.set_xlabel(f"Time ({units_time})", size=params_tfr.text_size)
    cw_panel.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
    cw_panel.tick_params(axis="y", which="both", left=True, labelleft=True, labelsize="large")
    cw_panel.grid(True)

    for i in range(len(power_panel_sigs)):
        power_panel.semilogx(power_panel_freqs[i], power_panel_sigs[i],
                             ls=power_panel_ls[i],
                             lw=power_panel_lw[i],
                             label=power_panel_sig_labels[i])

    power_panel.set_ylabel(power_panel_y_units, size=params_tfr.text_size)
    power_panel.set_xlabel(f"Frequency ({power_panel_x_units})", size=params_tfr.text_size)
    power_panel.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
    power_panel.tick_params(axis="y", which="both", left=True, labelleft=True, labelsize="large")
    power_panel.grid(True)
    power_panel.legend()

    fig.tight_layout()
    fig.subplots_adjust()
    return fig
