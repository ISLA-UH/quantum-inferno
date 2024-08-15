"""
This module contains functionalized versions of the quantum inferno plot templates used by examples.
These functions are intended to run with the full set of parameters for each chart, without the
user knowing the underlying classes that group the values.
"""
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt

from quantum_inferno.plot_templates import figure_attributes as fa
from quantum_inferno.plot_templates import plot_base as plt_base
from quantum_inferno.plot_templates import plot_templates as plt_tpl


def mesh_panel_colormap_scaling(
        mesh_panel_custom_color_scaling: Union[tuple, float, None],
        mesh_panel_tfr: np.ndarray,
        mesh_panel_cbar_units: str = "bits",
        mesh_panel_ytick_style: str = "sci") -> plt_base.MeshPanel:
    """
    Create a mesh panel with custom colormap scaling
    :param mesh_panel_custom_color_scaling: either a float, a tuple of floats, or None. If tuple, values in the tuple
        are used to set panel colormap limits according to (vmin, vmax). If float, range colormap scaling is used with
        the provided float as the range. If None, auto colormap scaling is used. Default is range colormap scaling at
        15.0 range.
    :param mesh_panel_tfr: array with mesh tfr data for mesh plot
    :param mesh_panel_cbar_units: units of colorbar for mesh plot. Default is "bits"
    :param mesh_panel_ytick_style: 'plain' or 'sci'. Default is "sci"
    :return: mesh panel object
    """
    if type(mesh_panel_custom_color_scaling) == tuple:
        mesh_panel = plt_base.MeshPanel(tfr=mesh_panel_tfr,
                                        colormap_scaling="else",
                                        color_max=mesh_panel_custom_color_scaling[1],
                                        color_min=mesh_panel_custom_color_scaling[0],
                                        cbar_units=mesh_panel_cbar_units,
                                        ytick_style=mesh_panel_ytick_style)
    elif type(mesh_panel_custom_color_scaling) == float:
        mesh_panel = plt_base.MeshPanel(tfr=mesh_panel_tfr,
                                        colormap_scaling="range",
                                        color_range=mesh_panel_custom_color_scaling,
                                        cbar_units=mesh_panel_cbar_units,
                                        ytick_style=mesh_panel_ytick_style)
    else:
        mesh_panel = plt_base.MeshPanel(colormap_scaling="auto",
                                        tfr=mesh_panel_tfr,
                                        cbar_units=mesh_panel_cbar_units,
                                        ytick_style=mesh_panel_ytick_style)
    return mesh_panel


def plot_wf_mesh_vert_example(
        station_id: str,
        wf_panel_a_sig: np.ndarray,
        wf_panel_a_time: np.ndarray,
        mesh_time: np.ndarray,
        mesh_frequency: np.ndarray,
        mesh_panel_b_tfr: np.ndarray,
        params_tfr=plt_base.AudioParams(),
        frequency_scaling: str = "log",
        mesh_shading: str = "auto",
        wf_panel_a_yscaling: str = "auto",
        wf_panel_a_ytick_style: str = "plain",
        mesh_panel_b_ytick_style: str = "sci",
        mesh_panel_b_custom_color_scaling: Union[tuple, float, None] = 15.0,
        start_time_epoch: float = 0,
        frequency_hz_ymin: float = None,
        frequency_hz_ymax: float = None,
        mesh_colormap: str = None,
        units_time: str = "s",
        units_frequency: str = "Hz",
        wf_panel_a_units: str = "Norm",
        mesh_panel_b_cbar_units: str = "bits",
        figure_title: str = "Time-Frequency Representation",
        figure_title_show: bool = True,
) -> plt.Figure:
    """
    Plot 2 vertical panels - mesh (top panel) and signal waveform (bottom panel)

    :param wf_panel_a_yscaling: 'auto', 'symmetric', 'positive'
    :param station_id: name of station
    :param wf_panel_a_sig: array with signal waveform for bottom panel
    :param wf_panel_a_time: array with signal timestamps for bottom panel
    :param mesh_time: array with mesh time
    :param mesh_frequency: array with mesh frequencies
    :param mesh_panel_b_tfr: array with mesh tfr data for mesh plot (top panel)
    :param params_tfr: Display parameters for tfr. Check AudioParams().
    :param frequency_scaling: "log" or "linear". Default is "log"
    :param mesh_shading: type of mesh shading, one of "auto", "gouraud" or "else". Default is "auto"
    :param mesh_panel_b_custom_color_scaling: either a float, a tuple of floats, or None. If tuple, values in the tuple
        are used to set panel b colormap limits according to (vmin, vmax). If float, range colormap scaling is used with
        the provided float as the range. If None, auto colormap scaling is used. Default is range colormap scaling at
        15.0 range.
    :param start_time_epoch: start time in epoch UTC. Default is 0.0
    :param frequency_hz_ymin: minimum frequency for y-axis
    :param frequency_hz_ymax: maximum frequency for y-axis
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
    wf_base = plt_base.WaveformPlotBase(station_id=station_id,
                                        figure_title=figure_title,
                                        figure_title_show=figure_title_show,
                                        start_time_epoch=start_time_epoch,
                                        params_tfr=params_tfr,
                                        units_time=units_time
                                        )
    mesh_base = plt_base.MeshBase(time=mesh_time, frequency=mesh_frequency,
                                  frequency_scaling=frequency_scaling, shading=mesh_shading,
                                  frequency_hz_ymin=frequency_hz_ymin,
                                  frequency_hz_ymax=frequency_hz_ymax,
                                  colormap=mesh_colormap,
                                  units_frequency=units_frequency)
    # build panels
    wf_panel = plt_base.WaveformPanel(sig=wf_panel_a_sig, time=wf_panel_a_time, units=wf_panel_a_units, label="(wf)",
                                      yscaling=wf_panel_a_yscaling, ytick_style=wf_panel_a_ytick_style)
    mesh_panel = mesh_panel_colormap_scaling(mesh_panel_custom_color_scaling=mesh_panel_b_custom_color_scaling,
                                             mesh_panel_tfr=mesh_panel_b_tfr,
                                             mesh_panel_cbar_units=mesh_panel_b_cbar_units,
                                             mesh_panel_ytick_style=mesh_panel_b_ytick_style)
    fig = plt_tpl.plot_n_mesh_wf_vert(mesh_base, [mesh_panel], wf_base, wf_panel,
                                      use_default_size=False)

    return fig


def plot_wf_mesh_mesh_vert_example(
        station_id: str,
        wf_panel_a_sig: np.ndarray,
        wf_panel_a_time: np.ndarray,
        mesh_time: np.ndarray,
        mesh_frequency: np.ndarray,
        mesh_panel_b_tfr: np.ndarray,
        mesh_panel_c_tfr: np.ndarray,
        params_tfr=plt_base.AudioParams(fa.AspectRatioType(3)),
        wf_panel_a_yscaling: str = "auto",
        wf_panel_a_ytick_style: str = "plain",
        mesh_panel_b_ytick_style: str = "sci",
        mesh_panel_c_ytick_style: str = "sci",
        frequency_scaling: str = "log",
        mesh_shading: str = "auto",
        mesh_panel_b_custom_color_scaling: Union[tuple, float, None] = 15.0,
        mesh_panel_c_custom_color_scaling: Union[tuple, float, None] = 15.0,
        start_time_epoch: float = 0,
        frequency_hz_ymin: float = None,
        frequency_hz_ymax: float = None,
        mesh_colormap: str = None,
        units_time: str = "s",
        units_frequency: str = "Hz",
        wf_panel_a_units: str = "Norm",
        mesh_panel_b_cbar_units: str = "bits",
        mesh_panel_c_cbar_units: str = "bits",
        figure_title: str = "Time-Frequency Representation",
        figure_title_show: bool = True,
) -> plt.Figure:
    """
    Plot 3 vertical panels - mesh (top panel), mesh (middle panel) and signal waveform (bottom panel)

    :param mesh_panel_b_ytick_style: y-tick style for the middle mesh panel
    :param wf_panel_a_ytick_style: y-tick style for the waveform panel
    :param wf_panel_a_yscaling: y-scaling for the waveform panel
    :param mesh_panel_c_ytick_style: y-tick style for the top mesh panel
    :param station_id: name of station
    :param wf_panel_a_sig: array with signal waveform for bottom panel
    :param wf_panel_a_time: array with signal timestamps for bottom panel
    :param mesh_time: array with mesh time
    :param mesh_frequency: array with mesh frequencies
    :param mesh_panel_b_tfr: array with mesh tfr data for mesh plot (middle panel)
    :param mesh_panel_c_tfr: array with mesh tfr data for mesh plot (top panel)
    :param params_tfr: parameters for tfr. Check AudioParams().
    :param frequency_scaling: "log" or "linear". Default is "log"
    :param mesh_shading: type of mesh shading, one of "auto", "gouraud" or "else". Default is "auto"
    :param mesh_panel_b_custom_color_scaling: either a float, a tuple of floats, or None. If tuple, values in the tuple
        are used to set panel b colormap limits according to (vmin, vmax). If float, range colormap scaling is used with
        the provided float as the range. If None, auto colormap scaling is used. Default is range colormap scaling at
        15.0 range.
    :param mesh_panel_c_custom_color_scaling: either a float, a tuple of floats, or None. If tuple, values in the tuple
        are used to set panel c colormap limits according to (vmin, vmax). If float, range colormap scaling is used with
        the provided float as the range. If None, auto colormap scaling is used. Default is range colormap scaling at
        15.0 range.
    :param start_time_epoch: start time in epoch UTC. Default is 0.0
    :param frequency_hz_ymin: minimum frequency for y-axis
    :param frequency_hz_ymax: maximum frequency for y-axis
    :param mesh_colormap: a Matplotlib Colormap instance or registered colormap name. If None, inherits style sheet spec
    :param units_time: units of time. Default is "s"
    :param units_frequency: units of frequency. Default is "Hz"
    :param wf_panel_a_units: units of waveform plot (bottom panel). Default is "Norm"
    :param mesh_panel_b_cbar_units: units of colorbar for mesh plot (middle panel). Default is "bits"
    :param mesh_panel_c_cbar_units: units of colorbar for mesh plot (top panel). Default is "bits"
    :param figure_title: title of figure. Default is "Time-Frequency Representation"
    :param figure_title_show: show title if True. Default is True
    :return: plot
    """
    plot_base = plt_base.WaveformPlotBase(station_id=station_id,
                                          figure_title=figure_title,
                                          figure_title_show=figure_title_show,
                                          start_time_epoch=start_time_epoch,
                                          params_tfr=params_tfr,
                                          units_time=units_time
                                          )
    mesh_base = plt_base.MeshBase(time=mesh_time, frequency=mesh_frequency,
                                  frequency_scaling=frequency_scaling, shading=mesh_shading,
                                  frequency_hz_ymin=frequency_hz_ymin,
                                  frequency_hz_ymax=frequency_hz_ymax,
                                  colormap=mesh_colormap,
                                  units_frequency=units_frequency)
    # build panels
    wf_panel = plt_base.WaveformPanel(sig=wf_panel_a_sig, time=wf_panel_a_time, units=wf_panel_a_units, label="(wf)",
                                      yscaling=wf_panel_a_yscaling, ytick_style=wf_panel_a_ytick_style)
    mesh_panel_b = mesh_panel_colormap_scaling(
        mesh_panel_custom_color_scaling=mesh_panel_b_custom_color_scaling,
        mesh_panel_tfr=mesh_panel_b_tfr,
        mesh_panel_cbar_units=mesh_panel_b_cbar_units,
        mesh_panel_ytick_style=mesh_panel_b_ytick_style)
    mesh_panel_c = mesh_panel_colormap_scaling(
        mesh_panel_custom_color_scaling=mesh_panel_c_custom_color_scaling,
        mesh_panel_tfr=mesh_panel_c_tfr,
        mesh_panel_cbar_units=mesh_panel_c_cbar_units,
        mesh_panel_ytick_style=mesh_panel_c_ytick_style)
    fig = plt_tpl.plot_n_mesh_wf_vert(mesh_base, [mesh_panel_c, mesh_panel_b], plot_base, wf_panel,
                                      use_default_size=False)

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
        params_tfr=fa.AudioParams(),
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
    cw_panel = plt_base.CwPanel(cw_panel_sig, cw_panel_time, cw_panel_units, units_time, cw_panel_title)
    power_panel_data = []
    for f in range(len(power_panel_sigs)):
        power_panel_data.append(plt_base.PowerPanelData(power_panel_sigs[f], power_panel_freqs[f], power_panel_ls[f],
                                                        power_panel_lw[f], power_panel_sig_labels[f]))
    power_panel = plt_base.PowerPanel(power_panel_data, power_panel_y_units, power_panel_x_units, power_panel_title)
    cw_pow_base = plt_base.CwPowerPlotBase(params_tfr, figure_title_show)
    return plt_tpl.plot_cw_and_power(cw_panel, power_panel, cw_pow_base)
