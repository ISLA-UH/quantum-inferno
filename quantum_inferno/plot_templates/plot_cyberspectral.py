"""
This module contains plotting functions for Time Frequency Representations
"""
import enum
from typing import List, Tuple
import math
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable, AxesDivider
from dataclasses import dataclass
from quantum_inferno.utilities.date_time import utc_datetime_to_utc_timestamp, utc_timestamp_to_utc_datetime

# import quantum_inferno.utils_date_time as dt


# TODO: Add native color stylings
#  https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
# TODO: For dark background
#  plt.style.use('dark_background')


class FigureAttributes:
    """
    This is the most basic parent class -- sets the plot canvas as well as figure handling functions. This is where
    figure size, font style and sizes, line weights and colors, etc. are established. All subsequent plot classes
    will inherit these attributes, overriding them if necessary.

    Attributes
    __________

    fig_size_ratio: 2d array of figure width, height ratio
    fontsize1_scale: int, scale for fontsize level 1 (titles, axes labels...)
    fontsize2: int, scale for fontsize level 2 (legend labels, ticks...)
    line_color: string, color for line in plot
    line_style: string, style of line in plot
    """

    def __init__(
        self, fig_size_ratio=np.array([640, 400]), fontsize1_scale=5, fontsize2_scale=4, line_color="k", line_style="-"
    ):

        self.fig_scale = 2.0
        self.fig_dpi = 300
        self.ratio = fig_size_ratio
        self.font_size_1st_level = np.rint(self.fig_scale * fontsize1_scale)
        self.font_size_2nd_level = np.rint(self.fig_scale * fontsize2_scale)
        self.line_color = line_color
        self.line_style = line_style
        self.fig_aspect_ratio = np.rint(self.fig_scale * self.ratio)  # was 640, 360, ratio 16:9
        self.fig_face_color = "w"
        self.fig_edge_color = self.fig_face_color
        self.fig_size = self.fig_aspect_ratio / self.fig_dpi
        self.font_color = "k"
        self.font_weight = "normal"
        self.line_weight = np.rint(self.fig_scale * 1)
        self.tick_size = self.font_size_2nd_level
        self.legend_label_size = self.font_size_2nd_level

        self.fig = None


class AspectRatioType(enum.Enum):
    """
    Enumeration denoting aspect ratios
    """

    R640x360 = 1
    R1280x720 = 2
    R1920x1080 = 3
    R2560x1440 = 4
    R3840x2160 = 5


class FigureParameters:
    """
    This class encapsulates the logic of computing figure size
    """

    def __init__(self, aspect_ratio: AspectRatioType):
        if aspect_ratio == AspectRatioType.R640x360:
            self.width = 640
            self.height = 360
            self.scale_factor = 1.0 / 3.0
        elif aspect_ratio == AspectRatioType.R1280x720:
            self.width = 1280
            self.height = 720
            self.scale_factor = 2.0 / 3.0
        elif aspect_ratio == AspectRatioType.R1920x1080:
            self.width = 1920
            self.height = 1080
            self.scale_factor = 1.25
        elif aspect_ratio == AspectRatioType.R2560x1440:
            self.width = 2560
            self.height = 1440
            self.scale_factor = 4.0 / 3.0
        else:
            self.width = 3840
            self.height = 2160
            self.scale_factor = 2.0

        scale = self.scale_factor * self.height / 8
        self.figure_size_x = int(self.width / scale)
        self.figure_size_y = int(self.height / scale)
        self.text_size = int(2.0 * self.height / scale)


# Set Aspect Ratio
@dataclass
class AudioParams:
    fill_gaps: bool = True
    figure_parameters: FigureParameters = FigureParameters(AspectRatioType.R1920x1080)


def origin_time_correction(
    time_input: np.ndarray, start_time_epoch: float, units_time: str, start_time_sanitized: bool = True
) -> Tuple[str, np.ndarray]:
    """
    Sanitize time
    :param time_input: array with timestamps
    :param start_time_epoch: start time in epoch UTC
    :param units_time: units of time
    :param start_time_sanitized: True or False
    :return: time label and time elapsed from start
    """
    # Sanitizing/scrubbing time is a key function.
    # TEST EXTENSIVELY!
    # Elapsed time from start of the record. Watch out with start alignment.
    # TODO: THIS WILL BREAK: TEST EXTENSIVELY!
    if start_time_sanitized:
        time_elapsed_from_start = time_input - time_input[0]
    else:
        time_elapsed_from_start = time_input
    # Need to construct a function to return time from start_time_epoch if they are not the same
    if start_time_epoch != time_input[0]:
        time_from_epoch_start = time_input[0] - start_time_epoch
    if start_time_epoch == 0:
        # Time sanitized if no input provided
        time_label: str = f"Time ({units_time})"
    else:
        start_datetime_epoch = utc_timestamp_to_utc_datetime(timestamp=start_time_epoch)
        dt_str: str = start_datetime_epoch.strftime("%Y-%m-%d %H:%M:%S")
        time_label: str = f"Time ({units_time}) from UTC {dt_str}"

    return time_label, time_elapsed_from_start


def mesh_time_frequency_edges(
    frequency: np.ndarray,
    time: np.ndarray,
    frequency_ymin: float,
    frequency_ymax: float,
    frequency_scaling: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Find time and frequency edges for plotting

    :param frequency: array with frequencies
    :param time: array with timestamps
    :param frequency_ymin: minimum frequency for y-axis
    :param frequency_ymax: maximum frequency for y-axis
    :param frequency_scaling: "log" or "linear". Default is "linear"
    :return: min and max frequency for plot, time and frequency edges
    """

    if frequency_ymin > frequency_ymax:
        print("Higher frequency must be greater than lower frequency")
    if frequency[2] < frequency[1]:
        print("Frequency must be increasing, flip it")
    if time[2] < time[1]:
        print("Time must be increasing, flip it")

    t_half_bin: float = np.abs(time[2] - time[1]) / 2.0
    t_edge: np.ndarray = np.append(time[0] - t_half_bin, time + t_half_bin)

    if frequency_scaling == "log":
        k_edge: float = np.sqrt(frequency[-1] / frequency[-2])
        f_edge: np.ndarray = np.append(frequency / k_edge, k_edge * frequency[-1])
    else:
        f_half_bin: float = (frequency[2] - frequency[1]) / 2.0
        f_edge: np.ndarray = np.append(frequency[0] - f_half_bin, frequency + f_half_bin)

    # Initialize
    frequency_fix_ymin = 1.0 * frequency_ymin
    frequency_fix_ymax = 1.0 * frequency_ymax

    if frequency_ymin < f_edge[1]:
        frequency_fix_ymin = f_edge[0]
    if frequency_fix_ymin <= 0 and frequency_scaling == "log":
        frequency_fix_ymin = f_edge[1]
    if frequency_ymax > f_edge[-1]:
        frequency_fix_ymax = f_edge[-1]

    return t_edge, f_edge, frequency_fix_ymin, frequency_fix_ymax


def mesh_colormap_limits(mesh_array: np.ndarray, colormap_scaling: str = "auto", color_range: float = 16):
    """
    Find colormap limits for plotting

    :param mesh_array: array with mesh
    :param colormap_scaling: one of: "auto" (max/min of input mesh), "range" (max of input mesh minus color range given)
        or "abs" (absolute max/min of input mesh)
    :param color_range: default is 16.0
    :return: colormap min and max values
    """

    if colormap_scaling == "auto":
        color_max = np.max(mesh_array)
        color_min = np.min(mesh_array)
    elif colormap_scaling == "range":
        color_max = np.max(mesh_array)
        color_min = color_max - color_range
    else:
        print("Specify mesh color limits, using min max")
        color_max = np.max(np.abs(mesh_array))
        color_min = np.min(np.abs(mesh_array))

    return color_min, color_max


# # BEGIN PLOTS: THESE ARE THE MAIN TEMPLATES
def plot_wf_wf_wf_vert(
    station_id: str,
    wf_panel_a_sig: np.ndarray,
    wf_panel_a_time: np.ndarray,
    wf_panel_b_sig: np.ndarray,
    wf_panel_b_time: np.ndarray,
    wf_panel_c_sig: np.ndarray,
    wf_panel_c_time: np.ndarray,
    start_time_epoch: float = 0,
    start_time_sanitized: bool = True,
    wf_panel_a_units: str = "Norm",
    wf_panel_b_units: str = "Norm",
    wf_panel_c_units: str = "Norm",
    params_tfr=AudioParams(),
    units_time: str = "s",
    figure_title: str = "Time Domain Representation",
    figure_title_show: bool = True,
    label_panel_show: bool = False,
    labels_panel_a: str = "(a)",
    labels_panel_b: str = "(b)",
    labels_panel_c: str = "(c)",
    labels_fontweight: str = None,
) -> None:
    """
    Template for aligned time-series display

    :param station_id: name of station
    :param wf_panel_c_sig: array with signal waveform for top panel
    :param wf_panel_c_time: array with signal timestamps for top panel
    :param wf_panel_b_sig: array with signal waveform for middle panel
    :param wf_panel_b_time: array with signal timestamps for middle panel
    :param wf_panel_a_sig: array with signal waveform for bottom panel
    :param wf_panel_a_time: array with signal timestamps for bottom panel
    :param start_time_epoch: start time in epoch UTC. Default is 0.0
    :param wf_panel_c_units: units of signal (top panel). Default is "Norm"
    :param wf_panel_b_units: units of signal (middle panel). Default is "Norm"
    :param wf_panel_a_units: units of signal (bottom panel). Default is "Norm"
    :param params_tfr: parameters for tfr. Check AudioParams().
    :param waveform_color: color of waveforms. Default is "midnightblue"
    :param units_time: units of time. Default is "s"
    :param figure_title: title of figure. Default is "Time Domain Representation"
    :param figure_title_show: True to display title, False for publications
    :param label_panel_show: show panel labelling. Default is False, True for publication
    :param labels_panel_a: label for bottom panel. Default is (a)
    :param labels_panel_b: label for middle panel. Default is (b)
    :param labels_panel_c: label for top panel. Default is (c)
    :param labels_fontweight: matplotlib.text property
    :return: plot
    """

    if start_time_epoch == 0:
        time_label: str = f"Time ({units_time})"
        if start_time_sanitized:
            # Time sanitized if no input provided
            wf_panel_c_time_zero = wf_panel_c_time - wf_panel_a_time[0]
            wf_panel_b_time_zero = wf_panel_b_time - wf_panel_a_time[0]
            wf_panel_a_time_zero = wf_panel_a_time - wf_panel_a_time[0]
        else:
            wf_panel_c_time_zero = wf_panel_c_time
            wf_panel_b_time_zero = wf_panel_b_time
            wf_panel_a_time_zero = wf_panel_a_time
    else:
        # TODO: Allow for different time formats
        start_datetime_epoch = utc_timestamp_to_utc_datetime(start_time_epoch)
        dt_str: str = start_datetime_epoch.strftime("%Y-%m-%d %H:%M:%S")
        time_label: str = f"Time ({units_time}) from UTC {dt_str}"
        wf_panel_c_time_zero = wf_panel_c_time - start_time_epoch
        wf_panel_b_time_zero = wf_panel_b_time - start_time_epoch
        wf_panel_a_time_zero = wf_panel_a_time - start_time_epoch

    # Catch cases where there may not be any data
    time_xmin = wf_panel_a_time_zero[0]
    time_xmax = wf_panel_a_time_zero[-1]
    if time_xmin == time_xmax:
        time_xmin = wf_panel_b_time_zero[0]
        time_xmax = wf_panel_b_time_zero[-1]
    if time_xmin == time_xmax:
        time_xmin = wf_panel_c_time_zero[0]
        time_xmax = wf_panel_c_time_zero[-1]
    if time_xmin == time_xmax:
        print("No data to plot for " + figure_title)
        return

    # print("Panel time:", wf_panel_c_time_zero[0], wf_panel_c_time_zero[-1])
    # Figure starts here
    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = plt.subplots(
        3,
        1,
        figsize=(params_tfr.figure_parameters.figure_size_x, params_tfr.figure_parameters.figure_size_y),
        sharex=True,
    )
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    wf_panel_c: plt.Axes = axes[0]
    wf_panel_b: plt.Axes = axes[1]
    wf_panel_a: plt.Axes = axes[2]

    if figure_title_show:
        wf_panel_c.set_title(f"{figure_title} at Station {station_id}")

    wf_panel_c.plot(wf_panel_c_time_zero, wf_panel_c_sig)
    if label_panel_show:
        wf_panel_c.text(
            0.01,
            0.95,
            labels_panel_c,
            transform=wf_panel_c.transAxes,
            fontsize=params_tfr.figure_parameters.text_size,
            fontweight=labels_fontweight,
            va="top",
        )
    wf_panel_c.set_ylabel(wf_panel_c_units, size=params_tfr.figure_parameters.text_size)
    wf_panel_c.set_xlim(time_xmin, time_xmax)
    wf_panel_c.tick_params(axis="x", which="both", bottom=False, labelbottom=False, labelsize="large")
    wf_panel_c.grid(True)
    wf_panel_c.tick_params(axis="y", labelsize="large")
    wf_panel_c.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
    wf_panel_c.yaxis.get_offset_text().set_x(-0.034)

    wf_panel_b.plot(wf_panel_b_time_zero, wf_panel_b_sig)
    if label_panel_show:
        wf_panel_b.text(
            0.01,
            0.95,
            labels_panel_b,
            transform=wf_panel_b.transAxes,
            fontsize=params_tfr.figure_parameters.text_size,
            fontweight=labels_fontweight,
            va="top",
        )
    wf_panel_b.set_ylabel(wf_panel_b_units, size=params_tfr.figure_parameters.text_size)
    wf_panel_b.set_xlim(time_xmin, time_xmax)
    wf_panel_b.tick_params(axis="x", which="both", bottom=False, labelbottom=False, labelsize="large")
    wf_panel_b.grid(True)
    wf_panel_b.tick_params(axis="y", labelsize="large")
    wf_panel_b.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
    wf_panel_b.yaxis.get_offset_text().set_x(-0.034)

    wf_panel_a.plot(wf_panel_a_time_zero, wf_panel_a_sig)
    if label_panel_show:
        wf_panel_a.text(
            0.01,
            0.95,
            labels_panel_a,
            transform=wf_panel_a.transAxes,
            fontsize=params_tfr.figure_parameters.text_size,
            fontweight=labels_fontweight,
            va="top",
        )
    wf_panel_a.set_ylabel(wf_panel_a_units, size=params_tfr.figure_parameters.text_size)
    wf_panel_a.set_xlim(time_xmin, time_xmax)
    wf_panel_a.tick_params(axis="x", which="both", bottom=False, labelbottom=True, labelsize="large")
    wf_panel_a.grid(True)
    wf_panel_a.tick_params(axis="y", labelsize="large")
    wf_panel_a.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
    wf_panel_a.yaxis.get_offset_text().set_x(-0.034)

    fig.text(0.5, 0.01, time_label, ha="center", size=params_tfr.figure_parameters.text_size)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, hspace=0.13)


def plot_wf_mesh_mesh_vert(
    station_id: str,
    wf_panel_a_sig: np.ndarray,
    wf_panel_a_time: np.ndarray,
    mesh_time: np.ndarray,
    mesh_frequency: np.ndarray,
    mesh_panel_b_tfr: np.ndarray,
    mesh_panel_c_tfr: np.ndarray,
    params_tfr=AudioParams(),
    frequency_scaling: str = "log",
    mesh_shading: str = "auto",
    mesh_panel_b_colormap_scaling: str = "auto",
    mesh_panel_b_color_max: float = 15,
    mesh_panel_b_color_range: float = 15,
    mesh_panel_b_color_min: float = 0,
    mesh_panel_c_colormap_scaling: str = "auto",
    mesh_panel_c_color_max: float = 15,
    mesh_panel_c_color_range: float = 15,
    mesh_panel_c_color_min: float = 0,
    start_time_epoch: float = 0,
    frequency_hz_ymin: float = None,
    frequency_hz_ymax: float = None,
    mesh_colormap: str = "inferno",
    units_time: str = "s",
    units_frequency: str = "Hz",
    wf_panel_a_units: str = "Norm",
    mesh_panel_b_cbar_units: str = "bits",
    mesh_panel_c_cbar_units: str = "bits",
    figure_title: str = "Time-Frequency Representation",
    figure_title_show: bool = True,
):

    """
    Plot 3 vertical panels - mesh (top panel), mesh (middle panel) and signal waveform (bottom panel)

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
     :param mesh_panel_b_colormap_scaling: color scaling for mesh plot (middle panel). One of: "auto", "range" or "else"
        (use inputs given in mesh_panel_b_color_max, mesh_panel_b_color_range, mesh_panel_b_color_min). Default is "auto"
    :param mesh_panel_b_color_max: maximum value for color scaling for mesh plot (middle panel). Default is 15.0
    :param mesh_panel_b_color_range: range between maximum and minimum values in color scaling for mesh plot
        (middle panel). Default is 15.0
    :param mesh_panel_b_color_min: minimum value for color scaling for mesh plot (middle panel). Default is 0.0
    :param mesh_panel_c_colormap_scaling: color scaling for mesh plot (top panel). One of: "auto", "range" or "else"
        (use inputs given in mesh_panel_c_color_max, mesh_panel_c_color_range, mesh_panel_c_color_min). Default is "auto"
    :param mesh_panel_c_color_max: maximum value for color scaling for mesh plot (top panel). Default is 15.0
    :param mesh_panel_c_color_range:range between maximum and minimum values in color scaling for scatter plot
        (top panel). Default is 15.0
    :param mesh_panel_c_color_min: minimum value for color scaling for mesh plot (top panel). Default is 0.0
    :param start_time_epoch: start time in epoch UTC. Default is 0.0
    :param frequency_hz_ymin: minimum frequency for y-axis
    :param frequency_hz_ymax: maximum frequency for y-axis
    :param waveform_color: color of waveform for bottom panel. Default is "midnightblue"
    :param mesh_colormap: a Matplotlib Colormap instance or registered colormap name. Default is "inferno"
    :param units_time: units of time. Default is "s"
    :param units_frequency: units of frequency. Default is "Hz"
    :param wf_panel_a_units: units of waveform plot (bottom panel). Default is "Norm"
    :param mesh_panel_b_cbar_units: units of colorbar for mesh plot (middle panel). Default is "bits"
    :param mesh_panel_c_cbar_units: units of colorbar for mesh plot (top panel). Default is "bits"
    :param figure_title: title of figure. Default is "Time-Frequency Representation"
    :param figure_title_show: show title if True. Default is True
    :return: plot
    """

    # Autoscale to mesh frequency range
    if frequency_hz_ymax is None:
        frequency_hz_ymax = mesh_frequency[-1]
    if frequency_hz_ymin is None:
        frequency_hz_ymin = mesh_frequency[0]

    # Time zeroing and scrubbing, if needed
    time_label, wf_panel_a_elapsed_time = origin_time_correction(wf_panel_a_time, start_time_epoch, units_time)

    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # frequency and time must be increasing!
    t_edge, f_edge, frequency_fix_ymin, frequency_fix_ymax = mesh_time_frequency_edges(
        frequency=mesh_frequency,
        time=mesh_time,
        frequency_ymin=frequency_hz_ymin,
        frequency_ymax=frequency_hz_ymax,
        frequency_scaling=frequency_scaling,
    )

    # Figure starts here
    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = plt.subplots(
        3,
        1,
        figsize=(params_tfr.figure_parameters.figure_size_x, params_tfr.figure_parameters.figure_size_y),
        sharex=True,
    )
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    mesh_panel_c: plt.Axes = axes[0]
    mesh_panel_b: plt.Axes = axes[1]
    wf_panel_a: plt.Axes = axes[2]
    # bottom_panel_picker: plt.Axes = axes[3]

    # Top panel mesh --------------------------
    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # frequency and time must be increasing!

    # Display preference
    wf_panel_a_time_xmin: int = wf_panel_a_elapsed_time[0]
    wf_panel_a_time_xmax: int = t_edge[-1]

    # Override, default is autoscaling to min and max values
    if mesh_panel_b_colormap_scaling == "auto":
        mesh_panel_b_color_min, mesh_panel_b_color_max = mesh_colormap_limits(
            mesh_panel_b_tfr, mesh_panel_b_colormap_scaling, mesh_panel_b_color_range
        )
    elif mesh_panel_b_colormap_scaling == "range":
        mesh_panel_b_color_min, mesh_panel_b_color_max = mesh_colormap_limits(
            mesh_panel_b_tfr, mesh_panel_b_colormap_scaling, mesh_panel_b_color_range
        )
    else:
        "Mesh 1 color scaling with user inputs"

    if mesh_panel_c_colormap_scaling == "auto":
        mesh_panel_c_color_min, mesh_panel_c_color_max = mesh_colormap_limits(
            mesh_panel_c_tfr, mesh_panel_c_colormap_scaling, mesh_panel_c_color_range
        )
    elif mesh_panel_c_colormap_scaling == "range":
        mesh_panel_c_color_min, mesh_panel_c_color_max = mesh_colormap_limits(
            mesh_panel_c_tfr, mesh_panel_c_colormap_scaling, mesh_panel_c_color_range
        )
    else:
        "Mesh 0 color scaling with user inputs"

    # Setup color map ticks
    all_cbar_ticks_lens: List[int] = [
        len(str(math.ceil(mesh_panel_c_color_min))),
        len(str(math.floor(mesh_panel_c_color_max))),
        len(str(math.ceil(mesh_panel_b_color_min))),
        len(str(math.floor(mesh_panel_b_color_max))),
    ]
    max_cbar_tick_len: int = sorted(all_cbar_ticks_lens)[-1]
    cbar_tick_fmt: str = f"%-{max_cbar_tick_len}s"

    if mesh_shading == "auto":
        pcolormesh_top: QuadMesh = mesh_panel_c.pcolormesh(
            mesh_time,
            mesh_frequency,
            mesh_panel_c_tfr,
            vmin=mesh_panel_c_color_min,
            vmax=mesh_panel_c_color_max,
            cmap=mesh_colormap,
            shading=mesh_shading,
            snap=True,
        )
    elif mesh_shading == "gouraud":
        pcolormesh_top: QuadMesh = mesh_panel_c.pcolormesh(
            mesh_time,
            mesh_frequency,
            mesh_panel_c_tfr,
            vmin=mesh_panel_c_color_min,
            vmax=mesh_panel_c_color_max,
            cmap=mesh_colormap,
            shading=mesh_shading,
            snap=True,
        )
    else:
        pcolormesh_top: QuadMesh = mesh_panel_c.pcolormesh(
            t_edge,
            f_edge,
            mesh_panel_c_tfr,
            vmin=mesh_panel_c_color_min,
            vmax=mesh_panel_c_color_max,
            cmap=mesh_colormap,
            snap=True,
        )

    mesh_panel_c_div: AxesDivider = make_axes_locatable(mesh_panel_c)
    mesh_panel_c_cax: plt.Axes = mesh_panel_c_div.append_axes("right", size="1%", pad="0.5%")
    mesh_panel_c_cbar: Colorbar = fig.colorbar(
        pcolormesh_top,
        cax=mesh_panel_c_cax,
        ticks=[math.ceil(mesh_panel_c_color_min), math.floor(mesh_panel_c_color_max)],
        format=cbar_tick_fmt,
    )
    mesh_panel_c_cbar.set_label(mesh_panel_c_cbar_units, rotation=270, size=params_tfr.figure_parameters.text_size)
    mesh_panel_c_cax.tick_params(labelsize="large")
    # if figure_title_show:
    #     mesh_panel_c.set_title(f"{figure_title} at Station {station_id}")
    if figure_title_show:
        mesh_panel_c.set_title(f"{figure_title} ({station_id})")
    mesh_panel_c.set_ylabel(units_frequency, size=params_tfr.figure_parameters.text_size)
    mesh_panel_c.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
    mesh_panel_c.set_ylim(frequency_fix_ymin, frequency_fix_ymax)
    mesh_panel_c.set_yscale(frequency_scaling)
    mesh_panel_c.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    mesh_panel_c.tick_params(axis="y", labelsize="large")

    # Middle panel mesh --------------------------

    if mesh_shading == "auto":
        pcolormesh_mid: QuadMesh = mesh_panel_b.pcolormesh(
            mesh_time,
            mesh_frequency,
            mesh_panel_b_tfr,
            vmin=mesh_panel_b_color_min,
            vmax=mesh_panel_b_color_max,
            cmap=mesh_colormap,
            shading=mesh_shading,
            snap=True,
        )
    elif mesh_shading == "gouraud":
        pcolormesh_mid: QuadMesh = mesh_panel_b.pcolormesh(
            mesh_time,
            mesh_frequency,
            mesh_panel_b_tfr,
            vmin=mesh_panel_b_color_min,
            vmax=mesh_panel_b_color_max,
            cmap=mesh_colormap,
            shading=mesh_shading,
            snap=True,
        )
    else:
        pcolormesh_mid: QuadMesh = mesh_panel_b.pcolormesh(
            t_edge,
            f_edge,
            mesh_panel_b_tfr,
            vmin=mesh_panel_b_color_min,
            vmax=mesh_panel_b_color_max,
            cmap=mesh_colormap,
            snap=True,
        )

    mesh_panel_b_div: AxesDivider = make_axes_locatable(mesh_panel_b)
    mesh_panel_b_cax: plt.Axes = mesh_panel_b_div.append_axes("right", size="1%", pad="0.5%")
    mesh_panel_b_cbar: Colorbar = fig.colorbar(
        pcolormesh_mid,
        cax=mesh_panel_b_cax,
        ticks=[math.ceil(mesh_panel_b_color_min), math.floor(mesh_panel_b_color_max)],
        format=cbar_tick_fmt,
    )
    mesh_panel_b_cbar.set_label(mesh_panel_b_cbar_units, rotation=270, size=params_tfr.figure_parameters.text_size)
    mesh_panel_b_cax.tick_params(labelsize="large")

    mesh_panel_b.set_ylabel(units_frequency, size=params_tfr.figure_parameters.text_size)
    mesh_panel_b.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
    mesh_panel_b.set_ylim(frequency_fix_ymin, frequency_fix_ymax)
    mesh_panel_b.margins(x=0)
    mesh_panel_b.set_yscale(frequency_scaling)
    mesh_panel_b.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    mesh_panel_b.tick_params(axis="y", labelsize="large")

    # Waveform panel
    wf_panel_a.plot(wf_panel_a_elapsed_time, wf_panel_a_sig)
    wf_panel_a.set_ylabel(wf_panel_a_units, size=params_tfr.figure_parameters.text_size)
    wf_panel_a.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
    wf_panel_a.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
    wf_panel_a.grid(True)
    wf_panel_a.tick_params(axis="y", labelsize="large")

    # TODO: ADD OPTIONS AS BELOW
    # wf_panel_a.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
    wf_panel_a.ticklabel_format(style="plain", scilimits=(0, 0), axis="y")
    wf_panel_a.yaxis.get_offset_text().set_x(-0.034)

    wf_panel_a_div: AxesDivider = make_axes_locatable(wf_panel_a)
    wf_panel_a_cax: plt.Axes = wf_panel_a_div.append_axes("right", size="1%", pad="0.5%")
    wf_panel_a_cax.axis("off")

    fig.text(0.5, 0.01, time_label, ha="center", size=params_tfr.figure_parameters.text_size)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, hspace=0.13)

    return fig


def plot_wf_mesh_vert(
    station_id: str,
    wf_panel_a_sig: np.ndarray,
    wf_panel_a_time: np.ndarray,
    mesh_time: np.ndarray,
    mesh_frequency: np.ndarray,
    mesh_panel_b_tfr: np.ndarray,
    params_tfr=AudioParams(),
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
    start_time_sanitized: bool = True,
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
        (use inputs given in mesh_panel_b_color_max, mesh_panel_b_color_range, mesh_panel_b_color_min). Default is "auto"
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

    # Autoscale to mesh frequency range
    if frequency_hz_ymax is None:
        frequency_hz_ymax = mesh_frequency[-1]
    if frequency_hz_ymin is None:
        frequency_hz_ymin = mesh_frequency[0]

    # Time zeroing and scrubbing, if needed
    time_label, wf_panel_a_elapsed_time = origin_time_correction(
        wf_panel_a_time, start_time_epoch, units_time, start_time_sanitized
    )

    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # frequency and time must be increasing!
    t_edge, f_edge, frequency_fix_ymin, frequency_fix_ymax = mesh_time_frequency_edges(
        frequency=mesh_frequency,
        time=mesh_time,
        frequency_ymin=frequency_hz_ymin,
        frequency_ymax=frequency_hz_ymax,
        frequency_scaling=frequency_scaling,
    )

    # Figure starts here
    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = plt.subplots(
        2,
        1,
        figsize=(params_tfr.figure_parameters.figure_size_x, params_tfr.figure_parameters.figure_size_y),
        sharex=True,
    )
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    mesh_panel_b: plt.Axes = axes[0]
    wf_panel_a: plt.Axes = axes[1]
    # bottom_panel_picker: plt.Axes = axes[3]

    # Top panel mesh --------------------------
    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # frequency and time must be increasing!

    # Display preference
    wf_panel_a_time_xmin: int = wf_panel_a_elapsed_time[0]
    wf_panel_a_time_xmax: int = t_edge[-1]

    if mesh_panel_b_colormap_scaling == "auto":
        mesh_panel_b_color_min, mesh_panel_b_color_max = mesh_colormap_limits(
            mesh_panel_b_tfr, mesh_panel_b_colormap_scaling, mesh_panel_b_color_range
        )
    elif mesh_panel_b_colormap_scaling == "range":
        mesh_panel_b_color_min, mesh_panel_b_color_max = mesh_colormap_limits(
            mesh_panel_b_tfr, mesh_panel_b_colormap_scaling, mesh_panel_b_color_range
        )
    else:
        "Mesh 0 color scaling with user inputs"

    # Setup color map ticks
    all_cbar_ticks_lens: List[int] = [
        len(str(math.ceil(mesh_panel_b_color_min))),
        len(str(math.floor(mesh_panel_b_color_max))),
    ]
    max_cbar_tick_len: int = sorted(all_cbar_ticks_lens)[-1]
    cbar_tick_fmt: str = f"%-{max_cbar_tick_len}s"

    if mesh_shading == "auto":
        pcolormesh_top: QuadMesh = mesh_panel_b.pcolormesh(
            mesh_time,
            mesh_frequency,
            mesh_panel_b_tfr,
            vmin=mesh_panel_b_color_min,
            vmax=mesh_panel_b_color_max,
            cmap=mesh_colormap,
            shading=mesh_shading,
            snap=True,
        )
    elif mesh_shading == "gouraud":
        pcolormesh_top: QuadMesh = mesh_panel_b.pcolormesh(
            mesh_time,
            mesh_frequency,
            mesh_panel_b_tfr,
            vmin=mesh_panel_b_color_min,
            vmax=mesh_panel_b_color_max,
            cmap=mesh_colormap,
            shading=mesh_shading,
            snap=True,
        )
    else:
        pcolormesh_top: QuadMesh = mesh_panel_b.pcolormesh(
            t_edge,
            f_edge,
            mesh_panel_b_tfr,
            vmin=mesh_panel_b_color_min,
            vmax=mesh_panel_b_color_max,
            cmap=mesh_colormap,
            snap=True,
        )

    mesh_panel_b_div: AxesDivider = make_axes_locatable(mesh_panel_b)
    mesh_panel_b_cax: plt.Axes = mesh_panel_b_div.append_axes("right", size="1%", pad="0.5%")
    mesh_panel_b_cbar: Colorbar = fig.colorbar(
        pcolormesh_top,
        cax=mesh_panel_b_cax,
        ticks=[math.ceil(mesh_panel_b_color_min), math.floor(mesh_panel_b_color_max)],
        format=cbar_tick_fmt,
    )
    # mesh_panel_b_cbar: Colorbar = fig.colorbar(pcolormesh, cax=mesh_panel_b_cax,
    #                                         ticks=[colormin_top, colormax_top],
    #                                         format=cbar_tick_fmt)
    mesh_panel_b_cbar.set_label(mesh_panel_b_cbar_units, rotation=270, size=params_tfr.figure_parameters.text_size)
    mesh_panel_b_cax.tick_params(labelsize="large")
    if figure_title_show:
        mesh_panel_b.set_title(f"{figure_title} {station_id}")
    mesh_panel_b.set_ylabel(units_frequency, size=params_tfr.figure_parameters.text_size)
    mesh_panel_b.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)
    mesh_panel_b.set_ylim(frequency_fix_ymin, frequency_fix_ymax)
    # mesh_panel_b.get_xaxis().set_ticklabels([])
    mesh_panel_b.set_yscale(frequency_scaling)
    mesh_panel_b.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    mesh_panel_b.tick_params(axis="y", labelsize="large")
    if frequency_scaling == "linear":
        # Only works for linear range
        mesh_panel_b.ticklabel_format(style=mesh_panel_b_ytick_style, scilimits=(0, 0), axis="y")
    # Waveform panel
    wf_panel_a.plot(wf_panel_a_elapsed_time, wf_panel_a_sig, color=waveform_color)
    wf_panel_a.set_ylabel(wf_panel_a_units, size=params_tfr.figure_parameters.text_size)
    wf_panel_a.set_xlim(wf_panel_a_time_xmin, wf_panel_a_time_xmax)

    wf_panel_a.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelsize="large")
    wf_panel_a.grid(True)
    wf_panel_a.tick_params(axis="y", labelsize="large")
    # TODO: Standardize with inputs
    if wf_panel_a_yscaling == "auto":
        wf_panel_a.set_ylim(np.min(wf_panel_a_sig), np.max(wf_panel_a_sig))
        wf_panel_a.ticklabel_format(style="plain", scilimits=(0, 0), axis="y")
    elif wf_panel_a_yscaling == "symmetric":
        wf_panel_a.set_ylim(-np.max(np.abs(wf_panel_a_sig)), np.max(np.abs(wf_panel_a_sig)))
    elif wf_panel_a_yscaling == "positive":
        wf_panel_a.set_ylim(0, np.max(np.abs(wf_panel_a_sig)))
    else:
        wf_panel_a.set_ylim(-10, 10)
    wf_panel_a.ticklabel_format(style=wf_panel_a_ytick_style, scilimits=(0, 0), axis="y")
    wf_panel_a.yaxis.get_offset_text().set_x(-0.034)

    wf_panel_a_div: AxesDivider = make_axes_locatable(wf_panel_a)
    wf_panel_a_cax: plt.Axes = wf_panel_a_div.append_axes("right", size="1%", pad="0.5%")
    wf_panel_a_cax.axis("off")

    fig.text(0.5, 0.01, time_label, ha="center", size=params_tfr.figure_parameters.text_size)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1, hspace=0.13)

    return fig
