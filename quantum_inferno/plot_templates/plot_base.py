"""
Base classes, constants, and functions used to create plots
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from quantum_inferno.plot_templates.figure_attributes import AudioParams

# sets the default y limits for plots
DEFAULT_YLIM_MIN = -10
DEFAULT_YLIM_MAX = 10

# todo: all enumerated values for plots
WF_Y_SCALING_VALS = ["auto", "symmetric", "positive", "else"]
MESH_SHADING_VALS = ["auto", "gouraud", "flat", "nearest"]  # pyplot pcolormesh.shading
COLORMAP_SCALING_VALS = ["auto", "range", "else"]
YTICK_STYLE_VALS = ["sci", "scientific", "plain"]  # pyplot ytick style
AXIS_SCALE_VALS = ["function", "linear", "log", "functionlog", "symlog", "logit", "asinh"]  # pyplot yscale and xscale


@dataclass
class PlotBase:
    """
    Plotting base class with parameters used by all plots

    Attributes:
        station_id: str, id of the station being plotted.  Required
        figure_title: str, the title of the plot.  Required
        figure_title_show: bool, if True, show the figure title.  Default True
        start_time_epoch: float, the epoch start time of the data.  Default 0.
        params_tfr: AudioParams, parameters for plotting audio data.  Default AudioParams()
        units_time: str, label of units for time component
    """
    station_id: str
    figure_title: str
    figure_title_show: bool = True
    start_time_epoch: float = 0.
    params_tfr: AudioParams = AudioParams()
    units_time: str = "s"


@dataclass
class MeshBase:
    """
    Base class for Mesh plots

    Attributes:
        time: np.ndarray of the timestamps.  Required
        frequency: np.ndarray of the frequency data.  Required
        frequency_scaling: str, scaling for the frequency.  refer to AXIS_SCALE_VALS for options.  Default "log"
        shading: str, shading of the plot.  options: "auto", "gouraud", "flat", "nearest".  Default "auto"
        frequency_hz_ymin: optional float, minimum frequency.  Default None
        frequency_hz_ymax: optional float, maximum frequency.  Default None
        colormap: Optional str, colormap.  Default None
        units_frequency: str, units of the frequency.  Default "Hz"
    """
    time: np.ndarray
    frequency: np.ndarray
    frequency_scaling: str = "log"
    shading: str = "auto"
    frequency_hz_ymin: Optional[float] = None
    frequency_hz_ymax: Optional[float] = None
    colormap: Optional[str] = None
    units_frequency: str = "Hz"

    def __post_init__(self):
        # Autoscale to mesh frequency range
        if self.frequency_hz_ymax is None:
            self.frequency_hz_ymax = self.frequency[-1]
        if self.frequency_hz_ymin is None:
            self.frequency_hz_ymin = self.frequency[0]
        if self.frequency_scaling not in AXIS_SCALE_VALS:
            self.frequency_scaling = "log"
        if self.shading not in MESH_SHADING_VALS:
            self.shading = "auto"


def mesh_colormap_limits(
        mesh_array: np.ndarray,
        colormap_scaling: str = "auto",
        color_range: float = 16.
) -> Tuple[float, float]:
    """
    Find colormap limits for plotting

    :param mesh_array: array with mesh
    :param colormap_scaling: one of: "auto" (max/min of input mesh),
        "range" (max of input mesh - color range given) or
        "abs" (absolute max/min of input mesh).  Default "auto"
    :param color_range: range of colors.  Default is 16.0
    :return: colormap min and max values
    """
    if colormap_scaling == "auto":
        color_max = np.max(mesh_array)
        color_min = np.min(mesh_array)
    elif colormap_scaling == "range":
        color_max = np.max(mesh_array)
        color_min = color_max - color_range
    else:
        # print("User specified mesh color limits will be applied.")
        color_max = np.max(np.abs(mesh_array))
        color_min = np.min(np.abs(mesh_array))

    return color_min, color_max


@dataclass
class MeshPanel:
    """
    Panel for Mesh plots

    Attributes:
        tfr: np.ndarray of the data to plot.  Required
        colormap_scaling: str, scaling for the colormap.  options: "auto", "range", "else".  Default "auto"
        color_max: float, maximum color value.  Default 15
        color_range: float, range of color values.  Default 15
        color_min: float, minimum color value.  Default 0
        cbar_units: str, units to display for colorbar.  Default "bits"
        ytick_style: str, style for yticks.  options: "sci", "scientific", "plain".  Default "sci"
    """
    tfr: np.ndarray
    colormap_scaling: str = "auto"
    color_max: float = 15.
    color_range: float = 15.
    color_min: float = 0.
    cbar_units: str = "bits"
    ytick_style: str = 'sci'

    def __post_init__(self):
        if self.colormap_scaling not in COLORMAP_SCALING_VALS:
            self.colormap_scaling = "else"
        else:
            self.set_color_min_max()
        if self.ytick_style not in YTICK_STYLE_VALS:
            self.ytick_style = "sci"

    def set_color_min_max(self):
        """
        Set color min-max if the colormap_scaling is "auto" or "range"
        """
        if self.is_auto_color_min_max():
            self.color_min, self.color_max = mesh_colormap_limits(self.tfr, self.colormap_scaling, self.color_range)

    def is_auto_color_min_max(self) -> bool:
        """
        :return: True if colormap_scaling is "auto" or "range", which auto-sets the color range
        """
        return self.colormap_scaling in ["auto", "range"]


@dataclass
class WaveformBase(PlotBase):
    """
    Base class for Waveform plots

    Attributes:
        label_panel_show: bool, if True, show the label.  Default False
        labels_fontweight: optional str, font weight of the labels.  Default None
        waveform_color: optional str, color of the waveform.  Default None
    """
    label_panel_show: bool = False
    labels_fontweight: Optional[str] = None
    waveform_color: Optional[str] = None


@dataclass
class WaveformPanel:
    """
    Panel for Waveform plots

    Attributes:
        sig: np.ndarray, the signal to plot.  Required
        time: np.ndarray, the timestamps of the data.  Required
        units: str, units of the signal.  Default "Norm"
        label: str, label for the data.  Default "(wf)"
        yscaling: str, scaling for y-axis.  options: "auto", "symmetric", "positive", "else".  Default "auto"
        ytick_style: str, style for yticks.  options: "sci", "scientific", "plain".  Default "plain"
    """
    sig: np.ndarray
    time: np.ndarray
    units: str = "Norm"
    label: str = "(wf)"
    yscaling: str = "auto"
    ytick_style: str = "plain"

    def __post_init__(self):
        if self.ytick_style not in YTICK_STYLE_VALS:
            self.ytick_style = "plain"
        if self.yscaling not in WF_Y_SCALING_VALS:
            self.yscaling = "else"
