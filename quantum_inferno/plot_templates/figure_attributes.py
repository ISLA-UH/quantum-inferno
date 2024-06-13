"""
Base class for plotting figures.  Sets most common attributes for all plots.
"""
import enum

import numpy as np


class FigureAttributes:
    """
    This is the most basic parent class -- sets the plot canvas as well as figure handling functions. This is where
    figure size, font style and sizes, line weights and colors, etc. are established. All subsequent plot classes
    will inherit these attributes, overriding them if necessary.

    Attributes:
        fig_scale: float, scale factor for the figure.  Default 2.0
        fig_dpi: int, dots per inch for the figure.  Default 300
        ratio: np.array, 2d array of figure width, height ratio.  Default [640, 400]
        font_size_1st_level: int, scale for fontsize level 1 (titles, axes labels...).  Default 5
        font_size_2nd_level: int, scale for fontsize level 2 (legend labels, ticks...).  Default 4
        line_color: str, color for line in plot.  Default 'k' (black)
        line_style: str, style of line in plot.  Default '-' (solid line)
        fig_aspect_ratio: np.array, figure aspect ratio.  Default fig_scale * ratio
        fig_face_color: str, face color of the figure.  Default "w" (white)
        fig_edge_color: str, edge color of the figure.  Default fig_face_color
        fig_size: np.array, size of the figure.  Default fig_aspect_ratio / fig_dpi
        font_color: str, color of the font.  Default "k" (black)
        font_weight: str, weight of the font.  Default "normal"
        line_weight: int, weight of the line.  Default fig_scale * 1
        tick_size: int, size of the ticks.  Default font_size_2nd_level
        legend_label_size: int, size of the legend labels.  Default font_size_2nd_level
        fig: figure object.  Default None
    """
    def __init__(self, fig_size_ratio=np.array([640, 400]), fontsize1_scale=5, fontsize2_scale=4,
                 line_color='k', line_style='-'):
        """
        Initialize the FigureAttributes

        :param fig_size_ratio: 2d array of figure width, height ratio.  Default [640, 400]
        :param fontsize1_scale: int, scale for fontsize level 1 (titles, axes labels...).  Default 5
        :param fontsize2_scale: int, scale for fontsize level 2 (legend labels, ticks...)  Default 4
        :param line_color: string, color for line in plot.  Default 'k' (black)
        :param line_style: string, style of line in plot.  Default '-' (solid line)
        """
        self.fig_scale = 2.0
        self.fig_dpi = 300
        self.ratio = fig_size_ratio
        self.font_size_1st_level = np.rint(self.fig_scale * fontsize1_scale)
        self.font_size_2nd_level = np.rint(self.fig_scale * fontsize2_scale)
        self.line_color = line_color
        self.line_style = line_style
        self.fig_aspect_ratio = np.rint(self.fig_scale * self.ratio)
        self.fig_face_color = "w"
        self.fig_edge_color = self.fig_face_color
        self.fig_size = self.fig_aspect_ratio / self.fig_dpi
        self.font_color = "k"
        self.font_weight = "normal"
        self.line_weight = np.rint(self.fig_scale * 1)
        self.tick_size = self.font_size_2nd_level
        self.legend_label_size = self.font_size_2nd_level
        self.fig = None


class FigureAttributesBackInBlack(FigureAttributes):
    """
    Adapts FigureAttributes to have black coloring
    Refer to FigureAttributes for attribute information.
    Notable changes to default values:
        fig_face_color: "k" (black)
        font_color: "w" (white)
    """
    def __init__(
            self, fig_size_ratio=np.array([640, 400]),
            fontsize1_scale=5, fontsize2_scale=4, line_color="w", line_style="-"
    ):
        """
        Initialize the FigureAttributes

        :param fig_size_ratio: 2d array of figure width, height ratio.  Default [640, 400]
        :param fontsize1_scale: int, scale for fontsize level 1 (titles, axes labels...).  Default 5
        :param fontsize2_scale: int, scale for fontsize level 2 (legend labels, ticks...)  Default 4
        :param line_color: string, color for line in plot.  Default 'w' (white)
        :param line_style: string, style of line in plot.  Default '-' (solid line)
        """
        super().__init__(fig_size_ratio, fontsize1_scale, fontsize2_scale, line_color, line_style)
        self.fig_face_color = "k"
        self.fig_edge_color = self.fig_face_color
        self.font_color = "w"


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
    Allows users to set the figure size.  Refer to AspectRatioType for available aspect ratios.
    Default aspect ratio is 3840x2160.

    Attributes:
        width: int, width in pixels of the plot
        height: int, height in pixels of the plot
        scale_factor: float, scale factor for text and other elements
        figure_size_x: int, size of the figure in x direction
        figure_size_y: int, size of the figure in y direction
        text_size: int, size of the text
    """
    def __init__(self, aspect_ratio: AspectRatioType):
        """
        :param aspect_ratio: refer to AspectRatioType class for valid options
        """
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
        self.figure_size_y = int(8. / self.scale_factor)
        self.text_size = int(16.0 / self.scale_factor)


# Set Aspect Ratio
class AudioParams(FigureParameters):
    """
    Audio plot figure parameters.  Refer to FigureParameters class for valid options.
    Uses 1920X1080 as default for FigureParameters

    Attributes:
        fill_gaps: bool, if True, gaps in audio are filled.  Default True
    """
    def __init__(self, aspect_ratio: AspectRatioType = AspectRatioType.R1920x1080, fill_gaps: bool = True):
        super().__init__(aspect_ratio)
        self.fill_gaps: bool = fill_gaps
