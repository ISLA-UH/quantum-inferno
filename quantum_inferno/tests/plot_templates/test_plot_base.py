"""
Tests for the plot_base module.
"""
import unittest

import numpy as np

import quantum_inferno.plot_templates.plot_base as pb


class PlotBaseTest(unittest.TestCase):
    def test_plot_base(self):
        test_plot_base = pb.PlotBase(station_id="test", figure_title="test")
        self.assertEqual(test_plot_base.station_id, "test")
        self.assertEqual(test_plot_base.figure_title, "test")
        self.assertTrue(test_plot_base.figure_title_show)
        self.assertEqual(test_plot_base.start_time_epoch, 0.)
        self.assertEqual(test_plot_base.units_time, "s")


class MeshBaseTest(unittest.TestCase):
    def test_mesh_base(self):
        test_mesh_base = pb.MeshBase(time=np.array([1, 2, 3]), frequency=np.array([1, 2, 3]))
        self.assertEqual(test_mesh_base.frequency_scaling, "log")
        self.assertEqual(test_mesh_base.shading, "auto")
        self.assertEqual(test_mesh_base.frequency_hz_ymin, 1)
        self.assertEqual(test_mesh_base.frequency_hz_ymax, 3)
        self.assertIsNone(test_mesh_base.colormap)
        self.assertEqual(test_mesh_base.units_frequency, "Hz")
        test_mesh_base = pb.MeshBase(time=np.array([0, 1, 2, 3]), frequency=np.array([0, 1, 2, 3]))
        self.assertEqual(test_mesh_base.frequency_hz_ymin, 1)
        self.assertEqual(test_mesh_base.frequency_hz_ymax, 3)

    def test_mesh_base_freq_scal(self):
        test_mesh_base = pb.MeshBase(time=np.array([1, 2, 3]), frequency=np.array([1, 2, 3]),
                                     frequency_scaling="test")
        self.assertEqual(test_mesh_base.frequency_scaling, "log")

    def test_mesh_base_shading(self):
        test_mesh_base = pb.MeshBase(time=np.array([1, 2, 3]), frequency=np.array([1, 2, 3]),
                                     shading="test")
        self.assertEqual(test_mesh_base.shading, "auto")


class MeshColormapLimitsTest(unittest.TestCase):
    def test_mesh_colormap_limits(self):
        mesh_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        test_min, test_max = pb.mesh_colormap_limits(mesh_array)
        self.assertEqual(test_min, 0.)
        self.assertEqual(test_max, 9.)

    def test_mesh_colormap_limits_short(self):
        mesh_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        test_min, test_max = pb.mesh_colormap_limits(mesh_array, colormap_scaling="range", color_range=4)
        self.assertEqual(test_min, 5.)
        self.assertEqual(test_max, 9.)

    def test_mesh_colormap_limits_long(self):
        mesh_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        test_min, test_max = pb.mesh_colormap_limits(mesh_array, colormap_scaling="range", color_range=12)
        self.assertEqual(test_min, -3.)
        self.assertEqual(test_max, 9.)

    def test_mesh_colormap_limits_abs_scaling(self):
        mesh_array = np.array([-1, 2, 3, 4, -5, 6, 7, 8, -9, 0])
        test_min, test_max = pb.mesh_colormap_limits(mesh_array, colormap_scaling="abs")
        self.assertEqual(test_min, 0.)
        self.assertEqual(test_max, 9.)


class MeshPanelTest(unittest.TestCase):
    def test_mesh_panel(self):
        test_mesh_panel = pb.MeshPanel(tfr=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
        self.assertTrue(test_mesh_panel.is_auto_color_min_max())
        self.assertEqual(test_mesh_panel.panel_label_color, "k")

    def test_mesh_user_panel(self):
        test_mesh_panel = pb.MeshPanel(tfr=np.array([1, 2, 3]), colormap_scaling="user", color_min=5, color_max=10)
        self.assertFalse(test_mesh_panel.is_auto_color_min_max())
        self.assertEqual(test_mesh_panel.color_min, 5)
        self.assertEqual(test_mesh_panel.color_max, 10)

    def test_mesh_ytick_style(self):
        test_mesh_panel = pb.MeshPanel(tfr=np.array([1, 2, 3]), ytick_style="test")
        self.assertEqual(test_mesh_panel.ytick_style, "sci")


class WaveformBaseTest(unittest.TestCase):
    def test_waveform_base(self):
        test_wf_base = pb.WaveformPlotBase(station_id="test", figure_title="test")
        self.assertFalse(test_wf_base.label_panel_show)
        self.assertEqual(test_wf_base.labels_fontweight, "bold")
        self.assertIsNone(test_wf_base.waveform_color)


class WaveformPanelTest(unittest.TestCase):
    def test_waveform_panel(self):
        test_wf_panel = pb.WaveformPanel(sig=np.array([1, 2, 3]), time=np.array([1, 2, 3]))
        self.assertEqual(test_wf_panel.units, "Norm")
        self.assertEqual(test_wf_panel.label, "(wf)")
        self.assertEqual(test_wf_panel.yscaling, "auto")
        self.assertEqual(test_wf_panel.ytick_style, "plain")
        self.assertEqual(test_wf_panel.panel_label_color, "k")

    def test_waveform_panel_yscaling(self):
        test_wf_panel = pb.WaveformPanel(sig=np.array([1, 2, 3]), time=np.array([1, 2, 3]), yscaling="test")
        self.assertEqual(test_wf_panel.yscaling, "else")

    def test_waveform_panel_ytick_style(self):
        test_wf_panel = pb.WaveformPanel(sig=np.array([1, 2, 3]), time=np.array([1, 2, 3]), ytick_style="test")
        self.assertEqual(test_wf_panel.ytick_style, "plain")
