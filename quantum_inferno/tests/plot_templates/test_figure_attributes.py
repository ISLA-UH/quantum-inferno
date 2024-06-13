import unittest

import quantum_inferno.plot_templates.figure_attributes as fa


class FigureAttributesTest(unittest.TestCase):
    def test_figure_attributes(self):
        test_fa = fa.FigureAttributes()
        self.assertEqual(test_fa.fig_scale, 2.0)
        self.assertEqual(test_fa.fig_dpi, 300)
        self.assertEqual(test_fa.ratio[0], 640)
        self.assertEqual(test_fa.ratio[1], 400)
        self.assertEqual(test_fa.font_size_1st_level, 10)
        self.assertEqual(test_fa.font_size_2nd_level, 8)
        self.assertEqual(test_fa.line_color, "k")
        self.assertEqual(test_fa.line_style, "-")
        self.assertEqual(test_fa.fig_aspect_ratio[0], 1280)
        self.assertEqual(test_fa.fig_aspect_ratio[1], 800)
        self.assertEqual(test_fa.fig_face_color, "w")
        self.assertEqual(test_fa.fig_edge_color, "w")
        self.assertAlmostEqual(test_fa.fig_size[0], 4.267, 2)
        self.assertAlmostEqual(test_fa.fig_size[1], 2.67, 2)
        self.assertEqual(test_fa.font_color, "k")
        self.assertEqual(test_fa.font_weight, "normal")
        self.assertEqual(test_fa.line_weight, 2)
        self.assertEqual(test_fa.tick_size, 8)
        self.assertEqual(test_fa.legend_label_size, 8)
        self.assertIsNone(test_fa.fig)


class FigureAttributesBlackTest(unittest.TestCase):
    def test_fa_black(self):
        test_fa = fa.FigureAttributesBackInBlack()
        self.assertEqual(test_fa.fig_face_color, "k")
        self.assertEqual(test_fa.fig_edge_color, "k")
        self.assertEqual(test_fa.font_color, "w")


class FigureParametersTest(unittest.TestCase):
    def test_figure_params_640_360(self):
        test_fp = fa.FigureParameters(fa.AspectRatioType.R640x360)
        self.assertEqual(test_fp.width, 640)
        self.assertEqual(test_fp.height, 360)
        self.assertAlmostEqual(test_fp.scale_factor, .33, 2)
        self.assertEqual(test_fp.figure_size_x, 42)
        self.assertEqual(test_fp.figure_size_y, 24)
        self.assertAlmostEqual(test_fp.text_size, 48)

    def test_figure_params_1280_720(self):
        test_fp = fa.FigureParameters(fa.AspectRatioType.R1280x720)
        self.assertEqual(test_fp.width, 1280)
        self.assertEqual(test_fp.height, 720)
        self.assertAlmostEqual(test_fp.scale_factor, .666, 2)
        self.assertEqual(test_fp.figure_size_x, 21)
        self.assertEqual(test_fp.figure_size_y, 12)
        self.assertAlmostEqual(test_fp.text_size, 24)

    def test_figure_params_1920_1080(self):
        test_fp = fa.FigureParameters(fa.AspectRatioType.R1920x1080)
        self.assertEqual(test_fp.width, 1920)
        self.assertEqual(test_fp.height, 1080)
        self.assertEqual(test_fp.scale_factor, 1.25)
        self.assertEqual(test_fp.figure_size_x, 11)
        self.assertEqual(test_fp.figure_size_y, 6)
        self.assertAlmostEqual(test_fp.text_size, 12)

    def test_figure_params_2560_1440(self):
        test_fp = fa.FigureParameters(fa.AspectRatioType.R2560x1440)
        self.assertEqual(test_fp.width, 2560)
        self.assertEqual(test_fp.height, 1440)
        self.assertAlmostEqual(test_fp.scale_factor, 1.33, 2)
        self.assertEqual(test_fp.figure_size_x, 10)
        self.assertEqual(test_fp.figure_size_y, 6)
        self.assertAlmostEqual(test_fp.text_size, 12)

    def test_figure_params_3840_2160(self):
        test_fp = fa.FigureParameters(fa.AspectRatioType.R3840x2160)
        self.assertEqual(test_fp.width, 3840)
        self.assertEqual(test_fp.height, 2160)
        self.assertEqual(test_fp.scale_factor, 2)
        self.assertEqual(test_fp.figure_size_x, 7)
        self.assertEqual(test_fp.figure_size_y, 4)
        self.assertAlmostEqual(test_fp.text_size, 8)


class AudioParamsTest(unittest.TestCase):
    def test_audio_params(self):
        test_ap = fa.AudioParams()
        self.assertEqual(test_ap.width, 1920)
        self.assertEqual(test_ap.height, 1080)
        self.assertEqual(test_ap.scale_factor, 1.25)
        self.assertEqual(test_ap.figure_size_x, 11)
        self.assertEqual(test_ap.figure_size_y, 6)
        self.assertAlmostEqual(test_ap.text_size, 12)
        self.assertTrue(test_ap.fill_gaps)
