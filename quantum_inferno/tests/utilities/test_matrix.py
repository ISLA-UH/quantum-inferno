import unittest
import numpy as np
import quantum_inferno.utilities.matrix as mat


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.array_1d = np.array([1, 2, 3])
        cls.array_1d_repeat_1x6 = np.array([1, 2, 3, 1, 2, 3])
        cls.array_2d_repeat_3x2 = np.array([[1, 2, 3], [1, 2, 3]])
        cls.array_3d_ones_4x4 = np.ones((3, 4, 4))

    def test_array_from_number(self):
        self.assertTrue(np.array_equal(mat.array_from_number(5, (3, 4)), np.full((3, 4), 5)))
        self.assertTrue(np.array_equal(mat.array_from_number(5, (3, 4, 4)), np.full((3, 4, 4), 5)))

    def test_n_tile_array(self):
        self.assertTrue(
            np.array_equal(mat.n_tile_array(self.array_1d, 2, mat.MatrixAxis.ROW), self.array_2d_repeat_3x2)
        )
        self.assertTrue(
            np.array_equal(mat.n_tile_array(self.array_1d, 2, mat.MatrixAxis.COLUMN), self.array_2d_repeat_3x2.T)
        )
        self.assertTrue(np.array_equal(mat.n_tile_array(self.array_1d, 0, mat.MatrixAxis.ROW), self.array_1d))
        self.assertTrue(np.array_equal(mat.n_tile_array(self.array_1d, 0, mat.MatrixAxis.COLUMN), self.array_1d))

    def test_tile_array_to_shape(self):
        self.assertTrue((mat.tile_array_to_shape(self.array_1d, (1, 6), mat.MatrixAxis.ROW) == self.array_1d).all())
        self.assertTrue(
            (mat.tile_array_to_shape(self.array_1d, (3, 2), mat.MatrixAxis.ROW) == self.array_2d_repeat_3x2).all()
        )
        self.assertTrue(
            (mat.tile_array_to_shape(self.array_1d, (2, 3), mat.MatrixAxis.COLUMN) == self.array_2d_repeat_3x2.T).all()
        )
