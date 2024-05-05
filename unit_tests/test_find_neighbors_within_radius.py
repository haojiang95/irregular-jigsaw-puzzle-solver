import unittest
import numpy as np
from cython_libs import find_neighbors_within_radius


class TestFindNeighborsWithinRadius(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFindNeighborsWithinRadius, self).__init__(*args, **kwargs)
        self._data = np.array(
            [
                [0, 0],
                [1.1, 0],
                [2.1, 0],
                [3.1, 0],
                [4, 0],
                [3.2, 0],
                [2.2, 0],
                [1.2, 0],
            ],
        )

    def test_no_wrapping(self):
        self.assertTrue(
            np.array_equal(
                np.sort(find_neighbors_within_radius(self._data, 2, 1.5), axis=0),
                np.array([[1.1, 0], [2.1, 0], [3.1, 0]]),
            )
        )

    def test_wrapping_from_the_start(self):
        self.assertTrue(
            np.array_equal(
                np.sort(find_neighbors_within_radius(self._data, 1, 1.5), axis=0),
                np.array([[0, 0], [1.1, 0], [1.2, 0], [2.1, 0], [2.2, 0]]),
            )
        )

    def test_wrapping_from_the_end(self):
        self.assertTrue(
            np.array_equal(
                np.sort(find_neighbors_within_radius(self._data, 7, 1.5), axis=0),
                np.array([[0, 0], [1.1, 0], [1.2, 0], [2.1, 0], [2.2, 0]]),
            )
        )

    def test_entire_array(self):
        self.assertTrue(
            np.array_equal(
                np.sort(find_neighbors_within_radius(self._data, 4, 10), axis=0),
                np.sort(self._data, axis=0),
            )
        )


if __name__ == "__main__":
    unittest.main()
