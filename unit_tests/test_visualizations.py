import unittest
from unittest import mock

import networkx as nx
import numpy as np
from skimage.transform import EuclideanTransform

import visualizations
from data_structures.puzzle_piece import PuzzlePiece


class TestVisualizations(unittest.TestCase):
    def test_overlay_mask_blends_color_into_masked_pixels(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        result = visualizations.overlay_mask(image, mask, visualizations.GREEN)

        np.testing.assert_array_equal(result[0, 0], np.array([0, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(
            result[1, 1], np.array([0, 76, 0], dtype=np.uint8)
        )

    def test_draw_contours_on_image_marks_contour_points_red(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        contour_points = [np.array([[1, 2], [3, 0]], dtype=int)]

        result = visualizations.draw_contours_on_image(image, contour_points)

        np.testing.assert_array_equal(result[2, 1], np.array(visualizations.RED))
        np.testing.assert_array_equal(result[0, 3], np.array(visualizations.RED))

    def test_draw_feature_points_marks_positive_and_negative_features(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        feature_point_positions = np.array([[10, 10], [20, 20]], dtype=int)
        normal_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        features = np.array([[1.0, 0.5], [-1.0, 0.5]])

        result = visualizations.draw_feature_points(
            image,
            feature_point_positions,
            normal_vectors,
            features,
        )

        np.testing.assert_array_equal(result[10, 5], np.array(visualizations.RED))
        np.testing.assert_array_equal(result[15, 20], np.array(visualizations.BLUE))
        np.testing.assert_array_equal(result[10, 18], np.array(visualizations.GREEN))

    def test_visualize_pairwise_matching_result_expands_canvas_and_draws_contours(self):
        mask1 = np.zeros((4, 4), dtype=bool)
        mask2 = np.zeros((2, 2), dtype=bool)
        mask1[1:3, 1:3] = True
        mask2[:, :] = True
        contour_points1 = np.array([[1, 1]], dtype=int)
        contour_points2 = np.array([[0, 0]], dtype=int)

        result = visualizations.visualize_pairwise_matching_result(
            mask1,
            mask2,
            EuclideanTransform(translation=(-3, -1)),
            contour_points1,
            contour_points2,
        )

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape[2], 3)
        self.assertGreater(result.shape[0], mask1.shape[0])
        self.assertGreater(result.shape[1], mask1.shape[1])
        self.assertTrue(
            np.any(np.all(result == np.array(visualizations.YELLOW), axis=2))
        )
        self.assertTrue(
            np.any(np.all(result == np.array(visualizations.MAGENTA), axis=2))
        )

    def test_visualize_matching_results_renders_pose_tree_with_deterministic_colors(
        self,
    ):
        piece_a = PuzzlePiece(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=bool),
            (0, 0),
        )
        piece_b = PuzzlePiece(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=bool),
            (0, 0),
        )
        pose_tree = nx.DiGraph()
        pose_tree.add_nodes_from([0, 1])
        pose_tree.add_edge(0, 1, relative_pose=EuclideanTransform(translation=(3, 0)))

        with mock.patch(
            "utils.misc.random_color",
            side_effect=[(10, 20, 30), (40, 50, 60)],
        ):
            result = visualizations.visualize_matching_results(
                [piece_a, piece_b], pose_tree
            )

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (2, 5, 3))
        self.assertTrue(np.any(np.all(result == np.array([10, 20, 30]), axis=2)))
        self.assertTrue(np.any(np.all(result == np.array([40, 50, 60]), axis=2)))


if __name__ == "__main__":
    unittest.main()
