import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from skimage.transform import EuclideanTransform

from data_structures.contour import (
    Contour,
    get_contour_feature_point_positions_in_puzzle_piece,
    get_contour_features_in_puzzle_piece,
    get_contour_normal_vectors_in_puzzle_piece,
    get_contour_points_in_puzzle_piece,
    get_contours_in_puzzle_piece,
)
from data_structures.contour_matching_results import (
    ContourMatchingResults,
    ContourMatchingSingleStepResults,
)
from data_structures.dataset import PuzzleDataset
from data_structures.puzzle_piece import PuzzlePiece


class TestContour(unittest.TestCase):
    def test_contour_stores_points_and_piece_id(self):
        points = np.array([[0, 0], [1, 0], [1, 1]], dtype=int)

        contour = Contour(points, puzzle_piece_id=2)

        np.testing.assert_array_equal(contour.points, points)
        self.assertEqual(contour.puzzle_piece_id, 2)
        self.assertIsNone(contour.feature_point_indices)
        self.assertIsNone(contour.normal_vectors)
        self.assertIsNone(contour.features)

    def test_contour_rejects_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            Contour(np.array([0, 1], dtype=int), puzzle_piece_id=0)
        with self.assertRaises(AssertionError):
            Contour(np.array([[0, 0]], dtype=int), puzzle_piece_id=-1)

    def test_contour_helpers_filter_and_stack_by_piece_id(self):
        contour_a = Contour(np.array([[0, 0], [1, 0], [1, 1]], dtype=int), 0)
        contour_b = Contour(np.array([[5, 5], [6, 5]], dtype=int), 1)
        contour_c = Contour(np.array([[2, 2], [3, 2]], dtype=int), 0)
        contour_a.feature_point_indices = np.array([0, 2], dtype=int)
        contour_c.feature_point_indices = np.array([1], dtype=int)
        contour_a.normal_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        contour_c.normal_vectors = np.array([[0.0, -1.0]])
        contour_a.features = np.array([[1.0, 2.0], [3.0, 4.0]])
        contour_c.features = np.array([[5.0, 6.0]])
        contours = [contour_a, contour_b, contour_c]

        self.assertEqual(
            get_contours_in_puzzle_piece(contours, 0), [contour_a, contour_c]
        )
        self.assertEqual(
            get_contour_points_in_puzzle_piece(contours, 1)[0].tolist(),
            [[5, 5], [6, 5]],
        )
        np.testing.assert_array_equal(
            get_contour_feature_point_positions_in_puzzle_piece(contours, 0),
            np.array([[0, 0], [1, 1], [3, 2]], dtype=int),
        )
        np.testing.assert_allclose(
            get_contour_normal_vectors_in_puzzle_piece(contours, 0),
            np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        )
        np.testing.assert_allclose(
            get_contour_features_in_puzzle_piece(contours, 0),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )


class TestContourMatchingResults(unittest.TestCase):
    def test_single_step_inverse_swaps_transform_and_indices(self):
        transform = EuclideanTransform(translation=(3, -2))
        source_indices = np.array([0, 2, 4], dtype=int)
        target_indices = np.array([1, 3, 5], dtype=int)
        result = ContourMatchingSingleStepResults(
            transform, source_indices, target_indices
        )

        inverse = result.inverse()

        np.testing.assert_allclose(inverse.transform.params, transform.inverse.params)
        np.testing.assert_array_equal(inverse.source_matching_indices, target_indices)
        np.testing.assert_array_equal(inverse.target_matching_indices, source_indices)

    def test_matching_results_inverse_swaps_contours_and_nested_results(self):
        init_result = ContourMatchingSingleStepResults(
            EuclideanTransform(translation=(1, 0)),
            np.array([0, 1], dtype=int),
            np.array([2, 3], dtype=int),
        )
        refined_result = ContourMatchingSingleStepResults(
            EuclideanTransform(translation=(2, 0)),
            np.array([4, 5], dtype=int),
            np.array([6, 7], dtype=int),
        )
        result = ContourMatchingResults(
            init_result,
            EuclideanTransform(translation=(3, 0)),
            refined_result,
            valid=True,
            source_contour_index=8,
            target_contour_index=9,
        )

        inverse = result.inverse()

        self.assertTrue(inverse.valid)
        self.assertEqual(inverse.source_contour_index, 9)
        self.assertEqual(inverse.target_contour_index, 8)
        np.testing.assert_allclose(
            inverse.icp_refined_transform.params,
            result.icp_refined_transform.inverse.params,
        )
        np.testing.assert_array_equal(
            inverse.init_matching_results.source_matching_indices,
            init_result.target_matching_indices,
        )
        np.testing.assert_array_equal(
            inverse.refined_matching_result.target_matching_indices,
            refined_result.source_matching_indices,
        )

    def test_single_step_rejects_empty_indices(self):
        with self.assertRaises(AssertionError):
            ContourMatchingSingleStepResults(
                EuclideanTransform(),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )


class TestPuzzleDataset(unittest.TestCase):
    def test_load_dataset_reads_background_and_piece_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            pieces_dir = dataset_root / "puzzle_piece_images"
            pieces_dir.mkdir()
            cv2.imwrite(
                str(dataset_root / "background.png"),
                np.full((4, 5, 3), 10, dtype=np.uint8),
            )
            cv2.imwrite(
                str(pieces_dir / "piece_a.png"),
                np.full((3, 4, 3), 20, dtype=np.uint8),
            )
            cv2.imwrite(
                str(pieces_dir / "piece_b.png"),
                np.full((2, 3, 3), 30, dtype=np.uint8),
            )
            dataset = PuzzleDataset(dataset_root)

            dataset.load_dataset()

            self.assertEqual(dataset.background_image.shape, (4, 5, 3))
            self.assertEqual(len(dataset.puzzle_piece_images), 2)
            self.assertTrue(
                all(image is not None for image in dataset.puzzle_piece_images)
            )

    def test_load_dataset_rejects_missing_or_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = PuzzleDataset(Path(tmp_dir))

            with self.assertRaises(AssertionError):
                dataset.load_dataset()

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            (dataset_root / "puzzle_piece_images").mkdir()
            cv2.imwrite(
                str(dataset_root / "background.png"),
                np.zeros((2, 2, 3), dtype=np.uint8),
            )
            dataset = PuzzleDataset(dataset_root)

            with self.assertRaises(AssertionError):
                dataset.load_dataset()


class TestPuzzlePiece(unittest.TestCase):
    def test_puzzle_piece_stores_inputs_and_default_input_id(self):
        patch = np.zeros((3, 4, 3), dtype=np.uint8)
        mask = np.ones((3, 4), dtype=bool)

        piece = PuzzlePiece(patch, mask, (5, 6))

        np.testing.assert_array_equal(piece.bgr_patch, patch)
        np.testing.assert_array_equal(piece.mask, mask)
        self.assertEqual(piece.upper_left_corner, (5, 6))
        self.assertEqual(piece.input_image_id, -1)

    def test_puzzle_piece_rejects_invalid_inputs(self):
        patch = np.zeros((3, 4, 3), dtype=np.uint8)
        mask = np.ones((3, 4), dtype=bool)

        with self.assertRaises(AssertionError):
            PuzzlePiece(patch[..., 0], mask, (0, 0))
        with self.assertRaises(AssertionError):
            PuzzlePiece(patch, mask.astype(np.uint8), (0, 0))
        with self.assertRaises(AssertionError):
            PuzzlePiece(patch, np.ones((4, 3), dtype=bool), (0, 0))
        with self.assertRaises(AssertionError):
            PuzzlePiece(patch, mask, (-1, 0))


if __name__ == "__main__":
    unittest.main()
