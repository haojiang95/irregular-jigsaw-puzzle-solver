import unittest
import warnings

import cv2
import networkx as nx
import numpy as np
from skimage.transform import EuclideanTransform

from algorithms.contour_extraction import extract_contours_from_mask
from algorithms.feature_extraction import curvature_by_circle_fitting, extract_features
from algorithms.incremental_matching import IncrementalMatching, merge_pose_forest_nodes
from algorithms.pairwise_matching import (
    calculate_contour_distance,
    match_feature_points_pairwise,
    minimize_overlap,
    validate_pairwise_matching_result,
)
from algorithms.puzzle_piece_segmentation import segment_puzzle_pieces
from data_structures.contour import Contour
from data_structures.contour_matching_results import (
    ContourMatchingResults,
    ContourMatchingSingleStepResults,
)


class TestContourExtraction(unittest.TestCase):
    def test_extract_contours_from_mask_filters_by_length(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True

        contours = extract_contours_from_mask(mask, min_contour_length=1)
        filtered_contours = extract_contours_from_mask(mask, min_contour_length=100)

        self.assertEqual(len(contours), 1)
        self.assertEqual(contours[0].dtype, int)
        self.assertEqual(contours[0].shape[1], 2)
        self.assertEqual(filtered_contours, [])

    def test_extract_contours_from_mask_rejects_non_binary_mask(self):
        with self.assertRaises(AssertionError):
            extract_contours_from_mask(np.zeros((4, 4), dtype=np.uint8), 0)


class TestFeatureExtraction(unittest.TestCase):
    def test_curvature_by_circle_fitting_returns_curvature_and_normals(self):
        angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        center = np.array([20.0, 20.0])
        radius = 10.0
        contour_points = np.column_stack(
            [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
        ).astype(float)

        curvatures, normal_vectors = curvature_by_circle_fitting(
            contour_points,
            window_radius=3.0,
            gaussian_smoothing_sigma=1.0,
        )

        self.assertEqual(curvatures.shape, (60,))
        self.assertEqual(normal_vectors.shape, (60, 2))
        np.testing.assert_allclose(curvatures, np.full(60, 0.1), atol=1e-2)
        np.testing.assert_allclose(np.linalg.norm(normal_vectors, axis=1), 1.0)

    def test_extract_features_finds_square_corners(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[8:22, 8:22] = True
        contour_points = extract_contours_from_mask(mask, 1)[0].astype(float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            feature_indices, normal_vectors, features = extract_features(
                contour_points,
                mask,
                curvature_window_radii=[2.5, 4.0],
                normal_vector_window_radius=2.5,
                curvature_gaussian_smoothing_sigma=1.0,
                maximum_filter_window_size=3,
                median_filter_window_size=3,
                salience_ratio=1.01,
                minimum_curvature=0.01,
                insideness_testing_stride=1.0,
            )

        feature_points = contour_points[feature_indices].astype(int)
        expected_corners = np.array([[8, 8], [8, 21], [21, 21], [21, 8]])
        self.assertEqual(len(feature_indices), 4)
        self.assertEqual(normal_vectors.shape, (4, 2))
        self.assertEqual(features.shape, (4, 2))
        self.assertEqual(
            {tuple(point) for point in feature_points},
            {tuple(point) for point in expected_corners},
        )


class TestPuzzlePieceSegmentation(unittest.TestCase):
    def test_segment_puzzle_pieces_extracts_two_components(self):
        background = np.zeros((20, 30, 3), dtype=np.uint8)
        puzzle_image = background.copy()
        puzzle_image[2:6, 3:8] = (255, 255, 255)
        puzzle_image[12:17, 20:26] = (200, 200, 200)

        puzzle_pieces = segment_puzzle_pieces(
            background,
            puzzle_image,
            blurring_kernel_size=1,
            margin_size=1,
            min_puzzle_piece_size_in_pixels=4,
        )

        self.assertEqual(len(puzzle_pieces), 2)
        self.assertEqual(
            [tuple(map(int, piece.upper_left_corner)) for piece in puzzle_pieces],
            [(1, 2), (11, 19)],
        )
        self.assertEqual([int(piece.mask.sum()) for piece in puzzle_pieces], [20, 30])
        self.assertTrue(
            all(
                piece.bgr_patch.shape[:2] == piece.mask.shape for piece in puzzle_pieces
            )
        )

    def test_segment_puzzle_pieces_rejects_even_blur_kernel(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        with self.assertRaises(AssertionError):
            segment_puzzle_pieces(image, image, 2, 0, 1)


class TestPairwiseMatching(unittest.TestCase):
    def test_calculate_contour_distance_reports_matches_and_wrapped_run(self):
        query_points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]]
        )
        target_points = np.array([[0.0, 0.0], [1.0, 0.0], [12.0, 0.0]])

        percentage, average_distance, indices, max_consecutive_percentage = (
            calculate_contour_distance(
                query_points,
                target_points,
                max_matching_contour_point_distance=0.1,
            )
        )

        self.assertEqual(percentage, 60.0)
        self.assertEqual(average_distance, 0.0)
        np.testing.assert_array_equal(indices, np.array([0, 1, 4], dtype=int))
        self.assertEqual(max_consecutive_percentage, 60.0)

    def test_match_feature_points_pairwise_returns_best_transform(self):
        source_contour = Contour(
            np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=int), 0
        )
        target_contour = Contour(
            np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=int), 1
        )
        source_contour.feature_point_indices = np.array([1], dtype=int)
        target_contour.feature_point_indices = np.array([1], dtype=int)
        source_contour.normal_vectors = np.array([[0.0, 1.0]])
        target_contour.normal_vectors = np.array([[0.0, -1.0]])
        source_contour.features = np.array([[1.0, 2.0]])
        target_contour.features = np.array([[-1.0, -2.0]])
        source_mask = np.zeros((5, 5), dtype=bool)
        target_mask = np.zeros((5, 5), dtype=bool)
        source_mask[2, 0:4] = True
        target_mask[1, 0:4] = True

        transform, source_indices, target_indices = match_feature_points_pairwise(
            source_contour,
            target_contour,
            source_mask,
            target_mask,
            max_matching_feature_distance=0.1,
            max_overlap_score=100.0,
            max_matching_contour_point_distance=1.1,
            min_matching_percentage=25.0,
            max_num_trials=10,
        )

        self.assertIsNotNone(transform)
        np.testing.assert_allclose(
            transform(source_contour.points[source_contour.feature_point_indices]),
            target_contour.points[target_contour.feature_point_indices],
        )
        np.testing.assert_array_equal(source_indices, np.array([0, 1, 2, 3], dtype=int))
        np.testing.assert_array_equal(target_indices, np.array([0, 1, 2, 3], dtype=int))

    def test_validate_pairwise_matching_result_returns_valid_result(self):
        source_mask = np.zeros((10, 20), dtype=bool)
        target_mask = np.zeros((10, 20), dtype=bool)
        source_mask[2:8, 2:8] = True
        target_mask[2:8, 8:14] = True
        source_points = np.array([[7.0, float(y)] for y in range(2, 8)])
        target_points = np.array([[8.0, float(y)] for y in range(2, 8)])

        valid, source_indices, target_indices = validate_pairwise_matching_result(
            EuclideanTransform(translation=(1, 0)),
            source_points,
            target_points,
            source_mask,
            target_mask,
            min_puzzle_piece_size_in_pixels=5,
            error_allowance_in_pixels=1,
            min_matching_percentage=50.0,
            manufacturing_error_length_in_pixels=2.0,
        )

        self.assertTrue(valid)
        np.testing.assert_array_equal(source_indices, np.arange(6))
        np.testing.assert_array_equal(target_indices, np.arange(6))

    def test_validate_pairwise_matching_result_returns_invalid_for_low_match(self):
        source_points = np.array([[float(i), 0.0] for i in range(10)])
        target_points = np.array(
            [[float(i), 0.0] for i in range(4)]
            + [[float(i), 5.0] for i in range(4, 10)]
        )
        source_mask = np.zeros((20, 20), dtype=bool)
        target_mask = np.zeros((20, 20), dtype=bool)
        source_mask[2:5, 2:5] = True
        target_mask[10:13, 10:13] = True

        valid, source_indices, target_indices = validate_pairwise_matching_result(
            EuclideanTransform(),
            source_points,
            target_points,
            source_mask,
            target_mask,
            min_puzzle_piece_size_in_pixels=5,
            error_allowance_in_pixels=1,
            min_matching_percentage=60.0,
            manufacturing_error_length_in_pixels=2.0,
        )

        self.assertFalse(valid)
        self.assertGreater(len(source_indices), 0)
        self.assertGreater(len(target_indices), 0)

    def test_minimize_overlap_moves_source_mask_to_reduce_overlap(self):
        source_mask = np.zeros((10, 10), dtype=bool)
        target_mask = np.zeros((10, 10), dtype=bool)
        source_mask[3:7, 3:7] = True
        target_mask[3:7, 4:8] = True
        initial_overlap = int(np.sum(source_mask & target_mask))

        transform = minimize_overlap(
            source_mask,
            target_mask,
            EuclideanTransform(),
            max_num_iters=10,
            step_size=0.5,
            exit_threshold=0.01,
        )
        transformed_source_mask = cv2.warpAffine(
            source_mask.astype(np.uint8),
            transform.params[:2],
            np.flip(target_mask.shape),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)

        self.assertGreater(np.linalg.norm(transform.translation), 0)
        self.assertLess(
            int(np.sum(transformed_source_mask & target_mask)), initial_overlap
        )


class TestIncrementalMatching(unittest.TestCase):
    def test_merge_pose_forest_nodes_reverses_target_ancestors_and_adds_edge(self):
        pose_forest = nx.DiGraph()
        pose_forest.add_nodes_from([0, 1, 2])
        pose_forest.add_edge(2, 1, relative_pose=EuclideanTransform(translation=(5, 0)))

        merge_pose_forest_nodes(
            pose_forest,
            source_node=0,
            target_node=1,
            relative_pose=EuclideanTransform(translation=(1, 0)),
        )

        self.assertTrue(pose_forest.has_edge(0, 1))
        self.assertTrue(pose_forest.has_edge(1, 2))
        self.assertFalse(pose_forest.has_edge(2, 1))
        np.testing.assert_allclose(
            pose_forest.edges[0, 1]["relative_pose"].translation,
            np.array([1.0, 0.0]),
        )
        np.testing.assert_allclose(
            pose_forest.edges[1, 2]["relative_pose"].translation,
            np.array([-5.0, 0.0]),
        )

    def test_incremental_matching_steps_through_best_edges_until_done(self):
        contours = [
            Contour(np.array([[0, 0], [1, 0]], dtype=int), 0),
            Contour(np.array([[0, 1], [1, 1]], dtype=int), 1),
            Contour(np.array([[0, 2], [1, 2]], dtype=int), 2),
        ]
        contour_graph = nx.Graph()
        contour_graph.add_edge(
            0,
            1,
            matching_info=self._matching_results(
                source_contour_index=0,
                target_contour_index=1,
                num_matches=4,
                translation=(1, 0),
            ),
        )
        contour_graph.add_edge(
            1,
            2,
            matching_info=self._matching_results(
                source_contour_index=1,
                target_contour_index=2,
                num_matches=2,
                translation=(2, 0),
            ),
        )
        matching = IncrementalMatching(contour_graph, 3, contours)

        self.assertFalse(matching.step())
        self.assertFalse(matching.step())
        self.assertTrue(matching.step())

        pose_forest = matching.pose_forest()
        self.assertEqual(pose_forest.number_of_edges(), 2)
        self.assertTrue(nx.is_tree(pose_forest.to_undirected()))

    def test_incremental_matching_step_with_result_records_skipped_candidates(self):
        contours = [
            Contour(np.array([[0, index], [1, index]], dtype=int), index)
            for index in range(4)
        ]
        contour_graph = nx.Graph()
        contour_graph.add_edge(
            0,
            1,
            matching_info=self._matching_results(
                source_contour_index=0,
                target_contour_index=1,
                num_matches=5,
                translation=(1, 0),
            ),
        )
        contour_graph.add_edge(
            1,
            2,
            matching_info=self._matching_results(
                source_contour_index=1,
                target_contour_index=2,
                num_matches=4,
                translation=(2, 0),
            ),
        )
        contour_graph.add_edge(
            0,
            2,
            matching_info=self._matching_results(
                source_contour_index=0,
                target_contour_index=2,
                num_matches=3,
                translation=(3, 0),
            ),
        )
        contour_graph.add_edge(
            2,
            3,
            matching_info=self._matching_results(
                source_contour_index=2,
                target_contour_index=3,
                num_matches=2,
                translation=(4, 0),
            ),
        )
        matching = IncrementalMatching(contour_graph, 4, contours)

        first_result = matching.step_with_result()
        second_result = matching.step_with_result()
        third_result = matching.step_with_result()

        self.assertIsNotNone(first_result)
        self.assertIsNotNone(second_result)
        self.assertIsNotNone(third_result)
        self.assertEqual(first_result.step, 1)
        self.assertEqual(first_result.accepted_candidate.match_score, 10)
        self.assertEqual(third_result.step, 3)
        self.assertEqual(third_result.accepted_candidate.source_puzzle_piece_id, 2)
        self.assertEqual(third_result.accepted_candidate.target_puzzle_piece_id, 3)
        self.assertEqual(third_result.accepted_candidate.match_score, 4)
        self.assertEqual(len(third_result.skipped_candidates), 1)
        self.assertEqual(third_result.skipped_candidates[0].source_puzzle_piece_id, 0)
        self.assertEqual(third_result.skipped_candidates[0].target_puzzle_piece_id, 2)
        self.assertEqual(third_result.skipped_candidates[0].match_score, 6)
        self.assertEqual(third_result.components_before, ((0, 1, 2), (3,)))
        self.assertEqual(third_result.source_component_before, (0, 1, 2))
        self.assertEqual(third_result.target_component_before, (3,))
        self.assertEqual(third_result.merged_component, (0, 1, 2, 3))
        self.assertEqual(third_result.unmatched_pieces, ())
        self.assertEqual(third_result.pose_forest_edge, (3, 2))
        np.testing.assert_allclose(
            third_result.transform.translation,
            np.array([4.0, 0.0]),
        )
        self.assertIsNone(matching.step_with_result())

    def test_incremental_matching_step_preserves_boolean_contract(self):
        contours = [
            Contour(np.array([[0, 0], [1, 0]], dtype=int), 0),
            Contour(np.array([[0, 1], [1, 1]], dtype=int), 1),
        ]
        contour_graph = nx.Graph()
        contour_graph.add_edge(
            0,
            1,
            matching_info=self._matching_results(
                source_contour_index=0,
                target_contour_index=1,
                num_matches=2,
                translation=(1, 0),
            ),
        )
        matching = IncrementalMatching(contour_graph, 2, contours)

        self.assertFalse(matching.step())
        self.assertTrue(matching.step())

    @staticmethod
    def _matching_results(
        source_contour_index: int,
        target_contour_index: int,
        num_matches: int,
        translation: tuple[int, int],
    ) -> ContourMatchingResults:
        indices = np.arange(num_matches, dtype=int)
        single_step = ContourMatchingSingleStepResults(
            EuclideanTransform(translation=translation),
            indices,
            indices,
        )
        return ContourMatchingResults(
            single_step,
            EuclideanTransform(translation=translation),
            single_step,
            True,
            source_contour_index,
            target_contour_index,
        )


if __name__ == "__main__":
    unittest.main()
