import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import networkx as nx
import numpy as np
from skimage.transform import EuclideanTransform

import utils.assertions as assertions
import utils.geometry as geometry
import utils.graph_utils as graph_utils
import utils.image_operations as image_operations
import utils.misc as misc
import utils.profiling_tools as profiling_tools
import utils.incremental_matching_debug as incremental_matching_debug
import utils.puzzle_solver_output_structure as output_structure


class TestAssertions(unittest.TestCase):
    def test_image_assertions_accept_valid_inputs(self):
        assertions.assert_binary_image(np.ones((2, 3), dtype=bool))
        assertions.assert_color_image(np.zeros((2, 3, 3), dtype=np.uint8))
        assertions.assert_same_size_image(
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.zeros((2, 3), dtype=bool),
        )

    def test_image_assertions_reject_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            assertions.assert_binary_image(np.ones((2, 3), dtype=np.uint8))
        with self.assertRaises(AssertionError):
            assertions.assert_color_image(np.zeros((2, 3), dtype=np.uint8))
        with self.assertRaises(AssertionError):
            assertions.assert_same_size_image(
                np.zeros((2, 3, 3), dtype=np.uint8),
                np.zeros((3, 2), dtype=bool),
            )

    def test_vector_and_transform_assertions(self):
        assertions.assert_nd_vectors(np.zeros((3, 4), dtype=float), float, 4)
        assertions.assert_2d_vectors(np.zeros((3, 2), dtype=int), int)
        assertions.assert_flat_array(np.zeros(3, dtype=int), int)
        assertions.assert_2d_transformation_matrix(np.zeros((2, 3), dtype=float))
        assertions.assert_2d_rigid_transform(EuclideanTransform())

        with self.assertRaises(AssertionError):
            assertions.assert_nd_vectors(np.zeros(3, dtype=float), float)
        with self.assertRaises(AssertionError):
            assertions.assert_2d_vectors(np.zeros((3, 3), dtype=float), float)
        with self.assertRaises(AssertionError):
            assertions.assert_flat_array(np.zeros((1, 3), dtype=int), int)
        with self.assertRaises(AssertionError):
            assertions.assert_2d_transformation_matrix(np.zeros((3, 3), dtype=float))


class TestGeometry(unittest.TestCase):
    def test_nearest_neighbors_respects_radius(self):
        query_points = np.array([[0.0, 0.0], [5.0, 0.0]])
        target_points = np.array([[1.0, 0.0], [10.0, 0.0]])

        distances, indices = geometry.nearest_neighbors(
            query_points, target_points, max_radius=2.0
        )

        np.testing.assert_allclose(distances, np.array([1.0, np.inf]))
        np.testing.assert_array_equal(indices, np.array([0, len(target_points)]))

    def test_nearest_neighbors_multiple_results(self):
        query_points = np.array([[0.0, 0.0], [5.0, 0.0]])
        target_points = np.array([[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])

        distances, indices = geometry.nearest_neighbors(
            query_points, target_points, num_neighbors=2, max_radius=3.1
        )

        self.assertEqual(distances.shape, (2, 2))
        self.assertEqual(indices.shape, (2, 2))
        np.testing.assert_allclose(distances[0], np.array([1.0, 2.0]))

    def test_icp2d_recovers_small_translation(self):
        source_points = np.array(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 3.0]]
        )
        target_points = source_points + np.array([0.5, 0.2])

        transform = geometry.icp2d(
            source_points,
            target_points,
            EuclideanTransform(),
            max_num_iters=20,
            max_matching_radius=1.0,
            rtol=1e-9,
            ttol=1e-9,
        )

        np.testing.assert_allclose(transform.translation, np.array([0.5, 0.2]))
        self.assertAlmostEqual(transform.rotation, 0.0)


class TestImageOperations(unittest.TestCase):
    def test_rotate_image_keeping_all_preserves_content_and_returns_matrix(self):
        image = np.full((3, 5), 255, dtype=np.uint8)

        rotated, rotation_matrix = image_operations.rotate_image_keeping_all(
            image,
            angle_in_degrees=45,
            fill_value=0,
            interpolation=cv2.INTER_NEAREST,
        )

        self.assertEqual(rotated.dtype, image.dtype)
        self.assertGreater(rotated.shape[0], image.shape[0])
        self.assertGreater(rotated.shape[1], image.shape[0])
        self.assertEqual(rotation_matrix.shape, (2, 3))
        self.assertGreater(rotated.max(), 0)


class TestGraphUtils(unittest.TestCase):
    def test_find_root_returns_node_without_parent(self):
        tree = nx.DiGraph()
        tree.add_edges_from([(0, 1), (1, 2)])

        self.assertEqual(graph_utils.find_root(tree), 0)

    def test_find_root_rejects_non_tree(self):
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 0)])

        with self.assertRaises(AssertionError):
            graph_utils.find_root(graph)


class TestMisc(unittest.TestCase):
    def test_set_random_seeds_makes_random_generators_repeatable(self):
        misc.set_random_seeds(123)
        first_random = random.random()
        first_numpy = np.random.rand()

        misc.set_random_seeds(123)

        self.assertEqual(random.random(), first_random)
        self.assertEqual(np.random.rand(), first_numpy)

    def test_prepare_output_dir_copies_config_and_invokes_source_archive_command(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text("seed: 1\n")
            output_dir = tmp_path / "output"

            with mock.patch("utils.misc.os.system", return_value=0) as os_system:
                misc.prepare_output_dir(output_dir, config_path)

            self.assertEqual((output_dir / "config.yaml").read_text(), "seed: 1\n")
            os_system.assert_called_once()
            self.assertIn(".source_code.tar.gz", os_system.call_args.args[0])

    def test_random_color_respects_channel_range(self):
        misc.set_random_seeds(5)

        color = misc.random_color(0.4)

        self.assertEqual(len(color), 3)
        self.assertTrue(all(isinstance(channel, int) for channel in color))
        self.assertTrue(all(0 <= channel <= 255 for channel in color))

    def test_random_color_rejects_invalid_brightness(self):
        with self.assertRaises(AssertionError):
            misc.random_color(1.0)


class TestProfilingTools(unittest.TestCase):
    def setUp(self):
        profiling_tools.profiling_data.clear()

    def tearDown(self):
        profiling_tools.profiling_data.clear()

    def test_profile_context_records_elapsed_time_and_count(self):
        with mock.patch(
            "utils.profiling_tools.time.perf_counter", side_effect=[10.0, 10.25]
        ):
            with profiling_tools.profile("block"):
                pass

        self.assertEqual(profiling_tools.profiling_data["block"][1], 1)
        self.assertAlmostEqual(profiling_tools.profiling_data["block"][0], 0.25)

    def test_profile_decorator_uses_function_name_by_default(self):
        @profiling_tools.profile()
        def decorated_function():
            return "result"

        with mock.patch(
            "utils.profiling_tools.time.perf_counter", side_effect=[1.0, 1.5]
        ):
            result = decorated_function()

        self.assertEqual(result, "result")
        self.assertEqual(profiling_tools.profiling_data["decorated_function"][1], 1)
        self.assertAlmostEqual(
            profiling_tools.profiling_data["decorated_function"][0], 0.5
        )

    def test_save_profiling_data_writes_table(self):
        profiling_tools.profiling_data["slow"] = [1.0, 2]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "profiling.txt"
            profiling_tools.save_profiling_data(path)

            text = path.read_text()

        self.assertIn("slow", text)
        self.assertIn("Average Time", text)


class TestPuzzleSolverOutputStructure(unittest.TestCase):
    def test_puzzle_piece_paths_are_created_and_formatted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            piece_dir = output_structure.puzzle_piece_dir(output_dir, 7)

            self.assertEqual(piece_dir, output_dir / "puzzle_pieces" / "000007")
            self.assertTrue(piece_dir.is_dir())
            self.assertEqual(
                output_structure.puzzle_piece_mask_visualization_path(output_dir, 7),
                piece_dir / "mask_visualization.png",
            )
            self.assertEqual(
                output_structure.puzzle_piece_contour_visualization_path(output_dir, 7),
                piece_dir / "contour_visualization.png",
            )
            self.assertEqual(
                output_structure.contour_feature_points_visualization_path(
                    output_dir, 7
                ),
                piece_dir / "feature_points.png",
            )

    def test_pairwise_matching_paths_are_created_and_formatted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            matching_dir = output_structure.pairwise_matching_dir(output_dir, 3, 12)

            self.assertEqual(
                matching_dir, output_dir / "pairwise_matching" / "0003" / "0012"
            )
            self.assertTrue(matching_dir.is_dir())
            self.assertEqual(
                output_structure.initial_pairwise_matching_visualization_path(
                    output_dir, 3, 12
                ),
                matching_dir / "initial_matching_result.png",
            )
            self.assertEqual(
                output_structure.icp_pairwise_matching_visualization_path(
                    output_dir, 3, 12
                ),
                matching_dir / "icp_matching_result.png",
            )
            self.assertEqual(
                output_structure.refined_pairwise_matching_visualization_path(
                    output_dir, 3, 12
                ),
                matching_dir / "refined_matching_result.png",
            )

    def test_top_level_and_incremental_paths_are_formatted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            self.assertEqual(
                output_structure.contour_graph_visualization_path(output_dir),
                output_dir / "contour_graph.png",
            )
            self.assertEqual(
                output_structure.matching_result_visualization_path(output_dir),
                output_dir / "matching_visualization.png",
            )
            self.assertEqual(
                output_structure.pose_tree_visualization_path(output_dir),
                output_dir / "pose_tree.png",
            )
            self.assertEqual(
                output_structure.profiling_data_path(output_dir),
                output_dir / "profiling_data.txt",
            )
            incremental_dir = output_structure.incremental_matching_output_dir(
                output_dir, 2
            )
            self.assertEqual(
                incremental_dir, output_dir / "incremental_matching" / "0002"
            )
            self.assertTrue(incremental_dir.is_dir())
            self.assertEqual(
                output_structure.incremental_matching_dir(output_dir),
                output_dir / "incremental_matching",
            )
            self.assertEqual(
                output_structure.incremental_matching_manifest_path(output_dir),
                output_dir / "incremental_matching" / "manifest.json",
            )
            self.assertEqual(
                output_structure.incremental_matching_viewer_path(output_dir),
                output_dir / "incremental_matching" / "index.html",
            )
            self.assertEqual(
                output_structure.incremental_matching_step_matching_visualization_path(
                    output_dir, 2
                ),
                incremental_dir / "matching_visualization.png",
            )
            self.assertEqual(
                output_structure.incremental_matching_step_pose_forest_visualization_path(
                    output_dir, 2
                ),
                incremental_dir / "pose_forest.png",
            )
            self.assertEqual(
                output_structure.incremental_matching_step_change_visualization_path(
                    output_dir, 2
                ),
                incremental_dir / "change_visualization.png",
            )
            self.assertEqual(
                output_structure.incremental_matching_puzzle_visualization_path(
                    output_dir, 2, 4
                ),
                incremental_dir / "matching_visualization_0004.png",
            )
            self.assertEqual(
                output_structure.incremental_matching_pose_tree_visualization_path(
                    output_dir, 2, 4
                ),
                incremental_dir / "pose_tree_0004.png",
            )

    def test_negative_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            with self.assertRaises(AssertionError):
                output_structure.puzzle_piece_dir(output_dir, -1)
            with self.assertRaises(AssertionError):
                output_structure.pairwise_matching_dir(output_dir, 1, 1)


class TestIncrementalMatchingDebugViewer(unittest.TestCase):
    def test_write_debug_viewer_writes_manifest_and_embedded_html(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            manifest = incremental_matching_debug.build_manifest(
                2,
                [
                    {
                        "step": 1,
                        "accepted": {
                            "source_piece": 0,
                            "target_piece": 1,
                            "match_score": 4,
                        },
                        "skipped_candidate_count": 0,
                        "skipped_candidates": [],
                        "pose_forest_edge": [1, 0],
                        "transform": {
                            "translation": [1.0, 0.0],
                            "rotation_degrees": 0.0,
                        },
                        "source_component_before": [0],
                        "target_component_before": [1],
                        "merged_component": [0, 1],
                        "unmatched_pieces": [],
                        "assets": {
                            "matching_visualization": "0001/matching_visualization.png",
                            "pose_forest": "0001/pose_forest.png",
                            "change_visualization": "0001/change_visualization.png",
                        },
                    }
                ],
            )

            incremental_matching_debug.write_debug_viewer(output_dir, manifest)

            manifest_path = output_structure.incremental_matching_manifest_path(
                output_dir
            )
            viewer_path = output_structure.incremental_matching_viewer_path(output_dir)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(viewer_path.is_file())
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            viewer_html = viewer_path.read_text()
            self.assertIn("manifest-data", viewer_html)
            self.assertIn("[hidden]", viewer_html)
            self.assertIn("display: none !important", viewer_html)
            self.assertIn("0001/matching_visualization.png", viewer_html)
            self.assertNotIn("fetch(", viewer_html)


if __name__ == "__main__":
    unittest.main()
