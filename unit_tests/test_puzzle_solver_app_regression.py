import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))

import networkx as nx
import numpy as np
import yaml

import utils.misc
from applications.puzzle_solver_app import run_puzzle_solver


class TestPuzzleSolverAppRegression(unittest.TestCase):
    EXPECTED_CONTOUR_GRAPH_EDGES = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 5),
        (0, 6),
        (1, 6),
        (2, 8),
        (3, 6),
        (4, 6),
        (5, 8),
        (6, 7),
    ]
    EXPECTED_POSE_FOREST_EDGES = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 5),
        (5, 8),
        (6, 0),
        (6, 4),
        (7, 6),
    ]

    def test_demo_dataset_solver_structure_matches_baseline(self):
        config = self._regression_config()
        captures = {}

        utils.misc.set_random_seeds(1234)
        old_error_settings = np.seterr(divide="raise")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with (
                    mock.patch(
                        "applications.building_blocks.tqdm",
                        side_effect=self._passthrough_progress,
                    ),
                    mock.patch(
                        "applications.puzzle_solver_app.tzip",
                        side_effect=self._passthrough_zip,
                    ),
                    mock.patch(
                        "applications.building_blocks.PuzzleSolver.run_contour_graph_visualization",
                        side_effect=lambda solver, graph: self._capture_contour_graph(
                            captures, graph
                        ),
                        autospec=True,
                    ),
                    mock.patch(
                        "applications.building_blocks.PuzzleSolver.run_matching_result_visualization",
                        side_effect=lambda solver, puzzle_pieces, pose_forest: self._capture_pose_forest(
                            captures, puzzle_pieces, pose_forest
                        ),
                        autospec=True,
                    ),
                ):
                    run_puzzle_solver(Path("demo_dataset"), Path(tmp_dir), config)
        finally:
            np.seterr(**old_error_settings)

        self.assertEqual(captures["num_pieces"], 9)
        self.assertEqual(captures["contour_graph_nodes"], 9)
        self.assertEqual(captures["contour_graph_edges"], 11)
        self.assertEqual(captures["contour_graph_components"], [9])
        self.assertEqual(
            captures["contour_graph_edge_pairs"], self.EXPECTED_CONTOUR_GRAPH_EDGES
        )
        self.assertEqual(captures["pose_forest_nodes"], 9)
        self.assertEqual(captures["pose_forest_edges"], 8)
        self.assertEqual(captures["pose_forest_components"], [9])
        self.assertTrue(captures["pose_forest_is_forest"])
        self.assertEqual(
            captures["pose_forest_edge_pairs"], self.EXPECTED_POSE_FOREST_EDGES
        )

    @staticmethod
    def _regression_config() -> dict:
        with open("configs/puzzle_solver_config.yaml") as file:
            config = yaml.safe_load(file)
        config = copy.deepcopy(config)
        config["outputs"] = {output_name: False for output_name in config["outputs"]}
        return config

    @staticmethod
    def _passthrough_progress(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    @staticmethod
    def _passthrough_zip(*iterables, **kwargs):
        return zip(*iterables)

    @staticmethod
    def _capture_contour_graph(captures: dict, graph: nx.Graph) -> None:
        captures["contour_graph_nodes"] = graph.number_of_nodes()
        captures["contour_graph_edges"] = graph.number_of_edges()
        captures["contour_graph_components"] = sorted(
            len(component) for component in nx.connected_components(graph)
        )
        captures["contour_graph_edge_pairs"] = sorted(
            tuple(sorted(edge)) for edge in graph.edges()
        )

    @staticmethod
    def _capture_pose_forest(
        captures: dict, puzzle_pieces: list, pose_forest: nx.DiGraph
    ) -> None:
        captures["num_pieces"] = len(puzzle_pieces)
        captures["pose_forest_nodes"] = pose_forest.number_of_nodes()
        captures["pose_forest_edges"] = pose_forest.number_of_edges()
        captures["pose_forest_components"] = sorted(
            len(component) for component in nx.weakly_connected_components(pose_forest)
        )
        captures["pose_forest_is_forest"] = nx.is_forest(pose_forest)
        captures["pose_forest_edge_pairs"] = sorted(pose_forest.edges())


if __name__ == "__main__":
    unittest.main()
