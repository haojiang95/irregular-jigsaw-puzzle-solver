from pathlib import Path
import numpy as np
from data_structures.dataset import PuzzleDataset
from tqdm.contrib import tzip
from applications.building_blocks import PuzzleSolver
import networkx as nx
from utils.profiling_tools import profile


@profile()
def run_puzzle_solver(input_dir: Path, output_dir: Path, configs: dict) -> None:
    solver = PuzzleSolver(output_dir, configs, True)

    # Step 1: Load dataset
    dataset = PuzzleDataset(input_dir)
    dataset.load_dataset()

    # Step 2: Segment the puzzle pieces
    puzzle_pieces = solver.run_puzzle_piece_segmentation(dataset)

    # Visualize puzzle piece segmentation result
    solver.run_puzzle_piece_mask_visualization(puzzle_pieces)

    # Step 3: Extract contours
    contours = solver.run_contour_extraction(puzzle_pieces)

    # Visualize contour extraction result
    solver.run_contour_visualization(puzzle_pieces, contours)

    # Step 4: Extract features from contours
    solver.run_feature_extraction(puzzle_pieces, contours)

    # Visualize feature points
    solver.run_feature_point_visualization(puzzle_pieces, contours)

    # Step 5: Pairwise matching
    # Puzzle piece graph is a graph where each node represents a puzzle piece and each edge represents a match.
    # A node carries the puzzle piece index as its label. An edge carries a ContourMatchingResults as data.
    # The transformation carried by the edge pointing from node A to node B transforms A to B.
    contour_graph = nx.Graph()
    contour_graph.add_nodes_from(range(len(puzzle_pieces)))
    for source_contour_index, target_contour_index in tzip(
        *np.triu_indices(len(contours), k=1), desc="Running pairwise matching"
    ):
        results = solver.run_contour_matching(
            source_contour_index, target_contour_index, puzzle_pieces, contours
        )

        if results is None:
            continue

        # Visualize matching results
        source_contour = contours[source_contour_index]
        target_contour = contours[target_contour_index]
        solver.run_contour_matching_visualization(
            results, source_contour, target_contour, puzzle_pieces
        )

        if results.valid:
            source_puzzle_piece_id = source_contour.puzzle_piece_id
            target_puzzle_piece_id = target_contour.puzzle_piece_id
            assert not contour_graph.has_edge(
                source_puzzle_piece_id, target_puzzle_piece_id
            )
            contour_graph.add_edge(
                source_puzzle_piece_id, target_puzzle_piece_id, matching_info=results
            )

    # Visualize puzzle piece graph
    solver.run_contour_graph_visualization(contour_graph)

    # Step 6: Incremental matching
    pose_forest = solver.run_incremental_matching(
        contour_graph, puzzle_pieces, contours
    )
    solver.run_matching_result_visualization(puzzle_pieces, pose_forest)
