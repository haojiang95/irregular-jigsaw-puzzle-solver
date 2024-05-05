from data_structures.dataset import PuzzleDataset
from data_structures.puzzle_piece import PuzzlePiece
from algorithms.puzzle_piece_segmentation import segment_puzzle_pieces
from tqdm import tqdm
import logging
import cv2
import visualizations
import utils.puzzle_solver_output_structure as output_structure
from pathlib import Path
from data_structures.contour import Contour
from algorithms.contour_extraction import extract_contours_from_mask
import data_structures.contour as contour_utils
from algorithms.feature_extraction import extract_features
from data_structures.contour_matching_results import (
    ContourMatchingResults,
    ContourMatchingSingleStepResults,
)
from algorithms.pairwise_matching import (
    match_feature_points_pairwise,
    validate_pairwise_matching_result,
    minimize_overlap,
)
import utils.geometry
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.profiling_tools import profile
from algorithms.incremental_matching import IncrementalMatching

logger = logging.getLogger(__name__)


class PuzzleSolver:
    def __init__(self, output_dir: Path, configs: dict, verbose: bool) -> None:
        self._output_dir: Path = output_dir
        self._configs: dict = configs
        self._algorithm_configs = configs["algorithms"]
        self._output_configs = configs["outputs"]
        self._verbose = verbose
        if verbose:
            plt.set_loglevel("info")
        else:
            logger.setLevel(logging.WARNING)
            plt.set_loglevel("warning")

    @profile()
    def run_puzzle_piece_segmentation(
        self, dataset: PuzzleDataset
    ) -> list[PuzzlePiece]:
        segmentation_configs = self._algorithm_configs["puzzle_piece_segmentation"]
        puzzle_pieces = []
        puzzle_piece_images = dataset.puzzle_piece_images
        num_puzzle_piece_images = len(puzzle_piece_images)
        for i, puzzle_piece_image in tqdm(
            enumerate(puzzle_piece_images),
            total=num_puzzle_piece_images,
            desc="Detecting puzzle pieces",
            disable=not self._verbose,
        ):
            puzzle_pieces_local = segment_puzzle_pieces(
                dataset.background_image,
                puzzle_piece_image,
                segmentation_configs["blurring_kernel_size"],
                segmentation_configs["margin_size"],
                self._algorithm_configs["min_puzzle_piece_size_in_pixels"],
            )
            for puzzle_piece in puzzle_pieces_local:
                puzzle_piece.input_image_id = i
            puzzle_pieces.extend(puzzle_pieces_local)
        logger.info(
            f"Detected {len(puzzle_pieces)} puzzle pieces from {num_puzzle_piece_images} puzzle piece images."
        )
        return puzzle_pieces

    @profile()
    def run_puzzle_piece_mask_visualization(
        self, puzzle_pieces: list[PuzzlePiece]
    ) -> None:
        if self._output_configs["puzzle_piece_mask_visualization"]:
            for i, puzzle_piece in tqdm(
                enumerate(puzzle_pieces),
                total=len(puzzle_pieces),
                desc="Visualizing puzzle piece masks",
                disable=not self._verbose,
            ):
                cv2.imwrite(
                    str(
                        output_structure.puzzle_piece_mask_visualization_path(
                            self._output_dir, i
                        )
                    ),
                    visualizations.overlay_mask(
                        puzzle_piece.bgr_patch, puzzle_piece.mask
                    ),
                )

    @profile()
    def run_contour_extraction(self, puzzle_pieces: list[PuzzlePiece]) -> list[Contour]:
        num_puzzle_pieces = len(puzzle_pieces)
        contours = []
        for i, puzzle_piece in tqdm(
            enumerate(puzzle_pieces),
            total=num_puzzle_pieces,
            desc="Extracting contours",
            disable=not self._verbose,
        ):
            contours.extend(
                [
                    Contour(contour, i)
                    for contour in extract_contours_from_mask(
                        puzzle_piece.mask, self._algorithm_configs["min_contour_length"]
                    )
                ]
            )
        num_contours = len(contours)
        logger.info(
            f"Extracted {num_contours} contours from {num_puzzle_pieces} puzzle pieces."
        )
        return contours

    @profile()
    def run_contour_visualization(
        self, puzzle_pieces: list[PuzzlePiece], contours: list[Contour]
    ) -> None:
        if self._output_configs["contour_visualization"]:
            for i, puzzle_piece in tqdm(
                enumerate(puzzle_pieces),
                total=len(puzzle_pieces),
                desc="Visualizing contours",
                disable=not self._verbose,
            ):
                cv2.imwrite(
                    str(
                        output_structure.puzzle_piece_contour_visualization_path(
                            self._output_dir, i
                        )
                    ),
                    visualizations.draw_contours_on_image(
                        puzzle_piece.bgr_patch,
                        contour_utils.get_contour_points_in_puzzle_piece(contours, i),
                    ),
                )

    @profile()
    def run_feature_extraction(
        self, puzzle_pieces: list[PuzzlePiece], contours: list[Contour]
    ) -> None:
        feature_extraction_configs = self._algorithm_configs["feature_extraction"]
        for contour in tqdm(
            contours, desc="Extracting features", disable=not self._verbose
        ):
            contour.feature_point_indices, contour.normal_vectors, contour.features = (
                extract_features(
                    contour.points.astype(float),
                    puzzle_pieces[contour.puzzle_piece_id].mask,
                    feature_extraction_configs["curvature_window_radii"],
                    feature_extraction_configs["normal_vector_window_radius"],
                    feature_extraction_configs["curvature_gaussian_smoothing_sigma"],
                    feature_extraction_configs["maximum_filter_window_size"],
                    feature_extraction_configs["median_filter_window_size"],
                    feature_extraction_configs["salience_ratio"],
                    feature_extraction_configs["minimum_curvature"],
                    feature_extraction_configs["insideness_testing_stride"],
                )
            )

    @profile()
    def run_feature_point_visualization(
        self, puzzle_pieces: list[PuzzlePiece], contours: list[Contour]
    ) -> None:
        if self._output_configs["feature_points"]:
            for i, puzzle_piece in tqdm(
                enumerate(puzzle_pieces),
                total=len(puzzle_pieces),
                desc="Visualizing feature points",
                disable=not self._verbose,
            ):
                cv2.imwrite(
                    str(
                        output_structure.contour_feature_points_visualization_path(
                            self._output_dir, i
                        )
                    ),
                    visualizations.draw_feature_points(
                        puzzle_piece.bgr_patch,
                        contour_utils.get_contour_feature_point_positions_in_puzzle_piece(
                            contours, i
                        ),
                        contour_utils.get_contour_normal_vectors_in_puzzle_piece(
                            contours, i
                        ),
                        contour_utils.get_contour_features_in_puzzle_piece(contours, i),
                    ),
                )

    @profile()
    def run_contour_matching(
        self,
        source_contour_index: int,
        target_contour_index: int,
        puzzle_pieces: list[PuzzlePiece],
        contours: list[Contour],
    ) -> ContourMatchingResults | None:
        source_contour = contours[source_contour_index]
        target_contour = contours[target_contour_index]
        source_mask_index = source_contour.puzzle_piece_id
        target_mask_index = target_contour.puzzle_piece_id
        # Skip if the two contours are from the same puzzle piece because they cannot match
        if source_mask_index == target_mask_index:
            return None

        # Step 1 Initial matching
        source_mask = puzzle_pieces[source_mask_index].mask
        target_mask = puzzle_pieces[target_mask_index].mask
        pairwise_matching_configs = self._algorithm_configs["pairwise_matching"]
        init_transform, init_source_matching_indices, init_target_matching_indices = (
            match_feature_points_pairwise(
                source_contour,
                target_contour,
                source_mask,
                target_mask,
                pairwise_matching_configs["max_matching_feature_distance"],
                pairwise_matching_configs["max_overlap_score"],
                pairwise_matching_configs["max_matching_contour_point_distance"],
                pairwise_matching_configs["min_matching_percentage"],
                pairwise_matching_configs["max_num_trials"],
            )
        )

        if init_transform is None:
            return None

        # Step 2: ICP pose refinement
        icp_refinement_configs = self._algorithm_configs["icp_pose_refinement"]
        source_contour_points_float = source_contour.points.astype(float)
        target_contour_points_float = target_contour.points.astype(float)
        icp_transform = utils.geometry.icp2d(
            source_contour_points_float,
            target_contour_points_float,
            init_transform,
            icp_refinement_configs["max_num_iters"],
            icp_refinement_configs["max_matching_radius"],
            np.deg2rad(icp_refinement_configs["rtol_in_degrees"]),
            icp_refinement_configs["ttol_in_pixels"],
        )

        # Step 3: Minimize puzzle piece overlap
        minimize_overlap_configs = self._algorithm_configs["minimize_overlap"]
        refined_transform = minimize_overlap(
            source_mask,
            target_mask,
            icp_transform,
            minimize_overlap_configs["max_num_iters"],
            minimize_overlap_configs["step_size"],
            minimize_overlap_configs["exit_threshold"],
        )

        # Step 4: Validate matching result
        validation_configs = self._algorithm_configs["matching_result_validation"]
        (
            valid,
            refined_source_matching_indices,
            refined_target_matching_indices,
        ) = validate_pairwise_matching_result(
            refined_transform,
            source_contour_points_float,
            target_contour_points_float,
            source_mask,
            target_mask,
            self._algorithm_configs["min_puzzle_piece_size_in_pixels"],
            validation_configs["error_allowance_in_pixels"],
            validation_configs["min_matching_percentage"],
            validation_configs["manufacturing_error_length_in_pixels"],
        )

        return ContourMatchingResults(
            ContourMatchingSingleStepResults(
                init_transform,
                init_source_matching_indices,
                init_target_matching_indices,
            ),
            icp_transform,
            ContourMatchingSingleStepResults(
                refined_transform,
                refined_source_matching_indices,
                refined_target_matching_indices,
            ),
            valid,
            source_contour_index,
            target_contour_index,
        )

    @profile()
    def run_contour_matching_visualization(
        self,
        contour_matching_results: ContourMatchingResults,
        source_contour: Contour,
        target_contour: Contour,
        puzzle_pieces: list[PuzzlePiece],
    ) -> None:
        source_mask_index = source_contour.puzzle_piece_id
        target_mask_index = target_contour.puzzle_piece_id
        source_mask = puzzle_pieces[source_mask_index].mask
        target_mask = puzzle_pieces[target_mask_index].mask
        if self._output_configs["pairwise_matching_results"]:
            init_matching_results = contour_matching_results.init_matching_results
            cv2.imwrite(
                str(
                    output_structure.initial_pairwise_matching_visualization_path(
                        self._output_dir, source_mask_index, target_mask_index
                    )
                ),
                visualizations.visualize_pairwise_matching_result(
                    target_mask,
                    source_mask,
                    init_matching_results.transform,
                    target_contour.points[
                        init_matching_results.target_matching_indices
                    ],
                    source_contour.points[
                        init_matching_results.source_matching_indices
                    ],
                ),
            )
            cv2.imwrite(
                str(
                    output_structure.icp_pairwise_matching_visualization_path(
                        self._output_dir, source_mask_index, target_mask_index
                    )
                ),
                visualizations.visualize_pairwise_matching_result(
                    target_mask,
                    source_mask,
                    contour_matching_results.icp_refined_transform,
                ),
            )
            refined_matching_results = contour_matching_results.refined_matching_result
            cv2.imwrite(
                str(
                    output_structure.refined_pairwise_matching_visualization_path(
                        self._output_dir, source_mask_index, target_mask_index
                    )
                ),
                visualizations.visualize_pairwise_matching_result(
                    target_mask,
                    source_mask,
                    refined_matching_results.transform,
                    target_contour.points[
                        refined_matching_results.target_matching_indices
                    ],
                    source_contour.points[
                        refined_matching_results.source_matching_indices
                    ],
                ),
            )

    @profile()
    def run_contour_graph_visualization(self, graph: nx.Graph) -> None:
        if self._output_configs["contour_graph"]:
            plt.clf()
            nx.draw(graph, with_labels=True, font_weight="bold")
            plt.savefig(
                output_structure.contour_graph_visualization_path(self._output_dir)
            )

    @profile()
    def run_matching_result_visualization(
        self, puzzle_pieces: list[PuzzlePiece], pose_forest: nx.DiGraph
    ) -> None:
        assert nx.is_forest(pose_forest)

        pose_tree = nx.DiGraph(
            pose_forest.subgraph(
                max(nx.weakly_connected_components(pose_forest), key=len)
            ).copy()
        )
        plt.clf()
        nx.draw_planar(pose_tree, with_labels=True, font_weight="bold")
        plt.savefig(output_structure.pose_tree_visualization_path(self._output_dir))
        cv2.imwrite(
            output_structure.matching_result_visualization_path(self._output_dir),
            visualizations.visualize_matching_results(puzzle_pieces, pose_tree),
        )

    @profile()
    def run_incremental_matching(
        self,
        contour_graph: nx.Graph,
        puzzle_pieces: list[PuzzlePiece],
        contours: list[Contour],
    ) -> nx.DiGraph:
        logger.info("Running incremental matching...")
        incremental_matching = IncrementalMatching(
            contour_graph, len(puzzle_pieces), contours
        )
        step = 0
        while not incremental_matching.step():
            step += 1
            if not self._output_configs["incremental_matching_intermediate_results"]:
                continue
            pose_forest = incremental_matching.pose_forest()
            for component_index, component in enumerate(
                nx.weakly_connected_components(pose_forest)
            ):
                if len(component) <= 1:
                    continue
                pose_tree = nx.DiGraph(pose_forest.subgraph(component).copy())
                cv2.imwrite(
                    str(
                        output_structure.incremental_matching_puzzle_visualization_path(
                            self._output_dir, step, component_index + 1
                        )
                    ),
                    visualizations.visualize_matching_results(puzzle_pieces, pose_tree),
                )
                plt.clf()
                nx.draw_planar(pose_tree, with_labels=True, font_weight="bold")
                plt.savefig(
                    output_structure.incremental_matching_pose_tree_visualization_path(
                        self._output_dir, step, component_index + 1
                    )
                )

        pose_forest = incremental_matching.pose_forest()
        logger.info(
            f"Incremental matching matched {len(max(nx.weakly_connected_components(pose_forest), key=len))}/{len(puzzle_pieces)} puzzle pieces"
        )
        return pose_forest
