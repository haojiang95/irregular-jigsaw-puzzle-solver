from pathlib import Path
from data_structures.dataset import PuzzleDataset
from applications.building_blocks import PuzzleSolver
from utils.profiling_tools import profile


def write_validity(output_dir: Path, valid: bool) -> None:
    assert output_dir.is_dir()
    open(output_dir / ("valid" if valid else "invalid"), "w").close()

@profile()
def run_pairwise_puzzle_solver(
    input_dir: Path, output_dir: Path, configs: dict
) -> None:
    solver = PuzzleSolver(output_dir, configs, False)

    # Step 1: Load dataset
    dataset = PuzzleDataset(input_dir)
    dataset.load_dataset()

    # Step 2: Segment the puzzle pieces
    puzzle_pieces = solver.run_puzzle_piece_segmentation(dataset)
    assert len(puzzle_pieces) == 2

    # Visualize puzzle piece segmentation result
    solver.run_puzzle_piece_mask_visualization(puzzle_pieces)

    # Step 3: Extract contours
    contours = solver.run_contour_extraction(puzzle_pieces)
    assert len(contours) == 2

    # Visualize contour extraction result
    solver.run_contour_visualization(puzzle_pieces, contours)

    # Step 4: Extract features from contours
    solver.run_feature_extraction(puzzle_pieces, contours)

    # Visualize feature points
    solver.run_feature_point_visualization(puzzle_pieces, contours)

    # Step 5: Match contours
    results = solver.run_contour_matching(0, 1, puzzle_pieces, contours)
    if results is None:
        write_validity(output_dir, False)
        return

    write_validity(output_dir, results.valid)

    # Visualize matching results
    source_contour = contours[0]
    target_contour = contours[1]
    solver.run_contour_matching_visualization(
        results, source_contour, target_contour, puzzle_pieces
    )
