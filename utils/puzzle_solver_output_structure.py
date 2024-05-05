from pathlib import Path


def puzzle_piece_dir(output_dir: Path, puzzle_piece_index: int) -> Path:
    assert puzzle_piece_index >= 0
    result = output_dir / "puzzle_pieces" / f"{puzzle_piece_index:06}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def puzzle_piece_mask_visualization_path(
    output_dir: Path, puzzle_piece_index: int
) -> Path:
    assert puzzle_piece_index >= 0
    return puzzle_piece_dir(output_dir, puzzle_piece_index) / "mask_visualization.png"


def puzzle_piece_contour_visualization_path(
    output_dir: Path, puzzle_piece_index: int
) -> Path:
    assert puzzle_piece_index >= 0
    return (
        puzzle_piece_dir(output_dir, puzzle_piece_index) / "contour_visualization.png"
    )


def pairwise_matching_dir(
    output_dir: Path, puzzle_piece_index1: int, puzzle_piece_index2: int
) -> Path:
    assert 0 <= puzzle_piece_index1 != puzzle_piece_index2 >= 0
    result = (
        output_dir
        / "pairwise_matching"
        / f"{puzzle_piece_index1:04}"
        / f"{puzzle_piece_index2:04}"
    )
    result.mkdir(parents=True, exist_ok=True)
    return result


def initial_pairwise_matching_visualization_path(
    output_dir: Path, puzzle_piece_index1: int, puzzle_piece_index2: int
) -> Path:
    assert 0 <= puzzle_piece_index1 != puzzle_piece_index2 >= 0
    return (
        pairwise_matching_dir(output_dir, puzzle_piece_index1, puzzle_piece_index2)
        / "initial_matching_result.png"
    )


def icp_pairwise_matching_visualization_path(
    output_dir: Path, puzzle_piece_index1: int, puzzle_piece_index2: int
) -> Path:
    assert 0 <= puzzle_piece_index1 != puzzle_piece_index2 >= 0
    return (
        pairwise_matching_dir(output_dir, puzzle_piece_index1, puzzle_piece_index2)
        / "icp_matching_result.png"
    )


def refined_pairwise_matching_visualization_path(
    output_dir: Path, puzzle_piece_index1: int, puzzle_piece_index2: int
) -> Path:
    assert 0 <= puzzle_piece_index1 != puzzle_piece_index2 >= 0
    return (
        pairwise_matching_dir(output_dir, puzzle_piece_index1, puzzle_piece_index2)
        / "refined_matching_result.png"
    )


def contour_feature_points_visualization_path(
    output_dir: Path, puzzle_piece_index: int
) -> Path:
    assert puzzle_piece_index >= 0
    return puzzle_piece_dir(output_dir, puzzle_piece_index) / "feature_points.png"


def contour_graph_visualization_path(output_dir: Path):
    return output_dir / "contour_graph.png"


def matching_result_visualization_path(output_dir: Path):
    return output_dir / "matching_visualization.png"


def pose_tree_visualization_path(output_dir: Path):
    return output_dir / "pose_tree.png"


def profiling_data_path(output_dir: Path) -> Path:
    return output_dir / "profiling_data.txt"


def incremental_matching_output_dir(output_dir: Path, step: int) -> Path:
    result = output_dir / "incremental_matching" / f"{step:04}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def incremental_matching_puzzle_visualization_path(
    output_dir: Path, step: int, component_index: int
) -> Path:
    return (
        incremental_matching_output_dir(output_dir, step)
        / f"matching_visualization_{component_index:04}.png"
    )


def incremental_matching_pose_tree_visualization_path(
    output_dir: Path, step: int, component_index: int
) -> Path:
    return (
        incremental_matching_output_dir(output_dir, step)
        / f"pose_tree_{component_index:04}.png"
    )
