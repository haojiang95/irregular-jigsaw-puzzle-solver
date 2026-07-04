import cv2
import numpy as np
import numpy.typing as npt
from pathlib import Path
import utils.assertions as assertions
from skimage.transform import EuclideanTransform
from data_structures.puzzle_piece import PuzzlePiece
import networkx as nx
import utils.graph_utils as graph_utils
from typing import Union
import utils.misc

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
POSE_FOREST_EDGE_COLOR = "#697487"
POSE_FOREST_HIGHLIGHT_EDGE_COLOR = "#d62728"


def overlay_mask(
    bgr_image: npt.NDArray[np.uint8],
    mask: npt.NDArray[bool],
    color: tuple[int, int, int] = GREEN,
) -> npt.NDArray[np.uint8]:
    """
    Render mask on top of image
    """
    assertions.assert_color_image(bgr_image)
    assertions.assert_binary_image(mask)
    assertions.assert_same_size_image(bgr_image, mask)

    color = np.array(color, dtype=np.uint8)
    result = bgr_image.copy()
    result[mask] = result[mask] * 0.7 + color * 0.3

    assertions.assert_color_image(result)
    assertions.assert_same_size_image(result, bgr_image)
    return result


def visualize_pairwise_matching_result(
    mask1: npt.NDArray[bool],
    mask2: npt.NDArray[bool],
    transform: EuclideanTransform,
    contour_points1: Union[npt.NDArray[int], None] = None,
    contour_points2: Union[npt.NDArray[int], None] = None,
) -> npt.NDArray[np.uint8]:
    """
    The transform brings mask2 to mask1
    """
    assertions.assert_binary_image(mask1)
    assertions.assert_binary_image(mask2)
    if contour_points1 is not None:
        assertions.assert_2d_vectors(contour_points1, int)
    if contour_points2 is not None:
        assertions.assert_2d_vectors(contour_points2, int)
    assertions.assert_2d_rigid_transform(transform)

    transformation_matrix = transform.params[:2].copy()

    mask2_height, mask2_width = mask2.shape
    mask2_corners = np.array(
        [
            [0, 0, 1],
            [0, mask2_height, 1],
            [mask2_width, 0, 1],
            [mask2_width, mask2_height, 1],
        ]
    ).T
    mask2_corners = (transformation_matrix @ mask2_corners)[
        1::-1
    ]  # These are (height, width)

    mask2_upper_left_corner = np.floor(np.min(mask2_corners, axis=1)).astype(int)
    mask2_lower_right_corner = np.ceil(np.max(mask2_corners, axis=1)).astype(int)
    mask1_upper_left_corner = np.zeros(2, dtype=int)
    mask1_lower_right_corner = np.array(mask1.shape, dtype=int)
    canvas_upper_left_corner = np.minimum(
        mask1_upper_left_corner, mask2_upper_left_corner
    )
    canvas_lower_right_corner = np.maximum(
        mask1_lower_right_corner, mask2_lower_right_corner
    )
    canvas_shape = np.append(canvas_lower_right_corner - canvas_upper_left_corner, 3)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    mask1_start_indices = mask1_upper_left_corner - canvas_upper_left_corner
    mask1_end_indices = (
        mask1_start_indices + mask1_lower_right_corner - mask1_upper_left_corner
    )

    transformation_matrix[:, 2] += np.flip(mask1_start_indices)
    canvas[
        mask1_start_indices[0] : mask1_end_indices[0],
        mask1_start_indices[1] : mask1_end_indices[1],
        0,
    ] = (
        mask1 * 255
    )
    canvas[..., 1] = cv2.warpAffine(
        mask2.astype(np.uint8) * 255,
        transformation_matrix,
        canvas_shape[1::-1],
        flags=cv2.INTER_NEAREST,
    )

    # Render contours
    if contour_points1 is not None:
        canvas[
            contour_points1[:, 1] + mask1_start_indices[0],
            contour_points1[:, 0] + mask1_start_indices[1],
        ] = YELLOW
    if contour_points2 is not None:
        transformed_contour_points2 = np.round(transform(contour_points2)).astype(int)
        canvas[
            transformed_contour_points2[:, 1] + mask1_start_indices[0],
            transformed_contour_points2[:, 0] + mask1_start_indices[1],
        ] = MAGENTA

    assertions.assert_color_image(canvas)
    return canvas


def draw_contours_on_image(
    image: npt.NDArray[np.uint8], contour_points_all: list[npt.NDArray[int]]
) -> npt.NDArray[np.uint8]:
    assertions.assert_color_image(image)

    canvas = image.copy()
    for contour_points in contour_points_all:
        assertions.assert_2d_vectors(contour_points, int)
        canvas[contour_points[:, 1], contour_points[:, 0]] = RED

    assertions.assert_color_image(canvas)
    assertions.assert_same_size_image(canvas, image)
    return canvas


def draw_feature_points(
    image: npt.NDArray[np.uint8],
    feature_point_positions: npt.NDArray[int],
    normal_vectors: npt.NDArray[float],
    features: npt.NDArray[float],
) -> npt.NDArray[np.uint8]:
    assertions.assert_color_image(image)
    assertions.assert_2d_vectors(feature_point_positions, int)
    assertions.assert_2d_vectors(normal_vectors, float)
    assertions.assert_nd_vectors(features, float, features.shape[1])
    assert len(feature_point_positions) == len(normal_vectors) == len(features)

    canvas = image.copy()
    for position, normal_vector, feature in zip(
        feature_point_positions, normal_vectors, features
    ):
        cv2.circle(
            canvas, position, 5, RED if feature[0] > 0 else BLUE, thickness=cv2.FILLED
        )
        cv2.line(
            canvas,
            position,
            (position + normal_vector * 10).astype(int),
            GREEN,
            thickness=2,
        )

    assertions.assert_color_image(canvas)
    assertions.assert_same_size_image(canvas, image)
    return canvas


def visualize_matching_results(
    puzzle_pieces: list[PuzzlePiece],
    pose_tree: nx.DiGraph,
    piece_colors: dict[int, tuple[int, int, int]] | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Visualizes the matching results of assembling puzzle pieces based on a pose tree.

    This function takes a list of puzzle pieces and a directed graph that represents
    the poses (relative transformations) of the pieces relative to each other. It computes
    the absolute poses of all pieces and renders them on a canvas to provide an overview
    of their alignment and assembly.

    :param puzzle_pieces: A list of puzzle pieces, where each piece includes data
        such as its mask used for rendering on the canvas.
    :type puzzle_pieces: list[PuzzlePiece]
    :param pose_tree: A directed graph representing the hierarchical relationship
        between puzzle pieces, where edge attributes specify relative transformations.
    :type pose_tree: networkx.DiGraph
    :raises AssertionError: If the input directed graph `pose_tree` is not a tree,
        or if required transformations are missing during computation.
    :return: A color image representing the assembled puzzle, where each puzzle piece
        is displayed with a unique color.
    :rtype: npt.NDArray[np.uint8]
    """
    assert nx.is_tree(pose_tree)

    poses: list[EuclideanTransform | None] = [None] * len(puzzle_pieces)
    root = graph_utils.find_root(pose_tree)
    poses[root] = EuclideanTransform()  # Identity

    # Get the pose of each puzzle piece from the pose tree
    for edge in nx.dfs_edges(pose_tree, root):
        source_index = edge[0]
        target_index = edge[1]
        assert poses[source_index] is not None
        poses[target_index] = (
            pose_tree.edges[edge]["relative_pose"] + poses[source_index]
        )

    puzzle_piece_indices_and_poses = [
        (puzzle_piece_index, pose)
        for (puzzle_piece_index, pose) in enumerate(poses)
        if pose is not None
    ]

    # Figure out the canvas range
    min_point = np.array([np.inf, np.inf])
    max_point = np.array([-np.inf, -np.inf])
    for puzzle_piece_index, transform in puzzle_piece_indices_and_poses:
        height, width = puzzle_pieces[puzzle_piece_index].mask.shape
        transformed_corners = transform(
            [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
        )
        min_point = np.minimum(min_point, transformed_corners.min(axis=0))
        max_point = np.maximum(max_point, transformed_corners.max(axis=0))
    transform_offset = np.floor(min_point)
    canvas_size = (np.ceil(max_point) - transform_offset + 1).astype(
        int
    )  # (width, height)

    # Draw puzzle pieces
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    for puzzle_piece_index, transform in puzzle_piece_indices_and_poses:
        transform_matrix = transform.params[:2].copy()
        transform_matrix[:, 2] -= transform_offset
        color = (
            piece_colors[puzzle_piece_index]
            if piece_colors is not None and puzzle_piece_index in piece_colors
            else utils.misc.random_color(0.2)
        )
        canvas[
            cv2.warpAffine(
                puzzle_pieces[puzzle_piece_index].mask.astype(np.uint8),
                transform_matrix,
                canvas_size,
                flags=cv2.INTER_NEAREST,
            ).astype(bool)
        ] = color

    assertions.assert_color_image(canvas)
    return canvas


def stable_puzzle_piece_colors(
    num_puzzle_pieces: int,
) -> dict[int, tuple[int, int, int]]:
    assert num_puzzle_pieces > 0

    result = {}
    golden_ratio_conjugate = 0.61803398875
    for puzzle_piece_index in range(num_puzzle_pieces):
        hue = (puzzle_piece_index * golden_ratio_conjugate) % 1.0
        hsv = np.array([[[hue * 179, 155, 230]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        result[puzzle_piece_index] = (
            int(bgr[0]),
            int(bgr[1]),
            int(bgr[2]),
        )
    return result


def visualize_matching_result_forest(
    puzzle_pieces: list[PuzzlePiece],
    pose_forest: nx.DiGraph,
    include_singletons: bool = False,
) -> npt.NDArray[np.uint8]:
    assert nx.is_forest(pose_forest)

    piece_colors = stable_puzzle_piece_colors(len(puzzle_pieces))
    components = [
        tuple(sorted(component))
        for component in nx.weakly_connected_components(pose_forest)
        if include_singletons or len(component) > 1
    ]
    components.sort(key=lambda component: component[0])

    labeled_images = []
    for component in components:
        pose_tree = nx.DiGraph(pose_forest.subgraph(component).copy())
        label = f"Component: {', '.join(str(piece_id) for piece_id in component)}"
        labeled_images.append(
            (
                label,
                visualize_matching_results(puzzle_pieces, pose_tree, piece_colors),
            )
        )
    if not labeled_images:
        labeled_images.append(("No matched components", _blank_image(320, 120)))
    return _compose_labeled_images(labeled_images, max_columns=2)


def visualize_incremental_matching_change(
    puzzle_pieces: list[PuzzlePiece],
    pose_forest_before: nx.DiGraph,
    pose_forest_after: nx.DiGraph,
    source_component_before: tuple[int, ...],
    target_component_before: tuple[int, ...],
    merged_component: tuple[int, ...],
) -> npt.NDArray[np.uint8]:
    assert nx.is_forest(pose_forest_before)
    assert nx.is_forest(pose_forest_after)

    piece_colors = stable_puzzle_piece_colors(len(puzzle_pieces))
    source_tree = nx.DiGraph(
        pose_forest_before.subgraph(source_component_before).copy()
    )
    target_tree = nx.DiGraph(
        pose_forest_before.subgraph(target_component_before).copy()
    )
    merged_tree = nx.DiGraph(pose_forest_after.subgraph(merged_component).copy())

    return _compose_labeled_images(
        [
            (
                f"Before: {', '.join(str(piece_id) for piece_id in source_component_before)}",
                visualize_matching_results(puzzle_pieces, source_tree, piece_colors),
            ),
            (
                f"Before: {', '.join(str(piece_id) for piece_id in target_component_before)}",
                visualize_matching_results(puzzle_pieces, target_tree, piece_colors),
            ),
            (
                f"After: {', '.join(str(piece_id) for piece_id in merged_component)}",
                visualize_matching_results(puzzle_pieces, merged_tree, piece_colors),
            ),
        ],
        max_columns=2,
    )


def pose_forest_edge_styles(
    pose_forest: nx.Graph, highlighted_edge: tuple[int, int] | None = None
) -> tuple[list[str], list[float]]:
    edge_colors = []
    edge_widths = []
    for edge in pose_forest.edges:
        highlighted = highlighted_edge is not None and edge == highlighted_edge
        edge_colors.append(
            POSE_FOREST_HIGHLIGHT_EDGE_COLOR if highlighted else POSE_FOREST_EDGE_COLOR
        )
        edge_widths.append(3.0 if highlighted else 1.4)
    return edge_colors, edge_widths


def save_pose_forest_visualization(
    pose_forest: nx.DiGraph,
    output_path: Path,
    highlighted_edge: tuple[int, int] | None = None,
) -> None:
    assert nx.is_forest(pose_forest)
    save_graph_visualization(pose_forest, output_path, highlighted_edge)


def save_graph_visualization(
    graph: nx.Graph,
    output_path: Path,
    highlighted_edge: tuple[int, int] | None = None,
) -> None:
    result = cv2.imwrite(
        str(output_path),
        visualize_graph(graph, highlighted_edge),
    )
    assert result


def visualize_pose_forest(
    pose_forest: nx.DiGraph,
    highlighted_edge: tuple[int, int] | None = None,
) -> npt.NDArray[np.uint8]:
    assert nx.is_forest(pose_forest)
    return visualize_graph(pose_forest, highlighted_edge)


def visualize_graph(
    graph: nx.Graph,
    highlighted_edge: tuple[int, int] | None = None,
) -> npt.NDArray[np.uint8]:
    width = max(640, min(1600, graph.number_of_nodes() * 85))
    height = max(360, min(1200, graph.number_of_nodes() * 60))
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if graph.number_of_nodes() == 0:
        return canvas

    positions = nx.spring_layout(graph.to_undirected(), seed=1234)
    points = _layout_points_to_canvas_positions(positions, width, height)
    edge_colors, edge_widths = pose_forest_edge_styles(graph, highlighted_edge)

    for edge, edge_color, edge_width in zip(graph.edges, edge_colors, edge_widths):
        source, target = edge
        color = _hex_to_bgr(edge_color)
        if graph.is_directed():
            cv2.arrowedLine(
                canvas,
                points[source],
                points[target],
                color,
                thickness=int(edge_width),
                tipLength=0.08,
                line_type=cv2.LINE_AA,
            )
        else:
            cv2.line(
                canvas,
                points[source],
                points[target],
                color,
                thickness=int(edge_width),
                lineType=cv2.LINE_AA,
            )

    for node in sorted(graph.nodes):
        point = points[node]
        cv2.circle(canvas, point, 22, (251, 247, 244), thickness=cv2.FILLED)
        cv2.circle(canvas, point, 22, (85, 65, 51), thickness=2)
        label = str(node)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        text_x = point[0] - text_size[0] // 2
        text_y = point[1] + (text_size[1] - baseline) // 2
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (42, 32, 15),
            2,
            cv2.LINE_AA,
        )

    assertions.assert_color_image(canvas)
    return canvas


def _layout_points_to_canvas_positions(
    positions: dict[int, npt.NDArray[np.float64]], width: int, height: int
) -> dict[int, tuple[int, int]]:
    margin = 52
    position_values = np.array(list(positions.values()), dtype=float)
    min_position = position_values.min(axis=0)
    max_position = position_values.max(axis=0)
    span = max_position - min_position
    span[span == 0] = 1.0

    result = {}
    for node, position in positions.items():
        normalized_position = (position - min_position) / span
        x = margin + normalized_position[0] * (width - margin * 2)
        y = margin + normalized_position[1] * (height - margin * 2)
        result[node] = (int(round(x)), int(round(y)))
    return result


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    assert len(hex_color) == 7
    assert hex_color[0] == "#"
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return blue, green, red


def _blank_image(width: int, height: int) -> npt.NDArray[np.uint8]:
    assert width > 0
    assert height > 0
    return np.full((height, width, 3), 18, dtype=np.uint8)


def _labeled_panel(label: str, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    assertions.assert_color_image(image)

    padding = 14
    header_height = 32
    panel_width = max(image.shape[1] + padding * 2, 260)
    panel_height = image.shape[0] + padding * 2 + header_height
    panel = np.full((panel_height, panel_width, 3), 18, dtype=np.uint8)
    cv2.putText(
        panel,
        label,
        (padding, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 241, 245),
        1,
        cv2.LINE_AA,
    )
    image_x = (panel_width - image.shape[1]) // 2
    image_y = header_height + padding
    panel[image_y : image_y + image.shape[0], image_x : image_x + image.shape[1]] = (
        image
    )
    return panel


def _compose_labeled_images(
    labeled_images: list[tuple[str, npt.NDArray[np.uint8]]], max_columns: int
) -> npt.NDArray[np.uint8]:
    assert len(labeled_images) > 0
    assert max_columns > 0

    panels = [_labeled_panel(label, image) for label, image in labeled_images]
    gap = 16
    columns = min(max_columns, len(panels))
    rows = [panels[index : index + columns] for index in range(0, len(panels), columns)]

    row_images = []
    for row in rows:
        row_height = max(panel.shape[0] for panel in row)
        row_width = sum(panel.shape[1] for panel in row) + gap * (len(row) - 1)
        row_image = np.full((row_height, row_width, 3), 10, dtype=np.uint8)
        x_offset = 0
        for panel in row:
            y_offset = (row_height - panel.shape[0]) // 2
            row_image[
                y_offset : y_offset + panel.shape[0],
                x_offset : x_offset + panel.shape[1],
            ] = panel
            x_offset += panel.shape[1] + gap
        row_images.append(row_image)

    canvas_width = max(row.shape[1] for row in row_images)
    canvas_height = sum(row.shape[0] for row in row_images) + gap * (
        len(row_images) - 1
    )
    canvas = np.full((canvas_height, canvas_width, 3), 10, dtype=np.uint8)
    y_offset = 0
    for row_image in row_images:
        x_offset = (canvas_width - row_image.shape[1]) // 2
        canvas[
            y_offset : y_offset + row_image.shape[0],
            x_offset : x_offset + row_image.shape[1],
        ] = row_image
        y_offset += row_image.shape[0] + gap

    assertions.assert_color_image(canvas)
    return canvas
