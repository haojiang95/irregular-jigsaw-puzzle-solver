import cv2
import numpy as np
import numpy.typing as npt
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
    puzzle_pieces: list[PuzzlePiece], pose_tree: nx.DiGraph
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
        canvas[
            cv2.warpAffine(
                puzzle_pieces[puzzle_piece_index].mask.astype(np.uint8),
                transform_matrix,
                canvas_size,
                flags=cv2.INTER_NEAREST,
            ).astype(bool)
        ] = utils.misc.random_color(0.2)

    assertions.assert_color_image(canvas)
    return canvas
