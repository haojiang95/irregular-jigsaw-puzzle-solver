import numpy as np
import numpy.typing as npt
import utils.assertions as assertions
from skimage.transform import estimate_transform, EuclideanTransform
import cv2
from data_structures.contour import Contour
import utils.geometry
from scipy import ndimage
from utils.profiling_tools import profile


def calculate_contour_distance(
    query_contour_points: npt.NDArray[float],
    target_contour_points: npt.NDArray[float],
    max_matching_contour_point_distance: float,
) -> tuple[float, float, npt.NDArray[int], float]:
    """
    Given two sets of contour points, find how well the query points are matched to the target contour points.
    :param query_contour_points: (N, 2), float.
    :param target_contour_points: (N, 2), float.
    :param max_matching_contour_point_distance: Points closer to this distance are matched points.
    :return: (1) The percentage of the query points matched to the target points.
             (2) The average distance between the matched points.
             (3) The indices of the query points which have matching target points.
             (4) The percentage of the max consecutive matched query points
    """
    assertions.assert_2d_vectors(query_contour_points, dtype=float)
    assertions.assert_2d_vectors(target_contour_points, dtype=float)
    assert max_matching_contour_point_distance > 0

    distances, _ = utils.geometry.nearest_neighbors(
        query_contour_points,
        target_contour_points,
        max_radius=max_matching_contour_point_distance,
    )
    valid_mask = np.isfinite(distances)
    indices = np.nonzero(valid_mask)[0].astype(int)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid_mask.astype(np.uint8)[np.newaxis, :]
    )
    num_pixels = stats[:, 4]
    max_consecutive_length = np.max(num_pixels[1:]) if len(stats) > 1 else 0
    first_label = labels[0, 0]
    last_label = labels[0, -1]
    if first_label != 0 and last_label != 0:
        max_consecutive_length = max(
            max_consecutive_length, num_pixels[first_label] + num_pixels[last_label]
        )

    assertions.assert_flat_array(indices, int)
    num_query_points = len(distances)
    return (
        100 * len(indices) / num_query_points,
        np.mean(distances[indices]).item(),
        indices,
        100 * max_consecutive_length / num_query_points,
    )


@profile()
def match_feature_points_pairwise(
    source_contour: Contour,
    target_contour: Contour,
    source_mask: npt.NDArray[bool],
    target_mask: npt.NDArray[bool],
    max_matching_feature_distance: float,
    max_overlap_score: float,
    max_matching_contour_point_distance: float,
    min_matching_percentage: float,
    max_num_trials: int,
) -> tuple[EuclideanTransform | None, npt.NDArray[int], npt.NDArray[int]]:
    """
    Match two puzzle pieces based on feature points.
    :return: (1) The initial transform or None one can't be found. The transform aligns the source contour to the target contour.
             (2) The indices of the source contour points that match with target contour points, or None.
             (3) The indices of the target contour points that match with source contour points, or None.
    """
    assertions.assert_binary_image(source_mask)
    assertions.assert_binary_image(target_mask)
    assert max_matching_feature_distance > 0
    assert max_num_trials > 0
    assert max_overlap_score > 0
    assert max_matching_contour_point_distance > 0
    assert 0 <= min_matching_percentage <= 100

    # Find potential feature point pairs
    with profile("match_feature_points_pairwise / find potential matching"):
        source_feature_rows = source_contour.features[:, np.newaxis, :]
        target_feature_columns = target_contour.features[np.newaxis, ...]
        feature_point_distances = np.linalg.norm(
            source_feature_rows + target_feature_columns, axis=2
        ) / np.sqrt(
            np.linalg.norm(source_feature_rows, axis=2)
            * np.linalg.norm(target_feature_columns, axis=2)
        )
        potentially_matching_feature_point_index_pairs = sorted(
            np.column_stack(
                np.nonzero(feature_point_distances <= max_matching_feature_distance)
            ),
            key=lambda pair: feature_point_distances[tuple(pair)],
        )

    # Determine the best matching pair
    min_mask_size = min(np.sum(source_mask), np.sum(target_mask))

    source_euclidean_distance_transform = ndimage.distance_transform_edt(source_mask)
    target_euclidean_distance_transform = ndimage.distance_transform_edt(target_mask)

    source_feature_points = source_contour.points[source_contour.feature_point_indices]
    target_feature_points = target_contour.points[target_contour.feature_point_indices]

    target_size_opencv = np.flip(target_mask.shape)

    source_mask_float = source_mask.astype(float)

    source_contour_points = source_contour.points.astype(float)
    target_contour_points = target_contour.points.astype(float)

    best_matching_distance = np.inf
    best_source_indices = None
    best_target_indices = None
    best_transform = None
    for source_index, target_index in potentially_matching_feature_point_index_pairs[
        :max_num_trials
    ]:
        # Estimate the transform based on the current pair
        source_position = source_feature_points[source_index]
        target_position = target_feature_points[target_index]
        transform = estimate_transform(
            "euclidean",
            np.stack(
                (
                    source_position,
                    source_position + source_contour.normal_vectors[source_index],
                )
            ),
            np.stack(
                (
                    target_position,
                    target_position + target_contour.normal_vectors[target_index],
                )
            ),
        )
        assert isinstance(transform, EuclideanTransform)

        # There must be enough overlap between transformed source contour and target contour, in the sense that the
        # matching part covers the most part of one of the contours
        with profile("match_feature_points_pairwise / ensure contour overlap"):
            transformed_source_contour_points = transform(source_contour_points)
            _, source_distance, source_indices, source_percentage = (
                calculate_contour_distance(
                    transformed_source_contour_points,
                    target_contour_points,
                    max_matching_contour_point_distance,
                )
            )
            _, target_distance, target_indices, target_percentage = (
                calculate_contour_distance(
                    target_contour_points,
                    transformed_source_contour_points,
                    max_matching_contour_point_distance,
                )
            )
            matching_distance = (source_distance + target_distance) / 2
            if (
                max(source_percentage, target_percentage) < min_matching_percentage
                or matching_distance >= best_matching_distance
            ):
                continue

        # Transformed source mask shouldn't overlap the target mask
        with profile("match_feature_points_pairwise / ensure no mask overlap"):
            transform_matrix = transform.params[:2]
            if (
                np.maximum(
                    np.sum(
                        cv2.warpAffine(
                            source_mask_float, transform_matrix, target_size_opencv
                        )
                        * target_euclidean_distance_transform
                    ),
                    np.sum(
                        cv2.warpAffine(
                            source_euclidean_distance_transform,
                            transform_matrix,
                            target_size_opencv,
                        )
                        * target_mask
                    ),
                )
                / min_mask_size
                < max_overlap_score
            ):
                best_matching_distance = matching_distance
                best_transform = transform
                best_source_indices = source_indices
                best_target_indices = target_indices

    if best_transform is not None:
        assertions.assert_2d_rigid_transform(best_transform)
        assertions.assert_flat_array(best_source_indices, int)
        assertions.assert_flat_array(best_target_indices, int)

    return best_transform, best_source_indices, best_target_indices


@profile()
def validate_pairwise_matching_result(
    transform: EuclideanTransform,
    source_contour_points: npt.NDArray[float],
    target_contour_points: npt.NDArray[float],
    source_mask: npt.NDArray[bool],
    target_mask: npt.NDArray[bool],
    min_puzzle_piece_size_in_pixels: int,
    error_allowance_in_pixels: int,
    min_matching_percentage: float,
    manufacturing_error_length_in_pixels: float,
) -> tuple[bool, npt.NDArray[int], npt.NDArray[int]]:
    """
    Validate refined transform.
    :return: (1) Whether the transform is valid.
             (2) The indices of the source contour points that match with target contour points.
             (3) The indices of the target contour points that match with source contour points.
    """
    assertions.assert_2d_rigid_transform(transform)
    assertions.assert_2d_vectors(source_contour_points, dtype=float)
    assertions.assert_2d_vectors(target_contour_points, dtype=float)
    assertions.assert_binary_image(source_mask)
    assertions.assert_binary_image(target_mask)
    assert min_puzzle_piece_size_in_pixels > 0
    assert error_allowance_in_pixels > 0
    assert 0 <= min_matching_percentage <= 100
    assert manufacturing_error_length_in_pixels > error_allowance_in_pixels

    # A large enough amount of contour should match
    transformed_source_contour_points = transform(source_contour_points)
    _, _, source_indices, source_percentage = calculate_contour_distance(
        transformed_source_contour_points,
        target_contour_points,
        manufacturing_error_length_in_pixels,
    )
    _, _, target_indices, target_percentage = calculate_contour_distance(
        target_contour_points,
        transformed_source_contour_points,
        manufacturing_error_length_in_pixels,
    )
    assertions.assert_flat_array(source_indices, int)
    assertions.assert_flat_array(target_indices, int)
    if max(source_percentage, target_percentage) < min_matching_percentage:
        return False, source_indices, target_indices

    # There shouldn't be gaps smaller than a puzzle piece between two puzzle pieces, because nothing can fit in the gap
    transformed_source_mask = cv2.warpAffine(
        source_mask.astype(np.uint8),
        transform.params[:2],
        np.flip(target_mask.shape),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~(transformed_source_mask | target_mask)).astype(np.uint8), connectivity=8
    )
    erosion_size = int(round(manufacturing_error_length_in_pixels)) + 1
    for i, stat in enumerate(stats[1:]):
        assert isinstance(stat, np.ndarray)
        start_x, start_y, width, height, num_pixels = stat
        if (
            start_x != 0
            and start_y != 0
            and start_x + width != target_mask.shape[1]
            and start_y + height != target_mask.shape[0]
            and num_pixels < min_puzzle_piece_size_in_pixels
            and np.any(
                ndimage.binary_erosion(
                    labels == i + 1, structure=np.ones((erosion_size, erosion_size))
                )
            )
        ):
            return False, source_indices, target_indices

    # The masks shouldn't overlap up to error allowance
    erosion_size = error_allowance_in_pixels + 1
    return (
        not np.any(
            ndimage.binary_erosion(
                transformed_source_mask & target_mask,
                structure=np.ones((erosion_size, erosion_size)),
            )
        ),
        source_indices,
        target_indices,
    )


@profile()
def minimize_overlap(
    source_mask: npt.NDArray[bool],
    target_mask: npt.NDArray[bool],
    init_transform: EuclideanTransform,
    max_num_iters: int,
    step_size: float,
    exit_threshold: float,
) -> EuclideanTransform:
    """
    Minimize the overlap between two matched puzzle pieces.
    :param source_mask: Mask of the source puzzle piece.
    :param target_mask: Mask of the target puzzle piece.
    :param init_transform: Initial transform.
    :param max_num_iters: Max number of iterations of the refinement step.
    :param step_size: Size of each refinement step.
    :param exit_threshold: Exit when the average overlap is below this threshold.
    :return:
    """
    assertions.assert_binary_image(source_mask)
    assertions.assert_binary_image(target_mask)
    assertions.assert_2d_rigid_transform(init_transform)
    assert max_num_iters > 0
    assert step_size > 0
    assert exit_threshold > 0

    current_transform = init_transform
    offset_map = np.flipud(
        (
            ndimage.distance_transform_edt(
                target_mask, return_distances=False, return_indices=True
            )
            - np.mgrid[: len(target_mask), : target_mask.shape[1]]
        ).astype(float)
    ).transpose((1, 2, 0))
    for _ in range(max_num_iters):
        offsets = cv2.warpAffine(
            offset_map, current_transform.inverse.params[:2], np.flip(source_mask.shape)
        )[source_mask]
        offsets = offsets[np.nonzero(np.any(offsets != 0, axis=1))]
        if offsets.size == 0:
            break
        offsets = offsets.mean(axis=0)
        if np.linalg.norm(offsets) < exit_threshold:
            break
        current_transform += EuclideanTransform(translation=offsets * step_size)

    assertions.assert_2d_rigid_transform(current_transform)
    return current_transform
