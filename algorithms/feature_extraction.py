import cv2
import numpy as np
import circle_fit
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter
import utils.assertions as assertions
import numpy.typing as npt
from utils.profiling_tools import profile
from cython_libs import find_neighbors_within_radius


@profile()
def curvature_by_circle_fitting(
    contour_points: npt.NDArray[float],
    window_radius: float,
    gaussian_smoothing_sigma: float,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Calculate signed curvature as the inverse radius of the circle fitted using local points
    :param contour_points: Sampled points on the contour (N, 2), float
    :param window_radius: Calculate curvature using the fragment inside the window
    :param gaussian_smoothing_sigma: Sigma of the Gaussian smoothing applied to curvature
    :return: curvatures: (N, ) float;
             normal_vectors: The normalized vector pointing from the point on contour to the center of the circle
    """
    assertions.assert_2d_vectors(contour_points, float)
    assert window_radius > 1
    assert gaussian_smoothing_sigma > 0

    curvatures, normal_vectors = [], []
    for current_point_index, current_point in enumerate(contour_points):
        # Find the fragment within the radius to current_point
        points = find_neighbors_within_radius(
            contour_points, current_point_index, window_radius
        )

        # Calculate curvature and normal vector
        try:
            center_x, center_y, r, _ = circle_fit.taubinSVD(points)
            normal_vector = np.array([center_x, center_y]) - current_point
            normal_length = np.linalg.norm(normal_vector)
            normal_vectors.append(
                normal_vector / normal_length if normal_length > 0 else normal_vector
            )
        except FloatingPointError:
            r = np.inf
            normal_vectors.append(np.zeros(2))
        curvatures.append(1 / r)

    curvatures = gaussian_filter1d(
        np.array(curvatures), gaussian_smoothing_sigma, mode="wrap"
    )
    normal_vectors = np.stack(normal_vectors).astype(float)

    assertions.assert_flat_array(curvatures, float)
    assertions.assert_2d_vectors(normal_vectors, float)

    return curvatures, normal_vectors


def extract_features(
    contour_points: npt.NDArray[float],
    puzzle_piece_mask: npt.NDArray[bool],
    curvature_window_radii: list[float],
    normal_vector_window_radius: float,
    curvature_gaussian_smoothing_sigma: float,
    maximum_filter_window_size: int,
    median_filter_window_size: int,
    salience_ratio: float,
    minimum_curvature: float,
    insideness_testing_stride: float,
) -> tuple[npt.NDArray[int], npt.NDArray[float], npt.NDArray[float]]:
    """
    Detect feature point on the contour and calculate the features for each.
    :param contour_points: Array of points on the contour (N, 2) float.
    :param puzzle_piece_mask: Binary mask of the puzzle piece.
    :param curvature_window_radii: Calculate curvature using the fragment inside the window.
    :param normal_vector_window_radius: Calculate normal vectors using the fragment inside the window.
    :param curvature_gaussian_smoothing_sigma: Sigma of the Gaussian smoothing applied to curvature.
    :param maximum_filter_window_size: The curvature at feature points must be the maximum inside the window.
    :param median_filter_window_size: The curvature at feature points must be salient (more than salience_ratio times of the median of a neighborhood within median_filter_window_size).
    :param salience_ratio: The curvature at feature points must be salient (more than salience_ratio times of the median of a neighborhood within median_filter_window_size)
    :param minimum_curvature: Feature shouldn't be on low curvature region, e.g., a straight line
    :param insideness_testing_stride: Test insideness by checking position + stride * normal.
    :return: The indices of the feature points (N,) int, the normalized normal vectors at all scales (M, 2), float, and the signed curvatures at all scales (M, C), float. The curvatures are signed. Positive curvature points inwards and negative ones point outwards.
    """
    assertions.assert_2d_vectors(contour_points, float)
    assertions.assert_binary_image(puzzle_piece_mask)
    assert np.min(curvature_window_radii) > 1
    assert curvature_gaussian_smoothing_sigma > 0
    assert maximum_filter_window_size % 2 == 1 and maximum_filter_window_size > 1
    assert median_filter_window_size % 2 == 1 and median_filter_window_size > 1
    assert salience_ratio > 1

    # Calculate curvature and normal vector at all radii. The curvature array is (N, C), and the normal vector array
    # is (N, 2), where N is the number of points on the contour and C is the number of radii
    curvatures = np.column_stack(
        [
            curvature_by_circle_fitting(
                contour_points, r, curvature_gaussian_smoothing_sigma
            )[0]
            for r in curvature_window_radii
        ]
    )
    _, normal_vectors = curvature_by_circle_fitting(
        contour_points, normal_vector_window_radius, 1
    )

    # A feature point must have the largest curvature in the neighborhood and has curvature much larger than the
    # median of the neighborhood
    local_maximum = maximum_filter1d(
        curvatures, maximum_filter_window_size, axis=0, mode="wrap"
    )
    local_median = median_filter(
        curvatures, median_filter_window_size, mode="wrap", axes=(0,)
    )
    feature_point_mask = (
        (curvatures >= local_maximum)
        & (curvatures > salience_ratio * local_median)
        & (curvatures > minimum_curvature)
    )
    feature_point_indices = np.nonzero(
        feature_point_mask.any(axis=1) & np.any(normal_vectors != 0, axis=1)
    )[0].astype(int)
    normal_vectors = normal_vectors[feature_point_indices]
    curvatures = curvatures[feature_point_indices]

    # inside_map is 1 on the inside, 0 on the contour, and -1 on the outside
    inside_map = puzzle_piece_mask.astype(float) * 2 - 1
    # Flip the sign of the curvature if the normals are pointing away from the puzzle piece
    inside = cv2.remap(
        inside_map,
        (
            contour_points[feature_point_indices]
            + insideness_testing_stride * normal_vectors
        ).astype(np.float32)[:, np.newaxis, :],
        np.empty(0),
        cv2.INTER_LINEAR,
    )
    assert np.all(inside != 0)
    curvatures *= np.sign(inside)

    assertions.assert_flat_array(feature_point_indices, int)
    assertions.assert_2d_vectors(normal_vectors, float)
    assertions.assert_nd_vectors(curvatures, float, len(curvature_window_radii))
    return feature_point_indices, normal_vectors, curvatures
