import numpy as np
import utils.assertions as assertions
import numpy.typing as npt


class Contour:
    def __init__(self, points: npt.NDArray[int], puzzle_piece_id: int):
        assertions.assert_2d_vectors(points, int)
        assert puzzle_piece_id >= 0
        self.points = points  # (N, 2), in the order of (x, y)
        self.puzzle_piece_id = puzzle_piece_id
        self.feature_point_indices = None  # np.array, (N, ), int

        # the normalized normal vectors (N, 2)
        self.normal_vectors = None

        # Features are a list of curvatures calculated at a set of given radii
        self.features = None  # np.array, (N, M), float


def get_contours_in_puzzle_piece(
    contours: list[Contour], puzzle_piece_id: int
) -> list[Contour]:
    return [
        contour for contour in contours if contour.puzzle_piece_id == puzzle_piece_id
    ]


def get_contour_points_in_puzzle_piece(
    contours: list[Contour], puzzle_piece_id: int
) -> list[npt.NDArray[int]]:
    return [
        contour.points
        for contour in contours
        if contour.puzzle_piece_id == puzzle_piece_id
    ]


def get_contour_feature_point_positions_in_puzzle_piece(
    contours: list[Contour], puzzle_piece_id: int
) -> npt.NDArray[int]:
    return np.vstack(
        [
            contour.points[contour.feature_point_indices]
            for contour in contours
            if contour.puzzle_piece_id == puzzle_piece_id
        ]
    )


def get_contour_normal_vectors_in_puzzle_piece(
    contours: list[Contour], puzzle_piece_id: int
) -> npt.NDArray[float]:
    return np.vstack(
        [
            contour.normal_vectors
            for contour in contours
            if contour.puzzle_piece_id == puzzle_piece_id
        ]
    )


def get_contour_features_in_puzzle_piece(
    contours: list[Contour], puzzle_piece_id: int
) -> npt.NDArray[float]:
    return np.vstack(
        [
            contour.features
            for contour in contours
            if contour.puzzle_piece_id == puzzle_piece_id
        ]
    )
