import numpy.typing as npt
import utils.assertions as assertions
import numpy as np
from skimage.transform import estimate_transform, EuclideanTransform
from utils.profiling_tools import profile
import pykdtree


def nearest_neighbors(
    query_points: npt.NDArray[float],
    target_points: npt.NDArray[float],
    num_neighbors: int = 1,
    max_radius: float = np.inf,
) -> tuple[npt.NDArray[float], npt.NDArray[int]]:
    """
    Find the num_neighbors nearest neighbors within max_radius
    :return: (1) distances: (N, M) or (N, ), float, invalid entries are np.inf
             (2) indices: (N, M) or (N, ), int, invalid entries are len(target_points)
    """
    assertions.assert_nd_vectors(query_points, dtype=float)
    assertions.assert_nd_vectors(target_points, dtype=float)
    assert query_points.shape[1] == target_points.shape[1]
    assert num_neighbors > 0
    assert max_radius > 0

    distances, indices = pykdtree.kdtree.KDTree(target_points.copy()).query(
        query_points.copy(), distance_upper_bound=max_radius
    )

    if num_neighbors == 1:
        assertions.assert_flat_array(distances, dtype=float)
        assertions.assert_flat_array(indices, np.uint32)
    else:
        assertions.assert_nd_vectors(distances, dtype=float, dimension=num_neighbors)
        assertions.assert_nd_vectors(indices, dtype=np.uint32, dimension=num_neighbors)

    return distances, indices


@profile()
def icp2d(
    source_points: npt.NDArray[float],
    target_points: npt.NDArray[float],
    init_transform: EuclideanTransform,
    max_num_iters: int,
    max_matching_radius: float,
    rtol: float,
    ttol: float,
) -> EuclideanTransform:
    """
    Find the transform aligning source points to target points by ICP.
    :param source_points: (N, 2), float.
    :param target_points: (M, 2), float.
    :param init_transform: Initial transform.
    :param max_num_iters: Maximum number of iterations.
    :param max_matching_radius: The points within this radius are considered to match.
    :param rtol: Early stopping condition for rotation.
    :param ttol: Early stopping condition for translation.
    :return:
    """
    assertions.assert_2d_vectors(source_points, dtype=float)
    assertions.assert_2d_vectors(target_points, dtype=float)
    assertions.assert_2d_rigid_transform(init_transform)
    assert max_num_iters > 0
    assert max_matching_radius > 0
    assert rtol > 0
    assert ttol > 0

    curr_transform = init_transform
    for _ in range(max_num_iters):
        curr_source_points = curr_transform(source_points)
        _, indices = nearest_neighbors(
            curr_source_points, target_points, max_radius=max_matching_radius
        )
        index_validity = indices < len(target_points)
        delta_transform = estimate_transform(
            "euclidean",
            curr_source_points[index_validity],
            target_points[indices[index_validity]],
        )
        assert isinstance(delta_transform, EuclideanTransform)
        curr_transform += delta_transform
        if (
            abs(delta_transform.rotation) < rtol
            and np.linalg.norm(curr_transform.translation) < ttol
        ):
            break

    assertions.assert_2d_rigid_transform(curr_transform)
    return curr_transform
