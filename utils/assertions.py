import numpy as np
import numpy.typing as npt
from skimage.transform import EuclideanTransform


def assert_binary_image(image: npt.NDArray[bool]) -> None:
    assert image.ndim == 2
    assert image.dtype == bool
    assert image.size > 0


def assert_color_image(image: npt.NDArray[np.uint8]) -> None:
    assert image.ndim == 3
    assert image.dtype == np.uint8
    assert image.size > 0
    assert image.shape[2] == 3


def assert_same_size_image(image1: npt.NDArray, image2: npt.NDArray) -> None:
    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]


def assert_nd_vectors(
    vectors: npt.NDArray, dtype=None, dimension: int | None = None
) -> None:
    assert vectors.ndim == 2
    assert vectors.size > 0
    if dimension is not None:
        assert dimension > 0
        assert vectors.shape[1] == dimension
    if dtype is not None:
        assert vectors.dtype == dtype


def assert_2d_vectors(vectors: npt.NDArray, dtype=None) -> None:
    assert_nd_vectors(vectors, dtype, 2)


def assert_flat_array(array: npt.NDArray, dtype=None) -> None:
    assert array.ndim == 1
    assert array.size > 0
    if dtype is not None:
        assert array.dtype == dtype


def assert_2d_transformation_matrix(matrix: npt.NDArray[float]) -> None:
    assert matrix.shape == (2, 3)
    assert matrix.dtype == float


def assert_2d_rigid_transform(transform: EuclideanTransform):
    assert isinstance(transform, EuclideanTransform)
    assert transform.dimensionality == 2
