import numpy as np
import cv2
import utils.assertions as assertions
import numpy.typing as npt


def rotate_image_keeping_all(
    image: npt.NDArray, angle_in_degrees: float, fill_value, interpolation: int
) -> tuple[npt.NDArray, npt.NDArray[float]]:
    """
    Rotate an image counterclockwise by a given angle in rad. Resize and shift the image so that the entire image
    is still visible after rotation. The area not occupied by the original image is filled with a given value.
    :param image: Input image
    :param angle_in_degrees:
    :param fill_value: Scalar or tuple
    :param interpolation: One of the OpenCV interpolation flags, e.g., cv2.INTER_LINEAR
    :return: Rotated image and the rotation matrix
    """
    assert image.ndim in (2, 3)
    height, weight = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle_in_degrees, 1)
    image_corners = np.array(
        [[-1, -1], [-1, height + 1], [weight + 1, height + 1], [weight + 1, -1]],
        dtype=float,
    ).T
    rotated_image_corners = (
        rotation_matrix[:, :2] @ image_corners + rotation_matrix[:, 2, np.newaxis]
    )
    rotation_matrix[:, 2] -= np.min(rotated_image_corners, axis=1)
    new_size = np.ceil(np.ptp(rotated_image_corners, axis=1)).astype(
        int
    )  # in the order of x, y
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, new_size, flags=interpolation, borderValue=fill_value
    )

    assert rotated_image.dtype == image.dtype
    assertions.assert_2d_transformation_matrix(rotation_matrix)
    return rotated_image, rotation_matrix
