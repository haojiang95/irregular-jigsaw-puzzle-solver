import numpy as np
import cv2
import utils.assertions as assertions
import numpy.typing as npt


def extract_contours_from_mask(
    mask: npt.NDArray[bool], min_contour_length: int
) -> list[npt.NDArray[int]]:
    """
    Extract contour from a binary mask image
    :param mask: Binary mask image
    :param min_contour_length: Contour shorter than the threshold will be discarded
    :return: A list of contours. Each contour is of shape (N, 2), with dtype int32
    """
    assertions.assert_binary_image(mask)
    assert min_contour_length >= 0

    # findContours may find duplicate point in the contours of, e.g., a straight line.
    # Dilating the mask before contour extraction may help
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        contour.squeeze().astype(int)
        for contour in contours
        if len(contour) >= min_contour_length
    ]

    for contour in contours:
        assertions.assert_2d_vectors(contour, int)
    return contours
