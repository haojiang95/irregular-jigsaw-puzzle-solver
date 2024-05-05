import numpy as np
import utils.assertions as assertions
import numpy.typing as npt


class PuzzlePiece:
    def __init__(
        self,
        bgr_patch: npt.NDArray[np.uint8],
        mask: npt.NDArray[bool],
        upper_left_corner: tuple[int, int],
    ):
        assertions.assert_color_image(bgr_patch)
        assertions.assert_binary_image(mask)
        assertions.assert_same_size_image(bgr_patch, mask)
        assert upper_left_corner[0] >= 0 and upper_left_corner[1] >= 0

        self.bgr_patch: npt.NDArray[np.uint8] = bgr_patch
        self.mask: npt.NDArray[bool] = mask
        self.upper_left_corner: tuple[int, int] = upper_left_corner  # (y, x)
        # The id of the input image containing the current puzzle piece. -1 means unset.
        self.input_image_id: int = -1
