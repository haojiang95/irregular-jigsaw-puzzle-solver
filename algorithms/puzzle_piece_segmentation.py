import numpy as np
import cv2
from data_structures.puzzle_piece import PuzzlePiece
import utils.assertions as assertions
import numpy.typing as npt


def segment_puzzle_pieces(
    background_image: npt.NDArray[np.uint8],
    puzzle_pieces_image: npt.NDArray[np.uint8],
    blurring_kernel_size: int,
    margin_size: int,
    min_puzzle_piece_size_in_pixels: int,
) -> list[PuzzlePiece]:
    """
    Segment puzzle piece instances from input image
    :param background_image: BGR image
    :param puzzle_pieces_image: BGR image
    :param blurring_kernel_size: The kernel size of the Gaussian blur before using Otsu's method
    :param margin_size: The size of margin added to the mask, so that the border is always background
    :param min_puzzle_piece_size_in_pixels:
    :return: List of puzzle pieces
    """
    assertions.assert_color_image(background_image)
    assertions.assert_color_image(puzzle_pieces_image)
    assertions.assert_same_size_image(background_image, puzzle_pieces_image)
    assert blurring_kernel_size % 2 == 1 and blurring_kernel_size > 0
    assert margin_size >= 0
    assert min_puzzle_piece_size_in_pixels > 0

    background_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    puzzle_pieces_gray = cv2.cvtColor(puzzle_pieces_image, cv2.COLOR_BGR2GRAY)

    # Segment puzzle pieces
    diff_image = np.abs(
        puzzle_pieces_gray.astype(np.int16) - background_gray.astype(np.int16)
    ).astype(np.uint16)
    diff_image = cv2.GaussianBlur(
        diff_image, (blurring_kernel_size, blurring_kernel_size), 0
    )
    segmentation = cv2.threshold(
        diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1].astype(np.uint8)

    # Close small background gaps
    _, background_patches, background_patch_stats, _ = cv2.connectedComponentsWithStats(
        ~segmentation, connectivity=8
    )
    for i, stat in enumerate(background_patch_stats[1:]):
        # stat is (start x, start y, width, height, number of pixels)
        if stat[0] != 0 and stat[4] < min_puzzle_piece_size_in_pixels:
            segmentation[background_patches == i + 1] = True

    # Label piece instances
    _, segmentation, stats, _ = cv2.connectedComponentsWithStats(
        segmentation, connectivity=8
    )
    puzzle_pieces = []
    for i, stat in enumerate(stats[1:]):
        assert isinstance(stat, np.ndarray)
        puzzle_start_x, puzzle_start_y, width, height, num_pixels_foreground = stat
        if num_pixels_foreground < min_puzzle_piece_size_in_pixels:
            continue

        # Calculate the patch location and size
        patch_start = (
            max(puzzle_start_y - margin_size, 0),
            max(puzzle_start_x - margin_size, 0),
        )
        patch_end = (
            min(puzzle_start_y + height + margin_size, segmentation.shape[0]),
            min(puzzle_start_x + width + margin_size, segmentation.shape[1]),
        )
        mask = (
            segmentation[patch_start[0] : patch_end[0], patch_start[1] : patch_end[1]]
            == i + 1
        )
        assert isinstance(mask, np.ndarray)
        patch = puzzle_pieces_image[
            patch_start[0] : patch_end[0], patch_start[1] : patch_end[1]
        ]
        puzzle_pieces.append(PuzzlePiece(patch, mask, patch_start))

    return puzzle_pieces
