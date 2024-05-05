from pathlib import Path
import cv2
import numpy as np
from utils import image_operations as img_ops


# y = l / 2
# y = - 4x^2 / l + l
# y = 4x^2 / l - l
# (y - l)^2 + x^2 = l^2
# l = pattern_length_in_pixels / 3
def create_synthetic_dataset(output_dir: Path, config: dict) -> None:
    pattern_length_in_pixels = config["pattern_length_in_pixels"]
    margin_length_in_pixels = config["margin_length_in_pixels"]

    # Create patterns
    pattern1 = np.zeros(
        (pattern_length_in_pixels, pattern_length_in_pixels, 3), dtype=np.uint8
    )
    pattern2 = np.zeros(
        (pattern_length_in_pixels, pattern_length_in_pixels, 3), dtype=np.uint8
    )
    y, x = (
        np.mgrid[:pattern_length_in_pixels, :pattern_length_in_pixels]
        - pattern_length_in_pixels / 2
    )
    l = pattern_length_in_pixels / 4
    pattern1[(y < -4 * x**2 / l + l) & (y > 4 * x**2 / l - l) & (y < l / 2), :] = 255
    pattern2[((y + l) ** 2 + x**2 < l**2) & (y < 4 * x**2 / l - l), :] = 255

    # Rotate patterns
    pattern1, _ = img_ops.rotate_image_keeping_all(
        pattern1, config["rotation_in_degrees"], 0, cv2.INTER_LINEAR
    )

    # Put patterns on canvas
    canvas = np.zeros(
        (
            max(pattern1.shape[0], pattern2.shape[0]) + margin_length_in_pixels * 2,
            (pattern1.shape[0] + pattern2.shape[0]) + margin_length_in_pixels * 3,
            3,
        ),
        dtype=np.uint8,
    )

    cv2.imwrite(str(output_dir / "background.png"), canvas)

    canvas[
        margin_length_in_pixels : margin_length_in_pixels + pattern1.shape[0],
        margin_length_in_pixels : margin_length_in_pixels + pattern1.shape[0],
    ] = pattern1
    canvas[
        margin_length_in_pixels : margin_length_in_pixels + pattern2.shape[0],
        margin_length_in_pixels * 2 + pattern1.shape[0] : -margin_length_in_pixels,
    ] = pattern2

    puzzle_piece_dir = output_dir / "puzzle_piece_images"
    puzzle_piece_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(puzzle_piece_dir / "puzzle_pieces.png"), canvas)
