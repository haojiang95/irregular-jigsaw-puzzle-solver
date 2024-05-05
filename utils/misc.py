import random
import numpy as np
from pathlib import Path
import shutil
import os
import matplotlib


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_output_dir(output_dir: Path, config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / config_path.name)
    os.system(
        f"git ls-files -z | xargs -0 tar -czvf {output_dir / ".source_code.tar.gz"} > /dev/null 2>&1"
    )


def random_color(min_brightness: float) -> tuple[int, int, int]:
    """
    Generates a random RGB color with a minimum brightness constraint.

    This function generates a random color by first creating a random HSV
    (hue, saturation, value) value, then ensuring the value (brightness)
    component meets or exceeds the specified minimum brightness. The HSV
    is then converted to an RGB value, which is returned as a tuple of
    integers in the range 0-255.

    :param min_brightness: Minimum brightness value for the color, must
        be in the range [0, 1).
    :type min_brightness: float
    :return: A tuple representing the RGB color, with each component
        being an integer in the range [0, 255].
    :rtype: tuple[int, int, int]
    """
    assert 0 <= min_brightness < 1

    hsv = np.random.rand(3)
    hsv[2] = min_brightness + (1 - min_brightness) * hsv[2]
    rgb = (matplotlib.colors.hsv_to_rgb(hsv) * 255)

    assert np.all(rgb >= 0) and np.all(rgb <= 255)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])
