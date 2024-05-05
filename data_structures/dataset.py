from pathlib import Path
import cv2


class PuzzleDataset:
    """
    Dataset structure
    dataset_dir:
    - puzzle_pieces: A directory containing puzzle piece images
    - background.png: Background image
    """

    def __init__(self, dataset_root: Path):
        self.background_image_path = dataset_root / "background.png"
        self.puzzle_piece_images_dir = dataset_root / "puzzle_piece_images"
        self.background_image, self.puzzle_piece_images = None, None

    def load_dataset(self) -> None:
        assert (
            self.background_image_path.is_file()
            and self.puzzle_piece_images_dir.is_dir()
        )
        self.background_image = cv2.imread(str(self.background_image_path))
        self.puzzle_piece_images = [
            cv2.imread(str(p)) for p in self.puzzle_piece_images_dir.iterdir()
        ]
        assert len(self.puzzle_piece_images) > 0
