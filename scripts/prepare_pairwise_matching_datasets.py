import argparse
from pathlib import Path
import shutil


def main(input_dir: Path, output_dir: Path):
    assert input_dir.is_dir()
    background_filename = "background.png"
    background_path = input_dir / background_filename
    assert background_path.is_file()

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.iterdir():
        prefix_and_index = path.stem.split()
        prefix = prefix_and_index[0]
        if prefix not in ("positive", "negative"):
            continue
        index = 1 if len(prefix_and_index) == 1 else int(prefix_and_index[1]) + 1
        dataset_dir = output_dir / f"{prefix}_{index:02}"
        dataset_dir.mkdir()
        shutil.copyfile(background_path, dataset_dir / background_filename)
        puzzle_pieces_dir = dataset_dir / "puzzle_piece_images"
        puzzle_pieces_dir.mkdir()
        shutil.copyfile(path, puzzle_pieces_dir / "puzzle_pieces.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert the raw scanner output to pairwise matching dataset. The scanner output should have a background.png, positive.png, positive 01.png, ..., negative.png, negative 01.png, ..."
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, required=True, help="Input directory"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    main(Path(args.input_dir), Path(args.output_dir))
