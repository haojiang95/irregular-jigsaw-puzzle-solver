import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from applications.pairwise_puzzle_solver_app import run_pairwise_puzzle_solver
import numpy as np
import utils.misc
from tqdm import tqdm
from utils.profiling_tools import save_profiling_data
from utils.puzzle_solver_output_structure import profiling_data_path
import logging

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)


def main(input_dir: str, output_dir: str, config_path: str):
    utils.misc.set_random_seeds(1234)

    # Load config
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Check the validity of the input directory
    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f"{input_dir} is not a directory"

    # Prepare output directory
    output_dir = Path(output_dir)
    utils.misc.prepare_output_dir(output_dir, config_path)

    # Make numpy raise an exception when divide by zero, so that we can catch it
    np.seterr(divide="raise")

    failed_tests = []
    subdirectories = list(input_dir.iterdir())
    for input_subdir in tqdm(subdirectories):
        dataset_name = input_subdir.name
        output_subdir = output_dir / dataset_name
        output_subdir.mkdir(exist_ok=True)
        run_pairwise_puzzle_solver(input_subdir, output_subdir, config)
        valid = (output_subdir / "valid").is_file()
        invalid = (output_subdir / "invalid").is_file()
        assert valid ^ invalid
        if (dataset_name.startswith("positive") and invalid) or (
            dataset_name.startswith("negative") and valid
        ):
            failed_tests.append(dataset_name)
    failed_tests.sort()
    with open(output_dir / "failed_tests.txt", "w") as file:
        for test_name in failed_tests:
            file.write(f"{test_name}\n")
    logger.setLevel(logging.INFO)
    logger.info(f"{failed_tests} failed" if failed_tests else "All tests passed!")

    save_profiling_data(profiling_data_path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", "-i", type=str, required=True, help="Input directory"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--config_path", "-c", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.config_path)
