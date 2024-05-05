import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from applications.pairwise_puzzle_solver_app import run_pairwise_puzzle_solver
from applications.puzzle_solver_app import run_puzzle_solver
import numpy as np
import utils.misc
import logging
from utils.profiling_tools import save_profiling_data
from utils.puzzle_solver_output_structure import profiling_data_path


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

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    (run_pairwise_puzzle_solver if config["pairwise"] else run_puzzle_solver)(
        input_dir, output_dir, config
    )

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
