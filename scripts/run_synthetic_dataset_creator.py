import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from applications.synthetic_dataset_creator_app import create_synthetic_dataset
import utils.misc


def main(output_dir: str, config_path: str):
    # Load config
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Prepare output directory
    output_dir = Path(output_dir)
    utils.misc.prepare_output_dir(output_dir, config_path)

    create_synthetic_dataset(output_dir, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--config_path", "-c", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    main(args.output_dir, args.config_path)
