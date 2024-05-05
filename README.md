# Irregular Jigsaw Puzzle Solver

An application to solve irregular jigsaw puzzles.

## Getting Started

### Dependencies

* Git lfs is needed to pull the sample data.
* Python 3.12

### Installing

* Install Python libraries

```shell
pip install -r requirements.txt
```

* Build Cython libraries

```shell
bash scripts/build_project.sh
```

### Example Usage

```shell
python scripts/run_puzzle_solver.py -i demo_dataset -o output_dir -c configs/puzzle_solver_config.yaml
```

### Custom Datasets

To run the application with your own datasets, scan the puzzle pieces with the lid of your scanner open and use a ppi
close to 300. You can scan them into multiple images. Then, scan a background image. Put your puzzle piece images under
`my_dataset/puzzle_piece_images/` and your background image under `my_dataset/`.

## Known Issues

* This application doesn't work well on regular jigsaw puzzles because the puzzle pieces are too similar to each other.
* This application cannot handle ambiguous puzzles; each piece must have only one correct match.
* This application doesn't scale well. It tries to match every pair of puzzle pieces and takes O(N^2) time to solve a
  puzzle with N pieces. 