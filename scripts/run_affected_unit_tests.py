import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FULL_SUITE_TRIGGER_PATHS = {
    ".githooks/pre-commit",
    "requirements.txt",
    "scripts/build_project.sh",
    "scripts/run_affected_unit_tests.py",
    "setup.py",
}

PREFIX_TO_TESTS = (
    (
        "algorithms/",
        (
            "unit_tests/test_algorithms.py",
            "unit_tests/test_puzzle_solver_app_regression.py",
        ),
    ),
    (
        "applications/",
        ("unit_tests/test_puzzle_solver_app_regression.py",),
    ),
    (
        "data_structures/",
        (
            "unit_tests/test_data_structures.py",
            "unit_tests/test_puzzle_solver_app_regression.py",
        ),
    ),
    ("demo_dataset/", ("unit_tests/test_puzzle_solver_app_regression.py",)),
    (
        "utils/",
        ("unit_tests/test_utils.py", "unit_tests/test_puzzle_solver_app_regression.py"),
    ),
)

EXACT_PATH_TO_TESTS = {
    "configs/puzzle_solver_config.yaml": (
        "unit_tests/test_puzzle_solver_app_regression.py",
    ),
    "cython_libs.pyx": (
        "unit_tests/test_find_neighbors_within_radius.py",
        "unit_tests/test_algorithms.py",
        "unit_tests/test_puzzle_solver_app_regression.py",
    ),
    "scripts/run_puzzle_solver.py": (
        "unit_tests/test_puzzle_solver_app_regression.py",
    ),
    "visualizations.py": ("unit_tests/test_visualizations.py",),
}

TEST_ORDER = (
    "unit_tests/test_algorithms.py",
    "unit_tests/test_data_structures.py",
    "unit_tests/test_find_neighbors_within_radius.py",
    "unit_tests/test_puzzle_solver_app_regression.py",
    "unit_tests/test_utils.py",
    "unit_tests/test_visualizations.py",
)


@dataclass(frozen=True)
class TestSelection:
    run_full_suite: bool
    test_paths: tuple[str, ...]


def staged_files() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in result.stdout.splitlines() if path]


def select_affected_unit_tests(changed_paths: Iterable[str]) -> TestSelection:
    selected_tests: set[str] = set()
    extra_tests: list[str] = []

    for raw_path in changed_paths:
        path = raw_path.strip().replace(os.sep, "/")
        if not path:
            continue
        if path in FULL_SUITE_TRIGGER_PATHS:
            return TestSelection(run_full_suite=True, test_paths=())
        if path.startswith("unit_tests/") and Path(path).name.startswith("test_"):
            selected_tests.add(path)
            if path not in TEST_ORDER:
                extra_tests.append(path)
            continue
        for exact_path, test_paths in EXACT_PATH_TO_TESTS.items():
            if path == exact_path:
                selected_tests.update(test_paths)
                break
        else:
            for prefix, test_paths in PREFIX_TO_TESTS:
                if path.startswith(prefix):
                    selected_tests.update(test_paths)
                    break

    ordered_tests = [
        test_path for test_path in TEST_ORDER if test_path in selected_tests
    ]
    ordered_tests.extend(
        test_path for test_path in extra_tests if test_path in selected_tests
    )
    return TestSelection(run_full_suite=False, test_paths=tuple(ordered_tests))


def unittest_command(selection: TestSelection) -> list[str]:
    if selection.run_full_suite:
        return [sys.executable, "-m", "unittest", "discover", "-s", "unit_tests"]
    return [sys.executable, "-m", "unittest", *selection.test_paths]


def run_selection(selection: TestSelection) -> int:
    if not selection.run_full_suite and not selection.test_paths:
        print("No affected unit tests for staged files; skipping.")
        return 0

    env = os.environ.copy()
    temp_dir = Path(tempfile.gettempdir())
    env.setdefault("MPLCONFIGDIR", str(temp_dir / "mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(temp_dir / "xdg-cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    command = unittest_command(selection)
    if selection.run_full_suite:
        print("Running full unit test suite.", flush=True)
    else:
        print("Running affected unit tests.", flush=True)
    print(shlex.join(command), flush=True)
    return subprocess.run(command, env=env).returncode


def main() -> int:
    changed_paths = staged_files()
    if not changed_paths:
        print("No staged files found; skipping affected unit tests.")
        return 0
    return run_selection(select_affected_unit_tests(changed_paths))


if __name__ == "__main__":
    raise SystemExit(main())
