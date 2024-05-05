import time
from contextlib import ContextDecorator
from pathlib import Path
from prettytable import PrettyTable
import logging

logger = logging.getLogger(__name__)
profiling_data = {}


# The profiler overhead is about 4 us
class profile(ContextDecorator):
    def __init__(self, label: str = None):
        self._label = label

    def __call__(self, func):
        if self._label is None:
            self._label = func.__name__
        return super().__call__(func)

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        profiling_data.setdefault(self._label, [0.0, 0])
        entry = profiling_data[self._label]
        entry[0] += time.perf_counter() - self._start_time
        entry[1] += 1
        return False


def save_profiling_data(path: Path) -> None:
    """
    Write the profiling data to file as a table
    """
    table = PrettyTable()
    table.field_names = [
        "Name",
        "Total time (s)",
        "Average Time (s)",
        "Number of Times",
    ]
    table.add_rows(
        sorted(
            [
                [name, total_time, total_time / num_times, num_times]
                for name, (total_time, num_times) in profiling_data.items()
            ],
            key=lambda entry: entry[1],
            reverse=True,
        )
    )
    inaccurate_entries = [
        name
        for name, (total_time, num_times) in profiling_data.items()
        if total_time / num_times < 8e-5 # 20 x profiler overhead
    ]
    warning_string = f"The profiling data of the following entries are inaccurate because the overhead of the profiler is nonnegligible: {inaccurate_entries}"
    if len(inaccurate_entries) > 0:
        logger.warning(warning_string)

    with open(path, "w") as f:
        if len(inaccurate_entries) > 0:
            f.write(warning_string)
            f.write("\n")
        f.write(table.get_string())
