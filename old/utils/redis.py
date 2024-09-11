import redis
import hashlib
import numpy as np
import pandas as pd
from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from itertools import product

from utils.midi import split_filename

from typing import List


class Comparison:
    def __init__(self, sim, shift, trans, row, col, metric="pitch_histogram"):
        self.sim = float(sim)
        self.shift = int(shift)
        self.trans = int(trans)
        self.row = str(row)
        self.col = str(col)
        self.metric = str(metric)

    @classmethod
    def from_redis_values(cls, values):
        """Create a Comparison instance from a list of Redis values.

        Args:
            values: A list of values retrieved from Redis.

        Returns:
            A Comparison instance.
        """
        sim, shift, trans, row, col, metric = values
        return cls(
            sim.decode("utf-8"),
            shift.decode("utf-8"),
            trans.decode("utf-8"),
            row.decode("utf-8"),
            col.decode("utf-8"),
            metric.decode("utf-8"),
        )


def get_track_id(track_name: str) -> str:
    """Generate a unique ID for a track name."""
    # return hashlib.md5(track_name.encode()).hexdigest()[:5]
    return track_name


def store_vector(
    redis_client: redis.Redis, filename: str, metric: str, vector: np.ndarray
):
    """Store a vector in Redis using an efficient key-value structure.

    Args:
        redis_client: An instance of Redis client.
        filename: The name of the file whose metric being uploaded.
        metric: The name of the metric.
        vector: The vector to store.
    """
    # generate key parts
    basename, transpose, shift = split_filename(filename)
    track_id = get_track_id(basename)

    all_ids = redis_client.json().get("track_ids")
    if all_ids is None:
        redis_client.json().set("track_ids", "$", {})
        all_ids = redis_client.json().get("track_ids")

    if track_id not in all_ids.values():
        redis_client.json().set("track_ids", f"$.{basename}", track_id)

    # store the vector in the hash
    redis_client.hset(
        f"file:{track_id}:{metric}", f"{transpose}{shift}", vector.tobytes()
    )


def load_vector(
    redis_client: redis.Redis, filename: str, metric: str, dtype=np.float64, shape=(12,)
):
    """Load a vector from Redis using a key-value structure.

    Args:
        redis_client: An instance of Redis client.
        filename: The name of the file whose metric being uploaded.
        metric: The name of the metric.
        dtype: The data type of the vector.
        shape: The shape of the vector.

    Returns:
        The vector.
    """
    # generate key parts
    basename, transpose, shift = split_filename(filename)
    track_id = get_track_id(basename)

    # retrieve the vector from the hash
    vector_bytes = redis_client.hget(f"file:{track_id}:{metric}", f"{transpose}{shift}")

    if vector_bytes is None:
        print(f"WARNING: couldn't get vector for key 'file:{track_id}:{metric}'")
        return None

    # convert bytes back to vector
    return np.frombuffer(vector_bytes, dtype=dtype).reshape(shape)  # type: ignore


def load_vectors(
    redis_client: redis.Redis,
    filename: str,
    metric: str,
    fields: List[str] | None = None,
    dtype=np.float64,
    shape=(12,),
):
    """Load vectors from Redis using a key-value structure.

    Args:
        redis_client: An instance of Redis client.
        filename: The name of the file whose metric being uploaded.
        metric: The name of the metric.
        dtype: The data type of the vector.
        shape: The shape of the vector.

    Returns:
        The vector.
    """
    # generate key parts
    basename, _, _ = split_filename(filename)
    track_id = get_track_id(basename)

    # retrieve the vectors from the hash
    if fields is None:
        vector_dict = redis_client.hgetall(f"file:{track_id}:{metric}")
    else:
        vector_dict = redis_client.hmget(f"file:{track_id}:{metric}", fields)

    if vector_dict is None:
        print(f"WARNING: couldn't get vector for key 'file:{track_id}:{metric}'")
        return None

    # convert bytes back to vector
    byte2arr = lambda b: np.frombuffer(b, dtype=dtype).reshape(shape)

    return {k.decode("utf-8"): byte2arr(v) for k, v in vector_dict.items()}  # type: ignore


def verify_df(r, row, df, p):
    print(f"calculating mask...")
    mask = np.random.rand(*df.shape) < p
    print("arranging samples...")
    samples = [
        (row_idx, col_idx, df.iat[row_idx, col_idx])
        for row_idx, col_idx in zip(*np.where(mask))
    ]

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    check_task = progress.add_task(f"checking files...", total=len(samples))

    with progress:
        for row, col, val in samples:
            sim = r.json().get(f"cmp:{row}:{col}:pitch_histogram", "$.sim")

            if sim != val:
                print(f"value mismatch found at {row}:{col} -> {sim} != {val}")

            progress.advance(check_task)

    print("check complete")


def build_similarity_table(
    all_files: List[str], output_path: str, metric: str = "pitch_histogram"
):
    r = redis.Redis(host="localhost", port=6379, db=0)
    s_table = pd.DataFrame(index=all_files, columns=all_files)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    update_task = progress.add_task(
        f"gathering similarities", total=len(all_files) ** 2
    )

    with progress:
        for i in range(len(all_files)):
            row_file = all_files[i]
            keys = [
                f"cmp:{row_file[:-4]}:{col_file[:-4]}:{metric}"
                for col_file in all_files
            ]

            with r.pipeline() as pipe:
                for key in keys:
                    pipe.execute_command(
                        "HMGET",
                        key,
                        "sim",
                        "shift",
                        "trans",
                        "row_file",
                        "col_file",
                        "metric",
                    )
                results = pipe.execute()

            for key, value in zip(keys, results):
                result = Comparison.from_redis_values(value)
                s_table.at[row_file, result.col] = {
                    "sim": result.sim,
                    "transformations": {
                        "transpose": result.trans,
                        "shift": result.shift,
                    },
                }
                progress.advance(update_task)

    s_table.to_parquet(output_path)


def build_neighbor_table(all_files: List[str], output_path: str) -> None:
    column_names = ["prev_2", "prev", "current", "next", "next_2"]
    n_table = pd.DataFrame(index=all_files, columns=column_names)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    update_task = progress.add_task(f"gathering neighbors", total=len(all_files))
    with progress:
        for i, file in enumerate(all_files):
            neighbors = []
            curr_track, _ = file.split("_")
            for offset in range(-2, 3):
                idx = i + offset
                valid = 0 <= idx < len(all_files)
                filename = (
                    all_files[idx]
                    if valid and all_files[idx].split("_")[0] == curr_track
                    else None
                )
                neighbors.append(filename)

            n_table.loc[file] = neighbors
            progress.advance(update_task)

    n_table.to_parquet(output_path)


def build_transformation_table(
    all_files: List[str], output_path: str, metric: str = "pitch_histogram"
) -> None:
    r = redis.Redis(host="localhost", port=6379, db=0)
    column_names = [f"{t:02d}{s:02d}" for t, s in product(range(12), range(8))]
    t_table = pd.DataFrame(index=all_files, columns=column_names)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    update_task = progress.add_task(
        f"gathering transformations", total=len(all_files) * len(column_names)
    )

    with progress:
        for row_file in all_files:
            results = load_vectors(r, f"{row_file[:-4]}_DISCARDED", metric)
            for k, v in results.items():
                t_table.at[row_file, k] = v
                progress.advance(update_task)

    t_table.to_parquet(output_path)
