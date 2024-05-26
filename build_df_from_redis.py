import os
import json
import time
import redis
import numpy as np
import pandas as pd
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from typing import List


def verify_df(row, df, p):
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


def build_similarity_dataframe(all_files: List[str]):
    r = redis.Redis(host="localhost", port=6379, db=0)
    df = pd.DataFrame(index=all_files, columns=all_files)

    # process keys in batches
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    update_task = progress.add_task(
        f"gathering similarities", total=len(all_files)
    )

    with progress:
        for i in range(len(all_files)):
            row_file = all_files[i]
            keys = [
                f"cmp:{row_file}:{col_file}:pitch_histogram" for col_file in all_files
            ]

            with r.pipeline() as pipe:
                for key in keys:
                    pipe.execute_command("JSON.GET", key, "$")  # TODO: try MGET
                results = pipe.execute()

            # parse the results and update the dataframe
            for key, value in zip(keys, results):
                result = json.loads(value)[0]
                if value:
                    if result["col_file"] in all_files:
                        df.at[row_file, result["col_file"]] = {"sim": result["sim"], "transformations": result["mutations"]}
                    else:
                        print(
                            f"ERROR@key {result['col_file']}: not set on {key}"
                        )
                else:
                    print(f"ERROR@key {key}: no value")
            progress.advance(update_task)

    df.to_feather(
        os.path.join("outputs", "records", "chunks", "sim.feather")
    )

if __name__ == "__main__":
    # connect to redis
    r = redis.Redis(host="localhost", port=6379, db=0)

    # load filenames
    properties_path = "data/outputs/careful.json"
    properties = {}
    with open(properties_path, "r") as f:
        properties = json.load(f)
    names = list(properties.keys())
    names.sort()

    st = time.time()
    build_similarity_dataframe(names)
    et = time.time()

    big_df = pd.read_feather(
        os.path.join("outputs", "records", "chunks", "sim.feather")
    )

    memory_usage = big_df.memory_usage(index=True).sum()
    print(
        f"[MAIN] Time taken to generate DataFrame: {et - st:.2f}s"
    )
    print(f"[MAIN] Memory usage of DataFrame: {memory_usage / (1024 * 1024):.2f} MB")

    # verify_df(r, big_df, 10)
