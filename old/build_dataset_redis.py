
import os
from pathlib import Path
import json
import numpy as np
from itertools import product
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import pretty_midi
import redis
from concurrent.futures import ProcessPoolExecutor, as_completed

# from utils.metrics import blur_pr, energy, contour
# from utils.midi import transform
# from utils import get_tempo

dataset_path = "data/datasets/test"
metric = "pitch_histogram"

num_beats = 8
num_transpositions = 12


def update_best_matches(
    redis_client: redis.Redis, key, track_name_row, file_bm, sim_bm, metric
):
    """
    Update the pitch histogram in the Redis database based on the given criteria.

    Args:
        redis_client: Redis client object.
        key (str): The key to access the pitch histogram in Redis.
        track_name_row (str): The string to be compared with the substring before the underscore.
        file_bm (str): The filename to be inserted.
        sim_bm (float): The similarity value associated with the filename.
        metric (str): The metric to update.
    """

    # get the current list from Redis
    best_matches = redis_client.json().get(key, f"$.{metric}")
    if best_matches is None or len(best_matches) < 1:
        track_names = set()
    else:
        best_matches = best_matches[0]
        track_names = set(
            entry.split("@")[0].split("_")[0] for entry in best_matches
        )

    # check if the new track_name is already in the set or matches track_name_row
    new_track_name = file_bm.split("_")[0]
    if new_track_name in track_names or new_track_name == track_name_row:
        return  # dont update list

    # add the new entry
    best_matches.append(f"{file_bm}@{sim_bm}")

    # sort by similarity in descending order
    best_matches.sort(key=lambda x: float(x.split("@")[1]), reverse=True)

    redis_client.json().set(key, f"$.{metric}", best_matches[:3])


def calc_sims(
    rows,
    all_rows,
    metric,
    mod_table,
    index,
):
    print(f"[SUBR{index:02d}] starting subprocess {index:02d}")
    r = redis.Redis(host="localhost", port=6379, db=0)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    sim_task = progress.add_task(
        f"[SUBR{index:02d}] calculating sims", total=len(rows) * len(all_rows)
    )
    with progress:
        for i, row_file in enumerate(rows):
            print(
                f"[SUBR{index:02d}] {i:04d}/{len(rows):04d} calculating sims for file {row_file}"
            )
            track_name_row, _ = row_file.split("_")
            for col_file in all_rows:
                if col_file == row_file:
                    value = {
                        "sim": 1.0,
                        "mutations": {"shift": 0, "transpose": 0},
                        "row_file": row_file,
                        "col_file": col_file,
                        "metric": metric,
                    }
                else:
                    sim_best_mutation = -1
                    best_shift = -1
                    best_trans = -1

                    main_metric = list(
                        map(float, r.get(f"{metric}:{row_file}:0-0").decode().split(","))  # type: ignore
                    )
                    for t in mod_table:
                        s = 0  # TODO: REMOVE BEFORE SHIFTING
                        comp_metric = list(
                            map(float, r.get(f"{metric}:{col_file}:{s}-{t}").decode().split(","))  # type: ignore
                        )
                        similarity = np.dot(main_metric, comp_metric) / (
                            np.linalg.norm(main_metric) * np.linalg.norm(comp_metric)
                        )

                        if similarity > sim_best_mutation:
                            sim_best_mutation = similarity
                            best_shift = s
                            best_trans = t

                    value = {
                        "sim": sim_best_mutation,
                        "mutations": {"shift": best_shift, "trans": best_trans},
                        "row_file": row_file,
                        "col_file": col_file,
                        "metric": metric,
                    }

                # update comparison object
                r.json().set(f"cmp:{row_file}:{col_file}:{metric}", "$", value)

                # update row file object
                update_best_matches(
                    r,
                    f"file:{row_file}",
                    track_name_row,
                    col_file,
                    value["sim"],
                    metric,
                )
                progress.advance(sim_task)

    print(f"[SUBR{index:02d}] subprocess complete")

    return index


def main():
    names = os.listdir(dataset_path)
    names.sort()

    num_processes = os.cpu_count()
    split_keys = np.array_split(names, num_processes)  # type: ignore
    mod_table = list(product(range(num_transpositions), range(num_beats)))

    r = redis.Redis(host="localhost", port=6379, db=0)

    # calculate metrics for each file
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    pitch_histogram_task = progress.add_task(f"uploading '{metric}'", total=len(names))
    with progress:
        for file in names:
            r.json().set(f"file:{file}", "$", {f"{metric}": []}, nx=True)
            # for t, s in mod_table:
            #     transformed_path = transform(os.path.join(dataset_path, file), "outputs/tmp", get_tempo(file), {"transpose": t, "shift": s})
            file_path = os.path.join(dataset_path, file)
            transpose = file.split('_')[-1][:-4][:3]
            shift = file.split('_')[-1][:-4][3:]
            pch = pretty_midi.PrettyMIDI(file_path).get_pitch_class_histogram(
                use_duration=True
            )

            r.set(
                f"{metric}:{file[:-11]}:{transpose}-{shift}",
                ",".join(map(str, pch)),
                nx=True,
            )
            progress.advance(pitch_histogram_task)

    # calculate similarities
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(calc_sims, chunk, names, metric, mod_table, i): chunk
            for i, chunk in enumerate(split_keys)
        }

        for future in as_completed(futures):
            index = future.result()
            print(f"[MAIN]   subprocess {index} returned")


if __name__ == "__main__":
    main()
