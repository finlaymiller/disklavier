import os
import time
import zipfile
from pretty_midi import PrettyMIDI
import numpy as np
from rich import print
from rich.pretty import pprint
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import redis
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser

from utils.midi import insert_transformations
from utils.redis import *

from typing import List


def calc_sims(rows: List[str], all_rows: List[str], metric: str, index: int):
    print(
        f"[SUBR{index:02d}] starting subprocess {index:02d} with {len(rows)}/{len(all_rows)} rows"
    )
    # r = redis.Redis(host="localhost", port=6379, db=0)

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
                f"[SUBR{index:02d}] {i:04d}/{len(rows):04d} calculating sims for file '{row_file}'"
            )

            # find optimal transformation for each other file in dataset to maximize similarity
            for col_file in all_rows:
                best_sim = -1
                best_shift = -1
                best_trans = -1

                row_metric = load_vector(r, insert_transformations(row_file), metric)
                col_metrics = load_vectors(r, insert_transformations(col_file), metric)

                for key in col_metrics.keys():
                    similarity = np.dot(row_metric, col_metrics[key]) / (
                        np.linalg.norm(row_metric) * np.linalg.norm(col_metrics[key])
                    )

                    if similarity > best_sim:
                        best_sim = np.round(similarity, 5)
                        best_trans = f"{key}"[:2]
                        best_shift = f"{key}"[2:]

                value = {
                    "sim": best_sim,
                    "shift": best_shift,
                    "trans": best_trans,
                    "row_file": row_file,
                    "col_file": col_file,
                    "metric": metric,
                }

                # update comparison object
                for k, v in value.items():
                    r.hset(f"cmp:{row_file[:-4]}:{col_file[:-4]}:{metric}", k, v)

                progress.advance(sim_task)

    progress.remove_task(sim_task)
    print(f"[SUBR{index:02d}] subprocess complete")

    return index


def main(args):
    r = redis.Redis(host="localhost", port=6379, db=0)

    # filenames
    train_dir = os.path.join(args.data_dir, "train")
    all_filenames = [f for f in os.listdir(train_dir) if f.endswith(".mid")]
    all_filenames.sort()
    play_dir = os.path.join(args.data_dir, "play")
    base_filenames = [f for f in os.listdir(play_dir) if f.endswith(".mid")]
    base_filenames.sort()
    split_keys = np.array_split(base_filenames, os.cpu_count())  # type: ignore

    # calculate metrics for each file
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    metric_task = progress.add_task(
        f"uploading {args.metric}s", total=(len(all_filenames))
    )
    with progress:
        for file in all_filenames:
            match args.metric:
                case "pitch_histogram":
                    vector = PrettyMIDI(
                        os.path.join(train_dir, file)
                    ).get_pitch_class_histogram(True, True)
                case "clamp":
                    pass
            store_vector(r, file, args.metric, vector)
            progress.advance(metric_task)

    # calculate similarities
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(calc_sims, chunk, base_filenames, args.metric, i): chunk
            for i, chunk in enumerate(split_keys)
        }

        for future in as_completed(futures):
            index = future.result()
            print(f"[MAIN]   subprocess {index} returned")

    # build tables
    table_list = []
    if args.build_sim:
        parquet_path = os.path.join(
            "data", "tables", f"{args.dataset_name}_sim.parquet"
        )
        table_list.append(parquet_path)

        start_time = time.time()
        build_similarity_table(base_filenames, parquet_path)
        end_time = time.time()

        big_df = pd.read_parquet(parquet_path)
        print(f"successfully built sim table '{parquet_path}'")
        pprint(big_df.head())

        memory_usage = big_df.memory_usage(index=True).sum()
        print(f"Time taken to generate DataFrame: {end_time - start_time:.2f} s")
        print(f"Memory usage of DataFrame: {memory_usage / (1024 * 1024):.2f} MB")
        del big_df

    if args.build_neighbor:
        parquet_path = os.path.join(
            "data", "tables", f"{args.dataset_name}_neighbor.parquet"
        )
        table_list.append(parquet_path)

        start_time = time.time()
        build_neighbor_table(base_filenames, parquet_path)
        end_time = time.time()

        big_df = pd.read_parquet(parquet_path)
        print(f"successfully built neighbor table '{parquet_path}':")
        pprint(big_df.head())

        memory_usage = big_df.memory_usage(index=True).sum()
        print(f"Time taken to generate DataFrame: {end_time - start_time:.2f} s")
        print(f"Memory usage of DataFrame: {memory_usage / (1024 * 1024):.2f} MB")
        del big_df

    if args.build_transformation:
        parquet_path = os.path.join(
            "data", "tables", f"{args.dataset_name}_transformations.parquet"
        )
        table_list.append(parquet_path)

        start_time = time.time()
        build_transformation_table(base_filenames, parquet_path)
        end_time = time.time()

        big_df = pd.read_parquet(parquet_path)
        print(f"successfully built transformation table '{parquet_path}':")
        pprint(big_df.head())

        memory_usage = big_df.memory_usage(index=True).sum()
        print(f"Time taken to generate DataFrame: {end_time - start_time:.2f} s")
        print(f"Memory usage of DataFrame: {memory_usage / (1024 * 1024):.2f} MB")
        del big_df

    # CHATGPT UNTESTED
    zip_path = os.path.join("outputs", f"{args.dataset_name}_tables.zip")
    print(f"compressing to zipfile '{zip_path}'")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for table in table_list:
            zipf.write(table, os.path.basename(table))
    print("DONE")


if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--dataset_name", type=str, default="dataset", help="name of dataset"
    )
    parser.add_argument(
        "--metric", "-m", default="pitch_histogram", help="metric to use/calculate"
    )
    parser.add_argument(
        "--num_beats",
        type=int,
        default=8,
        help="number of beats each segment should have, not including the leading and trailing sections of each segment",
    )
    parser.add_argument(
        "--num_semitones",
        type=int,
        default=12,
        help="number of semitones to transpose",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    parser.add_argument(
        "--build_sim",
        "-s",
        action="store_true",
        default=False,
        help="actually build sim table, vs. just uploading the files",
    )
    parser.add_argument(
        "--build_neighbor",
        "-n",
        action="store_true",
        default=False,
        help="actually build neighbor table",
    )
    parser.add_argument(
        "--build_transformation",
        "-r",
        action="store_true",
        default=False,
        help="actually build transformation table",
    )
    parser.add_argument(
        "-t",
        action="store_true",
        default=False,
        help="test dataset mode (don't expect transformations)",
    )
    args = parser.parse_args()

    pprint(args)

    main(args)
import os
import time
import zipfile
from pretty_midi import PrettyMIDI
import numpy as np
from rich import print
from rich.pretty import pprint
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import redis
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser

from utils.midi import insert_transformations
from utils.redis import *

from typing import List


def calc_sims(rows: List[str], all_rows: List[str], metric: str, index: int):
    print(
        f"[SUBR{index:02d}] starting subprocess {index:02d} with {len(rows)}/{len(all_rows)} rows"
    )
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
                f"[SUBR{index:02d}] {i:04d}/{len(rows):04d} calculating sims for file '{row_file}'"
            )

            # find optimal transformation for each other file in dataset to maximize similarity
            for col_file in all_rows:
                best_sim = -1
                best_shift = -1
                best_trans = -1

                row_metric = load_vector(r, insert_transformations(row_file), metric)
                col_metrics = load_vectors(r, insert_transformations(col_file), metric)

                for key in col_metrics.keys():
                    similarity = np.dot(row_metric, col_metrics[key]) / (
                        np.linalg.norm(row_metric) * np.linalg.norm(col_metrics[key])
                    )

                    if similarity > best_sim:
                        best_sim = np.round(similarity, 5)
                        best_trans = f"{key}"[:2]
                        best_shift = f"{key}"[2:]

                value = {
                    "sim": best_sim,
                    "shift": best_shift,
                    "trans": best_trans,
                    "row_file": row_file,
                    "col_file": col_file,
                    "metric": metric,
                }

                # update comparison object
                for k, v in value.items():
                    r.hset(f"cmp:{row_file[:-4]}:{col_file[:-4]}:{metric}", k, v)

                progress.advance(sim_task)

    progress.remove_task(sim_task)
    print(f"[SUBR{index:02d}] subprocess complete")

    return index


def main(args):
    r = redis.Redis(host="localhost", port=6379, db=0)

    # filenames
    train_dir = os.path.join(args.data_dir, "train")
    all_filenames = [f for f in os.listdir(train_dir) if f.endswith(".mid")]
    all_filenames.sort()
    play_dir = os.path.join(args.data_dir, "play")
    base_filenames = [f for f in os.listdir(play_dir) if f.endswith(".mid")]
    base_filenames.sort()
    split_keys = np.array_split(base_filenames, os.cpu_count())  # type: ignore

    # calculate metrics for each file
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    pitch_histogram_task = progress.add_task(
        f"uploading {args.metric}s", total=(len(all_filenames))
    )
    with progress:
        for file in all_filenames:
            vector = PrettyMIDI(
                os.path.join(train_dir, file)
            ).get_pitch_class_histogram(True, True)
            store_vector(r, file, args.metric, vector)
            progress.advance(pitch_histogram_task)

    # calculate similarities
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(calc_sims, chunk, base_filenames, args.metric, i): chunk
            for i, chunk in enumerate(split_keys)
        }

        for future in as_completed(futures):
            index = future.result()
            print(f"[MAIN]   subprocess {index} returned")

    # build tables
    table_list = []
    if args.build_sim:
        parquet_path = os.path.join(
            "data", "tables", f"{args.dataset_name}_sim.parquet"
        )
        table_list.append(parquet_path)

        start_time = time.time()
        build_similarity_table(base_filenames, parquet_path)
        end_time = time.time()

        big_df = pd.read_parquet(parquet_path)
        print(f"successfully built sim table '{parquet_path}'")
        pprint(big_df.head())

        memory_usage = big_df.memory_usage(index=True).sum()
        print(f"Time taken to generate DataFrame: {end_time - start_time:.2f} s")
        print(f"Memory usage of DataFrame: {memory_usage / (1024 * 1024):.2f} MB")
        del big_df

    if args.build_neighbor:
        parquet_path = os.path.join(
            "data", "tables", f"{args.dataset_name}_neighbor.parquet"
        )
        table_list.append(parquet_path)

        start_time = time.time()
        build_neighbor_table(base_filenames, parquet_path)
        end_time = time.time()

        big_df = pd.read_parquet(parquet_path)
        print(f"successfully built neighbor table '{parquet_path}':")
        pprint(big_df.head())

        memory_usage = big_df.memory_usage(index=True).sum()
        print(f"Time taken to generate DataFrame: {end_time - start_time:.2f} s")
        print(f"Memory usage of DataFrame: {memory_usage / (1024 * 1024):.2f} MB")
        del big_df

    if args.build_transformation:
        parquet_path = os.path.join(
            "data", "tables", f"{args.dataset_name}_transformations.parquet"
        )
        table_list.append(parquet_path)

        start_time = time.time()
        build_transformation_table(base_filenames, parquet_path)
        end_time = time.time()

        big_df = pd.read_parquet(parquet_path)
        print(f"successfully built transformation table '{parquet_path}':")
        pprint(big_df.head())

        memory_usage = big_df.memory_usage(index=True).sum()
        print(f"Time taken to generate DataFrame: {end_time - start_time:.2f} s")
        print(f"Memory usage of DataFrame: {memory_usage / (1024 * 1024):.2f} MB")
        del big_df

    # CHATGPT UNTESTED
    zip_path = os.path.join("outputs", f"{args.dataset_name}_tables.zip")
    print(f"compressing to zipfile '{zip_path}'")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for table in table_list:
            zipf.write(table, os.path.basename(table))
    print("DONE")


if __name__ == "__main__":
    # load args
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument("--data_dir", default=None, help="location of MIDI files")
    parser.add_argument(
        "--dataset_name", type=str, default="dataset", help="name of dataset"
    )
    parser.add_argument(
        "--metric", "-m", default="pitch_histogram", help="metric to use/calculate"
    )
    parser.add_argument(
        "--num_beats",
        type=int,
        default=8,
        help="number of beats each segment should have, not including the leading and trailing sections of each segment",
    )
    parser.add_argument(
        "--num_semitones",
        type=int,
        default=12,
        help="number of semitones to transpose",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after a certain number of files",
    )
    parser.add_argument(
        "--build_sim",
        "-s",
        action="store_true",
        default=False,
        help="actually build sim table, vs. just uploading the files",
    )
    parser.add_argument(
        "--build_neighbor",
        "-n",
        action="store_true",
        default=False,
        help="actually build neighbor table",
    )
    parser.add_argument(
        "--build_transformation",
        "-r",
        action="store_true",
        default=False,
        help="actually build transformation table",
    )
    parser.add_argument(
        "-t",
        action="store_true",
        default=False,
        help="test dataset mode (don't expect transformations)",
    )
    args = parser.parse_args()

    pprint(args)

    main(args)
