import os
from pathlib import Path
import json
import numpy as np
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import pretty_midi
import redis
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.metrics import blur_pr, energy, contour

dataset_path = "data/datasets/careful"
properties_path = "data/outputs/careful.json"
metric = "pitch_histogram_wd"

num_beats = 8
num_transpositions = 12


def update_best_matches(
    redis_client: redis.Redis, key, track_name_row, file_bm, sim_bm, metric
):
    """
    Update the pitch histogram in the Redis database based on the given criteria.
    TODO: better variable names

    Args:
        redis_client: Redis client object.
        key (str): The key to access the pitch histogram in Redis.
        track_name_row (str): The string to be compared with the substring before the underscore.
        file_bm (str): The filename to be inserted.
        sim_bm (float): The similarity value associated with the filename.
        metric (str): The metric to update.
    """

    def extract_track_name(filename: str):
        return filename.split("_")[0]

    def format_entry(filename, sim):
        return f"{filename}@{sim}"

    # get the current list from Redis
    # TODO: handle this nested list better
    best_matches = redis_client.json().get(key, f"$.{metric}")
    if best_matches is None or len(best_matches) < 1:
        track_names = set()
    else:
        best_matches = best_matches[0]
        track_names = set(
            extract_track_name(entry.split("@")[0]) for entry in best_matches
        )

    # check if the new track_name is already in the set or matches track_name_row
    new_track_name = extract_track_name(file_bm)
    if new_track_name in track_names or new_track_name == track_name_row:
        return  # dont update list

    # add the new entry
    best_matches.append(format_entry(file_bm, sim_bm))

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


def transpose_and_shift_midi(
    midi_path: str, semitones: int, beats: int, total_beats=8
) -> pretty_midi.PrettyMIDI:
    """
    Transpose and shift a MIDI file by a specified number of semitones and beats.

    Args:
        midi_path (str): The path to the MIDI file.
        semitones (int): Number of semitones to transpose the MIDI file.
        beats (int): Number of beats to shift the MIDI events.
        total_beats (int): number of beats in the file (default 8)

    Returns:
        pretty_midi.PrettyMIDI: The modified MIDI file.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    tempo = int(Path(midi_path).stem.split("-")[1])
    beats_per_second = tempo / 60.0
    shift_seconds = 1 / beats_per_second
    s_t_midi = pretty_midi.PrettyMIDI()

    # shift
    for instrument in midi_data.instruments:
        new_inst = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum
        )
        for note in instrument.notes:
            # shift the start and end times of each note
            shifted_start = (note.start + shift_seconds * beats) % (
                total_beats / beats_per_second
            )
            shifted_end = (note.end + shift_seconds * beats) % (
                total_beats / beats_per_second
            )
            if shifted_end < shifted_start:  # handle wrapping around the cycle
                shifted_end += beats / beats_per_second
            s_t_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch + semitones,  # transpose
                start=shifted_start,
                end=shifted_end,
            )
            new_inst.notes.append(s_t_note)
        s_t_midi.instruments.append(new_inst)

    return s_t_midi


def main():
    properties = {}
    with open(properties_path, "r") as f:
        properties = json.load(f)

    names = list(properties.keys())
    names.sort()

    num_processes = os.cpu_count()
    split_keys = np.array_split(names, num_processes)  # type: ignore

    mod_table = list(range(12))  # []
    # for s in range(8):
    #     for t in range(12):
    #         mod_table.append([s, t])

    r = redis.Redis(host="localhost", port=6379, db=0)

    # calculate metrics for each file
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=1,
    )
    pitch_histogram_task = progress.add_task(
        f"uploading '{metric}'", total=(len(names) * 12)
    )
    with progress:
        for file in names:
            r.json().set(f"file:{file}", "$", {f"{metric}": []}, nx=True)
            for t in mod_table:
                s = 0  # NOTE: fix before switching to other metrics
                pch = transpose_and_shift_midi(
                    os.path.join(dataset_path, file), t, s
                ).get_pitch_class_histogram(use_duration=True)

                r.set(
                    f"{metric}:{file}:{s}-{t}",
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
