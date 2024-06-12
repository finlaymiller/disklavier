import os
from pathlib import Path
import redis
import pretty_midi
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import track

from typing import List

NUM_SEMITONES = 12
NUM_BEATS = 8


def transform(
    midi_path: str, semitones: int, beats: int, vel: float
) -> pretty_midi.PrettyMIDI:
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    tempo = int(Path(midi_path).stem.split("-")[1])
    beats_per_second = tempo / 60.0
    shift_seconds = 1 / beats_per_second
    new_midi = pretty_midi.PrettyMIDI()

    # shift
    for instrument in midi_data.instruments:
        new_inst = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum
        )
        for note in instrument.notes:
            # shift the start and end times of each note
            shifted_start = (note.start + shift_seconds * beats) % (
                NUM_BEATS / beats_per_second
            )
            shifted_end = (note.end + shift_seconds * beats) % (
                NUM_BEATS / beats_per_second
            )
            if shifted_end < shifted_start:  # handle wrapping around the cycle
                shifted_end += NUM_BEATS / beats_per_second
            new_note = pretty_midi.Note(
                velocity=int(np.round(note.velocity * vel)),  # vel scale
                pitch=note.pitch + semitones,  # transpose
                start=shifted_start,
                end=shifted_end,
            )
            new_inst.notes.append(new_note)
        new_midi.instruments.append(new_inst)

    return new_midi


# def augment(redis_url, midi_paths: List[str], transformations, index: int) -> int:
#     r = redis.Redis.from_url(redis_url)

#     print(f"SUBR{index:02d} processing {len(midi_paths)} files")

#     for file_name in track(
#         midi_paths,
#         description=f"[{index:02d}] augmenting files...",
#         refresh_per_second=1,
#         update_period=1.0,
#     ):
#         with r.pipeline() as pipeline:
#             file_path = os.path.join(input_dir, file_name)
#             pipeline.set(
#                 f"prs:{file_name[:-4]}:s00b00",
#                 pretty_midi.PrettyMIDI(file_path).get_piano_roll().tobytes(),
#             )

#             for semi, beat, vel in transformations:
#                 pipeline.set(
#                     f"prs:{file_name[:-4]}:s{semi:02d}b{beat:02d}",
#                     transform(file_path, semi, beat, vel).get_piano_roll().tobytes(),
#                 )

#             pipeline.execute()

#     return index


def augment(
    midi_paths: List[str], output_path: str, transformations, index: int
) -> int:
    # r = redis.Redis.from_url(redis_url)

    print(f"SUBR{index:02d} processing {len(midi_paths)} files")

    # all_prs = []
    for file_name in track(
        midi_paths,
        description=f"[{index:02d}] augmenting files...",
        refresh_per_second=1,
        update_period=1.0,
    ):
        file_path = os.path.join(input_dir, file_name)
        # piano_rolls = [
        #     {
        #         "name": f"{file_name[:-4]}_s00_b00",
        #         "pr": pretty_midi.PrettyMIDI(file_path).get_piano_roll(),
        #     }
        # ]

        # np.save(
        #     os.path.join(output_path, f"{file_name[:-4]}_s00b00"),
        #     pretty_midi.PrettyMIDI(file_path).get_piano_roll(),
        # )

        pretty_midi.PrettyMIDI(file_path).write(os.path.join(output_path, f"{file_name[:-4]}_s00b00.mid"))

        for semi, beat in transformations:
            # np.save(
            #     os.path.join(output_path, f"{file_name[:-4]}_s{semi:02d}b{beat:02d}"),
            #     transform(file_path, semi, beat, 1.0).get_piano_roll(),
            # )
            transform(file_path, semi, beat, 1.0).write(os.path.join(output_path, f"{file_name[:-4]}_s{semi:02d}b{beat:02d}.mid"))
        #     piano_rolls.append(
        #         {
        #             "name": f"{file_name[:-4]}_s{semi:02d}_b{beat:02d}",
        #             "pr": transform(file_path, semi, beat, 1.0).get_piano_roll(),
        #         }
        #     )

        # all_prs.extend(piano_rolls)

    # print(
    #     f"process {index} complete, saving to {os.path.join(output_path, f'prs_{index}')}"
    # )

    # np.savez_compressed(
    #     os.path.join(output_path, f"prs_{index}"),
    #     **{t["name"]: t["pr"] for t in all_prs},
    # )

    return index


if __name__ == "__main__":
    # file system
    redis_url = "redis://localhost:6379"
    dataset_name = "careful"
    input_dir = os.path.join("data", "datasets", f"{dataset_name}")
    output_dir = os.path.join("data", "outputs")
    files = [f for f in os.listdir(input_dir) if f.endswith(".mid")]

    num_processes = os.cpu_count()
    split_files = np.array_split(files, num_processes)  # type: ignore

    # transformations
    semis = list(range(NUM_SEMITONES))
    beats = list(range(NUM_BEATS))
    # vels = [0.8, 0.9, 1.0, 1.1, 1.2]
    transformation_table = [
        list(p) for p in itertools.product(semis, beats)
    ]  # , vels)]

    print(
        f"generating {len(transformation_table)} transformations for each of {len(files)} segments"
    )

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(augment, chunk, output_dir, transformation_table, i): chunk
            for i, chunk in enumerate(split_files)
        }

        for future in as_completed(futures):
            result = future.result()
            print(f"process {result} complete")
