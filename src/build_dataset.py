import os
import mido
import time
import random
import zipfile
import numpy as np
import pretty_midi
from shutil import copy2
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from itertools import product
from omegaconf import OmegaConf
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import basename, console
from utils.novelty import gen_ssm_and_novelty
from utils.midi import get_bpm, set_bpm, transform, trim_piano_roll
from utils.dataset import (
    add_metronome,
    add_novelty,
    modify_end_of_track,
    add_beats_to_file,
)
from typing import List

plt.style.use("dark_background")


class SegmentOptions:
    def __init__(
        self,
        num_beats: int,
        metronome: int,
        beats: bool,
        novelty: bool,
        pic_dir: str,
        lead_window_beat_frac: int,
    ):
        self.num_beats = num_beats
        self.lead_window_beat_frac = lead_window_beat_frac
        self.metronome = metronome
        self.beats = beats
        self.novelty = novelty
        self.pic_dir = pic_dir


class AugmentOptions:
    def __init__(self, num_shifts: int, num_transposes: int, tempo_fold: bool):
        self.num_shifts = num_shifts
        self.num_transposes = num_transposes
        self.tempo_fold = tempo_fold


def augment_midi(
    p: Progress,
    filename: str,
    new_segments: List[str],
    output_path: str,
    options: AugmentOptions,
) -> List[str]:
    augmented_files = []
    task_a = p.add_task(
        f"augmenting {basename(filename)}",
        total=len(new_segments) * options.num_transposes * options.num_shifts,
    )

    with p:
        for segment_filename in new_segments:
            transformations = [
                {"transpose": t, "shift": s}
                for t, s in product(
                    range(options.num_transposes), range(options.num_shifts)
                )
            ]
            for transformation in transformations:
                augmented_files.append(
                    transform(
                        segment_filename,
                        output_path,
                        get_bpm(segment_filename),
                        transformation,
                    )
                )
                p.update(task_a, advance=1)
        p.remove_task(task_a)

    return augmented_files


def segment_midi(
    midi_file_path: str,
    output_dir: str,
    options: SegmentOptions,
) -> List[str]:
    """
    Break a MIDI file into segments.

    Parameters
    ----------
    midi_file_path : str
        path to the midi file to segment
    output_dir : str
        path to the directory to save the segmented files
    options : SegmentOptions
        options for the segmentation

    Returns
    -------
    List[str]
        list of paths to the segmented files
    """
    trackname = basename(midi_file_path)
    # preserve tempo across all segments
    bpm = get_bpm(midi_file_path)
    set_bpm(midi_file_path, bpm)

    # calculate times and stuff
    midi_pm = pretty_midi.PrettyMIDI(midi_file_path)
    total_file_length_s = midi_pm.get_end_time()
    beat_length_s = 60 / bpm
    segment_length_s = options.num_beats * beat_length_s
    n_segments = int(np.round(total_file_length_s / segment_length_s))
    pre_beat_window_s = (
        beat_length_s / options.lead_window_beat_frac
    )  # capture when first note is a bit early

    if options.novelty:
        add_novelty(midi_file_path, options.pic_dir)

    # console.log(
    #     f"\tbreaking '{trackname}' ({total_file_length_s:.03f} s at {bpm} bpm) into {n_segments:03d} segments of {segment_length_s:.03f} s\n\t(pre window is {pre_beat_window_s:.03f} s)"
    # )

    new_files = []
    # -1 because the last segment often contains a mistake or isn't otherwise suitable for playback
    for n in list(range(n_segments - 1)):
        # add one beat since tracks now start with the lead-in beat included
        start = n * segment_length_s
        end = start + segment_length_s + beat_length_s
        if n > 0:
            start += beat_length_s - pre_beat_window_s

        # console.log(f"\t{n:03d} splitting from {start:08.03f} s to {end:07.03f} s")

        segment_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        instrument = pretty_midi.Instrument(
            program=midi_pm.instruments[0].program,
            name=f"{trackname}_{int(start):04d}-{int(end):04d}",
        )

        # add notes from the original MIDI that fall within the current segment
        for note in midi_pm.instruments[0].notes:
            if start <= note.start < end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start,
                    end=note.end - start,
                )
                # pad front of track to full bar for easier playback
                if n > 0:
                    new_note.start += pre_beat_window_s * (
                        options.lead_window_beat_frac - 1
                    )
                    new_note.end += pre_beat_window_s * (
                        options.lead_window_beat_frac - 1
                    )
                instrument.notes.append(new_note)

        # write out
        segment_filename = os.path.join(
            output_dir, f"{trackname}_{int(start):04d}-{int(end):04d}.mid"
        )

        segment_midi.instruments.append(instrument)
        segment_midi.write(segment_filename)
        set_bpm(segment_filename, bpm)
        if options.beats:
            add_beats_to_file(segment_filename, segment_filename, bpm)
        if options.metronome > 0:
            add_metronome(segment_filename, options.num_beats, options.metronome)
        modify_end_of_track(segment_filename, segment_length_s, bpm)

        new_files.append(segment_filename)

    return new_files


def process_files(
    args,
    pf_files: List,
    p_segments: str,
    p_augments: str,
    p_pictures: str,
    index: int,
) -> int:
    p = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=2,
        expand=True,
    )
    task_s = p.add_task(f"[SUBR{index:02d}] segmenting", total=len(pf_files))

    seg_opts = SegmentOptions(
        num_beats=args.num_beats,
        metronome=args.metronome,
        novelty=args.novelty,
        pic_dir=p_pictures,
        lead_window_beat_frac=args.lead_window_beat_frac,
        beats=args.beats,
    )
    aug_opts = AugmentOptions(
        num_shifts=args.num_beats,
        num_transposes=args.num_transposes,
        tempo_fold=args.tempo_fold,
    )

    # segment files
    segment_paths = []
    augment_paths = []
    with p:
        for pf_file in pf_files:
            # segment
            if args.segment:
                new_segments = segment_midi(
                    os.path.join(args.data_dir, pf_file),
                    p_segments,
                    seg_opts,
                )
            else:
                copy2(
                    os.path.join(args.data_dir, pf_file),
                    os.path.join(p_segments, pf_file),
                )
                new_segments = [os.path.join(args.data_dir, pf_file)]
            segment_paths.extend(new_segments)

            # augment
            if args.augment:
                augment_paths.extend(
                    augment_midi(
                        p,
                        os.path.splitext(pf_file)[0],
                        new_segments,
                        p_augments,
                        aug_opts,
                    )
                )

            p.update(task_s, advance=1)
    return index


def main(args, debug: bool = False) -> None:
    start_time = time.time()
    # set up filesystem
    if not os.path.exists(args.data_dir):
        console.log(f"no data dir found at {args.data_dir}")
        raise IsADirectoryError

    p_segments = os.path.join(args.out_dir, "segmented")
    p_augments = os.path.join(args.out_dir, "augmented")
    p_pictures = os.path.join(args.out_dir, "pictures")

    for dir in [p_segments, p_augments, p_pictures]:
        if not os.path.exists(dir):
            os.makedirs(dir)
            console.log(f"created new folder: '{dir}'")

    console.log(f"loading tracks from {args.data_dir}")
    tracks = []
    for root, _, files in os.walk(args.data_dir):
        for filename in files:
            if filename.endswith(".mid") or filename.endswith(".midi"):
                tracks.append(os.path.join(root, filename))
    tracks.sort()

    if args.tempo_fold:
        tmp_tracks = tracks.copy()
        for track in tmp_tracks:
            track_parts = basename(track).split("-")
            if int(track_parts[1]) <= args.tempo_fold_min:
                track_parts[1] = f"{int(int(track_parts[1]) * 2):03d}"
                new_track = os.path.join(
                    os.path.dirname(track), "-".join(track_parts) + ".mid"
                )
                if os.path.exists(new_track):
                    console.log(
                        f"not doubling {basename(track)} because it already exists"
                    )
                    continue
                tracks.append(new_track)
                copy2(track, new_track)
                console.log(
                    f"doubled tempo of {basename(track)} to {basename(new_track)}"
                )
            elif int(track_parts[1]) >= args.tempo_fold_max:
                track_parts[1] = f"{int(int(track_parts[1]) / 2):03d}"
                new_track = os.path.join(
                    os.path.dirname(track), "-".join(track_parts) + ".mid"
                )
                if os.path.exists(new_track):
                    console.log(
                        f"not halving {basename(track)} because it already exists"
                    )
                    continue
                tracks.append(new_track)
                copy2(track, new_track)
                console.log(
                    f"halved tempo of {basename(track)} to {basename(new_track)}"
                )

    if args.limit is not None:
        tracks = tracks[: args.limit]
    num_cores = os.cpu_count()
    if not num_cores:
        num_cores = 1
    split_keys = np.array_split(tracks, num_cores)

    console.log(f"segmenting {len(tracks)} tracks")
    if debug:
        for i, chunk in enumerate(split_keys):
            console.log(f"Processing chunk {i+1}/{len(split_keys)}")
            try:
                index = process_files(
                    args,
                    list(chunk),
                    p_segments,
                    p_augments,
                    p_pictures,
                    index=i,
                )
                console.log(f"Finished processing chunk {index}")
            except Exception as e:
                console.log(f"[red bold]Error processing chunk {i}: {e}")
                import traceback

                traceback.print_exc()
                # Decide whether to stop or continue
                # raise  # Re-raise the exception to stop execution
                console.log("[yellow]Continuing to next chunk...")
    else:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_files,
                    args,
                    list(chunk),
                    p_segments,
                    p_augments,
                    p_pictures,
                    index=i,
                ): chunk
                for i, chunk in enumerate(split_keys)
            }

            for future in as_completed(futures):
                index = future.result()
                console.log(f"subprocess {index} returned")

    zip_path = os.path.join("data", "datasets", f"{args.dataset_name}_segments.zip")
    console.log(f"compressing to zipfile '{zip_path}'")
    n_files = 0
    if args.segment:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(p_segments):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, p_segments)
                    zipf.write(file_path, arcname)
                    n_files += 1
    if args.augment:
        with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(p_augments):
                random_file = random.randint(0, len(files) - 1)
                for i, file in enumerate(files):
                    file_path = os.path.join(root, file)
                    if i == random_file:
                        console.log(
                            f"randomly selected file {i}: {basename(file_path)}"
                        )
                        mido.MidiFile(file_path).print_tracks()
                    arcname = os.path.relpath(file_path, p_augments)
                    zipf.write(file_path, arcname)
                    n_files += 1

    console.log(
        f"[green bold]segmentation complete, {n_files} files generated in {time.time() - start_time:.03f} s"
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset builder arguments")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="path to config file"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="run in debug mode (sequential execution)",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    console.log(config)
    main(config, args.debug)
