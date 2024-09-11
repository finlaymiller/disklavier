import os
from mido import MidiFile, MetaMessage, bpm2tempo, second2tick
from itertools import product
import pretty_midi
import numpy as np

from utils.midi import set_tempo, semitone_transpose, transform

from typing import List


def augment_midi(
    p, filename: str, new_segments: List[str], output_path: str
) -> List[str]:
    augmented_files = []

    task_a = p.add_task(f"augmenting {filename}", total=len(new_segments) * 96)

    with p:
        for segment_filename in new_segments:
            transformations = [
                {"transpose": t, "shift": s} for t, s in product(range(12), range(8))
            ]
            for transformation in transformations:
                augmented_files.append(
                    transform(
                        segment_filename,
                        output_path,
                        int(filename.split("-")[1]),
                        transformation,
                    )
                )
                p.update(task_a, advance=1)

        p.remove_task(task_a)
    return augmented_files


def segment_midi(
    midi_file_path: str,
    output_dir: str,
    num_beats: int = 8,
) -> List[str]:
    filename = os.path.basename(midi_file_path)[:-4]
    target_tempo = int(filename.split("-")[1])
    set_tempo(midi_file_path, target_tempo)
    midi_pm = pretty_midi.PrettyMIDI(midi_file_path)
from utils.midi import set_tempo, semitone_transpose

from utils import console


def segment_midi(input_file_path: str, params):
    """Segments a MIDI file into smaller parts based on tempo and beats per
    segment.

    Args:
        input_file_path (str): Path to the input MIDI file.
        params: A config object containing segmentation settings, such as:
            n__num_beats (int): Number of beats per segment.
            output_dir (str): Directory where segmented MIDI files will be stored.
            do_shift (int): Indicates how many semitones to transpose the segments.
            strip_tempo (bool): If True, strip existing tempo information from segments.

    Returns:
        int: The number of new files created through segmentation and optional transposition.

    This function reads a MIDI file, extracts its tempo from the filename, and segments it into smaller MIDI files each containing a specified number of beats. Optionally, it can transpose the segments by a specified number of semitones. The function adjusts tempo and track end as necessary for each segment.
    """
    p = "[red]datast[/red]:"
    target_tempo = int(os.path.basename(input_file_path).split("-")[1])
    set_tempo(input_file_path, target_tempo)

    # calculate timings
    midi_pm = pretty_midi.PrettyMIDI(input_file_path)
    total_length = midi_pm.get_end_time()
    segment_length = num_beats * 60 / target_tempo  # in seconds
    num_segments = int(np.round(total_length / segment_length))
    eighth_beat = segment_length / num_beats / 8  # eighth of a beat

    # print(
    #     f"\tbreaking '{filename}' ({total_length:.03f} s at {target_tempo} bpm) into {num_segments:03d} segments of {segment_length:.03f}s\n\t(pre window is {eighth_beat:.03f} s)"
    # )

    new_files = []
    for n in list(range(num_segments)):
        start = n * segment_length
        end = start + segment_length - eighth_beat
        if n > 0:
            start -= eighth_beat

        # print(f"\t{n:03d} splitting from {start:08.03f} s to {end:07.03f} s")

        segment_midi = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)
        instrument = pretty_midi.Instrument(
            program=midi_pm.instruments[0].program,
            name=f"{filename}_{int(start):04d}-{int(end):04d}",
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
                instrument.notes.append(new_note)

        # pad front of track to full bar for easier playback
        if n > 0:
            for note in instrument.notes:
                note.start += eighth_beat * 7
                note.end += eighth_beat * 7

        # write out
        segment_filename = os.path.join(
            output_dir, f"{filename}_{int(start):04d}-{int(end):04d}.mid"
        )

        segment_midi.instruments.append(instrument)
        segment_midi.write(segment_filename)
        set_tempo(segment_filename, target_tempo)
        modify_end_of_track(segment_filename, segment_length, target_tempo)

        new_files.append(segment_filename)

    return new_files


def modify_end_of_track(midi_file_path, new_end_time, tempo):
    midi = MidiFile(midi_file_path)
    new_end_time_t = second2tick(new_end_time, 220, bpm2tempo(tempo))

    for i, track in enumerate(midi.tracks):
        total_time_t = 0
        # Remove existing 'end_of_track' messages and calculate last note time
        for msg in track:
            if msg.type == "note_on":
                total_time_t += msg.time
            if msg.type == "end_of_track":
                track.remove(msg)
                # Add a new 'end_of_track' message at the calculated offset time
                offset = (
                    new_end_time_t - total_time_t
                    if new_end_time_t > total_time_t
                    else 0
                )
                track.append(MetaMessage("end_of_track", time=offset))

    # Save the modified MIDI file
    os.remove(midi_file_path)
    midi.save(midi_file_path)
