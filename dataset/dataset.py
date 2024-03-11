import os
from pathlib import Path
from mido import MidiFile
import pretty_midi
import numpy as np

from utils import console
from utils.midi import set_tempo, semitone_shift


def segment_midi(input_file_path: str, params):
    """do the segmentation"""
    p = "[red]datast[/red]:"
    target_tempo = int(os.path.basename(input_file_path).split("-")[1])
    set_tempo(input_file_path, target_tempo)

    # remove "-t" from filename
    filename = Path(input_file_path).stem
    filename_components = filename.split("-")
    filename = f"{filename_components[0]}-{int(np.round(float(filename_components[1]))):03d}-{filename_components[2]}"

    # figure out timings
    midi_pm = pretty_midi.PrettyMIDI(input_file_path)
    total_length = midi_pm.get_end_time()
    segment_length = 60 * params.n__num_beats / target_tempo  # in seconds
    num_segments_float = total_length / segment_length
    num_segments = int(np.round(num_segments_float))

    # pprint([total_length, segment_length, num_segments_float, num_segments, init_bpm])

    console.log(
        f"{p}breaking '{filename}' ({total_length:.03f} s at {target_tempo} bpm) into {num_segments:03d} segments of {segment_length:.03f} s"
    )

    new_files = 0
    for n in range(num_segments):
        start = n * segment_length
        end = start + segment_length
        # print(f"\tsplitting from {start:07.03f} s to {end:07.03f} s")
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
                    end=min(note.end, end) - start,
                )
                instrument.notes.append(new_note)

        # write out
        segment_midi.instruments.append(instrument)
        segment_filename = os.path.join(
            params.output_dir, f"{filename}_{int(start):04d}-{int(end):04d}.mid"
        )
        segment_midi.write(segment_filename)

        if params.strip_tempo:
            midi_md = MidiFile(segment_filename)
            for track in midi_md.tracks:
                for message in track:
                    if message.type == "set_tempo":
                        track.remove(message)
            midi_md.save(segment_filename)
        else:
            set_tempo(segment_filename, target_tempo)

        if params.do_shift:
            shifts = 12
            num_shifts = semitone_shift(segment_filename, params.output_dir, shifts)
            if num_shifts != shifts:
                console.log(
                    f"{p} [yellow]only shifted {num_shifts} times (expected {shifts})"
                )
            new_files += num_shifts
        else:
            new_files += 1

    return new_files
