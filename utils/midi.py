import os
from pretty_midi import PrettyMIDI, Instrument, Note
import mido
from mido import MidiFile, MetaMessage
import numpy as np
from pathlib import Path
import cv2
from scipy.signal import convolve2d

from utils import console

from typing import Tuple, List


def quantize_midi(filename, sections_per_beat) -> PrettyMIDI:
    """
    Quantizes a MIDI file into sections_per_beat sections per beat.

    Parameters:
        midi_file_path (str): Path to the MIDI file.
        sections_per_beat (int): Number of quantization sections per beat.

    Returns:
        pretty_midi.PrettyMIDI: A quantized PrettyMIDI object.
    """
    midi_data = PrettyMIDI(filename)
    bpm = int(filename.split("-")[1])
    section_duration = 60.0 / bpm / sections_per_beat

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            note.start = round(note.start / section_duration) * section_duration
            note.end = round(note.end / section_duration) * section_duration

    return midi_data


def blur_pr(midi: PrettyMIDI, do_center: bool = True, delta_max: int = 55):
    pr = midi.get_piano_roll()
    if do_center:
        pr = strip_and_pad(pr, delta_max)
    filter = np.full((3, 3), 1 / 9)
    width = 64
    height = int(pr.shape[0] / 4)
    small_img = cv2.resize(pr, (width, height), interpolation=cv2.INTER_AREA)
    blurred_img = convolve2d(small_img, filter)

    return np.asarray(blurred_img).flatten().tolist()


def strip_and_pad(pr, h_max):
    trimmed_pr = trim_piano_roll(pr)

    current_h = trimmed_pr.shape[0]
    total_padding = max(h_max + 2 - current_h, 0)  # Ensure non-negative padding

    top_padding = total_padding // 2
    bottom_padding = total_padding - top_padding

    padded_pr = np.pad(
        trimmed_pr,
        ((top_padding, bottom_padding), (0, 0)),
        mode="constant",  # type: ignore
        constant_values=0,
    )

    return padded_pr


def trim_piano_roll(piano_roll, min=None, max=None):
    """
    Trims the piano roll by removing rows above the highest note and below the
    lowest note.

    Parameters:
        piano_roll (np.array): A 2D NumPy array representing the piano roll.
        min (int): The note to remove everything below. If none is provided, the
        lowest note in the roll will be used
        max (int): The note to remove everything above. If none is provided, the
        highest note in the roll will be used

    Returns:
        np.array: The trimmed piano roll.
    """
    non_zero_rows = np.where(np.any(piano_roll > 0, axis=1))[0]

    if non_zero_rows.size == 0:
        return piano_roll

    lowest_note = non_zero_rows.min() if min is None else min
    highest_note = non_zero_rows.max() if max is None else max

    trimmed_piano_roll = piano_roll[lowest_note : highest_note + 1, :]

    return trimmed_piano_roll


def lstrip_midi(mid: PrettyMIDI):
    """Modify MIDI object so that the first note is at 0.0s."""
    for instrument in mid.instruments:
        if not instrument.notes:
            continue
        first_note_start = instrument.notes[0].start
        for note in instrument.notes:
            note.start -= first_note_start
            note.end -= first_note_start
    return mid


def stretch_midi_file(
    midi: MidiFile, new_duration_seconds: float, caller: str = "[cyan]utils[/cyan] : "
) -> MidiFile:
    """"""
    console.log(
        f"{caller} rescaling file from {midi.length:.02f} s to {new_duration_seconds:.02f} s (x {new_duration_seconds / midi.length:.03f})"
    )
    # Calculate stretch factor based on the original duration
    stretch_factor = new_duration_seconds / midi.length

    # Scale the time attribute of each message by the stretch factor
    for track in midi.tracks:
        for msg in track:
            msg.time = int(msg.time * stretch_factor)

    return midi


def set_tempo(input_file_path, target_tempo) -> None:
    """"""
    mid = MidiFile(input_file_path)
    tempo = mido.bpm2tempo(target_tempo)
    mid.tracks[0].insert(0, MetaMessage("set_tempo", tempo=tempo, time=0))
    mid.save(input_file_path)


def get_tempo(midi_file_path) -> float:
    """"""
    midi_file = MidiFile(midi_file_path)

    # Default MIDI tempo is 120 BPM, which equals 500000 microseconds per beat
    tempo = 500000  # Default tempo

    for track in midi_file.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        if tempo != 500000:
            break

    return mido.tempo2bpm(tempo)


def get_note_min_max(input_file_path) -> Tuple[int, int]:
    """returns the values of the highest and lowest notes in a midi file"""
    mid = MidiFile(input_file_path)

    lowest_note = 127
    highest_note = 0

    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta and msg.type in ["note_on", "note_off"]:
                # Update lowest and highest note if this is a note_on message
                if msg.velocity > 0:  # Considering note_on messages only
                    lowest_note = min(lowest_note, msg.note)
                    highest_note = max(highest_note, msg.note)

    return (lowest_note, highest_note)


def transpose_midi(input_file_path: str, output_file_path: str, semitones: int) -> None:
    """
    Transposes all the notes in a MIDI file by a specified number of semitones.

    Args:
    - input_file_path: Path to the input MIDI file.
    - output_file_path: Path where the transposed MIDI file will be saved.
    - semitones: Number of semitones to transpose the notes. Positive for up, negative for down.
    """

    midi = PrettyMIDI(input_file_path)
    for instrument in midi.instruments:
        # Don't want to shift drum notes
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += semitones
    midi.write(output_file_path)


def semitone_transpose(
    midi_path: str, output_dir: str, num_iterations: int = 1
) -> List[str]:
    """vertically shift a matrix
    chatgpt
    """
    new_filename = Path(midi_path).stem.split("_")
    new_filename = f"{new_filename[0]}_{new_filename[1]}"
    lowest_note, highest_note = get_note_min_max(midi_path)
    max_up = 108 - highest_note  # TODO double-check this IRL
    max_down = lowest_note

    # zipper up & down
    up = 1
    down = -1
    new_files = []
    for i in range(num_iterations):
        up_filename = f"{new_filename}_u{up:02d}.mid"
        up_filepath = os.path.join(output_dir, up_filename)
        down_filename = f"{new_filename}_d{abs(down):02d}.mid"
        down_filepath = os.path.join(output_dir, down_filename)

        if i % 2 == 0:
            if (
                up > max_up
            ):  # If exceeding max_up, adjust by switching to down immediately
                transpose_midi(midi_path, down_filepath, down)
                new_files.append(down_filepath)
                down -= 1
            else:
                transpose_midi(midi_path, up_filepath, up)
                new_files.append(up_filepath)
                up += 1
        else:
            if (
                abs(down) > max_down
            ):  # If exceeding max_down, adjust by switching to up immediately
                transpose_midi(midi_path, up_filepath, up)
                new_files.append(up_filepath)
                up += 1
            else:
                transpose_midi(midi_path, down_filepath, down)
                new_files.append(down_filepath)
                down -= 1

    return new_files


def get_velocities(midi_data: PrettyMIDI) -> List:
    """
    Analyzes MIDI file velocities.

    Args:
        midi_data (pretty_midi.PrettyMIDI): The MIDI file data.

    Returns:
        (list, list): A 2-element list containing the lowest and highest note velocities,
                    and a list of counts of note velocities broken into 10 bins.
    """
    velocities = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            velocities.append(note.velocity)

    if velocities:
        lowest_velocity = min(velocities)
        highest_velocity = max(velocities)
    else:
        lowest_velocity, highest_velocity = 0, 0

    bin_counts, _ = np.histogram(velocities, bins=10, range=(1, 127))

    return [[lowest_velocity, highest_velocity], bin_counts.tolist()]


def augment_recording(path: str, storage_dir: str, tempo: int = 70):
    midi = PrettyMIDI(path)
    midi_first_half = PrettyMIDI(initial_tempo=tempo)
    midi_second_half = PrettyMIDI(initial_tempo=tempo)
    midi_doubled = PrettyMIDI(path)

    length = midi.get_end_time()
    halfway_point = length / 2

    for i, instrument in enumerate(midi.instruments):
        inst_fh = Instrument(program=instrument.program, name=instrument.name)
        inst_sh = Instrument(program=instrument.program, name=instrument.name)

        for note in instrument.notes:
            if note.start < halfway_point:
                inst_fh.notes.append(note)
            else:
                new_note = Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - halfway_point,
                    end=note.end - halfway_point,
                )
                inst_sh.notes.append(new_note)

            shifted_note = Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start + length,
                end=note.end + length,
            )
            midi_doubled.instruments[i].notes.append(shifted_note)

        midi_first_half.instruments.append(inst_fh)
        midi_second_half.instruments.append(inst_sh)

    basename = Path(path).stem
    midi_first_half.write(os.path.join(storage_dir, f"{basename}_fh.mid"))
    midi_second_half.write(os.path.join(storage_dir, f"{basename}_sh.mid"))
    midi_doubled.write(os.path.join(storage_dir, f"{basename}_db.mid"))

    results = [
        os.path.join(storage_dir, f"{basename}_fh.mid"),
        os.path.join(storage_dir, f"{basename}_sh.mid"),
        os.path.join(storage_dir, f"{basename}_db.mid"),
    ]


    return results
