import os
from pretty_midi import PrettyMIDI, Instrument, Note
import mido
from mido import MidiFile, MetaMessage
import numpy as np
from pathlib import Path
import cv2
from scipy.signal import convolve2d

from utils import console

from typing import Dict, Tuple, List


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


def blur_pr(midi: PrettyMIDI, do_center: bool = True, delta_max: int = 55) -> List:
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


def set_tempo(input_file_path: str, bpm: int) -> None:
    """Sets the tempo of a MIDI file to a specified target tempo.

    Args:
        input_file_path (str): The path to the MIDI file whose tempo is to be adjusted.
        bpm (int): The target tempo in beats per minute (BPM) to set for the MIDI file.

    This function modifies the specified MIDI file by inserting a tempo change meta-message at the beginning of the first track, effectively setting the entire file to the specified tempo. The change is saved to the same file path, overwriting the original MIDI file.
    """
    midi = MidiFile(input_file_path)
    midi.tracks[0].insert(0, MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    midi.save(input_file_path)


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


def transform(file_path: str, out_dir: str, tempo: int, transformations: Dict, num_beats: int = 8) -> str:
    new_filename = f"{Path(file_path).stem}_t{transformations["transpose"]:02d}s{transformations["shift"]:02d}.mid"
    out_path = os.path.join(out_dir, new_filename)
    MidiFile(file_path).save(out_path) # in case transpose is 0

    if transformations["transpose"] != 0:
        t_midi = PrettyMIDI(initial_tempo=tempo)

        for instrument in PrettyMIDI(out_path).instruments:
            transposed_instrument = Instrument(program=instrument.program, name=new_filename[:-4])

            for note in instrument.notes:
                transposed_instrument.notes.append(
                    Note(
                        velocity=note.velocity,
                        pitch=note.pitch + int(transformations["transpose"]),
                        start=note.start,
                        end=note.end,
                    )
                )

            t_midi.instruments.append(transposed_instrument)

        t_midi.write(out_path)

    if transformations["shift"] != 0:
        s_midi = PrettyMIDI(initial_tempo=tempo)
        seconds_per_beat = 60 / tempo
        shift_seconds = transformations["shift"] * seconds_per_beat
        loop_point = (num_beats + 1) * seconds_per_beat

        for instrument in PrettyMIDI(out_path).instruments:
            shifted_instrument = Instrument(
                program=instrument.program, name=new_filename[:-4]
            )
            for note in instrument.notes:
                dur = note.end - note.start
                shifted_start = (note.start + shift_seconds) % loop_point
                shifted_end = shifted_start + dur

                if note.start + shift_seconds >= loop_point:
                    shifted_start += seconds_per_beat
                    shifted_end += seconds_per_beat

                shifted_instrument.notes.append(
                    Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=shifted_start,
                        end=shifted_end
                    )
                )

            s_midi.instruments.append(shifted_instrument)

        s_midi.write(out_path)

    change_tempo(out_path, tempo)

    return out_path


def get_tempo(filename: str) -> int:
    return int(filename.split('-')[1])


def split_filename(filename: str, split_track=False) -> List[str]:
    """Splits a filename into its components based on underscores.

    Args:
        filename (str): The filename to be split, expected to be in the format 
        'base_segment_transformations'.
        split_track (bool, optional): Whether to split the track part. Defaults to False.

    Returns:
        List[str]: A list containing the split parts of the filename. If split_track is False,
        the list will contain the basename, trans, and shift. If split_track is True,
        the list will contain basename, segment, trans, and shift.
    """

    b, s, t = filename.split("_")
    trans = t[1:3]
    shift = t[4:6]

    if split_track:
        return [b, s, trans, shift]
        
    basename = "_".join([b, s])

    return [basename, trans, shift]


def insert_transformations(filename: str, transformations: List[int]=[0,0]) -> str:
    return f"{filename[:-4]}_t{transformations[0]:02d}s{transformations[1]:02d}{filename[-4:]}"


def change_tempo(file_path: str, tempo: int):
    midi = mido.MidiFile(file_path)
    new_tempo = mido.bpm2tempo(tempo)
    new_message = mido.MetaMessage("set_tempo", tempo=new_tempo, time=0)
    tempo_added = False

    for track in midi.tracks:
        # remove existing set_tempo messages
        for msg in track:
            if msg.type == "set_tempo":
                track.remove(msg)

        # add new set_tempo message to the first track
        if not tempo_added:
            track.insert(0, new_message)
            tempo_added = True

    # if no tracks had a set_tempo message and no new one was added, add a new track with the tempo message
    if not tempo_added:
        new_track = mido.MidiTrack()
        new_track.append(new_message)
        midi.tracks.append(new_track)

    midi.save(file_path)


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


def augment_recording(path: str, storage_dir: str, tempo: int) -> List[str]:
    midi = PrettyMIDI(path)
    midi_first_half = PrettyMIDI(initial_tempo=tempo)
    midi_second_half = PrettyMIDI(initial_tempo=tempo)
    midi_doubled = PrettyMIDI(path)

    basename = Path(path).stem
    first_half_path = os.path.join(storage_dir, f"{basename}_fh.mid")
    second_half_path = os.path.join(storage_dir, f"{basename}_sh.mid")
    doubled_path = os.path.join(storage_dir, f"{basename}_db.mid")

    length = midi.get_end_time()
    halfway_point = length / 2

    for i, instrument in enumerate(midi.instruments):
        console.log(f"augmenting {i}")
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

            midi_doubled.instruments[i].notes.append(
                Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + length,
                    end=note.end + length,
                )
            )

        midi_first_half.instruments.append(inst_fh)
        midi_second_half.instruments.append(inst_sh)

    midi_first_half.write(first_half_path)
    midi_second_half.write(second_half_path)
    midi_doubled.write(doubled_path)

    return [first_half_path, second_half_path, doubled_path]


def b2t(time_beats: float, tempo: int) -> float:
    """Converts a time in beats to the equivalent time in seconds.

    Args:
        time_beats (float): The time in beats.
        tempo (int): The tempo in beats per minute.

    Returns:
        float: The equivalent time in seconds.

    Example:
        >>> b2t(2.0, 120)
        1.0
    """

    return time_beats * 60 / tempo


def calc_beats(tempo: int, start_time_seconds: float, end_time_seconds: float):
    """Compute the timing for each beat based on the given tempo, starting
    from 'start_time_seconds' until 'end_time_seconds'. Each beat's timing is
    calculated and appended to a list, which is then returned. The calculation
    assumes a constant tempo throughout the specified time range, and that
    'start_time_seconds' is a valid first beat.

    Args:
        tempo (int): The tempo of the music in beats per minute (BPM).
        start_time_seconds (float): The start time of the time range in seconds.
        end_time_seconds (float): The end time of the time range in seconds.

    Returns:
        list[float]: A list containing the timestamps (in seconds) where each
                        beat occurs within the specified time range.
    """
    beat_duration = 60 / tempo
    current_time = start_time_seconds
    beat_times = []

    while current_time <= end_time_seconds:
        beat_times.append(current_time)
        current_time += beat_duration

    return beat_times


def get_active_beats(beats, start):
    """Finds and returns the number immediately before a given start value and
    all numbers following it in the list.

    Args:
        beats (list[float]): A list of numeric values representing beats.
        start (float): The start value to search for in the list.

    Returns:
        list[float]: A list containing the value immediately before the start
                        value (if any) and all subsequent values. Returns an
                        empty list if the start value is greater than all
                        elements in the list.
    """
    index = next((i for i, beat in enumerate(beats) if beat > start), None)

    if index is not None and index > 0:
        return beats[index - 1 :]

    return []


def text_to_midi(filepath: str, tempo: int) -> None:
    if not os.path.exists(filepath):
        console.log(f"[red]unable to find notes file at[/red] '{filepath}'")
        return

    with open(filepath, "r") as f:
        new_midi = mido.MidiFile()
        new_track = mido.MidiTrack()
        for line in f.readlines():
            # console.log(f"[dark_sea_green2]midi[/dark_sea_green2]  : {line}")
            note_properties = line.split(" ")
