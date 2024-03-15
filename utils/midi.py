import os
import math
from pretty_midi import PrettyMIDI
import mido
from mido import MidiFile, MetaMessage
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import console

from typing import Dict, Tuple, List

DARK = True


#################################  plotting  ##################################
def plot_images(
    images,
    titles,
    save_path,
    shape=None,
    main_title=None,
    set_axis: str = "off",
) -> None:
    """Plot images vertically"""
    num_images = len(images)

    if shape is None:
        shape = [num_images, 1]

    plt.style.use("dark_background")
    plt.figure(figsize=(12, 12))

    if main_title:
        plt.suptitle(main_title)
    for num_plot in range(num_images):
        plt.subplot(shape[0], shape[1], num_plot + 1)
        plt.imshow(
            np.squeeze(images[num_plot]),
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        plt.title(titles[num_plot])
        plt.axis(set_axis)

    plt.savefig(save_path)


def draw_histogram(histogram, title="Pitch Histogram") -> None:
    if DARK:
        plt.style.use("dark_background")
    plt.bar(range(12), histogram)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(
        range(12), ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    )
    plt.show()


def draw_piano_roll(piano_roll, fs=100, title="Piano Roll") -> None:
    if DARK:
        plt.style.use("dark_background")
    plt.figure(figsize=(12, 8))
    plt.imshow(
        piano_roll, aspect="auto", origin="lower", cmap="magma", interpolation="nearest"
    )
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("MIDI Note Number")
    plt.colorbar()

    tick_spacing = 1
    ticks = np.arange(0, len(piano_roll.T) / fs, tick_spacing)
    plt.xticks(ticks * fs, labels=[f"{tick:.1f}" for tick in ticks])

    plt.show()


def plot_piano_roll_and_pitch_histogram(input_path: str, output_dir: str) -> None:
    """plot the piano roll and pitch histogram of a midi file side by side

    Parameters:
        input_path (str): the file to read
        output_dir (str): the folder to write the image out to

    Returns:
        None
    """
    if DARK:
        plt.style.use("dark_background")

    midi_data = PrettyMIDI(input_path)
    piano_roll = midi_data.get_piano_roll(fs=100)
    pitches = midi_data.get_pitch_class_histogram()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(piano_roll, aspect="auto", origin="lower", cmap="magma")
    plt.title("Piano Roll")
    plt.xlabel("Time")
    plt.ylabel("MIDI Note Number")

    plt.subplot(1, 2, 2)
    plt.bar(
        np.arange(12),
        pitches,
        tick_label=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
    )
    plt.title("Pitch Histogram")
    plt.xlabel("Pitch Class")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{Path(input_path).stem}_ph.png"))


################################  heuristics  #################################
################################  all in one  #################################
# TODO add manually-calculated "valid tempo range"


def all_properties(midi: PrettyMIDI, filename: str, config) -> Dict:
    """"""
    # properties from filename
    fn_segments_us = filename.split("_")
    tempo = fn_segments_us[0].split("-")[1]
    segment_start, segment_end = fn_segments_us[1].split("-")
    pitch_shift = fn_segments_us[-1] if len(fn_segments_us) > 2 else "none"

    # build dict
    num_bins = int(math.ceil(midi.get_end_time() / config.bin_length))
    properties = {
        "pitch_histogram": list(
            midi.get_pitch_class_histogram(
                use_duration=config.ph_weight_dur, use_velocity=config.ph_weight_vel
            )
        ),
        "tempo": tempo,
        "segment_start_s": segment_start,
        "segment_end_s": segment_end,
        "pitch_shift": pitch_shift,
        "file_len": midi.get_end_time(),
        "note_count": sum(len(instrument.notes) for instrument in midi.instruments),
        "velocities": [{"total_velocity": 0, "count": 0} for _ in range(num_bins)],
        "lengths": [0.0] * num_bins,
        "energies": [0.0] * num_bins,
        "simultaneous_counts": [0] * num_bins,
    }

    # properties that are calculated from notes
    for instrument in midi.instruments:
        for note in instrument.notes:
            start_bin = int(note.start // config.bin_length)
            end_bin = int(note.end // config.bin_length)
            properties["lengths"].append(note.end - note.start)

            for bin in range(start_bin, min(end_bin + 1, num_bins)):
                properties["velocities"][bin]["total_velocity"] += note.velocity
                properties["velocities"][bin]["count"] += 1
                properties["simultaneous_counts"][bin] += 1

    properties["lengths"] = sum(properties["lengths"]) / len(properties["lengths"])

    # properties that are calculated from other properties
    properties["energies"] = [
        config["w1"] * (vel["total_velocity"] / vel["count"])
        + config["w2"] * properties["lengths"]
        for vel in properties["velocities"]
        if vel["count"] > 0
    ]

    return properties


############################  individual properties  ##########################


def average_note_length(midi: PrettyMIDI) -> float:
    """
    Calculate the average length of notes in a MIDI file.

    Parameters:
        midi (PrettyMIDI): The prettyMIDI container object for the MIDI file.

    Returns:
        float: The average length of notes in the MIDI file in seconds. If the file
        has no notes, it returns 0.
    """
    note_lengths = []

    for instrument in midi.instruments:
        for note in instrument.notes:
            note_lengths.append(note.end - note.start)

    if note_lengths:
        average_length = sum(note_lengths) / len(note_lengths)
    else:
        average_length = 0.0

    return average_length


def total_number_of_notes(midi: PrettyMIDI) -> int:
    """
    Calculate the total number of notes in a MIDI file.

    Parameters:
        midi (PrettyMIDI): The prettyMIDI container object for the MIDI file.

    Returns:
        int: The total number of notes in the MIDI file.
    """

    return sum(len(instrument.notes) for instrument in midi.instruments)


def total_velocity(midi: PrettyMIDI, bin_length=None) -> list[dict[str, int]]:
    """
    Calculate the total velocity of all notes for each time bin in a MIDI file.

    Parameters:
        midi (PrettyMIDI): The prettyMIDI container object for the MIDI file.
        bin_length (float): The length of time each bin should occupy. (default 1s)

    Returns:
        list: A list whose values are the total velocity and number of notes in that bin.
    """
    if bin_length == None:
        bin_length = midi.get_end_time()
    num_bins = int(math.ceil(midi.get_end_time() / bin_length))
    bin_velocities = [{"total_velocity": 0, "count": 0} for _ in range(num_bins)]

    for instrument in midi.instruments:
        for note in instrument.notes:
            start_bin = int(note.start // bin_length)
            end_bin = int(note.end // bin_length)

            for bin in range(start_bin, min(end_bin + 1, num_bins)):
                bin_velocities[bin]["total_velocity"] += note.velocity
                bin_velocities[bin]["count"] += 1

    return bin_velocities


def simultaneous_notes(midi: PrettyMIDI, bin_length=None) -> list[int]:
    """
    Calculate the number of simultaneous notes being played each second in a MIDI file.

    Parameters:
        midi (PrettyMIDI): The prettyMIDI container object for the MIDI file.
        bin_length (float): The length of time each bin should occupy. (default 1s)

    Returns:
        list: A list whose values are the number of simultaneous notes being played in that second.
    """
    if bin_length == None:
        bin_length = midi.get_end_time()
    num_bins = int(math.ceil(midi.get_end_time() / bin_length))
    simultaneous_notes_counts = [0] * num_bins

    for instrument in midi.instruments:
        for note in instrument.notes:
            start_bin = int(note.start // bin_length)
            end_bin = int(note.end // bin_length)

            for bin in range(start_bin, min(end_bin + 1, num_bins)):
                simultaneous_notes_counts[bin] += 1

    return simultaneous_notes_counts


def energy(
    midi: PrettyMIDI,
    w1: float = 0.5,
    w2: float = 0.5,
    bin_length=None,
) -> list[float]:
    """
    Calculate a weighted average of note lengths and average velocities
    """
    if bin_length == None:
        bin_length = midi.get_end_time()
    num_bins = int(math.ceil(midi.get_end_time() / bin_length))
    energies = [0.0] * num_bins
    v = total_velocity(midi, bin_length)
    l = average_note_length(midi)

    for instrument in midi.instruments:
        for note in instrument.notes:
            start_bin = int(note.start // bin_length)
            end_bin = int(note.end // bin_length)

            for bin in range(start_bin, min(end_bin + 1, num_bins)):
                energies[bin] += (
                    w1 * (v[bin]["total_velocity"] / v[bin]["count"]) + w2 * l
                )

    return energies


def norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


#################################  other  ###################################
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
        f"{caller} rescaling file from {midi.length:.02f} s to {new_duration_seconds:.02f} s (x{new_duration_seconds / midi.length:.03f})"
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


def scale_vels(midi_data: PrettyMIDI, scale_factor: float = 1.0) -> PrettyMIDI:
    """
    Scales the velocities of all notes in a MIDI file by a set amount, with the results capped at 127.

    Args:
        midi_data (pretty_midi.PrettyMIDI): The MIDI file data to be modified.
        scale_factor (float): The factor by which to scale the velocities. A value of 1.0 leaves velocities unchanged,
                            0.5 halves them, etc.
    """
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            new_velocity = int(note.velocity * scale_factor)
            note.velocity = min(new_velocity, 127)

    return midi_data
