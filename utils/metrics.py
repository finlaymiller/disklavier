import os
import math
import mido
from pretty_midi import PrettyMIDI
import numpy as np

from utils.midi import blur_pr

from typing import Dict, List

TARGET_VEL = 80

def all_properties(midi_path: str, filename: str, config) -> Dict:
    """"""
    midi = PrettyMIDI(midi_path)
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
        "energy": [0.0] * num_bins,
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
    properties["energy"] = energy(midi_path)
    properties["pr_blur"] = blur_pr(midi, False)
    properties["pr_blur_c"] = blur_pr(midi)

    return properties


############################  individual properties  ##########################

def get_avg_vel(midi_file_path):
    midi = mido.MidiFile(midi_file_path)
    total_velocity = 0
    note_on_count = 0

    for track in midi.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                total_velocity += msg.velocity
                note_on_count += 1

    if note_on_count == 0:
        return 0
    else:
        return total_velocity / note_on_count


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


def energy(midi_path) -> List[int]:
    # new_path = scale_vels(midi_path, "outputs/graveyard")
    midi = PrettyMIDI(midi_path)
    envelope_length = 10  # seconds
    pitch_histogram = {i: 0 for i in range(12)}

    for instrument in midi.instruments:
        for note in instrument.notes:
            note_duration = note.end - note.start
            energy = note.velocity * envelope_length / 2

            if note_duration < envelope_length:
                energy = energy - (
                    (envelope_length - note_duration) * note.velocity / 2
                )

            pitch_histogram[note.pitch % 12] += energy

    return list(pitch_histogram.values())


def energy_old(
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

# helper functions


def scale_vels(midi_file_path, output_path):
    midi = PrettyMIDI(midi_file_path)

    avg_vel = get_avg_vel(midi_file_path)
    scaling_factor = TARGET_VEL / avg_vel

    if scaling_factor < 0.85 or scaling_factor > 1.15:
        return

    pre_min, pre_max = 128, 1
    post_min, post_max = 128, 1

    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.velocity < pre_min:
                pre_min = note.velocity
            if note.velocity > pre_max:
                pre_max = note.velocity

            new_velocity = int(note.velocity * scaling_factor)
            new_velocity = max(1, min(new_velocity, 127))

            if new_velocity < post_min:
                post_min = new_velocity
            if new_velocity > post_max:
                post_max = new_velocity

            note.velocity = new_velocity

    midi.write(output_path)
