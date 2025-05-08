import os
import mido
import numpy as np
import pretty_midi
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import zoom

from typing import Dict, Tuple

from . import basename, console

from utils.constants import TICKS_PER_BEAT


def get_bpm(file_path: str) -> int:
    """
    Extracts the bpm from a MIDI file. First, extraction is attempted from the filename, then from the file itself. If there are multiple `set_tempo` messages,
    the last one is used. A time signature of 4/4 is assumed.

    Parameters
    ----------
    file_path : str
        Path to the MIDI file.

    Returns
    -------
    int
        The BPM. Default is 120 BPM if not explicitly set.
    """
    try:
        tempo = int(os.path.basename(file_path).split("-")[1])
    except ValueError:
        tempo = 120
        midi_file = mido.MidiFile(file_path)
        for track in midi_file.tracks:
            for message in track:
                if message.type == "set_tempo":
                    tempo = mido.tempo2bpm(message.tempo)

    return tempo


def set_bpm(file_path: str, bpm: int) -> bool:
    """
    Sets the tempo of a MIDI file to a specified target tempo, provided as a bpm.

    Parameters
    ----------
    file_path : str
        The path to the MIDI file whose tempo is to be adjusted.
    bpm : int
        The target tempo in beats per minute (BPM) to set for the MIDI file.

    Returns
    -------
    bool
        True if the tempo was successfully set and matches the target bpm, False otherwise.
    """
    midi = mido.MidiFile(file_path)

    # remove existing set_tempo messages from all tracks
    for track in midi.tracks:
        tempo_indices = []
        for i, msg in enumerate(track):
            if msg.type == "set_tempo":
                tempo_indices.append(i)

        # remove in reverse order to avoid index shifting
        for index in sorted(tempo_indices, reverse=True):
            del track[index]

    # insert new tempo message at the beginning of the first track
    midi.tracks[0].insert(
        0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
    )
    midi.save(file_path)

    return get_bpm(file_path) == bpm


def change_tempo(in_path: str, out_path: str, tempo: int):
    midi = mido.MidiFile(in_path)
    new_tempo = mido.bpm2tempo(tempo)
    new_message = mido.MetaMessage("set_tempo", tempo=new_tempo, time=0)
    tempo_added = False

    for i, track in enumerate(midi.tracks):
        # remove existing set_tempo messages
        tempo_messages = []
        for j, msg in enumerate(track):
            if msg.type == "set_tempo":
                tempo_messages.append(j)

        for index in tempo_messages:
            midi.tracks[i][index] = new_message

        # add new set_tempo message to the first track
        if not tempo_added:
            track.insert(0, new_message)
            tempo_added = True

    # if no tracks had a set_tempo message and no new one was added, add a new track with the tempo message
    if not tempo_added:
        new_track = mido.MidiTrack()
        new_track.append(new_message)
        midi.tracks.append(new_track)

    midi.save(out_path)


def transform(
    file_path: str, out_dir: str, bpm: int, transformations: Dict, num_beats: int = 8
) -> str:
    new_filename = f"{Path(file_path).stem}_t{transformations['transpose']:02d}s{transformations['shift']:02d}"
    out_path = os.path.join(out_dir, f"{new_filename}.mid")

    if transformations["transpose"] != 0:
        t_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

        for instrument in pretty_midi.PrettyMIDI(file_path).instruments:
            # don't mess with the metronome
            if instrument.is_drum:
                t_midi.instruments.append(instrument)
                continue

            transposed_instrument = pretty_midi.Instrument(
                program=instrument.program, name=new_filename
            )

            for note in instrument.notes:
                transposed_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch + int(transformations["transpose"]),
                        start=note.start,
                        end=note.end,
                    )
                )

            t_midi.instruments.append(transposed_instrument)

        t_midi.write(out_path)
    else:
        mido.MidiFile(file_path).save(out_path)

    if transformations["shift"] != 0:
        s_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        seconds_per_beat = 60 / bpm
        shift_seconds = transformations["shift"] * seconds_per_beat
        loop_point = (num_beats + 0) * seconds_per_beat

        for instrument in pretty_midi.PrettyMIDI(out_path).instruments:
            # don't mess with the metronome
            if instrument.is_drum:
                s_midi.instruments.append(instrument)
                continue

            shifted_instrument = pretty_midi.Instrument(
                program=instrument.program, name=new_filename
            )
            for note in instrument.notes:
                dur = note.end - note.start
                shifted_start = (note.start + shift_seconds) % loop_point
                shifted_end = shifted_start + dur

                if note.start + shift_seconds >= loop_point:
                    shifted_start += seconds_per_beat
                    shifted_end += seconds_per_beat

                shifted_instrument.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=shifted_start,
                        end=shifted_end,
                    )
                )

            s_midi.instruments.append(shifted_instrument)

        s_midi.write(out_path)

    set_bpm(out_path, bpm)

    return out_path


def csv_to_midi(csv_path: str, midi_output_path: str, verbose: bool = False) -> bool:
    """
    Convert a CSV file containing piano note data to a MIDI file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with piano note data.
    midi_output_path : str
        Path where the MIDI file will be saved.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    bool
        True if the MIDI file was created successfully, False otherwise.
    """
    messages = []
    with open(csv_path, "r") as file:
        next(file)
        for line in file:
            _, msg_type, msg_pitch, msg_velocity, abs_time = line.strip().split(",")
            messages.append(
                mido.Message(
                    type=msg_type,
                    note=int(msg_pitch),
                    velocity=int(msg_velocity),
                    time=int(abs_time),
                )
            )

    if not messages:
        console.log("[orange]Warning: No notes found in CSV file")
        return False

    messages.sort(key=lambda msg: msg.time)
    if verbose:
        console.log(messages[:5])
        console.log(messages[-5:])

    # convert absolute timing to relative timing
    for i in range(len(messages) - 1, 0, -1):
        messages[i].time = messages[i].time - messages[i - 1].time
    # messages[0].time = 0

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    track.name = basename(midi_output_path)

    # add messages to track
    for msg in messages:
        track.append(msg)
    midi.tracks.append(track)

    midi.save(midi_output_path)

    console.log(f"[green]MIDI file created: '{midi_output_path}'")
    # if verbose:
    #     console.log("[bold]MIDI File Contents:[/bold]")
    #     midi.print_tracks()

    return os.path.isfile(midi_output_path)


def combine_midi_files(input_files: list[str], output_path: str) -> bool:
    """
    Combine multiple MIDI files into a single MIDI file, preserving separate tracks.

    Parameters
    ----------
    input_files : list[str]
        List of paths to input MIDI files.
    output_path : str
        Path where the combined MIDI file will be saved.

    Returns
    -------
    bool
        True if the combined MIDI file was created successfully, False otherwise.
    """
    combined = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    # add all tracks from all files
    for file_path in input_files:
        midi = mido.MidiFile(file_path)
        for track in midi.tracks:
            # remove all set_tempo messages (one is set in player recording file)
            track[:] = [msg for msg in track if msg.type != "set_tempo"]
            combined.tracks.append(track)

    combined.save(output_path)
    console.log(f"[green]combined MIDI file created: '{output_path}'")

    return os.path.isfile(output_path)


def generate_piano_roll(midi_path, output_path=None, figsize=(12, 8), dpi=100):
    """
    Generate a piano roll visualization from a MIDI file.
    TODO: have live velocity changes be reflected here

    Parameters
    ----------
    midi_path : str
        Path to the MIDI file.
    output_path : str, optional
        Path to save the piano roll image. If None, will save to the same directory
        as the MIDI file with the same name but with .png extension.
    figsize : tuple, optional
        Figure size in inches (width, height).
    dpi : int, optional
        DPI for the output image.

    Returns
    -------
    str
        Path to the generated piano roll image.
    """
    if output_path is None:
        output_path = os.path.splitext(midi_path)[0] + "-pianoroll.png"

    try:
        if not os.path.exists(midi_path):
            console.log(
                f"\t[orange]midi file '{os.path.basename(midi_path)}' not found[/orange]"
            )
            return None

        midi_data = pretty_midi.PrettyMIDI(midi_path)

        plt.figure(figsize=figsize)
        legend_elements = []
        colors = ["red", "blue"]  # player is red, system is blue
        for i, instrument in enumerate(midi_data.instruments):
            if not instrument.notes:
                continue

            for note in instrument.notes:
                plt.fill_betweenx(
                    [note.pitch, note.pitch + 0.8],
                    [note.start, note.start],
                    [note.end, note.end],
                    color=colors[i],
                    alpha=note.velocity / 127.0,
                )

            # add to legend
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[i],
                    lw=4,
                    label=instrument.name.split("-")[0],
                )
            )

        # add legend, labels and title
        plt.legend(handles=legend_elements, loc="upper right")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Pitch")
        plt.title(f"Piano Roll - {os.path.basename(midi_path)}")
        # set y limits to show only the piano range
        plt.ylim(20, 108)
        plt.xlim(0, midi_data.get_end_time())
        # display octaves
        octaves = list(range(24, 108, 12))
        octave_names = ["C" + str(i) for i in range(1, 8)]
        plt.yticks(octaves, octave_names)
        plt.grid(alpha=0.3)
        # save
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close()

        console.log(
            f"\t[green italic]saved piano roll to '{os.path.basename(output_path)}'[/green italic]"
        )
        return output_path
    except Exception as e:
        console.log(f"\t[red]error generating piano roll: {e}[/red]")
        return None


def get_note_min_max(input_file_path) -> Tuple[int, int]:
    """
    Returns the values of the highest and lowest notes in a MIDI file.

    Parameters
    ----------
    input_file_path : str
        Path to the MIDI file.

    Returns
    -------
    Tuple[int, int]
        The lowest and highest notes in the MIDI file.
    """
    mid = mido.MidiFile(input_file_path)
    lowest_note = 127
    highest_note = 0

    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta and msg.type in ["note_on", "note_off"]:
                if msg.velocity > 0:
                    lowest_note = min(lowest_note, msg.note)
                    highest_note = max(highest_note, msg.note)

    return (lowest_note, highest_note)


def trim_piano_roll(piano_roll):
    """
    trim piano roll to remove empty rows and columns

    Parameters
    ----------
    piano_roll : np.ndarray
        piano roll array

    Returns
    -------
    np.ndarray
        trimmed piano roll array
    """
    # find non-zero rows and columns
    rows = np.any(piano_roll, axis=1)
    cols = np.any(piano_roll, axis=0)

    # get indices of non-zero rows and columns
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    # if no non-zero elements, return empty array
    if len(row_indices) == 0 or len(col_indices) == 0:
        return np.array([])

    # trim array
    return piano_roll[
        row_indices[0] : row_indices[-1] + 1, col_indices[0] : col_indices[-1] + 1
    ]


def upsample_piano_roll(piano_roll, target_height=400, target_width=1200):
    """
    upsample piano roll to target resolution while preserving aspect ratio

    Parameters
    ----------
    piano_roll : np.ndarray
        piano roll array
    target_height : int
        target height in pixels
    target_width : int
        target width in pixels

    Returns
    -------
    np.ndarray
        upsampled piano roll array
    """
    if piano_roll.size == 0:
        return np.zeros((target_height, target_width))

    # calculate zoom factors
    height, width = piano_roll.shape
    zoom_height = target_height / height
    zoom_width = target_width / width

    # upsample using scipy zoom
    return zoom(piano_roll, (zoom_height, zoom_width), order=0)


def transpose_midi(input_file_path: str, output_file_path: str, semitones: int) -> None:
    """
    Transposes all the notes in a MIDI file by a specified number of semitones.

    Parameters
    ----------
    input_file_path : str
        Path to the input MIDI file.
    output_file_path : str
        Path where the transposed MIDI file will be saved.
    semitones : int
        Number of semitones to transpose the notes. Positive for up, negative for down.
    """

    midi = pretty_midi.PrettyMIDI(input_file_path)
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += semitones
    midi.write(output_file_path)


def change_tempo_and_trim(
    input_file: str, output_file: str, tempo: float = 93.75, cutoff_sec: float = 5.12
) -> bool:
    """Process a MIDI file to change its tempo and trim events after a cutoff.

    This function loads a MIDI file, replaces any existing tempo messages with a new tempo
    message, and trims events so that no event occurs after the specified cutoff time.
    If any note remains active at the cutoff, a note_off message is inserted.

    Parameters
    ----------
    input_file : str
        Path to the input MIDI file.
    output_file : str
        Path where the processed MIDI file will be saved.
    tempo : float, optional
        Desired tempo in BPM. Defaults to 93.75 so that 9 beats equals 5.12s.
    cutoff_sec : float, optional
        Time in seconds after which events are trimmed.
            Defaults to 5.12.

    Returns
    -------
    bool
        True if the output file exists after saving, False otherwise.
    """

    mid = mido.MidiFile(input_file)
    new_tempo_value = mido.bpm2tempo(tempo)
    ticks_per_beat = mid.ticks_per_beat

    # if no tracks exist, add a new track with the tempo message
    if not mid.tracks:
        new_track = mido.MidiTrack()
        new_track.append(mido.MetaMessage("set_tempo", tempo=new_tempo_value, time=0))
        mid.tracks.append(new_track)

    new_tracks = []
    for track_index, track in enumerate(mid.tracks):
        new_track = []
        current_abs_tick = 0
        active_notes = {}
        # insert new tempo message at beginning of first track
        if track_index == 0:
            new_track.append(
                mido.MetaMessage("set_tempo", tempo=new_tempo_value, time=0)
            )
        for msg in track:
            # skip existing tempo messages
            if msg.type == "set_tempo":
                continue
            next_abs_tick = current_abs_tick + msg.time
            event_time_sec = mido.tick2second(
                next_abs_tick, ticks_per_beat, new_tempo_value
            )
            if event_time_sec > cutoff_sec:
                current_time_sec = mido.tick2second(
                    current_abs_tick, ticks_per_beat, new_tempo_value
                )
                remaining_sec = cutoff_sec - current_time_sec
                remaining_ticks = int(
                    mido.second2tick(remaining_sec, ticks_per_beat, new_tempo_value)
                )
                first = True
                for channel, note in list(active_notes.keys()):
                    note_off_time = remaining_ticks if first else 0
                    new_track.append(
                        mido.Message(
                            "note_off",
                            note=note,
                            channel=channel,
                            velocity=0,
                            time=note_off_time,
                        )
                    )
                    first = False
                current_abs_tick += remaining_ticks
                break
            else:
                new_track.append(msg.copy(time=msg.time))
                current_abs_tick = next_abs_tick
                if msg.type == "note_on" and msg.velocity > 0:
                    active_notes[(msg.channel, msg.note)] = True
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    key = (msg.channel, msg.note)
                    if key in active_notes:
                        del active_notes[key]
        current_time_sec = mido.tick2second(
            current_abs_tick, ticks_per_beat, new_tempo_value
        )
        if current_time_sec < cutoff_sec and active_notes:
            remaining_sec = cutoff_sec - current_time_sec
            remaining_ticks = int(
                mido.second2tick(remaining_sec, ticks_per_beat, new_tempo_value)
            )
            first = True
            for channel, note in list(active_notes.keys()):
                note_off_time = remaining_ticks if first else 0
                new_track.append(
                    mido.Message(
                        "note_off",
                        note=note,
                        channel=channel,
                        velocity=0,
                        time=note_off_time,
                    )
                )
                first = False
        new_tracks.append(new_track)

    mid.tracks = new_tracks
    mid.save(output_file)

    return os.path.isfile(output_file)


def rearrange_midi(
    midi_path: str,
    output_path: str,
    augmentations: list[list[int]],
) -> list[str]:
    """
    Augment a MIDI file with a given list of augmentations.
    """
    bpm = get_bpm(midi_path)
    generated_paths = []

    for aug in augmentations:
        new_aug = os.path.join(output_path, f"{'_'.join(map(str, aug))}.mid")
        split_beats = beat_split(midi_path)
        beat_join(split_beats, aug, bpm).write(new_aug)
        generated_paths.append(new_aug)

    return generated_paths


def beat_split(midi_path: str, bpm: int | None = None) -> dict:
    """
    Split a MIDI file into beats.

    Parameters
    ----------
    midi_path : str
        Path to the MIDI file.

    Returns
    -------
    dict
        A dictionary of beats, where each key is an integer representing the beat number, and each value is a dictionary containing the notes and beats for that beat.
    """
    if bpm is None:
        bpm = get_bpm(midi_path)
    tempo = mido.bpm2tempo(bpm)
    midi = pretty_midi.PrettyMIDI(midi_path)
    beat_timings = range(
        TICKS_PER_BEAT,
        mido.second2tick(midi.get_end_time(), TICKS_PER_BEAT, tempo) + TICKS_PER_BEAT,
        TICKS_PER_BEAT,
    )
    beats = {
        i: {
            "notes": [],
            "beats": [
                mido.MetaMessage(
                    "text",
                    text=f"beat {i}: {mido.tick2second(b - TICKS_PER_BEAT, TICKS_PER_BEAT, tempo):.1f}",
                    time=0,
                ),
                mido.MetaMessage(
                    "text",
                    text=f"beat {i+1}: {mido.tick2second(b, TICKS_PER_BEAT, tempo):.1f}",
                    time=TICKS_PER_BEAT,
                ),
            ],
        }
        for i, b in enumerate(beat_timings)
    }

    for instrument in midi.instruments:
        for note in instrument.notes:
            note_start_ticks = mido.second2tick(note.start, TICKS_PER_BEAT, tempo)

            for i, b in enumerate(beat_timings):
                if note_start_ticks < b:
                    note_offset = mido.tick2second(
                        b - TICKS_PER_BEAT, TICKS_PER_BEAT, tempo
                    )
                    note.start -= note_offset
                    note.end -= note_offset
                    beats[i]["notes"].append(note)
                    break

    # Create a list of beat indices to remove
    beats_to_remove = []
    for i, beat in beats.items():
        if len(beat["notes"]) == 0:
            beats_to_remove.append(i)
            console.log(f"[yellow]beat {i} has no notes, removing[/yellow]")
    
    # Remove the empty beats after iteration is complete
    for i in beats_to_remove:
        del beats[i]

    return beats


def beat_join(
    beats: dict[int, dict], arrangement: list[int], bpm: int
) -> pretty_midi.PrettyMIDI:
    """
    Join a dictionary of beats into a single MIDI file.

    Parameters
    ----------
    beats : dict[int, dict]
        A dictionary of beats, where each key is an integer representing the beat number, and each value is a dictionary containing the notes and beats for that beat.
    arrangement: list[int]
        A list of integers representing the arrangement of the beats.
    bpm: int
        The tempo of the MIDI file in beats per minute.

    Returns
    -------
    pretty_midi.PrettyMIDI
        A PrettyMIDI object representing the joined MIDI file.
    """
    ts_beat = 60 / bpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, name="".join(map(str, arrangement)))

    for i, a in enumerate(arrangement):
        for note in beats[a]["notes"]:
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start + i * ts_beat,
                end=note.end + i * ts_beat,
            )
            inst.notes.append(new_note)
    pm.instruments.append(inst)

    return pm


def remove_notes(
    midi_path: str, output_path: str, amount: int | float, num_versions: int = 1
) -> list[str]:
    """
    Remove notes from a MIDI file with intermediate versions.

    Parameters
    ----------
    midi_path : str
        Path to the input MIDI file.
    output_path : str
        Directory to save output files.
    amount : int | float
        Number or percentage of notes to remove.
    num_versions : int, optional
        Number of versions to create with the same amount of notes removed.

    Returns
    -------
    list[str]
        Paths to the new MIDI files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    midi = pretty_midi.PrettyMIDI(midi_path)

    num_notes = 0
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                num_notes += 1

    # change from percentage to absolute number
    if isinstance(amount, float) and amount < 1:
        amount = int(num_notes * amount)

    amount = int(amount)  # catch things like 1.3

    # determine intermediate steps for note removal
    steps = []
    if amount <= 3:
        # if removing 3 or fewer notes, just save the final version
        steps = [amount]
    else:
        # save up to 3 intermediate versions
        if amount % 2 == 1:  # odd
            steps = [
                1,
                amount // 2 + (amount % 4 > 0),
                amount,
            ]  # first, middle-ish, last
        else:  # even
            steps = [
                2,
                amount // 2 + (amount % 4 > 0),
                amount,
            ]  # first, middle-ish, last

    # Create a single random selection of indices to remove
    all_indices = np.random.permutation(range(num_notes))

    new_files = []
    for version in range(num_versions):
        for step in steps:
            notes_to_remove = all_indices[:step]

            new_midi = pretty_midi.PrettyMIDI(
                initial_tempo=get_bpm(midi_path), resolution=TICKS_PER_BEAT
            )
            new_inst = pretty_midi.Instrument(program=0, name=f"{version+1}-{step}")
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for idx, note in enumerate(instrument.notes):
                        if idx not in notes_to_remove:
                            new_inst.notes.append(note)
            new_midi.instruments.append(new_inst)

            pf_new = os.path.join(
                output_path,
                f"{os.path.basename(midi_path).split('.')[0]}_{version+1:02d}-{step:02d}.mid",
            )
            new_midi.write(pf_new)
            new_files.append(pf_new)

    return new_files
