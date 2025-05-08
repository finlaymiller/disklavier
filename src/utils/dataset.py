import os
import mido
import pretty_midi
import numpy as np
from glob import glob
from math import floor
import soundfile as sf
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from matplotlib import pyplot as plt

from utils import basename, console
from utils.midi import trim_piano_roll
from typing import Optional
from matplotlib import pyplot as plt
from utils.novelty import gen_ssm_and_novelty
from utils.midi import trim_piano_roll


def add_metronome(midi_file_path: str, num_beats: int, velocity: int) -> None:
    """
    adds a clave on each beat of the segment.

    Parameters
    ----------
    midi_file_path : str
        path to the midi file.
    num_beats : int
        number of beats in the segment.
    velocity : int
        velocity of the clave.

    Returns
    -------
    None
    """
    midi = mido.MidiFile(midi_file_path)

    # create a new track for the clave
    clave_track = mido.MidiTrack()
    clave_track.append(mido.MetaMessage("track_name", name="metronome", time=0))

    # add clave notes on each beat
    # +1 because we want to include the last beat
    for beat in range(num_beats + 1):
        time_ticks = midi.ticks_per_beat - midi.ticks_per_beat // 8 if beat > 0 else 0
        clave_track.append(
            mido.Message(
                "note_on", note=76, velocity=velocity, time=time_ticks, channel=9
            )
        )

        # note off - make it short (1/8 of a beat)
        clave_track.append(
            mido.Message(
                "note_off",
                note=76,
                velocity=0,
                time=midi.ticks_per_beat // 8,
                channel=9,
            )
        )

    clave_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(clave_track)
    midi.save(midi_file_path)


def modify_end_of_track(midi_file_path: str, new_end_time: float, bpm: int) -> None:
    """
    Modifies the 'end_of_track' message in a MIDI file to match the new end time.

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file.
    new_end_time : float
        The new end time.
    bpm : int
        The BPM of the MIDI file.
    """
    midi = mido.MidiFile(midi_file_path)
    new_end_time_t = mido.second2tick(new_end_time, 220, mido.bpm2tempo(bpm))

    for _, track in enumerate(midi.tracks):
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
                track.append(mido.MetaMessage("end_of_track", time=offset))

    # Save the modified MIDI file
    midi.save(midi_file_path)


def midi_to_wav(
    input_folder: str,
    output_folder: str,
    sr: int = 48000,
    sf2_path: str = "src/ml/clap/Yamaha_C7__Normalized_.sf2",
) -> int:
    """
    Generate synthesized WAV files for each MIDI file in a folder.

    Parameters
    ----------
    input_folder : str
        Path to folder containing MIDI files.
    output_folder : str
        Path to folder where WAV files will be saved.
    sr : int
        Sample rate of the generated audio files in Hz.
    sf2_path : str
        Path to the SF2 file to use for synthesis.

    Returns
    -------
    int
        Number of files converted.
    """
    midi_files = glob(os.path.join(input_folder, "*.mid"))
    if not midi_files:
        console.log(
            f"[red]Warning: No MIDI files found in input folder '{input_folder}'"
        )
        return -1

    os.makedirs(output_folder, exist_ok=True)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    )
    wav_task = progress.add_task(f"synthesizing wav files", total=len(midi_files))
    n_files = 0
    with progress:
        for midi_path in midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                audio = midi_data.fluidsynth(fs=sr, sf2_path=sf2_path)

                filename = basename(midi_path) + ".wav"
                output_path = os.path.join(output_folder, filename)

                sf.write(output_path, audio, sr)
                n_files += 1
            except Exception as e:
                console.log(f"[yellow]Error converting '{midi_path}' to WAV: {e}")

            progress.advance(wav_task)

    console.log(f"[green]converted {n_files}/{len(midi_files)} MIDI files to WAV files")

    return n_files


def num_beats(midi: mido.MidiFile, bpm: int = 60) -> int:
    file_len_s = sum(msg.time for msg in midi)
    total_beats = file_len_s * bpm / 60

    return floor(total_beats)


def add_beats_to_file(in_path: str, out_path: str, bpm: Optional[int] = None):
    midi = mido.MidiFile(in_path)

    # extract timing information
    if bpm is None:
        bpm = int(basename(in_path).split("-")[1])
    tempo = mido.bpm2tempo(bpm)
    n_beats = num_beats(midi, bpm)
    beat_times_s = [
        mido.second2tick(60 / bpm, midi.ticks_per_beat, tempo)
        for _ in range(0, n_beats)
    ]

    # create tick track
    last_beat = 0
    tick_track = mido.MidiTrack()
    tick_track.append(mido.MetaMessage("track_name", name="tick", time=last_beat))
    for i, beat_time in enumerate(beat_times_s):
        beat_msg = mido.MetaMessage(
            "text",
            text=f"beat {i}",
            time=beat_time,
        )
        tick_track.append(beat_msg)
    midi.tracks.append(tick_track)

    midi.type = 1  # to allow saving with multiple tracks (some files generated by Live are type 0)
    midi.save(out_path)


def add_novelty(midi_file_path: str, pic_dir: str):
    midi_pm = pretty_midi.PrettyMIDI(midi_file_path)
    piano_roll = midi_pm.get_piano_roll(fs=220)
    ssm, novelty = gen_ssm_and_novelty(midi_file_path)
    trimmed_pr = trim_piano_roll(piano_roll)
    pr_height, pr_width = trimmed_pr.shape
    novelty_len = len(novelty)

    # generate x coordinates for novelty curve spanning the piano roll width
    novelty_x = np.linspace(0, pr_width - 1, novelty_len)
    # scale y coordinates to the piano roll height
    novelty_y = (1 - novelty / novelty.max()) * pr_height

    # save novelty curve
    plt.figure(figsize=(16, 4))
    plt.imshow(
        trimmed_pr,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    plt.plot(
        novelty_x,
        novelty_y,
        "g",
        linewidth=1.0,
        alpha=0.7,
    )
    plt.axis("off")
    plt.savefig(os.path.join(pic_dir, f"{basename(midi_file_path)}_novelty.png"))
    plt.close()

    # save ssm
    plt.figure(figsize=(16, 16))
    plt.imshow(ssm / ssm.max(), cmap="magma")
    plt.axis("off")
    plt.savefig(os.path.join(pic_dir, f"{basename(midi_file_path)}_ssm.png"))
    plt.close()
