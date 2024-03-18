import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pretty_midi import PrettyMIDI

import os
from pathlib import Path

DARK = True
SEMITONES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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

    plt.tight_layout()
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


def plot_pr_hists(midi: PrettyMIDI, energy_hist, title) -> None:
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    plt.suptitle(title)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(midi.get_piano_roll(), aspect="auto", origin="lower", cmap="magma")
    ax1.set_title("First Image")
    # ax1.axis("off")
    ax1.set_title("piano roll")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(
        np.arange(12),
        midi.get_pitch_class_histogram(normalize=False),
        tick_label=SEMITONES,
    )
    ax2.set_title("unweighted chromagram")
    ax2.set_xticks([])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(np.arange(12), energy_hist, tick_label=SEMITONES)
    ax3.set_title("energy weighted chromagram")
    # ax3.set_yticks("off")
    ax3.set_xticks(range(12), SEMITONES)

    plt.tight_layout()
