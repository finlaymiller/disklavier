import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import pretty_midi

# filesystem parameters
INPUT_DIR = "../../data/datasets/20240621/play"
OUTPUT_PATH = "../../data/tests"
OUTPUT_ID = "20240621"  # what identifies this run?

# dataset parameters
NUM_BEATS = 9
MIN_BPM = 50
MAX_BPM = 100

# bounds for outlier detection
OUTLIERS = {
    "min_seg_len": NUM_BEATS * 60 / MAX_BPM
    - 1,  # 9 beats per segment * 60 secs per min / 100 bpm (fastest recording tempo) - 1s for buffer = 4.4s
    "max_seg_len": NUM_BEATS * 60 / MIN_BPM
    + 1,  # 9 beats per segment * 60 secs per min / 50 bpm (slowest recording tempo) + 1s for buffer = 11.8s
    "min_note_len": 0.01,  # 10ms min note len
    "max_note_len": 12,  # 12s max note len
    "min_pitch": 0,  # lowest MIDI note, unusual
    "max_pitch": 127
    - 11,  # within an octave of the highest MIDI note, may cause issues transposing
    "min_vel": 5,  # anything quieter than this is unlikely
    "max_vel": 120,  # anything louder than this is unlikely
}


def analyze(folder_path):
    dur_counter = Counter()
    len_counter = Counter()
    pch_counter = Counter()
    phn_counter = Counter()
    vel_counter = Counter()

    outliers = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mid"):
            file_path = os.path.join(folder_path, file_name)

            try:
                midi = pretty_midi.PrettyMIDI(file_path)

                # count segment-level properties
                # segment length
                et = midi.get_end_time()
                len_counter[et] += 1
                if et <= OUTLIERS["min_seg_len"]:
                    print(f"short segment found w length {et:.03f}s:\t{file_name}")
                    outliers.append(
                        {
                            "file": file_name,
                            "type": "under min segment length",
                            "value": et,
                        }
                    )
                elif et >= OUTLIERS["max_seg_len"]:
                    print(f"long segment found w length {et:.03f}s:\t{file_name}")
                    outliers.append(
                        {
                            "file": file_name,
                            "type": "over max segment length",
                            "value": et,
                        }
                    )

                # segment pitch histogram
                phn = midi.get_pitch_class_histogram()
                for i, pitch in enumerate(phn):
                    phn_counter[i] += pitch

                # count note-level properties
                midi_data = pretty_midi.PrettyMIDI(file_path)
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        # note duration
                        dt = note.end - note.start
                        dur_counter[dt] += 1
                        if dt <= OUTLIERS["min_note_len"]:
                            print(
                                f"short note found w note dur {dt:.03f}:\t{file_name}"
                            )
                            outliers.append(
                                {
                                    "file": file_name,
                                    "type": "under min note length",
                                    "value": dt,
                                }
                            )
                        elif dt >= OUTLIERS["max_note_len"]:
                            print(f"long note found w note dur {dt:.03f}:\t{file_name}")
                            outliers.append(
                                {
                                    "file": file_name,
                                    "type": "over max note length",
                                    "value": dt,
                                }
                            )
                        # note pitch
                        pch_counter[note.pitch] += 1
                        if note.pitch <= OUTLIERS["min_pitch"]:
                            print(f"low note found w pitch {note.pitch}:\t{file_name}")
                            outliers.append(
                                {
                                    "file": file_name,
                                    "type": "under min pitch",
                                    "value": note.pitch,
                                }
                            )
                        elif note.pitch >= OUTLIERS["max_pitch"]:
                            print(f"high note found w pitch {note.pitch}:\t{file_name}")
                            outliers.append(
                                {
                                    "file": file_name,
                                    "type": "over max pitch",
                                    "value": note.pitch,
                                }
                            )
                        # note velocity
                        vel_counter[note.velocity] += 1
                        if note.velocity <= OUTLIERS["min_vel"]:
                            print(
                                f"quiet note found w vel {note.velocity:03d}:\t{file_name}"
                            )
                            outliers.append(
                                {
                                    "file": file_name,
                                    "type": "under min velocity",
                                    "value": note.velocity,
                                }
                            )
                        elif note.velocity >= OUTLIERS["max_vel"]:
                            print(
                                f"loud note found w vel {note.velocity:03d}:\t{file_name}"
                            )
                            outliers.append(
                                {
                                    "file": file_name,
                                    "type": "over max velocity",
                                    "value": note.velocity,
                                }
                            )

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                outliers.append(
                    {"file": file_name, "type": "file processing error", "value": None}
                )
                continue

    return (
        len_counter,
        pch_counter,
        vel_counter,
        dur_counter,
        phn_counter,
    ), outliers


def plot_histogram(
    histogram,
    save_path,
    show=True,
    grid=False,
    x_label="Time (s)",
    y_label="Count",
    x_tick_labels=None,
    title=None,
) -> None:
    plt.figure(figsize=(10, 6))

    plt.bar(list(histogram.keys()), list(histogram.values()))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid)

    if title:
        plt.title(title)
    if x_tick_labels:
        plt.xticks(range(len(x_tick_labels)), x_tick_labels)

    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()


def get_analytics():
    # parameter verification
    if len(os.listdir(INPUT_DIR)) < 1:
        print(f"input directory '{INPUT_DIR}' is empty")
        raise IndexError

    output_dir = os.path.join(OUTPUT_PATH, OUTPUT_ID)
    if os.path.isdir(output_dir):
        print(f"output folder '{output_dir}' already exists, pick a new one")
        raise IsADirectoryError

    os.makedirs(output_dir)

    (lc, pc, vc, dc, pn), outliers = analyze(INPUT_DIR)

    outlier_file = os.path.join(output_dir, "outliers.json")
    print(f"{len(outliers)} outliers found, writing out to '{outlier_file}'")
    with open(outlier_file, "w", encoding="utf-8") as f:
        json.dump(outliers, f)

    lc_plot_name = "segment_lengths.png"
    plot_histogram(
        lc,
        os.path.join(output_dir, lc_plot_name),
        title="Histogram of MIDI File Lengths",
    )
    pc_plot_name = "pitch_counts.png"
    plot_histogram(
        pc,
        os.path.join(output_dir, pc_plot_name),
        x_label="MIDI Pitch",
        title="Histogram of MIDI Note Pitches",
    )
    vc_plot_name = "velocity_counts.png"
    plot_histogram(
        vc,
        os.path.join(output_dir, vc_plot_name),
        x_label="MIDI Velocity",
        title="Histogram of MIDI Note Velocities",
    )
    dc_plot_name = "duration_counts.png"
    plot_histogram(
        dc,
        os.path.join(output_dir, dc_plot_name),
        x_label="Duration (s)",
        title="Histogram of MIDI Note Durations",
    )
    semitones = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pn_plot_name = "pitch_histogram_counts.png"
    plot_histogram(
        pn,
        os.path.join(output_dir, pn_plot_name),
        x_tick_labels=semitones,
        x_label="Note",
        title="Sum of Pitch Histograms",
    )


if __name__ == "__main__":
    get_analytics()
