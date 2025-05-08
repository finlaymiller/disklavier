import os
import sys
import mido
import faiss
import pretty_midi
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ml.specdiff.model import SpectrogramDiffusion
from src.utils import basename, console, midi
from src.utils.midi import upsample_piano_roll

np.random.seed(0)

FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
pf_augmentations = os.path.join("outputs", "augmentations")
os.makedirs(pf_augmentations, exist_ok=True)

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
specdiff = SpectrogramDiffusion(verbose=True)

all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()

chosen_files = np.random.choice(all_files, size=3, replace=False)
file_data = {}
for chosen_file in chosen_files:
    console.log(f"augmenting '{chosen_file}'")
    key = chosen_file
    file_data[key] = {}
    bpm = midi.get_bpm(chosen_file)
    new_path = os.path.join(pf_augmentations, f"{basename(chosen_file)}_")

    # add notes to start of file
    random_notes = midi.generate_random_midi(2, bpm)
    md_midi_joined = mido.MidiFile(chosen_file)
    md_midi = mido.MidiFile(ticks_per_beat=220)
    clean_track = mido.MidiTrack()
    md_midi.tracks.append(clean_track)

    for note in random_notes:
        clean_track.append(note)

    for note in md_midi_joined.tracks[1]:
        if note.type == "note_on" or note.type == "note_off":
            clean_track.append(note)

    start_path = new_path + "start.mid"
    md_midi.save(start_path)
    file_data[key][start_path] = specdiff.embed(start_path)
    file_data[key][start_path] /= np.linalg.norm(
        file_data[key][start_path], axis=1, keepdims=True
    )

    # add notes to end of file
    random_notes = midi.generate_random_midi(2, bpm)
    md_midi_joined = mido.MidiFile(chosen_file)

    for track in md_midi_joined.tracks:
        for msg in track:
            if msg.type == "end_of_track":
                track.remove(msg)

    md_midi_joined.tracks[1].extend(random_notes)
    md_midi_joined.tracks[1].append(mido.MetaMessage("end_of_track", time=0))

    end_path = new_path + "end.mid"
    md_midi_joined.save(end_path)
    file_data[key][end_path] = specdiff.embed(end_path)
    file_data[key][end_path] /= np.linalg.norm(
        file_data[key][end_path], axis=1, keepdims=True
    )

print(file_data.keys())

for base_file_path, data in file_data.items():
    bpm = midi.get_bpm(base_file_path)
    console.log(f"augmenting '{base_file_path}' (bpm {bpm})")
    for filepath, embedding in data.items():
        midi_paths = []
        split_beats = midi.beat_split(filepath, bpm)
        console.log(f"\tsplit '{basename(filepath)}' into {len(split_beats)} beats")
        ids = list(range(len(split_beats)))
        rearrangements = {
            "original": ids,  # original
            "first-half": ids[: len(ids) // 2],  # first half
            "first-half-twice": ids[: len(ids) // 2] * 2,  # first half twice
            "second-half": ids[len(ids) // 2 - 1 :],  # second half
            "second-half-twice": ids[len(ids) // 2 - 1 :] * 2,  # second half twice
        }
        for name, arrangement in rearrangements.items():
            console.log(f"\trearrangement {name}:\t{arrangement}")
            joined_midi: pretty_midi.PrettyMIDI = midi.beat_join(
                split_beats, arrangement, bpm
            )

            pf_joined_midi = os.path.join(
                pf_augmentations, f"{basename(filepath)}_{name}.mid"
            )
            joined_midi.write(pf_joined_midi)
            console.log(f"\tjoined midi is \t{joined_midi.get_end_time()} seconds")
            midi_paths.append(pf_joined_midi)

        # find best match for each augmentation and store pairs
        augmentation_paths = []
        best_match_paths = []
        similarities_list = []  # Keep track of similarities for logging if needed

        console.log(f"	Finding best matches for {len(midi_paths)} augmentations...")
        for mid_idx, mid in enumerate(midi_paths):
            # console.log(f"		Processing augmentation {mid_idx + 1}/{len(midi_paths)}: {basename(mid)}")
            embedding = specdiff.embed(mid)
            if embedding.size == 0:
                console.log(
                    f"		[yellow]Warning:[/yellow] Empty embedding for {basename(mid)}, skipping."
                )
                continue
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            similarities, indices = faiss_index.search(
                embedding, 1
            )  # Get only the top 1 match
            match = all_files[indices[0][0]]
            similarity = similarities[0][0]

            augmentation_paths.append(mid)
            best_match_paths.append(match)
            similarities_list.append(similarity)
            console.log(
                f"		Best match for '{basename(mid)}' is '{basename(match)}' with similarity {similarity:.4f}"
            )

        # Generate and save comparison plot
        n_augmentations = len(augmentation_paths)
        if n_augmentations == 0:
            console.log("	No valid augmentations found to generate plot.")
            continue

        console.log(
            f"	Generating comparison plot for {n_augmentations} augmentations..."
        )
        fig, axes = plt.subplots(
            n_augmentations, 2, figsize=(8, n_augmentations * 3.5), squeeze=False
        )
        # if n_augmentations == 1: # Ensure axes is always 2D - No longer necessary with squeeze=False
        #     axes = np.array(axes).reshape(1, 2) # Adjusted reshape

        # Find indices of top 2 similarities
        top_indices = []
        if n_augmentations >= 2:
            top_indices = np.argsort(similarities_list)[-1:]
        elif n_augmentations == 1:
            top_indices = [0]

        for i in range(n_augmentations):
            # Determine title highlight
            title_bbox = (
                dict(facecolor="yellow", alpha=0.5, pad=2) if i in top_indices else None
            )

            # Load and process augmentation
            aug_pm = pretty_midi.PrettyMIDI(augmentation_paths[i])
            aug_roll = aug_pm.get_piano_roll(fs=100)
            aug_roll_upsampled = upsample_piano_roll(aug_roll)

            # Load and process best match
            match_pm = pretty_midi.PrettyMIDI(best_match_paths[i])
            match_roll = match_pm.get_piano_roll(fs=100)
            match_roll_upsampled = upsample_piano_roll(match_roll)

            # Plot augmentation
            ax_aug = axes[i, 0]
            if aug_roll_upsampled.size > 0:
                ax_aug.imshow(
                    aug_roll_upsampled,
                    cmap="gray_r",
                    aspect="auto",
                    interpolation="nearest",
                )
            ax_aug.set_title(
                basename(augmentation_paths[i]), fontsize=8, bbox=title_bbox
            )
            ax_aug.axis("off")

            # Plot match
            ax_match = axes[i, 1]
            if match_roll_upsampled.size > 0:
                ax_match.imshow(
                    match_roll_upsampled,
                    cmap="gray_r",
                    aspect="auto",
                    interpolation="nearest",
                )
            ax_match.set_title(
                f"{basename(best_match_paths[i])} (sim: {similarities_list[i]:.3f})",
                fontsize=8,
                bbox=title_bbox,
            )
            ax_match.axis("off")

        plt.tight_layout(
            rect=(0, 0.03, 1, 0.95)
        )  # Adjust layout to prevent title overlap
        fig.suptitle(
            f"Augmentation vs. Best Match Piano Rolls for {basename(filepath)}",
            fontsize=12,
        )
        plot_filename = os.path.join(
            pf_augmentations, f"{basename(filepath)}_comparison.png"
        )
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)
        console.log(f"	Saved comparison plot to '{plot_filename}'")
