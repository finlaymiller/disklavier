import os
import sys
import faiss
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from numpy.linalg import norm
import pretty_midi
from scipy.stats import entropy

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import basename
from src.utils.midi import transpose_midi
from src.ml.specdiff.model import SpectrogramDiffusion

np.random.seed(0)

FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
pf_augmentations = os.path.join("outputs", "augmentations")
os.makedirs(pf_augmentations, exist_ok=True)

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
model = SpectrogramDiffusion(verbose=False)

all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()
base_files = glob(os.path.join(DATA_DIR, "*t00s00.mid"))
base_files.sort()

chosen_files = np.random.choice(base_files, size=12, replace=False)

transposition_pattern = re.compile(r"_t(\d+)s\d+\.mid$")

all_results = []

# Plotting setup for individual files
n_files = len(chosen_files)
n_cols = 3
n_rows = (n_files + n_cols - 1) // n_cols
fig_individual, axes_individual = plt.subplots(
    n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
)
axes_individual = axes_individual.flatten()

for i, base_file in enumerate(chosen_files):
    print(f"Processing base file: {basename(base_file)}")
    base_name_prefix = basename(base_file).replace("_t00s00", "")
    related_files = glob(os.path.join(DATA_DIR, f"{base_name_prefix}_t*s00.mid"))
    related_files.sort()
    related_files.append(f"{base_name_prefix}_t12s00.mid")

    try:
        base_embedding = faiss_index.reconstruct(all_files.index(base_file))
        if base_embedding is None:
            print(f"  Skipping {basename(base_file)} - Could not get base embedding.")
            continue
        base_embedding = base_embedding.flatten()  # Ensure it's 1D
    except Exception as e:
        print(f"  Error getting embedding for {basename(base_file)}: {e}")
        continue

    results = []
    transpositions_found = []

    for f in related_files:
        match = transposition_pattern.search(f)
        if match:
            transposition = int(match.group(1))
            transpositions_found.append(transposition)

            try:
                if f in all_files:
                    transposed_embedding = faiss_index.reconstruct(all_files.index(f))
                else:
                    tmp_path = f"/home/finlay/disklavier/tests/outputs/tmp/{f}"
                    print(f"transposing {base_file} to {tmp_path}")
                    transpose_midi(base_file, tmp_path, 12)
                    transposed_embedding = model.embed(tmp_path).squeeze()
                diff_vector = transposed_embedding - base_embedding
                magnitude_diff = norm(diff_vector)

                # Cosine similarity calculation, handle potential zero vectors
                norm_base = norm(base_embedding)
                norm_transposed = norm(transposed_embedding)
                cosine_sim = np.dot(base_embedding, transposed_embedding) / (
                    norm_base * norm_transposed
                )

                # Ensure cosine similarity is within [-1, 1] due to potential floating point errors
                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                cosine_diff = 1 - cosine_sim

                results.append(
                    {
                        "Transposition": transposition,
                        "Magnitude Difference": magnitude_diff,
                        "Cosine Difference": cosine_diff,
                        "File": basename(f),  # Keep track of the specific file
                    }
                )

            except Exception as e:
                print(f"    Error processing {basename(f)}: {e}")
                continue

    # Sort results by transposition for consistent table/plotting
    results.sort(key=lambda x: x["Transposition"])

    # Print table for the current base file
    df_results = pd.DataFrame(results)
    print(
        df_results.to_string(
            index=False,
            columns=["Transposition", "Magnitude Difference", "Cosine Difference"],
        )
    )
    print("-" * 70)

    # Add base file info to results before extending the main list
    for res in results:
        res["BaseFile"] = base_name_prefix

    # Store for aggregate plot
    all_results.extend(results)

    # Plotting for individual file
    ax = axes_individual[i]
    ax.plot(
        df_results["Transposition"],
        df_results["Magnitude Difference"],
        "o-",
        label="Magnitude Diff",
    )
    ax.plot(
        df_results["Transposition"],
        df_results["Cosine Difference"],
        "s--",
        label="Cosine Diff",
    )
    ax.set_title(f"{base_name_prefix}", fontsize=10)
    ax.set_xlabel("Transposition (Semitones)")
    ax.set_ylabel("Difference")
    ax.grid(True, linestyle=":")
    ax.legend()
    axes_individual[i] = ax
# Clean up empty subplots for individual plots
for j in range(i + 1, len(axes_individual)):
    fig_individual.delaxes(axes_individual[j])

fig_individual.suptitle(
    "Embedding Differences vs. Transposition (Individual Files)", fontsize=16
)
fig_individual.tight_layout(rect=(0, 0.03, 1, 0.97))
fig_individual.show()

df_all = pd.DataFrame(all_results)

fig_agg, ax_agg = plt.subplots(1, 2, figsize=(14, 6))


# Aggregate Magnitude Difference Plot
# Use lineplot to connect dots for the same base file
sns.lineplot(
    data=df_all,
    x="Transposition",
    y="Magnitude Difference",
    hue="BaseFile",
    ax=ax_agg[0],
    alpha=0.5,  # Make lines semi-opaque
    marker="o",  # Show markers
    legend=False,
)
ax_agg[0].set_title("Aggregate Magnitude Difference vs. Transposition")
ax_agg[0].set_xlabel("Transposition (Semitones)")
ax_agg[0].set_ylabel("Magnitude Difference (L2 Norm)")
ax_agg[0].grid(True, linestyle=":")

# Aggregate Cosine Difference Plot
# Use lineplot to connect dots for the same base file
sns.lineplot(
    data=df_all,
    x="Transposition",
    y="Cosine Difference",
    hue="BaseFile",
    ax=ax_agg[1],
    alpha=0.5,  # Make lines semi-opaque
    marker="o",  # Show markers
    legend=False,
)
ax_agg[1].set_title("Aggregate Cosine Difference vs. Transposition")
ax_agg[1].set_xlabel("Transposition (Semitones)")
ax_agg[1].set_ylim(0, 1)
ax_agg[1].set_ylabel("Cosine Difference")
ax_agg[1].grid(True, linestyle=":")

fig_agg.suptitle(
    "Aggregate Embedding Differences vs. Transposition Across All Files",
    fontsize=16,
)
fig_agg.tight_layout(rect=(0, 0.03, 1, 0.95))
fig_agg.show()


###############################################################################
# Phase 2: Align base files using pitch histograms and KL divergence         #
###############################################################################
print("\n" + "=" * 80)
print("Phase 2: Aligning base files using pitch histograms and KL divergence")
print("=" * 80 + "\n")

# Define temporary directory for transposed files
TMP_DIR = "/home/finlay/disklavier/tests/outputs/tmp_aligned"
os.makedirs(TMP_DIR, exist_ok=True)


def get_pitch_histogram(midi_path, smooth=1e-6):
    """
    calculate the pitch histogram for a midi file over the full pitch range (0-127).

    parameters
    ----------
    midi_path : str
        path to the midi file.
    smooth : float
        smoothing factor added to all bins to avoid zeros.

    returns
    -------
    np.ndarray or none
        normalized pitch histogram (128 bins), or none if midi cannot be loaded.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        # Initialize histogram for pitches 0-127
        hist = np.zeros(128)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Optional: ignore drums
                for note in instrument.notes:
                    # Ensure pitch is within valid range
                    if 0 <= note.pitch <= 127:
                        hist[note.pitch] += (
                            note.end - note.start
                        )  # Use duration as weight

        # Use total non-zero duration for normalization, or 1 if total duration is 0
        total_duration = hist.sum()
        if total_duration == 0:
            # Handle MIDI files with no notes or zero duration notes
            if midi_data.instruments and any(
                inst.notes for inst in midi_data.instruments
            ):
                # If there are notes but total duration is zero, use note count instead
                note_count = sum(
                    len(inst.notes)
                    for inst in midi_data.instruments
                    if not inst.is_drum
                )
                for instrument in midi_data.instruments:
                    if not instrument.is_drum:
                        for note in instrument.notes:
                            if 0 <= note.pitch <= 127:
                                hist[note.pitch] += 1
                if hist.sum() > 0:
                    hist /= hist.sum()
                else:  # Still no counts, return uniform distribution? Or zeros + smooth?
                    hist += smooth
                    hist /= hist.sum()

            else:  # No notes at all
                hist += smooth
                hist /= hist.sum()
        else:
            # Add smoothing and normalize by duration
            hist += smooth * total_duration  # Scale smoothing by total duration
            hist /= hist.sum()

        return hist
    except Exception as e:
        print(
            f"    Error loading MIDI {basename(midi_path)} or calculating histogram: {e}"
        )
        return None


def shift_histogram(hist, offset):
    """
    circularly shift a pitch histogram.

    parameters
    ----------
    hist : np.ndarray
        the pitch histogram (128 bins).
    offset : int
        the number of semitones to shift (positive or negative).

    returns
    -------
    np.ndarray
        the shifted histogram.
    """
    return np.roll(hist, offset)


def calculate_kl_divergence(p, q):
    """
    calculate kl divergence d_kl(p || q).

    parameters
    ----------
    p : np.ndarray
        first probability distribution.
    q : np.ndarray
        second probability distribution.

    returns
    -------
    float
        the kl divergence.
    """
    return entropy(p, q)


# --- Calculate Optimal Transpositions ---
optimal_offsets = {}
first_file = chosen_files[0]
print(f"Reference file for alignment: {basename(first_file)}")
hist_0 = get_pitch_histogram(first_file)

if hist_0 is None:
    print(
        "Error: Could not calculate histogram for the reference file. Skipping phase 2."
    )
    sys.exit(1)  # Or handle differently

optimal_offsets[first_file] = 0  # Offset for the first file is 0

for i, current_file in enumerate(chosen_files[1:], 1):
    print(f"Finding optimal offset for: {basename(current_file)}")
    hist_i = get_pitch_histogram(current_file)
    if hist_i is None:
        print(f"  Skipping {basename(current_file)} - could not calculate histogram.")
        optimal_offsets[current_file] = None  # Mark as failed
        continue

    min_kl = float("inf")
    best_offset = 0
    # Adjust the search range for full pitch histograms (e.g., +/- 2 octaves)
    possible_offsets = range(-24, 25)

    for offset in possible_offsets:
        # Note: np.roll for 128 bins means notes shifted off one end wrap to the other.
        # This is generally okay for KL divergence comparison if the distributions
        # are reasonably centered and don't have significant mass at the extremes.
        shifted_hist_i = shift_histogram(hist_i, offset)
        kl_div = calculate_kl_divergence(shifted_hist_i, hist_0)
        # print(f"  Offset {offset}: KL={kl_div:.4f}") # Debug print
        if kl_div < min_kl:
            min_kl = kl_div
            best_offset = offset

    print(f"  Best offset: {best_offset} (KL={min_kl:.4f})")
    optimal_offsets[current_file] = best_offset


# --- Recalculate Embeddings and Differences using Aligned Base ---
aligned_results = []
# Plotting setup for individual aligned files
fig_individual_aligned, axes_individual_aligned = plt.subplots(
    n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
)
axes_individual_aligned = axes_individual_aligned.flatten()
plot_idx = 0

for i, base_file in enumerate(chosen_files):
    print(f"Processing aligned base file: {basename(base_file)}")
    offset = optimal_offsets.get(base_file)

    if offset is None:
        print(f"  Skipping {basename(base_file)} - no optimal offset found.")
        continue

    base_name_prefix = basename(base_file).replace("_t00s00", "")
    aligned_base_file_path = os.path.join(
        TMP_DIR, f"{base_name_prefix}_t00s00_aligned_offset{offset}.mid"
    )

    # Transpose the original base file by the optimal offset
    try:
        print(
            f"  Transposing {basename(base_file)} by {offset} semitones to {basename(aligned_base_file_path)}"
        )
        transpose_midi(base_file, aligned_base_file_path, offset)
        aligned_base_embedding = model.embed(aligned_base_file_path).squeeze()
        if aligned_base_embedding is None or aligned_base_embedding.size == 0:
            print(
                f"  Skipping {basename(base_file)} - could not get embedding for aligned base."
            )
            continue
        aligned_base_embedding = aligned_base_embedding.flatten()  # Ensure 1D
    except Exception as e:
        print(
            f"  Error processing aligned base {basename(aligned_base_file_path)}: {e}"
        )
        continue

    # Now compare related files (_t01.._t12) to this *aligned* base embedding
    related_files = glob(os.path.join(DATA_DIR, f"{base_name_prefix}_t*s00.mid"))
    related_files.sort()
    related_files.append(f"{base_name_prefix}_t12s00.mid")  # Ensure t12 is included

    results_aligned = []
    for f in related_files:
        match = transposition_pattern.search(f)
        if match:
            transposition = int(
                match.group(1)
            )  # This is the original transposition label (0-12)

            try:
                # ALWAYS generate the embedding for the target transposition relative to the original t00 file
                tmp_path = os.path.join(
                    TMP_DIR, f"{base_name_prefix}_t{transposition:02d}s00_temp.mid"
                )
                # print(f"    Generating embedding for transposition {transposition}: {basename(tmp_path)}") # Debug
                # Ensure we transpose the *original* t00 base file
                transpose_midi(base_file, tmp_path, transposition)
                transposed_embedding = model.embed(tmp_path).squeeze()

                if transposed_embedding is None or transposed_embedding.size == 0:
                    print(
                        f"    Skipping t{transposition:02d} - Could not get transposed embedding."
                    )
                    continue
                transposed_embedding = transposed_embedding.flatten()  # Ensure 1D

                # Calculate difference relative to the ALIGNED base embedding
                diff_vector = transposed_embedding - aligned_base_embedding
                magnitude_diff = norm(diff_vector)

                norm_aligned_base = norm(aligned_base_embedding)
                norm_transposed = norm(transposed_embedding)

                # Handle potential zero vectors
                if norm_aligned_base == 0 or norm_transposed == 0:
                    cosine_sim = 0.0
                    print(
                        f"    Warning: Zero norm vector encountered for {basename(f)} or its aligned base."
                    )
                else:
                    cosine_sim = np.dot(
                        aligned_base_embedding, transposed_embedding
                    ) / (norm_aligned_base * norm_transposed)

                # Ensure cosine similarity is within [-1, 1]
                cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                cosine_diff = 1 - cosine_sim

                results_aligned.append(
                    {
                        "Transposition": transposition,  # Original transposition label
                        "Magnitude Difference": magnitude_diff,
                        "Cosine Difference": cosine_diff,
                        "File": basename(f),
                        "Aligned Base Offset": offset,  # Store the offset used for the base
                    }
                )

            except Exception as e:
                print(f"    Error processing {basename(f)} against aligned base: {e}")
                continue

    # Sort results by transposition for consistent table/plotting
    results_aligned.sort(key=lambda x: x["Transposition"])

    # Print table for the current aligned base file
    if results_aligned:
        df_results_aligned = pd.DataFrame(results_aligned)
        print(f"  Results relative to base aligned by {offset} semitones:")
        print(
            df_results_aligned.to_string(
                index=False,
                columns=["Transposition", "Magnitude Difference", "Cosine Difference"],
            )
        )
        print("-" * 70)

        # Add base file info to results before extending the main list
        for res in results_aligned:
            res["BaseFile"] = base_name_prefix  # Original base file identifier

        # Store for aggregate plot
        aligned_results.extend(results_aligned)

        # Plotting for individual file (aligned)
        ax = axes_individual_aligned[plot_idx]
        ax.plot(
            df_results_aligned["Transposition"],
            df_results_aligned["Magnitude Difference"],
            "o-",
            label="Magnitude Diff (Aligned)",
        )
        ax.plot(
            df_results_aligned["Transposition"],
            df_results_aligned["Cosine Difference"],
            "s--",
            label="Cosine Diff (Aligned)",
        )
        ax.set_title(f"{base_name_prefix} (Aligned, Offset={offset})", fontsize=10)
        ax.set_xlabel("Original Transposition Label (Semitones)")
        ax.set_ylabel("Difference from Aligned Base")
        ax.grid(True, linestyle=":")
        ax.legend()
        plot_idx += 1
    else:
        print(f"  No results generated for aligned base {basename(base_file)}")


# Clean up empty subplots for individual aligned plots
for j in range(plot_idx, len(axes_individual_aligned)):
    fig_individual_aligned.delaxes(axes_individual_aligned[j])

fig_individual_aligned.suptitle(
    "Embedding Differences vs. Transposition (Aligned Base Files)", fontsize=16
)
fig_individual_aligned.tight_layout(rect=(0, 0.03, 1, 0.97))
fig_individual_aligned.show()


# --- Aggregate Plots (Aligned) ---
if aligned_results:
    df_all_aligned = pd.DataFrame(aligned_results)
    fig_agg_aligned, ax_agg_aligned = plt.subplots(1, 2, figsize=(14, 6))

    # Aggregate Magnitude Difference Plot (Aligned)
    sns.lineplot(
        data=df_all_aligned,
        x="Transposition",
        y="Magnitude Difference",
        hue="BaseFile",  # Color by original base file group
        ax=ax_agg_aligned[0],
        alpha=0.5,
        marker="o",
        legend=False,
    )
    ax_agg_aligned[0].set_title("Aggregate Magnitude Difference (Aligned Base)")
    ax_agg_aligned[0].set_xlabel("Original Transposition Label (Semitones)")
    ax_agg_aligned[0].set_ylabel("Magnitude Difference (L2 Norm)")
    ax_agg_aligned[0].grid(True, linestyle=":")

    # Aggregate Cosine Difference Plot (Aligned)
    sns.lineplot(
        data=df_all_aligned,
        x="Transposition",
        y="Cosine Difference",
        hue="BaseFile",  # Color by original base file group
        ax=ax_agg_aligned[1],
        alpha=0.5,
        marker="o",
        legend=False,
    )
    ax_agg_aligned[1].set_title("Aggregate Cosine Difference (Aligned Base)")
    ax_agg_aligned[1].set_xlabel("Original Transposition Label (Semitones)")
    ax_agg_aligned[1].set_ylim(0, 1)
    ax_agg_aligned[1].set_ylabel("Cosine Difference")
    ax_agg_aligned[1].grid(True, linestyle=":")

    fig_agg_aligned.suptitle(
        "Aggregate Embedding Differences vs. Transposition (Aligned Base)",
        fontsize=16,
    )
    fig_agg_aligned.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig_agg_aligned.show()
else:
    print("\nNo aligned results were generated to plot.")

# Optional: Clean up temporary aligned files
# import shutil
# shutil.rmtree(TMP_DIR)
# print(f"Cleaned up temporary directory: {TMP_DIR}")

print("\nScript finished.")
