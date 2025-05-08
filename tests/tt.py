import os
import sys
import pretty_midi
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from numpy.linalg import norm
from scipy.stats import entropy
from itertools import combinations


project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import basename
from src.utils.midi import transpose_midi
from src.utils.constants import NOTE_NAMES
from src.ml.specdiff.model import SpectrogramDiffusion

np.random.seed(0)

FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
pf_augmentations = os.path.join("outputs", "augmentations")
os.makedirs(pf_augmentations, exist_ok=True)
# temporary directory for transposed files
TMP_DIR = "/home/finlay/disklavier/tests/outputs/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

model = SpectrogramDiffusion(verbose=False)

all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()
base_files = glob(os.path.join(DATA_DIR, "*t00s00.mid"))
base_files.sort()

chosen_files = np.random.choice(base_files, size=12, replace=False)

transposition_pattern = re.compile(r"_t(\d+)s\d+\.mid$")

all_results = []

def get_pitch_histogram(midi_path, normalize=True, smooth=1e-6):
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
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    # Initialize histogram for pitches 0-127
    hist = np.zeros(128)
    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # Optional: ignore drums
            for note in instrument.notes:
                # Ensure pitch is within valid range
                if 0 <= note.pitch <= 127:
                    if normalize:
                        hist[note.pitch] += (
                            note.end - note.start
                        )  # Use duration as weight
                    else:
                        hist[note.pitch] += 1

    if normalize:
        # Use total non-zero duration for normalization, or 1 if total duration is 0
        total_duration = hist.sum()
        # Add smoothing and normalize by duration
        hist += smooth * total_duration  # Scale smoothing by total duration
        hist /= hist.sum()

    return hist


def shift_histogram(hist, offset):
    """
    circularly shift a pitch histogram.

    parameters
    ----------
    hist : np.ndarray
        the pitch histogram (12 bins).
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


def plot_individual_difference(
    ax,
    df,
    title_prefix,
    offset=None,
    mag_label="Magnitude Diff",
    cos_label="Cosine Diff",
):
    """
    plot magnitude and cosine difference on primary and secondary y-axes using seaborn.

    parameters
    ----------
    ax : matplotlib.axes._axes.axes
        the axes to plot on.
    df : pd.dataframe
        dataframe containing 'transposition', 'magnitude difference', and 'cosine difference'.
    title_prefix : str
        prefix for the plot title.
    offset : int, optional
        transposition offset used for aligned plots. defaults to none.
    mag_label : str, optional
        label for the magnitude difference axis.
    cos_label : str, optional
        label for the cosine difference axis.
    """
    # Primary axis (Magnitude Difference)
    sns.lineplot(
        data=df,
        x="Transposition",
        y="Magnitude Difference",
        marker="o",
        label=mag_label,
        color="tab:blue",
        ax=ax,
    )
    ax.set_xlabel("Transposition (Semitones)")
    ax.set_ylabel(mag_label, color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, linestyle=":")

    # Secondary axis (Cosine Difference)
    ax2 = ax.twinx()
    sns.lineplot(
        data=df,
        x="Transposition",
        y="Cosine Difference",
        marker="s",
        linestyle="--",
        label=cos_label,
        color="tab:orange",
        ax=ax2,
    )
    ax2.set_ylabel(cos_label, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1)  # Set specific limits if needed for cosine diff

    # Title and Legend
    if offset is not None:
        ax.set_title(f"{title_prefix} (Aligned, Offset={offset})", fontsize=10)
        ax.set_xlabel("Original Transposition Label (Semitones)")
    else:
        ax.set_title(f"{title_prefix}", fontsize=10)

    # Combine legends from both axes
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc="best")
    ax.get_legend().remove()
    ax2.get_legend().remove()


def plot_heatmaps(embeddings_matrix, file_labels=None, transpose_labels=None):
    """
    generate heatmaps comparing embeddings between pairs of files across transpositions.

    parameters
    ----------
    embeddings_matrix : np.ndarray
        3d array of shape (n_files, n_transposes, len_embedding).
    file_labels : list[str], optional
        labels for the files (rows of the matrix).
    transpose_labels : list[str | int], optional
        labels for the transpositions (columns within each file).

    returns
    -------
    none
        displays the generated heatmaps.
    """
    n_files, n_transposes, _ = embeddings_matrix.shape

    if file_labels is None:
        file_labels = [f"file {i+1}" for i in range(n_files)]
    if transpose_labels is None:
        transpose_labels = list(range(n_transposes))

    # ensure labels match dimensions
    if len(file_labels) != n_files:
        raise ValueError(
            f"number of file labels ({len(file_labels)}) does not match number of files ({n_files})"
        )
    if len(transpose_labels) != n_transposes:
        raise ValueError(
            f"number of transpose labels ({len(transpose_labels)}) does not match number of transpositions ({n_transposes})"
        )

    for i, j in combinations(range(n_files), 2):
        file_i_embeddings = embeddings_matrix[i, :, :]
        file_j_embeddings = embeddings_matrix[j, :, :]

        magnitude_diff_matrix = np.zeros((n_transposes, n_transposes))
        cosine_sim_matrix = np.zeros((n_transposes, n_transposes))

        for t1 in range(n_transposes):
            for t2 in range(n_transposes):
                vec_i_t1 = file_i_embeddings[t1, :]
                vec_j_t2 = file_j_embeddings[t2, :]

                # magnitude difference
                magnitude_diff_matrix[t1, t2] = norm(vec_i_t1 - vec_j_t2)

                # cosine similarity
                norm_i_t1 = norm(vec_i_t1)
                norm_j_t2 = norm(vec_j_t2)
                if norm_i_t1 > 1e-9 and norm_j_t2 > 1e-9:  # check for non-zero vectors
                    cosine_sim = np.dot(vec_i_t1, vec_j_t2) / (norm_i_t1 * norm_j_t2)
                    # clip to handle potential floating point inaccuracies
                    cosine_sim_matrix[t1, t2] = np.clip(cosine_sim, -1.0, 1.0)
                else:
                    # assign 0 similarity if one or both vectors are zero
                    cosine_sim_matrix[t1, t2] = 0.0

        # plotting
        fig, axes = plt.subplots(
            1, 2, figsize=(16, 7)
        )  # increased figure size slightly

        # magnitude difference heatmap
        sns.heatmap(
            magnitude_diff_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis_r",  # reversed viridis: lower distance = brighter
            ax=axes[0],
            xticklabels=transpose_labels,
            yticklabels=transpose_labels,
            square=True,  # make cells square
            linewidths=0.5,  # add lines between cells
            cbar_kws={"shrink": 0.8},  # shrink color bar slightly
        )
        axes[0].set_title(f"magnitude difference")
        axes[0].set_xlabel(f"transpose index ({file_labels[j]})")
        axes[0].set_ylabel(f"transpose index ({file_labels[i]})")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].tick_params(axis="y", rotation=0)

        # cosine similarity heatmap
        sns.heatmap(
            cosine_sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",  # good for similarity (-1 to 1)
            ax=axes[1],
            vmin=-1,  # set range for cosine similarity
            vmax=1,
            xticklabels=transpose_labels,
            yticklabels=transpose_labels,
            square=True,  # make cells square
            linewidths=0.5,  # add lines between cells
            cbar_kws={"shrink": 0.8},  # shrink color bar slightly
        )
        axes[1].set_title(f"cosine similarity")
        axes[1].set_xlabel(f"transpose index ({file_labels[j]})")
        axes[1].set_ylabel(f"transpose index ({file_labels[i]})")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].tick_params(axis="y", rotation=0)

        title_suffix = f"comparison: {file_labels[i]} vs {file_labels[j]}"
        fig.suptitle(
            title_suffix,
            fontsize=16,
        )
        fig.tight_layout(
            rect=(0, 0.03, 1, 0.95)
        )  # adjust layout to prevent title overlap
        plt.show()

n_files = len(chosen_files)
n_cols = 3
n_rows = (n_files + n_cols - 1) // n_cols
fig_individual, axes_individual = plt.subplots(
    n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
)
axes_individual = axes_individual.flatten()

all_diffs = np.zeros((n_files, 13, 768))
all_prs = []

for i, base_file in enumerate(chosen_files):
    print(f"Processing base file: {basename(base_file)}")
    base_name_prefix = basename(base_file).replace("_t00s00", "")
    related_files = glob(os.path.join(DATA_DIR, f"{base_name_prefix}_t*s00.mid"))
    related_files.sort()
    related_files.append(f"{base_name_prefix}_t12s00.mid")

    try:
        base_embedding = model.embed(base_file).squeeze().flatten()
    except Exception as e:
        print(f"  Error getting embedding for {basename(base_file)}: {e}")
        continue

    results = []
    transpositions_found = []
    all_diffs[i, 0, :] = np.zeros_like(base_embedding)

    for j, f in enumerate(related_files):
        match = transposition_pattern.search(f)
        if match:
            transposition = int(match.group(1))
            transpositions_found.append(transposition)

            try:
                tmp_path = f"/home/finlay/disklavier/tests/outputs/tmp/{basename(f)}"
                transpose_midi(base_file, tmp_path, transposition)
                if i == 0:
                    all_prs.append(
                        pretty_midi.PrettyMIDI(tmp_path).get_piano_roll(fs=10)
                    )
                transposed_embedding = model.embed(tmp_path).squeeze()
                diff_vector = transposed_embedding - base_embedding
                all_diffs[i, j, :] = diff_vector
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
    plot_individual_difference(
        ax,
        df_results,
        base_name_prefix,
        mag_label="Magnitude Diff",
        cos_label="Cosine Diff",
    )
# Clean up empty subplots for individual plots
for j in range(i + 1, len(axes_individual)):
    fig_individual.delaxes(axes_individual[j])

fig_individual.suptitle(
    "Embedding Differences vs. Transposition (Individual Files)", fontsize=16
)
fig_individual.tight_layout(rect=(0, 0.03, 1, 0.97))
fig_individual.show()

# %%
plt.figure(figsize=(10, 5))
plt.imshow(np.hstack(all_prs), cmap="gray_r", aspect="auto", origin="lower")
# plt.colorbar(label="Velocity")
plt.title("Piano Roll of Base File")
plt.xlabel("Time")
plt.ylabel("Pitch")
plt.show()
