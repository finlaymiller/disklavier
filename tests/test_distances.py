import os
import sys
import faiss
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math  # For sqrt
from rich.progress import track
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, DefaultDict
import multiprocessing
from functools import partial

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils import basename

# --- configuration ---
FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
TRACK_DIR = "/media/scratch/sageev-midi/20250410/unsegmented"
OUTPUT_DIR = "tests/outputs/distances"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # ensure output dir exists
NUM_WORKERS = max(1, multiprocessing.cpu_count())


# --- Data Structures for Accumulation ---
@dataclass
class TrackAccumulator:
    min_dist: float = float("inf")
    max_dist: float = float("-inf")
    min_pair_idx: Optional[Tuple[int, int]] = None
    max_pair_idx: Optional[Tuple[int, int]] = None
    sum_dist: float = 0.0
    sum_sq_dist: float = 0.0
    count: int = 0
    vector_sum: Optional[np.ndarray] = None

    def update(
        self,
        dist: float,
        dist_sq: float,
        pair_idx: Tuple[int, int],
        diff_vector: np.ndarray,
    ):
        if self.vector_sum is None:
            self.vector_sum = np.zeros_like(diff_vector, dtype=np.float64)

        self.sum_dist += dist
        self.sum_sq_dist += dist_sq
        self.count += 1
        self.vector_sum += diff_vector

        if dist < self.min_dist:
            self.min_dist = dist
            self.min_pair_idx = pair_idx
        if dist > self.max_dist:
            self.max_dist = dist
            self.max_pair_idx = pair_idx

    def merge(self, other: "TrackAccumulator"):
        if other.count == 0:
            return

        self.sum_dist += other.sum_dist
        self.sum_sq_dist += other.sum_sq_dist
        self.count += other.count

        if self.vector_sum is None and other.vector_sum is not None:
            self.vector_sum = other.vector_sum.copy()
        elif self.vector_sum is not None and other.vector_sum is not None:
            self.vector_sum += other.vector_sum

        if other.min_dist < self.min_dist:
            self.min_dist = other.min_dist
            self.min_pair_idx = other.min_pair_idx
        if other.max_dist > self.max_dist:
            self.max_dist = other.max_dist
            self.max_pair_idx = other.max_pair_idx


@dataclass
class WorkerResult:
    # Histogram bins (local to worker) - Moved to top
    local_hist_counts: np.ndarray  # Field without default value comes first

    # Global accumulators (local to worker)
    local_min_dist: float = float("inf")
    local_max_dist: float = float("-inf")
    local_min_pair_idx: Optional[Tuple[int, int]] = None
    local_max_pair_idx: Optional[Tuple[int, int]] = None
    local_sum_dist: float = 0.0
    local_sum_sq_dist: float = 0.0
    local_count: int = 0
    local_vector_sum: Optional[np.ndarray] = None  # Initialize later based on dim

    # Histogram bins continued (with defaults)
    local_hist_underflow: int = 0
    local_hist_overflow: int = 0

    # Track accumulators (local to worker)
    local_track_accumulators: DefaultDict[str, TrackAccumulator] = field(
        default_factory=lambda: defaultdict(TrackAccumulator)
    )

    # Inter/Intra (local to worker)
    local_intra_track_sum_dist: float = 0.0
    local_intra_track_count: int = 0
    local_inter_track_sum_dist: float = 0.0
    local_inter_track_count: int = 0


# --- Load FAISS Index (Main Process) ---
print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
index = faiss.read_index(FAISS_INDEX_PATH)
expected_dim = index.d  # Get expected dimension from FAISS index
print(
    f"FAISS index loaded. Contains {index.ntotal} vectors of dimension {expected_dim}."
)
num_embeddings = index.ntotal
N = num_embeddings  # Use N for number of embeddings
del index  # Delete index from main process memory after getting info - workers will reload

# --- Prepare File Lists & Track Mapping (Main Process) ---
print("Scanning data directories...")
all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()
track_files = glob(os.path.join(TRACK_DIR, "*.mid"))
track_files.sort()

if len(all_files) != num_embeddings:
    print(
        f"Warning: Number of files ({len(all_files)}) does not match number of embeddings ({num_embeddings}). Adjusting file list."
    )
    all_files = all_files[:num_embeddings]

all_tracks = [basename(t) for t in track_files]
print(f"Found {len(all_tracks)} tracks.")

print("Mapping files to tracks...")
track_mapping_indices: DefaultDict[str, List[int]] = defaultdict(list)
file_to_track: Dict[int, str] = {}
for i, f in enumerate(all_files):
    fname = basename(f)
    track_name = fname.split("_")[0]
    if track_name in all_tracks:  # only consider files belonging to known tracks
        track_mapping_indices[track_name].append(i)
        file_to_track[i] = track_name
print(f"Mapped {len(file_to_track)} files to {len(track_mapping_indices)} tracks.")

# --- Histogram Setup (Main Process) ---
num_bins = 50
hist_min_range = 0.00
hist_max_range = 10.0
bin_edges = np.linspace(hist_min_range, hist_max_range, num_bins + 1)


# --- Worker Function Definition ---
def process_chunk(
    i_values: List[int],
    index_path: str,
    n_total: int,
    file_map: Dict[int, str],
    dim: int,
    hist_bin_edges: np.ndarray,
    hist_n_bins: int,
    hist_min: float,
    hist_max: float,
) -> WorkerResult:
    """
    Processes a chunk of outer loop indices (i) to calculate pairwise distances.

    Parameters
    ----------
    i_values : List[int]
        The list of 'i' indices this worker is responsible for.
    index_path : str
        Path to the FAISS index file.
    n_total : int
        Total number of embeddings (N).
    file_map : Dict[int, str]
        Mapping from file index to track name.
    dim : int
        Dimension of the embeddings.
    hist_bin_edges : np.ndarray
        Edges of the histogram bins.
    hist_n_bins : int
        Number of histogram bins.
    hist_min : float
        Minimum value of the histogram range.
    hist_max : float
        Maximum value of the histogram range.

    Returns
    -------
    WorkerResult
        Accumulated statistics for the processed chunk.
    """
    worker_hist_counts = np.zeros(hist_n_bins, dtype=np.int64)
    result = WorkerResult(local_hist_counts=worker_hist_counts)
    result.local_vector_sum = np.zeros(dim, dtype=np.float64)

    try:
        local_index = faiss.read_index(index_path)
        if local_index.d != dim:
            raise ValueError(f"Worker index dim {local_index.d} != expected {dim}")
        if local_index.ntotal != n_total:
            raise ValueError(
                f"Worker index ntotal {local_index.ntotal} != expected {n_total}"
            )
    except Exception as e:
        print(f"[Worker Error] Failed to load FAISS index: {e}")
        return result

    for i in i_values:
        try:
            emb_i = local_index.reconstruct(i).astype(np.float64)
        except Exception as e:
            continue

        track_i = file_map.get(i)

        for j in range(i + 1, n_total):
            try:
                emb_j = local_index.reconstruct(j).astype(np.float64)
            except Exception as e:
                continue

            track_j = file_map.get(j)

            diff_vector = emb_i - emb_j
            dist_sq = np.dot(diff_vector, diff_vector)
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0

            result.local_sum_dist += dist
            result.local_sum_sq_dist += dist_sq
            result.local_count += 1
            result.local_vector_sum += diff_vector

            if dist < result.local_min_dist:
                result.local_min_dist = dist
                result.local_min_pair_idx = (i, j)
            if dist > result.local_max_dist:
                result.local_max_dist = dist
                result.local_max_pair_idx = (i, j)

            if dist < hist_min:
                result.local_hist_overflow += 1
            elif dist >= hist_max:
                result.local_hist_overflow += 1
            else:
                bin_index = np.searchsorted(hist_bin_edges, dist, side="right") - 1
                if 0 <= bin_index < hist_n_bins:
                    result.local_hist_counts[bin_index] += 1

            if track_i is not None and track_j is not None:
                if track_i == track_j:
                    track = track_i
                    acc = result.local_track_accumulators[track]
                    acc.update(dist, dist_sq, (i, j), diff_vector)

                    result.local_intra_track_sum_dist += dist
                    result.local_intra_track_count += 1
                else:
                    result.local_inter_track_sum_dist += dist
                    result.local_inter_track_count += 1

    del local_index
    return result


# --- Main Processing Logic --- (To be added in next step)
if __name__ == "__main__":
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")

    indices = list(range(N))
    chunk_size = math.ceil(N / NUM_WORKERS)
    chunks = [indices[k : k + chunk_size] for k in range(0, N, chunk_size)]
    print(f"Divided {N} indices into {len(chunks)} chunks (approx size {chunk_size}).")

    worker_func = partial(
        process_chunk,
        index_path=FAISS_INDEX_PATH,
        n_total=N,
        file_map=file_to_track,
        dim=expected_dim,
        hist_bin_edges=bin_edges,
        hist_n_bins=num_bins,
        hist_min=hist_min_range,
        hist_max=hist_max_range,
    )

    final_global_min_dist = float("inf")
    final_global_max_dist = float("-inf")
    final_global_min_pair_idx = None
    final_global_max_pair_idx = None
    final_global_sum_dist = 0.0
    final_global_sum_sq_dist = 0.0
    final_global_count = 0
    final_global_vector_sum = np.zeros(expected_dim, dtype=np.float64)

    final_hist_counts = np.zeros(num_bins, dtype=np.int64)
    final_hist_underflow = 0
    final_hist_overflow = 0

    final_track_accumulators: DefaultDict[str, TrackAccumulator] = defaultdict(
        TrackAccumulator
    )
    final_intra_track_sum_dist = 0.0
    final_intra_track_count = 0
    final_inter_track_sum_dist = 0.0
    final_inter_track_count = 0

    print("Processing chunks...")
    results: List[WorkerResult] = []
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        for result in track(
            pool.imap_unordered(worker_func, chunks),
            description="Processing chunks",
            total=len(chunks),
        ):
            results.append(result)

    print(f"\nFinished processing {len(results)} chunks. Aggregating results...")

    for res in results:
        final_global_sum_dist += res.local_sum_dist
        final_global_sum_sq_dist += res.local_sum_sq_dist
        final_global_count += res.local_count
        if res.local_vector_sum is not None:
            final_global_vector_sum += res.local_vector_sum

        if res.local_min_dist < final_global_min_dist:
            final_global_min_dist = res.local_min_dist
            final_global_min_pair_idx = res.local_min_pair_idx
        if res.local_max_dist > final_global_max_dist:
            final_global_max_dist = res.local_max_dist
            final_global_max_pair_idx = res.local_max_pair_idx

        final_hist_counts += res.local_hist_counts
        final_hist_underflow += res.local_hist_underflow
        final_hist_overflow += res.local_hist_overflow

        final_intra_track_sum_dist += res.local_intra_track_sum_dist
        final_intra_track_count += res.local_intra_track_count
        final_inter_track_sum_dist += res.local_inter_track_sum_dist
        final_inter_track_count += res.local_inter_track_count

        for track_name, local_acc in res.local_track_accumulators.items():
            final_track_accumulators[track_name].merge(local_acc)

    print("Aggregation complete.")

    print("--- Global Statistics ---")
    if final_global_count > 0:
        global_avg_dist = final_global_sum_dist / final_global_count
        global_variance = (final_global_sum_sq_dist / final_global_count) - (
            global_avg_dist**2
        )
        global_std_dev = math.sqrt(max(0, global_variance))
        global_avg_vector = final_global_vector_sum / final_global_count
        global_avg_vector_magnitude = np.linalg.norm(global_avg_vector)

        min_file1 = (
            basename(all_files[final_global_min_pair_idx[0]])
            if final_global_min_pair_idx
            else "N/A"
        )
        min_file2 = (
            basename(all_files[final_global_min_pair_idx[1]])
            if final_global_min_pair_idx
            else "N/A"
        )
        max_file1 = (
            basename(all_files[final_global_max_pair_idx[0]])
            if final_global_max_pair_idx
            else "N/A"
        )
        max_file2 = (
            basename(all_files[final_global_max_pair_idx[1]])
            if final_global_max_pair_idx
            else "N/A"
        )

        print(f"Smallest Distance: {final_global_min_dist:.4f}")
        print(f"  Files: {min_file1}, {min_file2}")
        print(f"Largest Distance: {final_global_max_dist:.4f}")
        print(f"  Files: {max_file1}, {max_file2}")
        print(f"Average Distance: {global_avg_dist:.4f}")
        print(
            "Median Distance: N/A (Requires storing all distances or different algorithm)"
        )
        print(f"Standard Deviation of Distances: {global_std_dev:.4f}")
        print(f"Average Distance Vector Magnitude: {global_avg_vector_magnitude:.4f}")
    else:
        print("No pairs processed, cannot calculate global statistics.")
        global_avg_dist = 0

    print("--- Track-Specific Statistics ---")
    final_track_stats_report = {}
    processed_tracks = set()

    for track, acc in final_track_accumulators.items():
        processed_tracks.add(track)
        stats = {}
        count = acc.count
        if count > 0:
            avg = acc.sum_dist / count
            variance = (acc.sum_sq_dist / count) - (avg**2)
            std_dev = math.sqrt(max(0, variance))
            avg_vec = (
                acc.vector_sum / count
                if acc.vector_sum is not None
                else np.zeros(expected_dim)
            )
            avg_vec_mag = np.linalg.norm(avg_vec)

            min_f1 = (
                basename(all_files[acc.min_pair_idx[0]]) if acc.min_pair_idx else "N/A"
            )
            min_f2 = (
                basename(all_files[acc.min_pair_idx[1]]) if acc.min_pair_idx else "N/A"
            )
            max_f1 = (
                basename(all_files[acc.max_pair_idx[0]]) if acc.max_pair_idx else "N/A"
            )
            max_f2 = (
                basename(all_files[acc.max_pair_idx[1]]) if acc.max_pair_idx else "N/A"
            )

            stats = {
                "min_dist": acc.min_dist,
                "min_files": (min_f1, min_f2),
                "max_dist": acc.max_dist,
                "max_files": (max_f1, max_f2),
                "avg_dist": avg,
                "median_dist": "N/A",
                "std_dev_dist": std_dev,
                "avg_vector_mag": avg_vec_mag,
                "pair_count": count,
                "file_count": len(track_mapping_indices.get(track, [])),
            }
            print(
                f"Track '{track}' ({stats['file_count']} files, {stats['pair_count']} pairs):"
            )
            print(
                f"  Smallest Distance: {stats['min_dist']:.4f} between {stats['min_files'][0]} and {stats['min_files'][1]}"
            )
            print(
                f"  Largest Distance: {stats['max_dist']:.4f} between {stats['max_files'][0]} and {stats['max_files'][1]}"
            )
            print(f"  Average Distance: {stats['avg_dist']:.4f}")
            print("  Median Distance: N/A")
            print(f"  Std Dev Distance: {stats['std_dev_dist']:.4f}")
            print(f"  Avg Vector Magnitude: {stats['avg_vector_mag']:.4f}")
        else:
            stats = {
                "pair_count": 0,
                "file_count": len(track_mapping_indices.get(track, [])),
            }
            print(
                f"Track '{track}' ({stats['file_count']} files, 0 pairs): No pairs processed."
            )

        final_track_stats_report[track] = stats

    for track, indices in track_mapping_indices.items():
        if track not in processed_tracks:
            file_count = len(indices)
            print(
                f"Track '{track}' ({file_count} files, 0 pairs): Skipped (no pairs processed)."
            )
            final_track_stats_report[track] = {
                "pair_count": 0,
                "file_count": file_count,
            }

    print("--- Inter vs Intra Track Statistics ---")
    avg_intra_track_dist = (
        final_intra_track_sum_dist / final_intra_track_count
        if final_intra_track_count > 0
        else 0
    )
    avg_inter_track_dist = (
        final_inter_track_sum_dist / final_inter_track_count
        if final_inter_track_count > 0
        else 0
    )

    print(
        f"Average Intra-Track Distance: {avg_intra_track_dist:.4f} (calculated over {final_intra_track_count} pairs)"
    )
    print(
        f"Average Inter-Track Distance: {avg_inter_track_dist:.4f} (calculated over {final_inter_track_count} pairs)"
    )

    print("--- Generating Plot ---")
    plot_filename = "distance_distribution.png"
    plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)

    total_counted_in_hist = final_hist_counts.sum()
    if total_counted_in_hist > 0 or final_hist_underflow > 0 or final_hist_overflow > 0:
        plt.figure(figsize=(12, 7))
        plt.stairs(
            final_hist_counts,
            bin_edges,
            fill=True,
            color="skyblue",
            edgecolor="black",
            label="Frequency",
        )
        plot_title = "Distribution of Pairwise Euclidean Distances"
        plt.title(plot_title)
        plt.xlabel("Euclidean Distance Bins")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)

        if final_global_count > 0:
            plt.axvline(
                global_avg_dist,
                color="r",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean: {global_avg_dist:.2f}",
            )

        info_text = f"Hist Range: [{hist_min_range:.2f}, {hist_max_range:.2f})\n"
        info_text += f"Counts in Range: {total_counted_in_hist}\n"
        if final_hist_underflow > 0:
            info_text += f"Counts < {hist_min_range:.2f}: {final_hist_underflow}\n"
        if final_hist_overflow > 0:
            info_text += f"Counts >= {hist_max_range:.2f}: {final_hist_overflow}"

        plt.text(
            0.97,
            0.97,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.legend(loc="upper left")
        plt.tight_layout(rect=(0.0, 0.0, 0.95, 1.0))
        try:
            plt.savefig(plot_filepath)
            print(f"Distance distribution plot saved to: {plot_filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close()
    else:
        print("No distances counted in histogram range, skipping plot generation.")

    print("--- Generating Markdown Report ---")
    report_filename = "embedding_distance_report.md"
    report_filepath = os.path.join(OUTPUT_DIR, report_filename)

    try:
        with open(report_filepath, "w") as f:
            f.write("# Embedding Distance Analysis Report\n\n")
            f.write(f"Analysis based on FAISS index: `{FAISS_INDEX_PATH}`\n")
            f.write(f"Data directory: `{DATA_DIR}`\n")
            f.write(f"Number of embeddings analyzed: {N}\n")
            f.write(f"Number of files considered: {len(all_files)}\n")
            f.write(f"Number of tracks identified: {len(track_mapping_indices)}\n")
            f.write(f"Total pairs processed: {final_global_count}\n")
            f.write(f"Parallel workers used: {NUM_WORKERS}\n\n")

            f.write("## Global Statistics\n\n")
            if final_global_count > 0:
                f.write(f"*   **Smallest Distance:** {final_global_min_dist:.4f}\n")
                f.write(f"    *   Files: `{min_file1}`, `{min_file2}`\n")
                f.write(f"*   **Largest Distance:** {final_global_max_dist:.4f}\n")
                f.write(f"    *   Files: `{max_file1}`, `{max_file2}`\n")
                f.write(f"*   **Average Distance:** {global_avg_dist:.4f}\n")
                f.write("*   **Median Distance:** N/A\n")
                f.write(f"*   **Standard Deviation:** {global_std_dev:.4f}\n")
                f.write(
                    f"*   **Average Distance Vector Magnitude:** {global_avg_vector_magnitude:.4f}\n\n"
                )
            else:
                f.write("*   No pairs processed.\n\n")

            f.write("## Distance Distribution Plot\n\n")
            if (
                total_counted_in_hist > 0
                or final_hist_underflow > 0
                or final_hist_overflow > 0
            ):
                f.write(
                    f"The distribution of pairwise Euclidean distances (based on {final_global_count} pairs) is plotted below.\n"
                )
                f.write(
                    f"Histogram generated using {num_bins} bins over the fixed range [{hist_min_range:.2f}, {hist_max_range:.2f}).\n\n"
                )
                if final_hist_underflow > 0:
                    f.write(
                        f"*   Note: {final_hist_underflow} distances were below {hist_min_range:.2f}.\n"
                    )
                if final_hist_overflow > 0:
                    f.write(
                        f"*   Note: {final_hist_overflow} distances were at or above {hist_max_range:.2f}.\n"
                    )
                f.write("\n")
                f.write(f"![Distance Distribution](./{plot_filename})\n\n")
            else:
                f.write(
                    "Plot skipped as no distances were counted for the histogram.\n\n"
                )

            f.write("## Track-Specific Statistics\n\n")
            f.write(
                "Statistics calculated for pairs of files within the same track.\n\n"
            )
            for track in sorted(final_track_stats_report.keys()):
                stats = final_track_stats_report[track]
                file_count = stats.get(
                    "file_count", len(track_mapping_indices.get(track, []))
                )
                pair_count = stats.get("pair_count", 0)

                f.write(f"### Track: `{track}` ({file_count} files)\n\n")
                if pair_count > 0:
                    f.write(f"*   Pairs Processed: {pair_count}\n")
                    f.write(
                        f"*   **Smallest Distance:** {stats['min_dist']:.4f} (`{stats['min_files'][0]}`, `{stats['min_files'][1]}`)\n"
                    )
                    f.write(
                        f"*   **Largest Distance:** {stats['max_dist']:.4f} (`{stats['max_files'][0]}`, `{stats['max_files'][1]}`)\n"
                    )
                    f.write(f"*   **Average Distance:** {stats['avg_dist']:.4f}\n")
                    f.write("*   **Median Distance:** N/A\n")
                    f.write(
                        f"*   **Standard Deviation:** {stats['std_dev_dist']:.4f}\n"
                    )
                    f.write(
                        f"*   **Average Vector Magnitude:** {stats['avg_vector_mag']:.4f}\n\n"
                    )
                else:
                    f.write(f"*   Pairs Processed: 0\n")
                    f.write(f"*   Skipped: No pairs processed for this track.\n\n")

            f.write("## Inter vs. Intra-Track Comparison\n\n")
            f.write(
                f"*   **Average Intra-Track Distance:** {avg_intra_track_dist:.4f} (over {final_intra_track_count} pairs)\n"
            )
            f.write(
                f"*   **Average Inter-Track Distance:** {avg_inter_track_dist:.4f} (over {final_inter_track_count} pairs)\n\n"
            )

            if final_intra_track_count > 0 and final_inter_track_count > 0:
                if avg_inter_track_dist > avg_intra_track_dist:
                    obs_str = "Observation: On average, files from different tracks are further apart than files within the same track, suggesting some track-level clustering in the embedding space.\n"
                    f.write(obs_str)
                elif avg_intra_track_dist > avg_inter_track_dist:
                    obs_str = "Observation: On average, files from different tracks are closer than files within the same track. This might indicate that track boundaries are not well-defined in the embedding space or that variation within tracks is high.\n"
                    f.write(obs_str)
                else:
                    obs_str = "Observation: Average intra-track and inter-track distances are very similar.\n"
                    f.write(obs_str)
            else:
                f.write(
                    "Observation: Cannot compare inter/intra distances as one or both categories have 0 pairs.\n"
                )

        print(f"Markdown report saved to: {report_filepath}")

    except Exception as e:
        print(f"Error writing markdown report: {e}")

    print("Analysis complete.")
