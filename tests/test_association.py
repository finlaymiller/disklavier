import os
import random
import numpy as np
import pretty_midi
import faiss
import tempfile
import shutil
from glob import glob
from rich.pretty import pprint

from utils import basename

from ml.specdiff.model import SpectrogramDiffusion, config as specdiff_config

# --- configuration ---
FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
NUM_FILES_TO_SELECT = 10
TRANSPOSE_SEMITONES = 12  # 1 octave up
NEIGHBORS_TO_FIND = 3


# --- helper functions ---
def transpose_midi(input_path, output_path, semitones):
    """
    Transposes a MIDI file and saves the result.

    Parameters
    ----------
    input_path : str
        Path to the input MIDI file.
    output_path : str
        Path to save the transposed MIDI file.
    semitones : int
        Number of semitones to transpose by.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(input_path)
        for instrument in midi_data.instruments:
            # only transpose non-drum instruments
            if not instrument.is_drum:
                for note in instrument.notes:
                    note.pitch += semitones
                    # clamp pitch to valid midi range (0-127)
                    note.pitch = max(0, min(127, note.pitch))
        midi_data.write(output_path)
        print(
            f"Transposed '{os.path.basename(input_path)}' by {semitones} semitones to '{os.path.basename(output_path)}'"
        )
    except Exception as e:
        print(f"Error transposing {input_path}: {e}")
        # create an empty midi file if transposition fails to avoid errors downstream
        try:
            pretty_midi.PrettyMIDI().write(output_path)
        except Exception as write_e:
            print(
                f"  Additionally failed to write empty MIDI file to {output_path}: {write_e}"
            )


# --- main script ---
def main():
    index = faiss.read_index(FAISS_INDEX_PATH)
    expected_dim = index.d  # Get expected dimension from FAISS index
    print(
        f"FAISS index loaded. Contains {index.ntotal} vectors of dimension {expected_dim}."
    )

    print(f"\nSearching for files in: {DATA_DIR} ending with 't00s00.mid'")
    all_target_files = glob(os.path.join(DATA_DIR, "*t00s00.mid"))
    all_files = glob(os.path.join(DATA_DIR, "*.mid"))

    selected_files = random.sample(all_target_files, NUM_FILES_TO_SELECT)
    print("\nSelected Files:")
    pprint([os.path.basename(f) for f in selected_files])

    print("\nInitializing Spectrogram Diffusion model...")
    specdiff = SpectrogramDiffusion(specdiff_config, verbose=True)

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="transposed_midi_")
        print(f"\nCreated temporary directory for transposed files: {temp_dir}")
        transposed_files = []
        for i, file_path in enumerate(selected_files):
            base_name = os.path.basename(file_path)
            # ensure unique names in case of identical base names (unlikely)
            transposed_name = f"{os.path.splitext(base_name)[0]}_idx{i}_transposed_{TRANSPOSE_SEMITONES}.mid"
            transposed_path = os.path.join(temp_dir, transposed_name)
            transpose_midi(file_path, transposed_path, TRANSPOSE_SEMITONES)
            transposed_files.append(transposed_path)

        # --- Generate Original Embeddings ---
        print("\nGenerating embeddings for original files...")
        original_embeddings_list = []
        skipped_original_indices = set()
        for i, f in enumerate(selected_files):
            emb_tensor = specdiff.embed(f)
            # convert tensor to numpy
            emb = emb_tensor.detach().cpu().numpy()
            if emb.shape == (1, expected_dim):
                original_embeddings_list.append(emb.squeeze())
            elif emb.shape == (expected_dim,):
                original_embeddings_list.append(emb)

        original_embeddings = np.array(original_embeddings_list)
        # Adjust selected_files and transposed_files lists to remove skipped items
        # We need a mapping from the new index in original_embeddings to the original selected_files index
        original_indices_map = [
            i for i in range(NUM_FILES_TO_SELECT) if i not in skipped_original_indices
        ]
        selected_files = [selected_files[i] for i in original_indices_map]
        transposed_files = [
            transposed_files[i] for i in original_indices_map
        ]  # Keep original transposed paths list aligned
        print(
            f"Successfully generated {len(original_embeddings)} original embeddings. Final shape: {original_embeddings.shape}"
        )  # Should be (N, expected_dim)

        # --- Generate Transposed Embeddings ---
        print("\nGenerating embeddings for transposed files...")
        transposed_embeddings_dict = (
            {}
        )  # maps original index (from original_indices_map) to embedding
        valid_transposed_files_info = []  # Store tuples of (original_index, path)

        # Identify valid transposed files *after* potentially skipping originals
        for new_idx, original_idx in enumerate(original_indices_map):
            # Get the correct transposed file path using the new index for the filtered transposed_files list
            t_path = transposed_files[new_idx]
            if os.path.exists(t_path) and os.path.getsize(t_path) > 0:
                valid_transposed_files_info.append((original_idx, t_path))
            else:
                # Use new_idx to get the corresponding original file name from the filtered selected_files list
                original_file_name = (
                    os.path.basename(selected_files[new_idx])
                    if new_idx < len(selected_files)
                    else "Unknown"
                )
                print(
                    f"Warning: Transposed file {os.path.basename(t_path)} is missing or empty (original: {original_file_name}). Will be skipped."
                )

        print(
            f"Attempting to generate embeddings for {len(valid_transposed_files_info)} valid transposed files..."
        )
        processed_count = 0
        for original_idx, t_path in valid_transposed_files_info:
            emb_tensor = specdiff.embed(t_path)
            # convert tensor to numpy
            emb = emb_tensor.detach().cpu().numpy()
            if emb.shape == (1, expected_dim):
                transposed_embeddings_dict[original_idx] = emb.squeeze()
                processed_count += 1
            elif emb.shape == (expected_dim,):
                transposed_embeddings_dict[original_idx] = emb
                processed_count += 1

        print(f"Successfully generated {processed_count} transposed embeddings.")

        # Check if we have enough embeddings to proceed, especially the base file
        current_num_files = len(
            original_embeddings
        )  # Number of files we actually have embeddings for

        # --- Check Base File Transposition ---
        # The base file is the *last* one in the *successfully processed* original embeddings list
        base_file_original_index = original_indices_map[
            -1
        ]  # Original index of the last successfully processed file

        # Print base file info only if we are sure it's valid and has a transposed version
        print(
            f"\nBase file for analysis (last successfully processed original with valid transposed): {os.path.basename(selected_files[-1])}"
        )

        # --- calculations ---
        print("\n--- Association Analysis ---")

        base_embedding_idx_in_array = (
            current_num_files - 1
        )  # Index in the actual original_embeddings array
        base_file_embedding = original_embeddings[base_embedding_idx_in_array]
        transposed_base_file_embedding = transposed_embeddings_dict[
            base_file_original_index
        ]  # Fetch using original index

        # calculate the vector representing the transposition of the base file
        transpose_vector = transposed_base_file_embedding - base_file_embedding
        transpose_distance = np.linalg.norm(transpose_vector)
        print(f"\nBase File Details:")
        print(f"  - Path: {os.path.basename(selected_files[-1])}")
        print(f"  - Original Index: {base_file_original_index}")
        print(f"  - Index in Processed Array: {base_embedding_idx_in_array}")
        print(
            f"  - distance (base file <-> transposed base file): {transpose_distance:.4f}"
        )

        # Iterate through the files *before* the base file in the processed list
        for i in range(current_num_files - 1):
            query_original_index = original_indices_map[i]

            print(
                f"\n--- Analyzing Pair (Query Idx {i}, Original Idx {query_original_index} vs Base Original Idx {base_file_original_index}) ---"
            )
            query_file_path = selected_files[i]
            transposed_query_file_path = transposed_files[i]
            query_embedding = original_embeddings[i]
            transposed_query_embedding = transposed_embeddings_dict[
                query_original_index
            ]

            print(f"Query File: {os.path.basename(query_file_path)}")
            # 6.i. calculate the distance between the embedding of the file and the embedding of last of the original 10 files
            base_distance = np.linalg.norm(query_embedding - base_file_embedding)
            print(f"  Distance (original query <-> original base): {base_distance:.4f}")

            # Interpret 6.ii and 6.iii: predict transposed location and find error
            # Predicted location = original query embedding + transpose vector derived from base file
            predicted_transposed_embedding = query_embedding + transpose_vector
            # ensure it's float32 and reshaped for faiss search (shape must be (1, D) for search)
            predicted_transposed_embedding_faiss = (
                predicted_transposed_embedding.astype(np.float32).reshape(1, -1)
            )

            # 6.iii. find the distance between the *predicted* transposed embedding and the *actual* transposed embedding
            prediction_error_distance = np.linalg.norm(
                transposed_query_embedding - predicted_transposed_embedding
            )
            # Added print for clarity on step 6.ii implicitly used in prediction
            print(
                f"  6.ii. Base Transpose Vector applied to Query Embedding (Result used in 6.iii & 6.iv)"
            )
            print(
                f"  6.iii. Prediction Error Distance (actual transposed query <-> predicted transposed query): {prediction_error_distance:.4f}"
            )

            # (Goal Check) also calculate the distance between the actual transposed query and actual transposed base
            actual_transposed_distance = np.linalg.norm(
                transposed_query_embedding - transposed_base_file_embedding
            )
            print(
                f"     (Goal Check) Distance (actual transposed query <-> actual transposed base): {actual_transposed_distance:.4f}"
            )
            print(
                f"     (Goal Check) Difference between original distance (6.i) and transposed distance: {abs(base_distance - actual_transposed_distance):.4f}"
            )

            # 6.iv. print the 3 nearest neighbors to the vector resulting from prediction
            print(
                f"\n  6.iv. Finding {NEIGHBORS_TO_FIND} nearest neighbors in FAISS index to the *predicted* transposed vector..."
            )
            if predicted_transposed_embedding_faiss.shape[1] != index.d:
                print(
                    f"   Error: Predicted vector dimension ({predicted_transposed_embedding_faiss.shape[1]}) doesn't match FAISS index ({index.d}). Skipping search."
                )
                continue  # Skip search for this pair

            try:
                distances, indices = index.search(
                    predicted_transposed_embedding_faiss, NEIGHBORS_TO_FIND
                )

                print(
                    f"   Nearest Neighbors to Predicted Transposition of '{os.path.basename(query_file_path)}':"
                )
                if (
                    indices.size > 0 and indices[0, 0] != -1
                ):  # faiss returns -1 if no neighbors found within range (if using IndexIVF etc.)
                    for j in range(min(NEIGHBORS_TO_FIND, len(indices[0]))):
                        neighbor_index = indices[0, j]
                        if neighbor_index == -1:
                            continue  # skip if index is -1
                        neighbor_distance = distances[0, j]
                        # we don't know the file names corresponding to the indices in the FAISS index
                        # unless we load a mapping file or assume the index corresponds directly to some known list.
                        # just printing index and distance for now.
                        print(
                            f"    - Neighbor {j+1}: {basename(all_files[neighbor_index])}, Distance {neighbor_distance:.4f}"
                        )
                else:
                    print(
                        "    - No valid neighbors found in the index for the predicted vector."
                    )
            except Exception as e:
                print(f"   Error during FAISS search: {e}")

            # 6.v. Calculate the vector pointing back towards the original query
            print(
                f"\n  6.v. Calculating vector pointing back to original query embedding..."
            )
            # Target vector = actual transposed query - transpose vector (derived from base)
            reverse_transposed_vector = transposed_query_embedding - transpose_vector
            reverse_transposed_vector_faiss = reverse_transposed_vector.astype(
                np.float32
            ).reshape(1, -1)
            # print(f"     Target Vector (first 5 dims): {reverse_transposed_vector[:5]}...") # Optional debug print

            # Find neighbors near this 'reverse transposed' vector
            print(
                f"     Finding {NEIGHBORS_TO_FIND} nearest neighbors in FAISS index to this 'reverse transposed' vector..."
            )
            if reverse_transposed_vector_faiss.shape[1] != index.d:
                print(
                    f"     Error: Reverse transposed vector dimension ({reverse_transposed_vector_faiss.shape[1]}) doesn't match FAISS index ({index.d}). Skipping search."
                )
            else:
                try:
                    distances_rev, indices_rev = index.search(
                        reverse_transposed_vector_faiss, NEIGHBORS_TO_FIND
                    )

                    print(
                        f"     Nearest Neighbors to Reverse Transposed Vector of '{os.path.basename(query_file_path)}':"
                    )
                    if indices_rev.size > 0 and indices_rev[0, 0] != -1:
                        for j in range(min(NEIGHBORS_TO_FIND, len(indices_rev[0]))):
                            neighbor_index_rev = indices_rev[0, j]
                            if neighbor_index_rev == -1:
                                continue
                            neighbor_distance_rev = distances_rev[0, j]
                            # Use the user's added all_files list and basename function
                            if 0 <= neighbor_index_rev < len(all_files):
                                neighbor_filename = basename(
                                    all_files[neighbor_index_rev]
                                )
                            else:
                                neighbor_filename = f"Index {neighbor_index_rev} (Out of Bounds for all_files list)"
                            print(
                                f"      - Neighbor {j+1}: {neighbor_filename}, Distance {neighbor_distance_rev:.4f}"
                            )

                    else:
                        print(
                            "      - No valid neighbors found in the index for the reverse transposed vector."
                        )
                except Exception as e:
                    print(
                        f"     Error during FAISS search for reverse transposed vector: {e}"
                    )

    finally:
        # --- cleanup ---
        if temp_dir and os.path.exists(temp_dir):
            print(f"\nCleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print("Temporary directory removed.")
            except Exception as e:
                print(f"Error removing temporary directory {temp_dir}: {e}")


if __name__ == "__main__":
    main()

"""
import os
import sys

from ml.specdiff.model import SpectrogramDiffusion

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import faiss
import pretty_midi
import numpy as np
from glob import glob

from ml.specdiff.model import SpectrogramDiffusion, config as specdiff_config
from utils import basename, console, midi

np.random.seed(0)

dataset_path = os.path.join("data", "datasets", "20250421", "augmented")
table_path = os.path.join("data", "tables", "20250421")
pf_augmentations = os.path.join("outputs", "augmentations")
tmp_dir = "outputs/tmp"
os.makedirs(tmp_dir, exist_ok=True)

seed_rearrange = True
seed_remove = 0.25

faiss_index = faiss.read_index(os.path.join(table_path, "specdiff.faiss"))
specdiff = SpectrogramDiffusion(specdiff_config, verbose=True)

all_files = glob(os.path.join(dataset_path, "*.mid"))
all_files.sort()

chosen_files = np.random.choice(all_files, size=2, replace=False)
file_data = {}
for chosen_file in chosen_files:
    midi_pm = pretty_midi.PrettyMIDI(chosen_file)
    for instrument in midi_pm.instruments:
        for note in instrument.notes:
            note.pitch += 12
    tpose_path = os.path.join(tmp_dir, basename(chosen_file) + "_tposed.mid")
    midi_pm.write(tpose_path)
    file_data[basename(chosen_file)] = {
        "og_midi": chosen_file,
        "og_embedding": specdiff.embed(chosen_file),
        "tposed_midi": tpose_path,
        "tposed_embedding": specdiff.embed(tpose_path),
    }
    file_data[basename(chosen_file)]["og_embedding"] /= np.linalg.norm(
        file_data[basename(chosen_file)]["og_embedding"], axis=1, keepdims=True
    )
    file_data[basename(chosen_file)]["tposed_embedding"] /= np.linalg.norm(
        file_data[basename(chosen_file)]["tposed_embedding"], axis=1, keepdims=True
    )
"""