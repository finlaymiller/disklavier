import os
import sys
import random
import numpy as np
import faiss
import tempfile
from shutil import rmtree
from glob import glob
from rich.pretty import pprint
from sklearn.metrics.pairwise import cosine_similarity

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils import basename
from src.utils.midi import transpose_midi
from src.ml.specdiff.model import SpectrogramDiffusion

# --- configuration ---
FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
NUM_FILES_TO_SELECT = 10
TRANSPOSE_SEMITONES = 12  # 1 octave up

random.seed(0)
index = faiss.read_index(FAISS_INDEX_PATH)
expected_dim = index.d  # Get expected dimension from FAISS index
print(
    f"FAISS index loaded. Contains {index.ntotal} vectors of dimension {expected_dim}."
)

all_target_files = glob(os.path.join(DATA_DIR, "*t00s00.mid"))
all_target_files.sort()
all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()

# select files
selected_files = random.sample(all_target_files, NUM_FILES_TO_SELECT)
print("\nselected files:")
pprint([os.path.basename(f) for f in selected_files])

# init model
print("\nInitializing Spectrogram Diffusion model...")
specdiff = SpectrogramDiffusion(verbose=True)

# generate transposed files
temp_dir = None
temp_dir = tempfile.mkdtemp(prefix="transposed_midi_")
print(f"\nCreated temporary directory for transposed files: {temp_dir}")
transposed_files = []
for i, file_path in enumerate(selected_files):
    base_name = os.path.basename(file_path)
    # ensure unique names in case of identical base names (unlikely)
    transposed_name = (
        f"{os.path.splitext(base_name)[0]}_idx{i}_transposed_{TRANSPOSE_SEMITONES}.mid"
    )
    transposed_path = os.path.join(temp_dir, transposed_name)
    transpose_midi(file_path, transposed_path, TRANSPOSE_SEMITONES)
    transposed_files.append(transposed_path)

# generate embeddings
original_embeddings_list = []
for file in selected_files:
    embedding = specdiff.embed(file)
    embedding = embedding.detach().cpu().numpy().squeeze()
    original_embeddings_list.append(embedding)
original_embeddings = np.array(original_embeddings_list)
print("generated embeddings:")
pprint(original_embeddings.shape)

# generate transposed embeddings
transposed_embeddings_list = []
for file in transposed_files:
    embedding = specdiff.embed(file)
    embedding = embedding.detach().cpu().numpy()
    transposed_embeddings_list.append(embedding)
transposed_embeddings = np.array(transposed_embeddings_list).squeeze()
print("generated transposed embeddings:")
pprint(transposed_embeddings.shape)

# find distances between original and transposed embeddings
normed_original_embeddings = original_embeddings / np.linalg.norm(original_embeddings, axis=1, keepdims=True)
normed_transposed_embeddings = transposed_embeddings / np.linalg.norm(transposed_embeddings, axis=1, keepdims=True)
cosine_distances = cosine_similarity(normed_original_embeddings, normed_transposed_embeddings)
print("cosine distances:")
pprint(cosine_distances.shape)
separation_vectors = original_embeddings - transposed_embeddings
print("separation vectors:")
pprint(separation_vectors.shape)
normed_separation_vectors = normed_original_embeddings - normed_transposed_embeddings
print("normed_separation vectors:")
pprint(normed_separation_vectors.shape)

for i, (normed_separation_vector, separation_vector, cosine_distance) in enumerate(zip(normed_separation_vectors, separation_vectors, cosine_distances)):
    print(
        f"'{basename(selected_files[i])}' transpose mag: {np.linalg.norm(separation_vector):.04f} ({np.linalg.norm(normed_separation_vector):.04f}), cosine distance: {cosine_distance}"
    )

# find nearest neighbors

if temp_dir and os.path.exists(temp_dir):
    try:
        rmtree(temp_dir)
        print("Temporary directory removed.")
    except Exception as e:
        print(f"Error removing temporary directory {temp_dir}: {e}")
