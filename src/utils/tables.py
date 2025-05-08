import os
import time
import h5py
import json
import faiss
import torch
import numpy as np
import pandas as pd
import networkx as nx
from glob import glob
from pretty_midi import PrettyMIDI
from collections import defaultdict
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from torch.utils.data import DataLoader, TensorDataset


from typing import List
from . import basename, console

from ml.specdiff.model import SpectrogramDiffusion, DEFAULT_CONFIG
from ml.classifier.model import Classifier
from ml.clamp.model import Clamp
from ml.clamp.clamp_utils import M3_HIDDEN_SIZE


def build_neighbor_table(
    all_files: List[str], output_path: str, hdf_key: str = "neighbors"
) -> bool:
    """
    Build a table of segment neighbors. HDF dataset key is "neighbors" by default.

    Parameters
    ----------
    all_files : List[str]
        List of all files to build the table from.
    output_path : str
        Path to save the table to.
    hdf_key : str
        Key to save the table to in the HDF file.

    Returns
    -------
    bool
        True if the table was built successfully, False otherwise.
    """
    column_names = ["prev_2", "prev", "current", "next", "next_2"]
    n_table = pd.DataFrame(
        index=[basename(file) for file in all_files], columns=column_names
    )

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    )
    update_task = progress.add_task(f"gathering neighbors", total=len(all_files))
    with progress:
        for i, file in enumerate(all_files):
            neighbors = []
            curr_track, _ = file.split("_")
            for offset in range(-2, 3):
                idx = i + offset
                valid = 0 <= idx < len(all_files)
                filename = (
                    basename(all_files[idx])
                    if valid and all_files[idx].split("_")[0] == curr_track
                    else None
                )
                neighbors.append(filename)

            n_table.loc[basename(file)] = neighbors
            progress.advance(update_task)

    try:
        n_table.to_hdf(output_path, key=hdf_key, mode="w")
    except Exception as e:
        console.print_exception(show_locals=True)
        return False

    console.log(n_table.head())

    return os.path.exists(output_path)


def pitch_histograms(all_files: List[str], output_path: str) -> bool:
    """
    Calculate the pitch histograms for all files. output is a hdf5 file with keys "filenames" and "pitch-histogram" where the latter is a numpy array of shape (n_files, 12). pitch histograms are weighted by duration and velocity and are only calculated once per beat shift to speed things up a bit.

    Parameters
    ----------
    all_files : List[str]
        List of all files to calculate the pitch histograms for.
    output_path : str
        Path to save the pitch histograms to.

    Returns
    -------
    bool
        True if the pitch histograms were calculated successfully, False otherwise.
    """

    num_files = len(all_files)
    # initialize faiss index
    faiss_path = os.path.join(os.path.dirname(output_path), "pitch-histogram.faiss")
    index = faiss.IndexFlatIP(12)
    vecs = np.zeros((num_files, 12), dtype=np.float32)

    with h5py.File(output_path, "w") as f:
        console.log(f"creating pitch histograms for {num_files} files")

        d_filenames = f.create_dataset(
            "filenames",
            (num_files),
            dtype=h5py.string_dtype(encoding="utf-8"),
            fillvalue="",
        )
        d_histograms = f.create_dataset(
            "embeddings",
            shape=(num_files, 12),
            dtype=np.float32,
            fillvalue=np.zeros(12),
        )

        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
        )
        update_task = progress.add_task(f"generating pitch histograms", total=num_files)
        with progress:
            for i, file in enumerate(all_files):
                ph = PrettyMIDI(file).get_pitch_class_histogram(True, True)
                d_histograms[i] = ph
                d_filenames[i] = basename(file)
                vecs[i] = ph
                progress.advance(update_task)

        console.log("copying vectors to FAISS index")
        index.add(vecs)  # type: ignore
        faiss.write_index(index, faiss_path)
        console.log(f"FAISS index saved to '{faiss_path}'")

    return _check_hdf5(output_path, ["filenames", "embeddings"])


def specdiff(
    all_files: List[str],
    output_path: str,
    fix_time: bool = True,
    batch_size: int = 1,
    device_name: str = "cuda:0",
) -> bool:
    """
    Calculate the spectrogram diffusion embedding for input files.

    Parameters
    ----------
    all_files : List[str]
        List of all files to calculate the specdiff for.
    output_path : str
        Path to save the specdiff to.
    fix_time : Optional[bool]
        Whether to change the tempo of the files such that one segment
        fits exactly into one tokenization window.
    device_name : str
        Device to run the specdiff on.

    Returns
    -------
    bool
        True if the specdiff was calculated successfully, False otherwise.
    """

    num_files = len(all_files)
    all_files.sort()
    # initialize encoder
    DEFAULT_CONFIG["device"] = device_name
    model = SpectrogramDiffusion(DEFAULT_CONFIG, fix_time)

    # initialize faiss index
    faiss_path = os.path.join(os.path.dirname(output_path), "specdiff.faiss")
    index = faiss.IndexFlatIP(DEFAULT_CONFIG["encoder_config"]["d_model"])
    vecs = np.zeros(
        (num_files, DEFAULT_CONFIG["encoder_config"]["d_model"]), dtype=np.float32
    )

    with h5py.File(output_path, "w") as out_file:
        # create output datasets
        d_embeddings = out_file.create_dataset(
            "embeddings",
            (num_files, DEFAULT_CONFIG["encoder_config"]["d_model"]),
            fillvalue=0,
        )
        d_filenames = out_file.create_dataset(
            "filenames",
            (num_files),
            dtype=h5py.string_dtype(encoding="utf-8"),
            fillvalue="",
        )

        # calculate embeddings
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            refresh_per_second=1,
        )
        emb_task = progress.add_task("embedding", total=num_files)
        with progress:
            for i in range(0, num_files, batch_size):
                embeddings = []
                for file in all_files[i : i + batch_size]:
                    embeddings.append(model.embed(file))
                embeddings = torch.cat(embeddings)

                d_embeddings[i : i + batch_size] = embeddings
                vecs[i : i + batch_size] = [
                    e / float(np.linalg.norm(e, keepdims=True)) for e in embeddings
                ]
                d_filenames[i : i + batch_size] = [
                    basename(f) for f in all_files[i : i + batch_size]
                ]
                progress.advance(emb_task, batch_size)

        console.log("copying vectors to FAISS index")
        index.add(vecs)  # type: ignore
        faiss.write_index(index, faiss_path)
        console.log(f"FAISS index saved to '{faiss_path}'")

    return _check_hdf5(output_path, ["filenames", "embeddings"])


def classifier(
    type: str,
    all_files: List[str],
    output_path: str,
    model_path: str,
    batch_size: int = 1,
    device_name: str = "cuda:0",
) -> bool:
    num_files = len(all_files)
    device = torch.device(device_name)

    # FAISS index
    index = faiss.IndexFlatIP(128)
    vecs = np.zeros((num_files, 128), dtype=np.float32)

    # load embeddings
    embeddings_path = os.path.join(os.path.dirname(output_path), "specdiff.h5")
    if not os.path.exists(embeddings_path):
        console.log(f"couldn't find embeddings at '{embeddings_path}'")
        return False
    console.log(f"loading embeddings from '{embeddings_path}'")
    # load input datasets
    with h5py.File(embeddings_path, "r") as in_file:
        filenames: h5py.Dataset = in_file["filenames"][:]  # type: ignore
        console.log(f"processing {num_files} files, e.g.:\n{filenames[:5]}")
        embeddings: h5py.Dataset = in_file["embeddings"][:]  # type: ignore
        console.log(f"processing {num_files} files, e.g.:\n{embeddings[:5]}")

    # load the classifier
    console.log(f"loading classifier from '{model_path}'")
    clf = Classifier(768, [128], 120).cuda(device=device)
    clf.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    clf.eval()

    # generate embeddings
    with h5py.File(output_path, "w") as f:
        dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        updated_embeddings = []
        with torch.no_grad():
            progress = Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                refresh_per_second=1,
            )
            update_task = progress.add_task("classifying", total=num_files)
            for i, batch in enumerate(dataloader):
                batch = batch[0].cuda(device=device)
                hidden_output = clf(batch, return_hidden=True).cpu().numpy()
                updated_embeddings.append(hidden_output)
                vecs[i : i + batch_size] = hidden_output  # TODO: normalize
                progress.advance(update_task, batch_size)
        updated_embeddings = np.vstack(updated_embeddings)

        # save to new HDF5 file
        console.log(f"saving embeddings to '{output_path}'")
        f.create_dataset("embeddings", data=updated_embeddings)
        f.create_dataset("filenames", data=filenames)

    # save index
    console.log("copying vectors to FAISS index")
    index.add(vecs)  # type: ignore
    faiss_path = os.path.join(os.path.dirname(output_path), f"{type}.faiss")
    faiss.write_index(index, faiss_path)
    console.log(f"FAISS index saved to '{faiss_path}'")

    return _check_hdf5(output_path, ["filenames", "embeddings"])


def clamp(all_files: List[str], output_path: str, device_name: str = "cuda:0") -> bool:
    """
    Calculate the clamp embedding for all files.
    """
    num_files = len(all_files)
    all_files.sort()
    model = Clamp(device_name)

    # initialize faiss index
    faiss_path = os.path.join(os.path.dirname(output_path), "clamp.faiss")
    index = faiss.IndexFlatIP(M3_HIDDEN_SIZE)
    vecs = np.zeros((num_files, M3_HIDDEN_SIZE), dtype=np.float32)

    with h5py.File(output_path, "w") as out_file:
        # create output datasets
        d_embeddings = out_file.create_dataset(
            "embeddings",
            (num_files, M3_HIDDEN_SIZE),
            fillvalue=0,
        )
        d_filenames = out_file.create_dataset(
            "filenames",
            (num_files),
            dtype=h5py.string_dtype(encoding="utf-8"),
            fillvalue="",
        )

        # calculate embeddings
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            refresh_per_second=1,
        )
        emb_task = progress.add_task("embedding", total=num_files)
        with progress:
            for i in range(num_files):
                embedding = model.embed(all_files[i])
                # console.log(f"embedding {i} of {num_files} has shape {embedding.shape}")
                # console.log(embedding)
                d_embeddings[i] = embedding
                vecs[i] = embedding / float(np.linalg.norm(embedding, keepdims=True))
                d_filenames[i] = basename(all_files[i])
                progress.advance(emb_task)

        console.log("copying vectors to FAISS index")
        index.add(vecs)  # type: ignore
        faiss.write_index(index, faiss_path)
        console.log(f"FAISS index saved to '{faiss_path}'")

    return _check_hdf5(output_path, ["filenames", "embeddings"])


def _check_hdf5(path: str, keys: List[str]) -> bool:
    with h5py.File(path, "r") as f:
        console.log(f"checking HDF5 file with keys {f.keys()} for keys: {keys}")
        for key in keys:
            if key not in f:
                console.log(f"key {key} not found in HDF5 file")
                return False

            console.log(f"{key}:")
            if key == "filenames":
                for filename in f[key][:5]:  # type: ignore
                    console.log(f"\t{str(filename, 'utf-8')}")
            else:
                for val in f[key][:5]:  # type: ignore
                    console.log(f"\t{val.shape}")
        return True


def graph(
    embeddings_path: str, data_dir: str, output_path: str, top_k: int = 1000
) -> int:
    n_graphs = 0

    console.log(f"loading and normalizing embeddings from '{embeddings_path}'")
    with h5py.File(embeddings_path, "r") as f:
        embeddings = [e / np.linalg.norm(e, keepdims=True) for e in f["embeddings"]]  # type: ignore
        filenames = [str(name, "utf-8") for name in f["filenames"]]  # type: ignore
    df = pd.DataFrame({"embeddings": embeddings}, index=filenames, dtype=np.float32)

    console.log("grouping files")
    grouped_files = defaultdict(list[str])
    for filename in glob(os.path.join(data_dir, "*.mid")):
        track_name = basename(filename).split("_")[0]
        grouped_files[track_name].append(basename(filename))

    console.log(f"grouped {len(grouped_files)} tracks:")
    for track, files in list(grouped_files.items())[-3:]:
        console.log(f"\t{track}: {len(files):04d} files")
        console.log(f"\t{files[:3]}")

    console.log("building graphs")
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=2,
    )
    prim_task = progress.add_task("generating graphs", total=len(grouped_files.items()))
    debug_print = False
    with progress:
        for track, files in grouped_files.items():
            print(f"building graph for track {track} with {len(files)} files")

            emb_group = np.stack([df.loc[basename(file), "embeddings"] for file in files])  # type: ignore
            # compute cosine similarity matrix (dot product works since embeddings are normalized)
            similarity = emb_group.dot(emb_group.T)
            # exclude self-similarity by setting the diagonal to -infinity
            np.fill_diagonal(similarity, -np.inf)

            # for each node, get indices of top_k neighbors with highest cosine similarity
            neighbors = np.argsort(-similarity, axis=1)[:, :top_k]

            edges = []
            sec_task = progress.add_task("connecting nodes", total=len(files))
            for i in range(len(files)):
                for j in neighbors[i]:
                    if i < j:
                        if float(1 - similarity[i, j]) < 0:
                            raise ValueError(f"similarity is negative: {similarity[i, j]}")
                        edges.append((files[i], files[j], float(1 - similarity[i, j])))
                progress.advance(sec_task)

            if not debug_print:
                console.log(f"printing some debug stats for {track}")
                console.log(f"\t{len(files)} files, e.g.:\n\t{files[:5]}")
                console.log(f"\t{len(edges)} edges, e.g.:\n\t{edges[:5]}")
                console.log(
                    f"emb_group.shape: {emb_group.shape}, e.g.:\n\t{emb_group[:5]}"
                )
                console.log(f"\t{len(neighbors)} neighbors, e.g.:\n\t{neighbors[:5]}")
                console.log(f"\t{len(similarity)} similarity matrix")
                console.log(f"\t{similarity}")
                debug_print = True

            graph = nx.Graph(name=track)
            graph.add_nodes_from(files)
            graph.add_weighted_edges_from(edges)

            with open(f"{output_path}/{track}.json", "w") as f:
                json.dump(nx.node_link_data(graph, edges="edges"), f)  # type: ignore
            del graph

            n_graphs += 1
            progress.remove_task(sec_task)
            progress.advance(prim_task)

    return n_graphs
