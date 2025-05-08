import os
import json
import faiss
import numpy as np
import pandas as pd
import networkx as nx
from glob import glob
from shutil import copy2
from pretty_midi import PrettyMIDI
from scipy.spatial.distance import cosine

from workers.panther import Panther
from typing import Optional

from .worker import Worker
from utils.modes import find_path
from workers.panther import Panther
from utils import basename, console, panther
from utils.constants import SUPPORTED_EXTENSIONS, EMBEDDING_SIZES

from utils.constants import SUPPORTED_EXTENSIONS, EMBEDDING_SIZES
from typing import Optional


class Seeker(Worker):
    # required tables
    filenames: list[str] = []
    faiss_index: faiss.IndexFlatIP
    neighbor_table: pd.DataFrame | pd.Series
    neighbor_col_priorities = ["next", "next_2", "prev", "prev_2"]
    # forced track change interval
    n_transition_interval: int = 8  # 16 bars since segments are 2 bars
    # number of repeats for easy mode
    n_segment_repeats: int = 0
    # playback trackers
    played_files: list[str] = []
    allow_multiple_plays = False
    # playlist mode position tracker
    matches_pos: int = 0
    # TODO: i forget what this does tbh, rewrite entire playlist mode
    matches_mode: str = "cplx"
    playlist: dict[str, dict[str, list[tuple[str, float]]]] = {}
    # force pitch match after finding next segment
    pitch_match = False
    # path tracking variables
    current_path: list[tuple[str, float]] = []
    path_position: int = 0
    num_segs_diff_tracks: int = 5
    # filename to index mapping
    filename_to_index: dict[str, int] = {}

    # velocity tracking
    _recorder = None
    _avg_velocity: float = 0.0

    # reference to panther worker
    panther_worker: Optional[Panther] = None

    def __init__(
        self,
        params,
        aug_path: str,
        table_path: str,
        dataset_path: str,
        playlist_path: str,
        model,
        bpm: int,
    ):
        super().__init__(params, bpm=bpm)
        self.p_aug = aug_path
        self.p_table = table_path
        self.p_dataset = dataset_path
        self.p_playlist = playlist_path
        self.model = model
        self.rng = np.random.default_rng(self.params.seed)
        self.playlist = {}
        self.filenames = []
        self.played_files = []
        self.current_path = []
        self.filename_to_index = {}

        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

        console.log(
            f"{self.tag} loading FAISS index from '{self.p_table}/{self.params.metric}.faiss'"
        )
        self.filenames = sorted(
            [
                basename(f)
                for f in os.listdir(self.p_dataset)
                if f.endswith(SUPPORTED_EXTENSIONS)
            ]
        )

        # create a filename to index mapping for O(1) lookups
        console.log(f"{self.tag} creating filename to index mapping")
        self.filename_to_index = {name: idx for idx, name in enumerate(self.filenames)}
        console.log(
            f"{self.tag} created mapping with {len(self.filename_to_index)} entries"
        )
        for k, v in list(self.filename_to_index.items())[:3]:
            console.log(f"{self.tag}\t\t'{k}' -> {v}")

        # load FAISS index
        self.faiss_index = faiss.read_index(
            os.path.join(self.p_table, f"{self.params.metric}.faiss")
        )
        console.log(f"{self.tag} FAISS index loaded ({self.faiss_index.ntotal})")
        # if self.faiss_index.ntotal > 0:
        #     console.log(f"{self.tag} Checking norms of first few vectors in loaded index:")
        #     for i in range(min(100, self.faiss_index.ntotal)):
        #         try:
        #             vector = self.faiss_index.reconstruct(i)
        #             norm = np.linalg.norm(vector)
        #             console.log(f"{self.tag}   Vector {i} norm: {norm:.4f}")
        #         except Exception as e:
        #             console.log(f"{self.tag} Error reconstructing/checking vector {i}: {e}")

        # load neighbor table
        # old version -- backwards compat
        pf_neighbor_table = os.path.join(self.p_table, "neighbor.parquet")
        if not os.path.isfile(pf_neighbor_table):
            pf_neighbor_table = os.path.join(self.p_table, "neighbors.h5")
        console.log(f"{self.tag} looking for neighbor table '{pf_neighbor_table}'")
        if os.path.isfile(pf_neighbor_table):
            with console.status("\t\t\t      loading neighbor file..."):
                if os.path.splitext(pf_neighbor_table)[1] == ".h5":
                    self.neighbor_table = pd.read_hdf(pf_neighbor_table, key="20250420")
                else:
                    self.neighbor_table = pd.read_parquet(pf_neighbor_table)
                self.neighbor_table.head()
            console.log(
                f"{self.tag} loaded {len(self.neighbor_table)}*{len(self.neighbor_table.columns)} neighbor table"
            )
            console.log(self.neighbor_table.head())
        else:
            console.log(f"{self.tag} error loading neighbor table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        console.log(f"{self.tag} initialization complete")

    def set_recorder(self, recorder):
        """
        Set the reference to the MidiRecorder.

        Parameters
        ----------
        recorder : MidiRecorder
            Reference to the MidiRecorder instance.
        """
        self._recorder = recorder
        console.log(f"{self.tag} connected to recorder for velocity updates")

    def check_velocity_updates(self) -> bool:
        if self._recorder is None:
            console.log(f"{self.tag} no recorder connected")
            return False
        else:
            self._avg_velocity = self._recorder.avg_velocity

            if self.verbose:
                console.log(
                    f"{self.tag} updated velocity stats: avg={self._avg_velocity:.2f}"
                )
            return True

    def get_next(self) -> tuple[str, float]:
        """
        Get the next segment to play, based on the current mode.

        Returns
        -------
        tuple[str, float]
            Path to MIDI file to play next, and similarity measure.
        """
        similarity = 0.0
        match self.params.mode:
            case "best":
                next_file, similarity = self._get_best(hop=False)
            case "timed_hops":
                next_file, similarity = self._get_best(hop=True)
            case "easy":
                next_file = self._get_easy()
            case "playlist":
                next_file, similarity = self._read_playlist()
            case "repeat":
                next_file = self.played_files[-1]
            case "sequential":
                next_file = self._get_neighbor()
            case "graph":
                # TODO: fix similarities if possible
                next_file, similarity = self._get_graph()
            case "random" | "shuffle" | _:
                next_file = self._get_random()

        if self.pitch_match and len(self.played_files) > 0:
            console.log(
                f"{self.tag} pitch matching '{self.base_file(next_file)}' to '{self.played_files[-1]}'"
            )
            if "player" in self.played_files[-1].split("_")[0]:
                base_pch = PrettyMIDI(
                    os.path.join(self.params.pf_recording)
                ).get_pitch_class_histogram(True, True)
            else:
                base_pch = PrettyMIDI(
                    os.path.join(self.p_dataset, self.played_files[-1])
                ).get_pitch_class_histogram(True, True)
            best_match = {"file": None, "sim": -1}
            for pitch in range(12):
                shift = next_file.split("_")[-1][4:]  # also contains .mid
                transposed_file = f"{next_file[:-11]}_t{pitch:02d}s{shift}"
                shifted_pch = PrettyMIDI(transposed_file).get_pitch_class_histogram(
                    use_duration=True, use_velocity=True, normalize=True
                )
                similarity = float(1 - cosine(shifted_pch, base_pch))
                if similarity > best_match["sim"]:
                    best_match["sim"] = similarity
                    best_match["file"] = transposed_file
                    if self.verbose:
                        console.log(f"{self.tag} improved match:\n{best_match}")
            if best_match["file"] == None and self.verbose:
                console.log(f"{self.tag} next file was already optimally pitch matched")
            else:
                next_file = best_match["file"]

        self.played_files.append(os.path.basename(next_file))

        full_path = os.path.join(self.p_dataset, next_file)
        return full_path, similarity

    def get_random(self) -> str:
        """returns a random file from the dataset"""
        random_file = self._get_random()
        return os.path.join(self.p_dataset, random_file)

    def _get_best(self, hop: bool = False) -> tuple[str, float]:
        if self.verbose:
            console.log(
                f"{self.tag} finding most similar file to '{self.played_files[-1]}'"
            )
            console.log(
                f"{self.tag} {len(self.played_files)} played files:\n{self.played_files[-5:]}"
            )

        query_file = basename(self.played_files[-1])

        if self.params.match != "current" and query_file in self.filenames:
            track, segment, augmentation = query_file.split("_")
            new_query_file = self.neighbor_table.loc[
                f"{track}_{segment}", self.params.match
            ]
            new_query_file = f"{new_query_file}_{augmentation}"
            if new_query_file is not None:
                console.log(
                    f"{self.tag} finding match for '{new_query_file}' instead of '{query_file}' due to match mode {self.params.match}"
                )
                query_file = new_query_file

        # load query embedding
        # first, look in existing index
        # then, if no embedding is found, calculate manually
        try:
            # Use dictionary lookup instead of list.index() with fallback
            try:
                embedding = self.faiss_index.reconstruct(int(self.filename_to_index[query_file]))  # type: ignore
            except KeyError:
                embedding = self.faiss_index.reconstruct(int(self.filenames.index(query_file)))  # type: ignore
        except (ValueError, KeyError):
            pf_new = self.played_files[-1]
            # if "player" in pf_new:
            #     pf_new = self.augment_recording(pf_new)

            console.log(
                f"{self.tag}[yellow] unable to find embedding for '{query_file}', calculating manually from '{pf_new}'"
            )

            # Add the new embedding to the index and update filenames list and dictionary
            embedding = self.get_embedding(pf_new)
            embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
            self.faiss_index.add(embedding)  # type: ignore
            self.filenames.append(query_file)
            self.filename_to_index[query_file] = len(self.filenames) - 1
            console.log(
                f"{self.tag} added [dark_red bold]{query_file}[/dark_red bold] to index"
            )
        query_embedding = np.array(
            embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        # query index
        if self.verbose and query_file in self.filenames:
            console.log(f"{self.tag} querying with key '{query_file}'")

        indices, similarities = self._faiss_search(query_embedding)  # type: ignore
        if self.verbose:
            for i in range(3):
                console.log(
                    f"{self.tag}\t{i}: {self.filenames[indices[i]]} {similarities[i]:.05f}"
                )

        # find most similar valid match
        if "player" in query_file:
            next_file = self._get_random()
        else:
            next_file = self._get_neighbor()
        # TODO: fix this (what the hell did i mean by this?)
        played_files = [os.path.basename(self.base_file(f)) for f in self.played_files]
        if self.params.match != "current":
            played_files.append(query_file)
        for idx, similarity in zip(indices, similarities):
            if self.filenames[idx] == query_file:
                # this should always skip the first file
                continue
            segment_name = str(self.filenames[idx])
            if "player" in query_file:
                next_file = f"{segment_name}.mid"
                console.log(f"{self.tag} found player match: '{next_file}'")
                break
            # dont replay files
            if segment_name in played_files:
                continue

            next_segment_name = self.base_file(segment_name)
            next_track = next_segment_name.split("_")[0]
            last_track = self.played_files[-1].split("_")[0]
            # switch to different track after self.n_transition_interval segments
            if hop and self.n_segment_repeats >= self.n_transition_interval:
                played_tracks = [file.split("_")[0] for file in self.played_files]
                if next_track in played_tracks:
                    console.log(
                        f"{self.tag} transitioning to next track and skipping '{next_segment_name}'"
                    )
                    continue
                else:
                    next_file = f"{segment_name}.mid"
                    break
            # no shift because it sounds bad
            if (
                next_segment_name
                not in played_files
                # and segment_name.endswith("s00")
                # and next_track == last_track
            ):
                next_file = f"{segment_name}.mid"
                break

        console.log(
            f"{self.tag} best match is '{next_file}' with similarity {similarity:.05f}"
        )
        return next_file, similarity

    def _get_easy(self) -> str:
        if self.verbose:
            console.log(f"{self.tag} played files: {self.played_files[:-5]}")
            console.log(f"{self.tag} num_repeats: {self.n_segment_repeats}")
        if self.n_segment_repeats < self.n_segment_repeats:
            console.log(f"{self.tag} transitioning to next segment")
            self.n_segment_repeats += 1
            return self._get_neighbor()
        else:
            console.log(f"{self.tag} transitioning to next track")
            self.n_segment_repeats = 0
            return (
                self._get_random()
            )  # TODO: modify this to get nearest neighbor from different track

    def _get_neighbor(self) -> str:
        current_file = os.path.basename(self.played_files[-1])

        # transforms not needed
        # TODO: panther breaks this
        if len(current_file.split("_")) > 2:
            current_file = self.base_file(current_file).split(".")[0]

        try:
            for col_name in self.neighbor_col_priorities:
                neighbor = self.neighbor_table.loc[current_file, col_name]

                # only play files once
                if neighbor in self.played_files and not self.allow_multiple_plays:
                    neighbor = None

                # found a neighbor
                if neighbor != None:
                    if self.verbose:
                        console.log(
                            f"{self.tag} found neighboring file '{neighbor}' at position '{col_name}'"
                        )
                    return str(neighbor)
        except KeyError:
            console.log(
                f"{self.tag} unable to find neighbor for '{current_file}', choosing randomly"
            )
        return self._get_random()

    def _get_graph(self) -> tuple[str, float]:
        """
        Find the nearest segment with a different track using FAISS,
        load the relevant networkx graph, and plot the path.

        Returns
        -------
        tuple[str, float]
            The next segment in the path to play and its similarity value.
        """
        seed_file = self.played_files[-1]
        seed_key = basename(seed_file)
        seed_track = seed_key.split("_")[0]
        console.log(f"{self.tag} using seed file '{seed_file}' for graph navigation")

        try:
            _ = self.filename_to_index[seed_key]
        except KeyError:
            console.log(
                f"{self.tag} unable to find embedding for '{seed_key}', calculating best match"
            )
            next_file, similarity = self._get_best(hop=False)
            return next_file, similarity

        # path already calculated, progress along the path
        if (
            len(self.current_path) > 0
            and self.path_position < len(self.current_path) - 1
        ):
            self.path_position += 1
            next_file, next_sim = self.current_path[self.path_position]
            console.log(
                f"{self.tag} returning file [{self.path_position}/{len(self.current_path)}] '{next_file}' with similarity {next_sim:.05f}"
            )
            console.log(self.current_path[-5:])
            return next_file, next_sim

        e = self.faiss_index.reconstruct(self.filename_to_index[seed_key])  # type: ignore
        seed_embedding = np.array([e])

        # find nearest segments using FAISS
        console.log(
            f"{self.tag} searching for nearest segment from a different track with embedding {seed_embedding.shape}"
        )
        indices, similarities = self._faiss_search(seed_embedding)

        # find top whatever nearest segments from different tracks (one per track)
        # TODO: alternate mode: find whatever segments from different tracks which have the highest similarities to segments from the same track as the seed file
        top_segments = []
        seen_tracks = set()

        # exclude the last three tracks we've played from consideration
        # this way, played tracks CAN be revisited, but only after a few segments
        for played_file in self.played_files:
            track = played_file.split("_")[0]
            seen_tracks.add(basename(track))
        console.log(f"{self.tag} seen tracks: {seen_tracks}")
        seen_tracks = set(
            list(seen_tracks)[-self.params.graph_track_revisit_interval :]
        )

        for idx, similarity in zip(indices, similarities):
            segment_name = str(self.filenames[idx])
            segment_track = segment_name.split("_")[0]

            if segment_track == seed_track or segment_track in seen_tracks:
                continue

            top_segments.append((segment_name, float(similarity)))
            # seen_tracks.add(segment_track)

            # only keep top whatever
            if len(top_segments) >= self.num_segs_diff_tracks:
                break

        for i, (target_segment, target_similarity) in enumerate(top_segments):
            console.log(
                f"{self.tag} target {i+1}/10: '{target_segment}' with similarity {target_similarity:.4f}"
            )

        # load relevant graph files
        graph_path = os.path.join(
            os.path.dirname(self.p_dataset), "graphs", f"{seed_track}.json"
        )
        console.log(f"{self.tag} loading source graph from '{graph_path}'")
        with open(graph_path, "r") as f:
            graph_json = json.load(f)
            graph = nx.node_link_graph(graph_json, edges="edges")  # type: ignore

        # Try each of the top segments until a path is found
        for i, (target_segment, target_similarity) in enumerate(top_segments):
            console.log(
                f"{self.tag} finding path from '{seed_key}' to target {i+1}/10: '{target_segment}' with similarity {target_similarity:.4f}"
            )

            # add target node to graph if not already present
            if target_segment not in graph:
                graph.add_node(target_segment)
                nodes = [
                    basename(node) for node in graph.nodes() if node != target_segment
                ]

                # get node embeddings
                try:
                    node_indices = [
                        self.filename_to_index[basename(node)] for node in nodes
                    ]
                except KeyError as e:
                    # If a key is not found, add it to the mapping and log the error
                    missing_node = str(e).strip("'")
                    console.log(
                        f"{self.tag} [yellow]Warning: Node '{missing_node}' not found in filename_to_index, rebuilding mapping[/yellow]"
                    )
                    # Rebuild the mapping
                    self.filename_to_index = {
                        name: idx for idx, name in enumerate(self.filenames)
                    }
                    # Try again with the updated mapping
                    node_indices = [
                        self.filename_to_index.get(
                            node, self.filenames.index(basename(node))
                        )
                        for node in nodes
                    ]
                node_embs = np.vstack(
                    [self.faiss_index.reconstruct(idx) for idx in node_indices]  # type: ignore
                )

                target_emb = self.faiss_index.reconstruct(self.filename_to_index[target_segment])  # type: ignore
                weights = np.array([cosine(emb, target_emb) for emb in node_embs])
                weighted_edges = [
                    (n, target_segment, float(w)) for n, w in zip(nodes, weights)
                ]
                graph.add_weighted_edges_from(weighted_edges)
            else:
                console.log(
                    f"{self.tag} [yellow]target '{target_segment}' already in graph???[/yellow]"
                )

            # find shortest path
            try:
                res = find_path(
                    graph,
                    seed_key,
                    target_segment,
                    self.played_files,
                    max_nodes=self.params.graph_steps,
                    max_visits=1,
                    allow_transpose=True,
                    allow_shift=not self.params.block_shift,
                    verbose=True,
                )
                if res:
                    path, cost = res
                    console.log(f"{self.tag} found path with cost {cost:.4f}")
                    console.log(f"{self.tag} {path}")
                else:
                    console.log(f"{self.tag} no path found to target {i+1}/10")
                    continue

                edge_weights = []
                for start_node, end_node in list(zip(path[:-1], path[1:])):
                    edge_weights.append(1 - graph[start_node][end_node]["weight"])

                if len(self.current_path) > 0:
                    if self.current_path[-1][1] == -1.0:
                        self.current_path[-1] = (
                            self.current_path[-1][0],
                            edge_weights[0],
                        )

                    path = path[1:]
                    edge_weights = edge_weights[1:]

                self.current_path.extend(
                    list(zip([p + ".mid" for p in path], edge_weights))
                )
                # add target node to the path with placeholder similarity of -1
                target_file = target_segment + ".mid"
                self.current_path.append((target_file, -1.0))
                self.path_position += 1
                if self.verbose:
                    console.log(
                        f"{self.tag} found path between '{seed_key}' and '{target_segment}' with {len(path)} nodes:"
                    )
                    for i, (node, sim) in enumerate(zip(path, edge_weights)):
                        console.log(
                            f"{self.tag}\t\t'{node}' --( {sim:.4f} )--> '{path[i+1]}'"
                        )

                sim = float(
                    1
                    - cosine(
                        self.faiss_index.reconstruct(self.filename_to_index[self.current_path[self.path_position][0][:-4]]),  # type: ignore
                        self.faiss_index.reconstruct(self.filename_to_index[basename(self.played_files[-1])]),  # type: ignore
                    )
                )

                return (
                    self.current_path[self.path_position][0],
                    sim,  # 1 - self.current_path[self.path_position][1],
                )

            except nx.NetworkXNoPath:
                console.log(
                    f"{self.tag} no path found to target {i+1}/10, trying next target"
                )
                continue

        # If we reach here, no path was found to any of the top 10 targets
        console.log(
            f"{self.tag} no path found to any of the top 10 targets, choosing randomly"
        )
        random_file = self._get_random()
        return random_file, 0.0

    def _get_random(self) -> str:
        options = [
            m for m in os.listdir(self.p_dataset) if m.endswith(SUPPORTED_EXTENSIONS)
        ]
        if self.params.block_shift:
            options = [m for m in options if "s00" in str(m)]
        if self.verbose:
            console.log(
                f"{self.tag} choosing randomly from '{self.p_dataset}':\n{options[:5]}"
            )
        random_file = self.rng.choice(options, 1)[0]

        # correct for missing augmentation information if not present
        if len(random_file.split("_")) < 3:
            random_file = random_file[:-4] + "_t00s00.mid"

        # only play files once
        if not self.allow_multiple_plays:
            base_file = self.base_file(random_file)
            while base_file in self.played_files:
                random_file = self.rng.choice(
                    [
                        m
                        for m in os.listdir(self.p_dataset)
                        if m.endswith(SUPPORTED_EXTENSIONS)
                    ]
                )
                base_file = self.base_file(random_file)

        if self.verbose:
            console.log(f"{self.tag} chose random file '{random_file}'")

        return str(random_file)

    def _read_playlist(self) -> tuple[str, float]:
        """Read next segment from playlist"""
        # TODO: modify this to work from current playlist paradigm
        if self.verbose:
            console.log(
                f"{self.tag} playing matches for '{[self.playlist.keys()][self.matches_pos]}'"
            )
            console.log(f"{self.tag} already played files:\n{self.played_files}")

        for i, (q, ms) in enumerate(self.playlist.items()):
            if i == self.matches_pos:
                console.log(f"{self.tag} [grey30]{i}\t'{q}'")
                for mode, matches in ms.items():
                    console.log(f"{self.tag} [grey30]\t'{mode}'")
                    if mode == self.matches_mode:
                        for f, s in matches[:5]:
                            base_files = [
                                os.path.basename(self.base_file(f))
                                for f in self.played_files
                            ]
                            if self.base_file(f) in base_files:
                                console.log(f"{self.tag} [grey30]\t\t'{f}'\t{s}")
                            else:
                                console.log(f"{self.tag} [grey70]\t\t'{f}'\t{s}")
                                file_path = f"{f}.mid"
                                return file_path, float(s)
                        if self.matches_mode == "cplx":
                            raise EOFError("playlist complete")
                        console.log(f"{self.tag} switching modes")
                        self.matches_mode = "cplx"
                        file_path = f"{q}.mid"
                        return file_path, 0.0
        return "", 0.0

    def get_embedding(self, pf_midi: str, model: str | None = None) -> np.ndarray:
        if not model:
            model = str(self.params.metric)
        if model == "pitch-histogram":
            if self.verbose:
                console.log(f"{self.tag} using pitch histogram metric")
            embedding = PrettyMIDI(pf_midi).get_pitch_class_histogram(True, True)
            embedding = embedding.reshape(1, -1)
            console.log(f"{self.tag} {embedding}")
        else:
            embedding = self.model.embed(pf_midi)
            embedding = embedding.numpy()
            # if self.panther_worker is None:
            #     console.log(
            #         f"{self.tag} [orange]error: panther worker not set[/orange]"
            #     )
            #     embedding = panther.send_embedding(pf_midi, model, "live")
            #     console.log(
            #         f"{self.tag} got embedding {embedding.shape} from panther"
            #     )
            # else:
            #     console.log(
            #         f"{self.tag} getting [italic bold]{model}[/italic bold] embedding for '{pf_midi}'"
            #     )
            #     embedding = self.panther_worker.get_embedding(
            #         file_path=pf_midi, model=model, mode="live"
            #     )
            if embedding is None:
                console.log(
                    f"{self.tag} [red]error: failed to get embedding from panther[/red]"
                )
                # TODO: handle failure: return dummy or raise error
                embedding = np.zeros((1, 512), dtype=np.float32)
            console.log(
                f"{self.tag} got [italic bold]{self.params.metric}[/italic bold] embedding {embedding.shape}"
            )
        return embedding

    def _faiss_search(
        self,
        query_embedding: np.ndarray,
        num_matches: int = 1000,
        index: faiss.IndexFlatIP | None = None,
    ) -> tuple[list[int], list[float]]:
        """
        search the FAISS index for the most similar embeddings.
        """
        # ensure that embedding is normalized
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)
        if not index:
            similarities, indices = self.faiss_index.search(query_embedding, num_matches)  # type: ignore
        else:
            similarities, indices = index.search(query_embedding, num_matches)  # type: ignore

        if np.array(similarities).any() > 1.0:
            console.log(f"{self.tag} [red]WARNING: similarity > 1.0[/red]")
            console.print_exception(show_locals=True)
            raise ValueError("similarity > 1.0")

        if self.params.block_shift:
            indices, similarities = zip(
                *[
                    (i, d)
                    for i, d in zip(indices[0], similarities[0])
                    if str(self.filenames[i]).endswith("s00")
                ]
            )
            if self.verbose:
                console.log(
                    f"{self.tag} filtered {num_matches-len(indices)}/{num_matches} shifted files"
                )
        else:
            indices, similarities = zip(
                *[(i, d) for i, d in zip(indices[0], similarities[0])]
            )

        return indices, similarities

    def transform(self, midi_file: str = "") -> str:
        pf_in = self.base_file(self.played_files[-1]) if midi_file == "" else midi_file
        pf_out = os.path.join(
            self.p_playlist,
            f"{len(self.played_files):02d} {os.path.basename(pf_in)}",
        )
        pf_out = os.path.join(self.p_playlist, f"{len(self.played_files):02d} {pf_in}")
        console.log(f"{self.tag} transforming '{pf_in}' to '{pf_out}'")

        copy2(
            os.path.join(os.path.dirname(self.p_dataset), pf_in),
            pf_out,
        )

        return pf_out

    def base_file(self, filename: str) -> str:
        if "player" in filename:
            return filename
        pieces = os.path.basename(filename).split("_")
        return f"{pieces[0]}_{pieces[1]}.mid"

    def get_match(
        self, query_embedding: np.ndarray, metric: str | None = None
    ) -> tuple[str, float]:
        if metric is None:
            metric = str(self.params.metric)
        elif metric not in EMBEDDING_SIZES.keys():
            raise ValueError(f"metric '{metric}' not found in {EMBEDDING_SIZES.keys()}")
        console.log(f"{self.tag} getting match using metric '{metric}'")
        if metric != self.params.metric:
            pf_index = os.path.join(self.p_table, f"{metric}.faiss")
            if not os.path.isfile(pf_index):
                raise FileNotFoundError(f"faiss index '{pf_index}' not found")
            faiss_index = faiss.read_index(pf_index)
            console.log(f"{self.tag}\tloaded faiss index from '{pf_index}'")

        # TODO: embed augmented set with clap and remove this
        if "clap" in metric:
            input_path = self.p_dataset.replace("augmented", "segmented")
            filenames = list(glob(os.path.join(input_path, "*.mid")))
            filenames.sort()
            console.log(f"{self.tag}\tfound {len(filenames)} files:\n\t{filenames[:5]}")
        else:
            filenames = self.filenames

        # get matches
        indices, similarities = self._faiss_search(
            query_embedding,
            num_matches=2000,
            index=faiss_index if "clap" in metric else None,
        )
        if self.verbose:
            for i in range(3):
                console.log(
                    f"{self.tag}\t\t'{filenames[indices[i]]}' ({indices[i]:04d}): {similarities[i]:.4f}"
                )

        best_match = filenames[indices[0]]
        best_similarity = similarities[0]

        return best_match, best_similarity
