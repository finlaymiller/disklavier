import os
import torch
import faiss
import numpy as np
import pandas as pd
from shutil import copy2
from pretty_midi import PrettyMIDI
from scipy.spatial.distance import cosine
import json
import networkx as nx

from .worker import Worker
from utils import basename, console, panther
from utils.models import Classifier
from utils.modes import find_path
from utils.udp import send_udp

SUPPORTED_EXTENSIONS = (".mid", ".midi")
EMBEDDING_SIZES = {
    "pitch-histogram": 12,
    "specdiff": 768,
    "clf-4note": 128,
    "clf-speed": 128,
    "clf-tpose": 128,
}


class Seeker(Worker):
    # required tables
    filenames: list[str] = []
    faiss_index: faiss.IndexFlatIP
    neighbor_table: pd.DataFrame
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
    # supported ml models for post-processing embeddings
    model = None
    model_list = [
        "clf-4note",
        "clf-speed",
        "clf-tpose",
    ]
    # path tracking variables
    current_path: list[tuple[str, float]] = []
    path_position: int = 0
    num_segs_diff_tracks: int = 5
    # filename to index mapping
    filename_to_index: dict[str, int] = {}

    # velocity tracking
    _recorder = None
    _avg_velocity: float = 0.0

    def __init__(
        self,
        params,
        table_path: str,
        dataset_path: str,
        playlist_path: str,
        bpm: int,
    ):
        super().__init__(params, bpm=bpm)
        self.p_table = table_path
        self.p_dataset = dataset_path
        self.p_playlist = playlist_path
        self.rng = np.random.default_rng(self.params.seed)
        # TODO: stop doing this
        if hasattr(self.params, "pf_recording"):
            self.pf_recording = self.params.pf_recording
        if hasattr(self.params, "graph_steps"):
            self.graph_steps = self.params.graph_steps
        if self.params.metric in self.model_list:
            self.model = self.load_model()

        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

        # load FAISS index
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

        self.faiss_index = faiss.read_index(
            os.path.join(self.p_table, f"{self.params.metric}.faiss")
        )
        console.log(f"{self.tag} FAISS index loaded ({self.faiss_index.ntotal})")

        # load neighbor table
        # old version -- ignore
        pf_neighbor_table = os.path.join(self.p_table, "neighbor.parquet")
        # pf_neighbor_table = os.path.join(self.p_table, "neighbors.h5")
        console.log(f"{self.tag} looking for neighbor table '{pf_neighbor_table}'")
        if os.path.isfile(pf_neighbor_table):
            with console.status("\t\t\t      loading neighbor file..."):
                # self.neighbor_table = pd.read_hdf(pf_neighbor_table, key="neighbors")
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
        """
        Check for velocity updates from the recorder.

        Returns
        -------
        bool
            True if velocity data was updated, False otherwise.
        """
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
                # TODO: FIX THIS
                if "player" in self.played_files[-1]:
                    next_file, similarity = self._get_best(hop=False)
                else:
                    next_file, similarity = self._get_graph()
            case "random" | "shuffle" | _:
                next_file = self._get_random()

        if self.pitch_match and len(self.played_files) > 0:
            console.log(
                f"{self.tag} pitch matching '{self.base_file(next_file)}' to '{self.played_files[-1]}'"
            )
            if "player" in self.played_files[-1].split("_")[0]:
                base_pch = PrettyMIDI(
                    os.path.join(self.pf_recording)
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

        send_udp(f"setSim {similarity}", "/sim")
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
                f"{self.tag} {len(self.played_files)} played files:\n{self.played_files}"
            )

        query_file = basename(self.played_files[-1])

        if self.verbose:
            console.log(
                f"{self.tag} extracted '{self.played_files[-1]}' -> '{query_file}'"
            )

        # load query embedding
        # first, look in existing index
        # then, if no embedding is found, calculate manually
        try:
            # Use dictionary lookup instead of list.index() with fallback
            try:
                embedding = self.faiss_index.reconstruct(self.filename_to_index[query_file])  # type: ignore
            except KeyError:
                embedding = self.faiss_index.reconstruct(self.filenames.index(query_file))  # type: ignore
        except (ValueError, KeyError) as e:
            if hasattr(self, "pf_recording"):
                pf_new = self.pf_recording
            else:
                pf_new = self.played_files[-1]
            console.log(
                f"{self.tag}[yellow] unable to find embedding for '{query_file}', calculating manually from '{pf_new}'"
            )
            match self.params.metric:
                case "specdiff" | "clf-4note" | "clf-speed" | "clf-tpose":
                    console.log(
                        f"{self.tag} getting [italic bold]{self.params.metric}[/italic bold] embedding for '{pf_new}'"
                    )
                    embedding = panther.calc_embedding(pf_new, model=self.params.metric)
                    console.log(
                        f"{self.tag} got [italic bold]{self.params.metric}[/italic bold] embedding {embedding.shape}"
                    )

                    if self.params.metric != "specdiff" and self.model is not None:
                        console.log(
                            f"{self.tag} applying {self.params.metric} model to embedding"
                        )
                        embedding = (
                            self.model.get_hidden(torch.from_numpy(embedding), 1)
                            .detach()
                            .numpy()
                        )
                        console.log(
                            f"{self.tag} got {self.params.metric} embedding {embedding.shape}"
                        )
                case "clamp":
                    raise NotImplementedError("CLaMP model is not currently supported")
                case "pitch-histogram":
                    if self.verbose:
                        console.log(f"{self.tag} using pitch histogram metric")
                    embedding = PrettyMIDI(self.pf_recording).get_pitch_class_histogram(
                        True, True
                    )
                    embedding = embedding.reshape(1, -1)
                    console.log(f"{self.tag} {embedding}")
                case _:
                    console.log(f"{self.tag} unsupported metric: {self.params.metric}")
                    raise ValueError(
                        f"{self.tag} unsupported metric: {self.params.metric}"
                    )

            # Add the new embedding to the index and update filenames list and dictionary
            self.faiss_index.add(embedding)  # type: ignore
            self.filenames.append(query_file)
            self.filename_to_index[query_file] = len(self.filenames) - 1
            console.log(f"{self.tag} added '{query_file}' to index")
        query_embedding = np.array(
            embedding,
            dtype=np.float32,
        ).reshape(1, -1)
        # ensure that embedding is normalized
        # TODO: move embedding normalization to dataset generation
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # query index
        if self.verbose and "player" not in query_file:
            console.log(f"{self.tag} querying with key '{query_file}'")
        similarities, indices = self.faiss_index.search(query_embedding, 1000)  # type: ignore
        if self.verbose:
            console.log(f"{self.tag} indices:\n\t", indices[0][:10])
            console.log(f"{self.tag} similarities:\n\t", similarities[0][:10])

        # reformat and filter shifted files as it often sounds bad
        indices, similarities = zip(
            *[
                (i, d)
                for i, d in zip(indices[0], similarities[0])
                if str(self.filenames[i]).endswith("s00")
            ]
        )

        # find most similar valid match
        if "player" in query_file:
            next_file = self._get_random()
        else:
            next_file = self._get_neighbor()
        # TODO: fix this
        played_files = [os.path.basename(self.base_file(f)) for f in self.played_files]
        for idx, similarity in zip(indices, similarities):
            segment_name = str(self.filenames[idx])
            if "player" in query_file:
                next_file = f"{segment_name}.mid"
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
                next_segment_name not in played_files
                and segment_name.endswith("s00")
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
            console.log(f"{self.tag} played files: {self.played_files}")
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
            track, segment, _ = current_file.split("_")
            current_file = f"{track}_{segment}.mid"

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

        seed_file = self.played_files[-1]
        seed_key = basename(seed_file)
        seed_track = seed_key.split("_")[0]
        console.log(f"{self.tag} using seed file '{seed_file}' for graph navigation")

        e = self.faiss_index.reconstruct(self.filename_to_index[seed_key])  # type: ignore
        seed_embedding = np.array([e])

        # find nearest segments using FAISS
        console.log(
            f"{self.tag} searching for nearest segment from a different track with embedding {seed_embedding.shape}"
        )
        similarities, indices = self.faiss_index.search(seed_embedding, 1000)  # type: ignore

        # find top whatever nearest segments from different tracks (one per track)
        # TODO: alternate mode: find whatever segments from different tracks which have the highest similarities to segments from the same track as the seed file
        top_segments = []
        seen_tracks = set()

        # exclude the last three tracks we've played from consideration
        # this way, played tracks CAN be revisited, but only after a few segments
        for played_file in self.played_files:
            track = played_file.split("_")[0]
            seen_tracks.add(basename(track))
        seen_tracks = set(
            list(seen_tracks)[-self.params.graph_track_revisit_interval :]
        )
        console.log(f"{self.tag} seen tracks: {seen_tracks}")

        for idx, similarity in zip(indices[0], similarities[0]):
            segment_name = str(self.filenames[idx])
            segment_track = segment_name.split("_")[0]

            if segment_track == seed_track or segment_track in seen_tracks:
                continue

            top_segments.append((segment_name, float(similarity)))
            seen_tracks.add(segment_track)

            # only keep top whatever
            if len(top_segments) >= self.num_segs_diff_tracks:
                break

        for i, (target_segment, target_similarity) in enumerate(top_segments):
            console.log(
                f"{self.tag} target {i+1}/10: '{target_segment}' with similarity {target_similarity:.4f}"
            )

        # load relevant graph files
        # TODO: build this path programmatically
        graph_path = os.path.join(
            "data", "datasets", "20250110", "graphs", f"{seed_track}.json"
        )
        console.log(f"{self.tag} loading source graph from '{graph_path}'")
        with open(graph_path, "r") as f:
            graph_json = json.load(f)

        graph = nx.node_link_graph(graph_json)

        # Try each of the top segments until a path is found
        for i, (target_segment, target_similarity) in enumerate(top_segments):
            console.log(
                f"{self.tag} finding path from '{seed_key}' to target {i+1}/10: '{target_segment}' with similarity {target_similarity:.4f}"
            )

            # add target node to graph if not already present
            if target_segment not in graph:
                graph.add_node(target_segment)
                nodes = [node for node in graph.nodes() if node != target_segment]

                # get node embeddings
                try:
                    node_indices = [self.filename_to_index[node] for node in nodes]
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
                        self.filename_to_index.get(node, self.filenames.index(node))
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
                    max_updates=50,
                    allow_transpose=True,
                    allow_shift=True,
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
        if self.verbose:
            console.log(
                f"{self.tag} choosing randomly from '{self.p_dataset}':\n{[m for m in os.listdir(self.p_dataset)][:5]}"
            )
        random_file = self.rng.choice(
            [m for m in os.listdir(self.p_dataset) if m.endswith(SUPPORTED_EXTENSIONS)],
            1,
        )[0]

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

        # no shift -- too risky
        random_file = os.path.splitext(random_file)[0][:-3] + "s00.mid"

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

    def construct_keys(self):
        for filename in self.played_files:
            yield f"files:{filename[:-4]}_t00s00.mid"

    def load_model(self):
        match self.params.metric:
            case "clamp":
                raise NotImplementedError("CLaMP model is no longer supported")
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    console.log(f"{self.tag} Using GPU {torch.cuda.get_device_name(0)}")
                else:
                    console.log(f"{self.tag} No GPU available, using the CPU instead")
                    device = torch.device("cpu")
                self.params.model = clamp.CLaMP.from_pretrained(clamp.CLAMP_MODEL_NAME)
                if self.verbose:
                    console.log(f"{self.tag} Loaded model:\n{self.params.model.eval}")
                self.params.model = self.params.model.to(device)  # type: ignore
            case "clf-4note" | "clf-speed" | "clf-tpose":
                clf = Classifier(768, [128], 120)
                clf.load_state_dict(
                    torch.load(
                        os.path.join("data", "models", self.params.metric + ".pth"),
                        weights_only=True,
                        map_location=torch.device("cpu"),
                    )
                )
                clf.eval()

                return clf
            case _:
                raise TypeError(
                    f"{self.tag} Unsupported model specified: {self.params.metric}"
                )
