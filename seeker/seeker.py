import os
import json
import pretty_midi
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from rich.progress import track, Progress

from utils import console
from utils.midi import all_properties


class Seeker:
    p = "[yellow]seek[/yellow]  :"
    table: pd.DataFrame
    properties = {}

    def __init__(
        self, params, input_dir: str, output_dir: str, force_rebuild: bool = False
    ) -> None:
        """"""
        self.params = params
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.force_rebuild = force_rebuild

    def build_properties(self) -> None:
        dict_file = os.path.join(
            self.output_dir,
            f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}_properties.json",
        )

        if os.path.exists(dict_file) and not self.force_rebuild:
            console.log(f"{self.p} found existing properties file '{dict_file}'")
            with open(dict_file, "r") as f:
                self.properties = json.load(f)
                console.log(
                    f"{self.p} loaded properties for {len(list(self.properties.keys()))} files"
                )
        else:
            console.log(f"{self.p} calculating properties from '{self.input_dir}'")
            for file in track(
                os.listdir(self.input_dir),
                description=f"{self.p} calculating properties",
            ):
                if file.endswith(".mid") or file.endswith(".midi"):
                    file_path = os.path.join(self.input_dir, file)
                    midi = pretty_midi.PrettyMIDI(file_path)
                    properties = all_properties(midi, file, self.params)
                    self.properties[file] = {
                        "filename": file,
                        "properties": properties,
                        "played": 0,
                    }
            console.log(
                f"{self.p} calculated properties for {len(list(self.properties.keys()))} files"
            )

            with open(dict_file, "w") as f:
                json.dump(self.properties, f)

            if os.path.isfile(dict_file):
                console.log(f"{self.p} succesfully saved properties file '{dict_file}'")
            else:
                console.log(f"{self.p} error saving properties file '{dict_file}'")
                raise FileNotFoundError

        self.reset_plays()

    def build_similarity_table(self):
        """"""

        sim_file = f"sims-{os.path.basename(self.input_dir).replace(' ', '_')}-m.parquet"
        console.log(f"{self.p} looking for similarity file '{sim_file}'")
        parquet = os.path.join(self.output_dir, sim_file)
        self.load_similarities(parquet)

        if self.table is not None:
            console.log(f"{self.p} loaded existing similarity file from '{parquet}'")
        else:
            vectors = [
                {
                    "name": filename,
                    "metric": details["properties"][self.params.property],
                }
                for filename, details in self.properties.items()
            ]

            names = [v["name"] for v in vectors]
            vecs = np.array([v["metric"] for v in vectors])

            console.log(f"{self.p} building similarity table for {len(vecs)} vectors")

            # histograms = np.array(vecs)

            # norms = np.linalg.norm(histograms, axis=1, keepdims=True)
            # normalized_histograms = histograms / norms

            # cosine_similarity_matrix = np.dot(
            #     normalized_histograms, normalized_histograms.T
            # )

            # self.table = pd.DataFrame(
            #     cosine_similarity_matrix, index=names, columns=names, dtype="float64"
            # )

            # compute cosine similarity for each pair of vectors
            self.table = pd.DataFrame(index=names, columns=names, dtype="float64")
            with Progress() as progress:
                sims_task = progress.add_task(
                    f"{self.p} calculating sims", total=len(vecs) ** 2
                )
                for i in range(len(vecs)):
                    for j in range(len(vecs)):
                        if i != j:
                            self.table.iloc[i, j] = 1 - cosine(vecs[i], vecs[j])  # type: ignore
                        else:
                            self.table.iloc[i, j] = 1
                        progress.update(sims_task, advance=1)

            console.log(
                f"{self.p} Generated a similarity table of shape {self.table.shape}"
            )

            self.table.to_parquet(parquet, compression="gzip", index=True)

            if os.path.isfile(parquet):
                console.log(f"{self.p} succesfully saved similarities file '{parquet}'")
            else:
                console.log(f"{self.p} error saving similarities file '{parquet}'")
                raise FileNotFoundError

    def get_most_similar_file(self, filename: str, different_parent: bool = True):
        """finds the filename and similarity of the next most similar unplayed file in the similarity table
        NOTE: will go into an infinite loop once all files are played!
        """
        console.log(f"{self.p} finding most similar file to '{filename}'")
        n = 1
        similarity = -1
        next_file_played = 1
        next_filename = None
        self.properties[filename]["played"] = 1  # mark current file as played

        while next_file_played:
            nl = self.table[filename].nlargest(n)  # get most similar columns
            next_filename = nl.index[-1]
            similarity = nl.iloc[-1]

            next_file_played = self.properties[next_filename]["played"]
            n += 1

        console.log(
            f"{self.p} found '{next_filename}' with similarity {similarity:03f}"
        )

        return next_filename, similarity

    def midi_to_ph(self, midi_file: str):
        """"""
        console.log(f"{self.p} calculating pitch histogram for '{midi_file}'")

        midi = pretty_midi.PrettyMIDI(midi_file)

        return midi.get_pitch_class_histogram()

    def find_most_similar_vector(self, target_vector):
        """"""
        console.log(f"{self.p} finding most similar vector to {target_vector}")
        most_similar_vector = None
        highest_similarity = -1  # since cosine similarity ranges from -1 to 1
        vector_array = [
            {"name": filename, "metric": details["properties"][self.params.property]}
            for filename, details in self.properties.items()
        ]

        for vector_data in vector_array:
            name, vector = vector_data.values()
            similarity = 1 - cosine(target_vector, vector)  # type: ignore
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_vector = name

        console.log(
            f"{self.p} found '{most_similar_vector}' with similarity {similarity:03f}"
        )

        return most_similar_vector, highest_similarity

    def reset_plays(self) -> None:
        for k in self.properties.keys():
            self.properties[k]["played"] = 0

    def load_similarities(self, parquet_path) -> None:
        if os.path.isfile(parquet_path) and not self.force_rebuild:
            self.table = pd.read_parquet(parquet_path)
        else:
            self.table = None  # type: ignore
