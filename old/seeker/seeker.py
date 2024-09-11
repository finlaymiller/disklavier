import os
import pretty_midi
import numpy as np
import pandas as pd
import itertools
from scipy.spatial.distance import cosine
from rich.progress import track

from utils import console, extract_transformations

from typing import Dict


class Seeker:
    p = "[yellow]seeker[/yellow]:"
    sim_table: pd.DataFrame
    count = 0
    transition_probability = 0. # max is 1.0
    transformation = {"transpose": 0, "shift": 0}
    played_files = []

    def __init__(
        self,
        params,
        input_dir: str,
        output_dir: str,
        tempo: int,
        dataset: str,
        mode: str,
    ) -> None:
        """"""
        self.params = params
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.params.tempo = tempo
        self.params.seed = 1 if self.params.seed is None else self.params.seed
        self.rng = np.random.default_rng(self.params.seed)
        self.mode=mode

        # load similarity table
        sim_table_path = os.path.join("data", "tables", f"{dataset}_sim.parquet")
        console.log(f"{self.p} looking for similarity table '{sim_table_path}'")
        if os.path.isfile(sim_table_path):
            with console.status("\t\t\t      loading similarities file..."):
                self.sim_table = pd.read_parquet(sim_table_path)
            console.log(f"{self.p} loaded {len(self.sim_table)}*{len(self.sim_table.columns)} sim table")
        else:
            console.log(f"{self.p} error loading similarity table, exiting...")
            exit()
        
        # load neighbor table
        neighbor_table_path = os.path.join("data", "tables", f"{dataset}_neighbor.parquet")
        console.log(f"{self.p} looking for neighbor table '{neighbor_table_path}'")
        if os.path.isfile(neighbor_table_path):
            console.log(f"{self.p} loading neighbor table at '{neighbor_table_path}'")
            with console.status("\t\t\t      loading neighbor file..."):
                self.neighbor_table = pd.read_parquet(neighbor_table_path)
            console.log(f"{self.p} loaded {len(self.neighbor_table)}*{len(self.neighbor_table.columns)} neighbor table")
        else:
            console.log(f"{self.p} error loading neighbor table, exiting...")
            exit()
        
        # load transformation table
        trans_table_path = os.path.join("data", "tables", f"{dataset}_transformations.parquet")
        console.log(f"{self.p} looking for tranformation table '{trans_table_path}'")
        if os.path.isfile(trans_table_path):
            console.log(f"{self.p} loading tranformation table at '{trans_table_path}'")
            with console.status("\t\t\t      loading tranformation file..."):
                self.trans_table = pd.read_parquet(trans_table_path)
            console.log(f"{self.p} loaded {len(self.trans_table)}*{len(self.trans_table.columns)} transformation table")
        else:
            console.log(f"{self.p} error loading tranformation table, exiting...")
            exit()
        console.log(f"{self.p} [green]successfully loaded tables")

        console.log(f"{self.p} initialized to use metric '{self.params.property}'")

    def get_most_similar_file(
        self, prev_filename: str, different_parent=True, bump_trans=False
    ) -> Dict:
        """"""
        console.log(f"{self.p} finding most similar file to '{prev_filename}' (MODE={self.mode})")

        if bump_trans:
            console.log(f"{self.p} increasing transition probability {self.transition_probability} -> {self.transition_probability + self.params.transition_increment}")
            self.transition_probability += self.params.transition_increment

            if self.transition_probability > 1.:
                self.transition_probability = 0

        [filename, transformations] = extract_transformations(prev_filename)
        parent_track, _ = filename.split("_")
        # console.log(f"{self.p} extracted '{prev_filename}' -> '{filename}' and {transformations}")

        change_track = self.rng.choice([True, False], p=[self.transition_probability, 1 - self.transition_probability])
        console.log(f"{self.p} rolled change track = {change_track} w/transition prob. {self.transition_probability:.02f}")
        sorted_row = self.sim_table.loc[filename].sort_values(key=lambda x: x.str['sim'], ascending=False)

        if change_track:
            for next_filename, val in sorted_row.items():
                next_track, _ = next_filename.split("_")
                if next_track == parent_track or next_filename == filename or next_filename in self.played_files:
                    console.log(f"{self.p} skipping invalid match\n\t'{next_track}' == '{parent_track}' or\n\t'{next_filename}' == '{filename}'")
                    continue
                if next_track != parent_track and next_filename != filename:
                    value = val
                    value["filename"] = next_filename
                    break
        else:
            # scan for next neighbor in neighbor table
            neighbor = None
            num_tries = 0
            max_tries = 10
            while neighbor == None and num_tries < max_tries:
                neighbor = self.rng.choice(self.neighbor_table.loc[filename])
                num_tries += 1
            # if we couldn't find one, get most similar file
            if neighbor == None or neighbor == filename:
                for next_filename, val in sorted_row.items():
                    next_track, _ = next_filename.split("_")

                    if next_track == parent_track or next_filename == filename or next_filename in self.played_files:
                        # console.log(f"{self.p} skipping invalid match\n\t'{next_track}' == '{parent_track}' or\n\t'{next_filename}' == '{filename}'")
                        continue
                    if next_track != parent_track and next_filename != filename:
                        value = {
                            "filename": next_filename,
                            'sim': val['sim'],
                            "transformations": {
                                "transpose": val[ "transformations"]["transpose"],
                                "shift": val[ "transformations"]["shift"],
                            },
                        }
                        break
                # console.log(f"{self.p} unable to find neighbor for '{filename}', got most similar next file: '{value['filename']}'@sim {value['sim']:.03f}", value)
            else:
                value = {
                    "filename": neighbor,
                    "sim": -1,
                    "transformations": {
                        "transpose": 0,
                        "shift": 0,
                    },
                }

        # track transformations
        console.log(f"{self.p} found '{value['filename']}' with similarity {value["sim"]:.03f} to '{filename}'", value)

        true_transpose = (value["transformations"]["transpose"] + self.transformation["transpose"]) % 12
        if true_transpose > 6:
            true_transpose = -1 * true_transpose % 6
        true_shift = 0# (value["transformations"]["shift"] + self.transformation["shift"]) % 8
        value["transformations"]["transpose"] = true_transpose
        value["transformations"]["shift"] = true_shift
        self.transformation["transpose"] = true_transpose
        self.transformation["shift"] = true_shift

        # console.log(f"{self.p} added transform: {self.transformation} and returning {value}")
        self.played_files.append(value["filename"])

        return value
  
    def match_recording(self, recording_path: str):
        console.log(
            f"{self.p} finding most similar vector to '{recording_path}' with metric '{self.params.property}'"
        )
        recording = pretty_midi.PrettyMIDI(recording_path)
        recording_metric = recording.get_pitch_class_histogram(True, True)

        most_similar_segment = ""
        highest_similarity = -1.0
        best_transformations = {}

        # TODO: VECTORIZE THIS
        for segment_name in track(self.trans_table.index.tolist(), "calculating similarities...", refresh_per_second=1, update_period=1.0):
            for semi in list(range(12)): # [list(p) for p in itertools.product(list(range(12)), list(range(8)))]
                beat = 0
                similarity = float(1 - cosine(recording_metric, self.trans_table.at[segment_name, f"{semi:02d}{beat:02d}"]))  # type: ignore
                if similarity > highest_similarity:
                    # console.log(f"{self.p} updating similarity {highest_similarity:.03f} -> {similarity:.03f}\n\t'{segment_name}' -> '{most_similar_segment}'\tt{semi} & s{beat}")
                    highest_similarity = similarity
                    most_similar_segment = segment_name
                    best_transformations = {
                        "transpose": semi,
                        "shift": beat
                    }

        console.log(
            f"{self.p} \tfound '{most_similar_segment}' with similarity {highest_similarity:.03f} using transformations {best_transformations}"
        )

        self.transformation = best_transformations

        return most_similar_segment, highest_similarity, best_transformations

    def reset_plays(self) -> None:
        for k in self.properties.keys():
            self.properties[k]["played"] = 0

    def get_random(self) -> str:
        return os.path.join(self.input_dir, self.rng.choice(os.listdir(self.input_dir)))
