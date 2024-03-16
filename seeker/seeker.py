import os
import json
import pretty_midi
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from rich.progress import (
    track,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)

from utils import console
from utils.metrics import all_properties


class Seeker:
    p = "[yellow]seeker[/yellow]:"
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

        # self.probs = [
        #     1 / 5,  # same
        #     1 / 6,  # next 1
        #     1 / 6,  # prev 1
        #     1 / 10,  # next 2
        #     1 / 10,  # prev 2
        #     0.0533,  # diff 1
        #     0.0533,  # diff 2
        #     0.0533,  # diff 3
        #     0.0533,  # diff 4
        # ]
        # self.probs.append(1 - sum(self.probs))  # add diff 5
        self.probs = self.params.probabilities
        self.cumprobs = np.cumsum(self.probs)
        self.rng = np.random.default_rng(1)

    def build_properties(self) -> None:
        dict_file = os.path.join(
            self.output_dir,
            f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}.json",
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
                    properties = all_properties(file_path, file, self.params)
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

    def build_similarity_table(self) -> None:
        """"""
        sim_file = f"sims-{os.path.basename(self.input_dir).replace(' ', '_')}.parquet"
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
            vecs = [v["metric"] for v in vectors]

            console.log(f"{self.p} building similarity table for {len(vecs)} vectors")

            self.table = pd.DataFrame(index=names, columns=names, dtype="float64")

            # compute cosine similarity for each pair of vectors
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

            self.table.to_parquet(parquet, index=False)

            if os.path.isfile(parquet):
                console.log(f"{self.p} succesfully saved similarities file '{parquet}'")
            else:
                console.log(f"{self.p} error saving similarities file '{parquet}'")
                raise FileNotFoundError

    def build_top_n_table(self, n: int = 10, vision: int = 2) -> None:
        """ """
        parquet = os.path.join(
            self.output_dir,
            f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}_{self.params.property}.parquet",
        )

        self.load_similarities(parquet)

        if self.table is not None:
            # column_labels = [
            #     [f"{prob}-{i + 1}", f"sim-{i + 1}"] for i, prob in enumerate(self.probs)
            # ]
            # column_labels = [label for sublist in column_labels for label in sublist]
            # console.log(f"{self.p} swapping probs to:\n", column_labels)
            # self.table.columns = column_labels

            console.log(
                f"{self.p} loaded existing similarity file from '{parquet}' ({self.table.shape})\n",
                self.table.columns,
                self.table.index[:4],
            )

            return

        if n % 2:
            console.log(
                f"{self.p} [yellow]uneven value passed in for n ([/yellow]{n}[yellow]), rounding down"
            )
            n -= 1

        vectors = [
            {
                "name": filename,
                "metric": details["properties"][self.params.property],
            }
            for filename, details in self.properties.items()
        ]

        if self.params.property == "pr_blur_c":
            vectors = [
                {"name": v["name"],
                 "metric": np.asarray(v["metric"]).flatten()
                }
                for v in vectors
                if v["name"].endswith("n00.mid")
            ]

        names = [v["name"] for v in vectors]
        vecs = [v["metric"] for v in vectors]

        console.log(
            f"{self.p} building top-{n} similarity table for {len(vecs)} vectors from '{self.input_dir}' using metric '{self.params.property}'"
        )

        labels = [
            "loop  ",
            "prev 1",
            "next 1",
            "prev 2",
            "next 2",
            "diff 1",
            "diff 2",
            "diff 3",
            "diff 4",
            "diff 5",
        ]
        column_labels = [[label, f"sim-{i + 1}"] for i, label in enumerate(labels)]
        # column_labels = [
        #     [f"{prob:.03f}-{i}", f"sim-{i + 1}"] for i, prob in enumerate(self.probs)
        # ]
        column_labels = [label for sublist in column_labels for label in sublist]

        self.table = pd.DataFrame(
            [["", -1.0] * n] * len(names),
            index=names,
            columns=column_labels,
        )

        console.log(
            f"{self.p} initialized table ({len(names)}, {len(column_labels)}) with rows:\n{names[:5]}\nand columns:\n{column_labels}"
        )

        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            refresh_per_second=1,
        )
        sims_task = progress.add_task("calculating sims", total=len(vecs) ** 2)
        with progress:
            for name in self.table.index:
                # console.log(f"\n{self.p} populating row '{name}'")

                i = int(self.table.index.get_loc(name))  # type: ignore
                i_name, i_seg_num, i_shift = name.split("_")
                i_seg_start, i_seg_end = i_seg_num.split("-")
                i_track_name = f"{i_name}_{i_seg_num}"
                # console.print(i_name, i_seg_num, i_shift, i_seg_start, i_seg_end)

                # populate first five columns
                same_track_range = range(2, n, 2)
                # get prev file(s)
                prv2_file = None
                prev_file = self.get_prev(name)
                if prev_file:
                    if int(i_seg_start) != 0:
                        prv2_file = self.get_prev(prev_file)

                # get next file(s)
                nxt2_file = None
                next_file = self.get_next(name)
                if next_file:
                    nxt2_file = self.get_next(next_file)

                names = [prev_file, next_file, prv2_file, nxt2_file]

                for j, k in zip(same_track_range, names):
                    self.table.iat[i, j] = k

                # update second five columns
                for other_name in self.table.index:
                    # console.log(f"{self.p} checking col '{names[j]}'")
                    j = int(self.table.index.get_loc(other_name))  # type: ignore
                    j_name, j_seg_num, j_shift = other_name.split("_")
                    j_seg_start, j_seg_end = j_seg_num.split("-")
                    j_track_name = f"{j_name}_{j_seg_num}"
                    # console.print(j_name, j_seg_num, j_shift, j_seg_start, j_seg_end)

                    sim = float(1 - cosine(vecs[i], vecs[j])) if i != j else 1.0

                    diff_track_range = range(n + 1, n * 2, 2)
                    if i_track_name != j_track_name:  # clip is from a different track
                        self.replace_smallest_sim(
                            name,
                            other_name,
                            sim,
                            diff_track_range,
                        )

                    progress.update(sims_task, advance=1)

        console.log(
            f"{self.p} Generated a similarity table of shape {self.table.shape}"
        )

        self.table.to_parquet(parquet, index=True)

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

    def get_msf_new(self, filename: str):
        """finds the filename and similarity of the next most similar unplayed file in the similarity table
        NOTE: will go into an infinite loop once all files are played!
        """
        console.log(
            f"{self.p} finding most similar file to '{filename}'",
        )

        self.properties[filename]["played"] += 1  # mark current file as played

        # columns = self.table.columns[::2].insert(0, 0)
        columns = list(self.table.columns[::2].values)
        roll = self.rng.choice(columns, p=self.probs)
        if columns.index(roll) > 5:
            console.log(f"{self.p}[blue1] TRACK TRANSITION[/blue1] (rolled '{roll}')")

            # console.log(f"{self.p} options: ", self.table.loc[filename])

        # console.log(f"{self.p} rolled {roll}")
        # console.log(f"{self.p} columns\n{columns}")
        # console.log(f"{self.p} table columns\n{self.table.columns}")

        # for i, col in enumerate(columns[1:]):
        #     if float(columns[i - 1]) < roll <= float(col):
        #         console.log(
        #             f"{self.p} found a match: {columns[i-1]} < {roll:.05f} <= {col}",
        #         )

        #         next_filename = self.table.at[filename, col]
        #         next_col = self.table.columns[i * 2 + 1]

        #         # console.log(
        #         #     f"{self.p} looking for similarity ({i} -> {i * 2 + 1}) at ['{filename}', '{next_col}']\n\t",
        #         #     self.table.at[filename, next_col],
        #         # )
        #         similarity = float(self.table.at[filename, next_col])

        #         break

        next_filename = self.table.at[filename, f"{roll}"]
        next_col = self.table.columns.get_loc(roll) + 1  # type: ignore
        # console.log(
        #     f"{self.p} looking for similarity at ['{filename}', '{self.table.columns[next_col]}']\n\t",
        #     self.table.at[filename, self.table.columns[next_col]],
        # )
        similarity = float(self.table.at[filename, self.table.columns[next_col]])

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

    def replace_smallest_sim(
        self, src_row: str, cmp_file: str, sim: float, col_range: range
    ) -> None:
        row_index = self.table.index.get_loc(src_row)
        smallest_value = float("inf")
        smallest_index = None

        # console.log(f"{self.p} checking row:\n{self.table.iloc[row_index]}")

        for col in col_range:
            current_value = self.table.iloc[row_index, col]  # type: ignore
            # console.log(f"{self.p} got value at [{row_index}, {col}]: {current_value}")

            if current_value < sim and current_value < smallest_value:
                smallest_value = current_value
                smallest_index = col

        # If a smaller value was found, replace the tuple at its index
        if smallest_index is not None:
            self.table.iat[row_index, smallest_index] = sim
            self.table.iat[row_index, smallest_index - 1] = cmp_file

    def reset_plays(self) -> None:
        for k in self.properties.keys():
            self.properties[k]["played"] = 0

    def load_similarities(self, parquet_path) -> None:
        if os.path.isfile(parquet_path) and not self.force_rebuild:
            self.table = pd.read_parquet(parquet_path)
        else:
            self.table = None  # type: ignore

    def get_prev(self, filename):
        i_name, i_seg_num, i_shift = filename.split("_")
        i_seg_start, i_seg_end = i_seg_num.split("-")
        delta = int(i_seg_end) - int(i_seg_start)

        if int(i_seg_start) == 0:
            return None

        prev_file = f"{i_name}_{int(i_seg_start) - delta:04d}-{i_seg_start}_{i_shift}"

        for key in self.properties.keys():
            k_name, k_seg_num, k_shift = key.split("_")
            k_seg_start, k_seg_end = k_seg_num.split("-")

            if (
                k_name == i_name
                and k_shift == i_shift
                and abs(int(k_seg_end) - int(i_seg_start)) <= 2
            ):
                prev_file = key

        return prev_file

    def get_next(self, filename):
        i_name, i_seg_num, i_shift = filename.split("_")
        i_seg_start, i_seg_end = i_seg_num.split("-")
        delta = int(i_seg_end) - int(i_seg_start)

        next_file = f"{i_name}_{i_seg_end}-{int(i_seg_end) + delta:04d}_{i_shift}"
        if next_file not in self.properties.keys():
            next_file = None
            for key in self.properties.keys():
                k_name, k_seg_num, k_shift = key.split("_")
                k_seg_start, k_seg_end = k_seg_num.split("-")

                if (
                    k_name == i_name
                    and k_shift == i_shift
                    and abs(int(k_seg_start) - int(i_seg_end)) <= 2
                ):
                    next_file = key

        return next_file
