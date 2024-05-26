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
import utils.metrics as metrics
from utils.plot import plot_histograms

from typing import Tuple, Dict


class Seeker:
    p = "[yellow]seeker[/yellow]:"
    table: pd.DataFrame
    properties = {}
    count = 0
    trans_options = [
        "u01.mid",
        "d01.mid",
        "u02.mid",
        "d02.mid",
        "u03.mid",
        "d03.mid",
        "u04.mid",
        "d04.mid",
        "u05.mid",
        "d05.mid",
        "u06.mid",
        "d06.mid",
    ]

    def __init__(
        self,
        params,
        input_dir: str,
        output_dir: str,
        tempo: int,
        force_rebuild: bool = False,
    ) -> None:
        """"""
        self.params = params
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.force_rebuild = force_rebuild
        self.probs = self.params.probabilities / np.sum(self.params.probabilities)
        self.params.tempo = tempo
        self.params.seed = 1 if self.params.seed is None else self.params.seed
        self.rng = np.random.default_rng(self.params.seed)

        console.log(f"{self.p} initialized to use metric '{self.params.property}'")

    def build_properties(self) -> None:
        dict_file = os.path.join(
            self.output_dir,
            f"{os.path.basename(os.path.normpath(self.input_dir)).replace(' ', '_')}.json",
        )

        if os.path.exists(dict_file) and not self.force_rebuild:
            console.log(f"{self.p} found existing properties file '{dict_file}'")
            with console.status("loading properties file..."):
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
                    properties = metrics.all_properties(file_path, file, self.params)
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
            console.log(
                f"{self.p} loaded existing similarity file from '{parquet}' ({self.table.shape})\n",
                self.table.columns,
                self.table.index[:4],
            )

            return

        if n % 2:
            console.log(
                f"{self.p} [yellow]odd value passed in for n ([/yellow]{n}[yellow]), rounding down"
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
                {"name": v["name"], "metric": np.asarray(v["metric"]).flatten()}
                for v in vectors
                if v["name"].endswith("n00.mid")
            ]
        if self.params.property == "pr_blur":
            vectors = [
                {"name": v["name"], "metric": np.asarray(v["metric"]).flatten()}
                for v in vectors
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
        column_labels = [label for sublist in column_labels for label in sublist]

        self.table = pd.DataFrame(
            [["", -1.0] * len(labels)] * len(names),
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
                # i_name, i_seg_num, i_shift = name.split("_")
                i_name, i_seg_num = name.split("_")
                i_seg_start, i_seg_end = i_seg_num.split("-")
                i_seg_end = i_seg_end.split(".")[0]

                # populate first five columns
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

                names = [name, prev_file, next_file, prv2_file, nxt2_file]

                for j, k in zip(range(0, n, 2), names):
                    self.table.iat[i, j] = k

                # update second five columns
                for other_name in self.table.index:
                    # console.log(f"{self.p} checking col '{names[j]}'")
                    j = int(self.table.index.get_loc(other_name))  # type: ignore
                    # j_name, j_seg_num, j_shift = other_name.split("_")
                    j_name, j_seg_num = other_name.split("_")

                    # console.log(f"{self.p} v i", vecs[i])
                    # console.log(f"{self.p} v j", vecs[j])

                    sim = float(1 - cosine(vecs[i], vecs[j]))

                    diff_track_range = range(n + 1, n * 2, 2)
                    if i_name != j_name:  # clip is from a different track
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

    def get_most_similar_file(
        self, filename: str, different_parent=True
    ) -> Dict:
        """"""
        console.log(f"{self.p} finding most similar file to '{filename}'")
        
        self.properties[filename]["num_plays"] += 1
        n = 1
        parent_track, _ = filename.split("_")
        next_largest = self.table[filename].nlargest(n)
        console.log(f"{self.p} checking index {next_largest}")
        next_filename: str = next_largest.index[-1] # type: ignore
        next_track, _ = next_filename.split("_")

        # TODO: refactor logic?
        if different_parent:
            while next_track == parent_track:
                n += 1
                value = next_largest.iloc[-1]
                next_largest = self.table[filename].nlargest(n)
                console.log(f"{self.p} checking index {next_largest}")
                next_filename: str = next_largest.index[-1] # type: ignore
                next_track, _ = next_filename.split("_")
                value["filename"] = next_filename
        else:
            while next_filename == filename:
                n += 1
                value = next_largest.iloc[-1]
                next_largest = self.table[filename].nlargest(n)
                next_filename: str = next_largest.index[-1] # type: ignore
                next_track, _ = next_filename.split("_")
                value["filename"] = next_filename

        console.log(
            f"{self.p} found '{value['filename']}' with similarity {value["sim"]:.03f}"
        )

        return value

    def get_most_similar_file_old(self, filename: str, different_parent: bool = True):
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
            f"{self.p} found '{next_filename}' with similarity {similarity:.03f}"
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

        columns = list(self.table.columns[::2].values)
        roll = self.rng.choice(columns, p=self.probs)
        if columns.index(roll) > 5:
            console.log(f"{self.p} \t[blue1]TRACK TRANSITION[/blue1] (rolled '{roll}')")

        if self.params.calc_trans and not filename.endswith("n00.mid"):
            self.last_trans = filename[-7:]
            filename = filename[:-7] + "n00.mid"

        if self.params.max_sim:
            roll = self.get_max_sim(filename)

        next_filename = self.table.at[filename, f"{roll}"]

        # when the source file is at the start or end of a track the prev/next
        # columns respectively can be None
        while next_filename == "" or next_filename == None:
            console.log(f"{self.p} \t[blue1]REROLL[/blue1] (rolled '{roll}')")
            roll = self.rng.choice(columns, p=self.probs)
            next_filename = self.table.at[filename, f"{roll}"]

        next_col = self.table.columns.get_loc(roll) + 1  # type: ignore
        similarity = float(self.table.at[filename, self.table.columns[next_col]])

        # check transposition if using centered blur
        if self.params.calc_trans:
            next_filename, similarity = self.pitch_transpose(
                os.path.join(self.input_dir, filename),
                os.path.join(self.input_dir, next_filename),
                similarity,
            )

        console.log(
            f"{self.p} \tfound '{next_filename}' with similarity {similarity:.03f}"
        )

        return next_filename, similarity

    def get_ms_to_recording(self, recording_path: str) -> Tuple[str | None, float]:
        console.log(
            f"{self.p} finding most similar vector to '{recording_path}' with metric '{self.params.property}'"
        )

        midi = pretty_midi.PrettyMIDI(recording_path)

        match self.params.property:
            case "energy":
                cmp_metric = metrics.energy(recording_path)
            case "pr_blur":
                cmp_metric = metrics.blur_pr(midi, False)
            case "pr_blur_c":
                cmp_metric = metrics.blur_pr(midi)
            case "contour":
                cmp_metric = metrics.contour(
                    midi, self.params.beats_per_seg, self.params.tempo
                )
            case "contour-complex":
                cmp_metric = metrics.contour(
                    midi, self.params.beats_per_seg, self.params.tempo, False
                )
            case _:
                cmp_metric = midi.get_pitch_class_histogram()

        most_similar_vector = None
        highest_similarity = -1.0  # since cosine similarity ranges from -1 to 1
        vector_array = [
            {"name": filename, "metric": details["properties"][self.params.property]}
            for filename, details in self.properties.items()
        ]

        for vector_data in vector_array:
            name, vector = vector_data.values()
            similarity = float(1 - cosine(cmp_metric, vector))  # type: ignore
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_vector = name

        console.log(
            f"{self.p} \tfound '{most_similar_vector}' with similarity {highest_similarity:.03f}"
        )

        if self.params.calc_trans:
            most_similar_vector, highest_similarity = self.pitch_transpose(
                recording_path, os.path.join(self.input_dir, str(most_similar_vector))
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
        # i_name, i_seg_num, i_shift = filename.split("_")
        i_name, i_seg_num = filename.split("_")
        i_seg_start, i_seg_end = i_seg_num.split("-")
        i_seg_end = i_seg_end.split(".")[0]
        delta = int(i_seg_end) - int(i_seg_start)

        if int(i_seg_start) == 0:
            return None

        # prev_file = f"{i_name}_{int(i_seg_start) - delta:04d}-{i_seg_start}_{i_shift}"
        prev_file = f"{i_name}_{int(i_seg_start) - delta:04d}-{i_seg_start}"

        for key in self.properties.keys():
            # k_name, k_seg_num, k_shift = key.split("_")
            k_name, k_seg_num = key.split("_")
            k_seg_start, k_seg_end = k_seg_num.split("-")
            k_seg_end = k_seg_end.split(".")[0]

            if (
                k_name == i_name
                # and k_shift == i_shift
                and abs(int(k_seg_end) - int(i_seg_start)) <= 2
            ):
                prev_file = key

        return prev_file

    def get_next(self, filename):
        # i_name, i_seg_num, i_shift = filename.split("_")
        i_name, i_seg_num = filename.split("_")
        i_seg_start, i_seg_end = i_seg_num.split("-")
        i_seg_end = i_seg_end.split(".")[0]
        delta = int(i_seg_end) - int(i_seg_start)

        # next_file = f"{i_name}_{i_seg_end}-{int(i_seg_end) + delta:04d}_{i_shift}"
        next_file = f"{i_name}_{i_seg_end}-{int(i_seg_end) + delta:04d}"
        if next_file not in self.properties.keys():
            next_file = None
            for key in self.properties.keys():
                # k_name, k_seg_num, k_shift = key.split("_")
                k_name, k_seg_num = key.split("_")
                k_seg_start, k_seg_end = k_seg_num.split("-")
                k_seg_end = k_seg_end.split(".")[0]

                if (
                    k_name == i_name
                    # and k_shift == i_shift
                    and abs(int(k_seg_start) - int(i_seg_end)) <= 2
                ):
                    next_file = key

        return next_file

    def pitch_transpose(
        self, seed: str, match: str, piano_roll_sim: float = -1.0
    ) -> Tuple[str, float]:

        seed_ph = pretty_midi.PrettyMIDI(seed).get_pitch_class_histogram()
        match_ph = pretty_midi.PrettyMIDI(match).get_pitch_class_histogram()
        match_ph_sim = float(1 - cosine(seed_ph, match_ph))
        # console.log(
        #     f"{self.p} \tunshifted match ('{os.path.basename(seed)}' :: '{os.path.basename(match)}') has ph sim {match_ph_sim:.03f}"
        # )

        if piano_roll_sim > 0:
            seed = seed[:-7] + self.last_trans
            seed_ph = pretty_midi.PrettyMIDI(seed).get_pitch_class_histogram()
            match_ph_sim = float(1 - cosine(seed_ph, match_ph))

            # console.log(f"{self.p} \tshifted '{os.path.basename(seed)}' has ph sim {match_ph_sim:.03f}")

        best_match = os.path.basename(match)
        best_sim = match_ph_sim

        t_files = []
        for transposition in self.trans_options:
            t_file = match[:-7] + transposition

            # not all transposition options for each file will exist
            if not os.path.exists(t_file):
                continue

            t_ph = pretty_midi.PrettyMIDI(t_file).get_pitch_class_histogram()
            t_sim = float(1 - cosine(seed_ph, t_ph))
            t_files.append((t_file, t_sim))

            if t_sim > best_sim:
                best_match = os.path.basename(t_file)
                best_sim = t_sim

                console.log(
                    f"{self.p} \tbetter tpos '{os.path.basename(t_file)}' @ sim {t_sim:.03f}"
                )

        # plot_histograms(
        #     [seed_ph, match_ph],
        #     [os.path.basename(f) for f in [seed, match]],
        #     os.path.join(
        #         self.output_dir,
        #         "plots",
        #         "_test",
        #         f"{self.count}-{os.path.basename(seed)[:-4]}-src.png",
        #     ),
        #     (2, 1),
        #     f"neutral sim = {match_ph_sim:.03f}",
        # )
        # plot_histograms(
        #     [pretty_midi.PrettyMIDI(f).get_pitch_class_histogram() for f, _ in t_files],
        #     [f"{os.path.basename(f)[-7:-4]} ({s:.02f})" for f, s in t_files],
        #     os.path.join(
        #         self.output_dir,
        #         "plots",
        #         "_test",
        #         f"{self.count}-{os.path.basename(seed)[:-4]}-phs.png",
        #     ),
        #     (4, 3),
        #     f"best match: {os.path.basename(best_match)[:-4]}",
        # )
        self.count += 1

        return best_match, best_sim

    def sort_row(self, row):
        """Sorts the specified sections of a DataFrame row in descending order based on 'sim' values.

        This function assumes the row contains a fixed part and a sortable part, where the sortable
        part consists of 'diff' and 'sim' column pairs. The function sorts these pairs by the 'sim'
        value in descending order while keeping each 'diff' directly in front of its corresponding 'sim'.

        Args:
            row (pd.Series): A single row of a DataFrame to be sorted. It is expected that the
                            row contains mixed data types with 'diff' being filenames (str) and
                            'sim' being numeric scores (float).

        Returns:
            pd.Series: A series with the first part unchanged and the last part sorted based on the
                    'sim' values.

        Note:
            The function is designed to operate within a DataFrame.apply() method which allows it
            to be applied row-wise. It specifically manages rows that split at index 10, where indices
            from 10 onward contain 'diff' and 'sim' pairs.

        Example:
            # Assuming 'df' is a DataFrame loaded with the appropriate columns and data structure:
            sorted_df = df.apply(sort_row, axis=1)
        """
        fixed_part = row.iloc[:10]  # Assumes the first 10 entries do not need sorting
        to_sort_part = row.iloc[10:]  # The part that needs sorting

        # Create a DataFrame from the parts that need sorting
        # Assuming every two columns are 'diff' and 'sim' pairs starting from index 10
        df_to_sort = pd.DataFrame(
            {
                "diff": to_sort_part[::2].values,  # Assumes even indices are 'diff'
                "sim": to_sort_part[1::2].values,  # Assumes odd indices are 'sim'
            },
            index=pd.MultiIndex.from_arrays(
                [to_sort_part[::2].index, to_sort_part[1::2].index]
            ),
        )

        # Sort the DataFrame based on 'sim' values in descending order
        sorted_df = df_to_sort.sort_values(by="sim", ascending=False)

        # Flatten the sorted DataFrame back into a Series
        sorted_series = pd.Series(
            data=sorted_df.values.flatten(),
            index=[idx for sub_idx in sorted_df.index for idx in sub_idx],
        )

        # Concatenate the fixed part and the sorted part
        return pd.concat([fixed_part, sorted_series])

    def get_max_sim(self, row_label):
        row = self.table.loc[row_label]
        most_similar_v = -1
        most_similar_i = 1
        for i, (k, v) in enumerate(row.items()):
            if str(k).startswith("sim") and float(v) > most_similar_v:
                most_similar_v = v
                most_similar_i = i

        return row.index[most_similar_i - 1]
