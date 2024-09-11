import os
import numpy as np
import pandas as pd

from utils import console, extract_transformations
from .worker import Worker

NEIGHBOR_COL_PRIORITIES = ["next", "next_2", "prev", "prev_2"]


class Seeker(Worker):
    sim_table: pd.DataFrame
    neighbor_table: pd.DataFrame
    trans_table: pd.DataFrame
    n_track_repeats: int = 0
    played_files: list[str] = []
    allow_multiple_plays = False
    transformation = {"transpose": 0, "shift": 0}

    def __init__(
        self, params, table_path: str, dataset_path: str, verbose: bool = False
    ):
        # load state
        self.tag = params.tag
        self.params = params
        self.mode = params.mode
        self.p_table = table_path
        self.p_dataset = dataset_path
        self.rng = np.random.default_rng(self.params.seed)
        self.verbose = verbose

        # load similarity table
        pf_sim_table = os.path.join(self.p_table, "sim.parquet")
        console.log(f"{self.tag} looking for similarity table '{pf_sim_table}'")
        if os.path.isfile(pf_sim_table):
            with console.status("\t\t\t      loading similarities file..."):
                self.sim_table = pd.read_parquet(pf_sim_table)
            console.log(
                f"{self.tag} loaded {len(self.sim_table)}*{len(self.sim_table.columns)} sim table"
            )
            console.log(self.sim_table.head())
        else:
            console.log(f"{self.tag} error loading similarity table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        # load neighbor table
        pf_neighbor_table = os.path.join(self.p_table, "neighbor.parquet")
        console.log(f"{self.tag} looking for neighbor table '{pf_neighbor_table}'")
        if os.path.isfile(pf_neighbor_table):
            with console.status("\t\t\t      loading neighbor file..."):
                self.neighbor_table = pd.read_parquet(pf_neighbor_table)
            console.log(
                f"{self.tag} loaded {len(self.neighbor_table)}*{len(self.neighbor_table.columns)} neighbor table"
            )
            console.log(self.neighbor_table.head())
        else:
            console.log(f"{self.tag} error loading neighbor table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        # load transformation table
        pf_trans_table = os.path.join(self.p_table, "transformations.parquet")
        console.log(f"{self.tag} looking for tranformation table '{pf_trans_table}'")
        if os.path.isfile(pf_trans_table):
            with console.status("\t\t\t      loading tranformation file..."):
                self.trans_table = pd.read_parquet(pf_trans_table)
            console.log(
                f"{self.tag} loaded {len(self.trans_table)}*{len(self.trans_table.columns)} transformation table"
            )
            console.log(self.trans_table.head())
        else:
            console.log(f"{self.tag} error loading tranformation table, exiting...")
            exit()  # TODO: handle this better (return an error, let main handle it)

        console.log(f"{self.tag} [green]successfully loaded tables")
        console.log(f"{self.tag} initialization complete")

    def get_next(self) -> str:
        match self.mode:
            case "best":
                next_file = self._get_next()
            case "easy":
                next_file = self._get_easy()
            case "sequential":
                next_file = self._get_neighbor()
            case "repeat":
                next_file = self.played_files[0]
            case "random" | "shuffle" | _:
                next_file = self._get_random()

        self.played_files.append(next_file)

        return os.path.join(self.p_dataset, next_file)

    def get_random(self) -> str:
        """returns a random file from the dataset"""
        random_file = self._get_random()
        self.played_files.append(random_file)

        return os.path.join(self.p_dataset, random_file)

    def _get_next(self) -> str:
        if self.verbose:
            console.log(
                f"{self.tag} finding most similar file to '{self.played_files[-1]}'"
            )

        [filename, transformations] = extract_transformations(self.played_files[-1])
        parent_track, _ = filename.split("_")
        console.log(
            f"{self.tag} extracted '{self.played_files[-1]}' -> '{filename}' and {transformations}"
        )
        sorted_row = self.sim_table.loc[filename].sort_values(
            ascending=False, key=lambda x: x.str["sim"]
        )

        return self._get_random()

    def _get_easy(self) -> str:
        if self.verbose:
            console.log(f"{self.tag} played files: {self.played_files}")
            console.log(f"{self.tag} num_repeats: {self.n_track_repeats}")
        if self.n_track_repeats < 8:
            console.log(f"{self.tag} transitioning to next segment")
            self.n_track_repeats += 1
            return self._get_neighbor()
        else:
            console.log(f"{self.tag} transitioning to next track")
            self.n_track_repeats = 0
            return self._get_random()

    def _get_neighbor(self) -> str:
        current_file = os.path.basename(self.played_files[-1])

        for col_name in NEIGHBOR_COL_PRIORITIES:
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

        console.log(
            f"{self.tag} unable to find neighbor for '{current_file}', choosing randomly"
        )

        return self._get_random()

    def _get_random(self) -> str:
        if self.verbose:
            console.log(f"{self.tag} choosing random file")
        random_file = self.rng.choice(
            [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
        )

        # only play files once
        if not self.allow_multiple_plays:
            while random_file in self.played_files:
                random_file = self.rng.choice(
                    [m for m in os.listdir(self.p_dataset) if m.endswith(".mid")]
                )

        return str(random_file)
