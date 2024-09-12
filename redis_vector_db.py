import os
import pandas as pd
import numpy as np
from itertools import product
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
import pretty_midi
import redis
from redis.exceptions import ResponseError
from redis.commands.search.field import (
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

DATASET = "test"
P_DATASET = f"data/datasets/{DATASET}"
P_CLAMP_EMBEDDINGS = f"data/{DATASET}-clamp.parquet"
METRIC = "pitch_histogram"
N_BEATS = 8
N_TRANSPOSITIONS = 12
DF = pd.read_parquet(P_CLAMP_EMBEDDINGS)

def get_pitch_histograms(fp_midis: list[str]) -> list[list[float]]:
    return [pretty_midi.PrettyMIDI(fp).get_pitch_class_histogram(use_duration=True, use_velocity=True).astype(np.float32).tolist() for fp in fp_midis]

def get_clamp_embeddings(fp_midis: list[str]) -> list[list[float]]:
    filenames = [os.path.basename(fp)[:-4] for fp in fp_midis]
    # df = pd.read_parquet(P_CLAMP_EMBEDDINGS)
    df = DF[DF.index.isin(filenames)]
    return df["embedding"].apply(lambda x: x.tolist()).tolist()

def upload_metric(r, names: list[str]):
    mod_table = list(product(range(N_TRANSPOSITIONS), range(N_BEATS)))
    vector_dim = 0

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        refresh_per_second=10,
    )
    metric_upload_task = progress.add_task(f"uploading '{METRIC}'", total=len(names))
    with progress:
        for f_base in names:
            # get metric for each transformation
            pf_full = []
            transforms = []
            for trans, beat in mod_table:
                transformation = f"t{trans:02d}s{beat:02d}"
                transforms.append(transformation)
                pf_full.append(os.path.join(P_DATASET, "train", f"{f_base}_{transformation}.mid"))
            match METRIC:
                case "pitch_histogram":
                    metrics = get_pitch_histograms(pf_full)
                case "clamp":
                    metrics = get_clamp_embeddings(pf_full)
                case _:
                    raise ValueError(f"Invalid metric: {METRIC}")
            vector_dim = len(metrics[0])

            # update redis
            for t, m in zip(transforms, metrics):
                r_key = f"files:{f_base}_{t}"
                track, seg = f_base.split('_')
                info = {"track": track, "segment": seg, "transforms": t}
                r.json().set(r_key, "$", info, nx=True)
                r.json().set(r_key, f"$.{METRIC}", m)

            progress.advance(metric_upload_task)
    return vector_dim

def main():
    # setup
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    all_names = [name[:-4] for name in os.listdir(os.path.join(P_DATASET, "play")) if name.endswith(".mid")]
    all_names.sort()
    num_processes = os.cpu_count()
    split_keys = np.array_split(all_names, num_processes) # type: ignore

    # actually upload the data
    vector_dim = upload_metric(r, all_names)

    # create index
    print(f"Creating index for {METRIC} with dim {vector_dim}")
    index_name = f"idx:files_{METRIC}_vss"
    try:
        r.ft().search(index_name)
        r.ft(index_name).dropindex(delete_documents=True)
    except ResponseError as e:
        print(e)
    schema = (
        TextField("$.track", no_stem=True, as_name="track"),
        TextField("$.segment", no_stem=True, as_name="segment"),
        TextField("$.transforms", no_stem=True, as_name="transforms"),
        VectorField(f"$.{METRIC}", "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": vector_dim,
            "DISTANCE_METRIC": "COSINE",
        }, as_name=METRIC),
    )
    definition = IndexDefinition(prefix=["files:"], index_type=IndexType.JSON)
    res = r.ft(index_name).create_index(fields=schema, definition=definition)
    print(f"Index creation result: {res}")

if __name__ == "__main__":
    main()