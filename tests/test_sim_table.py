import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from pretty_midi import PrettyMIDI, Instrument, Note
from mido import MidiFile, MetaMessage, MidiTrack, bpm2tempo
import matplotlib.pyplot as plt

# filesystem settings
dataset_name = "20240621"
table_dir = os.path.join("..", "..", "data", "tables")
table_name = f"{dataset_name}_sim.parquet"
data_dir = os.path.join("..", "..", "data", "datasets", dataset_name, "play")
output_path = os.path.join(
    "outputs", f"{datetime.now().strftime('%y%m%d-%H%M')}_{dataset_name}"
)

# test settings
num_neighbors = 5  # number of neighbors to find
exclude_same_parent = True  # only look for segments from different tracks

# random settings
seed = 9

def get_neighbors(
    table: pd.DataFrame, seed_file: str, count: int, diff_track: bool, nearest=True
):
    parent_track, segment = seed_file.split("_")
    segment = segment[:-4]

    sim_values = table.loc[seed_file].apply(lambda x: x["sim"])
    transformations = table.loc[seed_file].apply(lambda x: x["transformations"])
    top_neighbors = sim_values.nlargest(len(sim_values)).index.tolist()

    neighbors = [
        {
            "file": n,
            "sim": sim_values[n],
            "shift": transformations[n]["shift"],
            "trans": transformations[n]["transpose"],
        }
        for n in top_neighbors
    ]

    if diff_track:
        neighbors = [f for f in neighbors if f["file"].split("_")[0] != parent_track]

    neighbors.sort(key=lambda x: x["sim"], reverse=nearest)
    neighbors = neighbors[:count]

    return neighbors

def change_tempo(file_path: str, tempo: int):
    midi = MidiFile(file_path)
    new_tempo = bpm2tempo(tempo)
    new_message = MetaMessage("set_tempo", tempo=new_tempo, time=0)
    tempo_added = False

    for track in midi.tracks:
        # remove existing set_tempo messages
        for msg in track:
            if msg.type == "set_tempo":
                track.remove(msg)

        # add new set_tempo message to the first track
        if not tempo_added:
            track.insert(0, new_message)
            tempo_added = True

    # if no tracks had a set_tempo message and no new one was added, add a new track with the tempo message
    if not tempo_added:
        new_track = MidiTrack()
        new_track.append(new_message)
        midi.tracks.append(new_track)

    midi.save(file_path)
    

def transform(file_path: str, out_dir: str, tempo: int, transformations, num_beats: int = 8, prefix=0) -> str:
    print(f"transforming '{Path(file_path).stem}'", transformations)
    new_filename = f"{prefix}_{Path(file_path).stem}_t{transformations["transpose"]:02d}s{transformations["shift"]:02d}"
    out_path = os.path.join(out_dir, f"{new_filename}.mid")
    MidiFile(file_path).save(out_path) # in case transpose is 0

    if transformations["transpose"] != 0:
        t_midi = PrettyMIDI(initial_tempo=tempo)

        for instrument in PrettyMIDI(out_path).instruments:
            transposed_instrument = Instrument(program=instrument.program, name=new_filename)

            for note in instrument.notes:
                transposed_instrument.notes.append(
                    Note(
                        velocity=note.velocity,
                        pitch=note.pitch + int(transformations["transpose"]),
                        start=note.start,
                        end=note.end,
                    )
                )

            t_midi.instruments.append(transposed_instrument)

        t_midi.write(out_path)

    if transformations["shift"] != 0:
        s_midi = PrettyMIDI(initial_tempo=tempo)
        seconds_per_beat = 60 / tempo
        shift_seconds = transformations["shift"] * seconds_per_beat
        loop_point = (num_beats + 1) * seconds_per_beat

        for instrument in PrettyMIDI(out_path).instruments:
            shifted_instrument = Instrument(
                program=instrument.program, name=new_filename
            )
            for note in instrument.notes:
                dur = note.end - note.start
                shifted_start = (note.start + shift_seconds) % loop_point
                shifted_end = shifted_start + dur

                if note.start + shift_seconds >= loop_point:
                    shifted_start += seconds_per_beat
                    shifted_end += seconds_per_beat

                shifted_instrument.notes.append(
                    Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=shifted_start,
                        end=shifted_end
                    )
                )

            s_midi.instruments.append(shifted_instrument)

        s_midi.write(out_path)

    change_tempo(out_path, tempo)

    return out_path

def plot_histograms(
    histograms,
    titles,
    save_path=None,
    shape=None,
    main_title=None,
) -> None:
    plt.style.use("dark_background")

    if shape is None:
        shape = [len(histograms), 1]

    plt.figure(figsize=(12, 12))

    if main_title:
        plt.suptitle(main_title)

    for num_plot, _ in enumerate(histograms):
        plt.subplot(shape[0], shape[1], num_plot + 1)
        plt.bar(range(12), histograms[num_plot])
        plt.xticks(
            range(12), ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        )
        plt.title(titles[num_plot])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
    
def sim_table(table, rng):
	print(table.head())
	
	seed_file = rng.choice(table.columns)
	print(f"finding nearest neighbors to '{seed_file}'")

	plot_histograms(
		[PrettyMIDI(os.path.join(data_dir, seed_file)).get_pitch_class_histogram()],
		[Path(seed_file).stem],
	)

	neighbors = get_neighbors(table, seed_file, num_neighbors, exclude_same_parent, True)

	# transform and copy nearest neighbors
	for i, neighbor in enumerate(neighbors):
		print(
			f"'{neighbor['file']}' has similarity {neighbor['sim']:.03f} with the transformations: [s: {neighbor['shift']:02d}, t: {neighbor['trans']:02d}]"
		)
		print(table.loc[seed_file, neighbor["file"]])
		# plot
		titles = [Path(neighbors[i]["file"]).stem]
		phs = [
			PrettyMIDI(
				os.path.join(data_dir, neighbors[i]["file"])
			).get_pitch_class_histogram(True, True)
		]

		# transform
		tempo = int(neighbor["file"].split("-")[1])
		new_filename = transform(
			os.path.join(data_dir, neighbor["file"]),
			output_path,
			tempo,
			{"shift": neighbor["shift"], "transpose": neighbor["trans"]},
			prefix=f"n{i+1}",
		)
		neighbors[i]["file"] = new_filename

		phs.append(PrettyMIDI(neighbors[i]["file"]).get_pitch_class_histogram(True, True))
		titles.append(Path(neighbors[i]["file"]).stem)
		plot_histograms(
			phs,
			titles,
		)

	# copy over seed file
	MidiFile(os.path.join(data_dir, seed_file)).save(
		os.path.join(output_path, f"s_{seed_file}")
	)
	neighbors.insert(0, {"file": os.path.join(output_path, f"s_{seed_file}")})
     
	get_hist = lambda x: PrettyMIDI(x).get_pitch_class_histogram(True, True)
	plot_histograms(
		[get_hist(n["file"]) for n in neighbors],
		[Path(n["file"]).stem for n in neighbors],
		os.path.join(output_path, "nearest_phs.jpg"),
		main_title=f"Nearest Neighbors of {seed_file}",
	)
     
	print(f"finding furthest neighbors from '{seed_file}'")
	neighbors = get_neighbors(table, seed_file, num_neighbors, exclude_same_parent, False)

	# transform and copy furthest neighbors
	for i, neighbor in enumerate(neighbors):
		print(
			f"'{neighbor['file']}' has similarity {neighbor['sim']:.03f} with the transformations: [s: {neighbor['shift']:02d}, t: {neighbor['trans']:02d}]"
		)
		tempo = int(neighbor["file"].split("-")[1])
		new_filename = transform(
			os.path.join(data_dir, neighbor["file"]),
			output_path,
			tempo,
			{"shift": neighbor["shift"], "transpose": neighbor["trans"]},
			prefix=f"f{i + 1}",
		)
		neighbors[i]["file"] = new_filename

	neighbors.insert(0, {"file": os.path.join(output_path, f"s_{seed_file}")})

if __name__=="__main__":
    # filesystem setup
	table_path = os.path.join(table_dir, table_name)
	if not os.path.isdir(output_path):
		os.makedirs(output_path)
	if not os.path.exists(table_path):
		print(f"unable to find table at '{table_path}'")
		FileNotFoundError
          
	table = pd.read_parquet(table_path)	

	# init rng
	rng = np.random.default_rng(seed)
     
	sim_table(table, rng)
