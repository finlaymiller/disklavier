import os
import json
import mido
import numpy as np
import pretty_midi
from utils import basename
from itertools import product
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from queue import PriorityQueue
from argparse import ArgumentParser
from datetime import datetime, timedelta

from utils import basename, console, midi, write_log
from utils.novelty import gen_ssm_and_novelty
from workers import Seeker, Scheduler

TRIM_ROLL = False

CONSTANTS = {
    "system": "test",
    "seed": 0,
    "bpm": 60,
    "ticks_per_beat": 220,
    "n_beats_per_segment": 8,
    "n_transitions": 16,
    "n_min_queue_length": 100,
    "startup_delay": 10,
    "initialization": "kickstart",
    "kickstart_path": "data/datasets/test/test/intervals-060-09_1_t00s00.mid",
    "graph_track_revisit_interval": 5,
}

OPTIONS = {
    "dataset": [
        "20250320",
        "20250410",
        "20250420",
    ],
    "match": ["current", "next"],
    "mode": ["best", "graph"],
    "metric": [
        "pitch-histogram",
        "specdiff",
        "clamp",
    ],
    "graph_steps": [5, 9, 17],
    "seed_rearrange": [False, True],
    "seed_remove": [0.0, 1.0, 0.25, 0.5, 0.75],
}

seeker = None
scheduler = None
ts_queue = 0
n_files_queued = 0


def main(args):
    global seeker, scheduler, ts_queue, n_files_queued
    start_time = datetime.now()
    # get all possible combinations of settings
    keys = list(OPTIONS.keys())
    values = list(OPTIONS.values())
    product_combo = list(product(*values))
    settings = [dict(zip(keys, combo)) for combo in product_combo]
    console.log(f"running with {len(settings)} settings combinations")

    clean_conf = OmegaConf.load("params/template.yaml")
    for i, setting in enumerate(settings):
        # set up settings
        console.log(f"running with settings:\n\t{setting}")
        try:
            params = OmegaConf.merge(clean_conf, CONSTANTS, setting)
            params.tables = os.path.join(args.data_dir, params.dataset)
            params.dataset_path = os.path.join(params.tables, "augmented")
            params.scheduler.n_beats_per_segment = params.n_beats_per_segment
            params.seeker.system = params.system
            params.scheduler.system = params.system
            params.seeker.match = params.match
            params.seeker.mode = params.mode
            params.seeker.metric = params.metric
            if params.mode == "graph":
                params.seeker.graph_steps = params.graph_steps
                params.seeker.graph_track_revisit_interval = (
                    params.graph_track_revisit_interval
                )
            np.random.seed(params.seed)
            console.log(f"build config:")
            console.print_json(json.dumps(OmegaConf.to_object(params), indent=4))

            start_str = start_time.strftime("%Y%m%d_%H%M%S")
            run_name_parts = [f"{k.replace('_', '-')}={v}" for k, v in setting.items()]
            run_name = "_".join(run_name_parts)
            output_dir = f"data/tests/compare/{start_str}/{i}_{run_name}"
            os.makedirs(output_dir, exist_ok=True)
            console.log(f"output will be saved to '{output_dir}'")
            OmegaConf.save(params, os.path.join(output_dir, "config.yaml"))
            p_log = os.path.join(
                output_dir,
                f"{start_str}",
            )
            p_playlist = os.path.join(p_log, "playlist")
            os.makedirs(p_log)
            os.makedirs(p_playlist)
            pf_playlist = os.path.join(p_log, f"playlist_{start_str}.csv")
            write_log(pf_playlist, "position", "start time", "file path", "similarity")

            p_aug = os.path.join(p_log, "augmentations")
            os.makedirs(p_aug, exist_ok=True)

            # init workers
            seeker = Seeker(
                params.seeker,
                p_aug,
                params.tables,
                params.dataset_path,
                p_playlist,
                params.bpm,
            )
            scheduler = Scheduler(
                params.scheduler,
                params.bpm,
                p_log,
                p_playlist,
                start_time,
                params.n_transitions,
                params.initialization == "recording",
            )

            # ready system
            ts_queue = 0
            n_files_queued = 0
            scheduler.init_schedule(
                os.path.join(output_dir, "schedule.mid"),
            )

            q_playback = PriorityQueue()
            _queue_file(
                params.kickstart_path, None, q_playback, pf_playlist, start_time
            )

            td_start = datetime.now() + timedelta(seconds=5)
            scheduler.td_start = td_start
            similarities = []

            if params.seed_rearrange or params.seed_remove:
                pf_augmentations = augment_midi(
                    params.kickstart_path,
                    params.seed_rearrange,
                    params.seed_remove,
                    params.bpm,
                    p_aug,
                    params.dataset_path,
                )
                console.log(
                    f"got {len(pf_augmentations)} augmentations:\n\t{pf_augmentations}"
                )
                for pf_aug in pf_augmentations:
                    _queue_file(pf_aug, None, q_playback, pf_playlist, td_start)

            # load queue
            while n_files_queued < params.n_transitions:
                pf_next_file, similarity = seeker.get_next()
                _queue_file(pf_next_file, similarity, q_playback, pf_playlist, td_start)
                similarities.append(similarity)

            # build midi file
            tt_tracker = 0
            midi_file = os.path.join(output_dir, f"{start_str}-{i}.mid")
            midi_md = mido.MidiFile(ticks_per_beat=params.ticks_per_beat)
            # notes
            note_track = mido.MidiTrack()
            note_track.append(
                mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(params.bpm))
            )
            num_notes = 0
            note_track.name = f"{i}"
            while not q_playback.empty():
                tt_msg, msg = q_playback.get()
                if num_notes == 0:
                    msg.time = 0
                else:
                    msg.time = tt_msg - tt_tracker
                tt_tracker = tt_msg
                note_track.append(msg)
                num_notes += 1
            midi_md.tracks.append(note_track)

            # beats
            beat_track = mido.MidiTrack()
            beat_track.name = "beats"
            for i in range(params.n_beats_per_segment + 1):
                beat_track.append(
                    mido.MetaMessage(
                        "text",
                        text=f"beat {i}",
                        time=mido.second2tick(
                            60 / params.bpm if i > 0 else 0,
                            midi_md.ticks_per_beat,
                            mido.bpm2tempo(params.bpm),
                        ),
                    )
                )
            midi_md.tracks.append(beat_track)

            # transitions
            transition_track = mido.MidiTrack()
            transition_track.name = "transitions"
            for i in range(len(scheduler.queued_files)):
                transition_track.append(
                    mido.MetaMessage(
                        "text",
                        text=f"{basename(scheduler.queued_files[i])}",
                        time=mido.second2tick(
                            scheduler.ts_transitions[0] if i > 0 else 0,
                            midi_md.ticks_per_beat,
                            mido.bpm2tempo(params.bpm),
                        ),
                    )
                )
            midi_md.tracks.append(transition_track)
            midi_md.save(midi_file)

            # save PR
            sr = 100
            piano_roll = pretty_midi.PrettyMIDI(midi_file).get_piano_roll(sr)
            piano_roll_path = os.path.join(output_dir, f"{start_str}-{i}-pr.png")

            # create figure with minimal padding
            plt.figure(figsize=(12, 4))

            # trim and upsample piano roll to fixed resolution
            if TRIM_ROLL:
                trimmed_roll = midi.trim_piano_roll(piano_roll)
                upsampled_roll = midi.upsample_piano_roll(trimmed_roll)
            else:
                upsampled_roll = midi.upsample_piano_roll(piano_roll)
            plt.imshow(upsampled_roll, aspect="auto", origin="lower", cmap="gray_r")

            # add vertical lines at transition times
            for transition in transition_track:
                # convert ticks to seconds
                time_in_seconds = mido.tick2second(
                    transition.time, midi_md.ticks_per_beat, mido.bpm2tempo(params.bpm)
                )
                # convert seconds to piano roll samples and scale to upsampled width
                x_pos = int(time_in_seconds * sr * (1200 / piano_roll.shape[1]))
                plt.axvline(x=x_pos, color="blue", alpha=0.5, linewidth=0.5)

            # # generate novelty curve
            # ssm, novelty = gen_ssm_and_novelty(midi_file)
            # # account for downsampling factor of 10 in novelty curve
            # scaling_factor = upsampled_roll.shape[1] / piano_roll.shape[1]
            # x = (
            #     np.arange(len(novelty)) * scaling_factor * 10
            # )  # stretch x-axis to match piano roll width
            # plt.plot(
            #     x,
            #     (1 - novelty / novelty.max()) * 100,
            #     "g",
            #     linewidth=1.0,
            #     alpha=0.7,
            # )

            # plt.xticks([])
            # plt.yticks([])
            # plt.axis("off")

            # remove all padding and make the plot fill the figure
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(piano_roll_path, bbox_inches="tight", pad_inches=0)
            plt.close()

            # # save SSM
            # plt.figure(figsize=(16, 16))
            # plt.imshow(ssm / ssm.max(), cmap="magma")
            # plt.xticks([])
            # plt.yticks([])
            # plt.axis("off")
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            # plt.savefig(os.path.join(output_dir, f"{start_str}-{i}-ssm.png"))
            # plt.close()
        except Exception as e:
            console.log(f"[bold red]Error processing setting {i}: {setting}[/bold red]")
            console.print_exception(show_locals=True)
            continue  # continue to the next setting

    console.log(f"comparison completed in {datetime.now() - start_time}")


def augment_midi(
    pf_midi: str,
    seed_rearrange,
    seed_remove,
    bpm: int,
    aug_path: str,
    dataset_path: str,
) -> list[str]:
    # generate augmentations
    console.log(f"augmenting '{basename(pf_midi)}'")
    import pretty_midi

    global seeker
    if seeker is None:
        console.print_exception(show_locals=True)
        raise Exception("seeker not initialized")

    try:
        midi_paths = []
        if seed_rearrange:
            split_beats = midi.beat_split(pf_midi, bpm)
            console.log(
                f"\t\tsplit '{basename(pf_midi)}' into {len(split_beats)} beats"
            )
            ids = list(range(len(split_beats)))
            rearrangements: list[list[int]] = [
                ids,  # original
                ids[: len(ids) // 2],  # first half
                ids[: len(ids) // 2] * 2,  # first half twice
                ids[len(ids) // 2 + 1 :],  # second half
                ids[len(ids) // 2 + 1 :] * 2,  # second half twice
                ids[0:4],  # first four
                ids[0:4] * 2,  # first four twice
                ids[-4:],  # last four
                ids[-4:] * 2,  # last four twice
                [ids[-2], ids[-1]] * 4,  # last two beats
                [ids[-1]] * 8,  # last beat
            ]
            for i, arrangement in enumerate(rearrangements):
                console.log(f"\t\trearranging seed:\t{arrangement}")
                joined_midi: pretty_midi.PrettyMIDI = midi.beat_join(
                    split_beats, arrangement, bpm
                )

                console.log(f"\t\tjoined midi:\t{joined_midi.get_end_time()} s")

                if joined_midi.get_end_time() == 0:
                    console.log(
                        f"\t\tjoined midi is empty, skipping ({basename(pf_midi)}_a{i:02d}.mid)"
                    )
                    continue

                pf_joined_midi = os.path.join(
                    aug_path, f"{basename(pf_midi)}_a{i:02d}.mid"
                )
                joined_midi.write(pf_joined_midi)
                midi_paths.append(pf_joined_midi)
        else:
            midi_paths.append(pf_midi)  # no rearrangement, use the original path

        if seed_remove:
            # start with the paths generated (or the original if no rearrangement)
            current_paths = midi_paths
            midi_paths = []  # this will hold the final list of paths after removal
            num_options = 0
            paths_after_removal = (
                []
            )  # Keep track of paths generated by removal for each input

            for mid_path in current_paths:
                stripped_paths = midi.remove_notes(mid_path, aug_path, seed_remove)
                console.log(
                    f"\t\tstripped {seed_remove * 100 if isinstance(seed_remove, float) else seed_remove}{'%' if isinstance(seed_remove, float) else ''} notes from '{basename(mid_path)}' (+{len(stripped_paths)} versions)"
                )
                paths_after_removal.extend(stripped_paths)  # Add the stripped versions
                num_options += len(stripped_paths)

            # midi_paths now contains all versions after note removal
            midi_paths.extend(paths_after_removal)
            console.log(
                f"\t\taugmented '{basename(pf_midi)}' into {num_options} files after removal"
            )

            best_aug = ""
            best_match = ""
            best_similarity = 0.0
            # Iterate through the flat list of paths after removal
            for m in midi_paths:
                embedding = seeker.get_embedding(m)
                if embedding is None or embedding.sum() == 0:
                    console.log(
                        f"\t\t{basename(m)} has no notes or embedding failed, skipping"
                    )
                    continue
                match, similarity = seeker.get_match(embedding)
                if similarity > best_similarity:
                    best_aug = m
                    best_match = match
                    best_similarity = similarity
                console.log(
                    f"\t\tbest match for '{basename(m)}' is '{basename(match)}' with similarity {similarity}"
                )
        else:
            # If not removing notes, midi_paths already contains the rearranged (or original) paths
            best_aug = ""
            best_match = ""
            best_similarity = 0.0
            # find best match for each augmentation in the flat list
            for mid in midi_paths:
                embedding = seeker.get_embedding(mid, model=seeker.params.metric)
                if embedding is None or embedding.sum() == 0:
                    console.log(
                        f"\t\t{basename(mid)} has no notes or embedding failed, skipping"
                    )
                    continue
                match, similarity = seeker.get_match(embedding)
                if similarity > best_similarity:
                    best_aug = mid
                    best_match = match
                    best_similarity = similarity
                console.log(
                    f"\t\tbest match for '{basename(mid)}' is '{basename(match)}' with similarity {similarity}"
                )

        console.log(
            f"\tbest augmentation is '{basename(best_aug)}' with similarity {best_similarity} matches to {basename(best_match)}"
        )

        final_paths = []
        if seed_remove:
            if best_aug:
                final_paths.append(best_aug)

                # Add the best matching original file from the dataset
                if best_match:
                    final_paths.append(os.path.join(dataset_path, best_match + ".mid"))
            else:
                console.log(
                    "[yellow]Warning: No suitable augmentation found after removal, returning original file.[/yellow]"
                )
                return [pf_midi]
        else:
            # If not removing, just return all generated/original paths (which are already flat)
            final_paths = midi_paths

        return final_paths

    except Exception as e:
        console.log(
            f"[bold red]Error during MIDI augmentation for '{basename(pf_midi)}':[/bold red]"
        )
        console.print_exception(show_locals=True)
        return [pf_midi]  # return original path on error


def _queue_file(
    file_path: str,
    similarity: float | None,
    playback_queue: PriorityQueue,
    playlist_path: str,
    td_start: datetime,
):
    global seeker, scheduler, ts_queue, n_files_queued
    if seeker is None or scheduler is None:
        console.print_exception(show_locals=True)
        raise Exception("seeker or scheduler not initialized")

    try:
        # log queuing action
        console.log(f"queuing '{file_path}'")

        ts_seg_len, ts_seg_start, tt_end_tick = scheduler.enqueue_midi(
            file_path, playback_queue, similarity=similarity
        )

        if ts_seg_len is None or ts_seg_start is None:
            console.log(
                f"[yellow]Warning: Failed to enqueue '{basename(file_path)}'. Skipping.[/yellow]"
            )
            return  # don't proceed if enqueue failed

        ts_queue += ts_seg_len
        seeker.played_files.append(file_path)
        n_files_queued += 1
        start_time = td_start + timedelta(seconds=ts_seg_start)

        write_log(
            playlist_path,
            n_files_queued,
            start_time.strftime("%y-%m-%d %H:%M:%S"),
            file_path,
            similarity if similarity is not None else "----",
        )
    except Exception as e:
        console.log(f"[bold red]Error queuing file '{basename(file_path)}':[/bold red]")
        console.print_exception(show_locals=True)
        # Decide if we should re-raise or just log and continue
        # For now, just log and continue to avoid halting the whole process


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", type=str, default="/media/scratch/sageev-midi"
    )
    args = parser.parse_args()

    main(args)
