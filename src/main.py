import os
import csv
import time
from rich.table import Table
from threading import Thread
from omegaconf import OmegaConf
from queue import PriorityQueue
from argparse import ArgumentParser
from multiprocessing import Process
from datetime import datetime, timedelta

import workers
from utils import console, midi, write_log
from workers.panther import send_embedding

tag = "[white]main[/white]  :"

# TODO: UPDATE THIS TO MATCH RUNNER & APP


def main(args, params):
    td_system_start = datetime.now()
    ts_start = td_system_start.strftime("%y%m%d-%H%M%S")

    # filesystem setup
    # create session output directories
    p_log = os.path.join(
        args.output,
        f"{ts_start}_{params.seeker.metric}_{params.initialization}_{params.seeker.seed}",
    )
    p_playlist = os.path.join(p_log, "playlist")
    if not os.path.exists(p_log):
        if args.verbose:
            console.log(f"{tag} creating new logging folder at '{p_log}'")
        os.makedirs(p_log)
        os.makedirs(p_playlist)

    # specify recording files
    pf_master_recording = os.path.join(p_log, f"master-recording.mid")
    pf_system_recording = os.path.join(p_log, f"system-recording.mid")
    pf_player_query = os.path.join(p_log, f"player-query.mid")
    pf_player_accompaniment = os.path.join(p_log, f"player-accompaniment.mid")
    pf_schedule = os.path.join(p_log, f"schedule.mid")

    # copy old recording if replaying
    if args.replay and params.initialization == "recording":
        import shutil

        shutil.copy2(params.kickstart_path, pf_player_query)
        console.log(f"{tag} moved old recording to current folder '{pf_player_query}'")
        params.seeker.pf_recording = pf_player_query

    # initialize playlist file
    pf_playlist = os.path.join(p_log, f"playlist_{ts_start}.csv")
    write_log(pf_playlist, "position", "start time", "file path", "similarity")
    console.log(f"{tag} filesystem set up complete")

    # worker setup
    scheduler = workers.Scheduler(
        params.scheduler,
        args.bpm,
        p_log,
        p_playlist,
        td_system_start,
        params.n_transitions,
        params.initialization == "recording",
    )
    seeker = workers.Seeker(
        params.seeker, args.tables, args.dataset_path, p_playlist, args.bpm
    )
    player = workers.Player(params.player, args.bpm, td_system_start)
    midi_recorder = workers.MidiRecorder(
        params.recorder,
        args.bpm,
        pf_player_query,
    )
    midi_stop_event = None
    audio_recorder = workers.AudioRecorder(params.audio, args.bpm, p_log)
    audio_stop_event = None

    # data setup
    match params.initialization:
        case "recording":  # collect user recording
            if args.replay:  # use old recording
                from pretty_midi import PrettyMIDI

                ts_recording_len = PrettyMIDI(pf_player_query).get_end_time()
                console.log(
                    f"{tag} calculated time of last recording {ts_recording_len}"
                )
                if ts_recording_len == 0:
                    raise ValueError("no recording found")
            else:
                ts_recording_len = midi_recorder.run()
            pf_seed = pf_player_query
        case "audio":
            pf_player_query = pf_player_query.replace(".mid", ".wav")
            ts_recording_len = audio_recorder.record_query(pf_player_query)
            embedding = send_embedding(pf_player_query, model="clap")
            console.log(f"{tag} got embedding {embedding.shape} from pantherino")
            best_match, best_similarity = seeker.get_match(embedding)
            console.log(
                f"{tag} got best match '{best_match}' with similarity {best_similarity}"
            )
            pf_seed = best_match
        case "kickstart":  # use specified file as seed
            try:
                if params.kickstart_path:
                    pf_seed = params.kickstart_path
                    console.log(f"{tag} [cyan]KICKSTART[/cyan] - '{pf_seed}'")
            except AttributeError:
                console.log(
                    f"{tag} no file specified to kickstart from, choosing randomly"
                )
                pf_seed = seeker.get_random()
                console.log(f"{tag} [cyan]RANDOM INIT[/cyan] - '{pf_seed}'")
        case "random" | _:  # choose random file from library
            pf_seed = seeker.get_random()
            console.log(f"{tag} [cyan]RANDOM INIT[/cyan] - '{pf_seed}'")

    if params.seeker.mode == "playlist":
        # TODO: implement playlist mode using generated csvs
        raise NotImplementedError("playlist mode not implemented")

    ts_queue = 0
    q_playback = PriorityQueue()
    # calculate first few transitions
    scheduler.init_schedule(
        pf_schedule,
        ts_recording_len if params.initialization == "recording" else 0,
    )
    console.log(f"{tag} successfully initialized recording")

    # find first match
    # this is done before adding the seed to the queue to accomodate seed
    # augmentations for player recordings
    seeker.played_files.append(pf_seed)
    pf_next_file, similarity = seeker.get_next()

    # add seed to queue
    ts_seg_len, ts_seg_start = scheduler.enqueue_midi(pf_seed, q_playback)
    ts_queue += ts_seg_len
    n_files = 1  # number of files played so far
    console.log(f"{tag} enqueued seed at {ts_seg_start}")
    write_log(
        pf_playlist,
        n_files,
        datetime.fromtimestamp(ts_seg_start).strftime("%y-%m-%d %H:%M:%S"),
        pf_seed,
        "----",
    )

    # add first match to queue
    ts_seg_len, ts_seg_start = scheduler.enqueue_midi(pf_next_file, q_playback)
    ts_queue += ts_seg_len
    n_files += 1
    write_log(
        pf_playlist,
        n_files,
        datetime.fromtimestamp(
            td_system_start.timestamp()
            + timedelta(seconds=ts_seg_start).total_seconds()
        ).strftime("%y-%m-%d %H:%M:%S"),
        pf_next_file,
        similarity,
    )

    # run
    try:
        td_start = datetime.now() + timedelta(seconds=params.startup_delay / 2)
        # start metronome
        console.log(f"{tag} starting metronome for {td_start}")
        metronome = workers.Metronome(params.metronome, args.bpm, td_start)
        process_metronome = Process(target=metronome.run, name="metronome")
        process_metronome.start()

        scheduler.td_start = td_start

        # start audio recording in a separate thread
        # TODO: fix ~1-beat delay in audio recording startup
        audio_stop_event = audio_recorder.start_recording(td_start)
        # start player
        player.td_start = td_start
        player.td_last_note = td_start
        thread_player = Thread(target=player.play, name="player", args=(q_playback,))
        thread_player.start()

        # start midi recording
        midi_stop_event = midi_recorder.start_recording(td_start)
        # connect recorder to player for velocity updates
        player.set_recorder(midi_recorder)

        # play for set number of transitions
        # TODO: move this to be managed by scheduler and track scheduler state instead
        current_file = ""
        while n_files < params.n_transitions:
            if current_file != scheduler.get_current_file():
                current_file = scheduler.get_current_file()
                console.log(f"{tag} now playing '{current_file}'")

            if q_playback.qsize() < params.n_min_queue_length:
                pf_next_file, similarity = seeker.get_next()
                ts_seg_len, ts_seg_start = scheduler.enqueue_midi(
                    pf_next_file, q_playback
                )
                ts_queue += ts_seg_len
                console.log(f"{tag} queue time is now {ts_queue:.01f} seconds")
                n_files += 1
                write_log(
                    pf_playlist,
                    n_files,
                    datetime.fromtimestamp(
                        ts_seg_start - td_system_start.timestamp()
                    ).strftime("%y-%m-%d %H:%M:%S"),
                    pf_next_file,
                    similarity,
                )

            time.sleep(0.1)
            ts_queue -= 0.1
            if not thread_player.is_alive():
                console.log(f"{tag} player ran out of notes, exiting")
                thread_player.join(0.1)
                break

        metronome.stop()

        # all necessary files queued, wait for playback to finish
        console.log(f"{tag} waiting for playback to finish...")
        while q_playback.qsize() > 0:
            time.sleep(0.1)
        thread_player.join(timeout=0.1)
    except KeyboardInterrupt:
        console.log(f"{tag}[yellow] CTRL + C detected, saving and exiting...")
        # dump queue to stop player
        while q_playback.qsize() > 0:
            try:
                _ = q_playback.get()
            except:
                if args.verbose:
                    console.log(f"{tag} [yellow]ouch! tried to dump queue but failed")
                pass
        thread_player.join(timeout=0.1)

    # run complete, save and exit
    # kill metronome
    if args.verbose:
        console.log(f"{tag} stopping metronome")
    process_metronome.kill()
    process_metronome.join(timeout=0.5)
    if args.verbose:
        console.log(f"{tag} metronome stopped")

    # close raw notes file
    scheduler.raw_notes_file.close()

    # Stop audio recording if it's running
    if audio_stop_event is not None:
        console.log(f"{tag} stopping audio recording")
        audio_recorder.stop_recording()

    # Stop MIDI passive recording if its running
    if midi_stop_event is not None:
        console.log(f"{tag} stopping MIDI recording")
        midi_recorder.stop_recording()
        midi_recorder.save_midi(pf_player_accompaniment)

    # convert queue to midi
    _ = scheduler.queue_to_midi(pf_system_recording)
    _ = midi.combine_midi_files(
        [pf_system_recording, pf_player_accompaniment, pf_schedule], pf_master_recording
    )

    # print playlist
    table = Table(title="PLAYLIST")
    with open(pf_playlist, mode="r") as file:
        headers = file.readline().strip().split(",")
        for header in headers:
            table.add_column(header)

        for line in file:
            row = line.strip().split(",")
            row[2] = os.path.basename(row[2])  # only print filenames
            table.add_row(*row)
    console.print(table)

    # plot piano roll if master recording exists
    if os.path.exists(pf_master_recording):
        console.log(f"{tag} generating piano roll visualization")
        midi.generate_piano_roll(pf_master_recording)

    console.save_text(os.path.join(p_log, f"{ts_start}.log"))
    console.log(f"{tag}[green bold] session complete, exiting")
    return 0

if __name__ == "__main__":
    # load/build arguments and parameters
    parser = ArgumentParser(description="Argparser description")
    parser.add_argument(
        "-d", "--dataset", type=str, default=None, help="name of the dataset"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="path to MIDI files"
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="name of parameter file (must be located at 'params/[NAME].yaml')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/outputs/logs",
        help="directory in which to store outputs (logs, recordings, etc...)",
    )
    parser.add_argument(
        "-t",
        "--tables",
        type=str,
        default=None,
        help="directory in which precomputed tables are stored",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="option to override seeker metric in yaml file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="option to override seeker mode in yaml file",
    )
    parser.add_argument(
        "-b",
        "--bpm",
        type=int,
        help="bpm to record and play at, in bpm",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose output",
    )
    parser.add_argument(
        "-r",
        "--replay",
        action="store_true",
        help="run again using last seed file",
    )
    args = parser.parse_args()
    params = OmegaConf.load(f"params/{args.params}.yaml")

    # handle overrides
    if args.dataset_path == None:
        args.dataset_path = f"data/datasets/{args.dataset}/augmented"
    if args.tables == None:
        args.tables = f"data/tables/{args.dataset}"
    if args.metric != None:
        console.log(f"{tag} overriding seeker metric to '{args.metric}'")
        params.seeker.metric = args.metric
    if args.mode != None:
        console.log(f"{tag} overriding seeker mode to '{args.mode}'")
        params.seeker.mode = args.mode

    # copy these so that they only have to be specified once
    params.scheduler.n_beats_per_segment = params.n_beats_per_segment
    params.metronome.n_beats_per_segment = params.n_beats_per_segment

    if args.replay:
        # get path to last seed file
        entries = os.listdir(args.output)
        folders = [
            entry
            for entry in entries
            if os.path.isdir(os.path.join(args.output, entry))
        ]
        folders.sort()
        last_folder = folders[-1]
        console.log(f"{tag} last run is in folder '{last_folder}'")

        last_timestamp, _, last_initialization, _ = last_folder.split("_")
        pf_last_playlist = os.path.join(
            args.output, last_folder, f"playlist_{last_timestamp}.csv"
        )
        pf_last_seed = None
        with open(pf_last_playlist, newline="") as csvfile:
            import csv

            first_row = next(csv.DictReader(csvfile), None)
            pf_last_seed = (
                first_row["file path"]
                if first_row and "file path" in first_row
                else None
            )

        if pf_last_seed is None:
            raise FileNotFoundError("couldn't load seed file path")

        params.initialization = last_initialization
        params.kickstart_path = pf_last_seed

    console.log(f"{tag} loading with arguments:\n\t{args}")
    console.log(f"{tag} loading with parameters:\n\t{params}")

    if not os.path.exists(args.tables):
        console.log(
            f"{tag} [red bold]ERROR[/red bold]: table directory not found, exiting..."
        )
        raise FileNotFoundError("table directory not found")

    main(args, params)
