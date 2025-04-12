import os
import time
from pretty_midi import PrettyMIDI
from PySide6 import QtCore
from threading import Thread
from rich.table import Table
from queue import PriorityQueue
from datetime import datetime, timedelta

import workers
from workers import Staff
from utils import basename, console, midi, write_log

from typing import Optional


class RunWorker(QtCore.QThread):
    """
    worker thread that handles the main application run loop.
    """

    tag = "[#008700]runner[/#008700]:"

    # session tracking
    n_files = 0
    ts_queue = 0
    pf_augmentations = None

    # signals
    s_status = QtCore.Signal(str)
    s_start_time = QtCore.Signal(datetime)
    s_switch_to_pr = QtCore.Signal(object)
    s_transition_times = QtCore.Signal(list)

    # queues
    q_playback = PriorityQueue()
    q_gui = PriorityQueue()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.args = main_window.args
        self.params = main_window.params
        self.staff: Staff = main_window.workers
        self.td_system_start = main_window.td_system_start

        # Connect signals
        self.s_switch_to_pr.connect(self.main_window.switch_to_piano_roll)

        # File paths from main window
        self.p_log = main_window.p_log
        self.p_playlist = main_window.p_playlist
        self.pf_master_recording = main_window.pf_master_recording
        self.pf_system_recording = main_window.pf_system_recording
        self.pf_player_query = main_window.pf_player_query
        self.pf_player_accompaniment = main_window.pf_player_accompaniment
        self.pf_schedule = main_window.pf_schedule
        self.pf_playlist = main_window.pf_playlist

        # metronome
        self.metronome = workers.Metronome(
            self.params.metronome, self.params.bpm, self.td_system_start
        )

    def run(self):
        # get seed
        match self.params.initialization:
            case "recording":  # collect user recording
                if self.args.replay:  # use old recording
                    from pretty_midi import PrettyMIDI

                    ts_recording_len = PrettyMIDI(self.pf_player_query).get_end_time()
                    console.log(
                        f"{self.tag} calculated time of last recording {ts_recording_len}"
                    )
                    if ts_recording_len == 0:
                        raise ValueError("no recording found")
                else:
                    ts_recording_len = self.staff.midi_recorder.run()
                self.pf_seed = self.pf_player_query

                # augment
                if self.params.seed_rearrange or self.params.seed_remove:
                    self.pf_augmentations = self.augment_midi(self.pf_seed)
                    console.log(
                        f"{self.tag} got {len(self.pf_augmentations)} augmentations:\n\t{self.pf_augmentations}"
                    )
            case "audio":
                self.pf_player_query = self.pf_player_query.replace(".mid", ".wav")
                ts_recording_len = self.staff.audio_recorder.record_query(
                    self.pf_player_query
                )
                embedding = self.staff.seeker.get_embedding(
                    self.pf_player_query, model="clap"
                )
                console.log(
                    f"{self.tag} got embedding {embedding.shape} from pantherino"
                )
                best_match, best_similarity = self.staff.seeker.get_match(
                    embedding, metric="clap-sgm"
                )
                console.log(
                    f"{self.tag} got best match '{best_match}' with similarity {best_similarity}"
                )
                self.pf_seed = best_match
            case "kickstart":  # use specified file as seed
                try:
                    if self.params.kickstart_path:
                        self.pf_seed = self.params.kickstart_path
                        console.log(
                            f"{self.tag} [cyan]KICKSTART[/cyan] - '{self.pf_seed}'"
                        )
                except AttributeError:
                    console.log(
                        f"{self.tag} no file specified to kickstart from, choosing randomly"
                    )
                    self.pf_seed = self.staff.seeker.get_random()
                    console.log(
                        f"{self.tag} [cyan]RANDOM INIT[/cyan] - '{self.pf_seed}'"
                    )
            case "random" | _:  # choose random file from library
                self.pf_seed = self.staff.seeker.get_random()
                console.log(f"{self.tag} [cyan]RANDOM INIT[/cyan] - '{self.pf_seed}'")

        if self.params.seeker.mode == "playlist":
            # TODO: implement playlist mode using generated csv's
            raise NotImplementedError("playlist mode not implemented")

        self.staff.scheduler.init_schedule(
            self.pf_schedule,
            ts_recording_len if self.params.initialization == "recording" else 0,
        )
        console.log(f"{self.tag} successfully initialized recording")

        # add seed to queue
        self._queue_file(self.pf_seed, None)

        # add first match to queue
        if self.pf_augmentations is None:
            pf_next_file, similarity = self.staff.seeker.get_next()
            self._queue_file(pf_next_file, similarity)

        try:
            td_start = datetime.now() + timedelta(seconds=self.params.startup_delay)
            console.log(
                f"{self.tag} start time set to {td_start.strftime('%y-%m-%d %H:%M:%S')}"
            )
            self.s_start_time.emit(td_start)
            self.staff.scheduler.td_start = td_start
            self.metronome.td_start = td_start
            self.metronome.start()  # Start the QThread directly

            # start audio recording in a separate thread
            # TODO: fix ~1-beat delay in audio recording startup
            self.e_audio_stop = self.staff.audio_recorder.start_recording(td_start)

            # Switch to piano roll view using signal
            self.s_switch_to_pr.emit(self.q_gui)
            # start player
            self.staff.player.td_start = td_start
            self.staff.player.td_last_note = td_start
            self.thread_player = Thread(
                target=self.staff.player.play, name="player", args=(self.q_playback,)
            )
            self.thread_player.start()

            # start midi recording
            self.midi_stop_event = self.staff.midi_recorder.start_recording(td_start)
            self.main_window.midi_stop_event = self.midi_stop_event
            # connect recorder to player for velocity updates
            self.staff.player.set_recorder(self.staff.midi_recorder)

            if self.pf_augmentations is not None:
                for aug in self.pf_augmentations:
                    self._queue_file(aug, None)

            # play for set number of transitions
            # TODO: move this to be managed by scheduler and track scheduler state instead
            current_file = ""
            last_time = time.time()
            while self.n_files < self.params.n_transitions:
                current_time = time.time()
                elapsed = current_time - last_time
                last_time = current_time

                if current_file != self.staff.scheduler.get_current_file():
                    current_file = self.staff.scheduler.get_current_file()
                    console.log(f"{self.tag} now playing '{current_file}'")
                    self.s_status.emit(f"now playing '{current_file}'")

                if self.ts_queue < self.params.startup_delay * 2:
                    pf_next_file, similarity = self.staff.seeker.get_next()
                    self._queue_file(pf_next_file, similarity)
                    console.log(
                        f"{self.tag} queue time is now {self.ts_queue:.01f} seconds"
                    )

                time.sleep(0.001)
                self.ts_queue -= elapsed
                if not self.thread_player.is_alive():
                    console.log(f"{self.tag} player ran out of notes, exiting")
                    self.thread_player.join(0.1)
                    break

            # all necessary files queued, wait for playback to finish
            console.log(f"{self.tag} waiting for playback to finish...")
            while self.q_playback.qsize() > 0:
                if current_file != self.staff.scheduler.get_current_file():
                    current_file = self.staff.scheduler.get_current_file()
                    console.log(f"{self.tag} now playing '{current_file}'")
                    self.s_status.emit(f"now playing '{current_file}'")
                time.sleep(0.1)
            self.thread_player.join(timeout=0.1)
            console.log(f"{self.tag} playback complete")
            self.s_status.emit("playback complete")
        except KeyboardInterrupt:
            console.log(f"{self.tag}[yellow] CTRL + C detected, saving and exiting...")

        self.shutdown()

    def shutdown(self):
        console.log(f"{self.tag}[yellow] shutdown called, saving and exiting...")
        # dump queue to stop player
        while self.q_playback.qsize() > 0:
            try:
                _ = self.q_playback.get()
            except:
                if self.args.verbose:
                    console.log(
                        f"{self.tag} [yellow]ouch! tried to dump queue but failed"
                    )
                pass

        if hasattr(self, "thread_player"):
            self.thread_player.join(timeout=0.1)

        console.log(f"{self.tag} stopping metronome")
        self.metronome.stop()
        console.log(f"{self.tag} metronome stopped")

        # run complete, save and exit
        # close raw notes file
        self.staff.scheduler.raw_notes_file.close()

        # Stop audio recording if it's running
        if self.e_audio_stop is not None:
            console.log(f"{self.tag} stopping audio recording")
            self.staff.audio_recorder.stop_recording()

        # Stop MIDI passive recording if its running
        if self.midi_stop_event is not None:
            console.log(f"{self.tag} stopping MIDI recording")
            self.staff.midi_recorder.stop_recording()
            self.staff.midi_recorder.save_midi(self.pf_player_accompaniment)

        # convert queue to midi
        _ = self.staff.scheduler.queue_to_midi(self.pf_system_recording)
        _ = midi.combine_midi_files(
            [self.pf_system_recording, self.pf_player_accompaniment, self.pf_schedule],
            self.pf_master_recording,
        )

        # print playlist
        table = Table(title="PLAYLIST")
        with open(self.pf_playlist, mode="r") as file:
            headers = file.readline().strip().split(",")
            for header in headers:
                table.add_column(header)

            for line in file:
                row = line.strip().split(",")
                row[2] = os.path.basename(row[2])  # only print filenames
                table.add_row(*row)
        console.print(table)

        # plot piano roll if master recording exists
        if os.path.exists(self.pf_master_recording):
            console.log(f"{self.tag} generating piano roll visualization")
            midi.generate_piano_roll(self.pf_master_recording)

        console.save_text(os.path.join(self.p_log, f"{self.td_system_start}.log"))
        console.log(f"{self.tag}[green bold] shutdown complete, exiting")

    def _queue_file(self, file_path: str, similarity: float | None) -> None:
        ts_seg_len, ts_seg_start = self.staff.scheduler.enqueue_midi(
            file_path, self.q_playback, self.q_gui, similarity
        )

        self.ts_queue += ts_seg_len
        self.staff.seeker.played_files.append(file_path)
        self.n_files += 1
        self.s_transition_times.emit(self.staff.scheduler.ts_transitions)

        write_log(
            self.pf_playlist,
            self.n_files,
            datetime.fromtimestamp(ts_seg_start).strftime("%y-%m-%d %H:%M:%S"),
            file_path,
            similarity if similarity is not None else "----",
        )

    def augment_midi(
        self,
        pf_midi: str,
        seed_rearrange: Optional[bool] = None,
        seed_remove: Optional[float] = None,
    ) -> list[str]:
        # generate augmentations
        console.log(f"{self.tag} augmenting '{basename(pf_midi)}'")
        import pretty_midi

        # load from parameters if not provided
        if not seed_rearrange:
            seed_rearrange = self.params.seed_rearrange
        if not seed_remove:
            seed_remove = self.params.seed_remove

        pf_augmentations = os.path.join(self.p_log, "augmentations")
        if not os.path.exists(pf_augmentations):
            os.makedirs(pf_augmentations)

        midi_paths = []
        if seed_rearrange:
            split_beats = midi.beat_split(pf_midi, self.params.bpm)
            console.log(
                f"{self.tag}\tsplit '{basename(pf_midi)}' into {len(split_beats)} beats"
            )
            ids = list(range(len(split_beats)))
            rearrangements: list[list[int]] = [
                ids,  # original
                ids[: len(ids) // 2] * 2,  # first half twice
                ids[len(ids) // 2 :] * 2,  # second half twice need +1?
                [ids[-2], ids[-1]] * 4,  # last two beats
                [ids[-1]] * 8,  # last beat
            ]
            for i, arrangement in enumerate(rearrangements):
                console.log(f"{self.tag}\trearranging seed:\t{arrangement}")
                joined_midi: pretty_midi.PrettyMIDI = midi.beat_join(
                    split_beats, arrangement, self.params.bpm
                )

                console.log(f"{self.tag}\tjoined midi:\t{joined_midi.get_end_time()}")

                pf_joined_midi = os.path.join(
                    pf_augmentations, f"{basename(pf_midi)}_a{i:02d}.mid"
                )
                joined_midi.write(pf_joined_midi)
                midi_paths.append(pf_joined_midi)
        else:
            midi_paths.append(pf_midi)

        if seed_remove:
            joined_paths = midi_paths
            midi_paths = []  # TODO: stop overloading this
            num_options = 0
            for mid in joined_paths:
                stripped_paths = midi.remove_notes(mid, pf_augmentations, seed_remove)
                console.log(
                    f"{self.tag}\tstripped notes from '{basename(mid)}' (+{len(stripped_paths)})"
                )
                midi_paths.append(stripped_paths)
                num_options += len(stripped_paths)
            console.log(
                f"{self.tag}\taugmented '{basename(pf_midi)}' into {num_options} files"
            )

            best_aug = ""
            best_path = []
            best_match = ""
            best_similarity = 0.0
            for ps in midi_paths:
                console.log(f"{self.tag}\tps: {ps}")
                for m in ps:
                    embedding = self.staff.seeker.get_embedding(m)
                    match, similarity = self.staff.seeker.get_match(embedding)
                    console.log(
                        f"{self.tag}\t\tbest match for '{basename(m)}' is '{basename(best_match)}' with similarity {best_similarity}"
                    )
                    if similarity > best_similarity:
                        best_aug = m
                        best_path = ps
                        best_match = match
                        best_similarity = similarity
        else:
            best_aug = ""
            best_match = ""
            best_similarity = 0.0
            for ps in midi_paths:
                for m in ps:
                    console.log(f"{self.tag}\t\t{basename(m)}")

            # find best match for each augmentation
            for mid in midi_paths:
                embedding = self.staff.seeker.get_embedding(mid, model="clap")
                match, similarity = self.staff.seeker.get_match(embedding)
                console.log(
                    f"{self.tag}\t\tbest match for '{basename(mid)}' is '{basename(best_match)}' with similarity {best_similarity}"
                )
                if similarity > best_similarity:
                    best_aug = mid
                    best_match = match
                    best_similarity = similarity

        console.log(
            f"{self.tag}\tbest augmentation is '{basename(best_aug)}' with similarity {best_similarity} matches to {best_match}"
        )

        if seed_remove:
            # add em all up
            midi_paths = [
                *best_path[: best_path.index(best_aug) + 1],
                os.path.join(self.args.dataset_path, best_match + ".mid"),
            ]
            console.log(f"{self.tag}\t\tbp: {best_path}")
            console.log(f"{self.tag}\t\tmp: {midi_paths}")

        return midi_paths
