import os
import time
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
    ts_queue = 0
    n_files_queued = 0
    pf_augmentations = None
    playing_file = None

    # signals
    s_status = QtCore.Signal(str)
    s_start_time = QtCore.Signal(datetime)
    s_switch_to_pr = QtCore.Signal(object)
    s_transition_times = QtCore.Signal(list)
    s_segments_remaining = QtCore.Signal(int)

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

        # connect signals
        self.s_switch_to_pr.connect(self.main_window.switch_to_piano_roll)

        # file paths from main window
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
            last_time = time.time()
            while self.n_files_queued < self.params.n_transitions:
                current_time = time.time()
                elapsed = current_time - last_time
                last_time = current_time

                self._emit_current_file()

                if self.ts_queue < self.params.startup_delay * 2:
                    pf_next_file, similarity = self.staff.seeker.get_next()
                    self._queue_file(pf_next_file, similarity)
                    console.log(
                        f"{self.tag} queue time is now {self.ts_queue:.01f} seconds"
                    )

                time.sleep(0.1)
                self.ts_queue -= elapsed
                if not self.thread_player.is_alive():
                    console.log(f"{self.tag} player ran out of notes, exiting")
                    self.thread_player.join(0.1)
                    break

            # all necessary files queued, wait for playback to finish
            console.log(f"{self.tag} waiting for playback to finish...")
            while self.q_playback.qsize() > 0:
                self._emit_current_file()
                time.sleep(0.1)
            self.thread_player.join(timeout=0.1)
            console.log(f"{self.tag} playback complete")
            self.s_status.emit("playback complete")
            self.s_segments_remaining.emit(0)
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
        self.n_files_queued += 1
        self.s_transition_times.emit(self.staff.scheduler.ts_transitions)
        start_time = self.td_system_start + timedelta(seconds=ts_seg_start)

        write_log(
            self.pf_playlist,
            self.n_files_queued,
            start_time.strftime("%y-%m-%d %H:%M:%S"),
            file_path,
            similarity if similarity is not None else "----",
        )

    def augment_midi(
        self,
        pf_midi: str,
        seed_rearrange: Optional[bool] = None,
        seed_remove: Optional[float] = None,
    ) -> list[str]:
        """
        generate augmentations (rearrangements, note removals) for a seed midi file.

        Parameters
        ----------
        pf_midi : str
            path to the seed midi file.
        seed_rearrange : Optional[bool], optional
            whether to generate beat rearrangements, defaults to param value.
        seed_remove : Optional[float], optional
            percentage/amount of notes to remove, defaults to param value.

        Returns
        -------
        list[str]
            list of paths to augmented midi files, ordered by quality (best first),
            followed by the best matching file from the dataset.
        """
        console.log(f"{self.tag} augmenting '{basename(pf_midi)}'")
        import pretty_midi

        # load from parameters if not provided
        rearrange = seed_rearrange if seed_rearrange is not None else self.params.seed_rearrange
        remove_amount = seed_remove if seed_remove is not None else self.params.seed_remove
        pf_augmentations_dir = os.path.join(self.p_log, "augmentations")
        if not os.path.exists(pf_augmentations_dir):
            os.makedirs(pf_augmentations_dir)

        # --- Step 1: Generate Base Set (Original + Rearrangements) ---
        base_midi_paths = []
        if rearrange:
            split_beats = midi.beat_split(pf_midi, self.params.bpm)
            if not split_beats:
                console.log(f"[yellow]warning: could not split '{basename(pf_midi)}' into beats. skipping rearrangement.[/yellow]")
                base_midi_paths.append(pf_midi)
            else:
                console.log(f"{self.tag}\t\tsplit '{basename(pf_midi)}' into {len(split_beats)} beats")
                ids = list(split_beats.keys()) # use actual beat keys
                # define rearrangements (consider making this configurable)
                rearrangements: list[list[int]] = [
                    ids, # original order
                    ids[0:4], # first four
                    ids[0:4] * 2, # first four twice
                    ids[-4:], # last four
                    ids[-4:] * 2, # last four twice
                    [ids[-2], ids[-1]] * 4, # last two beats repeated
                    [ids[-1]] * 8, # last beat repeated
                ]
                # ensure original segment is always first, even if empty rearrangement list is defined
                if not rearrangements or rearrangements[0] != ids:
                    rearrangements.insert(0, ids)

                for i, arrangement in enumerate(rearrangements):
                    # Filter out invalid beat indices
                    valid_arrangement = [b for b in arrangement if b in split_beats]
                    if not valid_arrangement:
                        console.log(f"[yellow]warning: rearrangement {i} resulted in empty sequence after filtering invalid beats. skipping.[/yellow]")
                        continue

                    # console.log(f"{self.tag}\t\trearranging seed: {valid_arrangement}")
                    joined_midi: pretty_midi.PrettyMIDI = midi.beat_join(
                        split_beats, valid_arrangement, self.params.bpm
                    )
                    if not joined_midi or not any(inst.notes for inst in joined_midi.instruments):
                        console.log(f"[yellow]warning: rearrangement {i} resulted in empty midi. skipping.[/yellow]")
                        continue

                    # console.log(f"{self.tag}\t\tjoined midi: {joined_midi.get_end_time()} s")
                    pf_joined_midi = os.path.join(
                        pf_augmentations_dir, f"{basename(pf_midi)}_r{i:02d}.mid"
                    )
                    joined_midi.write(pf_joined_midi)
                    base_midi_paths.append(pf_joined_midi)
        else:
            base_midi_paths.append(pf_midi) # Start with the original if no rearrangement

        # --- Step 2: Apply Note Removal (if enabled) ---
        all_augmented_paths = []
        if remove_amount and remove_amount > 0:
            console.log(f"{self.tag}\tapplying note removal (amount={remove_amount}")
            for base_path in base_midi_paths:
                # Add the base path itself (0 notes removed)
                all_augmented_paths.append(base_path)
                # Generate versions with notes removed
                try:
                    
                    removed_note_paths = midi.remove_notes(
                        base_path,
                        pf_augmentations_dir,
                        remove_amount,
                        num_versions=1
                    )
                    if removed_note_paths:
                        console.log(
                             f"{self.tag}\t\tremoved {remove_amount * 100 if isinstance(remove_amount, float) and remove_amount < 1 else remove_amount}
                             {'%' if isinstance(remove_amount, float) and remove_amount < 1 else ''} notes from '{basename(base_path)}' -> +{len(removed_note_paths)} versions"
                         )
                        all_augmented_paths.extend(removed_note_paths)
                    else:
                         console.log(f"[yellow]warning: note removal on '{basename(base_path)}' produced no valid files.[/yellow]")
                except Exception as e:
                    console.log(f"[red]error removing notes from {basename(base_path)}: {e}[/red]")
            console.log(f"{self.tag}\tgenerated {len(all_augmented_paths)} total versions including removals.")
        else:
            all_augmented_paths = base_midi_paths # Use only base paths if no removal

        # --- Step 3: Find Best Augmentation Based on Similarity --- 
        best_aug_path = ""
        best_match_path = ""
        best_similarity = -1.0 # Initialize to allow any valid similarity

        # Deduplicate paths before embedding calculation
        unique_augmented_paths = sorted(list(set(all_augmented_paths)))
        console.log(f"{self.tag}\tevaluating {len(unique_augmented_paths)} unique augmentations...")

        augmentation_scores = [] # Store (similarity, aug_path, match_path)

        for aug_path in unique_augmented_paths:
            if not os.path.exists(aug_path):
                # console.log(f"[yellow]warning: skipping non-existent augmentation path '{aug_path}'[/yellow]")
                continue
            try:
                embedding = self.staff.seeker.get_embedding(aug_path) # Using default model
                # Check for empty embeddings (e.g., file had no notes after processing)
                if embedding is None or embedding.sum() == 0:
                    # console.log(f"{self.tag}\t\t'{basename(aug_path)}' has no notes or failed embedding, skipping.")
                    continue

                match_path_stem, similarity = self.staff.seeker.get_match(embedding)
                # Ensure get_match returns a valid path stem or handles errors
                if match_path_stem is None:
                    console.log(f"[yellow]warning: could not find match for '{basename(aug_path)}'[/yellow]")
                    continue

                match_abs_path = os.path.join(self.args.dataset_path, match_path_stem + ".mid")
                augmentation_scores.append((similarity, aug_path, match_abs_path))

                # console.log(f"{self.tag}\t\t'{basename(aug_path)}' -> '{basename(match_abs_path)}' (sim: {similarity:.4f})")

            except FileNotFoundError:
                 console.log(f"[yellow]warning: file not found during embedding/matching for '{aug_path}'. skipping.[/yellow]")
            except Exception as e:
                console.log(f"[red]error processing augmentation '{basename(aug_path)}': {e}[/red]")
                # Decide if we should continue or stop
                continue

        if not augmentation_scores:
            console.log("[red]error: no valid augmentations could be evaluated.[/red]")
            return [pf_midi]

        # Sort by similarity descending
        augmentation_scores.sort(key=lambda x: x[0], reverse=True)

        best_similarity, best_aug_path, best_match_path = augmentation_scores[0]

        console.log(
            f"{self.tag}\tbest augmentation is '{basename(best_aug_path)}' (sim: {best_similarity:.4f} to '{basename(best_match_path)}')"
        )

        # --- Step 4: Construct Final Playlist --- 
        # Return the sequence starting from the best augmentation, possibly including its
        # intermediate steps if removal was applied, and finally the best match.

        final_playlist = []
        # Try to find the generation lineage of the best augmentation
        best_aug_basename = os.path.basename(best_aug_path).split('.')[0]
        parts = best_aug_basename.split('_')

        # Check if it came from a removal step (e.g., _v1-R05)
        if len(parts) > 1 and parts[-1].startswith(('v','R','L','S')):
            base_name_parts = parts[:-1]
            version_info = parts[-1]
            # Try to find the original base file (rearranged or original seed)
            potential_base_name = "_".join(base_name_parts) + ".mid"
            potential_base_path = os.path.join(pf_augmentations_dir, potential_base_name)

            # Find all related removal files for this base and version
            related_files = []
            if os.path.exists(potential_base_path):
                # Add the base rearrangement/original first
                final_playlist.append(potential_base_path)
                # Find files matching the pattern
                version_prefix = version_info.split('-')[0] # e.g., v1
                method_char = version_info[len(version_prefix)+1] # R, L, S
                pattern = f"{"_".join(base_name_parts)}_{version_prefix}-{method_char}*.mid"
                console.log(f"searching for pattern: {pattern}")
                import glob
                matches = glob.glob(os.path.join(pf_augmentations_dir, pattern))
                # Extract step number and sort
                sorted_matches = sorted(
                    matches,
                    key=lambda p: int(os.path.basename(p).split('-')[-1].split('.')[0][1:]) # Extract number after R/L/S
                 )
                final_playlist.extend(sorted_matches)
                # Ensure the best one is the last one from the sequence
                if final_playlist[-1] != best_aug_path:
                     console.log(f"[yellow]warning: best augmentation path {basename(best_aug_path)} not found at end of inferred sequence. using best path directly.[/yellow]")
                     final_playlist = [best_aug_path] # Fallback

            else: # Base not found, just use the best directly
                 final_playlist = [best_aug_path]
        else: # Best was likely an original or rearranged file without removal
            final_playlist = [best_aug_path]

        # Add the best matching file from the dataset
        if best_match_path and os.path.exists(best_match_path):
            final_playlist.append(best_match_path)
        else:
            console.log(f"[yellow]warning: best match path '{best_match_path}' is invalid or missing. skipping.[/yellow]")

        # Deduplicate final list while preserving order (in case base == best_aug_path)
        seen = set()
        final_playlist_unique = [p for p in final_playlist if not (p in seen or seen.add(p))]

        console.log(f"{self.tag} final augmentation sequence: {[basename(p) for p in final_playlist_unique]}")

        return final_playlist_unique

    def _emit_current_file(self):
        current_status = self.staff.scheduler.get_current_file()
        if current_status is not None and self.playing_file != current_status[0]:
            self.playing_file = current_status[0]
            console.log(f"{self.tag} now playing '{self.playing_file}'")
            self.s_status.emit(f"now playing '{self.playing_file}'")
            num_files_remaining = self.params.n_transitions - current_status[1]
            self.s_segments_remaining.emit(num_files_remaining)
