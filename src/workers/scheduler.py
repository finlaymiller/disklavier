import os
import mido
from queue import PriorityQueue
from datetime import datetime, timedelta

from .worker import Worker
from utils import basename, console
from utils.midi import csv_to_midi
from utils.constants import TICKS_PER_BEAT

from typing import Optional, Tuple, List


class Scheduler(Worker):
    lead_bar: bool = True
    tt_offset: int = 0
    ts_transitions: List[Tuple[float, bool]] = []
    tt_all_messages: list[int] = []
    n_files_queued: int = 0
    n_beats_per_segment: int = 8
    queued_files: list[str] = []
    first_file_avg_velocity: Optional[float] = None
    previous_track_name: Optional[str] = None

    def __init__(
        self,
        params,
        bpm: int,
        log_path: str,
        playlist_path: str,
        start_time: datetime,
        n_transitions: int,
        recording_mode: bool,
    ):
        super().__init__(params, bpm=bpm)
        self.lead_bar = params.lead_bar
        self.n_beats_per_segment = params.n_beats_per_segment
        self.pf_log = log_path
        self.p_playlist = playlist_path
        self.td_start = start_time
        self.n_transitions = n_transitions
        self.recording_mode = recording_mode
        self.tt_all_messages = []
        self.ts_transitions = []
        self.queued_files = []
        self.previous_track_name = None

        # initialize queue file
        self.raw_notes_filepath = os.path.join(self.pf_log, "queue_dump.csv")
        self.raw_notes_file = open(self.raw_notes_filepath, "w")
        self.raw_notes_file.write("file,type,note,velocity,time\n")

        console.log(f"{self.tag} initialization complete")
        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def enqueue_midi(
        self,
        pf_midi: str,
        q_piano: PriorityQueue,
        q_gui: Optional[PriorityQueue] = None,
        similarity: Optional[float] = None,
    ) -> Tuple[float, float, int]:
        midi_in = mido.MidiFile(pf_midi)

        # --- offset and track transition determination ---
        current_track_name = basename(pf_midi).split("_")[0]
        is_new_track_segment = False
        if self.n_files_queued == 0:
            if self.recording_mode and "player" in basename(pf_midi):
                n_recorded_beats = None
                for msg in midi_in.tracks[1]:
                    if msg.type == "text" and msg.text.startswith("beat"):
                        n_recorded_beats = int(msg.text.split(" ")[1]) - 1
                if n_recorded_beats is None:
                    ts_offset, tt_offset = 0, 0
                else:
                    ts_offset = self.n_beats_per_segment * 60 / self.bpm - (
                        n_recorded_beats * 60 / self.bpm
                    )
                    tt_offset = mido.second2tick(ts_offset, TICKS_PER_BEAT, self.tempo)
            else:
                ts_offset, tt_offset = 0, 0
            is_new_track_segment = False
        else:
            ts_offset, tt_offset = self._get_next_transition()
            if (
                self.previous_track_name is not None
                and current_track_name != self.previous_track_name
            ):
                is_new_track_segment = True

        self.ts_transitions.append((ts_offset, is_new_track_segment))
        self.previous_track_name = current_track_name

        tt_abs: int = tt_offset
        tt_sum: int = 0
        tt_max_abs_in_segment: int = tt_offset

        if midi_in.ticks_per_beat != TICKS_PER_BEAT:
            raise ValueError(
                f"{self.tag}[red] midi file ticks per beat mismatch!\n\tfile has {midi_in.ticks_per_beat} tpb but expected {TICKS_PER_BEAT}"
            )

        # --- velocity normalization ---
        if self.params.scale_velocity:
            note_on_messages = []
            for track in midi_in.tracks:
                if track[0].type == "track_name" and track[0].name == "metronome":
                    continue
                for msg in track:
                    if msg.type == "note_on" and msg.velocity > 0:
                        note_on_messages.append(msg)

            if note_on_messages:
                velocities = [m.velocity for m in note_on_messages]
                current_avg_velocity = (
                    sum(velocities) / len(velocities) if velocities else 0.0
                )

                if self.first_file_avg_velocity is None:
                    if current_avg_velocity > 0:
                        self.first_file_avg_velocity = current_avg_velocity
                        console.log(
                            f"{self.tag} first file enqueued, average velocity set to {self.first_file_avg_velocity:.2f}"
                        )
                    else:
                        console.log(
                            f"{self.tag} [yellow]first file enqueued, but no valid notes with positive velocity to set average velocity.[/yellow]"
                        )
                elif self.first_file_avg_velocity is not None:
                    console.log(
                        f"{self.tag} normalizing to target average velocity: {self.first_file_avg_velocity:.2f}. Original avg: {current_avg_velocity:.2f}"
                    )

                    # 1. clamp velocity to below max
                    for msg in note_on_messages:
                        if msg.velocity > self.params.max_velocity:
                            console.log(
                                f"{self.tag}\t[grey50]clamping velocity: {msg.velocity} -> {self.params.max_velocity}[/grey50]"
                            )
                            msg.velocity = self.params.max_velocity

                    velocities_after_clamp = [m.velocity for m in note_on_messages]
                    avg_velocity_after_clamp = (
                        sum(velocities_after_clamp) / len(velocities_after_clamp)
                        if velocities_after_clamp
                        else 0.0
                    )

                    # 2. rescale velocity
                    scale_factor = (
                        self.first_file_avg_velocity / avg_velocity_after_clamp
                    )
                    console.log(f"{self.tag} scaling velocity by {scale_factor:.2f}")
                    for msg in note_on_messages:
                        new_velocity = int(round(msg.velocity * scale_factor))
                        msg.velocity = max(
                            0, min(self.params.max_velocity, new_velocity)
                        )

                    final_velocities = [m.velocity for m in note_on_messages]
                    final_avg_vel = sum(final_velocities) / len(final_velocities)
                    console.log(
                        f"{self.tag} velocities normalized. avg after clamp: {avg_velocity_after_clamp:.2f}, final avg: {final_avg_vel:.2f}"
                    )
            else:
                console.log(
                    f"{self.tag} no note_on messages found in {basename(pf_midi)} for velocity normalization."
                )

        console.log(
            f"{self.tag} adding file {self.n_files_queued} to queue '{pf_midi}' with offset {tt_offset} ({ts_offset:.02f} s -> {self.td_start + timedelta(seconds=ts_offset):%H:%M:%S.%f})"
        )

        # --- add messages to queue(s) ---
        # add messages to queue first so that the player has access ASAP
        for track in midi_in.tracks:
            if track[0].type == "track_name":
                if track[0].name == "metronome":
                    console.log(f"{self.tag} skipping metronome track")
                    continue
            for msg in track:
                if msg.type == "note_on" or msg.type == "note_off":
                    tt_abs += msg.time
                    tt_sum += msg.time
                    # occasionally need to shift the message to avoid priority collisions
                    current_tt_abs = tt_abs  # store original intended time
                    if current_tt_abs in self.tt_all_messages:
                        # find the nearest integer that doesn't exist in tt_all_messages
                        tt_lower_bound = current_tt_abs - 1
                        tt_upper_bound = current_tt_abs + 1
                        while (
                            tt_lower_bound in self.tt_all_messages
                            or tt_upper_bound in self.tt_all_messages
                        ):
                            tt_lower_bound -= 1
                            tt_upper_bound += 1

                        # select the nearest available integer
                        if tt_lower_bound not in self.tt_all_messages:
                            current_tt_abs = tt_lower_bound
                        else:
                            current_tt_abs = tt_upper_bound
                    self.tt_all_messages.append(current_tt_abs)
                    tt_max_abs_in_segment = max(
                        tt_max_abs_in_segment, current_tt_abs
                    )  # update max tick time

                    # console.log(f"{self.tag} adding message to queue: ({current_tt_abs}, ({msg}))")

                    q_piano.put((current_tt_abs, msg))

                    # --- add to gui queue ---
                    if q_gui is not None:
                        # TODO: make this 10 seconds a global parameter
                        tt_delay = mido.second2tick(10, TICKS_PER_BEAT, self.tempo)
                        q_gui.put(
                            (
                                current_tt_abs - tt_delay,
                                (similarity if similarity is not None else 1.0, msg),
                            )
                        )

                    # --- write to raw notes file ---
                    # edge case, but it does happen sometimes that multiple recorded notes start at 0, resulting in one note getting bumped to time -1
                    if current_tt_abs < 0:  # Ensure absolute tick time is not negative
                        # This adjustment should be relative to the segment's start tick (tt_offset)
                        # The file writing expects absolute time from system start.
                        # If msg.time was 0, and tt_abs was tt_offset, then current_tt_abs is tt_offset.
                        # If a collision made current_tt_abs less than tt_offset (unlikely here, but generally),
                        # or if tt_offset itself was negative (which we try to avoid), this needs care.
                        # For now, simply ensuring it's not globally negative if it refers to global time.
                        # If current_tt_abs is a global time, then simply:
                        if current_tt_abs < 0:
                            current_tt_abs = 0

                    self.raw_notes_file.write(
                        f"{os.path.basename(pf_midi)},{msg.type},{msg.note},{msg.velocity},{current_tt_abs}\n"
                    )

        # --- update trackers ---
        # The following block about _gen_transitions seems to be for creating MIDI event markers,
        # not for managing self.ts_transitions which is now handled above.
        # If tt_abs (end tick of last note in current file) is beyond the last recorded transition start time,
        # it might indicate something about scheduling, but self.ts_transitions is already updated for current file.
        # Original condition: mido.tick2second(tt_abs, TICKS_PER_BEAT, self.tempo) > self.ts_transitions[-1][0]
        # This check might be for ensuring the schedule is long enough or for some other purpose.
        # For now, let's assume the primary role of _gen_transitions is not to populate self.ts_transitions.
        # if self.ts_transitions and mido.tick2second(tt_max_abs_in_segment, TICKS_PER_BEAT, self.tempo) > self.ts_transitions[-1][0]:
        # _ = self._gen_transitions(self.ts_transitions[-1][0]) # Original intent was perhaps to create MIDI markers

        self.n_files_queued += 1
        self.queued_files.append(basename(pf_midi))
        ts_seg_len = mido.tick2second(tt_sum, TICKS_PER_BEAT, self.tempo)

        console.log(
            f"{self.tag} added {ts_seg_len:.03f} seconds of music to queue ({self.n_files_queued} files in queue)"
        )

        if self._copy_midi(pf_midi):
            console.log(f"{self.tag} copied {basename(pf_midi)} to playlist folder")

        return ts_seg_len, ts_offset, tt_max_abs_in_segment

    def init_schedule(self, pf_midi: str, offset_s: float = 0):
        """Initialize a MIDI file to hold a playback recording."""
        if self.verbose:
            console.log(f"{self.tag} initializing output file with offset {offset_s} s")
        midi = mido.MidiFile()
        tick_track = mido.MidiTrack()

        # default timing messages
        tick_track.append(
            mido.MetaMessage("track_name", name=basename(pf_midi), time=0)
        )
        tick_track.append(
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=36,
                notated_32nd_notes_per_beat=8,
                time=0,
            )
        )
        tick_track.append(mido.MetaMessage("set_tempo", tempo=self.tempo, time=0))

        # transition messages
        mm_transitions = self._gen_transitions(
            ts_offset=offset_s, n_stamps=self.n_transitions, is_new_track_segment=False
        )
        for mm_transition in mm_transitions:
            tick_track.append(mm_transition)
        tick_track.append(mido.MetaMessage("end_of_track", time=1))

        midi.tracks.append(tick_track)

        # write to file
        midi.save(pf_midi)

    def _gen_transitions(
        self,
        ts_offset: float = 0,
        n_stamps: int = 1,
        do_ticks: bool = True,
        is_new_track_segment: bool = False,
    ) -> list[mido.MetaMessage]:
        ts_offset = 0
        self.tt_offset = mido.second2tick(ts_offset, TICKS_PER_BEAT, self.tempo)
        ts_beat_length = 60 / self.bpm  # time duration of each beat
        ts_interval = self.n_beats_per_segment * ts_beat_length

        # adjust ts_offset to the next interval
        # if ts_offset % ts_interval < ts_beat_length * N_BEATS_TRANSITION_OFFSET:
        #     if self.verbose:
        #         console.log(
        #             f"{self.tag} adjusting ts_offset from {ts_offset:.02f} s to {((ts_offset // ts_interval) + 1) * ts_interval:.02f} s"
        #         )
        #     ts_offset = ((ts_offset // ts_interval) + 1) * ts_interval
        self.ts_transitions.extend(
            [
                (ts_offset + i * ts_interval, is_new_track_segment)
                for i in range(n_stamps + 1)
            ]
        )

        if self.verbose:
            console.log(
                f"{self.tag} segment interval is {ts_interval:.03f} seconds",
                # [
                #     f"{t:02.01f}  -> {self.td_start + timedelta(seconds=t):%H:%M:%S.%f}"
                #     for t in self.ts_transitions[:-5]
                # ],
            )

        transitions = []
        for i, (ts_transition, is_new_track_segment) in enumerate(self.ts_transitions):
            # transition messages
            transitions.append(
                mido.MetaMessage(
                    "text",
                    text=f"transition {i+1} ({ts_transition:.02f}s)",
                    time=mido.second2tick(ts_transition, TICKS_PER_BEAT, self.tempo),
                )
            )
            # tick messages
            if do_ticks:
                transitions[-1].time = 0  # transition occurs at tick time
                for beat in range(self.n_beats_per_segment):
                    tick_time = ts_transition + (beat * ts_beat_length)
                    transitions.append(
                        mido.MetaMessage(
                            "text",
                            text=f"tick {i}-{beat + 1} ({tick_time:.02f}s)",
                            time=mido.second2tick(
                                ts_beat_length, TICKS_PER_BEAT, self.tempo
                            ),
                        )
                    )

        return transitions

    def _get_next_transition(self) -> Tuple[float, int]:
        # this method determines the start time (ts_offset) for the NEXT segment to be queued.
        ts_offset = 0

        if not self.ts_transitions:
            # this means no segments have been scheduled yet (e.g. init_schedule not called before first enqueue_midi)
            # which is handled by n_files_queued == 0 logic in enqueue_midi.
            # if this is called when ts_transitions is truly empty, next segment starts at 0.
            console.log(
                f"{self.tag} [yellow]warning: _get_next_transition called with empty ts_transitions. Defaulting next segment start to 0s.[/yellow]"
            )
            ts_offset = 0
        else:
            # the next segment should start after the last one.
            # ts_transitions[-1][0] is the start time of the *last scheduled segment*.
            # we need its end time, or use a fixed progression.
            last_segment_start_time_s = self.ts_transitions[-1][0]

            # estimate duration of the last segment.
            # a more robust way would be to get actual duration of self.queued_files[-1] if available.
            # for now, use n_beats_per_segment as an estimate for progression.
            # this logic might need to be more sophisticated if segments have variable lengths
            # and _get_next_transition is meant to find the precise end of the previous + any lead bar.
            estimated_duration_last_segment_s = self.n_beats_per_segment * (
                60 / self.bpm
            )

            ts_offset = last_segment_start_time_s + estimated_duration_last_segment_s

            # original code had complex logic involving self.n_files_queued to index ts_transitions
            # and also called _gen_transitions which itself appended to ts_transitions.
            # the new logic is: ts_transitions is a direct log of (start_time, new_track_flag)
            # added by enqueue_midi. _get_next_transition just calculates the start for the *next* one.

        if (
            self.lead_bar and self.n_files_queued > 0
        ):  # Apply lead bar only after the first file
            # This lead_bar logic might be tricky. If ts_offset is the *actual* start including lead-in,
            # it should be calculated carefully. If ts_offset is content start and lead-in is separate,
            # that's different. Assuming ts_offset is the point notes *start playing*.
            # The original _get_next_transition had:
            # ts_offset = self.ts_transitions[self.n_files_queued]
            # if self.lead_bar: ts_offset -= 60 / self.bpm
            # This implied ts_transitions stored content start times *after* potential lead bars.
            # For now, let's keep it simpler: ts_offset is the calculated start of the next content.
            # Lead bar application would be part of how this ts_offset is derived or used.
            # The current calculation of ts_offset = last_start + duration is a content-to-content progression.
            pass

        tt_offset = mido.second2tick(ts_offset, TICKS_PER_BEAT, self.tempo)
        return ts_offset, tt_offset

    def _copy_midi(self, pf_midi: str) -> bool:
        """Copy the MIDI file to the playlist folder.

        Parameters
        ----------
        pf_midi : str
            Path to the MIDI file to copy.

        Returns
        -------
        bool
            True if the MIDI file was copied successfully, False otherwise.
        """
        midi = mido.MidiFile(pf_midi)
        out_path = os.path.join(self.p_playlist, os.path.basename(pf_midi))
        midi.save(out_path)

        return os.path.isfile(out_path)

    def queue_to_midi(self, out_path: str) -> bool:
        return csv_to_midi(
            self.raw_notes_filepath,
            out_path,
            verbose=self.verbose,
        )

    def get_current_file(self) -> Optional[tuple[str, int]]:
        if not self.ts_transitions or not self.queued_files:
            return None

        elapsed_seconds = (datetime.now() - self.td_start).total_seconds()

        # we haven't started playing yet or ts_transitions is unexpectedly empty
        if not self.ts_transitions or elapsed_seconds < self.ts_transitions[0][0]:
            return None

        # find which segment we're in
        current_segment_idx = -1
        for i in range(len(self.ts_transitions)):
            segment_start_time = self.ts_transitions[i][0]

            # determine end time of this segment
            # if it's the last segment, it extends "indefinitely" for now or until a known total duration
            next_segment_start_time = float("inf")
            if i + 1 < len(self.ts_transitions):
                next_segment_start_time = self.ts_transitions[i + 1][0]

            if segment_start_time <= elapsed_seconds < next_segment_start_time:
                current_segment_idx = i
                break

        # if loop finishes and current_segment_idx is still -1, means we are in or after the last defined segment
        if current_segment_idx == -1:
            # This can happen if elapsed_seconds is >= self.ts_transitions[-1][0]
            # Ensure that the index is valid for queued_files
            if elapsed_seconds >= self.ts_transitions[-1][0] and self.queued_files:
                current_segment_idx = (
                    len(self.ts_transitions) - 1
                )  # Assuming last transition maps to last queued file
            else:  # Should not happen if logic is correct and lists are synced
                return None

        if 0 <= current_segment_idx < len(self.queued_files):
            return self.queued_files[current_segment_idx], current_segment_idx
        else:
            # Mismatch between length of ts_transitions and queued_files or bad index
            # console.log(f"[scheduler] warning: current_segment_idx {current_segment_idx} out of range. ts_transitions: {len(self.ts_transitions)}, queued_files: {len(self.queued_files)}")
            if (
                self.queued_files
            ):  # Fallback: return last known playing file if possible
                # This fallback might be problematic if ts_transitions is longer than queued_files
                # and current_segment_idx refers to a projected, not yet queued, file.
                # For safety, ensure index is valid for queued_files.
                safe_idx = min(current_segment_idx, len(self.queued_files) - 1)
                if safe_idx >= 0:
                    return self.queued_files[safe_idx], safe_idx
            return None

    def set_start_time(self, td_start: datetime):
        self.td_start = td_start
        console.log(f"{self.tag} start time set to {self.td_start}")
