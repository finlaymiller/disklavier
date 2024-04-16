import os
import mido
from mido import MidiFile, MidiTrack, Message
import time
from datetime import datetime
from threading import Thread, Event

from utils import console, tick
from utils.midi import stretch_midi_file


class Listener:
    p: str = "[magenta]listen[/magenta]:"
    is_recording: bool = False  # always either recording or listening
    recorded_notes = []
    outfile: str = ""

    def __init__(
        self, params, record_dir: str, rec_event: Event, kill_event: Event, reset_event: Event
    ) -> None:
        self.params = params
        self.record_dir = record_dir
        self.ready_event = rec_event
        self.reset_event = reset_event
        self.kill_event = kill_event

    def listen(self) -> None:
        """"""
        start_time = time.time()
        end_time = 0
        last_note_time = start_time

        with mido.open_input(self.params.in_port) as inport:  # type: ignore
            console.log(f"{self.p} listening on port '{self.params.in_port}'")
            for msg in inport:
                # record delta time of input message
                # mido doesn't do this by default for some reason
                current_time = time.time()
                msg.time = int((current_time - last_note_time) * 480)
                console.log(f"{self.p} \t{msg}")
                last_note_time = current_time

                # record pedal signal
                if msg.type == "control_change" and msg.control == self.params.record:
                    # record pedal released
                    if msg.value == 0:
                        end_time = time.time()
                        console.log(
                            f"{self.p} recorded {end_time - start_time:.02f} s"
                        )
                        self.is_recording = False

                        self.stop_tick_event.set()
                        self.metro_thread.join()

                        if len(self.recorded_notes) > 0:
                            # save file and notify overseer that its ready
                            self.save_midi(end_time - start_time)
                            self.ready_event.set()
                        else:
                            console.log(f"{self.p} no notes recorded")

                    # record pedal not released, but not already recording
                    elif self.is_recording == False:
                        console.log(f"{self.p} recording at {self.params.tempo} BPM")
                        self.reset_event.set()
                        self.is_recording = True

                        self.stop_tick_event = Event()
                        self.metro_thread = Thread(
                            target=tick,
                            args=(self.params.tempo, self.stop_tick_event),
                            name="player",
                        )
                        self.metro_thread.start()

                # record note on/off
                elif self.is_recording and msg.type in ["note_on", "note_off"]:
                    if len(self.recorded_notes) == 0:
                        # set times to start from now
                        start_time = time.time()
                        msg.time = 0
                    self.recorded_notes.append(msg)

                # die
                if self.kill_event.is_set():
                    console.log(f"{self.p} [orange]shutting down")
                    self.stop_tick_event.set()
                    self.metro_thread.join(1)
                    return

    def save_midi(self, dt) -> None:
        """Saves the recorded notes to a MIDI file."""
        self.outfile = f"recording-{self.params.tempo:03d}-{datetime.now().strftime('%y%m%d_%H%M%S')}.mid"
        console.log(f"{self.p} saving recording '{self.outfile}'")

        mid = MidiFile()
        track = MidiTrack()
        track.insert(
            0,
            mido.MetaMessage(
                type="set_tempo",
                tempo=mido.bpm2tempo(self.params.tempo),
                time=0,
            ),
        )
        track.append(Message("program_change", program=12))  # dunno what this does tbh
        for msg in self.recorded_notes:
            track.append(msg)
        mid.tracks.append(track)

        # mid = stretch_midi_file(mid, dt, caller=self.p)
        mid.save(os.path.join(self.record_dir, self.outfile))

        if os.path.exists(os.path.join(self.record_dir, self.outfile)):
            console.log(f"{self.p} successfully saved recording '{self.outfile}'")
        else:
            console.log(f"{self.p} failed to save recording '{self.outfile}'")
