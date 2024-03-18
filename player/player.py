import os
import mido
from mido import MidiFile, Message
from threading import Thread, Event
from queue import Queue

from utils import console, tick


class Player:
    p = "[blue]player[/blue]:"
    playing_file_path = ""
    next_file_path = ""
    volume = 1.0  # volume scaling factor
    last_volume = 1.0  # memory

    def __init__(
        self,
        params,
        record_dir: str,
        do_tick: bool,
        kill_event: Event,
        playback_event: Event,
        filename_queue: Queue,
        progress_queue: Queue,
        command_queue: Queue,
    ) -> None:
        self.params = params
        self.record_dir = record_dir
        self.do_tick = do_tick
        self.kill_event = kill_event
        self.get_next = playback_event
        self.file_queue = filename_queue
        self.playback_progress = progress_queue
        self.commands = command_queue
        self.out_port = mido.open_output(self.params.out_port)  # type: ignore

    def playback_loop(self, seed_file_path: str, fh):
        """"""
        self.playing_file_path = seed_file_path
        (self.next_file_path, similarity) = self.file_queue.get()
        next_file = os.path.basename(self.next_file_path)
        first_loop = True

        while self.next_file_path is not None and not self.kill_event.is_set():
            self.playing_file = os.path.basename(self.playing_file_path)
            file_tempo = int(os.path.basename(self.playing_file).split("-")[1])
            found_tempo = -1
            for track in MidiFile(self.playing_file_path).tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        found_tempo = msg.tempo

            if not first_loop:  # bad code!
                self.next_file_path, similarity = self.file_queue.get()
                next_file = os.path.basename(self.next_file_path)
            self.get_next.set()

            console.log(
                f"{self.p} playing '{self.playing_file}' ({file_tempo}) -- {similarity:.3f} --> '{next_file}' ({round(mido.tempo2bpm(found_tempo)):01d})"
            )

            self.play_midi_file(self.playing_file_path)
            self.playing_file_path = self.next_file_path

            first_loop = False

        console.log(f"{self.p} [orange]shutting down")

    def play_midi_file(self, midi_path: str) -> None:
        """"""
        midi = MidiFile(midi_path)
        found_tempo = -1
        runtime = 0
        printed_msg = False
        do_fade = False

        for track in midi.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    found_tempo = msg.tempo

        if self.do_tick:
            stop_tick = Event()
            tick_thread = Thread(
                target=tick,
                args=(int(mido.tempo2bpm(found_tempo)), stop_tick, self.p, False),
            )
        for msg in midi.play():
            if hasattr(msg, "time"):
                runtime += msg.time  # type: ignore

            # check for keypresses
            while not self.commands.qsize() == 0:
                try:
                    command = self.commands.get()
                    console.log(f"{self.p} got key command '{command}'")
                    match command:
                        case "FADE":
                            self.last_volume = self.volume
                            do_fade = not do_fade
                        case "MUTE":
                            if self.volume == 0.0:
                                self.volume = self.last_volume
                                self.last_volume = 0.0
                            else:
                                self.last_volume = self.volume
                                self.volume = 0.0
                        case "VOL DOWN":
                            self.last_volume = self.volume
                            self.volume -= 0.1
                        case "VOL UP":
                            self.last_volume = self.volume
                            self.volume += 0.1
                        case _:
                            pass

                    console.log(
                        f"{self.p} volume change {self.last_volume:.02f} -> {self.volume:.02f}"
                    )

                    self.commands.task_done()
                except:
                    console.log(f"{self.p} [bold orange]whoops[/bold orange]")

            if self.do_tick and not tick_thread.is_alive():
                tick_thread.start()

            # handle volume changes
            scaled_msg = msg
            if hasattr(msg, "velocity"):
                if do_fade:
                    self.last_volume = self.volume
                    self.volume = self.fade(self.volume, runtime, midi.length)

                scaled_msg.velocity = round(scaled_msg.velocity * self.volume)  # type: ignore

            self.out_port.send(scaled_msg)

            self.playback_progress.put(msg.time)  # type: ignore

            if self.kill_event.is_set() and not printed_msg:
                self.end_notes()
                self.volume = 1.0
                self.last_volume = 1.0
                break
                # console.log(f"{self.p} [yellow]finishing playback of the current file")
                # do_fade = True
                # printed_msg = True

        if self.do_tick:
            stop_tick.set()
            tick_thread.join()

        # do_fade = False

    def fade(self, current_value, current_time, end_time):
        """
        Scales the current value to zero by the end time.

        Parameters:
        - current_value: The current value to scale.
        - current_time: The current time in ticks
        - end_time: The end time by which the value should reach 0.

        Returns:
        - The scaled value, which linearly decreases to 0 by the end time. If the current time is past the end time, it returns 0.
        """

        scaling_factor = 1 - current_time / (end_time * 2.0)
        scaled_value = current_value * scaling_factor

        return scaled_value

    def end_notes(self):
        for note in range(128):  # MIDI notes range from 0 to 127
            msg = Message("note_off", note=note, velocity=0, channel=0)
            self.out_port.send(msg)
