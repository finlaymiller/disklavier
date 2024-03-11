import os
import mido
from queue import Queue
from threading import Thread, Event
from datetime import datetime

from player.player import Player
from listener.listener import Listener
from seeker.seeker import Seeker

from utils import console
from utils.midi import stretch_midi_file, plot_piano_roll_and_pitch_histogram


class Overseer:
    p = "[green]ovrsee[/green]:"
    playing_file = ""

    def __init__(
        self,
        params,
        args,
        record_dir: str,
        plot_dir: str,
        tempo: int,
    ):
        console.log(f"{self.p} initializing")

        self.params = params
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.record_dir = record_dir
        self.plot_dir = plot_dir
        self.tempo = tempo
        self.kickstart = args.kickstart
        self.params.listener.tempo = self.tempo
        self.params.player.tempo = self.tempo

        self._init_midi()  # make sure MIDI port is available first
        if len(os.listdir(self.data_dir)) < 10:
            console.log(
                f"{self.p} [red]less than 10 files in input folder. are you sure you didnt screw something up?"
            )

        # set up events & queues for threading
        self.kill_event = Event()
        self.give_next_event = Event()
        self.recording_ready_event = Event()
        self.playlist_queue = Queue()
        self.progress_queue = Queue()

        # initialize objects to be overseen
        self.seeker = Seeker(
            self.params.seeker, self.data_dir, self.output_dir, args.force_rebuild
        )
        self.seeker.build_properties()
        self.seeker.build_similarity_table()
        self.listener = Listener(
            self.params.listener,
            self.record_dir,
            self.recording_ready_event,
            self.kill_event,
        )
        self.player = Player(
            self.params.player,
            self.record_dir,
            args.tick,
            self.kill_event,
            self.give_next_event,
            self.playlist_queue,
            self.progress_queue,
        )

    def start(self) -> None:
        """run the system"""
        if not self.input_port or not self.output_port:
            return

        # kickstart process
        if self.kickstart:
            self.recording_ready_event.set()
            random_file_path = os.listdir(self.data_dir)[0]
            self.listener.outfile = random_file_path
        # start listening for recording
        else:
            listen_thread = Thread(
                target=self.listener.listen, args=(), name="listener"
            )
            listen_thread.start()

        try:
            while True:
                # check for recordings
                if self.recording_ready_event.is_set():
                    # get recording
                    if self.kickstart:
                        recording_path = os.path.join(
                            self.data_dir, self.listener.outfile
                        )
                    else:
                        recording_path = os.path.join(
                            self.record_dir, self.listener.outfile
                        )
                    console.log(
                        f"{self.p} triggering playback from recording '{recording_path}'"
                    )

                    # get most similar file to recording
                    recorded_ph = self.seeker.midi_to_ph(recording_path)
                    first_link = self.seeker.find_most_similar_vector(recorded_ph)
                    next_file_path = os.path.join(self.data_dir, str(first_link[0]))
                    next_file_path = self.change_tempo(next_file_path)

                    # save plots of both PHs
                    plot_dir = f"{datetime.now().strftime('%y%m%d-%H%M%S')}"
                    plot_path = os.path.join(self.output_dir, "plots", plot_dir)

                    if os.path.exists(plot_path):
                        plot_path += "_2"
                    os.mkdir(plot_path)
                    plot_piano_roll_and_pitch_histogram(recording_path, plot_path)
                    plot_piano_roll_and_pitch_histogram(next_file_path, plot_path)

                    # start up player
                    self.playlist_queue.put((next_file_path, float(first_link[1])))
                    playback_thread = Thread(
                        target=self.player.playback_loop,
                        args=(recording_path, recorded_ph),
                        name="player",
                    )
                    playback_thread.start()

                    # clear recording
                    self.listener.outfile = ""
                    self.recording_ready_event.clear()
                    console.log(f"{self.p} finished triggering playback from recording")

                # check for next file requests from player
                if self.give_next_event.is_set():
                    # get and prep next file
                    next_file, similarity = self.seeker.get_most_similar_file(
                        os.path.basename(next_file_path)
                    )
                    next_file_path = os.path.join(self.data_dir, str(next_file))
                    next_file_path = self.change_tempo(next_file_path)
                    # console.log(f"{self.p} player is playing '{self.player.playing_file}'\t(next up is '{next_file_path}')")

                    # send next file to player
                    self.playlist_queue.put((next_file_path, similarity))
                    console.log(
                        f"{self.p} added next file '{next_file}' to queue with similarity {similarity:.03f}"
                    )

                    self.give_next_event.clear()

        except KeyboardInterrupt:  # CTRL+C caught
            # end threads
            console.log(f"{self.p} [red]CTRL + C detected, shutting down")
            self.kill_event.set()

            playback_thread.join()
            console.log(f"{self.p} player killed successfully")
            listen_thread.join()
            console.log(f"{self.p} listener killed successfully")

    def _init_midi(self) -> None:
        """initialize the MIDI system based on the connections specified in
        the provided parameters (from the param file loaded by main.py)
        """
        console.log(f"{self.p} connecting to MIDI")
        available_inputs = mido.get_input_names()  # type: ignore
        available_outputs = mido.get_output_names()  # type: ignore

        console.log(f"{self.p} found input ports: {available_inputs}")
        console.log(f"{self.p} found output ports: {available_outputs}")

        if len(available_inputs) == 0 or len(available_outputs) == 0:
            console.log(f"{self.p} no MIDI device detected")
            raise ConnectionAbortedError  # wrong error type, i know

        # set up input connection
        if self.params.in_port in available_inputs:
            self.input_port = mido.open_input(self.params.in_port)  # type: ignore
            self.params.player.in_port = self.params.in_port
            self.params.listener.in_port = self.params.in_port
        elif len(available_inputs) > 0:
            console.log(
                f"{self.p} unable to find MIDI device '{self.params.in_port}' falling back on '{available_inputs[0]}'"
            )
            self.input_port = mido.open_input(available_inputs[0])  # type: ignore
            self.params.player.in_port = available_inputs[0]
            self.params.listener.in_port = available_inputs[0]
        else:
            console.log(f"{self.p} no MIDI input devices available")

        # set up output connection
        if self.params.out_port in available_inputs:
            self.output_port = mido.open_output(self.params.out_port)  # type: ignore
            self.params.player.out_port = self.params.out_port
            self.params.listener.out_port = self.params.out_port
        elif len(available_outputs) > 0:
            console.log(
                f"{self.p} unable to find MIDI device '{self.params.out_port}' falling back on '{available_outputs[0]}'"
            )
            self.output_port = mido.open_output(available_outputs[0])  # type: ignore
            self.params.player.out_port = available_outputs[0]
            self.params.listener.out_port = available_outputs[0]
        else:
            console.log(f"{self.p} no MIDI output devices available")

    def change_tempo(self, midi_file_path: str, do_stretch: bool = True) -> str:
        """
        update midi file tempo and note timings so that it is played back at the set tempo.

        Parameters:
            midi_file_path (str): the path to the midi file
            do_stretch (bool): also change the note timings to conform to the new tempo

        Returns:
            str: the path to the new midi file
        """
        midi = mido.MidiFile(midi_file_path)
        file_bpm = int(os.path.basename(midi_file_path).split("-")[1])
        new_tempo = mido.bpm2tempo(self.tempo)
        new_message = mido.MetaMessage("set_tempo", tempo=new_tempo, time=0)
        tempo_added = False

        for track in midi.tracks:
            # remove existing set_tempo messages
            for msg in track:
                if msg.type == "set_tempo":
                    # track.remove(msg)
                    track.remove(msg)
                    console.log(f"{self.p} [red]removed set tempo message", msg)

            # add new set_tempo message to the first track
            if not tempo_added:
                # console.log(f"{self.p} adding message (tempo={self.tempo}) {new_message}")
                track.insert(0, new_message)
                tempo_added = True

        # if no tracks had a set_tempo message and no new one was added, add a new track with the tempo message
        if not tempo_added:
            new_track = mido.MidiTrack()
            console.log(
                f"{self.p} [red]adding message to new track {new_message}", markup=True
            )
            new_track.append(new_message)

        # also stretch note timings
        new_len = midi.length * file_bpm / self.tempo
        if do_stretch:
            midi = stretch_midi_file(midi, new_len, self.p)

        # save the modified MIDI file
        new_file_path = os.path.join(
            "data", "playlist", f"{os.path.basename(midi_file_path)}"
        )
        # console.log(f"{self.p} saving modified MIDI file with new tempo {self.tempo} BPM to '{new_file_path}'")
        midi.save(new_file_path)

        return new_file_path
