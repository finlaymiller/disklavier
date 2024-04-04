import os
from shutil import copy2
from pathlib import Path
from datetime import datetime
from pynput import keyboard
from queue import Queue
from threading import Thread, Event
from pretty_midi import PrettyMIDI
import mido

from player.player import Player
from listener.listener import Listener
from seeker.seeker import Seeker
from controller.controller import Controller

from utils import console
from utils.midi import augment_recording
from utils.metrics import scale_vels
from utils.plot import plot_images, plot_piano_roll_and_pitch_histogram


class Overseer:
    p = "[green]ovrsee[/green]:"
    playing_file = ""
    playlist = []
    do_loop = False

    def __init__(
        self,
        params,
        args,
        playlist_dir: str,
        record_dir: str,
        plot_dir: str,
        tempo: int,
    ):
        console.log(f"{self.p} initializing")

        self.params = params
        self.playlist_dir = playlist_dir
        self.record_dir = record_dir
        self.plot_dir = plot_dir
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.kickstart = args.kickstart
        self.v_scale = args.velocity
        self.do_plot = args.plot
        self.tempo = tempo
        self.params.player.tempo = self.tempo
        self.params.listener.tempo = self.tempo

        self._init_midi()  # make sure MIDI port is available first

        if len(os.listdir(self.data_dir)) < 10:
            console.log(
                f"{self.p} [red]less than 10 files in input folder. are you sure you didnt screw something up?"
            )

        # set up events & queues for threading
        self.reset_event = Event()
        self.kill_player_event = Event()
        self.kill_listener_event = Event()
        self.kill_controller_event = Event()
        self.give_next_event = Event()
        self.recording_ready_event = Event()
        self.playback_commmand_queue = Queue()
        self.playlist_queue = Queue()
        self.progress_queue = Queue()
        self.keypress_queue = Queue()

        # initialize objects to be overseen
        self.seeker = Seeker(
            self.params.seeker, self.data_dir, self.output_dir, args.force_rebuild
        )
        self.seeker.build_properties()
        self.seeker.build_top_n_table()

        self.listener = Listener(
            self.params.listener,
            self.record_dir,
            self.recording_ready_event,
            self.kill_listener_event,
            self.reset_event,
        )
        self.player = Player(
            self.params.player,
            self.record_dir,
            args.tick,
            self.kill_player_event,
            self.give_next_event,
            self.playlist_queue,
            self.progress_queue,
            self.playback_commmand_queue,
        )
        self.controller = Controller(self.kill_controller_event, self.keypress_queue)

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

        controller_thread = Thread(
            target=self.controller.run, args=(), name="controller"
        )
        controller_thread.start()

        playback_thread = None

        try:
            while True:
                pass
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
                    first_file, first_similarity = self.seeker.get_ms_to_recording(
                        recording_path
                    )
                    # check augments for any better matches
                    if self.params.augment_recording:
                        options = augment_recording(
                            recording_path, self.plot_dir, self.tempo
                        )

                        change = None
                        for opt in options:
                            match_path, match_sim = self.seeker.get_ms_to_recording(opt)

                            if match_sim > first_similarity:
                                recording_path = opt
                                first_similarity = match_sim
                                first_file = match_path

                                if recording_path.endswith("fh.mid"):
                                    change = "first half"
                                elif recording_path.endswith("sh.mid"):
                                    change = "second half"
                                elif recording_path.endswith("db.mid"):
                                    change = "doubled"

                        if change is not None:
                            console.log(
                                f"{self.p} using alt version of recording :: [bold deep_pink3]{change}[/bold deep_pink3] :: '{recording_path}'"
                            )

                    next_file_path = os.path.join(self.data_dir, str(first_file))
                    next_file_path = self.change_tempo(next_file_path)
                    self.playlist_queue.put((next_file_path, first_similarity))
                    self.playlist.append(next_file_path)

                    # copy the version of the recording that we use to the playlist
                    copy2(
                        recording_path,
                        os.path.join(
                            self.playlist_dir, os.path.basename(recording_path)
                        ),
                    )

                    # start up player
                    self.playlist.append(
                        os.path.join(
                            self.playlist_dir, os.path.basename(recording_path)
                        )
                    )
                    playback_thread = Thread(
                        target=self.player.playback_loop,
                        args=(recording_path, "a"),
                        name="player",
                    )
                    playback_thread.start()

                    # while not self.playlist_queue.qsize() == 0:
                    #     queued_file = self.playlist_queue.get()
                    #     console.log(f"{self.p} removed queued segment: '{queued_file}'")
                    #     self.playlist_queue.task_done()
                    # save plots of both PHs
                    # plot_dir = f"pr_ph_{datetime.now().strftime('%y%m%d-%H%M%S')}"
                    # plot_path = os.path.join(self.output_dir, "plots", plot_dir)
                    # if os.path.exists(plot_path):
                    #     plot_path += "_2"
                    # os.mkdir(plot_path)
                    # plot_piano_roll_and_pitch_histogram(recording_path, plot_path)
                    # plot_piano_roll_and_pitch_histogram(next_file_path, plot_path)

                    # clear recording
                    self.listener.outfile = ""
                    self.recording_ready_event.clear()
                    console.log(f"{self.p} finished triggering playback from recording")

                # check for next file requests from player
                if self.give_next_event.is_set():
                    # get and prep next file
                    next_file, similarity = (self.player.playing_file, 1.0)
                    if not self.do_loop:
                        next_file, similarity = self.seeker.get_msf_new(
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

                    self.playlist.append(next_file_path)
                    self.give_next_event.clear()

                if self.reset_event.is_set():
                    console.log(f"{self.p} [bold deep_pink3]RESETTING")

                    # reset player
                    while not self.playlist_queue.qsize() == 0:
                        queued_file = self.playlist_queue.get()
                        console.log(
                            f"{self.p}\tremoved queued segment: '{queued_file}'"
                        )
                        self.playlist_queue.task_done()

                    if playback_thread is not None:
                        self.kill_player_event.set()
                        console.log(f"{self.p}\twaiting for player to die")
                        playback_thread.join()
                        self.kill_player_event.clear()

                    # clear recording
                    self.listener.outfile = ""
                    self.listener.recorded_notes = []
                    self.recording_ready_event.clear()

                    self.reset_event.clear()
                    console.log(f"{self.p} reset complete")

                # check for keypresses
                while not self.keypress_queue.qsize() == 0:
                    try:
                        command = self.keypress_queue.get()
                        console.log(f"{self.p} got key command '{command}'")
                        match command:
                            case "FADE" | "MUTE" | "VOL DOWN" | "VOL UP":
                                self.playback_commmand_queue.put(command)
                            case "LOOP":
                                self.do_loop = not self.do_loop

                                self.player.next_file_path = (
                                    self.player.playing_file_path
                                )

                                while not self.playlist_queue.qsize() == 0:
                                    queued_file, sim = self.playlist_queue.get()
                                    console.log(
                                        f"{self.p}\tremoved queued segment: '{queued_file}'"
                                    )
                                    self.playlist_queue.task_done()

                                self.give_next_event.set()

                            case "BACK":
                                console.log(f"\trewinding")

                                while not self.playlist_queue.qsize() == 0:
                                    queued_file, sim = self.playlist_queue.get()
                                    console.log(
                                        f"{self.p}\tremoved queued segment: '{queued_file}'"
                                    )
                                    self.playlist_queue.task_done()

                                self.player.next_file_path = self.playlist[-1]

                                self.give_next_event.set()
                            case _:
                                console.log(
                                    f"{self.p}\tcommand unsupported '{command}'"
                                )

                        self.keypress_queue.task_done()
                    except:
                        console.log(f"{self.p} [bold orange]whoops")

        except KeyboardInterrupt:  # ctrl + c
            # end threads
            console.log(f"{self.p} [red bold]CTRL + C detected, shutting down")

            with open(os.path.join(self.playlist_dir, "playlist.txt"), "a") as f:
                for sample in self.playlist:
                    f.write(f"{sample}\n")

            self.kill_controller_event.set()
            self.kill_player_event.set()
            self.kill_listener_event.set()

            if controller_thread.is_alive():
                controller_thread.join()
                console.log(f"{self.p} controller killed successfully")

            if playback_thread is not None:
                playback_thread.join()
                console.log(f"{self.p} player killed successfully")

            if listen_thread is not None:
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
                    track.remove(msg)
                    # console.log(f"{self.p} [red]removed set tempo message", msg)

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
            midi.tracks.append(new_track)

        new_file_path = os.path.join(
            self.playlist_dir, f"{Path(midi_file_path).stem}.mid"
        )
        midi.save(new_file_path)

        # also stretch note timings
        # if do_stretch:
        #     new_len = midi.length * file_bpm / self.tempo
        #     new_midi = um.stretch_midi_file(midi, new_len, self.p)

        # save the modified MIDI file

        # console.log(f"{self.p} saving modified MIDI file with new tempo {self.tempo} BPM to '{new_file_path}'")
        # new_midi.save(new_file_path)
        new_midi = mido.MidiFile(new_file_path)
        # segment_length = 60 * 16 / file_bpm  # in seconds
        # if np.round(segment_length, 3) != np.round(midi.length, 3):
        #     total_time_t = -1
        #     for track in midi.tracks:
        #         # Remove existing 'end_of_track' messages and calculate last note time
        #         for msg in track:
        #             total_time_t += msg.time
        #             if msg.type == "end_of_track":
        #                 track.remove(msg)
        # console.log(f"{self.p} expected len {segment_length:.04f} but found {midi.length:.04f} and calcd {mido.tick2second(total_time_t, 220, mido.bpm2tempo(file_bpm))}")

        # scale velocities
        # if self.v_scale != 1.0 :
        #     new_pm = PrettyMIDI(new_file_path)
        #     os.remove(new_file_path)
        #     [[v_min_o, v_max_o], v_hist_o] = um.get_velocities(new_pm)
        #     scaled_midi = scale_vels(new_pm, self.v_scale)
        #     [[v_min_i, v_max_i], v_hist_i] = um.get_velocities(scaled_midi)
        #     console.log(
        #         f"{self.p} scaled by factor {self.v_scale} ({v_min_o}, {v_max_o}) -> ({v_min_i}, {v_max_i})\n{v_hist_o} -> {v_hist_i}"
        #     )
        #     scaled_midi.write(new_file_path)

        # old_path = os.path.join(self.plot_dir, os.path.basename(midi_file_path))
        if self.do_plot:
            old_pr = PrettyMIDI(midi_file_path).get_piano_roll()
            new_pr = PrettyMIDI(new_file_path).get_piano_roll()
            plot_path = os.path.join(
                self.plot_dir, f"{os.path.basename(midi_file_path)[-4:]}.png"
            )

            plot_images(
                [old_pr, new_pr],
                [
                    f"{os.path.basename(midi_file_path)} ({midi.length:.02f}s)",
                    f"{os.path.basename(new_file_path)} ({new_midi.length:.02f}s)",
                ],
                plot_path,
                (2, 1),
                set_axis="on",
            )

        return new_file_path
