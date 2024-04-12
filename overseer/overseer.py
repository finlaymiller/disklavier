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
from player.metronome import Metronome
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

        # folders
        self.playlist_dir = playlist_dir  # re-tempo'd files
        self.record_dir = record_dir  # player recordings
        self.plot_dir = plot_dir
        self.data_dir = args.data_dir  # midi file library
        # behavior
        self.params = params
        self.kickstart = args.kickstart  # pick random start file
        self.v_scale = args.velocity  # TODO: unimplemented augmentation
        self.do_plot = args.plot  # generate a bunch of plots
        # tempo
        self.tempo = tempo
        self.params.metronome.tempo = self.tempo
        self.params.player1.tempo = self.tempo
        self.params.player2.tempo = self.tempo
        self.params.listener.tempo = self.tempo

        self._init_midi()  # make sure MIDI port is available first

        if len(os.listdir(self.data_dir)) < 10:
            console.log(
                f"{self.p} [red]less than 10 files in input folder. are you sure you didnt screw something up?"
            )

        if args.tick:
            self.params.player1

        # set up events & queues for threading
        self.reset_e = Event()
        # players
        self.play_e = Event()
        self.kill_player1_e = Event()
        self.kill_player2_e = Event()
        self.give_player1_e = Event()
        self.give_player2_e = Event()
        self.playlist1_q = Queue()
        self.playlist2_q = Queue()
        self.playback_commmand_q = Queue()
        # metronome
        self.kill_metro_e = Event()
        self.ready_e = Event()
        # listener
        self.kill_listener_e = Event()
        self.recording_ready_e = Event()
        # keyboard controller
        self.kill_controller_e = Event()
        self.keypress_q = Queue()

        # initialize objects to be overseen
        self.seeker = Seeker(
            self.params.seeker, self.data_dir, args.output_dir, args.force_rebuild
        )
        self.seeker.build_properties()
        self.seeker.build_top_n_table()

        self.listener = Listener(
            self.params.listener,
            self.record_dir,
            self.recording_ready_e,
            self.kill_listener_e,
            self.reset_e,
        )
        self.player1 = Player(
            self.params.player1,
            self.kill_player1_e,
            self.give_player1_e,
            self.play_e,
            self.playlist1_q,
            self.playback_commmand_q,
        )
        self.player2 = Player(
            self.params.player2,
            self.kill_player2_e,
            self.give_player2_e,
            self.play_e,
            self.playlist2_q,
            self.playback_commmand_q,
        )
        self.metronome = Metronome(
            self.params.metronome, self.kill_metro_e, self.ready_e, self.play_e
        )
        self.controller = Controller(self.kill_controller_e, self.keypress_q)

    def run(self) -> None:
        if not self.input_port or not self.output_port:
            console.log(f"{self.p}[red bold] input or output port unavailable, exiting")
            return

        # kickstart process
        if self.kickstart:
            self.recording_ready_e.set()
            random_file_path = os.listdir(self.data_dir)[0]
            self.listener.outfile = random_file_path
        # start listening for recording
        else:
            listen_thread = Thread(
                target=self.listener.listen, args=(), name="listener"
            )
            listen_thread.start()

        # start up other threads
        controller_thread = Thread(
            target=self.controller.run, args=(), name="controller"
        )
        controller_thread.start()
        player1_t = Thread(
            target=self.player1.play_loop,
            name="player1",
        )
        player2_t = Thread(
            target=self.player2.play_loop,
            name="player2",
        )
        player1_t.start()
        player2_t.start()
        metro_t = Thread(target=self.metronome.tick, name="metronome")

        p1_playing = False  # used to alternate between players

        try:
            while True:
                pass
                # check for recordings
                if self.recording_ready_e.is_set():
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
                            recording_path, self.record_dir, self.tempo
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

                    self.playlist.append(
                        os.path.join(
                            self.playlist_dir, os.path.basename(recording_path)
                        )
                    )
                    next_file_path = os.path.join(self.data_dir, str(first_file))
                    next_file_path = self.change_tempo(next_file_path)
                    console.log(
                        f"{self.p} queueing (ready: {self.give_player1_e.is_set()}) recording for p1: '{recording_path}'"
                    )
                    console.log(
                        f"{self.p} next file would be '{os.path.basename(next_file_path)}'"
                    )
                    self.playlist1_q.put((recording_path, -1.0))
                    self.give_player1_e.clear()
                    self.play_e.set()
                    metro_t.start()

                    # copy the version of the recording that we use to the playlist
                    copy2(
                        recording_path,
                        os.path.join(
                            self.playlist_dir, os.path.basename(recording_path)
                        ),
                    )

                    # clear recording
                    self.listener.outfile = ""
                    self.recording_ready_e.clear()
                    self.play_e.clear()
                    console.log(f"{self.p} finished triggering playback from recording")

                # metronome says get ready
                if self.ready_e.is_set():
                    # ready next file
                    next_file = (
                        self.player2.playing_file
                        if p1_playing
                        else self.player1.playing_file
                    )
                    similarity = -1.0
                    if not self.do_loop:
                        next_file, similarity = self.seeker.get_msf_new(
                            os.path.basename(next_file_path)
                        )
                    next_file_path = os.path.join(self.data_dir, str(next_file))
                    next_file_path = self.change_tempo(next_file_path)

                    # send to player
                    if p1_playing:
                        console.log(
                            f"{self.p} queueing (ready: {self.give_player1_e.is_set()}) next file for p1: '{next_file}' sim {similarity:.03f}"
                        )
                        self.playlist1_q.put((next_file_path, similarity))
                        self.give_player1_e.clear()
                    else:
                        console.log(
                            f"{self.p} queueing (ready: {self.give_player2_e.is_set()}) next file for p2: '{next_file}' sim {similarity:.03f}"
                        )
                        self.playlist2_q.put((next_file_path, similarity))
                        self.give_player2_e.clear()

                    p1_playing = not p1_playing
                    self.playlist.append(next_file_path)
                    self.ready_e.clear()

                # check for next file requests from player
                # if self.give_player1_e.is_set():
                #     # get and prep next file
                #     next_file, similarity = (self.player1.playing_file, 1.0)
                #     if not self.do_loop:
                #         next_file, similarity = self.seeker.get_msf_new(
                #             os.path.basename(next_file_path)
                #         )
                #     next_file_path = os.path.join(self.data_dir, str(next_file))
                #     next_file_path = self.change_tempo(next_file_path)

                #     # send next file to player
                #     self.playlist1_q.put((next_file_path, similarity))
                #     console.log(
                #         f"{self.p} added next file '{next_file}' to queue with similarity {similarity:.03f}"
                #     )

                #     self.playlist.append(next_file_path)
                #     self.give_player1_e.clear()

                # if self.reset_e.is_set():
                #     console.log(f"{self.p} [bold deep_pink3]RESETTING")

                #     # reset player
                #     while not self.playlist1_q.qsize() == 0:
                #         queued_file = self.playlist_q.get()
                #         console.log(
                #             f"{self.p}\tremoved queued segment: '{queued_file}'"
                #         )
                #         self.playlist_q.task_done()

                #     if player1_t is not None:
                #         self.kill_player1_e.set()
                #         console.log(f"{self.p}\twaiting for player to die")
                #         player1_t.join()
                #         self.kill_player1_e.clear()

                #     # clear recording
                #     self.listener.outfile = ""
                #     self.listener.recorded_notes = []
                #     self.recording_ready_e.clear()

                #     self.reset_e.clear()
                #     console.log(f"{self.p} reset complete")

                # check for keypresses
                # while not self.keypress_q.qsize() == 0:
                #     try:
                #         command = self.keypress_q.get()
                #         console.log(f"{self.p} got key command '{command}'")
                #         match command:
                #             case "FADE" | "MUTE" | "VOL DOWN" | "VOL UP":
                #                 self.playback_commmand_q.put(command)
                #             case "LOOP":
                #                 self.do_loop = not self.do_loop

                #                 self.player.next_file_path = (
                #                     self.player.playing_file_path
                #                 )

                #                 while not self.playlist_q.qsize() == 0:
                #                     queued_file, sim = self.playlist_q.get()
                #                     console.log(
                #                         f"{self.p}\tremoved queued segment: '{queued_file}'"
                #                     )
                #                     self.playlist_q.task_done()

                #                 self.give_player1_e.set()

                #             case "BACK":
                #                 console.log(f"\trewinding")

                #                 while not self.playlist_q.qsize() == 0:
                #                     queued_file, sim = self.playlist_q.get()
                #                     console.log(
                #                         f"{self.p}\tremoved queued segment: '{queued_file}'"
                #                     )
                #                     self.playlist_q.task_done()

                #                 self.player.next_file_path = self.playlist[-1]

                #                 self.give_player1_e.set()
                #             case _:
                #                 console.log(
                #                     f"{self.p}\tcommand unsupported '{command}'"
                #                 )

                #         self.keypress_q.task_done()
                #     except:
                #         console.log(f"{self.p} [bold orange]whoops")

        except KeyboardInterrupt:  # ctrl + c
            # end threads
            console.log(f"{self.p} [red bold]CTRL + C detected, shutting down")

            with open(os.path.join(self.playlist_dir, "playlist.txt"), "a") as f:
                for sample in self.playlist:
                    f.write(f"{sample}\n")

            self.kill_controller_e.set()
            self.kill_metro_e.set()
            self.kill_player1_e.set()
            self.kill_player2_e.set()
            self.kill_listener_e.set()

            if controller_thread.is_alive():
                controller_thread.join()
                console.log(f"{self.p} controller killed successfully")

            if metro_t.is_alive():
                metro_t.join()
                console.log(f"{self.p} metronome killed successfully")

            if player1_t.is_alive():
                player1_t.join()
                console.log(f"{self.p} player1 killed successfully")

            if player2_t.is_alive():
                player2_t.join()
                console.log(f"{self.p} player2 killed successfully")

            if listen_thread.is_alive():
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
            self.params.player1.in_port = self.params.in_port
            self.params.player2.in_port = self.params.in_port
            self.params.listener.in_port = self.params.in_port
        elif len(available_inputs) > 0:
            console.log(
                f"{self.p} unable to find MIDI device '{self.params.in_port}' falling back on '{available_inputs[0]}'"
            )
            self.input_port = mido.open_input(available_inputs[0])  # type: ignore
            self.params.player1.in_port = available_inputs[0]
            self.params.player2.in_port = available_inputs[0]
            self.params.listener.in_port = available_inputs[0]
        else:
            console.log(f"{self.p} no MIDI input devices available")

        # set up output connection
        if self.params.out_port in available_inputs:
            self.output_port = mido.open_output(self.params.out_port)  # type: ignore
            self.params.player1.out_port = self.params.out_port
            self.params.player2.out_port = self.params.out_port
            self.params.listener.out_port = self.params.out_port
        elif len(available_outputs) > 0:
            console.log(
                f"{self.p} unable to find MIDI device '{self.params.out_port}' falling back on '{available_outputs[0]}'"
            )
            self.output_port = mido.open_output(available_outputs[0])  # type: ignore
            self.params.player1.out_port = available_outputs[0]
            self.params.player2.out_port = available_outputs[0]
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
