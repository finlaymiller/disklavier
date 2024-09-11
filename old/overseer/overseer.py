import os
from shutil import copy2
from pathlib import Path
import time
from queue import Queue
from threading import Thread, Event
import mido

from player.player import Player
from player.metronome import Metronome
from listener.listener import Listener
from seeker.seeker import Seeker
from controller.controller import Controller

from utils import console
from utils.midi import augment_recording, text_to_midi, transform, change_tempo


class Overseer:
    p = "[green]ovrsee[/green]:"
    playing_file = ""
    playlist = []
    do_loop = False
    track_num = 0

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
        self.notes_path = os.path.join(self.playlist_dir, "all_notes.txt")
        # behavior
        self.params = params
        self.random_init = args.random_init  # pick random start file
        self.kickstart_file = args.kickstart  # start system from provided MIDI file
        self.read_commands = args.commands  # enable keyboard commands
        self.v_scale = args.velocity  # TODO: unimplemented augmentation
        self.playback_length = args.n
        # tempo
        self.tempo = tempo
        self.params.metronome.tempo = self.tempo
        self.params.player0.tempo = self.tempo
        self.params.player1.tempo = self.tempo
        self.params.player2.tempo = self.tempo
        self.params.listener.tempo = self.tempo
        self.params.seeker.tempo = self.tempo

        self._init_midi()  # make sure MIDI port is available first

        if len(os.listdir(self.data_dir)) < 10:
            console.log(
                f"{self.p} [red]fewer than 10 files in input folder. are you sure you didnt screw something up?"
            )

        if args.tick:
            self.params.player1

        # set up events & queues for threading
        self.reset_e = Event()
        # players
        self.play_e = Event()
        self.wait_player0_e = Event()
        self.wait_player1_e = Event()
        self.wait_player2_e = Event()
        self.kill_player0_e = Event()
        self.kill_player1_e = Event()
        self.kill_player2_e = Event()
        self.give_player0_e = Event()
        self.give_player1_e = Event()
        self.give_player2_e = Event()
        self.playlistR_q = Queue()
        self.playlist1_q = Queue()
        self.playlist2_q = Queue()
        self.playback_commmand_q = Queue()
        self.note_queue = Queue()
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
            self.params.seeker,
            self.data_dir,
            args.output_dir,
            tempo,
            args.dataset,
            "sequential" if args.s else "nearest",
        )
        self.listener = Listener(
            self.params.listener,
            self.record_dir,
            self.recording_ready_e,
            self.kill_listener_e,
            self.reset_e,
        )
        self.player0 = Player(
            self.params.player0,
            self.kill_player0_e,
            self.give_player0_e,
            self.wait_player0_e,
            self.play_e,
            self.playlistR_q,
            self.playback_commmand_q,
            self.note_queue,
        )
        self.player1 = Player(
            self.params.player1,
            self.kill_player1_e,
            self.give_player1_e,
            self.wait_player1_e,
            self.play_e,
            self.playlist1_q,
            self.playback_commmand_q,
            self.note_queue,
        )
        self.player2 = Player(
            self.params.player2,
            self.kill_player2_e,
            self.give_player2_e,
            self.wait_player2_e,
            self.play_e,
            self.playlist2_q,
            self.playback_commmand_q,
            self.note_queue,
        )
        self.metronome = Metronome(
            self.params.metronome, self.kill_metro_e, self.ready_e, self.play_e
        )
        self.controller = Controller(self.kill_controller_e, self.keypress_q)

    def run(self) -> None:
        if not self.input_port or not self.output_port:
            console.log(f"{self.p}[red bold] input or output port unavailable, exiting")
            return

        # random_init process
        if self.random_init:
            self.recording_ready_e.set()
            self.listener.outfile = os.listdir(self.data_dir)[0]
            console.log(f"{self.p} listener outfile set to '{self.listener.outfile}'")
        # kickstart
        if self.kickstart_file is not None:
            if self.kickstart_file == "RAND":
                self.kickstart_file = self.seeker.get_random()
                console.log(
                    f"{self.p} seeker kickstarting with random file '{self.kickstart_file}'"
                )
            self.recording_ready_e.set()
            self.listener.outfile = self.kickstart_file
        # start listening for recording
        else:
            self.listen_thread = Thread(
                target=self.listener.listen, args=(), name="listener"
            )
            self.listen_thread.start()

        # start up other threads
        self.controller_thread = Thread(
            target=self.controller.run, args=(), name="controller"
        )
        self.controller_thread.start()
        self.player0_t = Thread(
            target=self.player0.play_loop,
            name="player0",
        )
        self.player1_t = Thread(
            target=self.player1.play_loop,
            name="player1",
        )
        self.player2_t = Thread(
            target=self.player2.play_loop,
            name="player2",
        )
        self.player0_t.start()
        self.player1_t.start()
        self.player2_t.start()
        self.metro_t = Thread(target=self.metronome.tick, name="metronome")

        p1_playing = True  # used to alternate between players

        try:
            while (
                (self.track_num <= self.playback_length)
                if self.playback_length
                else True
            ):
                # check for recordings
                if self.recording_ready_e.is_set():
                    # get recording
                    if self.random_init:
                        recording_path = os.path.join(
                            self.playlist_dir,
                            f"{Path(os.path.join(self.data_dir, self.listener.outfile)).stem}.mid",
                        )
                        change_tempo(
                            os.path.join(self.data_dir, self.listener.outfile),
                            recording_path,
                            self.tempo,
                        )
                    elif self.kickstart_file:
                        recording_path = os.path.join(
                            self.playlist_dir, f"{Path(self.kickstart_file).stem}.mid"
                        )
                        change_tempo(self.kickstart_file, recording_path, self.tempo)
                    else:
                        recording_path = os.path.join(
                            self.record_dir, self.listener.outfile
                        )

                    # get most similar file to recording
                    first_file, first_similarity, first_transformations = (
                        self.seeker.match_recording(recording_path)
                    )
                    # check augments for any better matches
                    if self.params.augment_recording:
                        options = augment_recording(
                            recording_path, self.record_dir, self.tempo
                        )
                        console.log(
                            f"{self.p} augmented recording generated options", options
                        )

                        change = None
                        for opt in options:
                            match_path, match_sim, match_transformations = (
                                self.seeker.match_recording(opt)
                            )

                            if match_sim > first_similarity:
                                recording_path = opt
                                first_similarity = match_sim
                                first_file = match_path
                                first_transformations = match_transformations

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

                    console.log(
                        f"{self.p} triggering playback from recording '{recording_path}'"
                    )

                    self.playlist.append(
                        f"{self.track_num:02d} {os.path.join(self.playlist_dir, os.path.basename(recording_path))}"
                    )
                    self.ready_e.set()
                    self.play_e.set()
                    self.playlistR_q.put((recording_path, -1.0, {}))
                    next_file_path = os.path.join(self.data_dir, str(first_file))
                    next_file_path = transform(
                        next_file_path,
                        self.playlist_dir,
                        self.tempo,
                        first_transformations,
                    )
                    console.log(
                        f"{self.p} queueing (ready: {self.give_player0_e.is_set()}) recording for p1: '{recording_path}'"
                    )
                    console.log(
                        f"{self.p} next up is '{os.path.basename(next_file_path)}'"
                    )

                    # copy the version of the recording that we use to the playlist (not needed for kickstart)
                    if recording_path != os.path.join(
                        self.playlist_dir, os.path.basename(recording_path)
                    ):
                        copy2(
                            recording_path,
                            os.path.join(
                                self.playlist_dir, os.path.basename(recording_path)
                            ),
                        )

                    console.log(
                        f"{self.p} finished triggering playback from recording,\n\twaiting for playback to end"
                    )

                    # wait till recording playback is done
                    while not self.wait_player0_e.is_set():
                        time.sleep(0.001)

                    # kill recording player and start rest of system
                    console.log(
                        f"{self.p} recording playback ended, starting regular playback"
                    )
                    self.recording_ready_e.clear()
                    self.give_player0_e.clear()
                    self.kill_player0_e.set()
                    self.metro_t.start()
                    self.playlist1_q.put((next_file_path, first_similarity, {}))
                    self.give_player1_e.clear()

                # metronome says get ready
                if self.ready_e.is_set():
                    self.track_num += 1
                    # ready next file
                    next_file = (
                        self.player2.playing_file
                        if p1_playing
                        else self.player1.playing_file
                    )
                    if not self.do_loop:
                        # next_file, similarity = self.seeker.get_msf_new(
                        next_file_spec = self.seeker.get_most_similar_file(
                            os.path.basename(next_file_path),
                            bump_trans=self.track_num % 2 == 0,
                        )

                    console.log(
                        f"{self.p} applying transformations to '{next_file_spec['filename']}':",
                        next_file_spec["transformations"],
                        self.tempo,
                    )
                    next_file_path = os.path.join(
                        self.data_dir, next_file_spec["filename"]
                    )
                    out_dir = os.path.join(
                        self.playlist_dir, f"{Path(next_file_path).stem}.mid"
                    )
                    next_file_path = transform(
                        next_file_path,
                        self.playlist_dir,  # out_dir,
                        self.tempo,
                        next_file_spec["transformations"],
                    )

                    # send to player
                    if p1_playing:
                        console.log(
                            f"{self.p} queueing (ready: {self.give_player1_e.is_set()}) next file for p1: '{next_file_spec['filename']}' sim {next_file_spec['sim']:.03f}"
                        )
                        self.playlist1_q.put(
                            (
                                next_file_path,
                                next_file_spec["sim"],
                                next_file_spec["transformations"],
                            )
                        )
                        self.give_player1_e.clear()
                    else:
                        console.log(
                            f"{self.p} queueing (ready: {self.give_player2_e.is_set()}) next file for p2: '{next_file_spec['filename']}' sim {next_file_spec['sim']:.03f}"
                        )
                        self.playlist2_q.put(
                            (
                                next_file_path,
                                next_file_spec["sim"],
                                next_file_spec["transformations"],
                            )
                        )
                        self.give_player2_e.clear()

                    p1_playing = not p1_playing
                    self.playlist.append(f"{self.track_num:02d} {next_file_path}")
                    self.ready_e.clear()

                # check for keypresses
                if self.read_commands:
                    while not self.keypress_q.qsize() == 0:
                        try:
                            command = self.keypress_q.get()
                            console.log(f"{self.p} got key command '{command}'")
                            match command:
                                case "FADE" | "MUTE" | "VOL DOWN" | "VOL UP":
                                    self.playback_commmand_q.put(command)
                                case "LOOP":
                                    pass
                                    # self.do_loop = not self.do_loop

                                    # self.player.next_file_path = (
                                    #     self.player.playing_file_path
                                    # )

                                    # while not self.playlist_q.qsize() == 0:
                                    #     queued_file, sim = self.playlist_q.get()
                                    #     console.log(
                                    #         f"{self.p}\tremoved queued segment: '{queued_file}'"
                                    #     )
                                    #     self.playlist_q.task_done()

                                    # self.give_player1_e.set()

                                case "BACK":
                                    pass
                                    # console.log(f"\trewinding")

                                    # while not self.playlist_q.qsize() == 0:
                                    #     queued_file, sim = self.playlist_q.get()
                                    #     console.log(
                                    #         f"{self.p}\tremoved queued segment: '{queued_file}'"
                                    #     )
                                    #     self.playlist_q.task_done()

                                    # self.player.next_file_path = self.playlist[-1]

                                    # self.give_player1_e.set()
                                case _:
                                    console.log(
                                        f"{self.p}\tcommand unsupported '{command}'"
                                    )

                            self.keypress_q.task_done()
                        except:
                            console.log(f"{self.p} [bold yellow]whoops")

                # check for notes in queue
                if self.note_queue.not_empty:
                    while not self.note_queue.qsize() == 0:
                        with open(self.notes_path, "a") as f:
                            try:
                                note = self.note_queue.get_nowait()
                                f.write(f"{note}\n")
                            except:
                                console.log(f"{self.p} [bold yellow]whoops")
        except KeyboardInterrupt:  # catch ctrl + c
            console.log(f"{self.p} shutting down...")
        self.shutdown()

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
            self.params.player0.in_port = self.params.in_port
            self.params.player1.in_port = self.params.in_port
            self.params.player2.in_port = self.params.in_port
            self.params.listener.in_port = self.params.in_port
        elif len(available_inputs) > 0:
            console.log(
                f"{self.p} unable to find MIDI device '{self.params.in_port}' falling back on '{available_inputs[0]}'"
            )
            self.input_port = mido.open_input(available_inputs[0])  # type: ignore
            self.params.player0.in_port = available_inputs[0]
            self.params.player1.in_port = available_inputs[0]
            self.params.player2.in_port = available_inputs[0]
            self.params.listener.in_port = available_inputs[0]
        else:
            console.log(f"{self.p} no MIDI input devices available")

        # set up output connection
        if self.params.out_port in available_inputs:
            self.output_port = mido.open_output(self.params.out_port)  # type: ignore
            self.params.player0.out_port = self.params.out_port
            self.params.player1.out_port = self.params.out_port
            self.params.player2.out_port = self.params.out_port
            self.params.listener.out_port = self.params.out_port
        elif len(available_outputs) > 0:
            console.log(
                f"{self.p} unable to find MIDI device '{self.params.out_port}' falling back on '{available_outputs[0]}'"
            )
            self.output_port = mido.open_output(available_outputs[0])  # type: ignore
            self.params.player0.out_port = available_outputs[0]
            self.params.player1.out_port = available_outputs[0]
            self.params.player2.out_port = available_outputs[0]
            self.params.listener.out_port = available_outputs[0]
        else:
            console.log(f"{self.p} no MIDI output devices available")

    def shutdown(self):
        # end threads
        console.log(f"{self.p} [red bold]CTRL + C detected, shutting down")

        with open(os.path.join(self.playlist_dir, "playlist.txt"), "a") as f:
            for segment in self.playlist:
                f.write(f"{segment}\n")

        text_to_midi(self.notes_path, self.tempo)

        self.kill_controller_e.set()
        self.kill_metro_e.set()
        self.kill_player0_e.set()
        self.kill_player1_e.set()
        self.kill_player2_e.set()
        self.kill_listener_e.set()

        if self.listen_thread.is_alive():
            self.listen_thread.join()
            console.log(f"{self.p} listener killed successfully")

        if self.controller_thread.is_alive():
            self.controller_thread.join()
            console.log(f"{self.p} controller killed successfully")

        if self.metro_t.is_alive():
            self.metro_t.join()
            console.log(f"{self.p} metronome killed successfully")

        if self.player0_t.is_alive():
            self.player0_t.join()
            console.log(f"{self.p} player0 killed successfully")

        if self.player1_t.is_alive():
            self.player1_t.join()
            console.log(f"{self.p} player1 killed successfully")

        if self.player2_t.is_alive():
            self.player2_t.join()
            console.log(f"{self.p} player2 killed successfully")
