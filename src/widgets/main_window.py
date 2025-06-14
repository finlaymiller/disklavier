import os
import yaml
import time
from queue import Queue
from threading import Event
from omegaconf import OmegaConf
from PySide6 import QtWidgets, QtCore
from datetime import datetime

import workers
from workers import Staff
from workers.midi_control_listener import MidiControlListener
from utils import console, write_log, midi, constants
from widgets.runner import Runner
from widgets.param_editor import ParameterEditorWidget
from widgets.piano_roll import PianoRollWidget
from widgets.recording_view import RecordingWidget

from typing import Optional


class MainWindow(QtWidgets.QMainWindow):
    tag = "[white]main[/white]  :"
    workers: Staff
    midi_stop_event: Event
    th_run: Optional[Runner] = None
    midi_listener: Optional[MidiControlListener] = None

    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.params.bpm = self.args.bpm
        self.td_system_start = datetime.now()
        self.recording_offset = 0  # seconds

        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("Disklavier")

        # toolbar
        self.toolbar = QtWidgets.QToolBar("Time Toolbar")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self._build_toolbar()

        # start by editing parameters
        self.param_editor = ParameterEditorWidget(self.params, self)
        self.setCentralWidget(self.param_editor)

        # Instantiate RecordingWidget
        self.recording_widget = RecordingWidget(self)

        # status bar
        self.status = self.statusBar()
        self.status.showMessage("parameter editor loaded")
        self.status.setVisible(True)

        # Add buttons to status bar
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_clicked)
        self.stop_btn.setEnabled(False)
        self.status.addPermanentWidget(self.stop_btn)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.start_clicked)
        self.status.addPermanentWidget(self.start_btn)

        # window dimensions
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(800, 600)
        self.resize(int(geometry.width() * 0.9), int(geometry.height() * 0.9))

        # update status with parameter file path
        param_path = os.path.join(os.getcwd(), "params", f"{self.args.params}.yaml")
        self.status.showMessage(f"loaded parameter file: '{param_path}'")
        self.status_label.setText(f"parameter file editor")

    def _build_toolbar(self):
        # trackers
        ## velocity
        self.velocity_label = QtWidgets.QLabel("Velocity Tracking: ----")
        self.velocity_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.velocity_label.setMinimumWidth(130)
        self.toolbar.addWidget(self.velocity_label)
        ## player embedding
        self.duet_sens_label = QtWidgets.QLabel(
            "Duet Sensitivity: <b><font color='grey'>0%</font></b> "
        )
        self.duet_sens_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.duet_sens_label.setMinimumWidth(130)
        self.toolbar.addWidget(self.duet_sens_label)
        ## rhythm
        self.rhythm_sens_label = QtWidgets.QLabel(
            "Rhythm Tracking: <b><font color='yellow'>----</font></b> "
        )
        self.rhythm_sens_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.rhythm_sens_label.setMinimumWidth(130)
        self.toolbar.addWidget(self.rhythm_sens_label)
        ## melody
        self.melody_sens_label = QtWidgets.QLabel(
            "Melody Tracking: <b><font color='light blue'>----</font></b> "
        )
        self.melody_sens_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.melody_sens_label.setMinimumWidth(130)
        self.toolbar.addWidget(self.melody_sens_label)

        # spacer
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.toolbar.addWidget(spacer)

        # status + playing segment name
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMinimumWidth(300)
        self._update_status_style("Initializing...")
        self.toolbar.addWidget(self.status_label)

        # spacer
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.toolbar.addWidget(spacer)

        # segments remaining
        self.segments_label = QtWidgets.QLabel("Segments Remaining: ----")
        self.segments_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.segments_label.setMinimumWidth(150)
        self.toolbar.addWidget(self.segments_label)

        # runtime
        self.time_label = QtWidgets.QLabel()
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.time_label.setMinimumWidth(100)
        self.toolbar.addWidget(self.time_label)

        # timer to automatically update the time and velocity
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_toolbar)
        self.timer.start(100)  # ms

        self._update_toolbar()

    def _update_toolbar(
        self, duet_sens: Optional[int] = None, transpose: Optional[int] = None
    ):
        """
        update the toolbar

        TODO: this is poorly designed and should be refactored to avoid
        unnecessary updates and checks.
        """
        runtime = datetime.now() - self.td_system_start
        runtime_text = f"{runtime.seconds//3600:02d}:{(runtime.seconds//60)%60:02d}:{runtime.seconds%60:02d}  "
        self.time_label.setText(runtime_text)

        # update velocity display
        # TODO: use the same method as the values below
        if hasattr(self, "workers") and hasattr(self.workers, "midi_recorder"):
            avg_vel = self.workers.midi_recorder.avg_velocity
            if avg_vel == 0:
                color = "grey"
            elif avg_vel < 30:
                color = "light blue"
            elif avg_vel < 60:
                color = "yellow"
            elif avg_vel < 90:
                color = "orange"
            else:
                color = "red"

            self.velocity_label.setText(
                f"Velocity Factor: <b><font color='{color}'>{int(100 * avg_vel / constants.MAX_VEL)}%</font></b>"
            )

        # update player follow threshold
        if duet_sens is not None:
            scaled_sens = int(100 * duet_sens)
            if scaled_sens < 25:
                color = "green"
            elif scaled_sens < 50:
                color = "yellow"
            elif scaled_sens < 75:
                color = "orange"
            else:
                color = "red"
            self.duet_sens_label.setText(
                f"Duet Sensitivity: <b><font color='{color}'>{scaled_sens}%</font></b>"
            )

        if transpose is not None:
            if transpose < constants.MAX_TRANSPOSE * 0.25:
                color = "green"
            elif transpose < constants.MAX_TRANSPOSE * 0.5:
                color = "yellow"
            elif transpose < constants.MAX_TRANSPOSE * 0.75:
                color = "orange"
            else:
                color = "red"
            self.duet_sens_label.setText(
                f"Duet Sensitivity: <b><font color='{color}'>{transpose}%</font></b>"
            )

    def init_fs(self):
        """
        initialize the filesystem.
        """
        # filesystem setup
        # create session output directories
        if not os.path.exists(self.p_log) and self.args.verbose:
            console.log(f"{self.tag} creating new logging folder at '{self.p_log}'")
        self.p_playlist = os.path.join(self.p_log, "playlist")
        os.makedirs(self.p_playlist, exist_ok=True)
        self.p_aug = os.path.join(self.p_log, "augmentations")
        os.makedirs(self.p_aug, exist_ok=True)

        # specify recording files
        self.pf_master_recording = os.path.join(self.p_log, f"master-recording.mid")
        self.pf_system_recording = os.path.join(self.p_log, f"system-recording.mid")
        self.pf_player_query = os.path.join(self.p_log, f"player-query.mid")
        self.pf_player_accompaniment = os.path.join(
            self.p_log, f"player-accompaniment.mid"
        )
        self.pf_schedule = os.path.join(self.p_log, f"schedule.mid")

        # copy old recording if replaying
        if self.args.replay and self.params.initialization == "recording":
            import shutil

            shutil.copy2(self.params.kickstart_path, self.pf_player_query)
            console.log(
                f"{self.tag} moved old recording to current folder '{self.pf_player_query}'"
            )
            self.params.seeker.pf_recording = self.pf_player_query

        # initialize playlist file
        self.pf_playlist = os.path.join(
            self.p_log, f"playlist_{self.td_system_start.strftime('%y%m%d-%H%M%S')}.csv"
        )
        write_log(self.pf_playlist, "position", "start time", "file path", "similarity")
        console.log(f"{self.tag} filesystem set up complete")

    def init_workers(self):
        """
        initialize the workers.
        """
        console.log(f"{self.tag} initializing scheduler")
        self.status.showMessage(f"initializing scheduler")
        scheduler = workers.Scheduler(
            self.params.scheduler,
            self.params.bpm,
            self.p_log,
            self.p_playlist,
            self.td_system_start,
            self.params.n_transitions,
            self.params.initialization == "recording",
        )
        console.log(f"{self.tag} initializing seeker")
        self.status.showMessage(f"initializing seeker")
        time.sleep(0.1)  # small delay to ensure UI updates
        seeker = workers.Seeker(
            self.params.seeker,
            self.p_aug,
            self.args.tables,
            self.args.dataset_path,
            self.p_playlist,
            self.params.bpm,
        )
        self.status.showMessage(f"initializing player")
        player = workers.Player(
            self.params.player, self.params.bpm, self.td_system_start
        )
        midi_recorder = workers.MidiRecorder(
            self.params.recorder,
            self.params.bpm,
            self.pf_player_query,
        )
        audio_recorder = workers.AudioRecorder(
            self.params.audio, self.params.bpm, self.p_log
        )
        panther = workers.Panther(self.params.panther, self.params.bpm)
        self.workers = Staff(
            seeker, player, scheduler, midi_recorder, audio_recorder, panther
        )

        # connect seeker and panther
        self.workers.seeker.set_panther(self.workers.panther)
        self.workers.midi_recorder.s_recording_progress.connect(
            self.recording_widget.update_recording_time
        )

        # --- initialize midi control listener(s) ---
        if (
            self.params.midi_control.duet_sensitivity.enable
            or self.params.midi_control.transpose.enable
        ):
            console.log(f"{self.tag} initializing midi control listener...")
            self.params.midi_control.duet_sensitivity.normalize = (
                "average" in self.params.player_tracking
            )
            self.midi_listener = MidiControlListener(
                params=self.params.midi_control,
                runner_ref=None,
                player_ref=self.workers.player,
            )
        else:
            console.log(f"{self.tag} midi control listener is disabled")

        self.status.showMessage("all workers initialized")

    def switch_to_piano_roll(self, q_gui: Queue):
        console.log(f"{self.tag} switching to piano roll view")
        self.piano_roll = PianoRollWidget(q_gui, self)
        self.setCentralWidget(self.piano_roll)
        if hasattr(self, "th_run") and self.th_run is not None:
            self.th_run.s_transition_times.connect(
                self.piano_roll.pr_view.update_transitions
            )
        self.status.showMessage("piano roll view activated")

    def save_and_start(self, params):
        """
        save parameters to yaml file and start the application
        """
        self.params = params
        ts_start = self.td_system_start.strftime("%y%m%d-%H%M%S")

        # filesystem setup
        # create session output directories
        self.p_log = os.path.join(
            self.args.output,
            f"{ts_start}_{self.params.seeker.metric}_{self.params.initialization}_{self.params.seeker.seed}",
        )
        os.makedirs(self.p_log, exist_ok=True)

        # --- save parameter file to disk ---
        # TODO: do the following in a better place/way
        if "average" in self.params.player_tracking:
            self.params.midi_control.duet_sensitivity.max = 1.0
        param_file = os.path.join(self.p_log, "parameters.yaml")
        try:
            with open(param_file, "w") as f:
                yaml.dump(OmegaConf.to_container(params), f, default_flow_style=False)

            self.status.showMessage(f"Parameters saved to {param_file}")
            self.status_label.setText(f"Parameters saved")
            console.log(f"{self.tag} parameters saved to '{param_file}'")
        except Exception as e:
            self.status.showMessage(f"Error saving parameters: {str(e)}")
            self.status_label.setText(f"Error saving parameters")
            console.log(f"{self.tag} error saving parameters: {str(e)}")

        self.status.showMessage("initializing filesystem")
        self.status_label.setText("Initializing filesystem")
        self.init_fs()
        self.status.showMessage("initializing workers")
        self.status_label.setText("Initializing workers")
        self.init_workers()

        self.workers.midi_recorder.s_recording_progress.connect(
            self.recording_widget.update_recording_time
        )

        self.status.showMessage("system initialization complete, preparing UI")
        self.status_label.setText("System initialization complete, preparing UI")

        # switch to recording view
        self.recording_widget.reset_widget()
        self.setCentralWidget(self.recording_widget)
        self.status.showMessage("recording view active")
        self.status_label.setText("Recording View")

        # start the main processing in a QThread
        self.th_run = Runner(self)
        self.th_run.s_start_time.connect(self.update_start_time)
        self.th_run.s_status.connect(self.update_status)
        self.th_run.s_segments_remaining.connect(self.update_segments_display)
        # connect run worker signals to recording widget
        self.th_run.s_augmentation_started.connect(
            self.recording_widget.init_augmentation_progress
        )
        self.th_run.s_embedding_processed.connect(
            self.recording_widget.update_augmentation_progress
        )
        self.th_run.s_duet_sensitivity.connect(self.update_duet_sensitivity)
        self.th_run.s_transpose.connect(self.update_transpose)
        self.th_run.start()

        # enable stop button now that system is running
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

        # make sure status bar is visible
        self.status.setVisible(True)

        # start MIDI listener if it was initialized and we have a run_thread
        if self.midi_listener is not None:
            console.log(f"{self.tag} starting midi control listener with run_thread...")
            self.midi_listener.runner_ref = self.th_run
            self.midi_listener.start()

    def update_start_time(self, start_time):
        """
        update the system start time based on the signal from run thread.
        """
        self.td_start = start_time
        if hasattr(self, "piano_roll"):
            self.piano_roll.td_start = start_time

    def update_segments_display(self, remaining_count: int):
        """
        update the segments remaining display in the toolbar.

        parameters
        ----------
        remaining_count : int
            number of segments left to play.
        """
        self.segments_label.setText(f"Segments Remaining: {remaining_count}  ")

    def update_status(self, message: str):
        """
        update both status bar and toolbar status label with the message.

        parameters
        ----------
        message : str
            status message to display.
        """
        # self.status.showMessage(message)
        self._update_status_style(message)

    def _update_status_style(self, message: str):
        # default style
        style = "border-radius: 4px; padding: 2px 8px;"

        # check if message matches the pattern "now playing 'x_y_tNNsNN'"
        # TODO: update to not require the presence of the 'now playing' string
        if not self.params.seeker.block_shift and "s" in message:
            try:
                # extract the number after 's'
                s_index = message.rindex("s")
                number = int(message[s_index + 1 :].split("'")[0])

                # set background color based on even/odd
                if number % 2 == 0:
                    style += "background-color: #90EE90;"  # light green
                else:
                    style += "background-color: #ADD8E6;"  # light blue
            except (ValueError, IndexError):
                pass  # if parsing fails, use default style

        self.status_label.setStyleSheet(style)
        self.status_label.setText(message)

    def update_duet_sensitivity(self, cc_value: int) -> None:
        self._update_toolbar(duet_sens=cc_value)

    def update_transpose(self, cc_value: int) -> None:
        self._update_toolbar(transpose=cc_value)

    def cleanup_workers(self):
        """
        clean up worker threads.
        """
        if not hasattr(self, "workers"):
            return

        console.log(f"{self.tag} cleaning up workers...")

        # Stop the run thread if it exists
        if self.th_run is not None and self.th_run.isRunning():
            console.log(f"{self.tag} requesting run_thread to stop...")
            self.th_run.stop_requested = True
            self.th_run.quit()
            if not self.th_run.wait(5000):
                console.log(
                    f"{self.tag} [yellow]run_thread did not terminate gracefully, forcing termination...[/yellow]"
                )
                self.th_run.terminate()
                self.th_run.wait()
            else:
                console.log(f"{self.tag} run_thread finished.")
        self.th_run = None

        if self.midi_listener is not None and self.midi_listener.running:
            console.log(f"{self.tag} stopping midi listener during cleanup...")
            self.midi_listener.stop()

        if hasattr(self, "workers") and self.workers is not None:
            pass

        # Generate piano roll visualization
        if hasattr(self, "pf_master_recording") and os.path.exists(
            self.pf_master_recording
        ):
            console.log(f"{self.tag} Generating piano roll visualization...")
            try:

                midi.generate_piano_roll(self.pf_master_recording)
            except ImportError:
                console.log(
                    f"{self.tag} [red]Failed to import midi module for piano roll generation.[/red]"
                )
            except Exception as e:
                console.log(f"{self.tag} [yellow]Failed to generate piano roll: {e}")
        elif hasattr(self, "pf_master_recording"):
            console.log(
                f"{self.tag} [yellow]Master recording not found at {self.pf_master_recording}, skipping piano roll generation."
            )
        else:
            console.log(
                f"{self.tag} [yellow]pf_master_recording attribute not found, skipping piano roll generation."
            )

        console.log(f"{self.tag} worker thread cleanup process complete.")

    def start_clicked(self):
        """
        handles the start button click.
        retrieves updated parameters from the editor and initiates the save and start sequence.
        """
        console.log(
            f"{self.tag} start button clicked. retrieving params and starting..."
        )
        updated_params = self.param_editor.get_updated_params()
        self.save_and_start(updated_params)

    def stop_clicked(self):
        self.cleanup_workers()

        self.status.showMessage("stopped")
        self.status_label.setText("Stopped")

        # Reset recording widget state
        if hasattr(self, "recording_widget"):
            self.recording_widget.reset_widget()

        # Return to parameter editor (original state)
        if hasattr(self, "piano_roll"):
            self.param_editor = ParameterEditorWidget(self.params, self)
            self.setCentralWidget(self.param_editor)
            self.status.showMessage("returned to parameter editor")
        elif hasattr(
            self, "param_editor"
        ):  # Ensure param_editor exists if piano_roll was never shown
            self.setCentralWidget(self.param_editor)
            self.status.showMessage("returned to parameter editor")

    def closeEvent(self, event):
        """
        handle window close event.
        """
        console.log(f"{self.tag} close event triggered.")
        self.cleanup_workers()
        console.log(
            f"{self.tag} saving console log to '{os.path.join(self.p_log, f'console.log')}'"
        )

        # save console log
        try:
            console.save_text(os.path.join(self.p_log, f"console.log"))
        except Exception as e:
            print(f"Error saving console log: {e}")
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.window_height = self.height()
        self.window_width = self.width()
