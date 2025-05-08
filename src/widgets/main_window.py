import os
import time
import yaml
from queue import Queue
from threading import Event
from omegaconf import OmegaConf
from PySide6 import QtWidgets, QtCore
from datetime import datetime

import workers
from workers import Staff
from utils import console, write_log
from widgets.runner import RunWorker
from widgets.piano_roll import PianoRollWidget
from widgets.param_editor import ParameterEditorWidget


from typing import Optional


class MainWindow(QtWidgets.QMainWindow):
    tag = "[white]main[/white]  :"
    workers: Staff
    midi_stop_event: Event
    run_thread: Optional[RunWorker] = None

    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.params.bpm = self.args.bpm
        self.td_system_start = datetime.now()
        self.recording_offset = 0  # seconds

        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("disklavier")
        # toolbar
        self.toolbar = QtWidgets.QToolBar("Time Toolbar")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self._build_timer()

        # start by editing parameters
        self.param_editor = ParameterEditorWidget(self.params, self)
        self.setCentralWidget(self.param_editor)

        # status bar
        self.status = self.statusBar()
        self.status.showMessage("parameter editor loaded")
        self.status.setVisible(True)  # Ensure it's always visible by default

        # Add buttons to status bar
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_clicked)
        self.stop_btn.setEnabled(False)  # Disabled by default
        self.status.addPermanentWidget(self.stop_btn)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setDefault(True)  # Make it the default button
        self.start_btn.clicked.connect(self.start_clicked)
        self.status.addPermanentWidget(self.start_btn)

        # Window dimensions
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(800, 600)
        self.resize(int(geometry.width() * 0.8), int(geometry.height() * 0.8))

    def set_model(self, model):
        """
        set the model for the main window.

        parameters
        ----------
        model : object
            the model to be used by the window.
        """
        self.model = model
        # If there are components that depend on the model and are initialized
        # in __init__, you might need to re-initialize them or update them here.
        # For example, if param_editor depends on the model:
        # self.param_editor = ParameterEditorWidget(self.params, self)
        # self.setCentralWidget(self.param_editor)
        # Or, if workers depend on the model, you might need to re-initialize them
        # if they were already created.
        # if self.model is not None and hasattr(self, 'workers'):
        # self.init_workers() # Or a more specific update method

    def _build_timer(self):
        # Create velocity label (left side)
        self.velocity_label = QtWidgets.QLabel("Velocity: <b>0.0</b>")
        self.velocity_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.velocity_label.setMinimumWidth(120)
        font = self.velocity_label.font()
        font.setPointSize(font.pointSize() + 1)
        self.velocity_label.setFont(font)
        self.toolbar.addWidget(self.velocity_label)

        # Create segments label (after velocity)
        self.segments_label = QtWidgets.QLabel("Segments Left: N/A")
        self.segments_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.segments_label.setMinimumWidth(150)
        self.segments_label.setFont(font)  # use same font as velocity
        self.toolbar.addWidget(self.segments_label)

        # Create status label (middle)
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setMinimumWidth(300)
        self._update_status_style("Initializing...")
        status_font = self.status_label.font()
        status_font.setPointSize(status_font.pointSize() + 1)
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        self.toolbar.addWidget(self.status_label)

        # Add spacer between status and time
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.toolbar.addWidget(spacer)

        # Create time label (right side)
        self.time_label = QtWidgets.QLabel()
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.time_label.setMinimumWidth(100)
        self.toolbar.addWidget(self.time_label)

        # Timer to update the time and velocity
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_time)
        self.timer.start(100)  # update every 100ms for smoother velocity updates

        # Initial time update
        self._update_time()

    def _update_time(self):
        """
        update the time display in the toolbar.
        """
        current_time = QtCore.QTime.currentTime()
        time_text = current_time.toString("hh:mm:ss")
        delta = datetime.now() - self.td_system_start
        delta_text = f"{delta.seconds//3600:02d}:{(delta.seconds//60)%60:02d}:{delta.seconds%60:02d}"
        self.time_label.setText(time_text + " | " + delta_text)

        # Update velocity display if workers are initialized
        if hasattr(self, "workers") and hasattr(self.workers, "midi_recorder"):
            avg_vel = self.workers.midi_recorder.avg_velocity

            # Determine color based on velocity
            if avg_vel < 40:
                color = "light blue"
            elif avg_vel < 80:
                color = "orange"
            else:
                color = "red"

            self.velocity_label.setText(
                f"Velocity: <b><font color='{color}'>{avg_vel:.1f}</font></b>"
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
        scheduler = workers.Scheduler(
            self.params.scheduler,
            self.params.bpm,
            self.p_log,
            self.p_playlist,
            self.td_system_start,
            self.params.n_transitions,
            self.params.initialization == "recording",
        )
        seeker = workers.Seeker(
            self.params.seeker,
            self.p_aug,
            self.args.tables,
            self.args.dataset_path,
            self.p_playlist,
            self.model,
            self.params.bpm,
        )
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
        # seeker.set_panther(panther)
        self.workers = Staff(
            seeker, player, scheduler, midi_recorder, audio_recorder, panther
        )

    def switch_to_piano_roll(self, q_gui: Queue):
        console.log(f"{self.tag} switching to piano roll view")
        self.piano_roll = PianoRollWidget(q_gui, self)
        self.setCentralWidget(self.piano_roll)
        if hasattr(self, "run_thread") and self.run_thread is not None:
            self.run_thread.s_transition_times.connect(
                self.piano_roll.pr_view.update_transitions
            )
        self.status.showMessage("piano roll view activated")

        # Ensure status bar is visible
        self.status.setVisible(True)

    def save_and_start(self, params):
        """
        save parameters to yaml file and start the application
        """
        ts_start = self.td_system_start.strftime("%y%m%d-%H%M%S")

        # filesystem setup
        # create session output directories
        self.p_log = os.path.join(
            self.args.output,
            f"{ts_start}_{self.params.seeker.metric}_{self.params.initialization}_{self.params.seeker.seed}",
        )
        os.makedirs(self.p_log, exist_ok=True)

        # Create parameters file in p_log
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
        self.status.showMessage("system initialization complete")
        self.status_label.setText("System initialization complete")

        # Start the main processing in a QThread
        self.run_thread = RunWorker(self)
        self.run_thread.s_start_time.connect(self.update_start_time)
        self.run_thread.s_status.connect(self.update_status)
        self.run_thread.s_segments_remaining.connect(self.update_segments_display)
        self.run_thread.start()

        # Enable stop button now that system is running
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

        # Make sure status bar is visible
        self.status.setVisible(True)

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
        self.segments_label.setText(f"Segments Left: {remaining_count}")

    def update_status(self, message: str):
        """
        update both status bar and toolbar status label with the message.

        parameters
        ----------
        message : str
            status message to display.
        """
        self.status.showMessage(message)
        self._update_status_style(message)
        # self.segments_label.setText("Segments Left: N/A")

    def _update_status_style(self, message: str):
        # default style
        style = "border-radius: 4px; padding: 2px 8px;"

        # check if message matches the pattern "now playing 'x_y_tNNsNN'"
        if "now playing" in message and "s" in message:
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

    def cleanup_workers(self):
        """
        clean up all worker processes and threads.
        """
        if not hasattr(self, "workers"):
            return

        console.log(f"{self.tag} cleaning up workers...")

        # Stop the run thread if it exists
        if self.run_thread is not None and self.run_thread.isRunning():
            self.run_thread.shutdown()
            self.run_thread.requestInterruption()
            self.run_thread.wait(1000)  # Wait up to 1 second for thread to finish

        console.log(f"{self.tag} workers cleaned up")

        # Update button states
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.segments_label.setText("Segments Left: N/A")

    def stop_clicked(self):
        self.cleanup_workers()

        self.status.showMessage("stopped")
        self.status_label.setText("Stopped")

        # Return to parameter editor (original state)
        if hasattr(self, "piano_roll"):
            self.param_editor = ParameterEditorWidget(self.params, self)
            self.setCentralWidget(self.param_editor)
            self.status.showMessage("returned to parameter editor")

    def closeEvent(self, event):
        self.cleanup_workers()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.window_height = self.height()
        self.window_width = self.width()

    def start_clicked(self):
        """
        handle the start button click.
        """
        params = self.param_editor.get_updated_params()
        console.log(f"params: {params}")
        self.save_and_start(params)
