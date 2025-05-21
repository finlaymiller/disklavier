import mido
import time
from queue import Queue
from dataclasses import dataclass
from datetime import datetime, timedelta

from PySide6 import QtWidgets
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtCore import QThread, Signal, QTimer, QObject
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QWidget

from utils import console
from utils.midi import TICKS_PER_BEAT

from typing import Optional, List, Tuple


@dataclass
class Note:
    pitch: int
    velocity: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    similarity: float = 0.0
    is_active: bool = True
    is_playing: bool = False


class PianoRollBuilder(QThread):
    tag = "[#90EE90]pr bld[/#90EE90]:"
    note_on_signal = Signal(Note)
    note_off_signal = Signal(int, datetime)

    def __init__(self, midi_queue: Queue, bpm: int, td_start: datetime):
        super().__init__()
        self.running = True
        self.queue = midi_queue
        self.bpm = bpm
        self.td_start = td_start
        self.tempo = mido.bpm2tempo(bpm)

    def run(self):
        """
        process messages from the queue and emit signals for note events.
        """
        console.log(
            f"{self.tag} start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )

        while self.running:
            if self.queue.qsize() == 0:
                time.sleep(0.01)  # small sleep to prevent cpu hogging
            else:
                # --- calculate when to send the message ---
                tt_abs, (sim, message) = self.queue.get()
                ts_abs = mido.tick2second(tt_abs, TICKS_PER_BEAT, self.tempo)
                dt_scheduled_time = self.td_start + timedelta(seconds=ts_abs)
                now = datetime.now()
                dt_sleep = dt_scheduled_time - now
                if dt_sleep.total_seconds() > 0:
                    # console.log(
                    #     f"{self.tag} \twaiting until {(now + dt_sleep).strftime("%H:%M:%S.%f")[:-3]} ({dt_sleep.total_seconds():.3f}s) to play message: ({message}) scheduled for {dt_scheduled_time.strftime('%H:%M:%S.%f')[:-3]}"
                    # )
                    time.sleep(dt_sleep.total_seconds())
                # else: # Play immediately if late or on time
                #     console.log(
                #         f"{self.tag} \tplaying message immediately: ({message}) scheduled for {dt_scheduled_time.strftime('%H:%M:%S.%f')[:-3]} at {now.strftime('%H:%M:%S.%f')[:-3]} ({(now - dt_scheduled_time).total_seconds():.3f}s late)"
                #     )

                # --- send message ---
                # now emit the signal for the appropriate note event with scheduled times
                if message.type == "note_on" and message.velocity > 0:
                    note = Note(
                        message.note,
                        message.velocity,
                        start_time=dt_scheduled_time,
                        similarity=sim,
                    )
                    self.note_on_signal.emit(note)
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    self.note_off_signal.emit(message.note, dt_scheduled_time)

    def stop(self):
        self.running = False
        self.wait()


class PianoRollView(QGraphicsView):
    """
    graphical view that displays a piano roll with scrolling notes.
    """

    # constants
    debug = True
    tms_timestep = 33  # ms
    tms_roll_length = 10000  # visible roll duration in ms # TODO: make this 10 seconds a global parameter
    tms_max_note_len = 5000  # auto trim notes longer than this
    min_note_height = 4
    max_note_height = 20
    min_key_width = 30
    max_key_width = 80
    colors = {
        "white": QColor(230, 230, 230),
        "black": QColor(51, 51, 51),
        "playing": QColor(255, 128, 128),
        "grid": QColor(178, 178, 178, 128),
        "background": QColor(38, 38, 38),
        "transition": QColor(0, 200, 0, 128),
        "highlight": QColor(55, 55, 55, 150),
    }
    min_note = 21  # A0
    max_note = 108  # C8
    note_range = max_note - min_note + 1
    default_bpm = 60  # default tempo in BPM

    # variables
    window_height = 600
    window_width = 800
    key_width = 60
    note_height = 10
    white_key_width = 60
    black_key_width = white_key_width * 0.85
    dtms_current_time = 0
    notes: list[Note] = []
    active_notes: dict[int, Note] = {}
    playing_notes = [0] * note_range
    current_tempo = default_bpm
    transitions: List[Tuple[float, bool]] = []

    def __init__(self, parent=None):
        # --- pull attribs from parent ---
        super().__init__(parent)
        if hasattr(parent, "td_start") and parent is not None:
            self.start_time = parent.td_start
            self.bpm = parent.bpm
        else:
            console.log(
                f"[orange bold] no start time found, using current time [/orange bold]"
            )
            self.start_time = datetime.now()

        # --- initialize scene ---
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # appearance settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(self.colors["background"])

        # update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(self.tms_timestep)

        # calculate dimensions based on initial size
        self.resize(800, 1000)

    def resizeEvent(self, event):
        """
        handle resize events and update dimensions.

        parameters
        ----------
        event : QResizeEvent
            resize event.
        """
        super().resizeEvent(event)
        self.window_width = self.width()
        self.window_height = self.height()
        self.calculate_dimensions()
        self._scene.setSceneRect(0, 0, self.window_width, self.window_height)

    def calculate_dimensions(self):
        """
        calculate ui dimensions based on window size.
        """
        self.note_height = min(
            max(self.window_height / self.note_range, self.min_note_height),
            self.max_note_height,
        )

        self.key_width = min(
            max(self.window_width * 0.15, self.min_key_width), self.max_key_width
        )

        self.white_key_width = self.key_width
        self.black_key_width = self.key_width * 0.85

    def is_black_key(self, note: int) -> bool:
        return note % 12 in [1, 3, 6, 8, 10]

    def get_note_y(self, note: int) -> float:
        note = max(min(note, self.max_note), self.min_note)
        # from bottom of screen
        return (
            self.window_height
            - (note - self.min_note) * self.note_height
            - self.note_height
        )

    def time_to_x(self, tms_time: float) -> float:
        """
        convert a time in milliseconds relative to start_time to an x-coordinate.

        parameters
        ----------
        tms_time : float
            time in milliseconds relative to start_time.

        returns
        -------
        float
            x-coordinate on screen.
        """
        relative_ms_to_now = tms_time - self.dtms_current_time
        # adjust time based on tempo scale - note: this scales the *duration displayed*
        adjusted_relative_ms = relative_ms_to_now

        # map relative time to x coordinate
        return self.key_width + (
            (adjusted_relative_ms + self.tms_roll_length)
            / self.tms_roll_length  # Use roll_len_ms for time window
        ) * (self.window_width - self.key_width)

    def is_note_at_keyboard(self, note: Note) -> bool:
        if note.start_time is None:
            return False

        start_ms = (note.start_time - self.start_time).total_seconds() * 1000

        if note.end_time:
            end_ms = (note.end_time - self.start_time).total_seconds() * 1000
        else:  # If note is active, its end is the current time
            end_ms = self.dtms_current_time

        start_x = self.time_to_x(start_ms)
        end_x = self.time_to_x(end_ms)

        return start_x <= self.key_width and end_x >= self.key_width

    def update_playing_notes(self):
        # reset all playing notes
        self.playing_notes = [0] * self.note_range

        # update playing status for all notes
        for note in self.notes:
            if self.is_note_at_keyboard(note):
                note.is_playing = True

                # note index in the playing_notes array
                note_idx = note.pitch - self.min_note
                if 0 <= note_idx < self.note_range:
                    self.playing_notes[note_idx] = 1
            else:
                note.is_playing = False

    def update_time(self):
        # current time in milliseconds relative to start_time
        self.dtms_current_time = (
            datetime.now() - self.start_time
        ).total_seconds() * 1000

        # check for stuck notes
        now_dt = datetime.now()
        for note_num, note in list(self.active_notes.items()):
            if (
                note.start_time
                and (now_dt - note.start_time).total_seconds() * 1000
                > self.tms_max_note_len
            ):
                if self.debug:
                    console.log(
                        f"\t\tending stuck note {note_num} started at {note.start_time}"
                    )
                self.note_off(int(note_num), now_dt)

        # cleanup old notes based on relative end time
        cutoff_time_ms = self.dtms_current_time - self.tms_roll_length * 2
        self.notes = [
            n
            for n in self.notes
            if n.is_active
            or (
                n.end_time
                and (n.end_time - self.start_time).total_seconds() * 1000
                > cutoff_time_ms
            )
        ]

        # update playing notes status
        self.update_playing_notes()

        # redraw
        self._scene.update()

    def note_on(self, note: Note):
        # end the previous note with its own start time to avoid visual glitches if it was stuck
        if note.pitch in self.active_notes:
            if self.debug:
                console.log(
                    f"\twarning: note {note.pitch} already active, ending previous note"
                )
            self.note_off(
                note.pitch, self.active_notes[note.pitch].start_time or datetime.now()
            )

        # note.start_time should already be set by the builder
        self.notes.append(note)
        self.active_notes[note.pitch] = note

    def note_off(self, pitch: int, dt_scheduled_end_time: datetime):
        if pitch in self.active_notes:
            self.active_notes[pitch].end_time = dt_scheduled_end_time
            self.active_notes[pitch].is_active = False
            del self.active_notes[pitch]
        elif self.debug:
            console.log(f"\twarning: received noteOff for inactive note: {pitch}")

    def set_tempo(self, bpm: int):
        if bpm > 0:
            if self.debug:
                console.log(f"\ttempo changed to: {bpm} BPM")
            self.current_tempo = bpm
        elif self.debug:
            console.log(f"\tinvalid tempo value: {bpm}")

    def cleanup(self):
        active_note_nums = list(self.active_notes.keys())
        if self.debug:
            console.log(f"\tcleaning up {len(active_note_nums)} active notes")

        now_dt = datetime.now()
        for note_num in active_note_nums:
            self.note_off(int(note_num), now_dt)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)

        painter.fillRect(
            self.key_width,
            0,
            self.window_width - self.key_width,
            self.window_height,
            self.colors["background"],
        )

        # self.draw_track_change_backgrounds(painter)
        self.draw_grid(painter)
        self.draw_transition_lines(painter)
        self.draw_notes(painter)
        self.draw_keyboard(painter)

    def update_transitions(self, transitions: List[Tuple[float, bool]]):
        console.log(f"\tupdating transitions: {transitions}")
        self.transitions = []
        for t_sec, is_new_track_flag in transitions:
            self.transitions.append((t_sec * 1000, is_new_track_flag))

    def draw_track_change_backgrounds(self, painter):
        if not self.transitions:
            return

        highlight_color = self.colors.get("highlight", QColor(55, 55, 55, 150))

        for i, (start_ms, is_new_track_here) in enumerate(self.transitions):
            if is_new_track_here:
                # this segment (from start_ms to end_ms_of_this_segment) should be highlighted
                x_start_region = self.time_to_x(start_ms)

                end_ms_of_this_segment = -1
                if i + 1 < len(self.transitions):
                    end_ms_of_this_segment = self.transitions[i + 1][0]
                else:
                    end_ms_of_this_segment = (
                        self.dtms_current_time + self.tms_roll_length * 1.5
                    )

                x_end_region = self.time_to_x(end_ms_of_this_segment)
                draw_x_start = max(x_start_region, self.key_width)
                draw_x_end = min(x_end_region, self.window_width)

                if draw_x_start < draw_x_end:
                    painter.fillRect(
                        draw_x_start,
                        0,
                        draw_x_end - draw_x_start,
                        self.window_height,
                        highlight_color,
                    )

    def draw_transition_lines(self, painter):
        """
        draw vertical green lines at transition points.
        """
        painter.setPen(QPen(self.colors["transition"], 2))
        for tms_transition, _ in self.transitions:
            x = self.time_to_x(tms_transition)
            if self.key_width <= x <= self.window_width:
                painter.drawLine(int(x), 0, int(x), self.window_height)

    def draw_keyboard(self, painter):
        # background
        painter.fillRect(0, 0, self.key_width, self.window_height, QColor(25, 25, 25))

        # --- white keys ---
        for i in range(self.min_note, self.max_note + 1):
            if not self.is_black_key(i):
                is_playing = self.playing_notes[i - self.min_note] == 1
                color = self.colors["playing"] if is_playing else self.colors["white"]
                y = self.get_note_y(i)

                painter.fillRect(0, y, self.white_key_width, self.note_height, color)

                # border
                painter.setPen(QPen(QColor(128, 128, 128), 0.5))
                painter.drawRect(0, y, self.white_key_width, self.note_height)

        # --- black keys ---
        for i in range(self.min_note, self.max_note + 1):
            if self.is_black_key(i):
                is_playing = self.playing_notes[i - self.min_note] == 1
                color = self.colors["playing"] if is_playing else self.colors["black"]
                y = self.get_note_y(i)

                painter.fillRect(0, y, self.black_key_width, self.note_height, color)

    def draw_grid(self, painter):
        # draw main background for the rolling part using window dimensions
        painter.fillRect(
            self.key_width,
            0,
            self.window_width - self.key_width,
            self.window_height,
            self.colors["background"],
        )

        # draw horizontal grid lines for each octave
        painter.setPen(QPen(self.colors["grid"], 1))
        for octave in range(9):
            note_number = 12 * octave + 12  # C notes
            if self.min_note <= note_number <= self.max_note:
                y = self.get_note_y(note_number)
                painter.drawLine(self.key_width, y, self.window_width, y)

        # draw additional grid lines for F notes
        painter.setPen(QPen(QColor(102, 102, 102, 76), 0.5))
        for octave in range(9):
            note_number = 12 * octave + 5  # F notes
            if self.min_note <= note_number <= self.max_note:
                y = self.get_note_y(note_number)
                painter.drawLine(self.key_width, y, self.window_width, y)

        # --- current time marker ---
        current_x = self.time_to_x(self.dtms_current_time)
        painter.setPen(QPen(QColor(255, 77, 77, 204), 2))
        painter.drawLine(current_x, 0, current_x, self.window_height)

    def draw_notes(self, painter):
        dtms_visible_start_time = self.dtms_current_time - self.tms_roll_length

        for note in self.notes:
            if note.start_time is None:
                continue

            # convert start/end datetimes to milliseconds relative to self.start_time
            dtms_start = (note.start_time - self.start_time).total_seconds() * 1000

            if note.end_time:
                end_ms = (note.end_time - self.start_time).total_seconds() * 1000
                # skip notes that ended before the visible time window
                if end_ms < dtms_visible_start_time:
                    continue
            else:  # note is active, use current time as end for drawing
                end_ms = self.dtms_current_time

            # calculate start and end positions using relative milliseconds
            start_x = self.time_to_x(dtms_start)
            end_x = self.time_to_x(end_ms)

            # clip start_x to key_width (don't draw over keyboard)
            start_x = max(start_x, self.key_width)

            # skip notes that start after the visible time window (horizontal)
            if start_x > self.window_width:
                continue

            # adjust end_x if it's off-screen (horizontal)
            end_x = min(end_x, self.window_width)

            # calculate note box dimensions
            y = self.get_note_y(note.pitch)
            note_width = max(end_x - start_x, 1)  # ensure a minimum width of 1px

            # Skip drawing if width is non-positive (can happen with clipping)
            if note_width <= 0:
                continue

            # note background scaled by velocity
            alpha = 0.25 + (note.velocity / 127) * 0.75

            if note.is_active:
                color = QColor(77, 77, 77, int(alpha * 255))
            elif note.is_playing:
                color = QColor(255, 128, 128, int(alpha * 255))
            else:
                if note.similarity < 0.8:
                    color = QColor(255, 255, 0, int(alpha * 255))
                else:
                    # interpolate between yellow and dark gray based on similarity
                    # when similarity is 1.0, we get the original dark gray (77,77,77)
                    # when similarity is 0.8, we get yellow (255,255,0)
                    t = (note.similarity - 0.8) / 0.2  # normalize to 0-1 range
                    r = int(128 + (255 - 128) * (1 - t))
                    g = int(204 + (255 - 204) * (1 - t))
                    b = int(255 + (0 - 255) * (1 - t))
                    color = QColor(r, g, b, int(alpha * 255))

            painter.fillRect(start_x, y, note_width, self.note_height, color)

            # note border
            painter.setPen(QPen(QColor(51, 51, 51, 204), 0.5))
            painter.drawRect(start_x, y, note_width, self.note_height)


class PianoRollWidget(QWidget):
    tag = "[#90FF00]pr wgt[/#90FF00]:"

    def __init__(self, message_queue: Queue, parent=None):
        super().__init__(parent)
        self.q_message = message_queue
        if parent is not None:
            self.td_start = parent.td_start
            self.bpm = parent.params.bpm
            console.log(
                f"{self.tag} using start time: {self.td_start.strftime('%H:%M:%S.%f')[:-3]}, bpm: {self.bpm}"
            )
        else:
            self.td_start = datetime.now()
            self.bpm = 60
            console.log(
                f"{self.tag}[orange bold] no start time found, using current time: {self.td_start} [/orange bold]"
            )

        # setup layout
        self.setMinimumSize(800, 600)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # create piano roll view
        self.pr_view = PianoRollView(self)
        layout.addWidget(self.pr_view)  # add view to layout

        # connect transition signal
        if parent is not None and hasattr(parent, "s_transition_times"):
            console.log(f"{self.tag} connecting transition signal")
            parent.s_transition_times.connect(self.pr_view.update_transitions)
        else:
            console.log(f"{self.tag} [yellow]no transition signal found[/yellow]")

        # create worker thread
        self.pr_builder = PianoRollBuilder(message_queue, self.bpm, self.td_start)
        self.pr_builder.note_on_signal.connect(self.pr_view.note_on)
        # Connect to the updated note_off signal signature
        self.pr_builder.note_off_signal.connect(self.pr_view.note_off)
        self.pr_builder.start()

    def update_start_time(self, start_time: datetime):
        """
        update the start time for both the widget and view.
        """
        self.td_start = start_time
        self.pr_view.start_time = start_time
        console.log(f"{self.tag} updated start time to {start_time}")

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.pr_builder.stop()
        self.pr_view.cleanup()
        super().closeEvent(event)
