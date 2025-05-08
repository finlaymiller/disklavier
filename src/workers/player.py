import mido
import time
from queue import PriorityQueue
from datetime import datetime, timedelta

from utils import console
from utils.midi import TICKS_PER_BEAT
from .worker import Worker


class Player(Worker):
    """
    Plays MIDI from queue. scales velocities based on velocity stats from player.
    """

    td_last_note: datetime
    first_note = False
    n_notes = 0
    n_late_notes = 0

    # velocity tracking
    _recorder = None
    _avg_velocity: float = 0.0
    _min_velocity: int = 0
    _max_velocity: int = 0
    _velocity_adjustment_factor: float = 1.0
    _last_factor: float = 0

    def __init__(self, params, bpm: int, t_start: datetime):
        super().__init__(params, bpm=bpm)
        self.midi_port = mido.open_output(params.midi_port)  # type: ignore
        self.td_start = t_start
        self.td_last_note = t_start

        # if self.verbose:
        console.log(f"{self.tag} settings:\n{self.__dict__}")
        console.log(
            f"{self.tag} initialization complete, start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )

    def set_recorder(self, recorder):
        """
        Set the reference to the MidiRecorder.

        Parameters
        ----------
        recorder : MidiRecorder
            Reference to the MidiRecorder instance.
        """
        self._recorder = recorder
        console.log(f"{self.tag} connected to recorder for velocity updates")

    def check_velocity_updates(self) -> bool:
        """
        Check for velocity updates from the recorder.

        Returns
        -------
        bool
            True if velocity data was updated, False otherwise.
        """
        if self._recorder is None:
            console.log(f"{self.tag} no recorder connected")
            return False
        else:
            self._avg_velocity = self._recorder.avg_velocity
            self._min_velocity = self._recorder.min_velocity
            self._max_velocity = self._recorder.max_velocity

            if self.verbose:
                console.log(
                    f"{self.tag} updated velocity stats: avg={self._avg_velocity:.2f}, min={self._min_velocity}, max={self._max_velocity}"
                )
            return True

    def adjust_playback_based_on_velocity(self):
        if self._avg_velocity > 0:  # and self.verbose:
            if self._last_factor != self._velocity_adjustment_factor:
                console.log(
                    f"{self.tag} adjusting playback based on velocity: avg={self._avg_velocity:.2f}, min={self._min_velocity}, max={self._max_velocity}"
                )

            # Store velocity for future message adjustments
            self._last_factor = self._velocity_adjustment_factor
            self._velocity_adjustment_factor = (
                self._calculate_velocity_adjustment_factor()
            )

            if self._last_factor != self._velocity_adjustment_factor:
                console.log(
                    f"{self.tag} adjustment factor: {self._velocity_adjustment_factor:.2f}"
                )

    def _calculate_velocity_adjustment_factor(self):
        """
        Calculate velocity adjustment factor based on current velocity stats.

        Returns
        -------
        float
            Factor to scale message velocities.
        """
        # TODO: move this to class variables
        min_expected_velocity = 10
        max_expected_velocity = 100
        min_adjustment = 0.2  # minimum adjustment factor
        max_adjustment = 1.5  # maximum adjustment factor

        # default for middle-range velocity
        if self._avg_velocity == 0:
            return 1.0

        # calculate adjustment factor
        normalized_velocity = (self._avg_velocity - min_expected_velocity) / (
            max_expected_velocity - min_expected_velocity
        )
        normalized_velocity = max(0.0, min(1.0, normalized_velocity))  # clamp to [0, 1]

        adjustment_factor = min_adjustment + normalized_velocity * (
            max_adjustment - min_adjustment
        )

        return adjustment_factor

    def play(self, queue: PriorityQueue):
        console.log(
            f"{self.tag} start time is {self.td_start.strftime('%H:%M:%S.%f')[:-3]}"
        )

        while queue.qsize() > 0:
            # Check for velocity updates from recorder
            velocity_updated = self.check_velocity_updates()
            if velocity_updated:
                self.adjust_playback_based_on_velocity()

            tt_abs, msg = queue.get()
            ts_abs = mido.tick2second(tt_abs, TICKS_PER_BEAT, self.tempo)
            if self.verbose:
                console.log(
                    f"{self.tag} absolute time is {tt_abs} ticks (delta is {ts_abs:.03f} seconds)"
                )

            # may want to comment this if testing other player features
            if self.verbose:
                console.log(
                    f"{self.tag} \ttotal time should be {self.td_start.strftime('%H:%M:%S.%f')} + {ts_abs:.02f} = {(self.td_start + timedelta(seconds=ts_abs)).strftime(('%H:%M:%S.%f'))}"
                )

            td_now = datetime.now()
            if not self.first_note:
                self.td_last_note = td_now
            dt_sleep = self.td_start + timedelta(seconds=ts_abs) - td_now
            if dt_sleep.total_seconds() < -0.001:
                self.n_late_notes += 1
            if self.verbose:
                dt_tag = "yellow bold" if dt_sleep.total_seconds() < -0.001 else "blue"
                console.log(
                    f"{self.tag} \tit is {td_now.strftime('%H:%M:%S.%f')} and the last note was played at {self.td_last_note.strftime('%H:%M:%S.%f')}. I will sleep for [{dt_tag}]{dt_sleep.total_seconds()}[/{dt_tag}]s"
                )
            self.td_last_note += timedelta(seconds=ts_abs)

            if dt_sleep.total_seconds() > 0:
                if self.verbose:
                    console.log(
                        f"{self.tag} \twaiting until {(td_now + dt_sleep).strftime("%H:%M:%S.%f")[:-3]} to play message: ({msg})"
                    )
                time.sleep(dt_sleep.total_seconds())

            if msg.velocity > 0 and self.verbose:
                console.log(
                    f"{self.tag} playing ({msg})\t{queue.qsize():03d} events queued"
                )

            # Adjust message velocity based on player intensity
            if msg.type == "note_on" and msg.velocity > 0:
                original_velocity = msg.velocity
                adjusted_velocity = min(
                    127,
                    max(1, int(original_velocity * self._velocity_adjustment_factor)),
                )

                # TODO: only print this once per adjustment
                if adjusted_velocity != original_velocity and self.verbose:
                    console.log(
                        f"{self.tag} adjusting note velocity from {original_velocity} to {adjusted_velocity} (factor: {self._velocity_adjustment_factor:.2f})"
                    )

                msg = msg.copy(velocity=adjusted_velocity)

            self.midi_port.send(msg)
            self.n_notes += 1
            queue.task_done()

        # kill any remaining active notes
        for note in range(128):
            msg = mido.Message("note_off", note=note, velocity=0, channel=0)
            self.midi_port.send(msg)
        self.midi_port.close()
        console.log(
            f"{self.tag} [yellow bold]{self.n_late_notes}[/yellow bold]/{self.n_notes} notes were late (sent > 0.001 s after scheduled)"
        )
        console.log(f"{self.tag}[green] playback finished")

    def set_start_time(self, td_start: datetime):
        self.td_start = td_start
        console.log(f"{self.tag} start time set to {self.td_start}")
