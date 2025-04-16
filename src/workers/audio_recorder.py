import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import mido
from threading import Thread, Event
from datetime import datetime
from typing import Optional

from .worker import Worker
from utils import console, tick


class AudioRecorder(Worker):
    is_recording: bool = False
    stop_event: Optional[Event] = None
    audio_thread: Optional[Thread] = None

    def __init__(
        self,
        params,
        bpm: int,
        output_dir: str,
    ):
        super().__init__(params, bpm=bpm)
        self.output_dir = output_dir

        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def run(self, td_start: datetime, stop_event: Event) -> None:
        """
        Records audio from the specified start time until the stop event is set.

        Parameters
        ----------
        td_start : datetime
            Start time for the recording.
        stop_event : Event
            Event that signals when to stop recording.
        """
        pf_output = os.path.join(self.output_dir, f"audio-recording.wav")

        # wait until td_start
        t_wait = (td_start - datetime.now()).total_seconds()
        if t_wait > 0:
            console.log(f"{self.tag} waiting {t_wait:.2f} s until recording start")
            time.sleep(t_wait)

        console.log(f"{self.tag} starting audio recording")
        self.is_recording = True

        recorded_data = []

        def callback(indata, frames, time, status):
            if status:
                console.log(f"{self.tag} status: {status}")
            recorded_data.append(indata.copy())

        # start metronome tick
        self.stop_tick_event = Event()
        self.metro_thread = Thread(
            target=tick,
            args=(
                self.bpm,
                self.stop_tick_event,
                self.params.tag,
                False,
            ),
            name="recorder metronome",
        )
        self.metro_thread.start()

        # start the stream
        with sd.InputStream(
            samplerate=self.params.sample_rate,
            channels=self.params.channels,
            callback=callback,
        ):
            console.log(f"{self.tag} recording to '{pf_output}'")
            stop_event.wait()

        
        self.stop_tick_event.set()
        self.metro_thread.join()

        if len(recorded_data) > 0:
            recorded_array = np.concatenate(recorded_data)
            sf.write(pf_output, recorded_array, self.params.sample_rate)
            console.log(f"{self.tag} saved audio recording to {pf_output}")
        else:
            console.log(f"{self.tag} no audio recorded")

        self.is_recording = False

    def start_recording(self, td_start: datetime) -> Event:
        """
        Starts audio recording in a separate thread.

        Parameters
        ----------
        td_start : datetime
            Start time for the recording.

        Returns
        -------
        Event
            The stop event that can be used to signal the recording to stop.
        """
        self.stop_event = Event()
        self.audio_thread = Thread(
            target=self.run,
            args=(td_start, self.stop_event),
            name="audio recorder",
            daemon=True,
        )
        self.audio_thread.start()
        return self.stop_event

    def stop_recording(self) -> bool:
        """
        Stops the audio recording thread.

        Returns
        -------
        bool
            True if recording was successfully stopped, False otherwise.
        """
        if self.is_recording and self.stop_event is not None:
            console.log(f"{self.tag} stopping audio recording")
            self.stop_event.set()
            if self.audio_thread is not None:
                self.audio_thread.join()
            return True
        else:
            console.log(f"{self.tag} audio recording not active")
            console.log(f"{self.tag} \tis_recording: {self.is_recording}")
            console.log(f"{self.tag} \tstop_event: {self.stop_event}")
            return False

    def record_query(self, record_path: str) -> float:
        """
        Records audio while the keyboard's left pedal is held down.

        Parameters
        ----------
        record_path : str
            Path where the audio recording will be saved.

        Returns
        -------
        float
            Duration of the recording in seconds.
        """
        start_time = datetime.now()
        end_time = None
        recorded_data = []

        def callback(indata, frames, time, status):
            if status:
                console.log(f"{self.tag} status: {status}")
            recorded_data.append(indata.copy())

        console.log(
            f"{self.tag} waiting for pedal press (control 67) to start recording"
        )

        # Initialize audio input stream
        stream = sd.InputStream(
            samplerate=self.params.sample_rate,
            channels=self.params.channels,
            callback=callback,
        )

        try:
            # Open MIDI input port
            with mido.open_input(self.params.midi_port) as inport:  # type: ignore
                recording_started = False
                stream.start()

                # Start audio stream once MIDI port is open

                for msg in inport:
                    # Check for pedal press/release (control 67)
                    if (
                        msg.type == "control_change"
                        and msg.control == self.params.record
                    ):
                        # Pedal pressed
                        if msg.value > 0 and not recording_started:
                            start_time = datetime.now()
                            console.log(
                                f"{self.tag} pedal pressed, starting audio recording"
                            )
                            self.is_recording = True
                            recording_started = True

                        # Pedal released
                        elif msg.value == 0 and recording_started:
                            end_time = datetime.now()
                            self.is_recording = False
                            console.log(
                                f"{self.tag} pedal released, stopping audio recording"
                            )
                            break
        finally:
            # Stop and close the audio stream
            stream.stop()
            stream.close()

        # Calculate duration
        duration = 0.0
        if end_time is not None:
            duration = (end_time - start_time).total_seconds()
            console.log(f"{self.tag} recorded {duration:.2f} seconds of audio")

        # Save the recording if any data was collected
        if len(recorded_data) > 0:
            recorded_array = np.concatenate(recorded_data)
            sf.write(record_path, recorded_array, self.params.sample_rate)
            console.log(f"{self.tag} saved audio recording to {record_path}")
        else:
            console.log(f"{self.tag} no audio recorded")
            duration = 0.0

        return duration
