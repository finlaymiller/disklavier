import threading
import pyaudio
import wave
import os
from pydub import AudioSegment

class AudioRecorder:
    def __init__(self, filepath):
        self.filepath = filepath
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 2
        self.rate = 44100
        self.frames = []
        self.kill_event = threading.Event()

    def record(self):
        audio = pyaudio.PyAudio()

        stream = audio.open(format=self.format,
                            channels=self.channels,
                            rate=self.rate,
                            input=True,
                            frames_per_buffer=self.chunk)

        print("Recording...")

        while not self.kill_event.is_set():
            data = stream.read(self.chunk)
            self.frames.append(data)

        print("Recording stopped.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = self.filepath.replace('.mp3', '.wav')
        with wave.open(wave_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))

        audio_segment = AudioSegment.from_wav(wave_file)
        audio_segment.export(self.filepath, format="mp3")
        os.remove(wave_file)

def record_audio(filepath):
    recorder = AudioRecorder(filepath)
    thread = threading.Thread(target=recorder.record)
    thread.start()
    return recorder, thread

# example usage
# recorder, thread = record_audio('output.mp3')
# recorder.kill_event.set()
# thread.join()

