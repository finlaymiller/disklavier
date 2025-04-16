import mido
import sounddevice as sd
import numpy as np
import tempfile
import os
from scipy.io.wavfile import write
import librosa
import soundfile as sf
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
from mido import MidiFile
import mido


def record_audio(duration=5, samplerate=44100):
    """Record audio from the user's microphone.

    Args:
        duration (int): Duration of the recording in seconds. Defaults to 5.
        samplerate (int): Sampling rate in Hz. Defaults to 44100.

    Returns:
        tuple[str, np.ndarray, int]: Path to the recorded WAV file, the audio data, and the samplerate.
    """
    # record audio
    print("Recording...")
    audio_data = sd.rec(
        int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16"
    )
    sd.wait()
    print("Recording finished.")

    # save to a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_wav.name, samplerate, audio_data)
    return temp_wav.name, audio_data, samplerate


def process_audio_to_midi(wav_path):
    """process the recorded wav file to extract midi data using basic pitch.

    this function loads the audio, resamples it to 16khz, ensures it's mono,
    trims silence, normalizes the volume, saves the processed audio,
    and then runs basic pitch inference.

    parameters
    ----------
    wav_path : str
        path to the wav file.

    returns
    -------
    tuple[str, str]
        path to the generated midi file and path to the processed wav file.
    """
    # load audio, resample to 16khz, ensure mono
    print("preprocessing audio...")
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    # trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)  # adjust top_db as needed

    # normalize volume
    y_normalized = librosa.util.normalize(y_trimmed)

    # save normalized audio to a temporary file
    temp_processed_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_processed_wav.name, y_normalized, sr)
    print(f"processed audio saved to: {temp_processed_wav.name}")

    # play back the recorded audio
    print("playing recorded audio...")
    sd.play(y_normalized, 16000)
    sd.wait()
    print("playback finished.")

    # basic pitch model setup
    model = Model(ICASSP_2022_MODEL_PATH)

    # run basic pitch prediction with processed audio data
    # note: predict expects a tuple (audio_array, sample_rate) for direct audio input
    print("running basic pitch inference...")
    model_output, midi_data, note_events = predict(
        audio_path=temp_processed_wav.name,  # pass the path to the processed file
        model_or_model_path=model,
        # onset_threshold=0.3,
        # frame_threshold=0.1,
    )
    print("basic pitch inference complete.")

    # save midi data to a temporary file
    temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    midi_data.write(temp_midi.name)
    return temp_midi.name, temp_processed_wav.name


def play_midi(midi_path):
    with mido.open_output("Disklavier") as output:
        # load and play the MIDI file
        mid = MidiFile(midi_path)
        for msg in mid.play():
            output.send(msg)


def main():
    # record audio from the microphone
    samplerate = 44100  # define samplerate here or pass it if needed elsewhere
    wav_path, audio_data, _ = record_audio(duration=5, samplerate=samplerate)

    # play back the recorded audio
    print("playing recorded audio...")
    sd.play(audio_data, samplerate)
    sd.wait()
    print("playback finished.")

    # process the recorded audio to MIDI
    midi_path, processed_wav_path = process_audio_to_midi(wav_path)

    # play the MIDI file using the specified SoundFont
    play_midi(midi_path)

    # clean up temporary files
    print("cleaning up temporary files...")
    os.unlink(wav_path)
    os.unlink(processed_wav_path)  # clean up processed wav
    os.unlink(midi_path)
    print("cleanup complete.")


if __name__ == "__main__":
    main()
