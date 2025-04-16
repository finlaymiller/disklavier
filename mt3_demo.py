import sounddevice as sd
import numpy as np
import tempfile
import mido
import note_seq
import tensorflow as tf

from mt3 import configs
from mt3 import spec_utils
from mt3 import vocabularies
from mt3.models import mt3
from note_seq.protobuf import music_pb2
from note_seq import sequences_lib

SAMPLE_RATE = 16000
DURATION = 5  # seconds

def record_audio(duration=5, samplerate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]

def audio_to_midi(audio, sample_rate):
    # extract log-mel spectrogram
    config = configs.get_config('mt3')
    spec = spec_utils.compute_logmel(audio, sample_rate)
    spec = spec_utils.pad_or_trim(spec, config.audio_num_frames)
    spec = spec[None, ..., np.newaxis]  # add batch and channel dims

    # load model
    model = mt3.MT3(config, vocabularies.vocabularies['melody'])
    model.load_weights(config.pretrained_checkpoint_path).expect_partial()

    # predict tokens
    tokens = model.predict_tf(spec)
    ns = model.decode(tokens[0])

    # quantize to 4/4 MIDI
    ns = sequences_lib.quantize_note_sequence(ns, steps_per_quarter=4)
    return note_seq.sequence_to_pretty_midi(ns)

def send_midi(pmidi, port_name='Disklavier'):
    outport = mido.open_output(port_name)
    for msg in pmidi.instruments[0].midi_events:
        m = mido.Message.from_bytes(msg.message)
        outport.send(m)

if __name__ == '__main__':
    audio = record_audio(DURATION, SAMPLE_RATE)
    midi = audio_to_midi(audio, SAMPLE_RATE)
    send_midi(midi)