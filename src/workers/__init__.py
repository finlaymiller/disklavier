from dataclasses import dataclass
from .player import Player
from .scheduler import Scheduler
from .seeker import Seeker
from .metronome import Metronome
from .panther import Panther
from .midi_recorder import MidiRecorder
from .audio_recorder import AudioRecorder


@dataclass
class Staff:
    def __init__(
        self,
        seeker: Seeker,
        player: Player,
        scheduler: Scheduler,
        midi_recorder: MidiRecorder,
        audio_recorder: AudioRecorder,
        panther: Panther,
    ):
        self.seeker = seeker
        self.player = player
        self.scheduler = scheduler
        self.midi_recorder = midi_recorder
        self.audio_recorder = audio_recorder
        self.panther = panther