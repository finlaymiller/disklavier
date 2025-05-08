import os
import sys
import torch
import pretty_midi
from diffusers.pipelines.deprecated.spectrogram_diffusion.midi_utils import (
    MidiProcessor,
)
from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import (
    SpectrogramNotesEncoder,
)

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

    from utils import console, basename
    from utils.midi import change_tempo_and_trim, get_bpm
else:
    from src.utils import console, basename
    from src.utils.midi import change_tempo_and_trim, get_bpm

from typing import Optional


DEFAULT_CONFIG = {
    "device": "cuda:0",
    "encoder_config": {
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 768,
        "dropout_rate": 0.1,
        "feed_forward_proj": "gated-gelu_pytorch_tanh",
        "is_decoder": False,
        "max_length": 2048,
        "num_heads": 12,
        "num_layers": 12,
        "vocab_size": 1536,
    },
    "encoder_weights_path": "/home/finlay/disklavier/src/ml/specdiff/note_encoder.bin",
    "ts_min": 4.500,
    "ts_max": 5.119,
}

class SpectrogramDiffusion:
    tag = "[light_pink4]spcdif[/light_pink4]:"
    name = "SpectrogramDiffusion"

    def __init__(
        self,
        new_config: Optional[dict] = None,
        fix_time: bool = True,
        verbose: bool = False,
    ) -> None:
        console.log(f"{self.tag} initializing spectrogram diffusion model")

        # override config
        if new_config is None:
            config = DEFAULT_CONFIG
        else:
            config = DEFAULT_CONFIG
            for k, v in new_config.items():
                config[k] = v

        self.device = config["device"]
        self.ts_min = config["ts_min"]
        self.ts_max = config["ts_max"]
        self.fix_time = fix_time
        self.verbose = verbose
        torch.set_grad_enabled(False)
        self.processor = MidiProcessor()
        self.encoder = SpectrogramNotesEncoder(**config["encoder_config"]).to(
            self.device
        )
        self.encoder.eval()
        if os.path.exists(config["encoder_weights_path"]):
            sd = torch.load(config["encoder_weights_path"], weights_only=True)
            self.encoder.load_state_dict(sd)
        else:
            console.log(
                f"{self.tag} no encoder weights found at {config['encoder_weights_path']}"
            )
            console.log(f"{self.tag} {os.getcwd()}")
            raise ValueError(
                f"Encoder weights not found at {config['encoder_weights_path']}"
            )

        console.log(f"{self.tag} model initialization complete")

    def embed(self, path: str) -> torch.Tensor:
        if self.verbose:
            console.log(f"{self.tag} generating embedding for '{path}'")
        if self.fix_time:
            bpm = get_bpm(path)
            if self.verbose:
                console.log(f"{self.tag} guessing that {basename(path)} has bpm {bpm}")

            midi_len = pretty_midi.PrettyMIDI(path, initial_tempo=bpm).get_end_time()
            if midi_len == 0:
                console.log(
                    f"{self.tag} [yellow] midi duration is 0, skipping[/yellow]"
                )
                return torch.zeros(1, 768)

            if midi_len < self.ts_min or midi_len > self.ts_max:
                new_bpm = bpm * (midi_len / self.ts_max)
                if self.verbose:
                    console.log(
                        f"{self.tag} midi duration {midi_len:.03f} is out of bounds ({self.ts_min} to {self.ts_max}), changing tempo from {bpm} to {new_bpm:.03f}"
                    )
                tmp_dir = os.path.join(os.path.dirname(path), "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_file = os.path.join(tmp_dir, basename(path))
                change_tempo_and_trim(path, tmp_file, new_bpm)
                path = tmp_file

        tokens = self.processor(path)
        if self.verbose:
            console.log(
                f"{self.tag} generated ({len(tokens)}, {len(tokens[0])}) tokens"
            )
        if len(tokens) > 1:
            if self.verbose:
                console.log(
                    f"{self.tag}[yellow italic] too many pooled tokens, using first one[/yellow italic]"
                )
            tokens = [tokens[0]]

        console.log(f"{self.tag} inting")
        all_tokens = torch.IntTensor(tokens) #[torch.IntTensor(token) for token in tokens]

        console.log(f"{self.tag} embedding")
        embeddings = []
        for i in range(0, len(all_tokens)):
            batch = (
                all_tokens[i].view(1, -1).cuda(self.device)
                if "cuda" in self.device
                else all_tokens[i].view(1, -1)
            )
            with torch.autocast("cuda" if "cuda" in self.device else "cpu"):
                tokens_mask = batch > 0
                console.log(f"{self.tag} embedding {i}")
                tokens_embedded, tokens_mask = self.encoder(
                    encoder_input_tokens=batch, encoder_inputs_mask=tokens_mask
                )
            console.log(f"{self.tag} embedding {i} done")
            if self.verbose:
                console.log(
                    f"{self.tag} generated embedding {i} ({tokens_embedded.shape})"
                )

            embeddings.append(tokens_embedded[tokens_mask].detach().cpu())

        avg_embedding = torch.cat(embeddings).mean(0, keepdim=True)

        if self.verbose:
            console.log(
                f"{self.tag} embedding of '{basename(path)}' complete {avg_embedding.shape}"
            )
        return avg_embedding
