import os
import torch
from diffusers.pipelines.deprecated.spectrogram_diffusion.midi_utils import (
    MidiProcessor,
)
from diffusers.pipelines.deprecated.spectrogram_diffusion.notes_encoder import (
    SpectrogramNotesEncoder,
)

from utils import console, basename
from utils.midi import change_tempo_and_trim


default_config = {
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
    "encoder_weights_path": "src/ml/specdiff/note_encoder.bin",
}


class SpectrogramDiffusion:
    tag = "[#aaaaff]spcdif[/#aaaaff]:"
    name = "SpectrogramDiffusion"

    def __init__(
        self, config: dict | None = None, fix_time: bool = True, verbose: bool = False
    ) -> None:
        if config is None:
            config = default_config
        console.log(f"{self.tag} initializing spectrogram diffusion model")
        self.device = config["device"]
        self.fix_time = fix_time
        self.verbose = verbose
        torch.set_grad_enabled(False)
        self.processor = MidiProcessor()
        self.encoder = (
            SpectrogramNotesEncoder(**config["encoder_config"]).cuda(device=self.device)
            if "cuda" in self.device
            else SpectrogramNotesEncoder(**config["encoder_config"]).to(self.device)
        )
        self.encoder.eval()
        sd = torch.load(config["encoder_weights_path"], weights_only=True)
        self.encoder.load_state_dict(sd)

        console.log(f"{self.tag} model initialization complete")

    def embed(self, path: str) -> torch.Tensor:
        if self.fix_time:
            tmp_dir = os.path.join(os.path.dirname(path), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_file = os.path.join(tmp_dir, basename(path))
            change_tempo_and_trim(path, tmp_file)
            path = tmp_file

        if self.verbose:
            console.log(f"{self.tag} generating embedding for '{path}'")
        tokens = self.processor(path)
        if self.verbose:
            console.log(f"{self.tag} {len(tokens)} {torch.tensor(tokens[0]).shape}")
        all_tokens = [torch.IntTensor(token) for token in tokens]
        if self.verbose:
            console.log(
                f"{self.tag} generated ({len(all_tokens)}, {all_tokens[0].shape}) tokens"
            )

        if self.verbose:
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
                tokens_embedded, tokens_mask = self.encoder(
                    encoder_input_tokens=batch, encoder_inputs_mask=tokens_mask
                )
            if self.verbose:
                console.log(
                    f"{self.tag} generated embedding {i} ({tokens_embedded.shape})"
                )

            embeddings.append(tokens_embedded[tokens_mask].detach().cpu())

        avg_embedding = torch.cat(embeddings).mean(0, keepdim=True)

        if self.verbose:
            console.log(f"{self.tag} embedding complete {avg_embedding.shape}")
        return avg_embedding
