from rich.pretty import pprint
from ml.specdiff.model import SpectrogramDiffusion, default_config

pprint("starting test")
device = "cpu"
default_config["device"] = device
model = SpectrogramDiffusion(config=default_config, verbose=True)
model.embed("data/datasets/test/chords-060-04_amf_t00s00.mid")
pprint("DONE")
