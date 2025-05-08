from ml.specdiff.model import SpectrogramDiffusion, DEFAULT_CONFIG


config = DEFAULT_CONFIG
config["device"] = "cpu"
config["encoder_weights_path"] = (
	"/Users/finlay/Documents/Programming/disklavier/src/ml/specdiff/note_encoder.bin"
)
model = SpectrogramDiffusion(
	new_config=config,
	fix_time=True,
	verbose=True,
)

model.embed("data/datasets/test/velocitytweaks-060-05_c4vel100_t00s00.mid")
