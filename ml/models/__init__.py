from .bad_dae import *
from .base import *
from .vanilla_vae import *
from .vq_vae import *

vae_models = {"BadDAE": BadAutoEncoder, "VanillaVAE": VanillaVAE, "VQVAE": VQVAE}
