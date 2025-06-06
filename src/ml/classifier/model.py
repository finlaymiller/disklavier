import torch
import torch.nn as nn

from utils import console


class Classifier(nn.Module):
    tag = "[#ffafaf]clf   [/#ffafaf]:"

    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.SiLU):
        super(Classifier, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn())
            current_dim = hidden_dim
        self.hidden_layer = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x, return_hidden: bool = False) -> torch.Tensor:
        hidden_representation = self.hidden_layer(x)
        if return_hidden:
            return hidden_representation
        return self.output_layer(hidden_representation)

    def embed(self, path: str) -> torch.Tensor:
        x = torch.load(path, weights_only=True)
        if x.dim() == 1:
            x = x.unsqueeze(0).reshape(-1, 768)
        return self.forward(x, return_hidden=True).detach().cpu()
