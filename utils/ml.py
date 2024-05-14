import os
import importlib
import torch
import torch.nn as nn
import torch.nn.init as init


def get_model_list(directory=os.path.join("ml", "models")):
    return [
        file[:-3]
        for file in os.listdir(directory)
        if file.endswith(".py") and not file.startswith("__")
    ]


def load_model(
    model_name: str, model_file: str, model_directory=os.path.join("ml", "models")
):
    spec = importlib.util.spec_from_file_location(  # type: ignore
        model_name, os.path.join(model_directory, f"{model_file}.py")
    )
    if spec is None:
        raise ImportError(f"Couldn't find model file {model_file} in {model_directory}")

    module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)

    try:
        model = getattr(module, model_name)
    except AttributeError:
        raise ImportError(f"{model_name} not found in {model_directory}")

    return model()

def init_weights(m, a, b):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, a, b)
        if m.bias is not None:
            init.uniform_(m.bias, a, b)
