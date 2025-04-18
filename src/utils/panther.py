import os
import time
import torch
import paramiko
import numpy as np
import uuid
from shutil import copy2
from typing import Optional
from utils import console


USER = "finlay"
REMOTE_HOST = "129.173.66.44"
PORT = 22
P_REMOTE = "/home/finlay/disklavier/data/outputs/uploads"
tag = "[#5f00af]panthr[/#5f00af]:"


def send_embedding(
    file_path: str, model: str = "specdiff", mode: Optional[str] = None, verbose: bool = False
) -> np.ndarray:
    """
    Calculates the embedding of a MIDI file using the Panther model.

    Parameters
    ----------
    file_path : str
        The path to the MIDI file to calculate the embedding of.
    model : str
        The model to use for embedding calculation.

    Returns
    -------
    np.ndarray
        The embedding of the MIDI file.
    """
    console.log(
        f"{tag} sending embedding for '{file_path}' using model '{model}' and mode '{mode}'"
    )

    # fs setup
    local_folder = os.path.dirname(file_path)
    base_filename = os.path.basename(file_path)

    # add model name to the remote filename
    filename, ext = os.path.splitext(base_filename)
    random_id = str(uuid.uuid4())[:8]  # short 8-character random identifier
    model_filename = f"{filename}_{random_id}_{model}{ext}"
    remote_file_path = os.path.join(P_REMOTE, model_filename)
    pf_tensor_remote = os.path.splitext(remote_file_path)[0] + ".pt"
    pf_tensor_local = os.path.join(local_folder, os.path.basename(pf_tensor_remote))
    if verbose:
        console.log(
            f"{tag} using remote file path '{pf_tensor_remote}' and local file path '{pf_tensor_local}'"
        )

    # short circuit for testing
    # '/home/finlay/disklavier/data/outputs/uploads/intervals-060-09_1_t00s00_specdiff.pt
    #                          data/outputs/uploads/intervals-060-09_1_t00s00_specdiff.pt
    if mode == "test":
        if verbose:
            console.log(f"{tag} moving file from '{file_path}' to '{remote_file_path}'")
        if os.path.exists(pf_tensor_local):
            os.remove(pf_tensor_local)
        if os.path.exists(remote_file_path):
            os.remove(remote_file_path)
        copy2(file_path, remote_file_path)
        pf_tensor_local = os.path.abspath(remote_file_path).replace(".mid", ".pt")
        time.sleep(0.5)
    else:
        # panther login
        if verbose:
            console.log(f"{tag} connecting to panther at {USER}@{REMOTE_HOST}:{PORT}")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=REMOTE_HOST, port=PORT, username=USER)
        sftp = ssh.open_sftp()
        if verbose:
            console.log(f"{tag} connection opened")

        # clear old files
        try:
            sftp.stat(remote_file_path)
            sftp.remove(remote_file_path)
            sftp.remove(os.path.splitext(remote_file_path)[0] + ".pt")
            console.log(f"{tag} existing file '{remote_file_path}' deleted.")
        except FileNotFoundError:
            if verbose:
                console.log(
                    f"{tag} no file to delete at '{remote_file_path}', proceeding..."
                )
        # upload
        sftp.put(file_path, remote_file_path)
        if verbose:
            console.log(
                f"{tag} upload complete, waiting for embedding..."
            )

        # wait for new tensor
        while 1:
            try:
                sftp.stat(pf_tensor_remote)
                break
            except FileNotFoundError:
                time.sleep(0.01)

        sftp.get(pf_tensor_remote, pf_tensor_local)
        console.log(f"{tag} downloaded embedding file '{pf_tensor_local}'")

        sftp.close()
        ssh.close()

    embedding = torch.load(pf_tensor_local, weights_only=False).numpy().reshape(1, -1)
    if verbose:
        console.log(f"{tag} loaded embedding {embedding.shape}")

    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
