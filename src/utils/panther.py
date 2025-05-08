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
tag = "[#af5fff]panthr[/#af5fff]:"


def send_embedding(
    file_path: str,
    model: str = "specdiff",
    mode: Optional[str] = None,
    verbose: bool = False,
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
        if verbose:
            console.log(
                f"{tag} {os.path.exists(remote_file_path)} moved file from to '{remote_file_path}'"
            )

        if not wait_for_file(pf_tensor_local):
            raise ValueError(f"file '{pf_tensor_local}' is not valid, exiting")
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
            console.log(f"{tag} upload complete, waiting for embedding...")

        # wait for new tensor
        if not wait_for_remote_file(sftp, pf_tensor_remote):
            raise TimeoutError(
                f"{tag} remote embedding file '{pf_tensor_remote}' did not appear or stabilize within 2 seconds."
            )

        sftp.get(pf_tensor_remote, pf_tensor_local)
        if verbose:
            console.log(f"{tag} downloaded embedding file '{pf_tensor_local}'")

        sftp.close()
        ssh.close()

    if verbose:
        console.log(f"{tag} loading embedding from '{pf_tensor_local}'")
    embedding = torch.load(pf_tensor_local, weights_only=False).numpy().reshape(1, -1)

    console.log(f"{tag} loaded embedding {embedding.shape}")

    return embedding


def wait_for_file(
    file_path: str, max_wait: float = 5.0, check_interval: float = 0.01
) -> bool:
    """
    wait for a file to be completely written.

    parameters
    ----------
    file_path : str
        path to file to wait for.
    max_wait : float (default: 5.0)
        maximum time to wait in seconds.
    check_interval : float (default: 0.01)
        time between file size checks.

    returns
    -------
    bool
        true if file is valid and ready.
    """
    start_time = time.time()
    prev_size = -1

    while time.time() - start_time < max_wait:
        if not os.path.exists(file_path):
            time.sleep(check_interval)
            continue

        current_size = os.path.getsize(file_path)
        if current_size > 0 and current_size == prev_size:
            # File size hasn't changed since last check, probably done writing
            return True

        prev_size = current_size
        time.sleep(check_interval)

    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def wait_for_remote_file(
    sftp: paramiko.SFTPClient,
    remote_file_path: str,
    max_wait: float = 2.0,
    check_interval: float = 0.01,
) -> bool:
    """
    wait for a remote file to be completely written using sftp.

    parameters
    ----------
    sftp : paramiko.SFTPClient
        active sftp client session.
    remote_file_path : str
        path to the remote file to wait for.
    max_wait : float (default: 5.0)
        maximum time to wait in seconds.
    check_interval : float (default: 0.01)
        time between file checks.

    returns
    -------
    bool
        true if file exists and its size is stable.
    """
    start_time = time.time()
    prev_size = -1
    file_exists = False

    while time.time() - start_time < max_wait:
        try:
            file_attr = sftp.stat(remote_file_path)
            current_size = file_attr.st_size
            file_exists = True

            if (
                current_size is not None
                and current_size > 0
                and current_size == prev_size
            ):
                # file size hasn't changed, assume complete
                return True

            prev_size = current_size
            time.sleep(check_interval)

        except FileNotFoundError:
            # file doesn't exist yet
            file_exists = False
            time.sleep(check_interval)
            continue

    # final check after timeout
    if file_exists:
        try:
            final_attr = sftp.stat(remote_file_path)
            return (
                final_attr.st_size is not None
                and final_attr.st_size > 0
                and final_attr.st_size == prev_size
            )
        except FileNotFoundError:
            return False
    else:
        return False
