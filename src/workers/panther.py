import os
import time
import torch
import paramiko
import numpy as np
import uuid
from shutil import copy2
from typing import Optional
from utils import console
from workers.worker import Worker


class Panther(Worker):
    """
    worker for interacting with the panther embedding system via ssh.

    maintains an open ssh connection for faster file transfers and embedding retrieval.
    """

    tag = "[#5f00af]panthr[/#5f00af]:"
    user = "finlay"
    remote_host = "129.173.66.44"
    port = 22
    p_remote = "/home/finlay/disklavier/data/outputs/uploads"
    ssh: Optional[paramiko.SSHClient] = None
    sftp: Optional[paramiko.SFTPClient] = None

    def __init__(self, params, bpm: int):
        """
        initialize the panther connection manager.

        parameters
        ----------
        params : object
            configuration parameters object. expected to have attributes like
            `user`, `remote_host`, `port`, `remote_dir`, `verbose`, `tag`.
        bpm : int
            beats per minute (inherited from worker).
        """
        super().__init__(params, bpm=bpm)
        self.user = getattr(params, "user", self.user)
        self.remote_host = getattr(params, "remote_host", self.remote_host)
        self.port = getattr(params, "port", self.port)
        self.remote_dir = getattr(params, "remote_dir", self.p_remote)
        # override tag if provided in params
        self.tag = getattr(params, "tag", self.tag)
        self.ssh: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None
        # Allow configuring wait times via params
        self.remote_wait = getattr(params, "remote_wait", 2.0)
        self.local_wait = getattr(params, "local_wait", 5.0)
        self.check_interval = getattr(params, "check_interval", 0.01)
        self._connect()

        console.log(f"{self.tag} initialization complete")
        if self.verbose:
            console.log(f"{self.tag} settings:\n{self.__dict__}")

    def _connect(self):
        """establish ssh and sftp connections."""
        if self.ssh or self.sftp:  # prevent reconnecting if already connected
            if self.verbose:
                console.log(f"{self.tag} connection already established.")
            return
        if self.verbose:
            console.log(
                f"{self.tag} connecting to panther at {self.user}@{self.remote_host}:{self.port}"
            )
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                hostname=self.remote_host, port=self.port, username=self.user
            )
            self.sftp = self.ssh.open_sftp()
            if self.verbose:
                console.log(f"{self.tag} connection opened")
        except Exception as e:
            console.log(f"{self.tag} failed to connect: {e}")
            self.ssh = None
            self.sftp = None

    def close(self):
        """close the sftp and ssh connections."""
        if self.sftp:
            try:
                self.sftp.close()
                if self.verbose:
                    console.log(f"{self.tag} sftp connection closed")
            except Exception as e:
                console.log(f"{self.tag} error closing sftp: {e}")
            finally:
                self.sftp = None
        if self.ssh:
            try:
                self.ssh.close()
                if self.verbose:
                    console.log(f"{self.tag} ssh connection closed")
            except Exception as e:
                console.log(f"{self.tag} error closing ssh: {e}")
            finally:
                self.ssh = None

    def reset(self):
        """reset worker state and reconnect ssh."""
        # Note: Worker.reset() does generic attribute resetting.
        # We might not want that for ssh/sftp, so we handle connection manually.
        # super().reset() # Consider if base reset logic is needed/safe here
        # self.close()  # close existing connections if any
        # self._connect()  # re-establish connection
        pass

    def get_embedding(
        self,
        file_path: str,
        model: str = "specdiff",
        mode: Optional[str] = None,  # e.g., 'test'
    ) -> Optional[np.ndarray]:
        """
        calculates the embedding of a midi file using the panther system.

        parameters
        ----------
        file_path : str
            the path to the midi file to calculate the embedding of.
        model : str (default: "specdiff")
            the model identifier to use for embedding calculation.
        mode : optional[str] (default: none)
            operational mode, e.g., 'test'.

        returns
        -------
        optional[np.ndarray]
            the embedding of the midi file, or none if an error occurred.
        """
        console.log(
            f"{self.tag} getting embedding for '{file_path}' using model '{model}' and mode '{mode}'"
        )

        local_folder = os.path.dirname(file_path)
        base_filename = os.path.basename(file_path)
        filename, ext = os.path.splitext(base_filename)
        random_id = str(uuid.uuid4())[:8]
        model_filename = f"{filename}_{random_id}_{model}{ext}"
        remote_file_path = os.path.join(self.remote_dir, model_filename)
        pf_tensor_remote = os.path.splitext(remote_file_path)[0] + ".pt"
        pf_tensor_local = os.path.join(local_folder, os.path.basename(pf_tensor_remote))

        if self.verbose:
            console.log(
                f"{self.tag} remote: '{pf_tensor_remote}', local: '{pf_tensor_local}'"
            )

        embedding = None
        if mode == "test":
            embedding = self._handle_test_mode(
                file_path, remote_file_path, pf_tensor_local
            )
        else:
            embedding = self._handle_remote_mode(
                file_path, remote_file_path, pf_tensor_remote, pf_tensor_local
            )

        if embedding is not None:
            console.log(f"{self.tag} loaded embedding {embedding.shape}")
            # Optional: cleanup local tensor file
            # try:
            #     os.remove(pf_tensor_local)
            #     if self.verbose: console.log(f"{self.tag} cleaned up '{pf_tensor_local}'")
            # except OSError as e:
            #     console.log(f"{self.tag} failed to cleanup '{pf_tensor_local}': {e}")
        else:
            console.log(f"{self.tag} failed to get embedding for '{file_path}'")

        return embedding

    def _handle_test_mode(
        self, file_path: str, remote_file_path: str, pf_tensor_local: str
    ) -> Optional[np.ndarray]:
        """handle embedding generation in test mode (local file copy)."""
        if self.verbose:
            console.log(
                f"{self.tag} [test mode] copying '{file_path}' to '{remote_file_path}'"
            )
        try:
            # Ensure target directory exists (shutil.copy2 doesn't create dirs)
            os.makedirs(os.path.dirname(remote_file_path), exist_ok=True)
            # Clean up potential old files first
            pf_tensor_local_test = os.path.splitext(remote_file_path)[0] + ".pt"
            if os.path.exists(pf_tensor_local):
                os.remove(pf_tensor_local)
            if os.path.exists(pf_tensor_local_test):
                os.remove(pf_tensor_local_test)
            if os.path.exists(remote_file_path):
                os.remove(remote_file_path)

            copy2(file_path, remote_file_path)

            if self.verbose:
                console.log(
                    f"{self.tag} [test mode] waiting for local tensor '{pf_tensor_local_test}'"
                )

            if not self._wait_for_file(
                pf_tensor_local_test,
                max_wait=self.local_wait,
                check_interval=self.check_interval,
            ):
                console.log(
                    f"{self.tag} [test mode] file '{pf_tensor_local_test}' did not appear/stabilize"
                )
                # Cleanup copied file if tensor wasn't generated
                # if os.path.exists(remote_file_path): os.remove(remote_file_path)
                return None

            # Rename the generated .pt file to match the expected local name
            if pf_tensor_local_test != pf_tensor_local:
                os.rename(pf_tensor_local_test, pf_tensor_local)
                if self.verbose:
                    console.log(
                        f"{self.tag} [test mode] renamed '{pf_tensor_local_test}' to '{pf_tensor_local}'"
                    )
            # Cleanup the copied midi file
            # if os.path.exists(remote_file_path): os.remove(remote_file_path)

        except Exception as e:
            console.log(f"{self.tag} [test mode] error: {e}")
            return None

        return self._load_embedding(pf_tensor_local)

    def _handle_remote_mode(
        self,
        file_path: str,
        remote_file_path: str,
        pf_tensor_remote: str,
        pf_tensor_local: str,
    ) -> Optional[np.ndarray]:
        """handle embedding generation via remote ssh connection."""
        if not self.sftp or not self.ssh:
            console.log(f"{self.tag} ssh/sftp not connected. attempting reconnect.")
            self._connect()
            if not self.sftp or not self.ssh:
                console.log(f"{self.tag} reconnect failed. cannot proceed.")
                return None

        try:
            # clear old files on remote
            try:
                self.sftp.remove(remote_file_path)
                if self.verbose:
                    console.log(
                        f"{self.tag} removed existing remote '{remote_file_path}'"
                    )
            except FileNotFoundError:
                pass  # file didn't exist
            try:
                self.sftp.remove(pf_tensor_remote)
                if self.verbose:
                    console.log(
                        f"{self.tag} removed existing remote '{pf_tensor_remote}'"
                    )
            except FileNotFoundError:
                pass  # tensor file didn't exist

            # upload using self.sftp
            self.sftp.put(file_path, remote_file_path)
            if self.verbose:
                console.log(
                    f"{self.tag} uploaded '{file_path}', waiting for remote '{pf_tensor_remote}'..."
                )

            # wait for new tensor using self.sftp
            if not self._wait_for_remote_file(
                self.sftp,
                pf_tensor_remote,
                max_wait=self.remote_wait,
                check_interval=self.check_interval,
            ):
                console.log(
                    f"{self.tag} remote file '{pf_tensor_remote}' did not appear/stabilize."
                )
                # Cleanup uploaded file if tensor wasn't generated?
                # try: self.sftp.remove(remote_file_path) except Exception: pass
                return None

            # download using self.sftp
            # Clean local file first
            if os.path.exists(pf_tensor_local):
                os.remove(pf_tensor_local)
            self.sftp.get(pf_tensor_remote, pf_tensor_local)
            if self.verbose:
                console.log(
                    f"{self.tag} downloaded '{pf_tensor_remote}' to '{pf_tensor_local}'"
                )

            # Optional: cleanup remote files after successful download
            # try:
            #      self.sftp.remove(remote_file_path)
            #      self.sftp.remove(pf_tensor_remote)
            #      if self.verbose: console.log(f"{self.tag} cleaned up remote files")
            # except Exception as e:
            #      console.log(f"{self.tag} failed remote cleanup: {e}")

        except Exception as e:
            console.log(f"{self.tag} error during sftp operation: {e}")
            return None

        return self._load_embedding(pf_tensor_local)

    def _load_embedding(self, pf_tensor_local: str) -> Optional[np.ndarray]:
        """load the embedding tensor from a local file."""
        try:
            if self.verbose:
                console.log(f"{self.tag} loading embedding from '{pf_tensor_local}'")
            # Ensure file exists and is not empty before loading
            if (
                not os.path.exists(pf_tensor_local)
                or os.path.getsize(pf_tensor_local) == 0
            ):
                console.log(
                    f"{self.tag} error: local embedding file '{pf_tensor_local}' is missing or empty."
                )
                return None

            embedding = (
                torch.load(pf_tensor_local, weights_only=False).numpy().reshape(1, -1)
            )
            return embedding
        except FileNotFoundError:
            console.log(
                f"{self.tag} error: local embedding file '{pf_tensor_local}' not found during load."
            )
            return None
        except Exception as e:
            console.log(
                f"{self.tag} error loading embedding from '{pf_tensor_local}': {e}"
            )
            return None

    # Make wait_for_file a private method
    def _wait_for_file(
        self,
        file_path: str,
        max_wait: float = 5.0,
        check_interval: float = 0.01,
    ) -> bool:
        """
        wait for a local file to appear and stabilize.

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
            true if file appeared and size stabilized > 0.
        """
        start_time = time.time()
        prev_size = -1
        file_appeared = False

        while time.time() - start_time < max_wait:
            if not os.path.exists(file_path):
                time.sleep(check_interval)
                continue

            file_appeared = True  # Mark that we've seen the file at least once
            try:
                current_size = os.path.getsize(file_path)
                # Check if size is stable and greater than 0
                if current_size > 0 and current_size == prev_size:
                    if self.verbose:
                        console.log(
                            f"{self.tag} local file '{file_path}' stabilized at size {current_size}."
                        )
                    return True
                prev_size = current_size
            except OSError as e:  # Handle potential race condition or permission error
                console.log(f"{self.tag} error checking local file '{file_path}': {e}")
                prev_size = -1  # Reset prev_size on error
            time.sleep(check_interval)

        # Final check after loop: if file appeared, check its final state
        if file_appeared:
            try:
                final_size = os.path.getsize(file_path)
                is_stable = final_size > 0 and final_size == prev_size
                if self.verbose:
                    console.log(
                        f"{self.tag} final check for local file '{file_path}': size {final_size}, stable: {is_stable} (prev_size: {prev_size})"
                    )
                return is_stable
            except OSError:
                return False  # Error during final check
        else:
            if self.verbose:
                console.log(f"{self.tag} local file '{file_path}' never appeared.")
            return False  # File never appeared

    # Make wait_for_remote_file a private method
    def _wait_for_remote_file(
        self,
        sftp: paramiko.SFTPClient,
        remote_file_path: str,
        max_wait: float = 2.0,
        check_interval: float = 0.01,
    ) -> bool:
        """
        wait for a remote file to appear and stabilize using sftp.

        parameters
        ----------
        sftp : paramiko.SFTPClient
            active sftp client session.
        remote_file_path : str
            path to the remote file to wait for.
        max_wait : float (default: 2.0)
            maximum time to wait in seconds.
        check_interval : float (default: 0.01)
            time between file checks.

        returns
        -------
        bool
            true if file appeared and size stabilized > 0.
        """
        start_time = time.time()
        prev_size = -1
        file_appeared = False

        while time.time() - start_time < max_wait:
            try:
                file_attr = sftp.stat(remote_file_path)
                current_size = file_attr.st_size
                file_appeared = True  # Mark that we've seen the file

                # Check stability: size > 0 and unchanged since previous check
                if (
                    current_size is not None
                    and current_size > 0
                    and current_size == prev_size
                ):
                    if self.verbose:
                        console.log(
                            f"{self.tag} remote file '{remote_file_path}' size {current_size} stabilized."
                        )
                    return True

                prev_size = current_size  # Update previous size for next iteration
                time.sleep(check_interval)

            except FileNotFoundError:
                # File doesn't exist (yet or anymore). Reset prev_size.
                prev_size = -1
                time.sleep(check_interval)
                continue
            except Exception as e:
                console.log(
                    f"{self.tag} error checking remote file '{remote_file_path}': {e}"
                )
                prev_size = -1  # Reset prev_size on error
                time.sleep(check_interval * 5)  # Wait a bit longer on error

        # Final check after timeout
        if file_appeared:
            try:
                final_attr = sftp.stat(remote_file_path)
                is_stable = (
                    final_attr.st_size is not None
                    and final_attr.st_size > 0
                    and final_attr.st_size
                    == prev_size  # Compare with last successful size check
                )
                if self.verbose:
                    status = "stable" if is_stable else "not stable"
                    console.log(
                        f"{self.tag} final check for '{remote_file_path}': size {final_attr.st_size}, status: {status} (prev_size: {prev_size})"
                    )
                return is_stable
            except (FileNotFoundError, Exception) as e:
                if self.verbose:
                    console.log(
                        f"{self.tag} final check failed for '{remote_file_path}': {e}"
                    )
                return False
        else:
            if self.verbose:
                console.log(
                    f"{self.tag} remote file '{remote_file_path}' never appeared."
                )
            return False
