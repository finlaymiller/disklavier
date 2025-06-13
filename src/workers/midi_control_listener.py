import mido
import time
import threading
import numpy as np

from utils import console, midi


class MidiControlListener:
    """
    listens for midi control changes and note events to control application parameters.
    """

    tag = "[#00AADD]midicc[/#00AADD]:"

    running = False
    cc_thread = None
    cc_input_port = None
    transpose_thread = None
    transpose_input_port = None

    def __init__(
        self,
        params,
        runner_ref,
        player_ref,
    ):
        """
        initialize the midi control listener.

        parameters
        ----------
        params : OmegaConf
            the main application configuration object.
        runner_ref : Runner
            a reference to the main runner instance.
        player_ref : Player
            a reference to the player instance.
        """
        self.params = params
        self.runner_ref = runner_ref
        self.player_ref = player_ref

        # --- print configs ---
        console.log(self.params)
        if self.params.cc_listener.enable:
            console.log(
                f"{self.tag} CC listener config:\n\t\tPort='{self.params.cc_listener.port_name}',\tCC#={self.params.cc_listener.cc_number},\n\t\tMin={self.params.cc_listener.min},\tMax={self.params.cc_listener.max},\tNormalize={self.params.cc_listener.normalize}"
            )
        else:
            console.log(f"{self.tag} CC listener is disabled.")

        if self.params.transpose_listener.enable:
            console.log(
                f"{self.tag} Transpose Listener Config:\n\t\tPort='{self.params.transpose_listener.port_name}',\tMiddleC={self.params.transpose_listener.middle_c_note_number}"
            )
        else:
            console.log(f"{self.tag} Transpose Listener is disabled")

    def start(self):
        """
        start the midi listening threads if they are enabled.
        """
        self.running = True
        if self.params.cc_listener.enable:
            if not self.cc_thread or not self.cc_thread.is_alive():
                self.cc_thread = threading.Thread(
                    target=self._listen_cc_input, daemon=True
                )
                self.cc_thread.start()
        else:
            console.log(
                f"{self.tag} CC listener not starting as it is disabled in config."
            )

        if self.params.transpose_listener.enable:
            if not self.transpose_thread or not self.transpose_thread.is_alive():
                self.transpose_thread = threading.Thread(
                    target=self._listen_transpose_input, daemon=True
                )
                self.transpose_thread.start()
        else:
            console.log(
                f"{self.tag} Transpose listener not starting as it is disabled in config."
            )

    def stop(self):
        """
        stop the midi listening threads and release resources.
        """
        self.running = False
        if self.cc_thread and self.cc_thread.is_alive():
            self.cc_thread.join(timeout=1.0)
            if self.cc_thread.is_alive():
                console.log(
                    f"{self.tag} [yellow]cc listener thread did not stop in time.[/yellow]"
                )

        if self.cc_input_port:
            if not self.cc_input_port.closed:
                self.cc_input_port.close()
                console.log(
                    f"{self.tag} closed cc midi port: {self.params.cc_listener.port_name}"
                )
            self.cc_input_port = None

        if self.transpose_thread and self.transpose_thread.is_alive():
            self.transpose_thread.join(timeout=1.0)
            if self.transpose_thread.is_alive():
                console.log(
                    f"{self.tag} [yellow]transpose listener thread did not stop in time.[/yellow]"
                )

        if self.transpose_input_port:
            if not self.transpose_input_port.closed:
                self.transpose_input_port.close()
                console.log(
                    f"{self.tag} closed transpose midi port: {self.params.transpose_listener.port_name}"
                )
            self.transpose_input_port = None

    def _listen_cc_input(self):
        """
        continuously listen for control change messages on the specified midi port.
        this method is intended to be run in a separate thread.
        """
        # --- init connection ---
        if not self.params.cc_listener.port_name:
            console.log(
                f"{self.tag} [orange bold]cc port name not configured. cc listener will not run.[/orange bold]"
            )
            return
        try:
            self.cc_input_port = mido.open_input(self.params.cc_listener.port_name)  # type: ignore
            console.log(
                f"{self.tag} listening for cc#{self.params.cc_listener.cc_number} on '{self.params.cc_listener.port_name}'"
            )
        except Exception as e:
            console.log(
                f"{self.tag} [red]failed to open cc midi port '{self.params.cc_listener.port_name}': {e}. cc listener will not start.[/red]"
            )
            self.cc_input_port = None
            return

        while self.running:
            try:
                # non-blocking check for messages
                for msg in self.cc_input_port.iter_pending():
                    if (
                        msg.type == "control_change"
                        and msg.control == self.params.cc_listener.cc_number
                    ):
                        mapped_value = msg.value / 127
                        new_threshold = (
                            self.params.cc_listener.min
                            + (
                                self.params.cc_listener.max
                                - self.params.cc_listener.min
                            )
                            * mapped_value
                        )
                        new_threshold = round(new_threshold, 2)

                        if self.params.cc_listener.normalize:
                            new_threshold /= self.params.cc_listener.max
                        self.runner_ref.update_duet_sensitivity(new_threshold)

                        console.log(
                            f"{self.tag} player_embedding_diff_threshold set to {new_threshold} (cc value: {msg.value})"
                        )

                time.sleep(0.01)
            except Exception as e:
                if self.running:  # only log if we weren't intending to stop
                    console.log(f"{self.tag} [red]error in cc listener loop: {e}[/red]")

                if (
                    isinstance(e, (IOError, OSError))
                    and self.cc_input_port
                    and self.cc_input_port.closed
                ):
                    console.log(
                        f"{self.tag} [red]cc midi port '{self.params.cc_listener.port_name}' appears closed. stopping listener.[/red]"
                    )
                    break
                time.sleep(0.1)

        if self.cc_input_port and not self.cc_input_port.closed:
            self.cc_input_port.close()
        console.log(
            f"{self.tag} stopped listening on cc port '{self.params.cc_listener.port_name}'"
        )

    def _listen_transpose_input(self):
        """
        continuously listen for note_on messages on the transpose midi port.
        this method is intended to be run in a separate thread.
        """
        # --- init connection ---
        if not self.params.transpose_listener.port_name:
            console.log(
                f"{self.tag} [yellow]transpose port name not configured. transpose listener will not run.[/yellow]"
            )
            return
        if not hasattr(self.player_ref, "set_transposition"):
            console.log(
                f"{self.tag} [red]player_ref does not have set_transposition method. transpose listener will not start.[/red]"
            )
            return
        try:
            self.transpose_input_port = mido.open_input(self.params.transpose_listener.port_name)  # type: ignore
            console.log(
                f"{self.tag} listening for transpose notes on '{self.params.transpose_listener.port_name}'"
            )
        except Exception as e:
            console.log(
                f"{self.tag} [red]failed to open transpose midi port '{self.params.transpose_listener.port_name}': {e}. transpose listener will not start.[/red]"
            )
            self.transpose_input_port = None
            return

        while self.running:
            if self.transpose_input_port is None:
                break
            try:
                for msg in self.transpose_input_port.iter_pending():
                    if msg.type == "note_on" and msg.velocity > 0:
                        key_pressed = msg.note
                        transposition_interval = (
                            key_pressed
                            - self.params.transpose_listener.middle_c_note_number
                        )
                        self.player_ref.set_transposition(transposition_interval)

                time.sleep(0.01)
            except Exception as e:
                if self.running:
                    console.log(
                        f"{self.tag} [red]error in transpose listener loop: {e}[/red]"
                    )
                if (
                    isinstance(e, (IOError, OSError))
                    and self.transpose_input_port
                    and self.transpose_input_port.closed
                ):
                    console.log(
                        f"{self.tag} [red]transpose midi port '{self.params.transpose_listener.port_name}' appears closed. stopping listener.[/red]"
                    )
                    break
                time.sleep(0.1)

        if self.transpose_input_port and not self.transpose_input_port.closed:
            self.transpose_input_port.close()
        console.log(
            f"{self.tag} stopped listening on transpose port '{self.params.transpose_listener.port_name}'"
        )

    def _listen_chord_input(self):
        """
        continuously listen for note_on messages on the transpose midi port.
        this method is intended to be run in a separate thread.
        """
        current_hist = np.zeros(128)

        # --- init connection ---
        if not self.params.chord_listener.port_name:
            console.log(
                f"{self.tag} [yellow]chord port name not configured. chord listener will not run.[/yellow]"
            )
            return
        if not hasattr(self.player_ref, "set_transposition"):
            console.log(
                f"{self.tag} [red]player_ref does not have set_transposition method. chord listener will not start.[/red]"
            )
            return
        try:
            self.chord_input_port = mido.open_input(self.params.chord_listener.port_name)  # type: ignore
            console.log(
                f"{self.tag} listening for chord notes on '{self.params.chord_listener.port_name}'"
            )
        except Exception as e:
            console.log(
                f"{self.tag} [red]failed to open chord midi port '{self.params.chord_listener.port_name}': {e}. chord listener will not start.[/red]"
            )
            self.chord_input_port = None
            return

        # --- get current histogram ---
        current_file = self.runner_ref.staff.scheduler.get_current_file()
        if current_file is None:
            console.log(
                f"{self.tag} [red]no current file. chord listener will not start.[/red]"
            )
            return
        console.log(f"{self.tag} current file: {current_file}")
        reference_hist = midi.get_pitch_histogram(current_file, False)
        console.log(f"{self.tag} reference histogram: {reference_hist}")

        # --- listen for chord notes ---
        while self.running:
            if self.chord_input_port is None:
                break
            try:
                for msg in self.chord_input_port.iter_pending():
                    if msg.type == "note_on" and msg.velocity > 0:
                        # add note to active notes
                        if not hasattr(self, "_active_notes"):
                            self._active_notes = set()
                        self._active_notes.add(msg.note)
                    elif msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                    ):
                        # remove note from active notes
                        if hasattr(self, "_active_notes"):
                            self._active_notes.discard(msg.note)

                    # generate histogram from active notes
                    if hasattr(self, "_active_notes") and self._active_notes:
                        # create a 128-element array (one for each midi note)
                        current_hist = np.zeros(128)
                        # set the active notes to 1
                        for note in self._active_notes:
                            current_hist[note] = 1
                        # normalize the histogram
                        if np.sum(current_hist) > 0:
                            current_hist = current_hist / np.sum(current_hist)

                    # find best transposition using kl divergence
                    min_kl = float("inf")
                    best_offset = 0
                    possible_offsets = range(-24, 25)  # +/- 2 octaves

                    for offset in possible_offsets:
                        shifted_hist = midi.shift_histogram(current_hist, offset)
                        kl_div = midi.calculate_kl_divergence(
                            shifted_hist, reference_hist
                        )
                        if kl_div < min_kl:
                            min_kl = kl_div
                            best_offset = offset

                    console.log(
                        f"{self.tag} found optimal transposition: {best_offset}"
                    )
                    self.player_ref.set_transposition(best_offset)

                time.sleep(0.01)
            except Exception as e:
                if self.running:
                    console.log(
                        f"{self.tag} [red]error in chord listener loop: {e}[/red]"
                    )
                if (
                    isinstance(e, (IOError, OSError))
                    and self.chord_input_port
                    and self.chord_input_port.closed
                ):
                    console.log(
                        f"{self.tag} [red]chord midi port '{self.params.chord_listener.port_name}' appears closed. stopping listener.[/red]"
                    )
                    break
                time.sleep(0.1)

        if self.chord_input_port and not self.chord_input_port.closed:
            self.chord_input_port.close()
        console.log(
            f"{self.tag} stopped listening on chord port '{self.params.chord_listener.port_name}'"
        )
