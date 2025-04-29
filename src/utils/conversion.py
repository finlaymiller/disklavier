import os
import shutil
import time
import uuid
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def request_basic_pitch_conversion(
    input_audio_path: str, config: dict, timeout: int = 30
) -> Optional[str]:
    """
    requests midi conversion by placing audio in a watched directory and waiting for the result.

    copies the input audio file to the basic pitch input directory with a unique id.
    polls the basic pitch output directory for the corresponding midi file.

    parameters
    ----------
    input_audio_path : str
        path to the input audio file (.wav).
    config : dict
        configuration dictionary containing 'basic_pitch' settings.
        expects config['basic_pitch']['audio_in_dir'] and config['basic_pitch']['midi_out_dir'].
    timeout : int, optional
        time in seconds to wait for the midi file, by default 30.

    returns
    -------
    Optional[str]
        path to the generated midi file if successful within the timeout, otherwise none.
    """
    if not os.path.exists(input_audio_path):
        logging.error(f"input audio file not found: {input_audio_path}")
        return None

    if (
        "basic_pitch" not in config
        or "audio_in_dir" not in config["basic_pitch"]
        or "midi_out_dir" not in config["basic_pitch"]
    ):
        logging.error("basic pitch directories not configured in params file.")
        return None

    audio_in_dir = config["basic_pitch"]["audio_in_dir"]
    midi_out_dir = config["basic_pitch"]["midi_out_dir"]

    # ensure watch directories exist
    os.makedirs(audio_in_dir, exist_ok=True)
    os.makedirs(midi_out_dir, exist_ok=True)

    task_id = str(uuid.uuid4())
    base_filename = os.path.splitext(os.path.basename(input_audio_path))[0]

    watch_audio_filename = f"{base_filename}_{task_id}.wav"
    watch_midi_filename = f"{base_filename}_{task_id}.mid"

    watch_audio_path = os.path.join(audio_in_dir, watch_audio_filename)
    watch_midi_path = os.path.join(midi_out_dir, watch_midi_filename)

    try:
        shutil.copy(input_audio_path, watch_audio_path)
        logging.info(f"copied audio file to watch directory: {watch_audio_path}")
    except Exception as e:
        logging.error(f"failed to copy audio file to watch directory: {e}")
        return None

    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(watch_midi_path):
            logging.info(f"midi file found: {watch_midi_path}")
            # plan mentions optional move, returning path for now
            return watch_midi_path
        # logging.debug(f"waiting for midi file: {watch_midi_path}") # potentially too verbose
        time.sleep(0.1)

    logging.warning(
        f"basic pitch conversion timed out after {timeout} seconds for task {task_id}."
    )
    # cleanup the input file if timeout occurs? plan doesn't specify, leaving it for now.
    # if os.path.exists(watch_audio_path):
    #     os.remove(watch_audio_path)
    return None
