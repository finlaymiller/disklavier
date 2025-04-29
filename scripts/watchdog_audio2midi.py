import os
import time
import argparse
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    # these imports are expected to work only in the 'bp' conda environment
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    # Assuming basic_pitch.inference.predict returns midi_data with a .write method
except ImportError:
    logging.error(
        "failed to import basic-pitch. ensure script is run in the correct conda environment ('bp') with basic-pitch installed."
    )
    exit(1)  # exit if basic-pitch is not available


class BasicPitchHandler(FileSystemEventHandler):
    """handles new audio files created in the watched directory."""

    def __init__(self, model, midi_out_dir):
        self.model = model
        self.midi_out_dir = midi_out_dir
        os.makedirs(self.midi_out_dir, exist_ok=True)  # ensure output dir exists

    def on_created(self, event):
        """called when a file or directory is created."""
        if event.is_directory:
            return

        if not event.src_path.lower().endswith(".wav"):
            # logging.debug(f"ignoring non-wav file: {event.src_path}")
            return

        logging.info(f"detected new audio file: {event.src_path}")

        # construct the output midi path based on the input filename
        base_filename = os.path.splitext(os.path.basename(event.src_path))[0]
        output_midi_filename = f"{base_filename}.mid"
        output_midi_path = os.path.join(self.midi_out_dir, output_midi_filename)

        try:
            logging.info(f"processing {event.src_path} with basic-pitch...")
            # perform basic-pitch inference
            # model_output, midi_data, note_events = predict(event.src_path, self.model) # Original plan structure
            # using predict_and_save might be simpler if available and matches requirements,
            # but sticking to plan's predict() and manual save:
            model_output, midi_data, note_events = predict(event.src_path, self.model)

            if midi_data:
                # save the midi file
                midi_data.write(output_midi_path)
                logging.info(f"successfully created midi file: {output_midi_path}")

                # optionally remove the original audio file
                try:
                    os.remove(event.src_path)
                    logging.info(f"removed processed audio file: {event.src_path}")
                except OSError as e:
                    logging.error(f"error removing audio file {event.src_path}: {e}")
            else:
                logging.error(
                    f"basic-pitch processing failed for {event.src_path}, no midi data generated."
                )

        except Exception as e:
            logging.error(
                f"error processing file {event.src_path}: {e}", exc_info=True
            )  # include stack trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="watch a directory for audio files and convert them to midi using basic-pitch."
    )
    parser.add_argument(
        "--audio-in", required=True, help="directory to watch for incoming .wav files."
    )
    parser.add_argument(
        "--midi-out", required=True, help="directory to save outgoing .mid files."
    )
    args = parser.parse_args()

    logging.info("loading basic-pitch model...")
    try:
        # load the model (using the standardICASSP 2022 model path)
        model = ICASSP_2022_MODEL_PATH  # Or load a specific checkpoint
        # Note: basic_pitch might lazy-load the actual model on first predict call.
        # If direct loading is needed:
        # from tensorflow import saved_model
        # model = saved_model.load(str(ICASSP_2022_MODEL_PATH))
        logging.info("basic-pitch model ready.")
    except Exception as e:
        logging.error(f"failed to load basic-pitch model: {e}", exc_info=True)
        exit(1)

    event_handler = BasicPitchHandler(model=model, midi_out_dir=args.midi_out)
    observer = Observer()
    observer.schedule(event_handler, path=args.audio_in, recursive=False)

    logging.info(f"starting watchdog observer on directory: {args.audio_in}")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("received keyboard interrupt, stopping observer...")
        observer.stop()
    except Exception as e:
        logging.error(f"an unexpected error occurred: {e}", exc_info=True)
        observer.stop()

    observer.join()
    logging.info("watchdog observer stopped.")
