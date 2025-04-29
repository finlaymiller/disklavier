import argparse
import sys
import os
import fluidsynth


def synthesize_midi(midi_file, soundfont_file, output_wav_file, sample_rate=44100):
    """
    synthesizes audio from a MIDI file using FluidSynth.

    parameters
    ----------
    midi_file : str
        path to the input MIDI file.
    soundfont_file : str
        path to the SoundFont (.sf2) file.
    output_wav_file : str
        path to save the synthesized WAV audio file.
    sample_rate : int, optional
        the sample rate for the output audio (default is 44100).

    returns
    -------
    none
    """
    if not os.path.exists(midi_file):
        print(f"error: midi file not found at {midi_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(soundfont_file):
        print(f"error: soundfont file not found at {soundfont_file}", file=sys.stderr)
        sys.exit(1)

    print(f"initializing fluidsynth with sample rate {sample_rate}...")
    fs = fluidsynth.Synth(samplerate=float(sample_rate))

    print(f"loading soundfont: {soundfont_file}...")
    sfid = fs.sfload(soundfont_file)
    if sfid == fluidsynth.FLUID_FAILED:
        print(
            f"error: failed to load soundfont file: {soundfont_file}", file=sys.stderr
        )
        fs.delete()
        sys.exit(1)

    print(f"synthesizing midi: {midi_file} to {output_wav_file}...")
    try:
        # pyfluidsynth > 1.3.0 has midi_to_audio directly
        fs.midi_to_audio(midi_file, output_wav_file)
        print("synthesis complete.")
    except AttributeError:
        # fallback for older pyfluidsynth versions or different approach
        print(
            "note: direct midi_to_audio not available. using track-by-track synthesis (may be less accurate)."
        )
        # this part requires a more complex implementation involving reading midi messages
        # and manually triggering notes, which is non-trivial.
        # for simplicity, we'll just error out if the direct method isn't present.
        print(
            "error: this script requires pyfluidsynth > 1.3.0 for midi_to_audio.",
            file=sys.stderr,
        )
        fs.sfunload(sfid)
        fs.delete()
        sys.exit(1)
    except Exception as e:
        print(f"error during synthesis: {e}", file=sys.stderr)
        fs.sfunload(sfid)
        fs.delete()
        sys.exit(1)

    print(f"cleaning up fluidsynth...")
    fs.sfunload(sfid)
    fs.delete()
    print(f"output saved to: {output_wav_file}")


def main():
    """main function to parse arguments and call the synthesizer."""
    parser = argparse.ArgumentParser(
        description="synthesize audio from a MIDI file using FluidSynth."
    )
    parser.add_argument("midi_file", help="path to the input MIDI file (.mid)")
    parser.add_argument("soundfont_file", help="path to the SoundFont file (.sf2)")
    parser.add_argument(
        "output_wav_file", help="path to save the output WAV file (.wav)"
    )
    parser.add_argument(
        "-sr",
        "--sample_rate",
        type=int,
        default=44100,
        help="sample rate for the output audio (default: 44100)",
    )

    args = parser.parse_args()

    synthesize_midi(
        args.midi_file, args.soundfont_file, args.output_wav_file, args.sample_rate
    )


if __name__ == "__main__":
    main()
