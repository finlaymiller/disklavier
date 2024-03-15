import os
from pathlib import Path
from mido import MidiFile, MetaMessage, second2tick
import mido
import pretty_midi
import numpy as np

from utils.midi import set_tempo, semitone_transpose



def segment_midi(input_file_path: str, params):
    """do the segmentation"""
    target_tempo = int(os.path.basename(input_file_path).split("-")[1])
    set_tempo(input_file_path, target_tempo)

    # remove "-t" from filename
    filename = Path(input_file_path).stem
    filename_components = filename.split("-")
    filename = f"{filename_components[0]}-{int(np.round(float(filename_components[1]))):03d}-{filename_components[2]}"

    # calculate timings
    midi_pm = pretty_midi.PrettyMIDI(input_file_path)
    total_length = midi_pm.get_end_time()
    segment_length = 60 * params.n__num_beats / target_tempo  # in seconds
    num_segments_float = total_length / segment_length
    num_segments = int(np.round(num_segments_float))

    # print([total_length, segment_length, num_segments_float, num_segments, init_bpm])

    # print(
    #     f"breaking '{filename}' ({total_length:.03f} s at {target_tempo} bpm) into {num_segments:03d} segments of {segment_length:.03f} s"
    # )

    new_files = 0
    for n in list(range(num_segments)):
        start = n * segment_length
        end = start + segment_length
        # print(f"\tsplitting from {start:07.03f} s to {end:07.03f} s")
        segment_midi = pretty_midi.PrettyMIDI(initial_tempo=target_tempo)
        instrument = pretty_midi.Instrument(
            program=midi_pm.instruments[0].program,
            name=f"{filename}_{int(start):04d}-{int(end):04d}",
        )

        # add notes from the original MIDI that fall within the current segment
        for note in midi_pm.instruments[0].notes:
            if start <= note.start < end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start,
                    end=min(note.end, end) - start,
                )
                instrument.notes.append(new_note)

        # write out
        segment_midi.instruments.append(instrument)
        segment_filename = os.path.join(
            params.output_dir, f"{filename}_{int(start):04d}-{int(end):04d}_n00.mid"
        )
        segment_midi.write(segment_filename)

        # semitone shift
        if params.do_shift > 1:
            tpose_files = semitone_transpose(segment_filename, params.output_dir, params.do_shift)
            new_files += len(tpose_files)
        else:
            new_files += 1

        for filepath in tpose_files:
            # set tempo properly
            if params.strip_tempo:
                midi_md = MidiFile(filepath)
                for track in midi_md.tracks:
                    for message in track:
                        if message.type == "set_tempo":
                            track.remove(message)
                            track.append(
                                MetaMessage(
                                    "set_tempo", tempo=mido.bpm2tempo(target_tempo), time=0
                                )
                            )
                os.remove(filepath)
                midi_md.save(filepath)
            else:
                set_tempo(filepath, target_tempo)

            # make sure track end is correct
            modify_end_of_track(filepath, segment_length, target_tempo)
            test_mid = MidiFile(filepath)
            # if np.round(test_mid.length, 3) != np.round(segment_length, 3):
            #     print(
            #         f"'{os.path.basename(filepath)}' is {test_mid.length:.3f} s at {target_tempo} BPM but should be {segment_length:.3f} s ({test_mid.ticks_per_beat} tpb)"
            #     )
                # test_mid.print_tracks()

    return new_files


def modify_end_of_track(midi_file_path, new_end_time, tempo):
    mid = MidiFile(midi_file_path)
    total_time_t = -1
    new_e_time_t = second2tick(new_end_time, 220, mido.bpm2tempo(tempo))
    # print(f"{midi_file_path} at {mido.bpm2tempo(tempo)}")
    # mid.print_tracks()

    for track in mid.tracks:
        # Remove existing 'end_of_track' messages and calculate last note time
        for msg in track:
            total_time_t += msg.time
            if msg.type == "end_of_track":
                track.remove(msg)
                # Add a new 'end_of_track' message at the calculated offset time
                offset = (
                    new_e_time_t - total_time_t if new_e_time_t > total_time_t else 0
                )
                # print(
                #     f"modifying track to have end time {new_e_time_t}: {total_time_t} -> {offset}"
                # )
                track.append(MetaMessage("end_of_track", time=offset))

    # Save the modified MIDI file
    os.remove(midi_file_path)
    mid.save(midi_file_path)
