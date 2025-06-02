#!/usr/bin/env python3
"""
A minimal test script to record and playback audio block by block
using sounddevice InputStream and OutputStream.
"""

import sounddevice as sd
import numpy as np
import time
from queue import Queue

# Audio parameters
SAMPLE_RATE = 14400
CHANNELS = 1
DTYPE = 'int16'
BLOCK_DURATION_SECONDS = 0.03
BLOCK_SIZE = int(BLOCK_DURATION_SECONDS * SAMPLE_RATE)
RECORD_DURATION_SECONDS = 5

recorded_blocks_q = Queue()

def input_stream_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
    recorded_blocks_q.put(indata.copy())

def main():
    all_recorded_data = []

    # --- Recording Phase ---
    print(f"Recording for {RECORD_DURATION_SECONDS} seconds...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_SIZE,
        callback=input_stream_callback
    ):
        time.sleep(RECORD_DURATION_SECONDS) # Record for the duration
    print("Recording finished.")

    while not recorded_blocks_q.empty():
        all_recorded_data.append(recorded_blocks_q.get_nowait())

    if not all_recorded_data:
        print("No audio blocks were recorded.")
        return

    full_audio_recording = np.concatenate(all_recorded_data)
    print(f"Concatenated {len(all_recorded_data)} blocks. Total samples: {full_audio_recording.shape[0] if full_audio_recording.ndim > 0 else 0}")


    # --- Playback Phase ---
    if full_audio_recording.size > 0:
        print(f"Playing back {len(all_recorded_data)} recorded blocks...")
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE # Using the same blocksize for output for simplicity
        ) as stream:
            stream.write(full_audio_recording)
        print("Playback finished.")
    else:
        print("No audio data to play back (array is empty).")

    print("Simple stream audio test completed.")

if __name__ == "__main__":
    main() 