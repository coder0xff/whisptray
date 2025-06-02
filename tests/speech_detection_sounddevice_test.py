#!/usr/bin/env python3
"""Test speech detection using AudioCapture module."""

import sys
import os
import logging
import time
import numpy as np
import sounddevice as sd

# Add parent directory to path to import whisptray modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from whisptray.audio_capture import AudioCapture, SAMPLE_RATE, DTYPE, CHANNELS

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S.%f'[:-3]  # Include milliseconds
)

def main():
    all_audio_chunks = []

    def on_audio(audio_data):
        logging.debug(f"Got data.")
        nonlocal all_audio_chunks
        all_audio_chunks.append(audio_data)

    capture = AudioCapture(on_audio=on_audio)

    capture.start()
    time.sleep(10)
    capture.stop()

    full_audio = np.concatenate(all_audio_chunks)
    with sd.OutputStream(samplerate=SAMPLE_RATE, dtype=DTYPE, channels=CHANNELS) as stream:
        stream.write(full_audio)


if __name__ == "__main__":
    main()