#!/usr/bin/env python3
"""Test speech detection using sounddevice with improved voice activity detection."""

import logging
import time
import collections
import numpy as np
import sounddevice as sd
from threading import Lock
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S.%f'[:-3]  # Include milliseconds
)

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
BLOCK_DURATION_MS = 30  # Duration of each audio block in milliseconds
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)

# Voice activity detection parameters
AMBIENT_DURATION_SECONDS = 2.0  # Duration to measure ambient noise
ENERGY_THRESHOLD_MULTIPLIER = 1.5  # Multiply ambient RMS by this for speech threshold
PRE_SPEECH_BUFFER_CHUNKS = 5  # Number of chunks to keep before speech starts
POST_SPEECH_CHUNKS = 10  # Number of silent chunks needed to consider speech ended
MIN_SPEECH_CHUNKS = 5  # Minimum chunks for valid speech (avoid false positives)

class SpeechDetector:
    def __init__(self, device=None):
        self.device = device  # Store the selected device
        self.ambient_rms = None
        self.energy_threshold = None
        self.is_speech_active = False
        self.silent_chunks_count = 0
        self.speech_chunks_count = 0
        self.start_stop_count = 0
        self.pre_speech_buffer = collections.deque(maxlen=PRE_SPEECH_BUFFER_CHUNKS)
        self.speech_data = []
        self.lock = Lock()
        
        # For tracking speech segments
        self.current_speech_start_time = None
        
        # Log selected device info
        if device is not None:
            try:
                device_info = sd.query_devices(device, 'input')
                logging.info(f"Using device: {device_info['name']} (ID: {device})")
            except Exception as e:
                logging.warning(f"Could not query device {device}: {e}")
        else:
            default_device = sd.default.device[0]  # Input device
            if default_device is not None:
                device_info = sd.query_devices(default_device, 'input')
                logging.info(f"Using default device: {device_info['name']} (ID: {default_device})")
            else:
                logging.info("Using system default audio input")
        
    def calculate_rms(self, audio_data):
        """Calculate RMS (Root Mean Square) energy of audio data."""
        # Convert to float to avoid overflow
        audio_float = audio_data.astype(np.float64)
        return np.sqrt(np.mean(audio_float ** 2))
    
    def measure_ambient_noise(self):
        """Measure ambient noise level over a period of time."""
        logging.info(f"Measuring ambient noise for {AMBIENT_DURATION_SECONDS} seconds...")
        
        rms_values = []
        
        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Status: {status}")
            rms = self.calculate_rms(indata[:, 0])
            rms_values.append(rms)
        
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            callback=callback,
            device=self.device  # Use the selected device
        ):
            time.sleep(AMBIENT_DURATION_SECONDS)
        
        # Use median instead of mean to be more robust against occasional spikes
        self.ambient_rms = np.median(rms_values)
        self.energy_threshold = self.ambient_rms * ENERGY_THRESHOLD_MULTIPLIER
        
        logging.info(f"Ambient RMS: {self.ambient_rms:.2f}")
        logging.info(f"Energy threshold: {self.energy_threshold:.2f}")
        
    def audio_callback(self, indata, frames, time, status):
        """Process audio chunks for voice activity detection."""
        if status:
            logging.warning(f"Status: {status}")
            
        audio_chunk = indata[:, 0].copy()
        rms = self.calculate_rms(audio_chunk)
        
        with self.lock:
            # Always add to pre-speech buffer (it's a circular buffer)
            self.pre_speech_buffer.append((audio_chunk, rms))
            
            if not self.is_speech_active:
                # Check if speech is starting
                if rms > self.energy_threshold:
                    self.is_speech_active = True
                    self.speech_chunks_count = 1
                    self.silent_chunks_count = 0
                    self.current_speech_start_time = time.currentTime
                    
                    # Add pre-speech buffer to speech data
                    self.speech_data = []
                    for chunk, chunk_rms in self.pre_speech_buffer:
                        self.speech_data.append(chunk)
                        
                    logging.info(f"🎤 SPEECH STARTED (RMS: {rms:.2f}, includes {len(self.pre_speech_buffer)} pre-buffer chunks)")
                    
            else:
                # Speech is active, accumulate data
                self.speech_data.append(audio_chunk)
                self.speech_chunks_count += 1
                
                if rms > self.energy_threshold:
                    # Reset silent counter if we hear speech
                    self.silent_chunks_count = 0
                else:
                    # Count silent chunks
                    self.silent_chunks_count += 1
                    
                    if self.silent_chunks_count >= POST_SPEECH_CHUNKS:
                        # Speech has ended
                        self.is_speech_active = False
                        
                        # Only count as valid speech if it was long enough
                        if self.speech_chunks_count >= MIN_SPEECH_CHUNKS:
                            duration = (len(self.speech_data) * BLOCK_DURATION_MS) / 1000.0
                            logging.info(f"🔇 SPEECH ENDED (duration: {duration:.2f}s, {len(self.speech_data)} total chunks)")
                            
                            self.start_stop_count += 1
                            logging.info(f"Speech segment #{self.start_stop_count} completed")
                            
                            # Here you would normally process self.speech_data
                            # For now, we just clear it
                            self.speech_data = []
                        else:
                            logging.debug(f"Ignored short sound burst ({self.speech_chunks_count} chunks)")
                            
                        self.speech_chunks_count = 0
                        self.silent_chunks_count = 0
                        
    def run(self):
        """Run the speech detection test."""
        # First measure ambient noise
        self.measure_ambient_noise()
        
        logging.info("Starting speech detection... Speak to test!")
        logging.info("Will exit after detecting 3 speech segments")
        
        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE,
                callback=self.audio_callback,
                device=self.device  # Use the selected device
            ) as stream:
                # Keep running until we detect 3 start/stop cycles
                while self.start_stop_count < 3:
                    time.sleep(0.1)
                    
                logging.info("Test complete! Detected 3 speech segments.")
                
        except KeyboardInterrupt:
            logging.info("Test interrupted by user")
        except Exception as e:
            logging.error(f"Error during speech detection: {e}")

def list_audio_devices():
    """List all available audio input devices."""
    print("\n=== Available Audio Input Devices ===")
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            default_str = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default_str}")
            print(f"      Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']}Hz")
    
    if not input_devices:
        print("  No input devices found!")
    
    print()
    return input_devices

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test speech detection using sounddevice"
    )
    parser.add_argument(
        '-d', '--device',
        type=int,
        help='Input device ID (use -l to list devices)'
    )
    parser.add_argument(
        '-l', '--list-devices',
        action='store_true',
        help='List available audio input devices and exit'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    return parser.parse_args()

def main():
    """Main entry point for the test."""
    args = parse_arguments()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    logging.info("=== Speech Detection Test with Sounddevice ===")
    logging.info(f"Audio parameters: {SAMPLE_RATE}Hz, {CHANNELS} channel(s), {DTYPE}")
    logging.info(f"Block size: {BLOCKSIZE} samples ({BLOCK_DURATION_MS}ms)")
    
    # Create detector with specified device (or None for default)
    detector = SpeechDetector(device=args.device)
    detector.run()

if __name__ == "__main__":
    main() 