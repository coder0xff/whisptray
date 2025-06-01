#!/usr/bin/env python3
"""Test speech detection using AudioCapture module."""

import sys
import os
import logging
import time
import argparse
import numpy as np

# Add parent directory to path to import whisptray modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from whisptray.audio_capture import AudioCapture

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S.%f'[:-3]  # Include milliseconds
)


def test_audio_capture(device=None, duration=30):
    """
    Test AudioCapture by capturing audio for a specified duration.
    
    Args:
        device: Audio device to use (None for default)
        duration: How long to capture audio in seconds
    """
    logging.info("=== Testing AudioCapture ===")
    
    blocks_received = []
    current_segment_blocks = []
    segments = []
    in_speech = False
    sample_rate = None  # Will be set from capture instance
    
    def on_audio_block(audio_data):
        """Callback for audio blocks during speech (called every 0.5 seconds)."""
        nonlocal in_speech, current_segment_blocks
        
        if not in_speech:
            # Start of new segment
            in_speech = True
            current_segment_blocks = []
            logging.info("Speech segment started")
        
        current_segment_blocks.append(audio_data)
        blocks_received.append(audio_data)
        
        duration_sec = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        total_duration_so_far = sum(len(block) / sample_rate for block in current_segment_blocks)
        
        logging.info(f"  Received audio block: {duration_sec:.2f}s (segment total: {total_duration_so_far:.2f}s)")
        logging.debug(f"    RMS: {rms:.2f}, Min: {audio_data.min()}, Max: {audio_data.max()}")
    
    # Create AudioCapture instance
    capture = AudioCapture(
        device=device,
        on_audio_block=on_audio_block,
        # Use slightly more aggressive settings for testing
        energy_threshold_multiplier=1.3,
        post_speech_blocks=8,
        min_speech_blocks=3,
        callback_buffer_duration=0.5  # Buffer 0.5 seconds before calling callback
    )
    
    # Start capture
    with capture:
        # Get sample rate from capture instance
        sample_rate = capture.sample_rate
        logging.info("Audio capture started. Speak to test voice activity detection!")
        logging.info(f"Will capture for {duration} seconds...")
        logging.info("Callback will be called every 0.5 seconds during speech")
        
        start_time = time.time()
        last_block_count = 0
        silence_start_time = None
        
        # Monitor for speech ending (when blocks stop coming)
        while time.time() - start_time < duration:
            time.sleep(0.1)
            
            # Check if we were in speech but blocks stopped coming
            if in_speech and len(blocks_received) == last_block_count:
                # No new blocks - track silence duration
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > 0.5:  # 0.5s of silence
                    # Speech has ended
                    in_speech = False
                    if current_segment_blocks:
                        # Concatenate blocks into segment
                        segment = np.concatenate(current_segment_blocks)
                        segments.append(segment)
                        duration_sec = len(segment) / sample_rate
                        rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2))
                        logging.info(f"Speech segment ended: {duration_sec:.2f} seconds, {len(current_segment_blocks)} blocks")
                        logging.info(f"  Overall RMS: {rms:.2f}, Min: {segment.min()}, Max: {segment.max()}")
                        current_segment_blocks = []
                    silence_start_time = None
            else:
                silence_start_time = None
            
            last_block_count = len(blocks_received)
    
    # Handle any remaining segment
    if current_segment_blocks:
        segment = np.concatenate(current_segment_blocks)
        segments.append(segment)
        duration_sec = len(segment) / sample_rate
        logging.info(f"Final segment: {duration_sec:.2f} seconds, {len(current_segment_blocks)} blocks")
    
    # Summary
    logging.info(f"\nTest completed. Received {len(blocks_received)} audio blocks in {len(segments)} segments")
    
    if segments:
        total_duration = sum(len(s) / sample_rate for s in segments)
        avg_duration = total_duration / len(segments)
        logging.info(f"Total speech duration: {total_duration:.2f} seconds")
        logging.info(f"Average segment duration: {avg_duration:.2f} seconds")
        
        # Show distribution of segment lengths
        lengths = [len(s) / sample_rate for s in segments]
        logging.info(f"Segment lengths: min={min(lengths):.2f}s, max={max(lengths):.2f}s")
    else:
        logging.warning("No speech segments were captured. Try speaking louder or adjusting the threshold.")
    
    return len(segments) > 0


def list_audio_devices():
    """List all available audio input devices."""
    import sounddevice as sd
    
    print("\n=== Available Audio Input Devices ===")
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_str = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default_str}")
            print(f"      Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']}Hz")
    print()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test AudioCapture module for speech detection"
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
        '-t', '--duration',
        type=int,
        default=30,
        help='Test duration in seconds (default: 30)'
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
    
    # Run the test
    success = test_audio_capture(
        device=args.device,
        duration=args.duration
    )
    
    if success:
        logging.info("Test passed! Audio capture is working correctly.")
    else:
        logging.error("Test failed! No audio was captured.")
        sys.exit(1)


if __name__ == "__main__":
    main() 