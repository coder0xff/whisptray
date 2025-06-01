"""Audio capture module using sounddevice with voice activity detection."""

import logging
import collections
from threading import Lock, Event
from typing import Callable, Optional
import numpy as np
import sounddevice as sd

# Audio parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = 'int16'
DEFAULT_BLOCK_DURATION_MS = 30  # Duration of each audio block in milliseconds

# Voice activity detection parameters
DEFAULT_AMBIENT_DURATION_SECONDS = 2.0  # Duration to measure ambient noise
DEFAULT_ENERGY_THRESHOLD_MULTIPLIER = 1.5  # Multiply ambient RMS by this for speech threshold
DEFAULT_PRE_SPEECH_BUFFER_BLOCKS = 5  # Number of blocks to keep before speech starts
DEFAULT_POST_SPEECH_BLOCKS = 10  # Number of silent blocks needed to consider speech ended
DEFAULT_MIN_SPEECH_BLOCKS = 5  # Minimum blocks for valid speech (avoid false positives)
DEFAULT_CONSECUTIVE_SECONDS_FOR_START = 0.4  # Consecutive seconds above threshold to start speech
DEFAULT_CONSECUTIVE_BLOCKS_FOR_START = int(DEFAULT_CONSECUTIVE_SECONDS_FOR_START / DEFAULT_BLOCK_DURATION_MS * 1000)
DEFAULT_CALLBACK_BUFFER_DURATION = 0.5  # Seconds of audio to buffer before calling callback


class AudioCapture:
    """Captures audio with voice activity detection and provides speech segments."""
    
    def __init__(
        self,
        device: Optional[int | str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        dtype: str = DEFAULT_DTYPE,
        block_duration_ms: int = DEFAULT_BLOCK_DURATION_MS,
        ambient_duration: float = DEFAULT_AMBIENT_DURATION_SECONDS,
        energy_threshold_multiplier: float = DEFAULT_ENERGY_THRESHOLD_MULTIPLIER,
        pre_speech_buffer_blocks: int = DEFAULT_PRE_SPEECH_BUFFER_BLOCKS,
        post_speech_blocks: int = DEFAULT_POST_SPEECH_BLOCKS,
        min_speech_blocks: int = DEFAULT_MIN_SPEECH_BLOCKS,
        on_audio_block: Optional[Callable[[np.ndarray], None]] = None,
        callback_buffer_duration: float = DEFAULT_CALLBACK_BUFFER_DURATION,
    ):
        """
        Initialize AudioCapture.
        
        Args:
            device: Audio input device (None for default)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            dtype: Audio data type ('int16', 'int32', 'float32')
            block_duration_ms: Duration of each audio block in milliseconds
            ambient_duration: Seconds to measure ambient noise
            energy_threshold_multiplier: Multiplier for ambient RMS to set threshold
            pre_speech_buffer_blocks: Blocks to keep before speech starts
            post_speech_blocks: Silent blocks needed to end speech
            min_speech_blocks: Minimum blocks for valid speech
            on_audio_block: Callback for each audio block during speech
            callback_buffer_duration: Seconds of audio to accumulate before calling callback
        """
        self.device = device
        self._sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = int(sample_rate * block_duration_ms / 1000)
        
        # VAD parameters
        self.ambient_duration = ambient_duration
        self.energy_threshold_multiplier = energy_threshold_multiplier
        self.pre_speech_buffer_blocks = pre_speech_buffer_blocks
        self.post_speech_blocks = post_speech_blocks
        self.min_speech_blocks = min_speech_blocks
        self.callback_buffer_duration = callback_buffer_duration
        self.callback_buffer_blocks = int(callback_buffer_duration * 1000 / block_duration_ms)
        
        # State
        self.ambient_rms = None
        self.energy_threshold = None
        self.is_speech_active = False
        self.silent_blocks_count = 0
        self.speech_blocks_count = 0
        self.consecutive_above_threshold = 0  # Track consecutive blocks above threshold
        self.pre_speech_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=pre_speech_buffer_blocks)
        self.speech_data: list[np.ndarray] = []
        self.callback_buffer: list[np.ndarray] = []  # Buffer for callback
        self.lock = Lock()
        self._stream = None
        self._stop_event = Event()
        
        # Callback
        self.on_audio_block = on_audio_block
        
        # Log device info
        self._log_device_info()
    
    @property
    def sample_rate(self) -> int:
        """Get the sample rate in Hz."""
        return self._sample_rate
    
    def _log_device_info(self):
        """Log information about the selected audio device."""
        try:
            if self.device is not None:
                device_info = sd.query_devices(self.device, 'input')
                logging.info(f"Audio device: {device_info['name']} (ID: {self.device})")
            else:
                default_device = sd.default.device[0]
                if default_device is not None:
                    device_info = sd.query_devices(default_device, 'input')
                    logging.info(f"Audio device: {device_info['name']} (default, ID: {default_device})")
                else:
                    logging.info("Audio device: System default")
        except Exception as e:
            logging.warning(f"Could not query audio device info: {e}")
    
    def calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) energy of audio data."""
        audio_float = audio_data.astype(np.float64)
        return np.sqrt(np.mean(audio_float ** 2))
    
    def measure_ambient_noise(self):
        """Measure ambient noise level over a period of time."""
        logging.info(f"Measuring ambient noise for {self.ambient_duration} seconds...")
        
        rms_values = []
        
        def callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio status during ambient measurement: {status}")
            rms = self.calculate_rms(indata[:, 0] if indata.shape[1] > 1 else indata.flatten())
            rms_values.append(rms)
        
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=callback,
            device=self.device
        ):
            # Use Event.wait instead of time.sleep for interruptibility
            self._stop_event.wait(self.ambient_duration)
        
        if rms_values:
            # Use 75th percentile instead of median to better capture ambient variation
            self.ambient_rms = np.percentile(rms_values, 75)
            self.energy_threshold = self.ambient_rms * self.energy_threshold_multiplier
            
            logging.info(f"Ambient RMS (75th percentile): {self.ambient_rms:.2f}")
            logging.info(f"Energy threshold: {self.energy_threshold:.2f}")
            
            # Log some statistics for debugging
            logging.debug(f"Ambient stats - Min: {np.min(rms_values):.2f}, "
                         f"25th: {np.percentile(rms_values, 25):.2f}, "
                         f"Median: {np.median(rms_values):.2f}, "
                         f"75th: {np.percentile(rms_values, 75):.2f}, "
                         f"Max: {np.max(rms_values):.2f}")
        else:
            logging.warning("No ambient noise samples collected")
            # Set reasonable defaults
            self.ambient_rms = 100.0
            self.energy_threshold = 150.0
    
    def _audio_callback(self, indata, frames, time, status):
        """Process audio blocks for voice activity detection."""
        if status:
            logging.warning(f"Audio status: {status}")
        
        if self._stop_event.is_set():
            return
            
        audio_block = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
        rms = self.calculate_rms(audio_block)
        
        # Log block RMS for debugging
        if rms > self.energy_threshold * 0.8:  # Only log blocks close to or above threshold
            logging.debug(f"Block RMS: {rms:.2f} {'[ABOVE THRESHOLD]' if rms > self.energy_threshold else '[approaching threshold]'}")
        
        with self.lock:
            # Always add to pre-speech buffer (it's a circular buffer)
            self.pre_speech_buffer.append(audio_block)
            
            if not self.is_speech_active:
                # Check if speech is starting
                if rms > self.energy_threshold:
                    self.consecutive_above_threshold += 1
                    
                    # Require multiple consecutive blocks above threshold
                    if self.consecutive_above_threshold >= DEFAULT_CONSECUTIVE_BLOCKS_FOR_START:
                        self.is_speech_active = True
                        self.speech_blocks_count = self.consecutive_above_threshold
                        self.silent_blocks_count = 0
                        
                        # Add pre-speech buffer to speech data
                        self.speech_data = list(self.pre_speech_buffer)
                        
                        logging.debug(f"Speech started after {self.consecutive_above_threshold} consecutive blocks "
                                    f"(RMS: {rms:.2f}, includes {len(self.pre_speech_buffer)} pre-buffer blocks)")
                        
                        # Add buffered blocks to callback buffer
                        if self.on_audio_block:
                            self.callback_buffer = list(self.speech_data)
                            self._check_callback_buffer()
                else:
                    # Reset consecutive counter if below threshold
                    self.consecutive_above_threshold = 0
                    
            else:
                # Speech is active, accumulate data
                self.speech_data.append(audio_block)
                self.speech_blocks_count += 1
                
                # Add to callback buffer
                if self.on_audio_block:
                    self.callback_buffer.append(audio_block)
                    self._check_callback_buffer()
                
                if rms > self.energy_threshold:
                    # Reset silent counter if we hear speech
                    self.silent_blocks_count = 0
                else:
                    # Count silent blocks
                    self.silent_blocks_count += 1
                    
                    if self.silent_blocks_count >= self.post_speech_blocks:
                        # Speech has ended
                        self.is_speech_active = False
                        
                        # Send any remaining buffered audio
                        if self.on_audio_block and self.callback_buffer:
                            audio_to_send = np.concatenate(self.callback_buffer)
                            self.on_audio_block(audio_to_send)
                            self.callback_buffer = []
                        
                        # Only process as valid speech if it was long enough
                        if self.speech_blocks_count >= self.min_speech_blocks:
                            # Log that speech segment ended
                            duration = len(self.speech_data) * self.blocksize / self._sample_rate
                            logging.debug(f"Speech ended (duration: {duration:.2f}s, {self.speech_blocks_count} blocks)")
                        else:
                            logging.debug(f"Ignored short sound burst ({self.speech_blocks_count} blocks)")
                        
                        # Reset for next speech segment
                        self.speech_data = []
                        self.speech_blocks_count = 0
                        self.silent_blocks_count = 0
                        self.consecutive_above_threshold = 0
    
    def _check_callback_buffer(self):
        """Check if callback buffer has enough data to send."""
        if len(self.callback_buffer) >= self.callback_buffer_blocks:
            # Send accumulated audio
            audio_to_send = np.concatenate(self.callback_buffer[:self.callback_buffer_blocks])
            self.on_audio_block(audio_to_send)
            # Keep any remaining blocks for next callback
            self.callback_buffer = self.callback_buffer[self.callback_buffer_blocks:]
    
    def start(self):
        """Start audio capture and voice activity detection."""
        if self._stream is not None:
            logging.warning("Audio capture already started")
            return
        
        # First measure ambient noise
        self.measure_ambient_noise()
        
        if self._stop_event.is_set():
            return
        
        logging.info("Starting audio capture with voice activity detection")
        
        # Create and start the audio stream
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self._audio_callback,
            device=self.device
        )
        self._stream.start()
    
    def stop(self):
        """Stop audio capture."""
        logging.info("Stopping audio capture")
        self._stop_event.set()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Clear buffers
        with self.lock:
            self.pre_speech_buffer.clear()
            self.speech_data = []
            self.callback_buffer = []
            self.is_speech_active = False
            self.silent_blocks_count = 0
            self.speech_blocks_count = 0
            self.consecutive_above_threshold = 0
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 