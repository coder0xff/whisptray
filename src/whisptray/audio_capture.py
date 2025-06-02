"""Audio capture module using sounddevice with voice activity detection."""

import collections
import logging
import time
from queue import Queue
from threading import Event, Thread
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

# Default parameters
DEFAULT_AMBIENT_DURATION_SECONDS = 1.0  # Duration to measure ambient noise
DEFAULT_ACTIVITY_AMBIENT_MULTIPLIER = 1.5  # Multiply ambient RMS by this for threshold

# Audio constants
SAMPLE_RATE = 14400
CHANNELS = 1
DTYPE = "int16"
BLOCK_SECONDS = 0.03  # Duration of each audio block in seconds

# Detection constants
ACTIVATION_SECONDS = 0.4  # Length of sound needed to activate
DEACTIVATION_SECONDS = 0.5  # Length of silence needed to deactivate
LEAD_SECONDS = 0.1  # Additional time before audio detection to include

# Callback constants
CALLBACK_BUFFER_SECONDS = 0.5

# Computed constants
BLOCK_SIZE = int(BLOCK_SECONDS * SAMPLE_RATE)  # Size of each audio block in samples
ACTIVATION_BLOCKS = int(ACTIVATION_SECONDS / BLOCK_SECONDS)
DEACTIVATION_BLOCKS = int(DEACTIVATION_SECONDS / BLOCK_SECONDS)
LEAD_BLOCKS = int(LEAD_SECONDS / BLOCK_SECONDS)
CALLBACK_BLOCKS = int(CALLBACK_BUFFER_SECONDS / BLOCK_SECONDS)


# pylint: disable=too-many-instance-attributes
class AudioCapture:
    """Captures audio with voice activity detection."""

    def __init__(
        self,
        on_audio: Callable[[np.ndarray], None],
        device: Optional[int | str] = None,
        ambient_duration: float = DEFAULT_AMBIENT_DURATION_SECONDS,
        activity_ambient_multiplier: float = DEFAULT_ACTIVITY_AMBIENT_MULTIPLIER,
    ):
        """
        Initialize AudioCapture.

        Args:
            on_audio_block: Callback for each audio block during speech
            device: Audio input device (None for default)
            ambient_duration: Seconds to measure ambient noise
            energy_threshold_multiplier: Multiplier for ambient RMS to set threshold
        """
        assert on_audio is not None
        self._on_audio = on_audio
        self._device = device
        self._ambient_duration = ambient_duration
        self._activity_ambient_multiplier = activity_ambient_multiplier

        # State
        self._activity_threshold: float = float("inf")
        self._active_blocks_count = 0
        self._silent_blocks_count = 0
        self._callback_blocks: Queue[np.ndarray] = Queue()
        self._capture_blocks: collections.deque[np.ndarray] = collections.deque()
        self._stream = None
        self._stop_event = Event()
        self._callback_thread = Thread(target=self._callback_worker, daemon=True)
        self._callback_thread.start()

    @staticmethod
    def query_devices():
        """
        Lists the available input audio devices using sounddevice.
        """
        print(sd.query_devices())

    @staticmethod
    def _iqr_rms(samples: np.ndarray) -> float:
        """Calculate a robust RMS-like value by filtering power outliers using Tukey's
        fences."""
        samples = samples.astype(np.float64)
        unbiased_samples = samples - np.mean(samples)
        powers = unbiased_samples**2
        q1 = np.percentile(powers, 25)
        q3 = np.percentile(powers, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered_powers = powers[(powers >= lower) & (powers <= upper)]
        return np.sqrt(np.mean(filtered_powers))

    def _measure_ambient_noise(self):
        """Measure ambient noise level over a period of time."""
        logging.info(
            "Measuring ambient noise for %s seconds...", self._ambient_duration
        )

        blocks = []

        def callback(
            indata: np.ndarray, _frames: int, _time, status: sd.CallbackFlags
        ) -> None:
            if status:
                raise RuntimeError(f"Audio status during ambient measurement: {status}")
            blocks.append(indata)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=callback,
            device=self._device,
        ):
            # Use Event.wait instead of time.sleep for interruptibility
            time.sleep(self._ambient_duration)

        audio_floats = np.concatenate(blocks)
        self._activity_threshold = (
            self._iqr_rms(audio_floats) * self._activity_ambient_multiplier
        )

    def _callback_worker(self):
        while not self._stop_event.is_set():
            data = self._callback_blocks.get()
            if data is None:
                break  # Poison pill
            self._on_audio(data)
            self._callback_blocks.task_done()

    def _flush_blocks_to_callback_thread(self):
        """Flush blocks to the callback."""
        if self._capture_blocks:
            self._callback_blocks.put(np.concatenate(self._capture_blocks))
            self._capture_blocks.clear()

    def _audio_callback(
        self, indata: np.ndarray, _frames: int, _time, status: sd.CallbackFlags
    ) -> None:
        """Process audio blocks for voice activity detection."""
        if status:
            logging.warning("Audio status: %s", status)

        if self._stop_event.is_set():
            return

        audio_level = self._iqr_rms(indata)
        self._capture_blocks.append(indata.copy())

        if audio_level >= self._activity_threshold:
            self._silent_blocks_count = 0
            self._active_blocks_count += 1
            if self._active_blocks_count == ACTIVATION_BLOCKS:
                logging.info("Detected start of speech")
        else:
            self._silent_blocks_count += 1
            if self._silent_blocks_count >= DEACTIVATION_BLOCKS:
                if self._active_blocks_count >= ACTIVATION_BLOCKS:
                    logging.info("Detected end of speech")
                    self._flush_blocks_to_callback_thread()
                self._active_blocks_count = 0

        if self._active_blocks_count >= ACTIVATION_BLOCKS:
            if len(self._capture_blocks) > CALLBACK_BLOCKS:
                logging.debug("Flushing blocks to callback thread")
                self._flush_blocks_to_callback_thread()
        else:
            # Discard old blocks
            while len(self._capture_blocks) > LEAD_BLOCKS + ACTIVATION_BLOCKS:
                self._capture_blocks.popleft()

    def start(self):
        """Start audio capture and voice activity detection."""
        if self._stream is not None:
            logging.warning("Audio capture already started")
            return

        self._stop_event.clear()
        self._measure_ambient_noise()

        # Create and start the audio stream
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._audio_callback,
            device=self._device,
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

    def shutdown(self):
        """Shutdown the audio capture."""
        self.stop()
        # Clear buffers
        self._callback_blocks.put(None)  # Poison pill
        self._callback_thread.join()
