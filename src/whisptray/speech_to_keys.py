import logging
import threading
from queue import Empty, Queue
from datetime import datetime, timedelta, timezone
from time import sleep

import numpy as np
import torch
import whisper
import speech_recognition
from pynput.keyboard import Controller as KeyboardController, Key

# pylint: disable=too-many-instance-attributes
class SpeechToKeys:
    """
    Class to convert speech to keyboard input.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self, model_name: str, energy_threshold: int, record_timeout: float, phrase_timeout: float, source: speech_recognition.Microphone
    ):
        self.model_name = model_name
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = timedelta(seconds=phrase_timeout)
        self.source = source
        self.data_queue = Queue[bytes]()
        self.phrase_bytes = b""
        self.phrase_time = None
        self.transcription_history = [""]
        self.dictation_active = False
        self.keyboard = KeyboardController()

        logging.debug("Loading Whisper model: %s", model_name)

        try:
            self.audio_model = whisper.load_model(model_name)
            logging.debug("Whisper model loaded successfully.")
        except (OSError, RuntimeError, ValueError) as e:
            logging.error("Error loading Whisper model: %s", e, exc_info=True)
            # Consider re-raising or setting a flag that initialization failed
            return

        self.recorder = speech_recognition.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        # Don't let it change, because eventually it will whisptray noise.
        self.recorder.dynamic_energy_threshold = False

        with self.source:
            try:
                self.recorder.adjust_for_ambient_noise(self.source, duration=1)
                logging.debug("Adjusted for ambient noise.")
            except (
                speech_recognition.WaitTimeoutError,
                OSError,
                ValueError,
                AttributeError,
            ) as e:
                logging.warning(
                    "Could not adjust for ambient noise: %s", e, exc_info=True
                )
                # Continue without adjustment if it fails

        # Start listening in background so it's ready, and
        # self.dictation_active controls actual processing.
        try:
            # The callback will now check dictation_active before putting data in queue
            self.recorder.listen_in_background(
                self.source,
                self._record_callback,
                phrase_time_limit=self.record_timeout,
            )
            logging.debug("Background listener started.")
        except (OSError, AttributeError, RuntimeError) as e:
            logging.error("Error starting background listener: %s", e, exc_info=True)
            # Consider re-raising or setting a flag
            return

        # Start audio processing thread
        audio_thread = threading.Thread(
            target=self._process_audio, daemon=True, name="AudioProcessThread"
        )
        audio_thread.start()
        logging.debug("Audio processing thread started.")

    def shutdown(self):
        """
        Shuts down the speech to keys.
        """
        self.enabled = False # This will set self.dictation_active to False
        if self.recorder and hasattr(
            self.recorder, "stop_listening"
        ):  # Check if listening
            logging.debug("Stopping recorder listener.")
            try:
                self.recorder.stop_listening(wait_for_stop=False)
            except Exception as e:
                 logging.error("Error stopping recorder listener: %s", e, exc_info=True)
        # Ensure data_queue is emptied or signal processing thread to terminate if it depends on queue state
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except Empty:
                break
        logging.debug("SpeechToKeys shutdown complete.")


    def _reset(self):
        self.phrase_bytes = b""
        self.phrase_time = None
        self.transcription_history = [""]
        while not self.data_queue.empty():  # Clear the queue
            try:
                self.data_queue.get_nowait()
            except Empty:
                break

    def _record_callback(self, _, audio: speech_recognition.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        if self.dictation_active:
            data = audio.get_raw_data()
            self.data_queue.put(data)

    def _process_audio(self):
        """Processes audio from the queue and performs transcription."""
        while True:
            if not self.dictation_active:
                sleep(0.1)
                continue

            try:
                now = datetime.now(timezone.utc)
                if not self.data_queue.empty():
                    phrase_complete = False
                    if self.phrase_time is not None and now - self.phrase_time > self.phrase_timeout:
                        self.phrase_bytes = b""
                        phrase_complete = True
                    self.phrase_time = now

                    temp_audio_list = []
                    while not self.data_queue.empty():
                        try:
                            temp_audio_list.append(self.data_queue.get_nowait())
                        except Empty:
                            break

                    audio_data = b"".join(temp_audio_list)
                    self.phrase_bytes += audio_data

                    if not self.phrase_bytes:
                        sleep(0.1)
                        continue

                    audio_np = (
                        np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32768.0
                    )

                    if self.audio_model:
                        self._transcribe(phrase_complete, audio_np)
                    else:
                        logging.warning("Audio model not loaded, cannot transcribe.")
                else:
                    sleep(0.1)
            except Empty: # Should be caught by `if not self.data_queue.empty()`
                sleep(0.1)
            except Exception as e: # Catch broader exceptions in the loop
                logging.error("Error in process_audio loop: %s", e, exc_info=True)
                sleep(0.1) # Prevent rapid looping on persistent error

    def _transcribe(self, phrase_complete, audio_np):
        try:
            result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text = result["text"].strip()
            logging.debug("Transcribed text: '%s'", text) # Can be very verbose

            if text:
                if phrase_complete:
                    assert self.transcription_history, "Transcription history is empty"
                    if self.transcription_history[-1] and not self.transcription_history[
                        -1
                    ].endswith(" "):
                        self.keyboard.type(" ")
                    self.keyboard.type(text)
                    self.transcription_history.append(text)
                else:
                    if self.transcription_history and self.transcription_history[-1]:
                        for _ in range(len(self.transcription_history[-1])):
                            self.keyboard.press(Key.backspace)
                            self.keyboard.release(Key.backspace)
                    self.keyboard.type(text)
                    self.transcription_history[-1] = text
        except Exception as e:
            logging.error("Error during transcription: %s", e, exc_info=True)

    @property
    def enabled(self) -> bool:
        """
        Returns the enabled state of the speech to keys.
        """
        return self.dictation_active

    @enabled.setter
    def enabled(self, value: bool):
        if value != self.dictation_active:
            logging.debug(f"SpeechToKeys.enabled changing from {self.dictation_active} to {value}")
            if value:
                self._reset() # Clear previous phrase data for a fresh start
            self.dictation_active = value 