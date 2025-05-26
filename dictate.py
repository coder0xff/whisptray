import argparse
import os
import threading
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from time import sleep

import numpy as np
import pystray
import speech_recognition as sr
import torch
import whisper
from PIL import Image, ImageDraw
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key

# --- Global Variables ---
keyboard = KeyboardController()
dictation_active = False
app_icon = None
audio_model = None
recorder = None
source = None
data_queue = Queue[bytes]()
phrase_time = None
phrase_bytes = b""
transcription_history = [""]  # Stores the history of transcriptions

# --- Configuration ---
MODEL_NAME = "turbo"
ENERGY_THRESHOLD = 1000
RECORD_TIMEOUT = 2.0  # Seconds for real-time recording
PHRASE_TIMEOUT = 3.0  # Seconds of silence before new line
DEFAULT_MICROPHONE = "pulse"  # For Linux


# --- Helper Functions ---
def create_tray_image(width, height, color1, color2):
    """Creates a simple image for the tray icon."""
    image = Image.new("RGB", (width, height), color1)
    dc = ImageDraw.Draw(image)
    dc.rectangle((width // 2, 0, width, height // 2), fill=color2)
    dc.rectangle((0, height // 2, width // 2, height), fill=color2)
    return image


# --- Speech Recognition Logic ---
def record_callback(_, audio: sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    global data_queue
    if dictation_active:
        data = audio.get_raw_data()
        data_queue.put(data)


def process_audio():
    """Processes audio from the queue and performs transcription."""
    global phrase_time, phrase_bytes, transcription_history, audio_model

    while True:
        if not dictation_active:
            sleep(0.1)  # Sleep briefly if dictation is not active
            continue

        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(
                    seconds=PHRASE_TIMEOUT
                ):
                    phrase_bytes = b""
                    phrase_complete = True
                phrase_time = now

                # Combine audio data from queue. Create a temporary list to avoid issues
                # if data_queue is modified during iteration.
                temp_audio_list = []
                while not data_queue.empty():
                    try:
                        temp_audio_list.append(data_queue.get_nowait())
                    except data_queue.Empty:
                        # Should not happen if initial check was true, but good for
                        # safety
                        break

                audio_data = b"".join(temp_audio_list)
                phrase_bytes += audio_data

                if not phrase_bytes:  # Skip if no audio data
                    sleep(0.1)
                    continue

                audio_np = (
                    np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                if audio_model:
                    result = audio_model.transcribe(
                        audio_np, fp16=torch.cuda.is_available()
                    )
                    text = result["text"].strip()

                    if text:  # Only process if there is new text
                        if phrase_complete:
                            # New phrase, type with a space if previous text exists and
                            # doesn't end with space
                            if transcription_history[-1] and not transcription_history[
                                -1
                            ].endswith(" "):
                                keyboard.type(" ")
                            keyboard.type(text)
                            transcription_history.append(text)
                        else:
                            # Continuing a phrase.
                            # Need to "backspace" the previous part of this phrase and
                            # type the new full phrase. This is a simplification. A more
                            # robust solution would be to diff the text.
                            if transcription_history and transcription_history[-1]:
                                for _ in range(len(transcription_history[-1])):
                                    keyboard.press(Key.backspace)
                                    keyboard.release(Key.backspace)
                            keyboard.type(text)
                            transcription_history[-1] = text
                else:
                    print("Audio model not loaded yet.")
            else:
                sleep(0.1)  # More responsive sleep
        except Exception as e:
            print(f"Error in process_audio: {e}")
            sleep(0.1)


# --- Tray Icon Functions ---
def toggle_dictation(icon, item):
    """Toggles dictation on/off."""
    global dictation_active, recorder, source
    print(f"[DEBUG] toggle_dictation called. Current state: {dictation_active}")
    dictation_active = not dictation_active
    if dictation_active:
        print("[DEBUG] Dictation started by toggle.")
        # Start listening in background if not already
        if recorder and source and not recorder.listening:
            # Ensure callback is only set once or reset it
            try:
                recorder.listen_in_background(
                    source, record_callback, phrase_time_limit=RECORD_TIMEOUT
                )
            except Exception as e:
                print(f"Error restarting listener: {e}")
        # Clear previous phrase data to avoid re-typing old text
        global phrase_bytes, phrase_time, transcription_history, data_queue
        phrase_bytes = b""
        phrase_time = None
        transcription_history = [""]
        while not data_queue.empty():  # Clear the queue
            try:
                data_queue.get_nowait()
            except data_queue.Empty:
                break

    else:
        print("[DEBUG] Dictation stopped by toggle.")
        # Consider stopping the listener if you want to save resources,
        # but be careful about restarting it correctly.
        # For now, we just set dictation_active to False and the callback/processing
        # will ignore new data.


def exit_program(icon, item):
    """Stops the program."""
    global dictation_active, app_icon, recorder
    print("[DEBUG] exit_program called.")
    dictation_active = False
    if recorder and hasattr(recorder, "stop_listening"):  # Check if listening
        print("[DEBUG] Stopping recorder listener.")
        recorder.stop_listening(wait_for_stop=False)
    if app_icon:
        print("[DEBUG] Stopping app_icon.")
        app_icon.stop()
    print("[DEBUG] Calling os._exit(0).")
    os._exit(0)  # Force exit if threads are hanging


def setup_tray_icon():
    """Sets up and runs the system tray icon."""
    global app_icon
    print("[DEBUG] setup_tray_icon called.")
    icon_image = create_tray_image(64, 64, "blue", "white")  # Placeholder icon
    menu = pystray.Menu(
        pystray.MenuItem(
            "Toggle Dictation",
            lambda: toggle_dictation(app_icon, None),  # Ensure lambda for direct call if needed
            checked=lambda item: dictation_active
        ),
        pystray.MenuItem("Exit", lambda: exit_program(app_icon, None)) # Ensure lambda
    )
    app_icon = pystray.Icon("dictate_app", icon_image, "Dictate App", menu)
    print("[DEBUG] pystray.Icon created. Calling app_icon.run().")
    app_icon.run()
    print("[DEBUG] app_icon.run() finished.") # Should not be reached if os._exit is called


# --- Main Function ---
def main():
    global audio_model
    global recorder
    global source
    global MODEL_NAME
    global ENERGY_THRESHOLD
    global RECORD_TIMEOUT
    global PHRASE_TIMEOUT
    global DEFAULT_MICROPHONE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
    )
    parser.add_argument(
        "--non_english", action="store_true", help="Don't use the english model."
    )
    parser.add_argument(
        "--energy_threshold",
        default=ENERGY_THRESHOLD,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=RECORD_TIMEOUT,
        help="How real time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=PHRASE_TIMEOUT,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default=DEFAULT_MICROPHONE,
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
    args = parser.parse_args()

    MODEL_NAME = args.model
    ENERGY_THRESHOLD = args.energy_threshold
    RECORD_TIMEOUT = args.record_timeout
    PHRASE_TIMEOUT = args.phrase_timeout
    if "linux" in platform:
        DEFAULT_MICROPHONE = args.default_microphone

    if not args.non_english and MODEL_NAME not in ["large", "turbo"]:
        temp = ".en"
    else:
        temp = ""

    # Load Whisper model
    print(f"Loading Whisper model: {MODEL_NAME}{temp}")
    effective_model_name = MODEL_NAME
    if MODEL_NAME not in ["large", "turbo"] and not args.non_english:
        effective_model_name = MODEL_NAME + ".en"

    try:
        audio_model = whisper.load_model(effective_model_name)
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return

    # Setup SpeechRecognition
    recorder = sr.Recognizer()
    recorder.energy_threshold = ENERGY_THRESHOLD
    recorder.dynamic_energy_threshold = False  # Important

    if "linux" in platform:
        mic_name = DEFAULT_MICROPHONE
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'Microphone with name "{name}" found')
            return
        else:
            source = None  # Initialize source
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    print(f"Using microphone: {name}")
                    break
            if source is None:
                print(
                    f"Microphone containing '{mic_name}' not found. Please check"
                    " available microphones."
                )
                print("Available microphone devices are: ")
                for index, name_available in enumerate(
                    sr.Microphone.list_microphone_names()
                ):
                    print(f'Microphone with name "{name_available}" found')
                return
    else:
        source = sr.Microphone(sample_rate=16000)
        print("Using default microphone.")

    with source:
        try:
            recorder.adjust_for_ambient_noise(source, duration=1)  # Adjust for 1 second
            print("Adjusted for ambient noise.")
        except Exception as e:
            print(f"Could not adjust for ambient noise: {e}")
            # Continue without adjustment if it fails

    # Start listening in background (but it will only process if dictation_active is
    # True). We start it here so it's ready, and toggle_dictation controls actual
    # processing.
    try:
        # The callback will now check dictation_active before putting data in queue
        recorder.listen_in_background(
            source, record_callback, phrase_time_limit=RECORD_TIMEOUT
        )
        print("Background listener started.")
    except Exception as e:
        print(f"Error starting background listener: {e}")
        return

    # Start audio processing thread
    audio_thread = threading.Thread(target=process_audio, daemon=True)
    audio_thread.start()
    print("Audio processing thread started.")

    # Start tray icon
    print("Starting tray icon...")
    setup_tray_icon()  # This will block until exit
    print("[DEBUG] main function finished after setup_tray_icon call.") # Should not be reached


if __name__ == "__main__":
    # It's good practice to ensure DISPLAY is set for GUI apps on Linux
    if "linux" in platform and not os.environ.get("DISPLAY"):
        print("Error: DISPLAY environment variable not set. GUI cannot be displayed.")
        print("Please ensure you are running this in a graphical environment.")
    else:
        main()
