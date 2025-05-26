import argparse
import ctypes
import ctypes.util
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from queue import Queue
from sys import platform
from time import sleep

# Conditional import for tkinter
try:
    import tkinter
    import tkinter.messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Don't use AppIndicator on Linux, because it doesn't support direct icon clicks.
if "linux" in platform:
    os.environ["PYSTRAY_BACKEND"] = "xorg"

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
last_click_time = 0.0
click_timer = None  # Will be a threading.Timer
EFFECTIVE_DOUBLE_CLICK_INTERVAL = 0.5  # Default in seconds, updated by system settings
app_is_exiting = threading.Event()

# --- Configuration ---
MODEL_NAME = "turbo"
ENERGY_THRESHOLD = 1000
RECORD_TIMEOUT = 2.0  # Seconds for real-time recording
PHRASE_TIMEOUT = 3.0  # Seconds of silence before new line
DEFAULT_MICROPHONE = "pulse"  # For Linux

# --- ALSA Error Handling Setup ---
# Define the Python callback function signature for ctypes
# Corresponds to:
# typedef void (*python_callback_func_t)(
#     const char *file,
#     int line,
#     const char *function,
#     int err,
#     const char *formatted_msg
# );
PYTHON_ALSA_ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None,  # Return type: void
    ctypes.c_char_p,  # const char *file
    ctypes.c_int,  # int line
    ctypes.c_char_p,  # const char *function
    ctypes.c_int,  # int err
    ctypes.c_char_p,  # const char *formatted_msg
)

alsa_logger = logging.getLogger("ALSA")


def python_alsa_error_handler(file_ptr, line, func_ptr, err, formatted_msg_ptr):
    """
    Python callback to handle ALSA error messages passed from C.
    Decodes char* to Python strings.
    """
    try:
        file = (
            ctypes.string_at(file_ptr).decode("utf-8", "replace")
            if file_ptr
            else "UnknownFile"
        )
        function = (
            ctypes.string_at(func_ptr).decode("utf-8", "replace")
            if func_ptr
            else "UnknownFunction"
        )
        formatted_msg = (
            ctypes.string_at(formatted_msg_ptr).decode("utf-8", "replace")
            if formatted_msg_ptr
            else ""
        )

        # Using python logging to output ALSA messages
        alsa_logger.info(f"{file}:{line} ({function}) - err {err}: {formatted_msg}")
    except Exception as e:
        # Fallback logging if there's an error within the error handler itself
        print(f"Error in python_alsa_error_handler: {e}")


# Keep a reference to the ctype function object to prevent garbage collection
py_error_handler_ctype = PYTHON_ALSA_ERROR_HANDLER_FUNC(python_alsa_error_handler)


def setup_alsa_error_handler():
    """
    Sets up a custom ALSA error handler using the C helper library.
    """
    if "linux" not in platform:
        logging.info("Skipping ALSA error handler setup on non-Linux platform.")
        return

    try:
        c_redirect_lib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "alsa_redirect.so"
        )
        if not os.path.exists(c_redirect_lib_path):
            try:
                c_redirect_lib = ctypes.CDLL("alsa_redirect.so")
                logging.info("Loaded alsa_redirect.so from system path.")
            except OSError:
                logging.error(
                    f"alsa_redirect.so not found at {c_redirect_lib_path} or in system"
                    " paths. ALSA logs will not be redirected."
                )
                return
        else:
            c_redirect_lib = ctypes.CDLL(c_redirect_lib_path)
            logging.info(f"Loaded alsa_redirect.so from: {c_redirect_lib_path}")

        # 2. Define argtypes and restype for functions in alsa_redirect.so
        # void register_python_alsa_callback(python_callback_func_t callback);
        c_redirect_lib.register_python_alsa_callback.argtypes = [
            PYTHON_ALSA_ERROR_HANDLER_FUNC
        ]
        c_redirect_lib.register_python_alsa_callback.restype = None

        # int initialize_alsa_error_handling();
        c_redirect_lib.initialize_alsa_error_handling.argtypes = []
        c_redirect_lib.initialize_alsa_error_handling.restype = ctypes.c_int

        # int clear_alsa_error_handling();
        c_redirect_lib.clear_alsa_error_handling.argtypes = []
        c_redirect_lib.clear_alsa_error_handling.restype = ctypes.c_int

        c_redirect_lib.register_python_alsa_callback(py_error_handler_ctype)
        logging.info("Registered Python ALSA error handler with C helper.")

        # 4. Ask the C library to set ALSA's error handler
        ret = c_redirect_lib.initialize_alsa_error_handling()
        if ret < 0:
            logging.error(
                f"C library failed to set ALSA error handler. Error code: {ret}"
            )
        else:
            logging.info("C library successfully set ALSA error handler.")
            try:
                asound_lib_name = ctypes.util.find_library("asound")
                if asound_lib_name:
                    asound = ctypes.CDLL(asound_lib_name)
                    # Ensure snd_config_update_free_global is defined before calling
                    if hasattr(asound, "snd_config_update_free_global"):
                        asound.snd_config_update_free_global.argtypes = []
                        asound.snd_config_update_free_global.restype = ctypes.c_int
                        asound.snd_config_update_free_global()
                        logging.debug(
                            "Called snd_config_update_free_global to test ALSA handler."
                        )
                    else:
                        logging.debug(
                            "snd_config_update_free_global not found in libasound,"
                            " skipping test call."
                        )
                else:
                    logging.warning("libasound not found, skipping ALSA test call.")
            except Exception as e_test:
                logging.debug(f"Exception during ALSA test call: {e_test}")

    except Exception as e:
        logging.error(f"Error setting up ALSA error handler: {e}", exc_info=True)


def _get_system_double_click_time() -> float | None:
    """Tries to get the system's double-click time in seconds."""
    try:
        if platform == "linux" or platform == "linux2":
            # Try GSettings first (common in GNOME-based environments)
            try:
                proc = subprocess.run(
                    [
                        "gsettings",
                        "get",
                        "org.gnome.settings-daemon.peripherals.mouse",
                        "double-click",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=0.5,
                )
                value_ms = int(proc.stdout.strip())
                return value_ms / 1000.0
            except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                ValueError,
                subprocess.TimeoutExpired,
            ):
                # Fallback to xrdb for other X11 environments
                try:
                    proc = subprocess.run(
                        ["xrdb", "-query"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=0.5,
                    )
                    for line in proc.stdout.splitlines():
                        if (
                            "DblClickTime" in line
                        ):  # XTerm*DblClickTime, URxvt.doubleClickTime etc.
                            value_ms = int(line.split(":")[1].strip())
                            return value_ms / 1000.0
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    ValueError,
                    IndexError,
                    subprocess.TimeoutExpired,
                ):
                    logging.debug(
                        "Could not determine double-click time from GSettings or xrdb."
                    )
                    pass  # Neither GSettings nor xrdb succeeded.
        elif platform == "win32":
            proc = subprocess.run(
                [
                    "reg",
                    "query",
                    "HKCU\\Control Panel\\Mouse",
                    "/v",
                    "DoubleClickSpeed",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=0.5,
            )
            # Output is like: '    DoubleClickSpeed    REG_SZ    500'
            value_ms = int(proc.stdout.split()[-1])
            return value_ms / 1000.0
        elif platform == "darwin":  # macOS
            # Getting this programmatically on macOS is non-trivial. Default for now.
            logging.debug("Using default double-click time for macOS.")
            pass
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        ValueError,
        IndexError,
        subprocess.TimeoutExpired,
    ) as e:
        logging.warning(f"Could not query system double-click time: {e}")
    return None


def initialize_double_click_interval():
    """Initializes the double-click interval, falling back to default if needed."""
    global EFFECTIVE_DOUBLE_CLICK_INTERVAL
    system_interval = _get_system_double_click_time()
    if (
        system_interval is not None and 0.1 <= system_interval <= 2.0
    ):  # Sanity check interval
        EFFECTIVE_DOUBLE_CLICK_INTERVAL = system_interval
        logging.info(
            "Using system double-click interval:"
            f" {EFFECTIVE_DOUBLE_CLICK_INTERVAL:.2f}s"
        )
    else:
        logging.info(
            "Using default double-click interval:"
            f" {EFFECTIVE_DOUBLE_CLICK_INTERVAL:.2f}s"
        )


def create_tray_image(width, height, shape_color, shape_type):
    """Creates an image for the tray icon (record or stop button) with a transparent
    background."""
    # RGBA mode for transparency, background color is (R, G, B, Alpha)
    # (0,0,0,0) means fully transparent black
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dc = ImageDraw.Draw(image)
    padding = int(width * 0.2)  # Add padding around the shape

    # The shape_color (e.g., "red") will be drawn as opaque on the transparent canvas
    if shape_type == "record":  # Draw a circle
        dc.ellipse(
            (padding, padding, width - padding, height - padding), fill=shape_color
        )
    elif shape_type == "stop":  # Draw a square
        dc.rectangle(
            (padding, padding, width - padding, height - padding), fill=shape_color
        )
    else:  # Default or fallback: a simple rectangle
        dc.rectangle(
            (width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=shape_color
        )
    return image


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
    global phrase_time, phrase_bytes, transcription_history, audio_model, app_icon

    while True:
        if not dictation_active:
            sleep(0.1)
            continue

        try:
            now = datetime.now(timezone.utc)
            if not data_queue.empty():
                logging.debug(f"Processing audio from queue at {now}")
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
                    logging.debug(f"Transcribed text: '{text}'")

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
                    logging.warning("Audio model not loaded yet.")
            else:
                sleep(0.1)  # More responsive sleep
        except Exception as e:
            logging.error(f"Error in process_audio: {e}", exc_info=True)
            sleep(0.1)


def toggle_dictation(icon, item):
    """Toggles dictation on/off."""
    global dictation_active, recorder, source, app_icon
    logging.debug(f"toggle_dictation called. Current state: {dictation_active}")
    dictation_active = not dictation_active
    if dictation_active:
        logging.debug("Dictation started by toggle.")
        if app_icon:
            app_icon.icon = create_tray_image(64, 64, "red", shape_type="stop")
        # The background listener is already started in main().
        # We just need to ensure data is cleared for a fresh start.
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
        logging.debug("Dictation stopped by toggle.")
        if app_icon:
            app_icon.icon = create_tray_image(64, 64, "red", shape_type="record")
        # Consider stopping the listener if you want to save resources,
        # but be careful about restarting it correctly.
        # For now, we just set dictation_active to False and the callback/processing
        # will ignore new data.


def show_exit_dialog_actual(icon_instance=None):  # Parameter for pystray callback
    """Shows an exit confirmation dialog or exits directly."""
    global app_icon, click_timer
    logging.debug("show_exit_dialog_actual called.")

    proceed_to_exit = False
    if TKINTER_AVAILABLE:
        try:
            # Ensure tkinter root window doesn't appear if not already running
            root = tkinter.Tk()
            root.withdraw()  # Hide the main window
            proceed_to_exit = tkinter.messagebox.askyesno(
                title="Exit Dictate App?",
                message="Are you sure you want to exit Dictate App?",
            )
            root.destroy()  # Clean up the hidden root window
        except Exception as e:
            logging.warning(
                f"Could not display tkinter exit dialog: {e}. Exiting directly."
            )
            proceed_to_exit = True  # Fallback to exit if dialog fails
    else:
        logging.info("tkinter not available, exiting directly without confirmation.")
        proceed_to_exit = True

    if proceed_to_exit:
        exit_program(app_icon, None)  # app_icon might be None if called early
    else:
        logging.debug("Exit cancelled by user.")


def delayed_single_click_action(icon_instance):
    """Action to perform for a single click after the double-click window."""
    if app_is_exiting.is_set():  # Don't toggle if we are already exiting
        return
    logging.debug("Delayed single click action triggered.")
    toggle_dictation(icon_instance, None)  # Parameter for pystray callback consistency


def icon_clicked_handler(icon_instance, item=None):  # item unused but pystray passes it
    """Handles icon clicks to differentiate single vs double clicks."""
    global last_click_time, click_timer
    current_time = time.monotonic()
    logging.debug(f"Icon clicked at {current_time}")

    if (
        click_timer and click_timer.is_alive()
    ):  # Timer is active, so this is a second click
        click_timer.cancel()
        click_timer = None
        last_click_time = 0.0  # Reset for next sequence
        logging.debug("Double click detected.")
        show_exit_dialog_actual(icon_instance)
    else:  # First click or click after timer expired
        last_click_time = current_time
        if click_timer:
            click_timer.cancel()  # Cancel any old timer, though it should be None here

        click_timer = threading.Timer(
            EFFECTIVE_DOUBLE_CLICK_INTERVAL,
            delayed_single_click_action,
            args=[icon_instance],
        )
        click_timer.daemon = True  # Ensure timer doesn't block exit
        click_timer.start()
        logging.debug(f"Started click timer for {EFFECTIVE_DOUBLE_CLICK_INTERVAL}s")


def exit_program(icon, item):
    """Stops the program."""
    global dictation_active, app_icon, recorder, click_timer
    logging.debug("exit_program called.")
    app_is_exiting.set()  # Signal that we are exiting

    if click_timer and click_timer.is_alive():
        click_timer.cancel()
        logging.debug("Cancelled pending click_timer on exit.")
    click_timer = None

    dictation_active = False
    if recorder and hasattr(recorder, "stop_listening"):  # Check if listening
        logging.debug("Stopping recorder listener.")
        recorder.stop_listening(wait_for_stop=False)
    if app_icon:
        logging.debug("Stopping app_icon.")
        app_icon.stop()
    logging.info("Exiting application via os._exit(0).")
    os._exit(0)  # Force exit if threads are hanging


def setup_tray_icon():
    """Sets up and runs the system tray icon."""
    global app_icon
    logging.debug("setup_tray_icon called.")
    # Initial icon is 'record' since dictation_active is False initially
    icon_image = create_tray_image(64, 64, "red", shape_type="record")

    if pystray.Icon.HAS_DEFAULT_ACTION:
        menu = pystray.Menu(
            pystray.MenuItem(
                text="Toggle Dictation",
                action=lambda icon, item: icon_clicked_handler(
                    icon
                ),  # Handles single/double click
                default=True,
                visible=False,
            )
        )
    else:
        menu = pystray.Menu(
            pystray.MenuItem(
                "Toggle Dictation",
                lambda: toggle_dictation(
                    app_icon, None
                ),  # Ensure lambda for direct call if needed
                checked=lambda item: dictation_active,
            ),
            pystray.MenuItem(
                "Exit", lambda: exit_program(app_icon, None)
            ),  # Ensure lambda
        )

    app_icon = pystray.Icon("dictate_app", icon_image, "Dictate App", menu)
    logging.debug("pystray.Icon created. Calling app_icon.run().")
    app_icon.run()
    logging.debug(
        "app_icon.run() finished."
    )  # Should not be reached if os._exit is called


def main():
    global audio_model
    global recorder
    global source
    global MODEL_NAME
    global ENERGY_THRESHOLD
    global RECORD_TIMEOUT
    global PHRASE_TIMEOUT
    global DEFAULT_MICROPHONE

    # Configure logging first
    # The default logging level will be WARNING.
    # If -v or --verbose is passed, it will be set to INFO.
    # The initial basicConfig is minimal, we'll add handlers and formatters later
    # if verbose is specified, or rely on the default print for critical errors
    # if not.
    logging.basicConfig(level=logging.WARNING)  # Set a default level
    initialize_double_click_interval()  # Initialize before parser for logging

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable informational logging. Debug logs are not affected by this flag.",
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

    # Configure logging properly based on verbosity
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,  # INFO and above for verbose
            format=(
                "%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("Verbose logging enabled.")
    else:
        logging.basicConfig(
            level=logging.WARNING,  # WARNING and above by default
            format=(
                "%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Setup ALSA error handler (if on Linux)
    # This should be done early, before any library might initialize ALSA.
    if "linux" in platform:
        setup_alsa_error_handler()
    else:
        logging.info("Skipping ALSA error handler setup on non-Linux platform.")

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
    logging.info(f"Loading Whisper model: {MODEL_NAME}{temp}")
    effective_model_name = MODEL_NAME
    if MODEL_NAME not in ["large", "turbo"] and not args.non_english:
        effective_model_name = MODEL_NAME + ".en"

    try:
        audio_model = whisper.load_model(effective_model_name)
        logging.info("Whisper model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading Whisper model: {e}", exc_info=True)
        return

    # Setup SpeechRecognition
    recorder = sr.Recognizer()
    recorder.energy_threshold = ENERGY_THRESHOLD
    recorder.dynamic_energy_threshold = (
        False  # Required for manual energy_threshold setting
    )

    if "linux" in platform:
        mic_name = DEFAULT_MICROPHONE
        if not mic_name or mic_name == "list":
            logging.info("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                logging.info(f'Microphone with name "{name}" found')
            return
        else:
            source = None  # Initialize source
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    logging.info(f"Using microphone: {name}")
                    break
            if source is None:
                logging.error(
                    f"Microphone containing '{mic_name}' not found. Please check"
                    " available microphones."
                )
                logging.info("Available microphone devices are: ")
                for index, name_available in enumerate(
                    sr.Microphone.list_microphone_names()
                ):
                    logging.info(f'Microphone with name "{name_available}" found')
                return
    else:
        source = sr.Microphone(sample_rate=16000)
        logging.info("Using default microphone.")

    with source:
        try:
            recorder.adjust_for_ambient_noise(source, duration=1)  # Adjust for 1 second
            logging.info("Adjusted for ambient noise.")
        except Exception as e:
            logging.warning(f"Could not adjust for ambient noise: {e}", exc_info=True)
            # Continue without adjustment if it fails

    # Start listening in background (but it will only process if dictation_active is
    # True). We start it here so it's ready, and toggle_dictation controls actual
    # processing.
    try:
        # The callback will now check dictation_active before putting data in queue
        recorder.listen_in_background(
            source, record_callback, phrase_time_limit=RECORD_TIMEOUT
        )
        logging.info("Background listener started.")
    except Exception as e:
        logging.error(f"Error starting background listener: {e}", exc_info=True)
        return

    # Start audio processing thread
    audio_thread = threading.Thread(
        target=process_audio, daemon=True, name="AudioProcessThread"
    )
    audio_thread.start()
    logging.info("Audio processing thread started.")

    # Start tray icon
    logging.info("Starting tray icon...")
    setup_tray_icon()  # This will block until exit
    logging.debug("main function finished after setup_tray_icon call.")


if __name__ == "__main__":
    # It's good practice to ensure DISPLAY is set for GUI apps on Linux
    if "linux" in platform and not os.environ.get("DISPLAY"):
        print("Error: DISPLAY environment variable not set. GUI cannot be displayed.")
        print("Please ensure you are running this in a graphical environment.")
        # Logging might not be configured yet if verbose flag isn't parsed.
        # So, print directly.
        # If main() were to proceed, logging would be set up, but we exit here.
    else:
        main()
