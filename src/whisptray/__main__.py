"""whisptray using your microphone to produce keyboard input."""

import argparse
import ctypes
import ctypes.util
import logging
import os
import subprocess
import threading
import time
from sys import platform
import glob

import speech_recognition
from PIL import Image, ImageDraw
from .speech_to_keys import SpeechToKeys

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

# pylint: disable=wrong-import-position
import pystray

try:
    import tkinter
    import tkinter.messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# --- Configuration ---
DEFAULT_MODEL_NAME = "turbo"
DEFAULT_ENERGY_THRESHOLD = 1000
DEFAULT_RECORD_TIMEOUT = 2.0  # Seconds for real-time recording
DEFAULT_PHRASE_TIMEOUT = 3.0  # Seconds of silence before new line
DEFAULT_MICROPHONE = "default"  # For Linux

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
        alsa_logger.info(
            "%s:%d (%s) - err %d: %s", file, line, function, err, formatted_msg
        )
    except (UnicodeDecodeError, AttributeError, TypeError, ValueError) as e:
        # Fallback logging if there's an error within the error handler itself
        print(f"Error in python_alsa_error_handler: {e}")


# Keep a reference to the ctype function object to prevent garbage collection
py_error_handler_ctype = PYTHON_ALSA_ERROR_HANDLER_FUNC(python_alsa_error_handler)


def setup_alsa_error_handler():
    """
    Sets up a custom ALSA error handler using the C helper library.
    """
    if "linux" not in platform:
        logging.debug("Skipping ALSA error handler setup on non-Linux platform.")
        return

    try:
        c_redirect_lib = None
        # Try to load the C library for redirecting ALSA messages
        # Path when installed or running from source with Makefile-built .so
        c_redirect_lib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "alsa_redirect.so"
        )

        if os.path.exists(c_redirect_lib_path):
            try:
                c_redirect_lib = ctypes.CDLL(c_redirect_lib_path)
                logging.debug("Loaded alsa_redirect.so from: %s", c_redirect_lib_path)
            except OSError as e:
                logging.error("Error loading alsa_redirect.so from %s: %s", c_redirect_lib_path, e)
        else:
            # Fallback for when installed as a package, where setuptools renames the .so file
            package_dir = os.path.dirname(os.path.abspath(__file__))
            found_libs = list(glob.glob(os.path.join(package_dir, "alsa_redirect*.so")))
            if found_libs:
                # Take the first one found (should ideally be only one)
                # Sort to get a deterministic choice if multiple somehow exist
                found_libs.sort()
                c_redirect_lib_path_found = found_libs[0]
                try:
                    c_redirect_lib = ctypes.CDLL(c_redirect_lib_path_found)
                    logging.debug("Loaded compiled C extension: %s", c_redirect_lib_path_found)
                except OSError as e:
                    logging.error("Error loading compiled C extension %s: %s", c_redirect_lib_path_found, e)
            else:
                # Last resort: try loading from system path (less reliable)
                try:
                    c_redirect_lib = ctypes.CDLL("alsa_redirect.so")
                    logging.debug("Loaded alsa_redirect.so from system path.")
                except OSError:
                    logging.error(
                        "alsa_redirect.so not found at %s, nor as alsa_redirect*.so in package dir, nor in system paths. ALSA logs "
                        "will not be redirected.",
                        c_redirect_lib_path,
                    )

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
        logging.debug("Registered Python ALSA error handler with C helper.")

        ret = c_redirect_lib.initialize_alsa_error_handling()
        if ret < 0:
            logging.error(
                "C library failed to set ALSA error handler. Error code: %d", ret
            )

    except (OSError, AttributeError, TypeError, ValueError, ctypes.ArgumentError) as e:
        logging.error("Error setting up ALSA error handler: %s", e, exc_info=True)


def configure_logging(verbose: bool):
    """
    Configures logging based on the verbose flag.
    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format=(
                "%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.debug("Verbose logging enabled.")
    else:
        logging.basicConfig(
            level=logging.INFO,
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
        logging.debug("Skipping ALSA error handler setup on non-Linux platform.")


def parse_args():
    """
    Parses the command line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mic",
        default=DEFAULT_MICROPHONE,
        help="Default microphone name for SpeechRecognition. "
        "Run this with 'list' to view available Microphones.",
        type=str,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
    )
    parser.add_argument(
        "--non_english", action="store_true", help="Don't use the english model."
    )
    parser.add_argument(
        "--energy_threshold",
        default=DEFAULT_ENERGY_THRESHOLD,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=DEFAULT_RECORD_TIMEOUT,
        help="How real time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=DEFAULT_PHRASE_TIMEOUT,
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
    return args


def open_microphone(mic_name: str) -> speech_recognition.Microphone:
    """
    Opens a microphone based on the microphone name.
    """
    assert mic_name

    result = None
    if "linux" in platform:
        for index, name in enumerate(
            speech_recognition.Microphone.list_microphone_names()
        ):
            if mic_name in name:
                result = speech_recognition.Microphone(
                    sample_rate=16000, device_index=index
                )
                logging.debug("Using microphone: %s", name)
                break
        if result is None:
            logging.error(
                "Microphone containing '%s' not found. Please check available"
                " microphones.",
                mic_name,
            )
            logging.debug("Available microphone devices are: ")
            for index, name_available in enumerate(
                speech_recognition.Microphone.list_microphone_names()
            ):
                logging.debug('Microphone with name "%s" found', name_available)
    else:
        result = speech_recognition.Microphone(sample_rate=16000)
        logging.debug("Using default microphone.")

    return result


# pylint: disable=too-many-instance-attributes
class whisptrayGui:
    """
    Class to run the whisptray App.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self, mic_name, model_name, energy_threshold, record_timeout, phrase_timeout
    ):
        self.last_click_time = 0.0
        self.click_timer = None
        # Default in seconds, updated by system settings
        self.effective_double_click_interval = 0.5
        self.app_is_exiting = threading.Event()
        self.app_icon = None # Initialize to None

        source = open_microphone(mic_name)
        if source is None:
            raise ValueError("No microphone found")

        self.speech_to_keys = SpeechToKeys(
            model_name, energy_threshold, record_timeout, phrase_timeout, source
        )
        self._initialize_double_click_interval()
        # Start tray icon
        logging.debug("Starting tray icon...")
        self._setup_tray_icon()  # This will set self.app_icon

        # Start icon health check thread
        if self.app_icon:
            self.health_check_thread = threading.Thread(
                target=self._icon_health_check, daemon=True, name="IconHealthCheckThread"
            )
            self.health_check_thread.start()
            logging.debug("Icon health check thread started.")
        else:
            logging.error(
                "App icon not initialized properly. Health check thread not started. "
                "The application might not function correctly."
            )

    def run(self):
        """
        Runs the whisptray App.
        """
        logging.debug("Calling app_icon.run().")
        self.app_icon.run()
        logging.debug("app_icon.run() finished.")

    def toggle_dictation(self):
        """Toggles dictation on/off."""
        logging.debug(
            "toggle_dictation called. Current state: %s", self.speech_to_keys.enabled
        )
        self.speech_to_keys.enabled = not self.speech_to_keys.enabled
        if self.speech_to_keys.enabled:
            logging.debug("Dictation started by toggle.")
            if self.app_icon:
                self.app_icon.icon = whisptrayGui._create_tray_image("record")

        else:
            logging.debug("Dictation stopped by toggle.")
            if self.app_icon:
                self.app_icon.icon = whisptrayGui._create_tray_image("stop")
            # Consider stopping the listener if you want to save resources,
            # but be careful about restarting it correctly.
            # For now, we just set dictation_active to False and the callback/processing
            # will ignore new data.

    def exit_program(self):
        """Stops the program."""
        logging.debug("exit_program called.")
        if self.app_is_exiting.is_set():
            return

        self.app_is_exiting.set()  # Signal that we are exiting

        if self.click_timer and self.click_timer.is_alive():
            self.click_timer.cancel()
            logging.debug("Cancelled pending click_timer on exit.")
        self.click_timer = None
        self.speech_to_keys.shutdown()

        if self.app_icon:
            logging.debug("Disabling tray icon.")
            self.app_icon.stop()

    def _setup_tray_icon(self):
        """Sets up and runs the system tray icon."""
        logging.debug("setup_tray_icon called.")
        # Initial icon is 'stop' since dictation_active is False initially
        icon_image = whisptrayGui._create_tray_image("stop")

        if pystray.Icon.HAS_DEFAULT_ACTION:
            menu = pystray.Menu(
                pystray.MenuItem(
                    text="Toggle Dictation",
                    action=self._icon_clicked_handler,
                    default=True,
                    visible=False,
                )
            )
        else:
            menu = pystray.Menu(
                pystray.MenuItem(
                    "Toggle Dictation",
                    self.toggle_dictation,
                    checked=lambda item: self.speech_to_keys.enabled,
                ),
                pystray.MenuItem("Exit", self.exit_program),
            )

        self.app_icon = pystray.Icon("whisptray_app", icon_image, "whisptray App", menu)
        logging.debug("pystray.Icon created.")

    @staticmethod
    def _get_system_double_click_time() -> float | None:
        """Tries to get the system's double-click time in seconds."""
        try:
            if platform in ("linux", "linux2"):
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
                        # Neither GSettings nor xrdb succeeded.
                        logging.debug(
                            "Could not determine double-click time from GSettings or"
                            " xrdb."
                        )
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
                # Getting this programmatically on macOS is non-trivial. Default.
                logging.debug("Using default double-click time for macOS.")
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            IndexError,
            subprocess.TimeoutExpired,
            OSError,
        ) as e:
            logging.warning("Could not query system double-click time: %s", e)
        return None

    def _initialize_double_click_interval(self):
        """Initializes the double-click interval, falling back to default if needed."""
        system_interval = whisptrayGui._get_system_double_click_time()
        if (
            system_interval is not None and 0.1 <= system_interval <= 2.0
        ):  # Sanity check interval
            self.effective_double_click_interval = system_interval
            logging.debug(
                "Using system double-click interval: %.2fs",
                self.effective_double_click_interval,
            )
        else:
            logging.debug(
                "Using default double-click interval: %.2fs",
                self.effective_double_click_interval,
            )

    @staticmethod
    def _create_tray_image(shape_type):
        """Creates an image for the tray icon (record or stop button) with a transparent
        background."""
        image = Image.new("RGB", (128, 128), (0, 0, 0))
        dc = ImageDraw.Draw(image)
        padding = int(128 * 0.2)  # Add padding around the shape

        if shape_type == "record":
            # Draw a circle
            dc.ellipse((padding, padding, 128 - padding, 128 - padding), fill="red")
        else:  # shape_type == "stop"
            # Draw a square
            dc.rectangle((padding, padding, 128 - padding, 128 - padding), fill="white")
        return image

    def _show_exit_dialog_actual(self):
        """Shows an exit confirmation dialog or exits directly."""
        logging.debug("show_exit_dialog_actual called.")

        proceed_to_exit = False
        if TKINTER_AVAILABLE:
            try:
                # Ensure tkinter root window doesn't appear if not already running
                root = tkinter.Tk()
                root.withdraw()  # Hide the main window
                proceed_to_exit = tkinter.messagebox.askyesno(
                    title="Exit whisptray App?",
                    message="Are you sure you want to exit whisptray App?",
                )
                root.destroy()  # Clean up the hidden root window
            except (tkinter.TclError, RuntimeError) as e:
                logging.warning(
                    "Could not display tkinter exit dialog: %s. Exiting directly.", e
                )
                proceed_to_exit = True  # Fallback to exit if dialog fails
        else:
            logging.debug(
                "tkinter not available, exiting directly without confirmation."
            )
            proceed_to_exit = True

        if proceed_to_exit:
            self.exit_program()  # app_icon might be None if called early
        else:
            logging.debug("Exit cancelled by user.")

    def _delayed_single_click_action(self):
        """Action to perform for a single click after the double-click window."""
        if self.app_is_exiting.is_set():  # Don't toggle if we are already exiting
            return
        logging.debug("Delayed single click action triggered.")
        self.toggle_dictation()

    def _icon_clicked_handler(self):  # item unused but pystray passes it
        """Handles icon clicks to differentiate single vs double clicks."""
        current_time = time.monotonic()
        logging.debug("Icon clicked at %s", current_time)

        if (
            self.click_timer and self.click_timer.is_alive()
        ):  # Timer is active, so this is a second click
            self.click_timer.cancel()
            self.click_timer = None
            self.last_click_time = 0.0  # Reset for next sequence
            logging.debug("Double click detected.")
            self._show_exit_dialog_actual()
        else:  # First click or click after timer expired
            self.last_click_time = current_time
            # Cancel any old timer, though it should be None here
            if self.click_timer:
                self.click_timer.cancel()

            self.click_timer = threading.Timer(
                self.effective_double_click_interval,
                self._delayed_single_click_action,
                args=[],
            )
            self.click_timer.daemon = True  # Ensure timer doesn't block exit
            self.click_timer.start()
            logging.debug(
                "Started click timer for %ss", self.effective_double_click_interval
            )

    def _handle_thread_exception(self, args):
        """Handles exceptions in threads."""
        thread_name = threading.current_thread().name
        logging.error(
            f"Exception in thread '{thread_name}': {args[1]}. Initiating application exit.",
            exc_info=True
        )
        self.exit_program()

    def _icon_health_check(self):
        """Periodically checks the health of the pystray icon and exits on failure."""
        if not self.app_icon:
            logging.warning("Icon health check: app_icon is None at start. Thread exiting.")
            return

        logging.debug("Icon health check loop starting.")
        while not self.app_is_exiting.is_set():
            try:
                # The act of getting and setting the icon can reveal issues with pystray.
                current_icon_image = self.app_icon.icon
                self.app_icon.icon = current_icon_image
            except Exception as e:  # Catching a broad exception as pystray errors can be varied
                logging.error(
                    "Pystray icon health check failed: %s. Initiating application exit.",
                    e,
                    exc_info=True
                )
                self.exit_program()
                break  # Exit health check loop

            time.sleep(1)  # Check every second

        logging.debug("Icon health check loop finished.")


def main():
    """
    Main function to run the whisptray App.
    """
    args = parse_args()
    configure_logging(args.verbose)

    if args.mic == "list":
        print(
            "Available microphones: ",
            ", ".join(speech_recognition.Microphone.list_microphone_names()),
        )
        return

    model_name = args.model
    if not args.non_english and model_name not in ["large", "turbo"]:
        model_name += ".en"

    gui = whisptrayGui(
        args.mic,
        model_name,
        args.energy_threshold,
        args.record_timeout,
        args.phrase_timeout,
    )
    # This will block until exit
    gui.run()


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
