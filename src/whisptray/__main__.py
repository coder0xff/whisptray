"""whisptray using your microphone to produce keyboard input."""

import argparse
import importlib.metadata
import logging
import os
import subprocess
import threading
import time
from sys import platform
from typing import Optional

from PIL import Image, ImageDraw

from .alsa_error_handler import setup_alsa_error_handler, teardown_alsa_error_handler
from .speech_to_keys import SpeechToKeys

# Conditional import for tkinter
try:
    import tkinter
    import tkinter.messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Don't use AppIndicator on Linux, because it doesn't support direct icon clicks.
if "linux" in platform and "PYSTRAY_BACKEND" not in os.environ:
    os.environ["PYSTRAY_BACKEND"] = "xorg"

# pylint: disable=wrong-import-position,wrong-import-order
import pystray

try:
    import tkinter
    import tkinter.messagebox

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# --- Configuration ---
DEFAULT_DEVICE = None  # Sounddevice will use default device
DEFAULT_MODEL_NAME = "turbo"
DEFAULT_AMBIENT_DURATION = 1.0  # Default for ambient_duration
DEFAULT_ENERGY_MULTIPLIER = 1.5  # Default for energy_threshold_multiplier
DEFAULT_MAX_KEY_RATE = 250.0  # Default for max_key_rate


def _configure_logging(verbose: bool):
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format=(
                "%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("Verbose logging enabled.")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if "linux" in platform:
        logging.info("Setting up ALSA error handler.")
        setup_alsa_error_handler()


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="whisptray is a tool that uses your microphone to produce keyboard input.",
        epilog="See https://github.com/coder0xff/whisptray for more information.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Microphone name or ID for sounddevice. "
        "Run with 'list' to view available Microphones.",
        type=str,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
    )
    parser.add_argument(
        "--max-key-rate",
        default=DEFAULT_MAX_KEY_RATE,
        help="Maximum key press rate in characters per second.",
        type=float,
    )
    parser.add_argument(
        "--ambient_duration",
        default=DEFAULT_AMBIENT_DURATION,
        help="Duration of time to measure ambient noise for energy threshold.",
        type=float,
    )
    parser.add_argument(
        "--energy_multiplier",
        default=DEFAULT_ENERGY_MULTIPLIER,
        help="Multiplier to ambient noise for energy threshold of speech detection.",
        type=float,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('whisptray')}",
        help="Show program's version number and exit.",
    )
    args = parser.parse_args()
    return args


class WhisptrayGui:
    """
    Class to run the whisptray App.
    """

    def __init__(
        self,
        device: Optional[str | int],
        model_name: str,
        max_key_rate: float,
        ambient_duration: float,
        energy_multiplier: float,  # Changed from energy_threshold
    ):
        self.last_click_time = 0.0
        self.click_timer = None
        # Default in seconds, updated by system settings
        self.effective_double_click_interval = (
            WhisptrayGui._get_system_double_click_time()
        )
        self.app_is_exiting = threading.Event()
        self.app_icon = None  # Initialize to None

        try:
            device = int(device)  # type: ignore
        except (TypeError, ValueError):
            pass

        self.speech_to_keys = SpeechToKeys(
            model_name=model_name,
            device=device,
            max_key_rate=max_key_rate,
            ambient_duration=ambient_duration,
            activity_ambient_multiplier=energy_multiplier,
        )

        logging.info("Starting tray icon...")
        self._setup_tray_icon()  # This will set self.app_icon

        self.health_check_thread = threading.Thread(
            target=self._icon_health_check,
            daemon=True,
            name="IconHealthCheckThread",
        )
        self.health_check_thread.start()
        logging.info("Icon health check thread started.")

    def run(self):
        """
        Runs the whisptray App.
        """
        logging.info("Calling app_icon.run().")
        self.app_icon.run()
        logging.info("app_icon.run() finished.")

    def toggle_dictation(self):
        """Toggles dictation on/off."""
        self.speech_to_keys.enabled = not self.speech_to_keys.enabled
        if self.speech_to_keys.enabled:
            logging.info("Dictation started by toggle.")
            if self.app_icon:
                self.app_icon.icon = WhisptrayGui._create_tray_image("record")

        else:
            logging.info("Dictation stopped by toggle.")
            if self.app_icon:
                self.app_icon.icon = WhisptrayGui._create_tray_image("stop")

    def exit_program(self):
        """Stops the program."""
        logging.info("exit_program called.")
        if self.app_is_exiting.is_set():
            return

        self.app_is_exiting.set()  # Signal that we are exiting

        if self.click_timer and self.click_timer.is_alive():
            self.click_timer.cancel()
            logging.info("Cancelled pending click_timer on exit.")
        self.click_timer = None
        self.speech_to_keys.shutdown()

        if "linux" in platform:
            logging.info("Tearing down ALSA error handler.")
            teardown_alsa_error_handler()

        if self.app_icon:
            logging.info("Shutting down tray icon.")
            self.app_icon.stop()

    def _setup_tray_icon(self):
        """Sets up and runs the system tray icon."""
        logging.info("setup_tray_icon called.")
        # Initial icon is 'stop' since dictation_active is False initially
        icon_image = WhisptrayGui._create_tray_image("stop")

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
        logging.info("pystray.Icon created.")

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
                        logging.info(
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
                logging.info("Using default double-click time for macOS.")
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
            IndexError,
            subprocess.TimeoutExpired,
            OSError,
        ) as e:
            logging.warning("Could not query system double-click time: %s", e)
        return 0.5

    @staticmethod
    def _create_tray_image(shape_type):
        """Creates an image for the tray icon (record or stop button) with a transparent
        background."""
        image = Image.new("RGB", (128, 128), (0, 0, 0))
        dc = ImageDraw.Draw(image)

        if shape_type == "record":
            dc.ellipse((0, 0, 128, 128), fill="red")
        else:  # shape_type == "stop"
            dc.rectangle((0, 0, 128, 128), fill="white")
        return image

    def _show_exit_dialog_actual(self):
        """Shows an exit confirmation dialog or exits directly."""
        logging.info("show_exit_dialog_actual called.")

        proceed_to_exit = True
        if TKINTER_AVAILABLE:
            # Ensure tkinter root window doesn't appear if not already running
            root = tkinter.Tk()
            root.withdraw()  # Hide the main window
            proceed_to_exit = tkinter.messagebox.askyesno(
                title="Exit whisptray App?",
                message="Are you sure you want to exit whisptray App?",
            )
            root.destroy()  # Clean up the hidden root window
        else:
            logging.info(
                "tkinter not available, exiting directly without confirmation."
            )

        if proceed_to_exit:
            self.exit_program()  # app_icon might be None if called early
        else:
            logging.info("Exit cancelled by user.")

    def _delayed_single_click_action(self):
        """Action to perform for a single click after the double-click window."""
        if self.app_is_exiting.is_set():  # Don't toggle if we are already exiting
            return
        logging.info("Delayed single click action triggered.")
        self.toggle_dictation()

    def _icon_clicked_handler(self):  # item unused but pystray passes it
        """Handles icon clicks to differentiate single vs double clicks."""
        current_time = time.monotonic()
        logging.info("Icon clicked at %s", current_time)

        if (
            self.click_timer and self.click_timer.is_alive()
        ):  # Timer is active, so this is a second click
            self.click_timer.cancel()
            self.click_timer = None
            self.last_click_time = 0.0  # Reset for next sequence
            logging.info("Double click detected.")
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
            logging.info(
                "Started click timer for %ss", self.effective_double_click_interval
            )

    def _icon_health_check(self):
        """Periodically checks the health of the pystray icon and exits on failure."""
        assert self.app_icon is not None, "app_icon failed"

        logging.info("Icon health check loop starting.")
        while not self.app_is_exiting.is_set():
            # The act of getting and setting the icon can help issues with pystray
            self.app_icon.icon = self.app_icon.icon

            time.sleep(1)

        logging.info("Icon health check loop finished.")


def main():
    """
    Main function to run the whisptray App.
    """
    if "linux" in platform and not os.environ.get("DISPLAY"):
        print("Error: DISPLAY environment variable not set. GUI cannot be displayed.")
        print("Please ensure you are running this in a graphical environment.")
        return 1

    args = _parse_args()
    _configure_logging(args.verbose)

    if isinstance(args.device, str) and args.device.lower() == "list":
        SpeechToKeys.query_devices()
        return 0

    model_name = args.model
    if not args.model.endswith(".en") and args.model not in ["large", "turbo"]:
        model_name += ".en"

    gui = WhisptrayGui(
        args.device,
        model_name,
        args.max_key_rate,
        args.ambient_duration,
        args.energy_multiplier,
    )
    gui.run()
    return 0


if __name__ == "__main__":
    main()
