"""Test program for SpeechToKeys."""

import logging
import sys
import platform
import termios
import tty
from whisptray.speech_to_keys import SpeechToKeys

logging.basicConfig(level=logging.DEBUG)

def main():
    """
    Creates a SpeechToKeys object, turns it on, waits for 10 chars from stdin, and exits.
    """
    speech_to_keys = SpeechToKeys()

    print("Turning on dictation. Speak into the microphone.")
    print("The script will auto-exit after you type 10 characters into this terminal.")
    speech_to_keys.enabled = True

    chars_typed_count = 0
    old_settings = None
    fd = None

    try:
        logging.info("Waiting for 10 characters to be typed into stdin...")
        if platform.system() == "Linux" or platform.system() == "Darwin":
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while chars_typed_count < 10:
                    char = sys.stdin.read(1)
                    if not char:  # EOF
                        logging.warning("EOF received from stdin. Exiting loop.")
                        break
                    if ord(char) == 3: # CTRL+C
                        raise KeyboardInterrupt
                    chars_typed_count += 1
            finally:
                if old_settings: # Ensure settings are restored
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        else: # Fallback for non-Unix systems (like Windows)
            logging.warning("Non-Unix system detected. Reading input line by line (press Enter to submit).")
            while chars_typed_count < 10:
                line_input = input()
                if not line_input: #EOF or empty line considered as break
                    logging.warning("Empty input or EOF received. Exiting loop.")
                    break
                chars_typed_count += len(line_input)

        logging.info(f"Read approximately {chars_typed_count} characters. Proceeding to shutdown.")

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
    finally:
        if old_settings and fd: # Ensure settings are restored if modified
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("Turning off dictation and shutting down.")
        speech_to_keys.enabled = False
        speech_to_keys.shutdown()
        print("Exited.")

if __name__ == "__main__":
    main()
