import speech_recognition
import logging
import sys
import platform
import termios
import tty
from whisptray.speech_to_keys import SpeechToKeys
from whisptray.__main__ import open_microphone, DEFAULT_MODEL_NAME, DEFAULT_ENERGY_THRESHOLD, DEFAULT_RECORD_TIMEOUT, DEFAULT_PHRASE_TIMEOUT

# Configure logging for testing
logging.basicConfig(level=logging.DEBUG)

def main():
    """
    Test program for SpeechToKeys.
    Creates a SpeechToKeys object, turns it on, waits for 10 chars from stdin, and exits.
    """
    mic_name = "default"  # Or use a specific microphone name
    source = open_microphone(mic_name)
    if source is None:
        logging.error(f"Microphone '{mic_name}' not found. Exiting.")
        return

    speech_to_keys = SpeechToKeys(
        model_name=DEFAULT_MODEL_NAME,
        energy_threshold=DEFAULT_ENERGY_THRESHOLD,
        record_timeout=DEFAULT_RECORD_TIMEOUT,
        phrase_timeout=DEFAULT_PHRASE_TIMEOUT,
        source=source,
    )

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
                    # logging.debug(f"Char read: '{char}' (ASCII: {ord(char)}), Total chars: {chars_typed_count}")
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
                # logging.debug(f"Line read: '{line_input}', Total chars: {chars_typed_count}")

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
