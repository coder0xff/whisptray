import time
import threading
from pynput import keyboard

# Event to signal that input() has completed
input_completed_event = threading.Event()
# To store the value received by input(), using a list for mutability
received_input_value = [None]

def read_input_in_thread():
    """Calls input() and signals when it returns, storing the value."""
    print("[INPUT_THREAD] Waiting for input() call to be satisfied...")
    try:
        # The prompt will be visible. The simulated Enter should satisfy this.
        val = input("Test: Simulated 'Enter' should satisfy this input: ")
        print(f"[INPUT_THREAD] input() returned: '{val}'")
        received_input_value[0] = val
    except Exception as e:
        print(f"[INPUT_THREAD] Exception in input thread: {e}")
    finally:
        input_completed_event.set() # Signal completion or error

def main_input_test():
    print("[MAIN] Starting input test using built-in input()...")

    # Start the thread that will block on input()
    input_thread = threading.Thread(target=read_input_in_thread, daemon=True)
    input_thread.start()

    # Give the input() in the other thread a moment to start and block
    print("[MAIN] Allowing time for input() to block...")
    time.sleep(1.0) # Adjust if needed, ensures input() is waiting

    # Simulate pressing the Enter key
    # This requires the terminal running the script to be the active window.
    print("[MAIN] Simulating 'Enter' key press using pynput.Controller...")
    try:
        controller = keyboard.Controller()
        controller.press(keyboard.Key.enter)
        controller.release(keyboard.Key.enter)
        print("[MAIN] 'Enter' key press simulated.")
    except Exception as e:
        print(f"[MAIN] Failed to simulate 'Enter' key press: {e}")
        # If controller fails, the event won't be set by simulation, test will likely fail on timeout.

    # Wait for the input_completed_event to be set by the input_thread
    print("[MAIN] Waiting for input() to complete...")
    if input_completed_event.wait(timeout=3.0):  # Timeout of 3 seconds
        if received_input_value[0] == "":
            print("[MAIN] SUCCESS: input() was satisfied by simulated 'Enter' and returned an empty string.")
        else:
            print(f"[MAIN] FAILURE: input() completed, but returned '{received_input_value[0]}' instead of an empty string.")
    else:
        print("[MAIN] FAILURE: input() did NOT complete within the timeout after simulated 'Enter' press.")
        print("       Ensure the terminal window running the script is active/focused when the key is simulated.")

    # Clean up the thread
    print("[MAIN] Ensuring input thread is joined...")
    input_thread.join(timeout=1.0)
    if input_thread.is_alive():
        print("[MAIN] Warning: Input thread did not terminate cleanly.")
    else:
        print("[MAIN] Input thread joined successfully.")
    
    print("[MAIN] Input test finished.")

if __name__ == "__main__":
    main_input_test() 