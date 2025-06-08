from contextlib import contextmanager
from threading import Thread
from time import sleep
import os

@contextmanager
def essential_thread():
    """A context manager that causes the program to exit if the thread exits
    unexpectedly."""
    failed = True
    try:
        yield
        failed = False
    finally:
        if failed:
            # Exit after a short delay to allow the exception to be logged.
            def exit_soon():
                sleep(1)
                os._exit(1)
            thread = Thread(target=exit_soon, daemon=True)
            thread.start()
