import pynput.keyboard
from time import sleep
from typing import Callable


class Controller:
    _delay: float
    _wrapped: pynput.keyboard.Controller
    _original_handle_method: Callable

    def __init__(self, delay):
        self._delay = delay
        self._wrapped = pynput.keyboard.Controller()
        self._original_handle_method = self._wrapped._handle
        self._wrapped._handle = self._handle

    def _handle(self, key, is_press):
        result = self._original_handle_method(key, is_press)
        sleep(self._delay)
        return result
        
    def __del__(self):
        self._wrapped._handle = self._original_handle_method

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)
