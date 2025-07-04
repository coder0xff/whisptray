from whisptray.rate_limited_keyboard import Controller as KeyboardController

if __name__ == "__main__":
    keys = KeyboardController(1)
    keys.type("Test")
