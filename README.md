# Whisptray

A simple dictation program that uses OpenAI's Whisper for speech-to-text, 
`pynput` for simulating keyboard input, and `pystray` for a system tray icon.

## Features

- Real-time dictation using Whisper.
- Types recognized text into the currently active application.
- System tray icon to toggle dictation and exit the application.
- Configurable Whisper model and audio parameters via command-line arguments.

## Installation

**Prerequisites:**

1.  **A working microphone** recognized by your operating system.

2.  **`ffmpeg`** (for Whisper audio processing):
    *   **Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

3.  **Audio System Libraries (Linux - if `sounddevice` fails):**
    `whisptray` uses the `sounddevice` library, which usually installs smoothly. However, on some minimal Linux systems, or if you encounter issues with `sounddevice` not finding audio devices, you might need to ensure underlying system audio libraries are present. The most common one is `libportaudio2`:
    *   **Debian/Ubuntu:** `sudo apt-get install libportaudio2`
    *   For other distributions, search for their PortAudio package.

**Installation Command:**

Once all prerequisites are met, you can install `whisptray`:

```bash
pip install whisptray
```

If `whisptray` fails to start with errors related to audio input (e.g., cannot find microphone, errors from `sounddevice` despite installing prerequisites), please double-check:
*   Your microphone is correctly connected and configured in your OS sound settings.
*   Your Python environment is correctly set up and `sounddevice` installed properly within it.

## Usage

Click the tray icon to toggle dictation. Double click to exit.

You can customize the behavior using command-line arguments. For example, to use a specific microphone (ID 2, found by running with `--device list`) and a different energy multiplier:

```bash
whisptray --device 2 --energy_multiplier 2.0
```

**Available arguments:**

*   `--device DEVICE`: Microphone name or ID to use (e.g., "pulse", "USB Microphone", or an integer ID like `1`). 
    Pass `list` to see available microphone IDs and names. If omitted, the system default microphone is used.
*   `--model MODEL`: Whisper model to use. (choices: "tiny", "base", "small", "medium", "large", "turbo"; default: "turbo"). 
    Non-English models are generally the base versions (e.g., "small" not "small.en"). "large" and "turbo" are multilingual by default.
*   `--ambient_duration SECONDS`: Duration (in seconds) to measure ambient noise before starting dictation. This helps set a baseline for voice activity detection. (default: 1.0)
*   `--energy_multiplier MULTIPLIER`: Multiplier applied to the measured ambient noise level to set the energy threshold for voice activity detection. Higher values are less sensitive. (default: 1.5)
*   `-v`, `--verbose`: Enable more detailed informational logging.
*   `--version`: Show program's version number and exit.

## Development

1. Ensure thesystem prerequisites are installed as described in the Installation section.
2. Clone this repository:
   ```bash
   git clone https://github.com/coder0xff/whisptray.git # Replace with your repo URL
   cd whisptray
   ```
3. `make develop`

