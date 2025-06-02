# Whisptray

A simple dictation program that uses OpenAI's Whisper for speech-to-text, 
`pynput` for simulating keyboard input, and `pystray` for a system tray icon.

## Features

- Real-time dictation using Whisper.
- Types recognized text into the currently active application.
- System tray icon to toggle dictation and exit the application.
- Configurable Whisper model and audio parameters via command-line arguments.

## Installation

Some common prerequisites are required before installation.

**Prerequisites for Ubuntu:**

```bash
sudo apt update && sudo apt install build-essential python3-dev libportaudio2 ffmpeg
```

**Prerequisites for Fedora:**

```bash
sudo dnf groupinstall "Development Tools" && sudo dnf install python3-devel alsa-lib-devel ffmpeg
```

**Installation Command:**

```bash
pip3 install whisptray
```

Or if you have a newer system that requires pipx to install Python applications:

```bash
pipx install whisptray
```

## Usage

```bash
whisptray
```

Click the tray icon to toggle dictation. Double click to exit.

If you have an older computer, try:

```bash
whisptray --model tiny
```

This and several other model options offer various performance vs. accurancy tradeoffs. The `turbo` model is the best quality, and is the default. Use the `--help` options for more details.

If `whisptray` fails to start with errors related to audio input (e.g., cannot find microphone, errors from `sounddevice` despite installing prerequisites), please double-check:
*   Your microphone is correctly connected and configured in your OS sound settings.
*   Your Python environment is correctly set up and `sounddevice` installed properly within it.

## Advanced Usage

You can customize the behavior using command-line arguments. For example, to use a specific microphone (ID 2, found by running with `--device list`) and a different energy multiplier:

```bash
whisptray --device 2 --energy_multiplier 2.0
```

**Available arguments:**

*   `--device DEVICE`: Microphone name or ID to use (e.g., "pulse", "USB Microphone", or an integer ID like `1`). 
    Pass `list` to see available microphone IDs and names. If omitted, the system default microphone is used.
*   `--model MODEL`: Whisper model to use. (choices: "tiny", "base", "small", "medium", "large", "turbo"; default: "turbo").
*   `--max_key_rate COUNT`: The maximum rate of generated keystrokes per second. Higher values are more responsive. Lower values give better compatibility with slow apps. (e.g. 1000 for native apps, 250 for glitchy web apps.)
*   `--ambient_duration SECONDS`: Duration (in seconds) to measure ambient noise before starting dictation. This helps set a baseline for voice activity detection. (default: 1.0)
*   `--energy_multiplier MULTIPLIER`: Multiplier applied to the measured ambient noise level to set the energy threshold for voice activity detection. Higher values are less sensitive. (default: 1.5)
*   `-v`, `--verbose`: Enable debug logging.
*   `--version`: Show program's version number and exit.
*   `--help`: Show the program's options and exit.

## Troubleshooting

If the text appears garbled in some applications, trying specifying a slower key rate using the `--max_key_rate` command line argument.

## Development

1. Ensure thesystem prerequisites are installed as described in the Installation section.
2. Clone this repository:
   ```bash
   git clone https://github.com/coder0xff/whisptray.git # Replace with your repo URL
   cd whisptray
   ```
3. `make develop`

