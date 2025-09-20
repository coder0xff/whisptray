# Whisptray

A fast, accurate, and private speech-to-text dictation program.

Speak into your microphone to type text into any program on your computer. Whisptray combines the state of the art Whisper audio-to-text engine with pynput and pystray for fast, accurate, and private dictation. Whisptray is intended for Linux (major commercial OSes include dictation already,) but should work on any platform with the requisite support.

**IMPORTANT**: Your system requires the correct audio capabilities to be setup before it can use Whisptray. See the [Installation](#installation) section for instructions.

After installation, run `whisptray`. Whisptray will create a record button, a red circle, in the corner of your screen. Dication begins when you click the record button. The record button changes into a stop button, a white square, to indicate that recording has started. Speak into your microphone, and the text will be entered into your currently active program. Click the stop button to stop dictation. You can double click the button at any time to exit Whisptray.

## Recommended System Requirements

Whisptray works best with a Pytorch-supported GPU.

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

Run:

```bash
whisptray
```

Click the tray icon to toggle dictation. Double click to exit.

If your computer doesn't have a modern GPU, try:

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
*   `--max_key_rate COUNT`: The maximum rate of generated keyboard events per second. Higher values are more responsive. Lower values give better compatibility with slow apps. Robust apps can tolerate up to `--max-key-rate 1000`, while the glitchiest only work as low as `--max-key-rate 50`.
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

