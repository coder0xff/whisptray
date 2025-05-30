# Whisptray

A simple dictation program that uses OpenAI's Whisper for speech-to-text, 
`pynput` for simulating keyboard input, and `pystray` for a system tray icon.

## Features

- Real-time dictation using Whisper.
- Types recognized text into the currently active application.
- System tray icon to toggle dictation and exit the application.
- Configurable Whisper model and audio parameters via command-line arguments.

## Installation And Use

```
pip install whisptray
whisptray
```

Click the tray icon to toggle dictation. Double click to exit.

You can customize the behavior using command-line arguments:

```bash
whisptray --model small --energy_threshold 1200
```

Available arguments:
- `--mic`: Name of the microphone to use. Pass "list" to see available mics. (default: "default)
- `--model`: Whisper model to use (choices: "tiny", "base", "small", "medium", "large", "turbo" - default: "turbo").
- `--non_english`: Use the multilingual model variant (if applicable for the chosen size).
- `--energy_threshold`: Energy level for mic to detect (default: 1000).
- `--record_timeout`: How real-time the recording is in seconds (default: 2.0).
- `--phrase_timeout`: Silence duration before a new phrase is considered (default: 3.0).
- `--default_microphone` (Linux only): Name or part of the name of the microphone to use (default: 'pulse'). Use `whisptray --default_microphone list` to see available microphones.

## Development

1. Clone this repository:
   ```bash
   git clone https://github.com/coder0xff/whisptray.git # Replace with your repo URL
   cd whisptray
   ```

2. **Linux System Dependency (PortAudio for PyAudio):**
   `PyAudio` is a dependency for microphone access and requires the PortAudio library. If installation in the previous step fails or `PyAudio` has issues, you may need to install the development headers.
   - **Debian/Ubuntu-based systems**:
     ```bash
     sudo apt-get update && sudo apt-get install portaudio19-dev
     ```
   - For other distributions, please consult their package manager for the appropriate PortAudio development package.

3. **System Dependency (ffmpeg for Whisper):**
   Ensure `ffmpeg` is installed on your system, as Whisper requires it for audio processing.
   - **Debian/Ubuntu-based systems**: `sudo apt update && sudo apt install ffmpeg`

4. `make develop`
