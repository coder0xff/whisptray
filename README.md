# whisptray App

A simple dictation program that uses OpenAI's Whisper for speech-to-text, 
`pynput` for simulating keyboard input, and `pystray` for a system tray icon.

## Features

- Real-time dictation using Whisper.
- Types recognized text into the currently active application.
- System tray icon to toggle dictation and exit the application.
- Configurable Whisper model and audio parameters via command-line arguments.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/whisptray_app.git # Replace with your repo URL
   cd whisptray_app
   ```

2. It is recommended to use a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `source .venv\Scripts\activate`
   ```

3. Install the package:
   This will install the `whisptray` command and its dependencies.
   ```bash
   pip install .
   ```
   Or for development (allows editing the code without reinstalling):
   ```bash
   pip install -e .[dev]
   ```

4. **Linux System Dependency (PortAudio for PyAudio):**
   `PyAudio` is a dependency for microphone access and requires the PortAudio library. If installation in the previous step fails or `PyAudio` has issues, you may need to install the development headers.
   - **Debian/Ubuntu-based systems**:
     ```bash
     sudo apt-get update && sudo apt-get install portaudio19-dev
     ```
   - For other distributions, please consult their package manager for the appropriate PortAudio development package.

5. **System Dependency (ffmpeg for Whisper):**
   Ensure `ffmpeg` is installed on your system, as Whisper requires it for audio processing.
   - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`


6. **System Dependency (AppIndicator & PyGObject for Tray Icon on Linux):**
   For the system tray icon to function reliably on many Linux desktop environments (especially those using GNOME Shell), `pystray` works best with the `AppIndicator` backend. This requires `PyGObject` (Python bindings for GObject) and the `AppIndicator` GObject introspection bindings.

   - **Debian/Ubuntu-based systems (e.g., Ubuntu 22.04 LTS):**
     You'll need to install `gir1.2-appindicator3-0.1` and the PyGObject development files. The specific PyGObject package might depend on your distribution version.
     ```bash
     sudo apt-get update && sudo apt-get install gir1.2-appindicator3-0.1 python3-gi python3-gi-cairo gir1.2-gtk-3.0
     ```
     If you encounter issues related to `libgirepository`, you might also need:
     ```bash
     sudo apt-get install libgirepository1.0-dev
     ```
     Or for newer systems, potentially `libgirepository2.0-dev`.

   - **Other Linux Distributions:**
     Please search your distribution's package manager for the equivalents of:
       - `appindicator3` or `libappindicator3` (e.g., `libappindicator-gtk3` on Fedora)
       - `PyGObject` or `python-gobject` (e.g., `python3-gobject` on Fedora)
       - The GObject Introspection development files (`gobject-introspection` or similar).

   After installing these system packages, you might need to reinstall the Python dependencies if you are using a virtual environment to ensure they pick up the new system libraries:
   ```bash
   pip install pystray --force-reinstall
   ```
   *(Note: The specific Python packages to reinstall might vary. `pystray` itself doesn't directly link to these system libraries at install time in a way that always necessitates reinstalling it, but ensuring `PyGObject` is correctly picked up by Python is key. Often, activating the virtual environment *after* system package installation is sufficient.)*

## Usage

Once installed, you can run the application using the `whisptray` command:

```bash
whisptray
```

A tray icon will appear. Click the icon to see options:
- **Toggle Dictation**: Starts or stops the dictation.
- **Exit**: Closes the application.

### Command-line Arguments

You can customize the behavior using command-line arguments:

```bash
whisptray --model small --energy_threshold 1200
```

Available arguments:
- `--model`: Whisper model to use (choices: "tiny", "base", "small", "medium", "large", "turbo" - default: "turbo").
- `--non_english`: Use the multilingual model variant (if applicable for the chosen size).
- `--energy_threshold`: Energy level for mic to detect (default: 1000).
- `--record_timeout`: How real-time the recording is in seconds (default: 2.0).
- `--phrase_timeout`: Silence duration before a new phrase is considered (default: 3.0).
- `--default_microphone` (Linux only): Name or part of the name of the microphone to use (default: 'pulse'). Use `whisptray --default_microphone list` to see available microphones.

## Development

To set up for development:

1. Clone the repository (if you haven't already).
2. Create and activate a virtual environment.
3. Install in editable mode: `pip install -e .`
4. (Optional) Install development tools: `pip install -e .[dev]` (if you add a `dev` extra in `pyproject.toml` for linters, formatters, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details (though a LICENSE file hasn't been created yet in this session, pyproject.toml specifies MIT). 