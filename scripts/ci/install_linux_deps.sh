#!/bin/bash
set -e
set -x

echo "Running install_linux_deps.sh to install runtime audio libraries (ALSA, PortAudio)"

if command -v yum &> /dev/null; then
    echo "Using yum"
    # alsa-lib-devel for ALSA, portaudio for PortAudio runtime
    yum install -y pkgconfig alsa-lib-devel portaudio
elif command -v apt-get &> /dev/null; then
    echo "Using apt-get"
    apt-get update
    # libasound2-dev for ALSA, libportaudio2 for PortAudio runtime
    apt-get install -y pkg-config libasound2-dev libportaudio2
elif command -v apk &> /dev/null; then
    echo "Using apk (Alpine Linux / musllinux)"
    # alsa-lib-dev for ALSA, portaudio for PortAudio runtime
    apk add --no-cache pkgconf alsa-lib-dev portaudio
else
    echo "Error: No known package manager (yum, apt-get, apk) found. Cannot install dependencies."
    exit 1
fi

echo "Verifying pkg-config installation..."
if command -v pkg-config &> /dev/null; then
    pkg-config --version
    echo "Checking for alsa.pc with pkg-config..."
    if pkg-config --exists alsa; then
        echo "SUCCESS: pkg-config found alsa.pc"
        echo "ALSA CFLAGS: $(pkg-config --cflags alsa)"
        echo "ALSA LIBS: $(pkg-config --libs alsa)"
    else
        echo "WARNING: pkg-config did NOT find alsa.pc."
    fi
else
    echo "Error: pkg-config command not found after attempting installation."
    exit 1
fi

# We no longer explicitly check for portaudio.h as we primarily need runtime libraries.
# The sounddevice wheels should ideally bundle PortAudio or link correctly if the runtime is present.

echo "Finished install_linux_deps.sh" 