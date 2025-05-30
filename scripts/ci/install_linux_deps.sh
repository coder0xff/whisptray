#!/bin/bash
set -e
set -x

echo "Running install_linux_deps.sh"
echo "Checking system package manager..."

if command -v yum &> /dev/null; then
    echo "Using yum"
    yum install -y pkgconfig alsa-lib-devel
    # Attempt to install portaudio-devel, but don't fail the script if it's not found
    yum install -y portaudio-devel || echo "Warning: portaudio-devel not found or failed to install via yum."
elif command -v apt-get &> /dev/null; then
    echo "Using apt-get"
    apt-get update
    apt-get install -y pkg-config libasound2-dev libportaudio-dev
else
    echo "Error: Neither yum nor apt-get found. Cannot install dependencies."
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
        echo "WARNING: pkg-config did NOT find alsa.pc. Your C extension might fail to build."
        echo "Searching for alsa.pc manually (this might be slow)..."
        find /usr -name alsa.pc -ls 2>/dev/null || echo "alsa.pc not found in /usr"
    fi
else
    echo "Error: pkg-config command not found after attempting installation."
    exit 1
fi

echo "Finished install_linux_deps.sh" 