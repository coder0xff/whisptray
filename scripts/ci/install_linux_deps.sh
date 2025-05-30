#!/bin/bash
set -e
set -x

echo "Running install_linux_deps.sh"
echo "Checking system package manager..."

if command -v yum &> /dev/null; then
    echo "Using yum"
    yum install -y pkgconfig alsa-lib-devel
    # Attempt to install portaudio-devel directly
    if ! yum install -y portaudio-devel; then
        echo "Warning: 'yum install -y portaudio-devel' failed. Searching for package providing portaudio.h..."
        yum whatprovides "*/portaudio.h" || echo "Warning: 'yum whatprovides */portaudio.h' also failed or found nothing."
        echo "Also trying 'yum search portaudio' for related packages..."
        yum search portaudio || echo "Warning: 'yum search portaudio' failed or found nothing."
    fi
elif command -v apt-get &> /dev/null; then
    echo "Using apt-get"
    apt-get update
    apt-get install -y pkg-config libasound2-dev apt-file
    apt-file update
    # Attempt to install libportaudio-dev directly
    if ! apt-get install -y libportaudio-dev; then
        echo "Warning: 'apt-get install -y libportaudio-dev' failed. Searching for package providing portaudio.h..."
        apt-file search portaudio.h || echo "Warning: 'apt-file search portaudio.h' also failed or found nothing."
        echo "Also trying 'apt-cache search portaudio' for related packages..."
        apt-cache search portaudio || echo "Warning: 'apt-cache search portaudio' failed or found nothing."
    fi
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