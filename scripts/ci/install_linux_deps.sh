#!/bin/bash
set -e
set -x

echo "Running install_linux_deps.sh"
echo "Checking system package manager..."

PORTAUDIO_H_PACKAGE_FOUND=false

if command -v yum &> /dev/null; then
    echo "Using yum"
    yum install -y pkgconfig alsa-lib-devel
    if yum install -y portaudio-devel; then
        echo "'yum install -y portaudio-devel' reported success."
        PORTAUDIO_H_PACKAGE_FOUND=true
    else
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
    if apt-get install -y libportaudio-dev; then
        echo "'apt-get install -y libportaudio-dev' reported success."
        PORTAUDIO_H_PACKAGE_FOUND=true
    else
        echo "Warning: 'apt-get install -y libportaudio-dev' failed. Searching for package providing portaudio.h..."
        apt-file search portaudio.h || echo "Warning: 'apt-file search portaudio.h' also failed or found nothing."
        echo "Also trying 'apt-cache search portaudio' for related packages..."
        apt-cache search portaudio || echo "Warning: 'apt-cache search portaudio' failed or found nothing."
    fi
elif command -v apk &> /dev/null; then
    echo "Using apk (Alpine Linux / musllinux)"
    apk add --no-cache pkgconf alsa-lib-dev
    # Try to install portaudio-dev for musllinux
    if apk add --no-cache portaudio-dev; then
        echo "'apk add --no-cache portaudio-dev' reported success."
        PORTAUDIO_H_PACKAGE_FOUND=true
    else
        echo "Warning: 'apk add --no-cache portaudio-dev' failed. Searching for portaudio packages..."
        apk search -v portaudio || echo "Warning: 'apk search portaudio' failed or found nothing."
        echo "Searching for who owns portaudio.h..."
        apk info --who-owns portaudio.h || echo "Warning: 'apk info --who-owns portaudio.h' failed or found nothing."
    fi
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
        echo "WARNING: pkg-config did NOT find alsa.pc. Your C extension might fail to build."
    fi
else
    echo "Error: pkg-config command not found after attempting installation."
    exit 1
fi

echo "Verifying portaudio.h presence on filesystem (after install attempts)..."
if find /usr/include /usr/local/include /opt -name portaudio.h -print -quit 2>/dev/null; then
    echo "SUCCESS: portaudio.h found on filesystem."
else
    echo "WARNING: portaudio.h NOT found on filesystem in /usr/include, /usr/local/include, or /opt."
    if [ "$PORTAUDIO_H_PACKAGE_FOUND" = true ] ; then
      echo "This is strange, as a PortAudio development package was reportedly installed."
      echo "Listing files of installed PortAudio dev packages (if possible)..."
      if command -v yum &> /dev/null && yum list installed 'portaudio-devel*' &>/dev/null; then
          echo "Files for portaudio-devel (yum):"
          rpm -ql $(yum list installed 'portaudio-devel*' -q | awk '{print $1}' | head -n 1) 2>/dev/null || echo "Could not list files for portaudio-devel."
      elif command -v dpkg &> /dev/null && dpkg -l 'libportaudio*-dev' 2>/dev/null | grep -q '^ii'; then
          echo "Files for libportaudio*-dev (apt):"
          dpkg -L $(dpkg -l 'libportaudio*-dev' 2>/dev/null | grep '^ii' | awk '{print $2}' | head -n 1) 2>/dev/null || echo "Could not list files for apt portaudio dev package."
      elif command -v apk &> /dev/null && apk info portaudio-dev &>/dev/null; then
          echo "Files for portaudio-dev (apk):"
          apk contents portaudio-dev 2>/dev/null || echo "Could not list files for apk portaudio-dev package."
      fi
    fi
fi

echo "Finished install_linux_deps.sh" 