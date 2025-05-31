# Makefile for whisptray App

.PHONY: all install develop clean run check format help package cibuildwheel

# Default Python interpreter - can be overridden
PYTHON ?= python3
PIP ?= pip3

# C Compiler and flags
CC ?= gcc
CFLAGS ?= -fPIC -Wall -Wextra -O2 # Basic warning and optimization flags
LDFLAGS ?= -shared
ALSA_CFLAGS ?= $(shell pkg-config --cflags alsa)
ALSA_LIBS ?= $(shell pkg-config --libs alsa)

# Virtual environment directory
VENV_DIR = .venv
# Activate script, depends on OS, but we'll assume bash-like for .venv/bin/activate
ACTIVATE = . $(VENV_DIR)/bin/activate

# Source files
# For Python tools
PYTHON_SRC_FILES = src/whisptray
# C helper library
C_HELPER_SRC = src/alsa_redirect.c
C_HELPER_OUTPUT = src/whisptray/alsa_redirect.so

# Variables for cibuildwheel builds
# These can be overridden from the environment if needed for specific CI jobs or local runs.
CIBW_PYTHON_VERSION ?= "3.8 3.9 3.10 3.11 3.12"
CIBW_BUILD_VERBOSITY ?= 0
CIBW_ARCHS_LINUX ?= "auto aarch64"
CIBW_ARCHS_MACOS ?= "x86_64 arm64"
CIBW_ARCHS_WINDOWS ?= "AMD64 ARM64"
CIBW_BEFORE_BUILD_LINUX ?= "sh {project}/scripts/ci/install_linux_deps.sh"
CIBW_MANYLINUX_AARCH64_IMAGE ?= "manylinux_2_28"
# cp37-*: project requires Python 3.8+. pp3*: PyPy, skip for now. cp313-*: openai-whisper build issue. *-manylinux_i686: torch dependency not available. *-musllinux_*: torch dependency not available or script needs apk for portaudio.
CIBW_SKIP_CONFIG ?= "cp37-* pp3* cp313-* *-manylinux_i686 *-musllinux_*"
CIBW_TEST_COMMAND ?= "echo 'No tests configured for wheels yet'" # Generic test command
CIBW_TEST_REQUIRES ?= ""
CIBW_OUTPUT_DIR ?= wheelhouse

# Build the package
package: $(VENV_DIR)/bin/activate
	@echo "Building the package..."
	$(VENV_DIR)/bin/$(PIP) install build
	$(VENV_DIR)/bin/python -m build --sdist
	@echo "Package build complete. Find artifacts in dist/ directory."

$(VENV_DIR)/bin/activate: # Target to create venv if activate script doesn't exist
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

# Target to build the C helper library
$(C_HELPER_OUTPUT): $(C_HELPER_SRC)
	@echo "Building ALSA C helper library..."
	$(CC) $(CFLAGS) $(ALSA_CFLAGS) $(LDFLAGS) -o $@ $< $(ALSA_LIBS)
	@echo "ALSA C helper library built as $(C_HELPER_OUTPUT)"

# Install the package and its dependencies
install: $(VENV_DIR)/bin/activate $(C_HELPER_OUTPUT)
	@echo "Installing the package..."
	$(VENV_DIR)/bin/$(PIP) install .
	@echo "Installation complete. Run with '$(VENV_DIR)/bin/whisptray' or activate venv and run 'whisptray'"

# Install for development (editable mode) and include dev dependencies
develop: $(VENV_DIR)/bin/activate $(C_HELPER_OUTPUT)
	@echo "Installing for development (editable mode) with dev dependencies..."
	$(VENV_DIR)/bin/$(PIP) install -e .[dev]
	@echo "Development installation complete."

# Run the application (assumes it's installed in the venv)
run: $(VENV_DIR)/bin/activate
	@echo "Running whisptray app..."
	$(VENV_DIR)/bin/whisptray

# Run checks (linting, formatting, type checking) for Python files
check: $(VENV_DIR)/bin/activate
	@echo "Running checks for Python files..."
	$(VENV_DIR)/bin/flake8 $(PYTHON_SRC_FILES)
	$(VENV_DIR)/bin/pylint $(PYTHON_SRC_FILES)
	$(VENV_DIR)/bin/black --check $(PYTHON_SRC_FILES)
	$(VENV_DIR)/bin/isort --check-only $(PYTHON_SRC_FILES)
	$(VENV_DIR)/bin/mypy $(PYTHON_SRC_FILES)
	@echo "Python checks complete."

# Apply formatting to Python files
format: $(VENV_DIR)/bin/activate
	@echo "Formatting Python source files..."
	$(VENV_DIR)/bin/black $(PYTHON_SRC_FILES)
	$(VENV_DIR)/bin/isort $(PYTHON_SRC_FILES)
	@echo "Python formatting complete."

# Clean build artifacts and virtual environment
clean:
	@echo "Cleaning build artifacts and virtual environment..."
	rm -rf build dist wheelhouse src/**/*.egg-info src/**/*.so .mypy_cache $(VENV_DIR) $(C_HELPER_OUTPUT) $(C_HELPER_SRC:.c=.o)
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -delete
	@echo "Clean complete."

help:
	@echo "Makefile for whisptray App"
	@echo ""
	@echo "Usage:"
	@echo "  make check           Run linting, formatting checks, and type checking for Python files."
	@echo "  make clean           Remove build artifacts, .pyc files, __pycache__ directories, C helper library and the virtual environment."
	@echo "  make develop         Install for development (editable mode) with dev dependencies (including C helper) into a virtual environment."
	@echo "  make format          Apply formatting to Python source files."
	@echo "  make help            Show this help message."
	@echo "  make install         Install the package and dependencies (including C helper) into a virtual environment."
	@echo "  make package           Build the package (sdist and wheel) into the dist/ directory."
	@echo "  make run             Run the application (requires prior install/develop)."
	@echo "  make $(C_HELPER_OUTPUT)  Build the ALSA C helper library independently."
	@echo ""
	@echo "To use a specific python/pip version:"
	@echo "  make PYTHON=python3.9 PIP=pip3.9 install"
	@echo "To use a specific C compiler:"
	@echo "  make CC=clang install" 

# Target to build Linux wheels
build-wheels-linux: clean-c-helper scripts/ci/install_linux_deps.sh
	@echo "Building Linux wheels using cibuildwheel..."
	CIBW_BEFORE_BUILD_LINUX=$(CIBW_BEFORE_BUILD_LINUX) \
	CIBW_ARCHS_LINUX=$(CIBW_ARCHS_LINUX) \
	CIBW_MANYLINUX_AARCH64_IMAGE=$(CIBW_MANYLINUX_AARCH64_IMAGE) \
	CIBW_SKIP=$(CIBW_SKIP_CONFIG) \
	CIBW_TEST_COMMAND=$(CIBW_TEST_COMMAND) \
	CIBW_TEST_REQUIRES=$(CIBW_TEST_REQUIRES) \
	CIBW_BUILD_VERBOSITY=$(CIBW_BUILD_VERBOSITY) \
	cibuildwheel --platform linux --output-dir $(CIBW_OUTPUT_DIR)

# Target to build Windows wheels
build-wheels-windows: clean-c-helper
	@echo "Building Windows wheels using cibuildwheel..."
	CIBW_ARCHS_WINDOWS=$(CIBW_ARCHS_WINDOWS) \
	CIBW_SKIP=$(CIBW_SKIP_CONFIG) \
	CIBW_TEST_COMMAND=$(CIBW_TEST_COMMAND) \
	CIBW_TEST_REQUIRES=$(CIBW_TEST_REQUIRES) \
	CIBW_BUILD_VERBOSITY=$(CIBW_BUILD_VERBOSITY) \
	cibuildwheel --platform windows --output-dir $(CIBW_OUTPUT_DIR)

# Target to build macOS wheels
build-wheels-macos: clean-c-helper
	@echo "Building macOS wheels using cibuildwheel..."
	CIBW_ARCHS_MACOS=$(CIBW_ARCHS_MACOS) \
	CIBW_SKIP=$(CIBW_SKIP_CONFIG) \
	CIBW_TEST_COMMAND=$(CIBW_TEST_COMMAND) \
	CIBW_TEST_REQUIRES=$(CIBW_TEST_REQUIRES) \
	CIBW_BUILD_VERBOSITY=$(CIBW_BUILD_VERBOSITY) \
	cibuildwheel --platform macos --output-dir $(CIBW_OUTPUT_DIR)

# Clean C helper library specifically
clean-c-helper:
	@echo "Cleaning C helper library $(C_HELPER_OUTPUT)..."
	rm -f $(C_HELPER_OUTPUT) $(C_HELPER_SRC:.c=.o)
