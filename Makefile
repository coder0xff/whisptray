# Makefile for whisptray App

.PHONY: all install develop clean run check format help package

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
	rm -rf build dist src/**/*.egg-info src/**/*.so .mypy_cache $(VENV_DIR) $(C_HELPER_OUTPUT) $(C_HELPER_SRC:.c=.o)
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