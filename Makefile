# Makefile for Dictate App

.PHONY: all install develop clean run check format help

# Default Python interpreter - can be overridden
PYTHON ?= python3
PIP ?= pip3

# Virtual environment directory
VENV_DIR = .venv
# Activate script, depends on OS, but we'll assume bash-like for .venv/bin/activate
ACTIVATE = . $(VENV_DIR)/bin/activate

# Source files
SRC_FILES = dictate.py

all: install

$(VENV_DIR)/bin/activate: # Target to create venv if activate script doesn't exist
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

# Install the package and its dependencies
install: $(VENV_DIR)/bin/activate
	@echo "Installing the package..."
	$(VENV_DIR)/bin/$(PIP) install .
	@echo "Installation complete. Run with '$(VENV_DIR)/bin/dictate' or activate venv and run 'dictate'"

# Install for development (editable mode) and include dev dependencies
develop: $(VENV_DIR)/bin/activate
	@echo "Installing for development (editable mode) with dev dependencies..."
	$(VENV_DIR)/bin/$(PIP) install -e .[dev]
	@echo "Development installation complete."

# Run the application (assumes it's installed in the venv)
run: $(VENV_DIR)/bin/activate
	@echo "Running dictate app..."
	$(VENV_DIR)/bin/dictate

# Run checks (linting, formatting, type checking)
check: $(VENV_DIR)/bin/activate
	@echo "Running checks..."
	$(VENV_DIR)/bin/flake8 $(SRC_FILES)
	$(VENV_DIR)/bin/black --check $(SRC_FILES)
	$(VENV_DIR)/bin/isort --check-only $(SRC_FILES)
	$(VENV_DIR)/bin/mypy $(SRC_FILES)
	@echo "Checks complete."

# Apply formatting
format: $(VENV_DIR)/bin/activate
	@echo "Formatting source files..."
	$(VENV_DIR)/bin/black $(SRC_FILES)
	$(VENV_DIR)/bin/isort $(SRC_FILES)
	@echo "Formatting complete."

# Clean build artifacts and virtual environment
clean:
	@echo "Cleaning build artifacts and virtual environment..."
	rm -rf build dist src/*.egg-info .mypy_cache $(VENV_DIR)
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -type d -delete
	@echo "Clean complete."

help:
	@echo "Makefile for Dictate App"
	@echo ""
	@echo "Usage:"
	@echo "  make install         Install the package and dependencies into a virtual environment."
	@echo "  make develop         Install for development (editable mode) with dev dependencies into a virtual environment."
	@echo "  make run             Run the application (requires prior install/develop)."
	@echo "  make check           Run linting, formatting checks, and type checking."
	@echo "  make format          Apply formatting to source files."
	@echo "  make clean           Remove build artifacts, .pyc files, __pycache__ directories, and the virtual environment."
	@echo "  make help            Show this help message."
	@echo ""
	@echo "To use a specific python/pip version:"
	@echo "  make PYTHON=python3.9 PIP=pip3.9 install" 