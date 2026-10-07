#!/bin/bash
# Parfois project - one-time setup for macOS (double-click this file).
cd "$(dirname "$0")" || exit 1
echo "=== Parfois setup (macOS) ==="

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10+ from https://www.python.org/downloads/ and run this file again."
  read -r -p "Press Enter to close..."; exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv ..."
  python3 -m venv .venv || { echo "Could not create the virtual environment."; read -r -p "Press Enter..."; exit 1; }
fi
echo "Installing requirements ..."
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt || { echo "Installation failed."; read -r -p "Press Enter..."; exit 1; }

echo "Running the automatic checks ..."
.venv/bin/python src/sprint1_preprocess.py check || { echo "Checks failed."; read -r -p "Press Enter..."; exit 1; }

echo
echo "Setup complete. Next: double-click 2_review_mac.command"
read -r -p "Press Enter to close..."
