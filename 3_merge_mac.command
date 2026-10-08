#!/bin/bash
# Parfois project - merge the 5 reviewed batches (run by ONE person once all reviews are in).
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Run 1_setup_mac.command first."; read -r -p "Press Enter..."; exit 1; }
.venv/bin/python src/sprint1_preprocess.py check
.venv/bin/python src/sprint1_preprocess.py merge
read -r -p "Press Enter to close..."
