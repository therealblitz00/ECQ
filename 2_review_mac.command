#!/bin/bash
# Parfois project - open my Sprint 1 review batch (macOS).
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Run 1_setup_mac.command first."; read -r -p "Press Enter..."; exit 1; }

read -r -p "Your team member number (1-4): " ID
.venv/bin/python src/sprint1_preprocess.py batch --members 4 --id "$ID" || { read -r -p "Press Enter..."; exit 1; }

open "outputs/sprint1/batches/batch_0${ID}_of_04.html"
open "outputs/sprint1/batches"
echo
echo "The review page opened in your browser. Fill in batch_0${ID}_of_04.csv (see SPRINT1_GUIDE.md)."
read -r -p "Press Enter to close..."
