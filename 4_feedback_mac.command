#!/bin/bash
# Parfois project - teach the similarity engine in the browser (Phase 4, macOS).
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Run 1_setup_mac.command first."; read -r -p "Press Enter..."; exit 1; }

echo "Team numbers: 1=André  2=Pedro Correia  3=Pedro Meireles  4=Manuel  5=Zé"
read -r -p "Your team member number (1-5): " ID

echo
echo "The feedback app opens in your browser. Every query you sort is saved automatically."
echo "KEEP THIS WINDOW OPEN while you sort. Close it when you are done."
.venv/bin/python src/phase4_feedback.py app --id "$ID"
