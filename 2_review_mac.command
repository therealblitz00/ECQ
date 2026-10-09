#!/bin/bash
# Parfois project - review my Sprint 1 batch in the browser (macOS).
cd "$(dirname "$0")" || exit 1
[ -x .venv/bin/python ] || { echo "Run 1_setup_mac.command first."; read -r -p "Press Enter..."; exit 1; }

echo "Team numbers: 1=André  2=Pedro Correia  3=Pedro Meireles  4=Manuel  5=Zé"
read -r -p "Your team member number (1-5): " ID
if [ ! -f "outputs/sprint1/batches/batch_0${ID}_of_05.csv" ]; then
  .venv/bin/python src/sprint1_preprocess.py batch --members 5 --id "$ID" || { read -r -p "Press Enter..."; exit 1; }
fi

echo
echo "The review app opens in your browser. Your clicks are saved automatically."
echo "KEEP THIS WINDOW OPEN while you review. Close it when you are done."
.venv/bin/python src/review_app.py --id "$ID"
