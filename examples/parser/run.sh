#!/bin/sh
# Train the position-aware parser and build a .COM for it.
set -eu

PYTHON="${PYTHON:-python3}"

"$PYTHON" gendata.py
"$PYTHON" ../../feedme.py --position-bands 8 --epochs 400 \
    --hidden-sizes 96,64 -o parser.pt < training-data.txt
"$PYTHON" ../../exportmodel.py -m parser.pt -o parser.npz
"$PYTHON" ../../build.py --model parser.npz --output PARSER.COM

echo
echo "Built PARSER.COM. Word order decides the answer:"
echo "  iz-cpm PARSER.COM \"PUT KEY IN BOX\"   -> OK"
echo "  iz-cpm PARSER.COM \"PUT BOX IN KEY\"   -> NO"
