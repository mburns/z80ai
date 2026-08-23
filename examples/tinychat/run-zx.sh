#!/bin/sh
# Build ZX Spectrum 48K TAP file for tinychat
set -eu
# training-data.txt.gz is generated; canonicalize.py is the source of truth.
./canonicalize.py | gzip -9nc > training-data.txt.gz
zcat training-data.txt.gz | ../../feedme.py --epochs 300 --save-best
# model.npz is what ships; without this the release keeps the old weights.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz
../../buildz80tap.py -m model.npz -o CHAT.TAP
echo ""
echo "ZX Spectrum TAP file created: CHAT.TAP"
echo "Load in emulator or transfer to real hardware"
echo "In ZX Spectrum BASIC:"
echo "  CLEAR 24575"
echo "  LOAD \"\" CODE"
echo "  RANDOMIZE USR 24576"
