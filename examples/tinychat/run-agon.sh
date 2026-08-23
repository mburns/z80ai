#!/bin/sh
# Build an Agon Light / eZ80 binary for tinychat
set -eu
# training-data.txt.gz is generated; canonicalize.py is the source of truth.
./canonicalize.py | gzip -9nc > training-data.txt.gz
zcat training-data.txt.gz | ../../feedme.py --epochs 300 --save-best
# model.npz is what ships; without this the release keeps the old weights.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz
../../buildez80.py -m model.npz -o CHAT.bin
echo ""
echo "Agon eZ80 binary created: CHAT.bin"
echo "Copy it onto the SD card and run it by name at the MOS prompt:"
echo "  CHAT"
