#!/bin/sh
# Build an Agon Light / eZ80 binary for smalltalk
set -eu
zcat training-data.txt.gz | ../../feedme.py --epochs 300 --save-best
# model.npz is what ships; without this the release keeps the old weights.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz
../../buildez80.py -m model.npz -o TALK.bin
echo ""
echo "Agon eZ80 binary created: TALK.bin"
echo "Copy it onto the SD card and run it by name at the MOS prompt:"
echo "  TALK"
