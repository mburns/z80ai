#!/bin/sh
# Retrain the clinc150 phrasebook and build its Agon binary.
#
# Unlike the other examples there is no feedme/exportmodel step: a phrasebook
# is a classifier, not a character decoder, so classify.py trains it and writes
# model.npz directly. It is also seeded, so this reproduces.
set -eu
../../data/clinc150/subset.py --recipe clinc150 | gzip -9n > training-data.txt.gz
../../classify.py --file training-data.txt.gz -o model.npz \
                  --hidden-sizes 384,256 --epochs 900
../../buildez80.py -m model.npz -o CLINC.bin
echo ""
echo "Agon eZ80 binary created: CLINC.bin, with PHRASES.DAT beside it."
echo "Copy BOTH onto the SD card, in the same directory, then run:"
echo "  CLINC"
echo ""
echo "The replies live in PHRASES.DAT - that is the point. Without it the"
echo "binary says so and stops rather than answering out of an empty buffer."
