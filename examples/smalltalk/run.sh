#!/bin/sh
# Retrain the smalltalk model from the vendored CLINC150 data and run it.
#
# To rebuild the training data itself (or to change which intents it covers,
# by editing RECIPES in data/clinc150/subset.py):
#   ../../data/clinc150/subset.py --recipe smalltalk | gzip -9n > training-data.txt.gz
set -eu

zcat training-data.txt.gz | ../../feedme.py --epochs 300 --save-best

# model.npz is what ships; without this the release keeps the old weights.
../../exportmodel.py -m command_model_autoreg.pt -o model.npz

../../build.py -m model.npz -o TALK.COM
../../cpm TALK.COM
