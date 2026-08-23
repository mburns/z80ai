#!/bin/sh
# Build every shipped example for every target into a directory.
#
# This is exactly what CI builds and releases, so running it locally reproduces
# the release artifacts byte for byte.
#
#   ./build-examples.sh [output-dir]     (default: dist)
#
# Verify the results with:
#
#   python verify_artifacts.py --dist dist

set -eu

OUT="${1:-dist}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT"

for example in guess:GUESS tinychat:CHAT smalltalk:TALK; do
    model="examples/${example%%:*}/model.npz"
    name="${example##*:}"

    "$PYTHON" buildz80com.py     -m "$model" -o "$OUT/$name.COM"
    "$PYTHON" buildfastz80com.py -m "$model" -o "$OUT/$name-FAST.COM"
    "$PYTHON" buildcolz80com.py  -m "$model" -o "$OUT/$name-COL.COM"
    "$PYTHON" buildz80tap.py     -m "$model" -o "$OUT/$name.TAP"
    "$PYTHON" buildez80.py       -m "$model" -o "$OUT/$name.bin"
done

# The phrasebook builds are Agon-only: they answer with an index into a file on
# the SD card, which no Z80 target has. buildez80.py writes both the binary and
# the replies it loads, from one build, so the offset table and the text it
# indexes cannot drift apart. Each needs its own file name, since they land in
# the same directory.
"$PYTHON" buildez80.py -m examples/clinc150/model.npz -o "$OUT/CLINC.bin" \
                       --phrases CLINC.DAT

# smalltalk twice over: the same 19 intents as TALK.bin, answered from the card
# instead of spelled. Worth 88.0% macro against the character decoder's 80.7%,
# and the two sit side by side so the comparison is one directory listing.
"$PYTHON" buildez80.py -m examples/smalltalk/phrasebook.npz -o "$OUT/TALK-PHR.bin" \
                       --phrases TALK-PHR.DAT

echo
echo "Built into $OUT/:"
ls -l "$OUT"
