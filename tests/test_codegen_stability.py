"""Pin the exact bytes each backend emits for the shipped models.

Refactoring the code generator should not change the code it generates. These
hashes make that explicit: a refactor that is meant to be behaviour-preserving
keeps them, and a deliberate codegen change updates them in the same commit
that explains why.

To update after an intended change::

    ./build-examples.sh dist
    shasum -a 256 dist/*

The semantic tests elsewhere check that the generated code is *correct*; these
only check that it has not moved.
"""

from __future__ import annotations

import hashlib
import os

import pytest

import buildcolz80com
import buildcpc
import buildez80
import buildfastz80com
import buildnext
import buildz80com
import buildz80tap
import libcpc
import libzx
from libz80 import Z80Builder


def _tap(builder: Z80Builder, artifact: str) -> bytes:
    """The .TAP the ZX and Next build scripts write."""
    return libzx.build_tap(builder.build(), builder.org)


def _amsdos(builder: Z80Builder, artifact: str) -> bytes:
    """The AMSDOS binary the CPC build script writes, named as on disc."""
    return libcpc.build_binary(builder.build(), builder.org, artifact.split(".")[0])

# Every hash below changed when layer 1's query half was hoisted out of the
# generation loop: the query cannot change while one response is generated, so
# PREQ folds its contribution into layer 1's bias once per query instead of once
# per character. 1.30x on the packed builds, 1.24x on the index-list build, 1.16x
# on the eZ80 column kernel, for 600-5,000 bytes. Bit-identical output - see
# tests/test_hoisting.py.
#
# artifact -> (module, example, sha256 of the image as written to disk)
GOLDEN = {
    "GUESS.COM": (buildz80com, "guess",
                  "f6049e084e42de0ef3c6aadf8c8564d4c8ec4018960e0bbe4d2b75643ab1d883"),
    "GUESS-FAST.COM": (buildfastz80com, "guess",
                       "3a003a1fda25c6b12eea2025c1c5fb42dbf4fb7894c25ab71c666d0d72b92cbe"),
    # New: the column-major CP/M layout, 2.9x fewer instructions than the
    # index-list one for about 3KB more.
    "GUESS-COL.COM": (buildcolz80com, "guess",
                      "6d0b8f6a3d10d8ed83986fb9014c2c181f7772c3501f093c97782c655fc35e7a"),
    # Changed when eZ80 ARGMAX stopped counting neurons in B (no 256-output
    # cap, 24-bit MAXI/RESULT) and again when the default kernel became the
    # unrolled column-major one - 23x fewer instructions for 2.6x the size.
    "GUESS.bin": (buildez80, "guess",
                  "56484a20f181bbddd040aef16b679d976207f1113fb084aa9b793f83f41df9ac"),
    # Changed when tinychat's 502 replies were collapsed onto a 21-word
    # vocabulary and then onto 11; the model is retrained on it.
    "CHAT.COM": (buildz80com, "tinychat",
                 "facab199025f80ebee7a77110eb4a16af6e604805da4fe7e2ef5623d46f11a30"),
    "CHAT-FAST.COM": (buildfastz80com, "tinychat",
                      "e3b74660e5777ea82568ec13454e6e534e4a661990640cd9447df527f7f72977"),
    "CHAT-COL.COM": (buildcolz80com, "tinychat",
                     "4a522fb857ba0f55220344cbcc32d226a081f49a3d9c8b55b8d5cea998d14f92"),
    "CHAT.bin": (buildez80, "tinychat",
                 "cd09d2dc237e06977ecbf7e3dda28a63babc20c858c656519e03daf91859d305"),
    "TALK.COM": (buildz80com, "smalltalk",
                 "fdf6ee4f4d660e64435c1e0359f2cc6e393b91e4f76c1b98b56b78b6263f4722"),
    "TALK-FAST.COM": (buildfastz80com, "smalltalk",
                      "17d9070833d14e79f649f4a295bc9f3ebaedb005d05faa7e66f5a8eff3e810bf"),
    "TALK-COL.COM": (buildcolz80com, "smalltalk",
                     "522cfbfc438141e44c55f036764d32d1d50c8df66475c3f9e57aa1d9c97ef637"),
    "TALK.bin": (buildez80, "smalltalk",
                 "8370a5fe7731f33f607a3664223b31ce0f064d1b6699700f31feacca5c976937"),
}

#: The phrasebook builds, which are two files each: the image, and the replies
#: it loads off the SD card. Both are pinned, because the offset table is in
#: one and the text it indexes is in the same one - a build that changed either
#: without the other would print the wrong reply rather than fail.
#:
#: One forward pass over 128 query buckets, one argmax, and the text printed
#: from the card rather than spelled: no GENLOOP, no context window, and no
#: column kernel, whose query hoisting amortizes over the steps of a response
#: and there is one. `auto` lands on `row`.
#:
#: The file name is part of the image - the binary carries the string it asks
#: MOS for - so these hashes move if it is renamed.
#:
#: artifact -> (example, model file, companion name, image sha256, companion sha256)
GOLDEN_PHRASEBOOK = {
    # 151 replies. `column` would need 552KB and not fit in Agon SRAM.
    "CLINC.bin": (
        "clinc150", "model.npz", "CLINC.DAT",
        "a27cc8803409200a02a583e91df44e2787f75f21b44d9755b8e0865cfdaa30d1",
        "daaee683934dbad65c27986de6d9f83608ec49345a641ee53bdd07ee2f3930b4"),
    # smalltalk's 19 intents answered in sentences rather than spelled: 87.2%
    # macro against the character decoder's 80.7%, on the same labels.
    "TALK-PHR.bin": (
        "smalltalk", "phrasebook.npz", "TALK-PHR.DAT",
        "7bf2c0d56f463e0a943e5470a739d4cf4e172533d37a13444f9e19d91b21fbd7",
        "1c2035adfad893479834c767a80e3ab1094b8d3bc35bb477cb69a38c23b04431"),
}

# Some targets ship the image inside a container the machine's loader reads.
# Those hashes cover the whole file, header included, because a wrong load
# address or checksum is exactly as fatal as wrong code.
#
# artifact -> (module, wrap, example, sha256 of the file as written to disk)
GOLDEN_WRAPPED = {
    # Changed when the ZX build adopted the CP/M inner loop: same arithmetic,
    # 26% fewer instructions, 32 bytes smaller.
    "GUESS.TAP": (buildz80tap, _tap, "guess",
                  "9c6aa2b15841a2bc945b3d0c8468edb5b3432150b4d7f19b7bab5d96f2bb80dc"),
    "CHAT.TAP": (buildz80tap, _tap, "tinychat",
                 "fbd66cf962521294e59440ef6d6c610b8953920527397f2589dc25f4b22abe20"),
    "TALK.TAP": (buildz80tap, _tap, "smalltalk",
                 "b66150b7823e40b849cea774c1b205608decdba4fb04b19e91a266863be82540"),
    # New: the Next build. Byte-for-byte the Spectrum's, plus the six bytes
    # that ask for 28MHz, so its hash moves whenever the ZX one does.
    "GUESS-NEXT.TAP": (buildnext, _tap, "guess",
                       "c1f34daf0b1d6756af93c6c1ad41afca010aa87c628e249ce2c18ea3bc20d8de"),
    "CHAT-NEXT.TAP": (buildnext, _tap, "tinychat",
                      "9d4d904d2b556a81141bc17838f9e0786cbc36eea878628013dc898279fb4575"),
    "TALK-NEXT.TAP": (buildnext, _tap, "smalltalk",
                      "75a3717451a3f693ea7d7a6325fff3ec268655d290cd1b281c3a2b81acb668de"),
    # New: the Amstrad CPC build, inside the AMSDOS header RUN" reads.
    "GUESS-CPC.BIN": (buildcpc, _amsdos, "guess",
                      "2e8f5c7b281426778e6bc9095b9a8a55a073feb0bfdc326b1716b71e44f79cfb"),
    "CHAT-CPC.BIN": (buildcpc, _amsdos, "tinychat",
                     "e6f5fc84a8a46f3f72e9f3dfb7d87fa1195bae1564c1556f868705aa63633055"),
    "TALK-CPC.BIN": (buildcpc, _amsdos, "smalltalk",
                     "e2dce867cbb6c2d26ac759ad535f4c6cf20ab003abbfc41636adf3d46a04bad4"),
}


def model_path(examples_dir: str, example: str) -> str:
    path = os.path.join(examples_dir, example, "model.npz")
    if not os.path.exists(path):
        pytest.skip(f"{example} example model not present")
    return path


@pytest.mark.parametrize("artifact", sorted(GOLDEN))
def test_generated_image_is_unchanged(artifact, examples_dir):
    module, example, expected = GOLDEN[artifact]
    image = module.build_autoreg(model_path(examples_dir, example)).build()
    got = hashlib.sha256(image).hexdigest()
    assert got == expected, (
        f"{artifact} changed: {got}\n"
        f"If that was intended, update GOLDEN in this file and say why in the "
        f"commit message."
    )


@pytest.mark.parametrize("artifact", sorted(GOLDEN_WRAPPED))
def test_generated_file_is_unchanged(artifact, examples_dir):
    module, wrap, example, expected = GOLDEN_WRAPPED[artifact]
    builder = module.build_autoreg(model_path(examples_dir, example))
    got = hashlib.sha256(wrap(builder, artifact)).hexdigest()
    assert got == expected, (
        f"{artifact} changed: {got}\n"
        f"If that was intended, update GOLDEN_WRAPPED in this file and say why "
        f"in the commit message."
    )


@pytest.mark.parametrize("artifact", sorted(GOLDEN_PHRASEBOOK))
def test_generated_phrasebook_is_unchanged(artifact, examples_dir):
    example, model, companion, expected_bin, expected_dat = \
        GOLDEN_PHRASEBOOK[artifact]

    path = os.path.join(examples_dir, example, model)
    if not os.path.exists(path):
        pytest.skip(f"{example}/{model} not present")

    builder = buildez80.build_autoreg(path, phrases_file=companion)
    got_bin = hashlib.sha256(builder.build()).hexdigest()
    got_dat = hashlib.sha256(builder.phrase_blob).hexdigest()

    assert got_bin == expected_bin, (
        f"{artifact} changed: {got_bin}\n"
        f"If that was intended, update GOLDEN_PHRASEBOOK in this file and say "
        f"why in the commit message."
    )
    assert got_dat == expected_dat, (
        f"{companion} changed: {got_dat}\n"
        f"The replies moved without the image moving, which means the offset "
        f"table and the text it indexes were built apart."
    )


def test_every_shipped_artifact_is_covered():
    """build-examples.sh and the release list must not outgrow this file."""
    import verify_artifacts

    assert set(verify_artifacts.ARTIFACTS) == (
        set(GOLDEN) | set(GOLDEN_WRAPPED) | set(GOLDEN_PHRASEBOOK))
    assert set(verify_artifacts.COMPANIONS) == set(GOLDEN_PHRASEBOOK)
    for artifact, companion in verify_artifacts.COMPANIONS.items():
        assert GOLDEN_PHRASEBOOK[artifact][2] == companion, artifact
