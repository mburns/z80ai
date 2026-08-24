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
# Every *Z80* hash moved again when ENCODE_CTX was shared with the eZ80 build.
# libnn's copy opened with `LD A,0 / LD (CTXPOS),A` and then fell straight into
# CTX_NLOOP, which begins `XOR A / LD (CTXPOS),A` - five dead bytes the eZ80
# copy never had, and the only thing standing between the two loop nests. The
# eZ80 hashes are unchanged, which is the check that only the dead store went.
#
# Then *every* hash moved when TOKENIZE was shared too. The Z80 images lose a
# byte each: clearing the query buckets was `LD A,0 / LD (HL),A` where the eZ80
# wrote `LD (HL),0` - a byte shorter, and it leaves A alone. The eZ80 images
# gain 19, because the shared body folds case inline where the eZ80 called
# LOWER. TOKENIZE runs once per query, so 19 bytes in 380KB is what deleting a
# duplicated trigram walk costs - and that walk is where a divergence would be
# silent, since two tokenizers that disagree both produce plausible buckets.
#
# artifact -> (module, example, sha256 of the image as written to disk)
GOLDEN = {
    "GUESS.COM": (buildz80com, "guess",
                  "7345258bf7e2f14acfc5a5231ecdcc2c6aa8dda4fb75022cd50ce6af16b72f81"),
    "GUESS-FAST.COM": (buildfastz80com, "guess",
                       "62ad3bfdccae35043b5a8949a03fe7e930d5b0a8323b3e116467765c1ac8e8aa"),
    # New: the column-major CP/M layout, 2.9x fewer instructions than the
    # index-list one for about 3KB more.
    "GUESS-COL.COM": (buildcolz80com, "guess",
                      "8ae0b2f2f11b5c566db02e4a15d09297cba431fc9b6757739791d125bb5a4ed8"),
    # Changed when eZ80 ARGMAX stopped counting neurons in B (no 256-output
    # cap, 24-bit MAXI/RESULT) and again when the default kernel became the
    # unrolled column-major one - 23x fewer instructions for 2.6x the size.
    "GUESS.bin": (buildez80, "guess",
                  "f5bdd85b75e97994351e14e584a81e66058436d2ca44a5082f2d109195da826c"),
    # Changed when tinychat's 502 replies were collapsed onto a 21-word
    # vocabulary and then onto 11; the model is retrained on it.
    "CHAT.COM": (buildz80com, "tinychat",
                 "853e02504b69b2999c7b2e45de422b96b028ecf6db88cc097624e94408a8cc8b"),
    "CHAT-FAST.COM": (buildfastz80com, "tinychat",
                      "5650ccaaf0c2cb13031f93e2a4aee543127cefc4bd0cafcd5d88cf8cea915f79"),
    "CHAT-COL.COM": (buildcolz80com, "tinychat",
                     "1919e4ee192720def767322413e0472b93d20366fbad3ca64dea0d9dd54089cd"),
    "CHAT.bin": (buildez80, "tinychat",
                 "1706a1a6bdf540268fc372c4ecbebee3f033641d2049ce5299541a1e597ccb89"),
    "TALK.COM": (buildz80com, "smalltalk",
                 "15103b7585005752a016a117a74a02e0debc1fb5a4802b00e44096f05dbbe5df"),
    "TALK-FAST.COM": (buildfastz80com, "smalltalk",
                      "7e800c577e211c71185fe981b37126aa4ad836c88307e9d1d0032a6d14c7ef7b"),
    "TALK-COL.COM": (buildcolz80com, "smalltalk",
                     "5551c23134a55791c7ca45244954446ad020173c09ff1c21562b2bb13e6a87c3"),
    "TALK.bin": (buildez80, "smalltalk",
                 "fcb6f23650bb42a3c5e44382102346ec23e960b8064bb564b16f4a3b1ccbb4f7"),
}

#: The phrasebook builds, which are two files each: the image, and the replies
#: it loads off the SD card. Both are pinned, because the offset table is in
#: one and the text it indexes is in the same one - a build that changed either
#: without the other would print the wrong reply rather than fail.
#:
#: One forward pass over 128 query buckets, one argmax, and the text printed
#: from the card rather than spelled: no GENLOOP, no context window. The
#: column kernel is allowed now that its input scan covers the whole vector
#: (no query half to hoist when there is one pass); `auto` takes it whenever
#: the unrolled blocks fit in Agon SRAM.
#:
#: The file name is part of the image - the binary carries the string it asks
#: MOS for - so these hashes move if it is renamed.
#:
#: artifact -> (example, model file, companion name, image sha256, companion sha256)
GOLDEN_PHRASEBOOK = {
    # 151 replies. `column` would need 537KB, so `auto` lands on `row`.
    "CLINC.bin": (
        "clinc150", "model.npz", "CLINC.DAT",
        "1e575d2e38b0f5fdd01f2ee5019648c95a912805d44367e3fd257e4a31c5d10a",
        "daaee683934dbad65c27986de6d9f83608ec49345a641ee53bdd07ee2f3930b4"),
    # smalltalk's 19 intents answered in sentences rather than spelled: 87.2%
    # macro against the character decoder's 80.7%, on the same labels.
    # Changed when the column kernel learned single-pass phrasebooks: 460KB,
    # and it fits, so `auto` takes it - 112,800 -> 86,910 instructions per
    # question on this model (1.3x; sparser models gain more).
    #
    # The value moved again when this was rebased onto the shared eZ80
    # TOKENIZE, which is 19 bytes longer than the copy buildez80 used to
    # carry. Same kernel, same arithmetic, different address for everything
    # after the tokenizer - so the image was rebuilt and re-verified against
    # the reference rather than either side's pin being taken.
    "TALK-PHR.bin": (
        "smalltalk", "phrasebook.npz", "TALK-PHR.DAT",
        "95642711df081188912bfa1f53813637b14e7b5d0709d8645a1ec72e220737ac",
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
                  "efb61f6020fbb08235e31d9f0a1f908855167ab48c20b112175538d535f39105"),
    "CHAT.TAP": (buildz80tap, _tap, "tinychat",
                 "dfbd3ae158a4802d313cdf61cd41f88b4f8080c6716246a6c31283476e9c34de"),
    "TALK.TAP": (buildz80tap, _tap, "smalltalk",
                 "d0fb21d7812f4f3db38823a37799deb87ff7040338ed6902517f6680567a9604"),
    # New: the Next build. Byte-for-byte the Spectrum's, plus the six bytes
    # that ask for 28MHz, so its hash moves whenever the ZX one does.
    "GUESS-NEXT.TAP": (buildnext, _tap, "guess",
                       "122adf3499d3dd0d01bdb253b84e989e385f849968ded183c9ed611717e095d4"),
    "CHAT-NEXT.TAP": (buildnext, _tap, "tinychat",
                      "06c669dcd4746444eec2fdc01d19c5660b8d2e277876ea5597a0133ea31fca18"),
    "TALK-NEXT.TAP": (buildnext, _tap, "smalltalk",
                      "737a4b526bfcbce6ecf68c2e134630e1a02e847fdd118767891e33482c66cdae"),
    # New: the Amstrad CPC build, inside the AMSDOS header RUN" reads.
    "GUESS-CPC.BIN": (buildcpc, _amsdos, "guess",
                      "e443e98c98cd5503194d79816be90bd978787df7987bd067c8880f7762f977ad"),
    "CHAT-CPC.BIN": (buildcpc, _amsdos, "tinychat",
                     "540c5350864cf7a8df948b21d2816239222916db3d269728a0715ce256302218"),
    "TALK-CPC.BIN": (buildcpc, _amsdos, "smalltalk",
                     "ad7289e3fd54d442dfddee3acedac718b6c587f1f4918bbdb0d6989ce8bb65fc"),
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
