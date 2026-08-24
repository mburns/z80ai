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
# artifact -> (module, example, sha256 of the image as written to disk)
GOLDEN = {
    "GUESS.COM": (buildz80com, "guess",
                  "310cf36f5ec66da8fa2995c32256b06190287b754d7757fcd95385b1132a466b"),
    "GUESS-FAST.COM": (buildfastz80com, "guess",
                       "d2c426b078c3e1160b9ce23d959244f15e89ef83c637cbb1a9d6f411cbbfca33"),
    # New: the column-major CP/M layout, 2.9x fewer instructions than the
    # index-list one for about 3KB more.
    "GUESS-COL.COM": (buildcolz80com, "guess",
                      "5698b61cff56644aa938ae8f089b1b856e146976422335b09c6d526164e594f4"),
    # Changed when eZ80 ARGMAX stopped counting neurons in B (no 256-output
    # cap, 24-bit MAXI/RESULT) and again when the default kernel became the
    # unrolled column-major one - 23x fewer instructions for 2.6x the size.
    "GUESS.bin": (buildez80, "guess",
                  "56484a20f181bbddd040aef16b679d976207f1113fb084aa9b793f83f41df9ac"),
    # Changed when tinychat's 502 replies were collapsed onto a 21-word
    # vocabulary and then onto 11; the model is retrained on it.
    "CHAT.COM": (buildz80com, "tinychat",
                 "77e3e7a142464d81e5703033a1084d26516ba95bb025192ba4445920fcba542b"),
    "CHAT-FAST.COM": (buildfastz80com, "tinychat",
                      "06db7b9489ada8856558a3b82d4cf4f5f311c5c125d1cb55650e09f87c7003f5"),
    "CHAT-COL.COM": (buildcolz80com, "tinychat",
                     "8a705a41f3b2d1fa827bda29aa0025efcaba9901b95abe4ed17d2250968cbb6d"),
    "CHAT.bin": (buildez80, "tinychat",
                 "cd09d2dc237e06977ecbf7e3dda28a63babc20c858c656519e03daf91859d305"),
    "TALK.COM": (buildz80com, "smalltalk",
                 "41bf407ee5e8dbc653d69b8c39426acad1af49cb6309c630c5ed6fe2a24c8548"),
    "TALK-FAST.COM": (buildfastz80com, "smalltalk",
                      "4bfce44144dd515fb72ec770009c92003235160e2f2a7b9d5c1bef5c0f5b3237"),
    "TALK-COL.COM": (buildcolz80com, "smalltalk",
                     "4e97909eb7cb7d8c1989e61cb91497008a9ef6629aba9fa7b98ab415221c8df4"),
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
                  "36265ba54bb77115413acfb2e30a16bba41f15e30526c25055290190bbe9c030"),
    "CHAT.TAP": (buildz80tap, _tap, "tinychat",
                 "70f34caa8397e94860c5318a476e24efa43756c20a9461b4e36068b780592559"),
    "TALK.TAP": (buildz80tap, _tap, "smalltalk",
                 "c0824510e5933632adc29b7008f459a743953524edea7e5b7da0c65ea6f05696"),
    # New: the Next build. Byte-for-byte the Spectrum's, plus the six bytes
    # that ask for 28MHz, so its hash moves whenever the ZX one does.
    "GUESS-NEXT.TAP": (buildnext, _tap, "guess",
                       "8eb8b749a82d6d2a2d9bebbd5cb5190e6da62c522471248afb3e7679fcfc2476"),
    "CHAT-NEXT.TAP": (buildnext, _tap, "tinychat",
                      "223260a81f9d5f19a65f3ac9ae946daeeb0d8daa97ecb40c5a19f5d680d54bf3"),
    "TALK-NEXT.TAP": (buildnext, _tap, "smalltalk",
                      "f8a41626765ac6149cde0d4762ec5f849ab129a13bfdb9bec1034abe5e36e3ab"),
    # New: the Amstrad CPC build, inside the AMSDOS header RUN" reads.
    "GUESS-CPC.BIN": (buildcpc, _amsdos, "guess",
                      "db7f3c0299460d7452fa1ebb270c4448a08bd7ec54c426adf715012b8662bf81"),
    "CHAT-CPC.BIN": (buildcpc, _amsdos, "tinychat",
                     "f92338c0b4ce7eaa660a671cda51a760701b123d5eea7087d0ecd87681436558"),
    "TALK-CPC.BIN": (buildcpc, _amsdos, "smalltalk",
                     "90bbf67e27f729a5b3eb4d00b28ea582eee63a824050a30ab22625b08003eb5b"),
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
