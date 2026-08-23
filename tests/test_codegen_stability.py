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

import buildez80
import buildfastz80com
import buildz80com
import buildz80tap

# artifact -> (module, example, sha256 of the image as written to disk)
GOLDEN = {
    "GUESS.COM": (buildz80com, "guess",
                  "1c26b542228fff147991e4b2b8c1e92652d46b4fb8b9eeafbb0442169b085a5a"),
    "GUESS-FAST.COM": (buildfastz80com, "guess",
                       "cceee1ec8d138f707d5759f8100bfa9c14476ed04aa4aedc31839d11efe94baf"),
    # Changed when eZ80 ARGMAX stopped counting neurons in B (no 256-output
    # cap, 24-bit MAXI/RESULT) and again when the default kernel became the
    # unrolled column-major one - 23x fewer instructions for 2.6x the size.
    "GUESS.bin": (buildez80, "guess",
                  "8eb986d2c73574344be128c66a9a71e08727bcc948d4afe8d35451f35c3db751"),
    # Changed when tinychat's 502 replies were collapsed onto a 21-word
    # vocabulary; the model is retrained on the canonicalized data.
    "CHAT.COM": (buildz80com, "tinychat",
                 "22b53d001ac679633aa2490229af91303b4d3dfa569d2251210a81e8b6847e7c"),
    "CHAT-FAST.COM": (buildfastz80com, "tinychat",
                      "124e15d11ace998b3f669af6f2cbd3b705a689ba42018ad2c0709dfcc8ef7b6d"),
    "CHAT.bin": (buildez80, "tinychat",
                 "f97df19f7e07d2c979a2ce981294b9823836625227591ca114bd78e52e50287a"),
    "TALK.COM": (buildz80com, "smalltalk",
                 "75842adb7a8ec135d54f71bc082aaf4fef62691def1306ed4242f59808353e51"),
    "TALK-FAST.COM": (buildfastz80com, "smalltalk",
                      "33e8a588e06b1392a522e031bb9b478700c3cb0e2a87efcdcfef088acdae7d8c"),
    "TALK.bin": (buildez80, "smalltalk",
                 "6efec989a6d3bb11af1c3f3846b0a98ebe26b0a4b80e9f48ce725e29d6b740b2"),
}

# The .TAP hashes cover the container, not the raw image, so they are checked
# through the same header/data blocks the build script writes.
GOLDEN_TAP = {
    # Changed when the ZX build adopted the CP/M inner loop: same arithmetic,
    # 26% fewer instructions, 32 bytes smaller.
    "GUESS.TAP": ("guess",
                  "f357b874901daec1f8ced9df2b0ff529471eef04a7d1256165d232ad48b46d86"),
    "CHAT.TAP": ("tinychat",
                 "2283521cbd6d343ae870a560c0386eca66177d066de448b44d30abef049f9f09"),
    "TALK.TAP": ("smalltalk",
                 "4db2e3f77b05b85165cee46fc16151cfa4a615a81b059ac8b2b5d9f647d58213"),
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


@pytest.mark.parametrize("artifact", sorted(GOLDEN_TAP))
def test_generated_tap_is_unchanged(artifact, examples_dir):
    example, expected = GOLDEN_TAP[artifact]
    builder = buildz80tap.build_autoreg(model_path(examples_dir, example))
    image = builder.build()
    tap = buildz80tap.build_tap_header("CHAT", builder.org, len(image))
    tap += buildz80tap.build_tap_data(image)
    got = hashlib.sha256(tap).hexdigest()
    assert got == expected, (
        f"{artifact} changed: {got}\n"
        f"If that was intended, update GOLDEN_TAP in this file and say why in "
        f"the commit message."
    )


def test_every_shipped_artifact_is_covered():
    """build-examples.sh and the release list must not outgrow this file."""
    import verify_artifacts

    assert set(verify_artifacts.ARTIFACTS) == set(GOLDEN) | set(GOLDEN_TAP)
