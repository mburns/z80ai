#!/usr/bin/env python3
"""
Execute release artifacts and check they still answer correctly.

CI used to build binaries and ship them without ever running one. This walks a
dist directory, boots each artifact in the emulator on its own platform, and
compares what it prints against the NumPy reference model. It also re-checks the
container formats and the size limits each platform imposes.

A build that assembles but computes the wrong thing - which is exactly what the
MULADD borrow bug did for years - fails here.

Usage:
    python verify_artifacts.py --dist dist
    python verify_artifacts.py --dist dist --query "IS IT AN ANIMAL"
"""

from __future__ import annotations

import argparse
import os
import sys

import libcpc
import libinfer
import libnext
from libcpm import TPA as CPM_TPA
from libcpm import TPA_TOP as CPM_TPA_TOP
from libez80 import AGON_LOAD_ADDR as AGON_LOAD
from libez80 import AGON_SRAM_TOP, AGON_STACK_MARGIN
from libhost import run_agon, run_cpc, run_cpm, run_next, run_zx
from libzx import TAP_FLAG_DATA, TAP_FLAG_HEADER, ZX_RAM_TOP

#: artifact name -> (model path, platform)
ARTIFACTS = {
    "GUESS.COM": ("examples/guess/model.npz", "cpm"),
    "GUESS-FAST.COM": ("examples/guess/model.npz", "cpm"),
    "GUESS-COL.COM": ("examples/guess/model.npz", "cpm"),
    "GUESS.TAP": ("examples/guess/model.npz", "zx"),
    "GUESS-NEXT.TAP": ("examples/guess/model.npz", "next"),
    "GUESS-CPC.BIN": ("examples/guess/model.npz", "cpc"),
    "GUESS.bin": ("examples/guess/model.npz", "agon"),
    "CHAT.COM": ("examples/tinychat/model.npz", "cpm"),
    "CHAT-FAST.COM": ("examples/tinychat/model.npz", "cpm"),
    "CHAT-COL.COM": ("examples/tinychat/model.npz", "cpm"),
    "CHAT.TAP": ("examples/tinychat/model.npz", "zx"),
    "CHAT-NEXT.TAP": ("examples/tinychat/model.npz", "next"),
    "CHAT-CPC.BIN": ("examples/tinychat/model.npz", "cpc"),
    "CHAT.bin": ("examples/tinychat/model.npz", "agon"),
    "TALK.COM": ("examples/smalltalk/model.npz", "cpm"),
    "TALK-FAST.COM": ("examples/smalltalk/model.npz", "cpm"),
    "TALK-COL.COM": ("examples/smalltalk/model.npz", "cpm"),
    "TALK.TAP": ("examples/smalltalk/model.npz", "zx"),
    "TALK-NEXT.TAP": ("examples/smalltalk/model.npz", "next"),
    "TALK-CPC.BIN": ("examples/smalltalk/model.npz", "cpc"),
    "TALK.bin": ("examples/smalltalk/model.npz", "agon"),
    "CLINC.bin": ("examples/clinc150/model.npz", "agon-phrasebook"),
    "TALK-PHR.bin": ("examples/smalltalk/phrasebook.npz", "agon-phrasebook"),
}

#: Artifacts that need a companion file on the SD card to run at all. Kept
#: beside ARTIFACTS rather than folded into it because two tests iterate that
#: dict and expect two-tuples.
COMPANIONS = {
    "CLINC.bin": "CLINC.DAT",
    "TALK-PHR.bin": "TALK-PHR.DAT",
}


class VerificationError(Exception):
    """An artifact did not behave the way the reference model says it should."""


def parse_tap(data: bytes) -> tuple[bytes, int]:
    """Unwrap a .TAP container, validating both block checksums.

    A TAP is a sequence of [length:2][flag:1][payload][checksum:1] blocks; a
    CODE file is a 19-byte header block followed by the data block.

    Returns:
        The raw image and the load address declared in the header.
    """
    blocks, pos = [], 0
    while pos < len(data):
        if pos + 2 > len(data):
            raise VerificationError("truncated TAP block length")
        length = data[pos] | (data[pos + 1] << 8)
        body = data[pos + 2 : pos + 2 + length]
        if len(body) != length:
            raise VerificationError("truncated TAP block body")
        checksum = 0
        for byte in body[:-1]:
            checksum ^= byte
        if checksum != body[-1]:
            raise VerificationError(
                f"TAP checksum mismatch: got {body[-1]:02X}, want {checksum:02X}"
            )
        blocks.append(body)
        pos += 2 + length

    if (len(blocks) != 2 or blocks[0][0] != TAP_FLAG_HEADER
            or blocks[1][0] != TAP_FLAG_DATA):
        raise VerificationError("expected one header block and one data block")

    header = blocks[0]
    declared = header[12] | (header[13] << 8)
    start = header[14] | (header[15] << 8)
    image = blocks[1][1:-1]
    if declared != len(image):
        raise VerificationError(
            f"TAP header declares {declared} bytes, data block holds {len(image)}"
        )
    return image, start


def parse_amsdos(data: bytes) -> tuple[bytes, int]:
    """Unwrap an AMSDOS binary, validating the header checksum.

    The header is the first 128 bytes of the file as stored on disc; AMSDOS
    treats one whose checksum does not match as a headerless file and would
    load it to the wrong address, so the checksum is the thing worth checking.

    Returns:
        The raw image and the load address declared in the header.
    """
    if len(data) <= libcpc.AMSDOS_HEADER_LEN:
        raise VerificationError("file is shorter than an AMSDOS header")

    head, image = data[:libcpc.AMSDOS_HEADER_LEN], data[libcpc.AMSDOS_HEADER_LEN:]
    at = libcpc.AMSDOS_CHECKSUM_AT
    want = sum(head[:at]) & 0xFFFF
    got = head[at] | (head[at + 1] << 8)
    if got != want:
        raise VerificationError(
            f"AMSDOS checksum mismatch: got {got:04X}, want {want:04X}"
        )

    if head[18] != libcpc.AMSDOS_TYPE_BINARY:
        raise VerificationError(f"AMSDOS file type {head[18]} is not a binary")

    load = head[21] | (head[22] << 8)
    entry = head[26] | (head[27] << 8)
    declared = head[64] | (head[65] << 8) | (head[66] << 16)
    if declared != len(image):
        raise VerificationError(
            f"AMSDOS header declares {declared} bytes, file holds {len(image)}"
        )
    if entry != load:
        raise VerificationError(
            f"entry {entry:#06x} is not the load address {load:#06x}; "
            f"RUN\" would start in the middle of the image"
        )
    return image, load


def check_fits(org: int, size: int, top: int, where: str) -> None:
    """Refuse an artifact that cannot load on the machine it targets."""
    end = org + size
    if end > top:
        raise VerificationError(
            f"{size:,} bytes loaded at {org:#06x} runs to {end:#07x}, "
            f"past {where} ({top - 1:#06x}) by {end - top:,} bytes"
        )


def run_artifact(path: str, platform: str, query: str,
                 files: dict[str, bytes] | None = None) -> str:
    """Boot the artifact on its platform and return what it printed."""
    with open(path, "rb") as fh:
        data = fh.read()

    if platform == "cpm":
        check_fits(CPM_TPA, len(data), CPM_TPA_TOP, "the top of the CP/M TPA")
        out, host = run_cpm(data, cmdline=query)
        if not host.finished:
            raise VerificationError("program never returned to CP/M")
        return out

    if platform == "zx":
        # The load address comes from the TAP header, so this checks the
        # artifact as a Spectrum would actually see it.
        image, org = parse_tap(data)
        check_fits(org, len(image), ZX_RAM_TOP, "the top of 48K RAM")
        out, _zx_host = run_zx(image, stdin=[query, "!"], org=org)
        return out

    if platform == "next":
        image, org = parse_tap(data)
        check_fits(org, len(image), ZX_RAM_TOP, "the top of 48K RAM")
        out, next_host = run_next(image, stdin=[query, "!"], org=org)
        # The whole reason this target exists. A Next build that forgot to ask
        # for the faster clock is just the Spectrum build under another name.
        if next_host.cpu_speed != libnext.DEFAULT_SPEED:
            raise VerificationError(
                f"asked the Next for {next_host.cpu_speed}MHz, expected "
                f"{libnext.DEFAULT_SPEED}MHz"
            )
        return out

    if platform == "cpc":
        image, org = parse_amsdos(data)
        check_fits(org, len(image), libcpc.CPC_HIMEM, "the CPC's HIMEM")
        out, _cpc_host = run_cpc(image, stdin=[query, "!"], org=org)
        return out

    if platform in ("agon", "agon-phrasebook"):
        check_fits(AGON_LOAD, len(data), AGON_SRAM_TOP - AGON_STACK_MARGIN,
                   "the top of Agon SRAM")
        out, _agon_host = run_agon(data, stdin=[query, "!"], files=files)
        return out

    raise VerificationError(f"unknown platform {platform}")


def release_files() -> list[str]:
    """Every file a release should publish, in the order ARTIFACTS lists them.

    The release step in ci.yml reads this rather than restating it. It used to
    restate it, and had drifted: the column-major .COMs and both phrasebooks
    were built, verified and pinned, but never published - while the README
    told people to download one of them.
    """
    files: list[str] = []
    for name in ARTIFACTS:
        files.append(name)
        if name in COMPANIONS:
            files.append(COMPANIONS[name])
    return files


def missing_from(dist: str) -> list[str]:
    """Release files that are not in ``dist``.

    ``verify`` skips what is absent, which is convenient when checking a single
    target by hand and useless as a release gate - a dist directory holding one
    artifact would pass. CI calls this instead.
    """
    return [name for name in release_files()
            if not os.path.exists(os.path.join(dist, name))]


def verify(dist: str, query: str) -> list[tuple[str, str, str, bool]]:
    """Check every artifact present in ``dist``. Returns per-artifact results."""
    results = []
    for name, (model_path, platform) in ARTIFACTS.items():
        path = os.path.join(dist, name)
        if not os.path.exists(path):
            continue

        model = libinfer.Model.load(model_path)
        # The eZ80 accumulates in 24 bits and so never wraps; the Z80 targets
        # wrap at 16. Compare each against the semantics it actually implements.
        bits = 24 if platform.startswith("agon") else 16

        files = None
        if platform == "agon-phrasebook":
            # The replies are not in the binary - the whole point - so the
            # card has to be served too, and a missing companion must fail the
            # verification rather than silently checking a program that could
            # not answer anything.
            companion = os.path.join(dist, COMPANIONS[name])
            if not os.path.exists(companion):
                raise VerificationError(
                    f"{name} needs {COMPANIONS[name]} beside it and it is missing")
            with open(companion, "rb") as fh:
                files = {COMPANIONS[name]: fh.read()}
            expected = libinfer.classify(model, query, accum_bits=bits)
        else:
            expected = libinfer.generate(model, query, accum_bits=bits)

        printed = run_artifact(path, platform, query, files)
        # CP/M single-query mode prints only the reply; every other target is
        # chat-driven and wraps it in prompt and echo chrome.
        ok = printed == expected if platform == "cpm" else expected in printed
        results.append((name, expected, printed, ok))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dist", "-d", default="dist", help="Directory of built artifacts")
    parser.add_argument("--query", "-q", default="HELLO", help="Query to send")
    parser.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"),
                        help="Append a markdown table here (defaults to the CI summary)")
    parser.add_argument("--require-all", action="store_true",
                        help="Fail if any known artifact is missing (what CI uses, "
                             "so a release cannot ship a partial dist)")
    parser.add_argument("--list", action="store_true",
                        help="Print the files a release should publish, one per "
                             "line, and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(release_files()))
        return 0

    if args.require_all:
        absent = missing_from(args.dist)
        if absent:
            print(f"FAIL: {len(absent)} release file(s) missing from "
                  f"{args.dist}/: {', '.join(absent)}", file=sys.stderr)
            return 1

    try:
        results = verify(args.dist, args.query)
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(f"No known artifacts found in {args.dist}/", file=sys.stderr)
        return 1

    lines = ["| artifact | expected | got | |", "|---|---|---|---|"]
    failures = 0
    print(f"\nVerifying {len(results)} artifacts with query {args.query!r}\n")
    for name, expected, printed, ok in results:
        mark = "ok" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  {mark:4}  {name:16} expected {expected!r}")
        if not ok:
            print(f"        got {printed!r}")
        lines.append(
            f"| `{name}` | `{expected}` | `{printed.strip()[:24]}` | {'✅' if ok else '❌'} |"
        )

    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write(f"\n### Artifact verification (`{args.query}`)\n\n")
            fh.write("\n".join(lines) + "\n")

    print()
    if failures:
        print(f"{failures} of {len(results)} artifacts did not match the reference model")
        return 1
    print(f"All {len(results)} artifacts match the reference model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
