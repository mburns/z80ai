#!/usr/bin/env python3
"""
Build an Agon .bin that probes `mos_load`, the one firmware call the SD path
depends on.

`tools/optest.py` establishes that libz80emu decodes the eZ80's instructions the
way real silicon does. This establishes the other half: that `libhost.AgonHost`
models the *firmware* the way real MOS behaves. Nothing in CI can do that -
MOS would not boot under fab-agon-emulator on the machine this was written on
(see tools/README.md), which is why optest bypasses MOS entirely and runs as
firmware. This one cannot: mos_load only exists when MOS is there.

    python tools/mostest.py --output MOSTEST.bin
    # copy MOSTEST.bin and MOSTEST.DAT onto a card, run MOSTEST from MOS

It prints the status byte and the first sixteen bytes actually loaded. Both are
checked against libhost below, so `--expect` and the hardware run can be
compared line for line and the result recorded in tools/README.md.

Shipped code calls `mos_load` and nothing else, deliberately: it is one call,
it has no handle to leak, it exists in every MOS version, and it means this
probe covers the whole unvalidated surface rather than a sample of it.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from libez80 import AGON_LOAD_ADDR, EZ80Builder, agon_header
from libhost import AgonHost

MOS_OUTCHAR = 0x10
MOS_API = 0x08
MOS_LOAD = 0x01

#: The companion file. Distinctive bytes, so a wrong answer looks wrong rather
#: than plausibly zero - the same rule optest.py follows.
DATA_NAME = "MOSTEST.DAT"
DATA = bytes([0xC0, 0xFF, 0xEE, 0x01, 0x02, 0x03, 0x04, 0x05,
              0xA5, 0x5A, 0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34]) + bytes(240)

#: Where the file lands. Inside SRAM and clear of a 64KB image at 040000h.
DEST = 0x060000

PROBES = [
    ("PRESENT", DATA_NAME, "a file that exists: expect status 00"),
    ("ABSENT", "NOSUCH.DAT", "a file that does not: expect a nonzero status"),
]


def build(org: int = AGON_LOAD_ADDR) -> EZ80Builder:
    b = EZ80Builder(org=org)
    agon_header(b, "START")

    b.label("START")
    for name, _filename, _note in [(n, f, d) for n, f, d in PROBES]:
        b.ld_hl_label(f"T_{name}")
        b.call("PRSTR")
        b.ld_hl_label(f"F_{name}")
        b.ld_de_nn(DEST)
        b.ld_bc_nn(len(DATA))
        b.ld_a_n(MOS_LOAD)
        b.rst(MOS_API)
        # A is the status; keep it before anything else disturbs it.
        b.ld_mem_label_a("STATUS")
        b.ld_a_mem_label("STATUS")
        b.call("PRHEX")
        b.call("PRSP")
        b.call("PRDATA")
        b.call("PRNL")
    b.ret()

    # PRDATA: the first sixteen bytes at DEST, as hex pairs.
    b.label("PRDATA")
    b.ld_hl_nn(DEST)
    b.ld_b_n(16)
    b.label("PRDATA1")
    b.push_bc()
    b.push_hl()
    b.ld_a_hl()
    b.call("PRHEX")
    b.pop_hl()
    b.pop_bc()
    b.inc_hl()
    b.djnz("PRDATA1")
    b.ret()

    b.label("PRHEX")
    b.push_af()
    b.rrca()
    b.rrca()
    b.rrca()
    b.rrca()
    b.call("PRNYB")
    b.pop_af()
    b.call("PRNYB")
    b.ret()

    b.label("PRNYB")
    b.and_n(0x0F)
    b.cp_n(10)
    b.jr_c("PRNYB_D")
    b.add_a_n(ord("A") - 10)
    b.rst(MOS_OUTCHAR)
    b.ret()
    b.label("PRNYB_D")
    b.add_a_n(ord("0"))
    b.rst(MOS_OUTCHAR)
    b.ret()

    b.label("PRSTR")
    b.ld_a_hl()
    b.or_a()
    b.ret_z()
    b.rst(MOS_OUTCHAR)
    b.inc_hl()
    b.jr("PRSTR")

    b.label("PRSP")
    b.ld_a_n(ord(" "))
    b.rst(MOS_OUTCHAR)
    b.ret()

    b.label("PRNL")
    b.ld_a_n(13)
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(10)
    b.rst(MOS_OUTCHAR)
    b.ret()

    for name, filename, _note in PROBES:
        b.label(f"T_{name}")
        b.ascii(f"{name}: ")
        b.db(0)
        b.label(f"F_{name}")
        b.ascii(filename)
        b.db(0)

    b.label("STATUS")
    b.db(0)
    return b


def expected() -> str:
    """What libhost says, which is what hardware is being compared against."""
    host = AgonHost(files={DATA_NAME: DATA})
    return host.run(build().build())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o", help="Write the .bin for hardware")
    parser.add_argument("--data", help="Write the companion .DAT for the card")
    args = parser.parse_args()

    if args.output:
        builder = build()
        builder.save(args.output)
        print(f"wrote {args.output} ({len(builder.build()):,} bytes)")
    if args.data:
        with open(args.data, "wb") as fh:
            fh.write(DATA)
        print(f"wrote {args.data} ({len(DATA):,} bytes)")
    if not args.output and not args.data:
        print(f"# copy {DATA_NAME} alongside the .bin, then compare:\n")
        print(expected())
        for name, filename, note in PROBES:
            print(f"# {name}: {filename} - {note}")


if __name__ == "__main__":
    main()
