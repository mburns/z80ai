#!/usr/bin/env python3
"""
Build an Agon .bin that probes `mos_load`, the one firmware call the oracle's
SD path depends on, and the handle calls a saved game makes.

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

The oracle calls `mos_load` and nothing else, deliberately: it is one call, it
has no handle to leak, and it exists in every MOS version. A world binary
(`buildif.py`) also saves, restores and appends to the archive's log, which is
`mos_fopen` in three modes, `mos_fwrite`, `mos_fread` and `mos_fclose`. The
`WRITTEN` probe makes exactly those calls - four bytes created, two appended -
and then loads the file back, so the six calls are covered by one line of
output, and a wrong mode byte or a handle protocol `libhost` has wrong shows
up as the wrong bytes rather than as a save that silently did not happen.

The first sixteen bytes at `DEST` are zeroed before every load, so what is
printed after a failed or short load is zeros and not the previous probe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run as a script from anywhere: the repo root holds libez80. rsplit("/") here
# assumed a POSIX path and an invocation that spelled one out - the same bug
# tools/optest.py had, which is why it uses this form.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libagon import (
    FA_CREATE_ALWAYS,
    FA_OPEN_APPEND,
    FA_WRITE,
    MOS_API,
    MOS_FCLOSE,
    MOS_FOPEN,
    MOS_FWRITE,
    MOS_LOAD,
    MOS_OUTCHAR,
)
from libez80 import AGON_LOAD_ADDR, EZ80Builder, agon_header
from libhost import AgonHost

#: The companion file. Distinctive bytes, so a wrong answer looks wrong rather
#: than plausibly zero - the same rule optest.py follows.
DATA_NAME = "MOSTEST.DAT"
DATA = bytes([0xC0, 0xFF, 0xEE, 0x01, 0x02, 0x03, 0x04, 0x05,
              0xA5, 0x5A, 0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34]) + bytes(240)

#: Where the file lands. Inside SRAM and clear of a 64KB image at 040000h.
DEST = 0x060000

#: What the write probe puts on the card: `WRITTEN` bytes created in one
#: open, `APPENDED` bytes added in a second open with `FA_OPEN_APPEND`.
WRITE_NAME = "MOSTEST.SAV"
WRITTEN = DATA[:4]
APPENDED = DATA[4:6]

PROBES = [
    ("PRESENT", DATA_NAME, "a file that exists: expect status 00"),
    ("ABSENT", "NOSUCH.DAT", "a file that does not: expect a nonzero status"),
    ("WRITTEN", WRITE_NAME, "a file this program wrote then appended to: "
                            "expect 00 and C0FFEE010203 then zeros"),
]


def _emit_write(b: EZ80Builder) -> None:
    """Create `WRITE_NAME` with four bytes, then append two more."""
    for mode, label, count in ((FA_WRITE | FA_CREATE_ALWAYS, "WDATA", len(WRITTEN)),
                               (FA_WRITE | FA_OPEN_APPEND, "WDATA2", len(APPENDED))):
        b.ld_hl_label("F_WRITTEN")
        b.ld_c_n(mode)
        b.ld_a_n(MOS_FOPEN)
        b.rst(MOS_API)
        b.ld_mem_label_a("HANDLE")
        b.ld_c_a()
        b.ld_hl_label(label)
        b.ld_de_nn(count)
        b.ld_a_n(MOS_FWRITE)
        b.rst(MOS_API)
        b.ld_a_mem_label("HANDLE")
        b.ld_c_a()
        b.ld_a_n(MOS_FCLOSE)
        b.rst(MOS_API)


def build(org: int = AGON_LOAD_ADDR) -> EZ80Builder:
    b = EZ80Builder(org=org)
    agon_header(b, "START")

    b.label("START")
    _emit_write(b)
    for name, _filename, _note in [(n, f, d) for n, f, d in PROBES]:
        b.ld_hl_label(f"T_{name}")
        b.call("PRSTR")
        b.ld_hl_nn(DEST)                 # zero what the load will overwrite
        b.ld_b_n(16)
        b.xor_a()
        b.label(f"Z_{name}")
        b.ld_hl_a()
        b.inc_hl()
        b.djnz(f"Z_{name}")
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
    b.label("HANDLE")
    b.db(0)
    b.label("WDATA")
    b.blob(WRITTEN)
    b.label("WDATA2")
    b.blob(APPENDED)
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
