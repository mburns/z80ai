#!/usr/bin/env python3
"""
Build an Agon .bin that probes the eZ80 opcodes our kernels depend on.

The eZ80 backend's speed comes from instructions that exist only on the eZ80 in
ADL mode -- ``LD rr,(IY+d)``, ``LD (IY+d),rr``, ``MLT`` -- implemented in
``libz80emu`` from a published opcode list. Nothing in CI can catch an emulator
that agrees with itself, so this prints the results of each one and they can be
compared against a real machine or an independent emulator.

    python tools/optest.py --output OPTEST.bin      # for hardware
    python tools/optest.py --expect                 # what libz80emu says

Any line that differs is a bug in our emulator, and everything the eZ80 backend
does rests on the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run as a script from anywhere: the repo root holds libez80. rsplit("/") here
# assumed a POSIX path and an invocation that spelled one out.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libez80 import AGON_LOAD_ADDR, EZ80Builder, agon_header

MOS_OUTCHAR = 0x10

#: Each probe: a label, the code that leaves a 24-bit result in HL, and a note.
#: Values are arbitrary but distinctive, so a wrong answer is obviously wrong
#: rather than plausibly zero.
PROBES: list[tuple[str, str]] = [
    ("LDHLIY", "LD (IY+6),HL then LD HL,(IY+6)  - 24-bit store/load via IY"),
    ("LDHLIX", "LD (IX+9),HL then LD HL,(IX+9)  - same via IX"),
    ("LDDEIX", "LD (IX+3),DE then LD DE,(IX+3)  - the pair the layer kernel uses"),
    ("NEGDIS", "LD (IY-4),HL then LD HL,(IY-4)  - negative displacement"),
    ("MLTHL", "MLT HL with H=0Dh L=13h          - 8x8 multiply"),
    ("ADD24", "24-bit ADD HL,DE across a carry into bits 16-23"),
    ("POP24", "POP reads three bytes in ADL mode"),
]


#: Where the probes keep their scratch when running as firmware. Address zero
#: is flash on a real Agon, so a store there is silently discarded; RAM starts
#: at 040000h. Getting this wrong makes every store-then-load probe read back
#: zero and look like an emulator disagreement.
RAM_SCRATCH = 0x050000


def emit_probe(b: EZ80Builder, name: str, scratch: int | None = None) -> None:
    """Emit one probe, leaving its 24-bit result in HL.

    ``scratch`` is an absolute RAM address when the probe runs as firmware;
    None uses the SCRATCH label inside the image, which is only writable when
    the image itself is loaded into RAM.
    """

    def point_ix() -> None:
        if scratch is None:
            b.ld_ix_label("SCRATCH")
        else:
            b.ld_ix_nn(scratch)

    def point_iy() -> None:
        if scratch is None:
            b.ld_iy_label("SCRATCH")
        else:
            b.ld_iy_nn(scratch)

    if name == "LDHLIY":
        point_iy()
        b.ld_hl_nn(0xA1B2C3)
        b.ld_iyd_hl(6)
        b.ld_hl_nn(0)
        b.ld_hl_iyd(6)
    elif name == "LDHLIX":
        point_ix()
        b.ld_hl_nn(0x4D5E6F)
        b.ld_ixd_hl(9)
        b.ld_hl_nn(0)
        b.ld_hl_ixd(9)
    elif name == "LDDEIX":
        point_ix()
        b.ld_de_nn(0x778899)
        b.ld_ixd_de(3)
        b.ld_de_nn(0)
        b.ld_de_ixd(3)
        b.ex_de_hl()
    elif name == "NEGDIS":
        point_iy()
        b.ld_de_nn(16)
        b.add_iy_de()
        b.ld_hl_nn(0x0F1E2D)
        b.ld_iyd_hl(-4)
        b.ld_hl_nn(0)
        b.ld_hl_iyd(-4)
    elif name == "MLTHL":
        b.ld_hl_nn(0x000D13)  # H = 0Dh, L = 13h -> 0Dh * 13h = 00F7h
        b.mlt_hl()
    elif name == "ADD24":
        b.ld_hl_nn(0x00FFFF)
        b.ld_de_nn(0x000002)
        b.add_hl_de()  # 010001h if the add is really 24-bit
    elif name == "POP24":
        b.ld_hl_label("POPDATA")
        b.ld_mem_label_sp("SPSAVE")
        b.di()
        b.ld_sp_hl()
        b.pop_hl()
        b.ld_sp_mem_label("SPSAVE")
        b.ei()
    else:  # pragma: no cover
        raise ValueError(name)


#: fab-agon-emulator debug IO ports (see its README, "Debug IO space").
DEBUG_IO_SHUTDOWN = 0x00
DEBUG_IO_CPU_STATE = 0x20


def build_firmware() -> EZ80Builder:
    """Build the probe as MOS firmware, running from reset at 000000h.

    Handed to fab-agon-emulator with --mos, this replaces MOS entirely: no SD
    card, no autoexec, no VDP. The eZ80 resets into Z80 mode, so the first job
    is JP.LIL into ADL mode; after that it is the same probe code.

    Each probe point prints CPU state through the emulator's debug IO port, and
    the last instruction shuts the emulator down, so the run is self-contained
    and terminates on its own.
    """
    b = EZ80Builder(org=0x000000)

    # Reset state is ADL=0. JP.LIL (5B C3 nn nn nn) jumps and enters ADL mode.
    b.di()
    b.emit(0x5B, 0xC3)
    b.fixup_word("ADLSTART")

    b.label("ADLSTART")
    for name, _note in PROBES:
        emit_probe(b, name, scratch=RAM_SCRATCH)
        b.label(f"P_{name}")   # HL holds the result here
        b.ld_a_n(0)
        b.out_n_a(DEBUG_IO_CPU_STATE)   # dump registers to the terminal

    b.label("DONE")
    b.ld_a_n(0)
    b.out_n_a(DEBUG_IO_SHUTDOWN)        # stop the emulator
    b.halt()

    b.label("SPSAVE")
    b.d24(0)
    b.label("POPDATA")
    b.d24(0xC0FFEE)
    b.label("SCRATCH")
    b.ds(64)
    return b


def build(org: int = AGON_LOAD_ADDR) -> EZ80Builder:
    b = EZ80Builder(org=org)
    agon_header(b, "START")

    b.label("START")
    for name, _note in PROBES:
        b.ld_hl_label(f"T_{name}")
        b.call("PRSTR")
        emit_probe(b, name)
        b.label(f"P_{name}")   # HL holds the result here
        b.call("PRHL")
        b.call("PRNL")
    b.label("DONE")
    b.ret()

    # PRHL: print HL as six hex digits, most significant first.
    b.label("PRHL")
    b.ld_mem_label_hl("HEXVAL")
    for offset in (2, 1, 0):
        b.ld_a_mem_label(f"HEXVAL{offset}")
        b.call("PRHEX")
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

    # PRSTR: print the NUL-terminated string at HL.
    b.label("PRSTR")
    b.ld_a_hl()
    b.or_a()
    b.ret_z()
    b.rst(MOS_OUTCHAR)
    b.inc_hl()
    b.jr("PRSTR")

    b.label("PRNL")
    b.ld_a_n(13)
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(10)
    b.rst(MOS_OUTCHAR)
    b.ret()

    # Data.
    for name, _note in PROBES:
        b.label(f"T_{name}")
        b.ascii(f"{name} ")
        b.db(0)

    b.label("HEXVAL")
    b.label("HEXVAL0")
    b.db(0)
    b.label("HEXVAL1")
    b.db(0)
    b.label("HEXVAL2")
    b.db(0)
    b.label("SPSAVE")
    b.d24(0)
    b.label("POPDATA")
    b.d24(0xC0FFEE)
    b.label("SCRATCH")
    b.ds(64)
    return b


def expected() -> list[tuple[str, str]]:
    """Run the probe in our own emulator and report what it prints."""
    from libhost import run_agon

    image = build().build()
    out, _host = run_agon(image, max_cycles=50_000_000)
    return [
        (line.split()[0], line.split()[1])
        for line in out.replace("\r", "").split("\n")
        if len(line.split()) == 2
    ]


def triggers() -> str:
    """Debugger commands that dump HL at each probe point, for fab-agon-emulator."""
    b = build()
    b.build()
    lines = [f"trigger {b.labels[f'P_{name}']:x} state" for name, _ in PROBES]
    lines.append(f"trigger {b.labels['DONE']:x} state : exit")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", "-o", help="Write the .bin here")
    parser.add_argument("--expect", action="store_true",
                        help="Print what libz80emu says the results are")
    parser.add_argument("--triggers", action="store_true",
                        help="Print fab-agon-emulator debugger commands")
    parser.add_argument("--firmware", metavar="PATH",
                        help="Build the probe as MOS firmware for --mos")
    args = parser.parse_args()

    if args.firmware:
        fw = build_firmware()
        fw.save(args.firmware)
        print(f"{len(fw.code)} bytes, runs from reset at 000000h")
        print("\nRun with:")
        print(f"  fab-agon-emulator --mos {args.firmware} -d -u </dev/null")
        print("\nProbe points (each prints CPU state; HL holds the result):")
        for name, _note in PROBES:
            print(f"  {fw.labels[f'P_{name}']:06x}  {name}")
        return

    if args.triggers:
        print(triggers(), end="")
        return

    b = build()
    if args.output:
        b.save(args.output)
        print(f"{len(b.code)} bytes, loads at {AGON_LOAD_ADDR:06X}h")
        print("\nRun it on an Agon and compare against --expect.")

    if args.expect or not args.output:
        print(f"\n{'probe':8} {'libz80emu':>10}   what it checks")
        print("-" * 72)
        results = dict(expected())
        for name, note in PROBES:
            print(f"{name:8} {results.get(name, '(none)'):>10}   {note}")


if __name__ == "__main__":
    main()
