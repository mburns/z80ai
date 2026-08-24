#!/usr/bin/env python3
"""
Disassemble generated code, annotated with the labels that produced it.

Five places in this project print addresses "for cross-referencing a
disassembly" and until now nothing produced one, so reading generated code
meant reading the Python that emits it and imagining the bytes. That is how the
MULADD borrow bug survived for years.

    python tools/disasm.py --model examples/guess/model.npz --at LAYER
    python tools/disasm.py --model examples/guess/model.npz --target ez80 --at PREQ
    python tools/disasm.py --model examples/guess/model.npz --labels

Decoding uses the same ``x/y/z/p/q`` decomposition as ``libz80emu``, which is
what keeps a complete Z80 in a page of tables rather than a 256-entry switch.
The two are independent implementations of the same instruction set, and
``tests/test_disasm.py`` plays them against each other: for every instruction in
a shipped artifact, the length this decoder reports must equal the distance the
emulator's PC actually moved.

Bytes that decode to nothing are rendered as ``DB``, so a data region read as
code produces a listing rather than an exception.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The eight decode tables the Z80's encoding is built from.
R = ("B", "C", "D", "E", "H", "L", "(HL)", "A")
RP = ("BC", "DE", "HL", "SP")
RP2 = ("BC", "DE", "HL", "AF")
CC = ("NZ", "Z", "NC", "C", "PO", "PE", "P", "M")
ALU = ("ADD A,", "ADC A,", "SUB ", "SBC A,", "AND ", "XOR ", "OR ", "CP ")
ROT = ("RLC", "RRC", "RL", "RR", "SLA", "SRA", "SLL", "SRL")
IM = ("0", "0/1", "1", "2", "0", "0/1", "1", "2")
BLOCK = {
    (4, 0): "LDI", (4, 1): "CPI", (4, 2): "INI", (4, 3): "OUTI",
    (5, 0): "LDD", (5, 1): "CPD", (5, 2): "IND", (5, 3): "OUTD",
    (6, 0): "LDIR", (6, 1): "CPIR", (6, 2): "INIR", (6, 3): "OTIR",
    (7, 0): "LDDR", (7, 1): "CPDR", (7, 2): "INDR", (7, 3): "OTDR",
}

#: eZ80 instructions that are not Z80 encodings at all, keyed by their ED byte.
ED_EZ80 = {0x4C: "MLT BC", 0x5C: "MLT DE", 0x6C: "MLT HL", 0x7C: "MLT SP"}

#: eZ80 24-bit loads through an index register, keyed by (prefix, opcode).
#: ``LD (IX+d),HL`` and friends: the pair is chosen by the opcode's high nibble.
IDX_PAIR = {0x0: "BC", 0x1: "DE", 0x2: "HL", 0x3: "IY"}


@dataclass(frozen=True)
class Instruction:
    """One decoded instruction, or one byte of data that decoded to nothing."""

    addr: int
    raw: bytes
    text: str

    @property
    def length(self) -> int:
        return len(self.raw)

    @property
    def is_data(self) -> bool:
        """True when the bytes did not decode, and are shown as DB."""
        return self.text.startswith("DB ")


class _Reader:
    """A cursor over the image that records what it consumed."""

    def __init__(self, image: bytes, org: int, addr: int, adl: bool) -> None:
        self.image, self.org, self.start, self.adl = image, org, addr, adl
        self.pos = addr - org

    def byte(self) -> int:
        if not 0 <= self.pos < len(self.image):
            raise IndexError("ran off the end of the image")
        value = self.image[self.pos]
        self.pos += 1
        return value

    def signed(self) -> int:
        value = self.byte()
        return value - 256 if value > 127 else value

    def word(self) -> int:
        """An address operand: two bytes, or three in ADL mode."""
        value = self.byte() | (self.byte() << 8)
        if self.adl:
            value |= self.byte() << 16
        return value

    def consumed(self) -> bytes:
        return self.image[self.start - self.org : self.pos]


def _hex(value: int, width: int = 2) -> str:
    """Z80 assembly convention: a trailing h, and a leading digit.

    An address wider than 16 bits is padded to six digits rather than five, so
    an eZ80 operand looks like an address and :func:`annotate` can match it.
    """
    if width >= 4 and value > 0xFFFF:
        width = 6
    text = f"{value:0{width}X}h"
    return text if text[0].isdigit() else "0" + text


def _idx(prefix: int) -> str:
    return "IX" if prefix == 0xDD else "IY"


def _decode_cb(r: _Reader, prefix: int | None, disp: int | None) -> str:
    op = r.byte()
    x, y, z = op >> 6, (op >> 3) & 7, op & 7
    target = f"({_idx(prefix)}{disp:+d})" if prefix is not None else R[z]
    if x == 0:
        return f"{ROT[y]} {target}"
    return f"{('BIT', 'RES', 'SET')[x - 1]} {y},{target}"


def _decode_ed(r: _Reader) -> str:
    op = r.byte()
    if op in ED_EZ80:
        return ED_EZ80[op]
    x, y, z, p, q = op >> 6, (op >> 3) & 7, op & 7, (op >> 4) & 3, (op >> 3) & 1

    if x == 2 and z <= 3 and y >= 4:
        return BLOCK[(y, z)]
    if x != 1:
        # Undecodable: fall back to the caller's single-byte DB rather than
        # swallowing the operand too, so a linear sweep can resynchronise on
        # the next byte instead of skipping past it.
        return ""

    if z == 0:
        return "IN (C)" if y == 6 else f"IN {R[y]},(C)"
    if z == 1:
        return "OUT (C),0" if y == 6 else f"OUT (C),{R[y]}"
    if z == 2:
        return f"{'SBC' if q == 0 else 'ADC'} HL,{RP[p]}"
    if z == 3:
        # ED 5Bh is LD DE,(nn); the eZ80 build uses it for a 24-bit load.
        return (f"LD ({_hex(r.word(), 4)}),{RP[p]}" if q == 0
                else f"LD {RP[p]},({_hex(r.word(), 4)})")
    if z == 4:
        return "NEG"
    if z == 5:
        return "RETI" if y == 1 else "RETN"
    if z == 6:
        return f"IM {IM[y]}"
    return ("LD I,A", "LD R,A", "LD A,I", "LD A,R", "RRD", "RLD", "NOP", "NOP")[y]


def _decode_indexed(r: _Reader, prefix: int) -> str:
    """DD/FD: the Z80's HL-to-IX/IY substitution, plus the eZ80's extras."""
    op = r.byte()
    reg = _idx(prefix)

    # eZ80: LD (IX+d),rr and LD rr,(IX+d) - 24-bit pair moves that plain Z80
    # has no encoding for. Bit 3 of the opcode picks the direction.
    if (op & 0x07) == 0x07 and (op >> 4) <= 3 and op not in (0x07, 0x77):
        pair = IDX_PAIR[op >> 4]
        disp = r.signed()
        return (f"LD ({reg}{disp:+d}),{pair}" if op & 0x08 == 0
                else f"LD {pair},({reg}{disp:+d})")
    if (op & 0x07) == 0x0F and (op >> 4) <= 3:
        pair = IDX_PAIR[op >> 4]
        disp = r.signed()
        return (f"LD ({reg}{disp:+d}),{pair}" if op & 0x08 else
                f"LD {pair},({reg}{disp:+d})")

    if op == 0xCB:
        disp = r.signed()
        return _decode_cb(r, prefix, disp)
    if op == 0x77:  # eZ80 LD (IX+d),A in the build's 24-bit form
        disp = r.signed()
        return f"LD ({reg}{disp:+d}),A"
    if op == 0x2A:
        return f"LD {reg},({_hex(r.word(), 4)})"

    # Otherwise decode as a base instruction with HL swapped for the index
    # register, which is exactly what the prefix means.
    text = _decode_base(r, op, index=reg)
    return text


def _sub_index(name: str, index: str, disp: int | None) -> str:
    if name == "(HL)":
        return f"({index}{disp:+d})" if disp is not None else f"({index})"
    if name == "HL":
        return index
    if name in ("H", "L"):
        return index + name
    return name


def _decode_base(r: _Reader, op: int, index: str | None = None) -> str:
    x, y, z = op >> 6, (op >> 3) & 7, op & 7
    p, q = (op >> 4) & 3, (op >> 3) & 1

    def reg(i: int, disp_first: bool = False) -> str:
        if index is None:
            return R[i]
        disp = r.signed() if (R[i] == "(HL)" and disp_first) else None
        return _sub_index(R[i], index, disp)

    def pair(i: int) -> str:
        return _sub_index(RP[i], index, None) if index else RP[i]

    if x == 0:
        if z == 0:
            if y == 0:
                return "NOP"
            if y == 1:
                return "EX AF,AF'"
            if y == 2:
                return f"DJNZ {_hex(_rel(r), 4)}"
            if y == 3:
                return f"JR {_hex(_rel(r), 4)}"
            return f"JR {CC[y - 4]},{_hex(_rel(r), 4)}"
        if z == 1:
            return (f"LD {pair(p)},{_hex(r.word(), 4)}" if q == 0
                    else f"ADD {index or 'HL'},{pair(p)}")
        if z == 2:
            # Branch rather than index a tuple: every element of a tuple is
            # evaluated, so building one here would call r.word() twice and
            # consume four operand bytes for a two-byte operand.
            if p == 0:
                return "LD (BC),A" if q == 0 else "LD A,(BC)"
            if p == 1:
                return "LD (DE),A" if q == 0 else "LD A,(DE)"
            operand = _hex(r.word(), 4)
            if p == 2:
                return (f"LD ({operand}),{index or 'HL'}" if q == 0
                        else f"LD {index or 'HL'},({operand})")
            return f"LD ({operand}),A" if q == 0 else f"LD A,({operand})"
        if z == 3:
            return f"{'INC' if q == 0 else 'DEC'} {pair(p)}"
        if z in (4, 5):
            return f"{'INC' if z == 4 else 'DEC'} {reg(y, disp_first=True)}"
        if z == 6:
            target = reg(y, disp_first=True)
            return f"LD {target},{_hex(r.byte())}"
        return ("RLCA", "RRCA", "RLA", "RRA", "DAA", "CPL", "SCF", "CCF")[y]

    if x == 1:
        if y == 6 and z == 6:
            return "HALT"
        # Only one operand may carry the displacement, and it is whichever is
        # (HL) - the other stays a plain register even under a prefix.
        if index and R[y] == "(HL)":
            disp = r.signed()
            return f"LD ({index}{disp:+d}),{R[z]}"
        if index and R[z] == "(HL)":
            disp = r.signed()
            return f"LD {R[y]},({index}{disp:+d})"
        return f"LD {_sub_index(R[y], index, None) if index else R[y]}," \
               f"{_sub_index(R[z], index, None) if index else R[z]}"

    if x == 2:
        return f"{ALU[y]}{reg(z, disp_first=True)}"

    if z == 0:
        return f"RET {CC[y]}"
    if z == 1:
        if q == 0:
            return f"POP {_sub_index(RP2[p], index, None) if index else RP2[p]}"
        return ("RET", "EXX", f"JP ({index or 'HL'})", f"LD SP,{index or 'HL'}")[p]
    if z == 2:
        return f"JP {CC[y]},{_hex(r.word(), 4)}"
    if z == 3:
        if y == 0:
            return f"JP {_hex(r.word(), 4)}"
        if y == 1:
            return _decode_cb(r, None, None)
        if y == 2:
            return f"OUT ({_hex(r.byte())}),A"
        if y == 3:
            return f"IN A,({_hex(r.byte())})"
        if y == 4:
            return f"EX (SP),{index or 'HL'}"
        return ("", "", "", "", "", "EX DE,HL", "DI", "EI")[y]
    if z == 4:
        return f"CALL {CC[y]},{_hex(r.word(), 4)}"
    if z == 5:
        if q == 0:
            return f"PUSH {_sub_index(RP2[p], index, None) if index else RP2[p]}"
        if p == 0:
            return f"CALL {_hex(r.word(), 4)}"
        return ""  # DD/ED/FD, handled by the caller
    if z == 6:
        return f"{ALU[y]}{_hex(r.byte())}"
    return f"RST {_hex(y * 8)}"


def _rel(r: _Reader) -> int:
    """A relative jump target, resolved to the absolute address it reaches."""
    disp = r.signed()
    return r.org + r.pos + disp


def decode(image: bytes, org: int, addr: int, adl: bool = False) -> Instruction:
    """Decode one instruction at ``addr``.

    Bytes that decode to nothing come back as a one-byte ``DB``, so a data
    region read as code produces a listing rather than an exception.
    """
    r = _Reader(image, org, addr, adl)
    try:
        op = r.byte()
        if op == 0xCB:
            text = _decode_cb(r, None, None)
        elif op == 0xED:
            text = _decode_ed(r)
        elif op in (0xDD, 0xFD):
            text = _decode_indexed(r, op)
        elif op == 0xF2 and adl:
            text = f"JP P,{_hex(r.word(), 4)}"
        else:
            text = _decode_base(r, op)
    except IndexError:
        text = ""

    if not text:
        raw = image[addr - org : addr - org + 1]
        return Instruction(addr, raw, f"DB {_hex(raw[0])}")
    return Instruction(addr, r.consumed(), text)


def disassemble(image: bytes, org: int, start: int, count: int,
                adl: bool = False) -> list[Instruction]:
    """Decode ``count`` instructions from ``start``."""
    out, addr = [], start
    for _ in range(count):
        if not org <= addr < org + len(image):
            break
        instruction = decode(image, org, addr, adl)
        out.append(instruction)
        addr += instruction.length
    return out


#: A hex operand in rendered text: four or six digits, then h.
_ADDRESS = re.compile(r"\b0?([0-9A-F]{4}|[0-9A-F]{6})h\b")


def annotate(text: str, labels: dict[int, str]) -> str:
    """Name any address operand that lands on a label.

    Without this a listing is a wall of `CALL 0446h`, and cross-referencing it
    means looking every target up by hand - which is the job this tool exists
    to remove.
    """
    def name(match: re.Match[str]) -> str:
        addr = int(match.group(1), 16)
        return f"{labels[addr]}" if addr in labels else match.group(0)

    return _ADDRESS.sub(name, text)


def format_listing(instructions: list[Instruction], labels: dict[int, str],
                   width: int = 3) -> str:
    """Render a listing, marking where each label lands.

    ``width`` is the address width in bytes, so an eZ80 build gets the six
    digits its 24-bit addresses need.
    """
    digits = 2 * width
    lines = []
    for ins in instructions:
        if ins.addr in labels:
            lines.append(f"{labels[ins.addr]}:")
        raw = " ".join(f"{b:02X}" for b in ins.raw)
        lines.append(f"  {ins.addr:0{digits}X}  {raw:<12}  "
                     f"{annotate(ins.text, labels)}")
    return "\n".join(lines)


def label_map(builder) -> dict[int, str]:
    """Invert a builder's label table into address -> name.

    Several labels can share an address - CHAT and CHAT_LOOP do - so they are
    joined rather than one silently winning.
    """
    by_addr: dict[int, list[str]] = {}
    for name, addr in builder.labels.items():
        by_addr.setdefault(addr, []).append(name)
    return {addr: " / ".join(sorted(names)) for addr, names in by_addr.items()}


TARGETS = {
    "cpm": ("buildz80com", False),
    "cpm-fast": ("buildfastz80com", False),
    "cpm-column": ("buildcolz80com", False),
    "zx": ("buildz80tap", False),
    "next": ("buildnext", False),
    "cpc": ("buildcpc", False),
    "ez80": ("buildez80", True),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", required=True, help="Model to build")
    parser.add_argument("--target", "-t", default="cpm", choices=sorted(TARGETS),
                        help="Which backend to disassemble (default: cpm)")
    parser.add_argument("--at", default="START",
                        help="Label or address to start at (default: START)")
    parser.add_argument("--count", "-n", type=int, default=32,
                        help="Instructions to show (default: 32)")
    parser.add_argument("--labels", action="store_true",
                        help="List every label with its address, and exit")
    args = parser.parse_args()

    import importlib

    module_name, adl = TARGETS[args.target]
    module = importlib.import_module(module_name)
    builder = module.build_autoreg(args.model, max_output_len=4)
    image = builder.build()

    if args.labels:
        digits = 2 * builder.addr_size
        for name, addr in sorted(builder.labels.items(), key=lambda kv: kv[1]):
            print(f"{addr:0{digits}X}  {name}")
        return 0

    if args.at in builder.labels:
        start = builder.labels[args.at]
    else:
        try:
            start = int(args.at, 0)
        except ValueError:
            print(f"no label {args.at!r}; --labels lists them", file=sys.stderr)
            return 1

    print(f"\n{args.target}, {args.model}, from {args.at}:\n")
    print(format_listing(
        disassemble(image, builder.org, start, args.count, adl),
        label_map(builder), width=builder.addr_size,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
