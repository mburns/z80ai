"""
Printing and reading a line on an Agon, for any program that needs both.

Lifted out of `buildwikibin` unchanged when a second program wanted it. The
search card, the oracle and the turn loop all need the same four routines, and
the alternative was a second copy that would drift - `PRWRAP` in particular is
the sort of thing that gets fixed in one place and not the other.

    PRSTR       HL -> NUL-terminated string, printed as-is
    PRWRAP      the same, broken between words at WRAP_WIDTH
    PRNL        carriage return and line feed
    READ_INPUT  a line into INPBUF, with backspace, length in INPLEN

## What the caller owes it

`INPBUF` and `INPLEN` are labels the caller declares, because how much room a
program gives an input line is the program's business. `WRAPCOL` is one byte
`PRWRAP` keeps its column in and is likewise the caller's to reserve.

## Why nothing here emits a VDU sequence

Nothing in this repository ever has. Every character goes through `RST 10h` and
the terminal decides what to do with it, which was invisible while the longest
thing on a card was a 300-character lead and stopped being invisible when
`data/silo/authored/` put whole documents there.

`PRWRAP` is the smallest repair that helps: it decides where a line ends rather
than letting the column run out mid-word. A screen mode, a status line and
anything else that needs the terminal told something are still absent - see the
second scope of issue #62.
"""

from __future__ import annotations

from libagon import MOS_API, MOS_GETKEY, MOS_OUTCHAR
from libez80 import EZ80Builder

__all__ = ["MOS_API", "MOS_GETKEY", "MOS_OUTCHAR", "WRAP_WIDTH", "emit_console"]

#: Where `PRWRAP` breaks a line. The Agon's default mode is 80 columns and a
#: line printed to exactly 80 makes the terminal wrap it itself, which costs a
#: blank line; this leaves room and is still wider than any prose on a card
#: needs.
WRAP_WIDTH = 76


def emit_console(b: EZ80Builder, max_input_len: int) -> None:
    """Emit the four routines. `max_input_len` bounds what READ_INPUT takes."""
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

    # PRWRAP: PRSTR, but breaking between words instead of wherever the column
    # runs out.
    #
    # No lookahead buffer: whatever is being printed is already whole in RAM,
    # so the next word can be measured in place and the decision made before
    # the space in front of it is emitted. A word longer than a line is not
    # special-cased - it is measured up to the width, fails to fit whatever the
    # column, and starts a line of its own.
    b.label("PRWRAP")
    b.xor_a()
    b.ld_mem_label_a("WRAPCOL")

    b.label("PW_NEXT")
    b.ld_a_hl()
    b.or_a()
    b.ret_z()
    b.cp_n(10)
    b.jr_z("PW_BREAK")
    b.cp_n(32)
    b.jr_z("PW_SPACE")
    b.rst(MOS_OUTCHAR)               # an ordinary character
    b.inc_hl()
    b.ld_a_mem_label("WRAPCOL")
    b.inc_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    # A newline the author wrote: `data/silo/authored.py` keeps paragraph
    # breaks because they are the only formatting that survives to a screen
    # with no wrap.
    b.label("PW_BREAK")
    b.inc_hl()
    b.call("PRNL")
    b.xor_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    b.label("PW_SPACE")
    b.inc_hl()                       # step over the space
    b.push_hl()
    b.ld_c_n(0)
    b.label("PW_MEAS")
    b.ld_a_hl()
    b.or_a()
    b.jr_z("PW_MEASD")
    b.cp_n(32)
    b.jr_z("PW_MEASD")
    b.cp_n(10)
    b.jr_z("PW_MEASD")
    b.inc_hl()
    b.inc_c()
    b.ld_a_c()
    b.cp_n(WRAP_WIDTH)               # longer than a line: stop counting
    b.jr_nc("PW_MEASD")
    b.jr("PW_MEAS")

    b.label("PW_MEASD")
    b.pop_hl()
    b.ld_a_mem_label("WRAPCOL")
    b.or_a()
    b.jr_z("PW_NEXT")                # at the margin already: swallow the space
    b.add_a_c()
    b.inc_a()                        # and the space itself
    b.cp_n(WRAP_WIDTH + 1)
    b.jr_c("PW_FITS")
    b.call("PRNL")
    b.xor_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    b.label("PW_FITS")
    b.ld_a_n(32)
    b.rst(MOS_OUTCHAR)
    b.ld_a_mem_label("WRAPCOL")
    b.inc_a()
    b.ld_mem_label_a("WRAPCOL")
    b.jr("PW_NEXT")

    b.label("READ_INPUT")
    b.xor_a()
    b.ld_mem_label_a("INPLEN")

    b.label("RI_LOOP")
    b.ld_a_n(MOS_GETKEY)
    b.rst(MOS_API)
    b.or_a()
    b.jr_z("RI_LOOP")
    b.cp_n(13)
    b.jr_z("RI_DONE")
    b.cp_n(8)
    b.jr_z("RI_DEL")
    b.cp_n(127)
    b.jr_z("RI_DEL")
    b.cp_n(32)
    b.jr_c("RI_LOOP")
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.cp_n(max_input_len)
    b.jr_nc("RI_LOOP")
    b.ld_hl_label("INPBUF")
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_hl_c()
    b.ld_a_mem_label("INPLEN")
    b.inc_a()
    b.ld_mem_label_a("INPLEN")
    b.ld_a_c()
    b.rst(MOS_OUTCHAR)
    b.jr("RI_LOOP")

    b.label("RI_DEL")
    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("RI_LOOP")
    b.dec_a()
    b.ld_mem_label_a("INPLEN")
    for code in (8, 32, 8):
        b.ld_a_n(code)
        b.rst(MOS_OUTCHAR)
    b.jr("RI_LOOP")

    b.label("RI_DONE")
    b.call("PRNL")
    b.ret()
