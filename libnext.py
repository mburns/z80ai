"""
The ZX Spectrum Next target: the Spectrum, plus a clock the Z80 never had.

A Next is Spectrum-compatible, so everything in :mod:`libzx` applies unchanged -
the same ROM entry points, the same memory map, the same .TAP container. What
this module adds is the one thing worth having a separate target for: the Next
register that sets the CPU clock, which takes a generated character from 3.5MHz
to 28MHz for the cost of six bytes at startup.

The Next registers are reached through a select/value port pair rather than
mapped into memory, and they live above port 255, so ``OUT (C),A`` addresses
them from BC. On a machine that is not a Next nothing decodes those ports and
the writes are ignored, which is why this build still runs - at 3.5MHz - on a
plain 48K Spectrum.

What this does *not* yet do is use the Next's extra RAM. The column-major
weight layout needs ~48KB and the 48K map has ~41KB above the screen, so
reaching it means paging banks through C000h. That is worth doing and is
independent of the container - a Next can page from a .TAP as readily as from a
.NEX - but it wants testing on real hardware. See NEXT.md.
"""

from __future__ import annotations

import libzx
from libz80 import Z80Builder

#: Next register access: write the register number here...
NEXT_REG_SELECT = 0x243B
#: ...then its value here.
NEXT_REG_VALUE = 0x253B

#: Next register 07h selects the CPU clock.
NEXT_REG_CPU_SPEED = 0x07

#: The four clocks a Next offers, by the value register 07h takes.
SPEEDS = {"3.5": 0, "7": 1, "14": 2, "28": 3}
#: Megahertz for each, for reporting what a build will run at.
SPEED_MHZ = {"3.5": 3_500_000, "7": 7_000_000, "14": 14_000_000, "28": 28_000_000}
#: Default: the fastest a Next offers. There is no reason to ask for less.
DEFAULT_SPEED = "28"

#: The Next boots Spectrum-compatible, so the memory map is the Spectrum's.
ORG_ADDR = libzx.ORG_ADDR

KEY_LABELS = libzx.KEY_LABELS


class NextPlatform(libzx.ZXPlatform):
    """Same I/O as the Spectrum; only the clock differs."""

    name = "ZX Spectrum Next"


def emit_set_speed(b: Z80Builder, speed: str = DEFAULT_SPEED) -> None:
    """Emit the two port writes that set the CPU clock.

    Raises:
        ValueError: If ``speed`` is not one of :data:`SPEEDS`.
    """
    if speed not in SPEEDS:
        raise ValueError(
            f"unknown speed {speed!r}; choose from {sorted(SPEEDS)}"
        )
    b.ld_bc_nn(NEXT_REG_SELECT)
    b.ld_a_n(NEXT_REG_CPU_SPEED)
    b.out_c_a()
    b.ld_bc_nn(NEXT_REG_VALUE)
    b.ld_a_n(SPEEDS[speed])
    b.out_c_a()


def emit_entry(b: Z80Builder, speed: str = DEFAULT_SPEED) -> None:
    """Emit START and the chat loop, clocking the CPU up first.

    The speed is set before the screen is touched so that even the CLS runs at
    the new clock.
    """
    libzx.emit_entry(b, prologue=lambda bb: emit_set_speed(bb, speed))
