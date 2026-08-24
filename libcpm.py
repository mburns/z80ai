"""
The CP/M target: entry points, memory map, and the chat front-end.

Three backends emit a ``.COM`` - :mod:`buildz80com` (packed weights),
:mod:`buildfastz80com` (per-neuron index lists) and :mod:`buildcolz80com`
(per-column index lists).  They differ only in how a layer is computed; the
program around the layers - where the query comes from, how a character reaches
the screen, the ``>`` prompt loop, the BDOS line buffer - was byte-for-byte
identical in all three, so it lives here.

:mod:`libhost` reads the same entry-point constants from here, which is the
point: the emulator's idea of where BDOS sits and the code generator's cannot
drift apart if there is only one of them.
"""

from __future__ import annotations

import libnn
from libz80 import Z80Builder

#: BDOS entry point.
BDOS = 0x0005
#: Where CP/M loads a transient program, and so the origin of every ``.COM``.
TPA = 0x0100
#: CP/M leaves the command tail here: a length byte followed by the text.
CPM_CMDLINE = 0x0080

# BDOS function numbers.
BDOS_CONSOLE_OUT = 2
BDOS_PRINT_STRING = 9
BDOS_READ_LINE = 10

#: Size of the chat-mode input line (the BDOS function 10 buffer).
CHAT_BUFFER_SIZE = 62
#: Typing this as the first character of a line leaves chat mode.
CHAT_EXIT_CHAR = "!"

#: A stock CP/M 2.2 puts the BDOS at E400h.
TPA_TOP = 0xE400
#: Headroom left below the BDOS for the program's stack.
STACK_MARGIN = 0x0200

#: Routines whose address a standalone build prints, for cross-referencing a
#: disassembly. The union across the three backends: each emits some of these
#: and the rest are skipped.
KEY_LABELS = ("START", "GENERATE", "PREQ", "LAYER", "LAYER1", "ARGMAX",
              "TOKENIZE", "UPDATE_CTX", "CHARTBL")


class CPMPlatform(libnn.Platform):
    """CP/M: characters go through BDOS, the query arrives in the command tail."""

    name = "CP/M"
    buffer = "INBUF"

    def print_char(self, b: Z80Builder) -> None:
        b.ld_e_a()
        b.ld_c_n(BDOS_CONSOLE_OUT)
        b.call_addr(BDOS)

    def load_query_length(self, b: Z80Builder) -> None:
        b.ld_hl_nn(CPM_CMDLINE)
        b.ld_a_hl()

    def load_query_pointer(self, b: Z80Builder) -> None:
        b.ld_de_nn(CPM_CMDLINE + 1)


def fits_in_tpa(builder: Z80Builder) -> bool:
    """Would the assembled image load, with room left for its stack?"""
    return builder.org + len(builder.build()) + STACK_MARGIN <= TPA_TOP


def emit_entry(b: Z80Builder) -> None:
    """Emit START and the chat loop.

    A command tail means one query and a warm boot; no arguments means the
    interactive ``>`` prompt.  Both routes stage the query at
    :data:`CPM_CMDLINE` before calling TOKENIZE, so there is only one input
    path to get right.
    """
    b.label("START")
    b.ld_hl_nn(CPM_CMDLINE)
    b.ld_a_hl()
    b.or_a()
    b.jp_z("CHAT")

    b.call("TOKENIZE")
    b.call("CLEAR_CTX")
    b.call("GENERATE")
    b.rst(0)  # warm boot, back to CP/M

    b.label("CHAT")
    b.label("CHAT_LOOP")
    b.ld_de_label("CRLF")
    b.ld_c_n(BDOS_PRINT_STRING)
    b.call_addr(BDOS)
    for ch in "> ":
        b.ld_e_n(ord(ch))
        b.ld_c_n(BDOS_CONSOLE_OUT)
        b.call_addr(BDOS)

    b.ld_de_label("CHATBUF")
    b.ld_c_n(BDOS_READ_LINE)
    b.call_addr(BDOS)

    b.ld_de_label("CRLF")
    b.ld_c_n(BDOS_PRINT_STRING)
    b.call_addr(BDOS)

    b.ld_a_mem_label("CHATLEN")
    b.or_a()
    b.jr_z("CHAT_LOOP")  # empty line, prompt again

    b.ld_a_mem_label("CHATDAT")
    b.cp_n(ord(CHAT_EXIT_CHAR))
    b.jp_z("CHAT_EXIT")

    # Stage the line where TOKENIZE expects to find it.
    b.ld_a_mem_label("CHATLEN")
    b.ld_hl_nn(CPM_CMDLINE)
    b.ld_hl_a()
    b.ld_hl_label("CHATDAT")
    b.ld_de_nn(CPM_CMDLINE + 1)
    b.ld_c_a()
    b.ld_b_n(0)
    b.ldir()

    b.call("TOKENIZE")
    b.call("CLEAR_CTX")
    b.call("GENERATE")
    b.jr("CHAT_LOOP")

    b.label("CHAT_EXIT")
    b.rst(0)


def emit_crlf(b: Z80Builder) -> None:
    """Emit the newline string BDOS function 9 prints, ``$``-terminated."""
    b.label("CRLF")
    b.db(13, 10, ord("$"))


def emit_chat_buffer(b: Z80Builder) -> None:
    """Emit the BDOS function 10 line buffer: capacity, length, then the text."""
    b.label("CHATBUF")
    b.db(CHAT_BUFFER_SIZE)  # capacity, read by BDOS
    b.label("CHATLEN")
    b.db(0)  # length, written by BDOS
    b.label("CHATDAT")
    b.ds(CHAT_BUFFER_SIZE)
