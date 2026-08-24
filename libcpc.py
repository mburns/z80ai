"""
The Amstrad CPC target: firmware jumpblock, memory map, and the AMSDOS header.

The counterpart to :mod:`libcpm` and :mod:`libzx`.  A CPC is a Z80A with a
firmware ROM whose entry points are reached through a fixed jumpblock, so this
looks much like the Spectrum target with ``CALL &BB5A`` where that one has
``RST 10h``.

Two firmware facts shape the code:

- Firmware calls corrupt AF, BC, DE and HL.  That is survivable only because
  nothing in ``libnn``'s generation loop is live in a register across
  ``CALL PRINTCH`` - RESULT, GENCNT and CTXCHARS are all memory - so the I/O
  path may clobber whatever it likes.
- ``KM_WAIT_CHAR`` is fed by the interrupt-driven keyboard scan, so interrupts
  have to stay enabled.  The packed weight layout never disables them (only the
  index-list layout does, and that one does not fit a CPC anyway), so the
  engine is usable here unchanged.
"""

from __future__ import annotations

import libnn
from libz80 import Z80Builder

# Firmware jumpblock entry points.
TXT_OUTPUT = 0xBB5A  # print the character in A
KM_WAIT_CHAR = 0xBB06  # wait for a key, return it in A
SCR_SET_MODE = 0xBC0E  # set screen mode in A, and clear the screen

#: Screen mode 2 is 80 columns, which suits a chat prompt better than the
#: 40-column mode 1 a CPC boots into.
SCREEN_MODE = 2

# Character codes.
CPC_CR = 13
CPC_LF = 10
CPC_DELETE = 127  # the DEL key, which is what CPC keyboards send for backspace
CPC_BACKSPACE = 8  # cursor left, used to rub a character out
CPC_SPACE = 32

#: Longest query the input line will accept.
MAX_INPUT_LEN = 62

# Memory layout.
#
# 0000h-003Fh holds the restart vectors the firmware itself uses - RST 1 (LOW
# JUMP), RST 3 (FAR CALL) and the interrupt entry at 0038h - so 0040h is the
# first byte a program may have, and taking it is what makes a ~40KB model fit
# at all.
#
# The ceiling is HIMEM, which the disc ROM lowers to A67Bh when AMSDOS is
# active. That is the number to build against: a model that only fits on a
# tape-only machine would fail the moment anyone put it on a disc.
ORG_ADDR = 0x0040
CPC_HIMEM = 0xA67B

#: Routines whose address a standalone build prints, for cross-referencing a
#: disassembly.
KEY_LABELS = ("START", "GENERATE", "LAYER", "ARGMAX", "TOKENIZE",
              "UPDATE_CTX", "CHARTBL")

# AMSDOS binary header. The header is the first 128 bytes of the file as stored
# on disc, and is what lets `RUN"CHAT.BIN"` know where to load and where to
# start. A file whose checksum does not match is treated as headerless.
AMSDOS_HEADER_LEN = 128
AMSDOS_TYPE_BINARY = 2
#: The checksum covers bytes 0 to 66 inclusive and lands at 67.
AMSDOS_CHECKSUM_AT = 67
#: Longest filename an AMSDOS header holds, before the extension.
AMSDOS_NAME_LEN = 8


class CPCPlatform(libnn.Platform):
    """Amstrad CPC: characters go through TXT_OUTPUT, input via a keyboard loop."""

    name = "Amstrad CPC"
    buffer = "TOKBUF"

    def print_char(self, b: Z80Builder) -> None:
        b.call_addr(TXT_OUTPUT)

    def load_query_length(self, b: Z80Builder) -> None:
        b.ld_a_mem_label("INPLEN")

    def load_query_pointer(self, b: Z80Builder) -> None:
        b.ld_de_label("INPBUF")


def check_fits_in_ram(org: int, size: int) -> None:
    """Refuse an image that would assemble over the firmware's workspace.

    Raises:
        ValueError: If the image would not load at ``org``.
    """
    end = org + size
    if end > CPC_HIMEM:
        raise ValueError(
            f"image is {size:,} bytes and would run to {end:#06x} from "
            f"{org:#06x}, past HIMEM ({CPC_HIMEM:#06x}) by {end - CPC_HIMEM:,} "
            f"bytes. {CPC_HIMEM - org:,} bytes are available at {org:#06x}; "
            f"train a smaller model or free the disc ROM."
        )


def amsdos_header(filename: str, load: int, entry: int, length: int) -> bytes:
    """Build the 128-byte AMSDOS header for a binary file.

    Args:
        filename: Stored uppercased, truncated to 8 characters, extension BIN.
        load: Address AMSDOS loads the image at.
        entry: Address ``RUN"file"`` jumps to.
        length: Bytes of image following the header.
    """
    head = bytearray(AMSDOS_HEADER_LEN)
    head[0] = 0  # user number
    name = filename.upper()[:AMSDOS_NAME_LEN].ljust(AMSDOS_NAME_LEN)
    head[1:9] = name.encode("ascii")
    head[9:12] = b"BIN"
    head[18] = AMSDOS_TYPE_BINARY
    head[21] = load & 0xFF
    head[22] = (load >> 8) & 0xFF
    head[24] = length & 0xFF
    head[25] = (length >> 8) & 0xFF
    head[26] = entry & 0xFF
    head[27] = (entry >> 8) & 0xFF
    # The 24-bit length AMSDOS actually reads when loading.
    head[64] = length & 0xFF
    head[65] = (length >> 8) & 0xFF
    head[66] = (length >> 16) & 0xFF

    checksum = sum(head[:AMSDOS_CHECKSUM_AT])
    head[AMSDOS_CHECKSUM_AT] = checksum & 0xFF
    head[AMSDOS_CHECKSUM_AT + 1] = (checksum >> 8) & 0xFF
    return bytes(head)


def build_binary(image: bytes, org: int, filename: str = "CHAT") -> bytes:
    """Wrap an assembled image in the AMSDOS header `RUN"` reads."""
    return amsdos_header(filename, org, org, len(image)) + image


def emit_entry(b: Z80Builder) -> None:
    """Emit START and the chat loop.

    A CPC binary is entered from ``RUN"CHAT.BIN"`` with the firmware live and
    interrupts enabled, and returns to BASIC with a plain RET.  There is no
    command tail to check, so unlike CP/M there is only the interactive path.
    """
    b.label("START")
    # Mode 2 is 80 columns, and setting a mode clears the screen, so this is
    # both the layout choice and the CLS in one firmware call.
    b.ld_a_n(SCREEN_MODE)
    b.call_addr(SCR_SET_MODE)
    b.jp("CHAT")

    b.label("CHAT")
    b.label("CHAT_LOOP")
    b.call("PRNL")
    for ch in "> ":
        b.ld_a_n(ord(ch))
        b.call_addr(TXT_OUTPUT)

    b.call("READ_INPUT")

    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("CHAT_LOOP")  # empty line, prompt again

    b.ld_a_mem_label("INPBUF")
    b.cp_n(ord("!"))
    b.jp_z("CHAT_EXIT")

    b.call("TOKENIZE")
    b.call("CLEAR_CTX")
    b.call("GENERATE")
    b.jp("CHAT_LOOP")

    b.label("CHAT_EXIT")
    b.call("PRNL")
    b.ret()  # back to BASIC


def emit_newline(b: Z80Builder) -> None:
    """Emit PRNL: the CPC needs both a carriage return and a line feed."""
    b.label("PRNL")
    b.ld_a_n(CPC_CR)
    b.call_addr(TXT_OUTPUT)
    b.ld_a_n(CPC_LF)
    b.call_addr(TXT_OUTPUT)
    b.ret()


def emit_read_input(b: Z80Builder) -> None:
    """Emit READ_INPUT: a line editor over the firmware's key routine."""
    b.label("READ_INPUT")
    b.ld_hl_label("INPBUF")
    b.ld_b_n(MAX_INPUT_LEN)
    b.xor_a()
    b.ld_mem_label_a("INPLEN")

    b.label("RI_LOOP")
    b.push_bc()
    b.push_hl()
    b.call_addr(KM_WAIT_CHAR)  # corrupts everything except A's result
    b.pop_hl()
    b.pop_bc()

    b.cp_n(CPC_CR)
    b.jr_z("RI_DONE")
    b.cp_n(CPC_DELETE)
    b.jr_z("RI_DELETE")
    b.cp_n(CPC_SPACE)
    b.jr_c("RI_LOOP")  # ignore other control codes
    b.cp_n(CPC_DELETE)
    b.jr_nc("RI_LOOP")  # and anything above printable ASCII

    # Buffer full? Keep the character in C rather than on the stack: POP AF
    # would restore the flags from before the CP and the branch below would
    # test the CP 32 above instead. LD A,C preserves flags.
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.cp_b()
    b.ld_a_c()
    b.jr_nc("RI_LOOP")

    b.push_bc()
    b.push_hl()
    b.ld_hl_label("INPBUF")
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.ld_e_a()
    b.ld_d_n(0)
    b.add_hl_de()
    b.ld_a_c()
    b.ld_hl_a()

    b.ld_a_mem_label("INPLEN")
    b.inc_a()
    b.ld_mem_label_a("INPLEN")

    b.ld_a_c()
    b.call_addr(TXT_OUTPUT)  # echo
    b.pop_hl()
    b.pop_bc()
    b.jr("RI_LOOP")

    b.label("RI_DELETE")
    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("RI_LOOP")
    b.dec_a()
    b.ld_mem_label_a("INPLEN")
    b.push_bc()
    b.push_hl()
    for code in (CPC_BACKSPACE, CPC_SPACE, CPC_BACKSPACE):
        b.ld_a_n(code)
        b.call_addr(TXT_OUTPUT)
    b.pop_hl()
    b.pop_bc()
    b.jr("RI_LOOP")

    b.label("RI_DONE")
    b.call("PRNL")
    b.ret()


def emit_input_buffer(b: Z80Builder) -> None:
    """Emit the keyboard line buffer: a length byte followed by the text."""
    b.label("INPLEN")
    b.db(0)
    b.label("INPBUF")
    b.ds(MAX_INPUT_LEN)
