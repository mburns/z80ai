"""
The ZX Spectrum 48K target: ROM entry points, memory map, and the TAP container.

The counterpart to :mod:`libcpm`.  :mod:`buildz80tap` supplies the weight
layout; everything here is true of any Spectrum build, and :mod:`libhost` reads
the same entry points so the emulator cannot disagree with the code generator
about where the ROM routines live.
"""

from __future__ import annotations

import libnn
from libz80 import Z80Builder

# ROM entry points.
ZX_PRINT_A = 0x0010  # RST 10h - print the character in A
ZX_CLS = 0x0DAF  # clear screen
ZX_CHAN_OPEN = 0x1601  # open a stream channel
ZX_KEY_INPUT = 0x10A8  # wait for a key, return it in A

# Character codes.
ZX_ENTER = 13
ZX_DELETE = 12
ZX_BACKSPACE = 8
ZX_SPACE = 32

#: Channel 2 is the upper screen, which is where a program should print.
ZX_UPPER_SCREEN = 2

#: Longest query the input line will accept.
MAX_INPUT_LEN = 62

# Memory layout for ZX Spectrum 48K.
#
# RAM runs to FFFFh, so the load address bounds how large a model can be: at the
# old 8000h only 32,768 bytes were available, which both shipped examples
# exceed. 6000h is the lowest address that is clear of the screen (4000-5AFFh),
# the printer buffer (5B00-5BFFh) and the system variables (5C00-5CCAh), and
# leaves room below it for the BASIC loader once RAMTOP is moved down with
# CLEAR. That gives 40,960 bytes.
ORG_ADDR = 0x6000
ZX_RAM_TOP = 0x10000  # one past the last byte of RAM on a 48K machine

#: Routines whose address a standalone build prints, for cross-referencing a
#: disassembly.
KEY_LABELS = ("START", "GENERATE", "LAYER", "ARGMAX", "TOKENIZE",
              "UPDATE_CTX", "CHARTBL")

#: TAP file type 3: a CODE block, which is what a machine-code image is.
TAP_TYPE_CODE = 3
#: TAP block flags.
TAP_FLAG_HEADER = 0x00
TAP_FLAG_DATA = 0xFF
#: Longest filename a TAP header can hold; longer names are truncated.
TAP_NAME_LEN = 10


class ZXPlatform(libnn.Platform):
    """ZX Spectrum: characters go through RST 10h, input via a keyboard loop."""

    name = "ZX Spectrum"
    buffer = "TOKBUF"
    weight_layout = "rotated"

    def print_char(self, b: Z80Builder) -> None:
        b.rst(ZX_PRINT_A)

    def load_query_length(self, b: Z80Builder) -> None:
        b.ld_a_mem_label("INPLEN")

    def load_query_pointer(self, b: Z80Builder) -> None:
        b.ld_de_label("INPBUF")


def check_fits_in_ram(org: int, size: int) -> None:
    """Refuse an image that would assemble past the top of 48K RAM.

    A .TAP whose image runs past FFFFh cannot load on any Spectrum, so a build
    that would produce one fails here rather than shipping a tape that dies
    halfway through loading.

    Raises:
        ValueError: If the image would not load at ``org``.
    """
    end = org + size
    if end > ZX_RAM_TOP:
        raise ValueError(
            f"image is {size:,} bytes and would run to {end:#07x} from "
            f"{org:#06x}, past the top of RAM ({ZX_RAM_TOP - 1:#06x}) by "
            f"{end - ZX_RAM_TOP:,} bytes. Lower --org or train a smaller model: "
            f"{ZX_RAM_TOP - org:,} bytes are available at {org:#06x}."
        )


def build_tap_header(filename: str, start: int, length: int) -> bytes:
    """Build a TAP header block describing a CODE file.

    A TAP block is ``[length:2][flag:1][payload][checksum:1]``; the checksum is
    the XOR of the flag and payload bytes.
    """
    header = bytearray()
    header.append(TAP_TYPE_CODE)
    header.extend(filename[:TAP_NAME_LEN].ljust(TAP_NAME_LEN).encode("ascii"))
    header.append(length & 0xFF)
    header.append((length >> 8) & 0xFF)
    header.append(start & 0xFF)
    header.append((start >> 8) & 0xFF)
    header.extend((0, 0))  # unused for CODE

    checksum = 0
    for byte in header:
        checksum ^= byte  # the 00h flag byte XORs to nothing

    block = bytearray()
    block.append(19)  # flag + 17 header bytes + checksum
    block.append(0)
    block.append(TAP_FLAG_HEADER)
    block.extend(header)
    block.append(checksum)
    return bytes(block)


def build_tap_data(data: bytes) -> bytes:
    """Build a TAP data block wrapping ``data``."""
    checksum = TAP_FLAG_DATA  # seeded with the data block flag
    for byte in data:
        checksum ^= byte

    length = len(data) + 2  # flag + checksum
    block = bytearray()
    block.append(length & 0xFF)
    block.append((length >> 8) & 0xFF)
    block.append(TAP_FLAG_DATA)
    block.extend(data)
    block.append(checksum)
    return bytes(block)


def build_tap(image: bytes, org: int, filename: str = "CHAT") -> bytes:
    """Wrap an assembled image in the header and data blocks a Spectrum loads."""
    return build_tap_header(filename, org, len(image)) + build_tap_data(image)


def emit_entry(b: Z80Builder) -> None:
    """Emit START and the chat loop.

    There is no command tail on a Spectrum, so unlike the CP/M build there is
    only the interactive path: open the upper screen, clear it, then prompt.
    """
    b.label("START")
    b.di()
    b.ld_a_n(ZX_UPPER_SCREEN)
    b.call_addr(ZX_CHAN_OPEN)
    b.call_addr(ZX_CLS)
    b.ei()
    b.jp("CHAT")

    b.label("CHAT")
    b.label("CHAT_LOOP")
    b.ld_a_n(ZX_ENTER)
    b.rst(ZX_PRINT_A)
    for ch in "> ":
        b.ld_a_n(ord(ch))
        b.rst(ZX_PRINT_A)

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
    b.ld_a_n(ZX_ENTER)
    b.rst(ZX_PRINT_A)
    b.ret()  # back to BASIC


def emit_read_input(b: Z80Builder) -> None:
    """Emit READ_INPUT: a keyboard line editor over the ROM key routine."""
    b.label("READ_INPUT")
    b.ld_hl_label("INPBUF")
    b.ld_b_n(MAX_INPUT_LEN)
    b.xor_a()
    b.ld_mem_label_a("INPLEN")

    b.label("RI_LOOP")
    b.call_addr(ZX_KEY_INPUT)

    b.cp_n(ZX_ENTER)
    b.jr_z("RI_DONE")
    b.cp_n(ZX_DELETE)
    b.jr_z("RI_DELETE")
    b.cp_n(ZX_SPACE)
    b.jr_c("RI_LOOP")  # ignore other control codes

    # Buffer full? Stash the character in C rather than on the stack: POP AF
    # would restore the flags from before the CP and the branch below would
    # then test the CP 32 above instead. LD A,C preserves flags.
    b.ld_c_a()
    b.ld_a_mem_label("INPLEN")
    b.cp_b()
    b.ld_a_c()
    b.jr_nc("RI_LOOP")

    b.push_af()
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

    b.pop_hl()
    b.pop_af()
    b.rst(ZX_PRINT_A)  # echo
    b.jr("RI_LOOP")

    b.label("RI_DELETE")
    b.ld_a_mem_label("INPLEN")
    b.or_a()
    b.jr_z("RI_LOOP")
    b.dec_a()
    b.ld_mem_label_a("INPLEN")
    for code in (ZX_BACKSPACE, ZX_SPACE, ZX_BACKSPACE):
        b.ld_a_n(code)
        b.rst(ZX_PRINT_A)
    b.jr("RI_LOOP")

    b.label("RI_DONE")
    b.ld_a_n(ZX_ENTER)
    b.rst(ZX_PRINT_A)
    b.ret()


def emit_input_buffer(b: Z80Builder) -> None:
    """Emit the keyboard line buffer: a length byte followed by the text."""
    b.label("INPLEN")
    b.db(0)
    b.label("INPBUF")
    b.ds(MAX_INPUT_LEN)
