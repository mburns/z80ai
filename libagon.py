"""
The Agon Light target: MOS entry points and the program around the layers.

The counterpart to :mod:`libcpm`, :mod:`libzx` and :mod:`libcpc`, and the last
of the four to exist. :mod:`buildez80` had carried its own copy of the entry
code, the ``>`` prompt loop and the keyboard line editor since before those
modules did.

The split here is between CPU and machine. :mod:`libez80` is the eZ80 - the ADL
assembler, and the Agon memory map it is inseparable from in practice. This is
the machine: how a character reaches the screen, how a key is read, how a
program is entered and left.

MOS is reached through two restarts rather than a jumpblock or a BDOS call:
``RST 10h`` prints the character in A, and ``RST 08h`` dispatches on a function
number, also in A. Only three functions are used, all present in every MOS
version, and none of them leaves a handle to leak.
"""

from __future__ import annotations

from collections.abc import Callable

import libnn
from libez80 import EZ80Builder
from libz80 import Z80Builder

# MOS entry points.
MOS_API = 0x08  # RST 08h, function number in A
MOS_OUTCHAR = 0x10  # RST 10h, character in A

# MOS API function numbers.
MOS_GETKEY = 0x00
# HL=filename, DE=load address, BC=max size -> A=status. The only SD call this
# makes: one call, no handle to leak, and it exists in every MOS version, which
# matters because MOS itself cannot be exercised in CI. See tools/mostest.py.
MOS_LOAD = 0x01

# Character codes.
AGON_CR = 13
AGON_LF = 10
AGON_BACKSPACE = 8
AGON_DELETE = 127
AGON_SPACE = 32

#: Longest query the input line will accept. Longer than the Z80 targets allow,
#: because nothing here is competing with the model for 64KB.
MAX_INPUT_LEN = 120

#: Routines whose address a standalone build prints, for cross-referencing a
#: disassembly. Which of these exist depends on the kernel.
KEY_LABELS = ('START', 'GENERATE', 'LAYER', 'LAYER1', 'ARGMAX', 'TOKENIZE',
              'NEUREND', 'BIASES', 'WTS1')


class AgonPlatform(libnn.Platform):
    """Agon: characters go through RST 10h, the query from the line editor.

    The eZ80 backend went without one of these for a long time, which is part
    of why so little of libnn looked reusable from it: every emitter that takes
    a Platform was unreachable, including the ones whose bodies are entirely
    machine-independent.

    ``activation_size`` is the real difference. Three bytes rather than two is
    what stops the buffer-touching emitters being shared; everything that only
    reads RESULT or CTXCHARS does not care.
    """

    name = "Agon Light"
    buffer = "INBUF"
    activation_size = 3

    def print_char(self, b: Z80Builder) -> None:
        b.rst(MOS_OUTCHAR)

    def load_query_length(self, b: Z80Builder) -> None:
        b.ld_a_mem_label('INPLEN')

    def load_query_pointer(self, b: Z80Builder) -> None:
        b.ld_de_label('INPBUF')


def emit_entry(b: EZ80Builder, answer: Callable[[EZ80Builder], None],
               phrase_bytes: int | None = None) -> None:
    """Emit START and the chat loop.

    There is no command tail on an Agon, so as on the Spectrum there is only
    the interactive path. A MOS program is entered by a call and leaves by a
    plain RET.

    Args:
        b: The builder.
        answer: Emits whatever turns a tokenized query into output - the
            character decoder's CLEAR_CTX and GENERATE, or the phrasebook's
            CLASSIFY. The rest of the loop is the same either way.
        phrase_bytes: Size of the reply file to load at startup, for a
            phrasebook build. ``None`` means the replies are in the image.
    """
    b.label('START')
    b.jp('CHAT_LOOP' if phrase_bytes is None else 'LOAD_PHRASES')

    if phrase_bytes is not None:
        emit_load_phrases(b, phrase_bytes)

    b.label('CHAT_LOOP')
    b.call('PRNL')
    b.ld_a_n(ord('>'))
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(ord(' '))
    b.rst(MOS_OUTCHAR)

    b.call('READ_INPUT')

    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jr_z('CHAT_LOOP')

    b.ld_a_mem_label('INPBUF')
    b.cp_n(ord('!'))
    b.jp_z('CHAT_EXIT')

    b.call('TOKENIZE')
    answer(b)
    b.jp('CHAT_LOOP')

    b.label('CHAT_EXIT')
    b.call('PRNL')
    b.ret()  # back to MOS


def emit_load_phrases(b: EZ80Builder, size: int) -> None:
    """Emit LOAD_PHRASES: read the reply file off the SD card.

    The replies live on the card, which is the entire point: the model picks an
    index, so the text costs it nothing and is free to be sentences. mos_load
    is the only firmware call here beyond printing and reading keys - one call,
    no handle to leak, present in every MOS version. BC is the buffer size, so
    MOS refuses an oversized file rather than writing past the end of it.
    """
    b.label('LOAD_PHRASES')
    b.ld_hl_label('PHRNAME')
    b.ld_de_label('PHRBUF')
    b.ld_bc_nn(size)
    b.ld_a_n(MOS_LOAD)
    b.rst(MOS_API)
    b.or_a()
    b.jp_z('CHAT_LOOP')

    # Nonzero status: say so and stop, rather than jumping into whatever the
    # buffer happens to contain and printing it as an answer.
    b.ld_hl_label('PHRERR')
    b.label('PE_LOOP')
    b.ld_a_hl()
    b.or_a()
    b.ret_z()
    b.rst(MOS_OUTCHAR)
    b.inc_hl()
    b.jr('PE_LOOP')


def emit_newline(b: EZ80Builder) -> None:
    """Emit PRNL: MOS does not turn one of these into the other."""
    b.label('PRNL')
    b.ld_a_n(AGON_CR)
    b.rst(MOS_OUTCHAR)
    b.ld_a_n(AGON_LF)
    b.rst(MOS_OUTCHAR)
    b.ret()


def emit_read_input(b: EZ80Builder) -> None:
    """Emit READ_INPUT: a line editor over mos_getkey."""
    b.label('READ_INPUT')
    b.xor_a()
    b.ld_mem_label_a('INPLEN')

    b.label('RI_LOOP')
    b.ld_a_n(MOS_GETKEY)
    b.rst(MOS_API)
    b.or_a()
    b.jr_z('RI_LOOP')  # no key ready
    b.cp_n(AGON_CR)
    b.jr_z('RI_DONE')
    b.cp_n(AGON_BACKSPACE)
    b.jr_z('RI_DELETE')
    b.cp_n(AGON_DELETE)
    b.jr_z('RI_DELETE')
    b.cp_n(AGON_SPACE)
    b.jr_c('RI_LOOP')  # ignore other control codes

    # Buffer full? Keep the character in C so the compare's flags survive.
    b.ld_c_a()
    b.ld_a_mem_label('INPLEN')
    b.cp_n(MAX_INPUT_LEN)
    b.jr_nc('RI_LOOP')

    # INPBUF[INPLEN++] = C, then echo it. A still holds INPLEN.
    b.ld_hl_label('INPBUF')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.add_hl_de()
    b.ld_hl_c()
    b.inc_a()
    b.ld_mem_label_a('INPLEN')
    b.ld_a_c()
    b.rst(MOS_OUTCHAR)
    b.jr('RI_LOOP')

    b.label('RI_DELETE')
    b.ld_a_mem_label('INPLEN')
    b.or_a()
    b.jr_z('RI_LOOP')
    b.dec_a()
    b.ld_mem_label_a('INPLEN')
    for code in (AGON_BACKSPACE, AGON_SPACE, AGON_BACKSPACE):
        b.ld_a_n(code)
        b.rst(MOS_OUTCHAR)
    b.jr('RI_LOOP')

    b.label('RI_DONE')
    b.call('PRNL')
    b.ret()


# There is deliberately no emit_input_buffer here, unlike the Z80 targets.
# buildez80 puts INPLEN in with the rest of its single-byte scratch and INPBUF
# on its own, which packs better than a length byte followed by the text; the
# data layout is the backend's business, not the machine's.
