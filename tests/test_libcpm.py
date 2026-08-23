"""Tests for the shared CP/M target definition.

Three backends emit a .COM and they used to carry their own copy of the entry
code, the chat loop and the BDOS line buffer. These tests pin what libcpm emits
and, more importantly, that all three backends really do use it - a backend
that quietly kept its own copy is what this is here to catch.
"""

from __future__ import annotations

import pytest

import buildcolz80com
import buildfastz80com
import buildz80com
import libcpm
from libz80 import Z80Builder


@pytest.fixture(scope="module")
def entry() -> Z80Builder:
    """A builder holding just the CP/M entry code, with its labels resolved."""
    b = Z80Builder()
    libcpm.emit_entry(b)
    # The entry calls into the engine; stub the targets so fixups resolve.
    for name in ("TOKENIZE", "CLEAR_CTX", "GENERATE"):
        b.label(name)
        b.ret()
    libcpm.emit_crlf(b)
    libcpm.emit_chat_buffer(b)
    return b


def test_entry_defines_every_label_the_backends_call(entry):
    for name in ("START", "CHAT", "CHAT_LOOP", "CHAT_EXIT"):
        assert name in entry.labels


def test_start_is_the_first_byte_emitted(entry):
    """A .COM is entered at its origin, so START may not have anything above it."""
    assert entry.labels["START"] == entry.org


def test_chat_and_chat_loop_are_the_same_address(entry):
    """CHAT is the no-arguments entry; it falls straight into the loop."""
    assert entry.labels["CHAT"] == entry.labels["CHAT_LOOP"]


def test_both_paths_end_in_a_warm_boot(entry):
    """RST 0 returns to CP/M. Falling off the end instead would hang the machine."""
    image = entry.build()
    assert image[entry.labels["CHAT_EXIT"] - entry.org] == 0xC7  # RST 0


def test_crlf_is_a_dollar_terminated_newline(entry):
    """BDOS function 9 prints until it sees a '$'."""
    base = entry.labels["CRLF"] - entry.org
    assert entry.build()[base : base + 3] == bytes((13, 10, ord("$")))


def test_chat_buffer_is_in_bdos_function_10_format(entry):
    """Capacity byte, then a length byte BDOS writes, then the text."""
    labels, image = entry.labels, entry.build()
    assert image[labels["CHATBUF"] - entry.org] == libcpm.CHAT_BUFFER_SIZE
    assert image[labels["CHATLEN"] - entry.org] == 0
    assert labels["CHATLEN"] == labels["CHATBUF"] + 1
    assert labels["CHATDAT"] == labels["CHATLEN"] + 1
    assert len(image) - (labels["CHATDAT"] - entry.org) == libcpm.CHAT_BUFFER_SIZE


def test_the_line_buffer_cannot_overrun_the_command_tail():
    """TOKENIZE reads from 0080h, where CP/M allows 127 bytes of tail."""
    assert libcpm.CHAT_BUFFER_SIZE <= 0xFF - libcpm.CPM_CMDLINE


def _opcodes(builder: Z80Builder, length: int) -> bytes:
    """The first ``length`` bytes with every fixed-up address blanked.

    What survives is the instruction stream itself, which is comparable across
    builds that place the engine at different addresses.
    """
    code = bytearray(builder.build()[:length])
    for offset, _label, ftype, _addend in builder.fixups:
        width = builder.addr_size if ftype == "abs" else 1
        for k in range(width):
            if offset + k < length:
                code[offset + k] = 0
    return bytes(code)


@pytest.mark.parametrize(
    "module", [buildz80com, buildfastz80com, buildcolz80com],
    ids=["packed", "fast", "column"],
)
def test_every_cpm_backend_emits_the_shared_entry(module, tiny_model_path, entry):
    """The three .COM backends differ in their kernels, not their front end."""
    builder = module.build_autoreg(tiny_model_path, max_output_len=4)

    entry_len = entry.labels["TOKENIZE"] - entry.org  # everything before the stubs
    assert _opcodes(builder, entry_len) == _opcodes(entry, entry_len), (
        f"{module.__name__} does not emit libcpm's entry code"
    )
    for name in ("CHAT", "CHAT_LOOP", "CHAT_EXIT"):
        assert builder.labels[name] - builder.org == entry.labels[name] - entry.org


@pytest.mark.parametrize(
    "module", [buildz80com, buildfastz80com, buildcolz80com],
    ids=["packed", "fast", "column"],
)
def test_every_cpm_backend_emits_the_shared_chat_buffer(module, tiny_model_path):
    builder = module.build_autoreg(tiny_model_path, max_output_len=4)
    labels, image = builder.labels, builder.build()
    assert image[labels["CHATBUF"] - builder.org] == libcpm.CHAT_BUFFER_SIZE
    assert labels["CHATDAT"] == labels["CHATBUF"] + 2


def test_fits_in_tpa_rejects_an_image_that_reaches_the_bdos():
    b = Z80Builder(org=libcpm.TPA_TOP - libcpm.STACK_MARGIN)
    assert libcpm.fits_in_tpa(b)  # empty: ends exactly at the margin
    b.nop()
    assert not libcpm.fits_in_tpa(b)


def test_platform_reads_the_query_from_the_command_tail():
    """The length byte lives at 0080h and the text at 0081h."""
    b = Z80Builder()
    plat = libcpm.CPMPlatform()
    plat.load_query_length(b)
    plat.load_query_pointer(b)
    image = b.build()
    assert image[1] | (image[2] << 8) == libcpm.CPM_CMDLINE
    assert image[5] | (image[6] << 8) == libcpm.CPM_CMDLINE + 1


def test_the_emulator_and_the_code_generator_agree_on_the_bdos():
    """libhost hooks the address libcpm emits calls to; they must be the one value."""
    import libhost

    assert libhost.BDOS == libcpm.BDOS
    assert libhost.TPA == libcpm.TPA
    assert libhost.CPM_CMDLINE == libcpm.CPM_CMDLINE
