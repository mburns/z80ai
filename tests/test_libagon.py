"""Tests for the Agon target.

buildez80 carried its own copy of the entry code, the prompt loop and the line
editor long after the Z80 backends stopped doing that. These cover the module
it moved into, and - more to the point - that the eZ80 build really uses it.
"""

from __future__ import annotations

import pytest

import buildez80
import libagon
from libez80 import EZ80Builder


def test_mos_is_reached_through_two_restarts():
    """Not a jumpblock and not a BDOS-style call: RST 10h prints, RST 08h
    dispatches on a function number."""
    assert libagon.MOS_OUTCHAR == 0x10
    assert libagon.MOS_API == 0x08
    assert libagon.MOS_GETKEY == 0x00
    assert libagon.MOS_LOAD == 0x01


def test_the_input_line_is_longer_than_the_z80_targets_allow():
    """Nothing here is competing with the model for 64KB."""
    import libcpc
    import libzx

    assert libagon.MAX_INPUT_LEN > libzx.MAX_INPUT_LEN
    assert libagon.MAX_INPUT_LEN > libcpc.MAX_INPUT_LEN


def _entry(phrase_bytes: int | None = None) -> EZ80Builder:
    b = EZ80Builder()
    calls: list[str] = []

    def answer(bb: EZ80Builder) -> None:
        calls.append("answer")
        bb.call("GENERATE")

    libagon.emit_entry(b, answer, phrase_bytes=phrase_bytes)
    libagon.emit_newline(b)
    libagon.emit_read_input(b)
    for name in ("TOKENIZE", "GENERATE", "CLASSIFY", "PHRNAME", "PHRBUF",
                 "PHRERR", "INPLEN", "INPBUF"):
        b.label(name)
        b.ret()
    b.entry_calls = calls
    return b


def test_entry_defines_the_labels_the_engine_jumps_to():
    b = _entry()
    for name in ("START", "CHAT_LOOP", "CHAT_EXIT", "PRNL", "READ_INPUT",
                 "RI_LOOP", "RI_DELETE", "RI_DONE"):
        assert name in b.labels


def test_entry_returns_to_mos_rather_than_resetting():
    """A MOS program is entered by a call; RST 0 would reboot the machine."""
    b = _entry()
    image = b.build()
    exit_at = b.labels["CHAT_EXIT"] - b.org
    assert image[exit_at] == 0xCD  # CALL PRNL
    assert image[exit_at + 4] == 0xC9  # RET, after a 4-byte ADL call


def test_newline_sends_both_a_carriage_return_and_a_line_feed():
    b = _entry()
    image = b.build()
    at = b.labels["PRNL"] - b.org
    assert image[at + 1] == libagon.AGON_CR
    assert image[at + 3 + 1] == libagon.AGON_LF


def test_the_answer_callback_is_emitted_once_per_query():
    b = _entry()
    assert b.entry_calls == ["answer"]


def test_a_phrasebook_build_loads_its_replies_before_prompting():
    b = _entry(phrase_bytes=1234)
    assert "LOAD_PHRASES" in b.labels
    assert b.labels["START"] < b.labels["LOAD_PHRASES"] < b.labels["CHAT_LOOP"]


def test_a_character_decoder_build_has_no_loader_at_all():
    assert "LOAD_PHRASES" not in _entry().labels


def test_the_load_size_is_the_size_of_the_reply_file():
    """BC bounds the write, so MOS refuses an oversized file rather than
    running past the end of the buffer."""
    b = _entry(phrase_bytes=0x4321)
    image = b.build()
    at = b.labels["LOAD_PHRASES"] - b.org
    # LD HL,nn / LD DE,nn / LD BC,nn - each 4 bytes in ADL mode.
    assert image[at + 8] == 0x01  # LD BC,nn
    assert image[at + 9] | (image[at + 10] << 8) == 0x4321


# --- the build ---------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["compact", "row", "column"])
def test_every_kernel_uses_the_shared_entry(kernel, tiny_model_path):
    builder = buildez80.build_autoreg(tiny_model_path, max_output_len=4,
                                      kernel=kernel)
    for name in ("START", "CHAT_LOOP", "CHAT_EXIT", "PRNL", "READ_INPUT"):
        assert name in builder.labels


def test_the_build_no_longer_restates_the_mos_constants():
    """They live in libagon now; a second copy is how they drift."""
    assert buildez80.MOS_OUTCHAR is libagon.MOS_OUTCHAR
    assert buildez80.MAX_INPUT_LEN is libagon.MAX_INPUT_LEN
    assert buildez80.KEY_LABELS is libagon.KEY_LABELS


def test_the_emulator_hooks_the_restarts_the_build_uses():
    """A build RSTing 10h while the host stubs something else would hang."""
    import libhost

    assert libhost.MOS_RST_OUTCHAR == libagon.MOS_OUTCHAR
    assert libhost.MOS_RST_API == libagon.MOS_API
    assert libhost.MOS_GETKEY == libagon.MOS_GETKEY
    assert libhost.MOS_LOAD == libagon.MOS_LOAD
