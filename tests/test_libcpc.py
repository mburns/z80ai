"""Tests for the Amstrad CPC target.

The inference is covered by test_end_to_end; what is checked here is the
machine-specific half - the firmware addresses, the memory map that decides
whether a model fits at all, and the AMSDOS header, where a wrong checksum
means the file loads to the wrong place rather than failing loudly.
"""

from __future__ import annotations

import pytest

import buildcpc
import libcpc
import verify_artifacts
from libz80 import Z80Builder


def test_org_is_the_first_byte_above_the_restart_vectors():
    """0000h-003Fh is the firmware's own: RST 1, RST 3, and the interrupt entry.

    Taking 0040h is what makes a ~40KB model fit; starting even a page higher
    would cost more than the headroom the shipped models have.
    """
    assert libcpc.ORG_ADDR == 0x0040


def test_himem_is_the_ceiling_with_the_disc_rom_active():
    """Building against the tape-only ceiling would break the moment anyone
    put the file on a disc, which is the only way it ships."""
    assert libcpc.CPC_HIMEM == 0xA67B
    assert libcpc.CPC_HIMEM - libcpc.ORG_ADDR == 42_555


def test_firmware_addresses_are_the_documented_jumpblock_entries():
    assert libcpc.TXT_OUTPUT == 0xBB5A
    assert libcpc.KM_WAIT_CHAR == 0xBB06
    assert libcpc.SCR_SET_MODE == 0xBC0E


def test_check_fits_accepts_an_image_ending_on_the_last_free_byte():
    libcpc.check_fits_in_ram(libcpc.ORG_ADDR, libcpc.CPC_HIMEM - libcpc.ORG_ADDR)


def test_check_fits_reports_the_overrun_and_what_was_available():
    with pytest.raises(ValueError) as exc:
        libcpc.check_fits_in_ram(libcpc.ORG_ADDR, 0xC000)
    message = str(exc.value)
    assert "past HIMEM" in message
    assert f"{libcpc.CPC_HIMEM - libcpc.ORG_ADDR:,}" in message


# --- AMSDOS header -----------------------------------------------------------


def test_header_is_128_bytes_and_declares_load_entry_and_length():
    head = libcpc.amsdos_header("CHAT", 0x0040, 0x0040, 1234)
    assert len(head) == libcpc.AMSDOS_HEADER_LEN
    assert head[18] == libcpc.AMSDOS_TYPE_BINARY
    assert head[21] | (head[22] << 8) == 0x0040  # load
    assert head[26] | (head[27] << 8) == 0x0040  # entry
    assert head[64] | (head[65] << 8) | (head[66] << 16) == 1234


def test_checksum_covers_the_first_67_bytes():
    """AMSDOS treats a file whose checksum does not match as headerless, and
    would then load it at the wrong address rather than refusing."""
    head = libcpc.amsdos_header("CHAT", 0x0040, 0x0040, 1234)
    at = libcpc.AMSDOS_CHECKSUM_AT
    assert head[at] | (head[at + 1] << 8) == sum(head[:at])


def test_filename_is_uppercased_padded_and_truncated():
    head = libcpc.amsdos_header("verylongname", 0x40, 0x40, 1)
    assert head[1:9] == b"VERYLONG"
    assert head[9:12] == b"BIN"
    short = libcpc.amsdos_header("hi", 0x40, 0x40, 1)
    assert short[1:9] == b"HI      "


def test_build_binary_round_trips_through_the_verifier(tiny_model_path):
    """Write the container, then read it back the way verify_artifacts does."""
    builder = buildcpc.build_autoreg(tiny_model_path, max_output_len=4)
    image = builder.build()
    binary = libcpc.build_binary(image, builder.org, "TINY")

    got_image, got_org = verify_artifacts.parse_amsdos(binary)
    assert got_image == image
    assert got_org == builder.org


def test_a_corrupted_checksum_is_rejected(tiny_model_path):
    builder = buildcpc.build_autoreg(tiny_model_path, max_output_len=4)
    binary = bytearray(libcpc.build_binary(builder.build(), builder.org, "TINY"))
    binary[libcpc.AMSDOS_CHECKSUM_AT] ^= 0xFF
    with pytest.raises(verify_artifacts.VerificationError, match="checksum"):
        verify_artifacts.parse_amsdos(bytes(binary))


def test_a_truncated_file_is_rejected():
    with pytest.raises(verify_artifacts.VerificationError, match="shorter than"):
        verify_artifacts.parse_amsdos(b"\x00" * 64)


def test_a_declared_length_that_does_not_match_is_rejected(tiny_model_path):
    builder = buildcpc.build_autoreg(tiny_model_path, max_output_len=4)
    image = builder.build()
    head = bytearray(libcpc.amsdos_header("TINY", builder.org, builder.org,
                                          len(image) + 1))
    at = libcpc.AMSDOS_CHECKSUM_AT
    checksum = sum(head[:at])
    head[at], head[at + 1] = checksum & 0xFF, (checksum >> 8) & 0xFF
    with pytest.raises(verify_artifacts.VerificationError, match="declares"):
        verify_artifacts.parse_amsdos(bytes(head) + image)


# --- entry code --------------------------------------------------------------


@pytest.fixture(scope="module")
def entry() -> Z80Builder:
    b = Z80Builder(org=libcpc.ORG_ADDR)
    libcpc.emit_entry(b)
    libcpc.emit_newline(b)
    libcpc.emit_read_input(b)
    for name in ("TOKENIZE", "CLEAR_CTX", "GENERATE"):
        b.label(name)
        b.ret()
    libcpc.emit_input_buffer(b)
    return b


def test_entry_sets_a_screen_mode_which_also_clears_the_screen(entry):
    image = entry.build()
    start = entry.labels["START"] - entry.org
    assert image[start : start + 2] == bytes((0x3E, libcpc.SCREEN_MODE))  # LD A,n
    assert image[start + 2] == 0xCD  # CALL
    assert image[start + 3] | (image[start + 4] << 8) == libcpc.SCR_SET_MODE


def test_entry_returns_to_basic_rather_than_warm_booting(entry):
    """A CPC binary is CALLed by RUN"; RST 0 would reset the machine."""
    image = entry.build()
    exit_at = entry.labels["CHAT_EXIT"] - entry.org
    assert image[exit_at] == 0xCD  # CALL PRNL
    assert image[exit_at + 3] == 0xC9  # RET


def test_newline_sends_both_a_carriage_return_and_a_line_feed(entry):
    """TXT_OUTPUT does not translate one into the other."""
    image = entry.build()
    at = entry.labels["PRNL"] - entry.org
    assert image[at + 1] == libcpc.CPC_CR
    assert image[at + 6] == libcpc.CPC_LF


def test_read_input_defines_the_labels_the_entry_jumps_to(entry):
    for name in ("READ_INPUT", "RI_LOOP", "RI_DELETE", "RI_DONE"):
        assert name in entry.labels


def test_input_buffer_is_a_length_byte_followed_by_the_text(entry):
    labels, image = entry.labels, entry.build()
    assert labels["INPBUF"] == labels["INPLEN"] + 1
    assert image[labels["INPLEN"] - entry.org] == 0
    assert len(image) - (labels["INPBUF"] - entry.org) == libcpc.MAX_INPUT_LEN


# --- the build ---------------------------------------------------------------


def test_the_shipped_models_fit_below_himem(examples_dir):
    """The whole target rests on this: ~40KB of model into ~42.5KB of RAM."""
    import os

    for example in ("guess", "tinychat", "smalltalk"):
        path = os.path.join(examples_dir, example, "model.npz")
        if not os.path.exists(path):
            pytest.skip(f"{example} example model not present")
        builder = buildcpc.build_autoreg(path)
        end = builder.org + len(builder.build())
        assert end <= libcpc.CPC_HIMEM, (
            f"{example} runs to {end:#06x}, past HIMEM by "
            f"{end - libcpc.CPC_HIMEM:,} bytes"
        )


def test_the_builder_refuses_an_image_that_would_not_load(guess_model_path):
    """Assembling over the firmware must fail, not emit a broken binary."""
    with pytest.raises(ValueError, match="past HIMEM"):
        buildcpc.build_autoreg(guess_model_path, org=0x8000)


def test_expected_entry_points_exist(tiny_model_path):
    builder = buildcpc.build_autoreg(tiny_model_path, max_output_len=4)
    for name in ("START", "GENERATE", "ARGMAX", "TOKENIZE", "UPDATE_CTX",
                 "CHARTBL", "READ_INPUT"):
        assert name in builder.labels


def test_the_emulator_hooks_the_addresses_the_build_calls():
    """A build CALLing BB5Ah while the host stubs something else would hang."""
    import libhost

    assert libhost.CPC_TXT_OUTPUT == libcpc.TXT_OUTPUT
    assert libhost.CPC_KM_WAIT_CHAR == libcpc.KM_WAIT_CHAR
    assert libhost.CPC_SCR_SET_MODE == libcpc.SCR_SET_MODE
    assert libhost.CPC_ORG_ADDR == libcpc.ORG_ADDR
    assert libhost.CPC_HIMEM == libcpc.CPC_HIMEM
