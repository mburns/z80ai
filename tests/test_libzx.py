"""Tests for the shared ZX Spectrum target definition.

The .TAP container itself is covered in test_builders.py through
buildz80tap's re-exports; what is checked here is the rest of the target -
the memory map, the entry code, and that libhost hooks the same ROM addresses
the generated code calls.
"""

from __future__ import annotations

import pytest

import buildz80tap
import libzx
from libz80 import Z80Builder


def test_org_clears_the_screen_the_printer_buffer_and_the_system_variables():
    """5CCBh is the first byte above the 48K system variables."""
    assert libzx.ORG_ADDR >= 0x5CCB
    assert libzx.ZX_RAM_TOP == 0x10000


def test_build_tap_is_the_header_and_data_blocks_concatenated():
    image = bytes(range(32))
    tap = libzx.build_tap(image, 0x6000, "CHAT")
    assert tap == (libzx.build_tap_header("CHAT", 0x6000, len(image))
                   + libzx.build_tap_data(image))


def test_tap_header_records_the_load_address_and_length():
    block = libzx.build_tap_header("CHAT", libzx.ORG_ADDR, 1234)
    assert block[3] == libzx.TAP_TYPE_CODE
    assert block[2] == libzx.TAP_FLAG_HEADER
    assert block[14] | (block[15] << 8) == 1234
    assert block[16] | (block[17] << 8) == libzx.ORG_ADDR


def test_check_fits_in_ram_accepts_an_image_ending_on_the_last_byte():
    """FFFFh is usable; only FFFFh+1 is not."""
    libzx.check_fits_in_ram(libzx.ORG_ADDR, libzx.ZX_RAM_TOP - libzx.ORG_ADDR)


def test_check_fits_in_ram_reports_the_overrun_and_the_headroom():
    org = 0x8000
    with pytest.raises(ValueError) as exc:
        libzx.check_fits_in_ram(org, 0x9000)
    message = str(exc.value)
    assert "past the top of RAM" in message
    assert f"{0x10000 - org:,}" in message  # what would have fit


@pytest.fixture(scope="module")
def entry() -> Z80Builder:
    b = Z80Builder(org=libzx.ORG_ADDR)
    libzx.emit_entry(b)
    libzx.emit_read_input(b)
    for name in ("TOKENIZE", "CLEAR_CTX", "GENERATE"):
        b.label(name)
        b.ret()
    libzx.emit_input_buffer(b)
    return b


def test_entry_opens_the_upper_screen_before_printing(entry):
    """Printing without opening channel 2 lands in the lower editing area."""
    image = entry.build()
    start = entry.labels["START"] - entry.org
    assert image[start] == 0xF3  # DI
    assert image[start + 1 : start + 3] == bytes((0x3E, libzx.ZX_UPPER_SCREEN))
    assert image[start + 4] | (image[start + 5] << 8) == libzx.ZX_CHAN_OPEN


def test_entry_returns_to_basic_rather_than_warm_booting(entry):
    """RANDOMIZE USR expects a RET; there is no CP/M to go back to."""
    image = entry.build()
    exit_at = entry.labels["CHAT_EXIT"] - entry.org
    assert image[exit_at + 3] == 0xC9  # LD A,13 / RST 10h / RET


def test_input_buffer_is_a_length_byte_followed_by_the_text(entry):
    labels, image = entry.labels, entry.build()
    assert labels["INPBUF"] == labels["INPLEN"] + 1
    assert image[labels["INPLEN"] - entry.org] == 0
    assert len(image) - (labels["INPBUF"] - entry.org) == libzx.MAX_INPUT_LEN


def test_read_input_defines_the_labels_the_entry_jumps_to(entry):
    for name in ("READ_INPUT", "RI_LOOP", "RI_DELETE", "RI_DONE"):
        assert name in entry.labels


def test_the_tap_build_uses_the_shared_platform(tiny_model_path):
    builder = buildz80tap.build_autoreg(tiny_model_path, max_output_len=4)
    for name in ("START", "CHAT_LOOP", "CHAT_EXIT", "READ_INPUT", "INPBUF"):
        assert name in builder.labels
    assert builder.org == libzx.ORG_ADDR


def test_the_emulator_hooks_the_rom_addresses_the_build_calls():
    """A build calling 0DAFh while the host stubs something else would hang."""
    import libhost

    assert libhost.ZX_PRINT_A == libzx.ZX_PRINT_A
    assert libhost.ZX_CLS == libzx.ZX_CLS
    assert libhost.ZX_CHAN_OPEN == libzx.ZX_CHAN_OPEN
    assert libhost.ZX_KEY_INPUT == libzx.ZX_KEY_INPUT
    assert libhost.ZX_DEFAULT_ORG == libzx.ORG_ADDR


def test_buildz80tap_still_exports_the_container_helpers():
    """Callers written against the build script keep working."""
    assert buildz80tap.build_tap_header is libzx.build_tap_header
    assert buildz80tap.build_tap_data is libzx.build_tap_data
    assert buildz80tap.ORG_ADDR == libzx.ORG_ADDR
    assert buildz80tap.ZX_RAM_TOP == libzx.ZX_RAM_TOP
