"""Tests for the ZX Spectrum Next target.

The Next build is the Spectrum build plus the clock register, so most of what
matters is covered by the ZX tests already. What is checked here is the part
that is actually different: that the clock is asked for, that it is asked for
in a way a 48K Spectrum ignores rather than crashes on, and that nothing else
about the image moved.
"""

from __future__ import annotations

import pytest

import buildnext
import buildz80tap
import libnext
import libzx
from libhost import run_next, run_zx
from libz80 import Z80Builder

#: LD BC,nn / LD A,n / OUT (C),A, twice: select the register, then write it.
SPEED_PROLOGUE_LEN = 14


def test_the_memory_map_is_the_spectrums():
    """A Next boots Spectrum-compatible, so nothing here should have moved."""
    assert libnext.ORG_ADDR == libzx.ORG_ADDR
    assert libnext.KEY_LABELS == libzx.KEY_LABELS


def test_the_register_ports_are_above_255():
    """Which is why the build needs OUT (C),A rather than OUT (n),A."""
    assert libnext.NEXT_REG_SELECT > 0xFF
    assert libnext.NEXT_REG_VALUE > 0xFF


def test_the_default_speed_is_the_fastest_a_next_offers():
    assert libnext.DEFAULT_SPEED == "28"
    assert max(libnext.SPEEDS.values()) == libnext.SPEEDS[libnext.DEFAULT_SPEED]
    assert libnext.SPEED_MHZ.keys() == libnext.SPEEDS.keys()


def test_an_unknown_speed_is_refused_rather_than_silently_ignored():
    b = Z80Builder()
    with pytest.raises(ValueError, match="unknown speed"):
        libnext.emit_set_speed(b, "56")


def test_set_speed_selects_the_clock_register_then_writes_it():
    b = Z80Builder(org=0x6000)
    libnext.emit_set_speed(b, "28")
    image = b.build()
    assert len(image) == SPEED_PROLOGUE_LEN
    # LD BC,243Bh / LD A,07h / OUT (C),A
    assert image[0] == 0x01
    assert image[1] | (image[2] << 8) == libnext.NEXT_REG_SELECT
    assert image[3:5] == bytes((0x3E, libnext.NEXT_REG_CPU_SPEED))
    assert image[5:7] == bytes((0xED, 0x79))
    # LD BC,253Bh / LD A,03h / OUT (C),A
    assert image[7] == 0x01
    assert image[8] | (image[9] << 8) == libnext.NEXT_REG_VALUE
    assert image[10:12] == bytes((0x3E, libnext.SPEEDS["28"]))
    assert image[12:14] == bytes((0xED, 0x79))


@pytest.mark.parametrize("speed", sorted(libnext.SPEEDS))
def test_every_offered_speed_assembles_and_is_requested(speed, tiny_model_path):
    builder = buildnext.build_autoreg(tiny_model_path, max_output_len=2,
                                      speed=speed)
    _out, host = run_next(builder.build(), stdin=["HI", "!"],
                          max_cycles=400_000_000)
    assert host.cpu_speed == speed


def test_the_clock_is_set_before_the_screen_is_touched(tiny_model_path):
    """So that even the CLS runs at the new clock."""
    builder = buildnext.build_autoreg(tiny_model_path, max_output_len=2)
    image = builder.build()
    start = builder.labels["START"] - builder.org
    assert image[start] == 0x01  # LD BC,nn - the prologue, not the DI
    assert image[start + SPEED_PROLOGUE_LEN] == 0xF3  # DI, the Spectrum entry


def test_the_image_is_the_spectrums_plus_exactly_the_prologue(tiny_model_path):
    """A fork would drift; this is meant to stay a thin wrapper."""
    nxt = buildnext.build_autoreg(tiny_model_path, max_output_len=4).build()
    zx = buildz80tap.build_autoreg(tiny_model_path, max_output_len=4).build()
    assert len(nxt) - len(zx) == SPEED_PROLOGUE_LEN


def test_a_48k_spectrum_ignores_the_clock_write_rather_than_crashing(
    tiny_model_path, tiny_model
):
    """ZXHost has no Next registers, exactly like real 48K hardware."""
    import libinfer

    builder = buildnext.build_autoreg(tiny_model_path, max_output_len=8)
    out, _host = run_zx(builder.build(), stdin=["HELLO", "!"],
                        max_cycles=400_000_000)
    assert libinfer.generate(tiny_model, "HELLO", 8) in out


def test_the_container_is_the_spectrums(tiny_model_path):
    """A Next loads .TAP as readily as a Spectrum, so there is no new format."""
    builder = buildnext.build_autoreg(tiny_model_path, max_output_len=4)
    tap = libzx.build_tap(builder.build(), builder.org)
    assert tap[:2] == bytes((19, 0))  # a 19-byte header block
    assert tap[16] | (tap[17] << 8) == libnext.ORG_ADDR


def test_the_host_records_the_register_the_build_writes():
    """The emulator has one clock, so the write is otherwise invisible."""
    from libhost import NextHost

    host = NextHost()
    host._io_write(libnext.NEXT_REG_SELECT, libnext.NEXT_REG_CPU_SPEED)
    host._io_write(libnext.NEXT_REG_VALUE, libnext.SPEEDS["14"])
    assert host.registers == {libnext.NEXT_REG_CPU_SPEED: libnext.SPEEDS["14"]}
    assert host.cpu_speed == "14"


def test_a_value_write_with_nothing_selected_is_ignored():
    from libhost import NextHost

    host = NextHost()
    host._io_write(libnext.NEXT_REG_VALUE, 3)
    assert host.registers == {}
    assert host.cpu_speed is None
