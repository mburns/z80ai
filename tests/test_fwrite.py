"""Writing to the card, which nothing here has ever done.

Every file on an Agon card in this repository is read-only by construction:
`.IDX`, `.DAT` and `.GRF` are built on a host and copied across, and the
program seeks and reads. Issue #62's second scope needs the other direction -
a saved game is a flag vector, an object-location overlay and a few counters -
and named `mos_fwrite` as the one call missing.

It is two. A handle opened `FA_READ` cannot be written to, so `mos_fopen` has
to learn a mode nothing has ever passed it, and until now it raised on any.

These drive real eZ80 through `AgonHost` rather than calling the handlers in
Python, because the thing worth pinning is the register protocol: which
register carries the handle, which the count, and what comes back in DE.
"""

from __future__ import annotations

import pytest

from libez80 import AGON_LOAD_ADDR, EZ80Builder, agon_header
from libhost import (
    FA_CREATE_ALWAYS,
    FA_READ,
    FA_WRITE,
    MOS_FCLOSE,
    MOS_FOPEN,
    MOS_FREAD,
    MOS_FWRITE,
    MOS_RST_API,
    AgonHost,
    Z80Error,
)

SCRATCH = AGON_LOAD_ADDR + 0x8000


def program(build) -> bytes:
    """Wrap `build` in the entry and exit an Agon binary needs."""
    b = EZ80Builder(org=AGON_LOAD_ADDR)
    agon_header(b, "START")
    b.label("START")
    build(b)
    b.ld_hl_nn(0)
    b.ret()
    return b.build()


def saver(name: str, payload: bytes, mode: int = FA_WRITE | FA_CREATE_ALWAYS):
    """Open `name`, write `payload` from scratch RAM, close."""
    def build(b: EZ80Builder) -> None:
        for i, byte in enumerate(payload):
            b.ld_a_n(byte)
            b.ld_hl_nn(SCRATCH + i)
            b.ld_hl_a()

        b.ld_hl_label("NAME")
        b.ld_c_n(mode)
        b.ld_a_n(MOS_FOPEN)
        b.rst(MOS_RST_API)
        b.ld_mem_label_a("HANDLE")

        b.ld_a_mem_label("HANDLE")
        b.ld_c_a()
        b.ld_hl_nn(SCRATCH)
        b.ld_de_nn(len(payload))
        b.ld_a_n(MOS_FWRITE)
        b.rst(MOS_RST_API)

        b.ld_a_mem_label("HANDLE")
        b.ld_c_a()
        b.ld_a_n(MOS_FCLOSE)
        b.rst(MOS_RST_API)
        b.jr("DONE")

        b.label("NAME")
        b.ascii(name)
        b.db(0)
        b.label("HANDLE")
        b.db(0)
        b.label("DONE")
    return build


# --- the round trip -----------------------------------------------------------


def test_a_write_reaches_the_card():
    host = AgonHost(stdin=[], files={})
    host.run(program(saver("SAVE.BIN", b"OK")), max_cycles=1_000_000)
    assert host.files["SAVE.BIN"] == b"OK"


def test_a_created_file_did_not_have_to_exist():
    """`FA_CREATE_ALWAYS` is how a first save works, and every other file on
    this card was put there by a host."""
    host = AgonHost(stdin=[], files={})
    assert "SAVE.BIN" not in host.files
    host.run(program(saver("SAVE.BIN", b"\x01\x02\x03")), max_cycles=1_000_000)
    assert host.files["SAVE.BIN"] == b"\x01\x02\x03"


def test_writing_reports_how_many_bytes_went():
    """DE comes back as the count, the same protocol `mos_fread` uses - so a
    short write is detectable rather than silent."""
    def build(b: EZ80Builder) -> None:
        saver("S.BIN", b"12345")(b)

    host = AgonHost(stdin=[], files={})
    host.run(program(build), max_cycles=1_000_000)
    assert len(host.files["S.BIN"]) == 5


def test_what_was_written_reads_back():
    """The property a save file needs and the only one worth calling a round
    trip: bytes out of RAM, onto the card, and back into RAM somewhere else."""
    def build(b: EZ80Builder) -> None:
        saver("S.BIN", b"RESTORED")(b)
        b.ld_hl_label("NAME2")
        b.ld_c_n(FA_READ)
        b.ld_a_n(MOS_FOPEN)
        b.rst(MOS_RST_API)
        b.ld_c_a()
        b.ld_hl_nn(SCRATCH + 0x100)
        b.ld_de_nn(8)
        b.ld_a_n(MOS_FREAD)
        b.rst(MOS_RST_API)
        b.jr("DONE2")
        b.label("NAME2")
        b.ascii("S.BIN")
        b.db(0)
        b.label("DONE2")

    host = AgonHost(stdin=[], files={})
    host.run(program(build), max_cycles=1_000_000)
    assert host.files["S.BIN"] == b"RESTORED"
    assert bytes(host.cpu.mem[SCRATCH + 0x100:SCRATCH + 0x108]) == b"RESTORED"


# --- the failures that must be loud -------------------------------------------


def test_writing_to_a_read_handle_is_refused():
    """The mistake that would otherwise corrupt a card silently: every other
    file this program opens is one it must not write to."""
    host = AgonHost(stdin=[], files={"S.BIN": b"original"})
    with pytest.raises(Z80Error, match="opened for reading"):
        host.run(program(saver("S.BIN", b"clobber", mode=FA_READ)),
                 max_cycles=1_000_000)
    assert host.files["S.BIN"] == b"original"


def test_a_mode_the_host_does_not_emulate_raises():
    """0x40 is no FatFs mode at all. (0x30, `FA_OPEN_APPEND`, used to be the
    example here, until the archive's log needed it.)"""
    host = AgonHost(stdin=[], files={})
    with pytest.raises(Z80Error, match="not emulated"):
        host.run(program(saver("S.BIN", b"x", mode=0x40)),
                 max_cycles=1_000_000)


def test_appending_starts_at_the_end():
    """`FA_OPEN_APPEND` is what the archive's log is written with: open or
    create, and every write lands after what is there."""
    from libhost import FA_OPEN_APPEND

    host = AgonHost(stdin=[], files={"S.BIN": b"abc"})
    host.run(program(saver("S.BIN", b"de", mode=FA_WRITE | FA_OPEN_APPEND)),
             max_cycles=1_000_000)
    assert host.files["S.BIN"] == b"abcde"
    fresh = AgonHost(stdin=[], files={})
    fresh.run(program(saver("S.BIN", b"de", mode=FA_WRITE | FA_OPEN_APPEND)),
              max_cycles=1_000_000)
    assert fresh.files["S.BIN"] == b"de"


def test_a_write_from_outside_sram_is_refused():
    """`write_block` has always refused a read *into* a wrong address. A save
    from one is the same mistake in the other direction, and would put
    plausible bytes on the card."""
    def build(b: EZ80Builder) -> None:
        b.ld_hl_label("NAME")
        b.ld_c_n(FA_WRITE | FA_CREATE_ALWAYS)
        b.ld_a_n(MOS_FOPEN)
        b.rst(MOS_RST_API)
        b.ld_c_a()
        b.ld_hl_nn(0x1000)               # below Agon SRAM
        b.ld_de_nn(16)
        b.ld_a_n(MOS_FWRITE)
        b.rst(MOS_RST_API)
        b.jr("DONE")
        b.label("NAME")
        b.ascii("S.BIN")
        b.db(0)
        b.label("DONE")

    host = AgonHost(stdin=[], files={})
    with pytest.raises(Z80Error, match="leaves Agon SRAM"):
        host.run(program(build), max_cycles=1_000_000)
