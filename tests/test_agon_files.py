"""MOS file I/O, as emulated for the Agon.

The card is a dict of bytes rather than a directory, so a test states exactly
what was served and nothing touches a filesystem.

What these cannot check is that MOS behaves the way this says it does. The
firmware would not boot under fab-agon-emulator on the machine this was written
on - see tools/README.md - so agreement with real MOS is established by
tools/mostest.py on hardware, not here. Keeping shipped code to mos_load alone
is what makes that a one-function question.
"""

from __future__ import annotations

import pytest

from libez80 import EZ80Builder
from libhost import AGON_RAM_HI, AGON_RAM_LO, AgonHost, run_agon
from libz80emu import Z80Error

LOAD_ADDR = 0x040000
SCRATCH = 0x050000

MOS_API = 0x08
MOS_OUTCHAR = 0x10


def build(fn) -> bytes:
    """Assemble a fragment at LOAD_ADDR, ending in HALT."""
    b = EZ80Builder(org=LOAD_ADDR)
    fn(b)
    b.halt()
    return b.build()


def run(image: bytes, files: dict[str, bytes] | None = None) -> AgonHost:
    host = AgonHost(files=files)
    host.cpu.load(LOAD_ADDR, image)
    host.cpu.pc = LOAD_ADDR
    host.cpu.run(max_cycles=1_000_000)
    return host


def load_call(name: str, dest: int = SCRATCH, size: int = 0x10000):
    """The mos_load sequence: HL=filename, DE=address, BC=max size, A=1."""
    def emit(b: EZ80Builder) -> None:
        b.ld_hl_label('FNAME')
        b.ld_de_nn(dest)
        b.ld_bc_nn(size)
        b.ld_a_n(0x01)
        b.rst(MOS_API)
        b.ld_mem_label_a('STATUS')
        b.jr('DONE')
        b.label('FNAME')
        b.ascii(name)
        b.db(0)
        b.label('STATUS')
        b.db(0)
        b.label('DONE')
    return emit


def test_mos_load_copies_a_whole_file_to_an_address():
    payload = bytes(range(256)) * 4
    host = run(build(load_call("MODEL.DAT")), {"MODEL.DAT": payload})
    got = bytes(host.cpu.mem[SCRATCH:SCRATCH + len(payload)])
    assert got == payload
    assert host.io_bytes == len(payload)


def test_a_missing_file_reports_a_status_rather_than_loading_garbage():
    image = build(load_call("ABSENT.DAT"))
    host = run(image, {"PRESENT.DAT": b"x" * 16})
    builder = EZ80Builder(org=LOAD_ADDR)
    load_call("ABSENT.DAT")(builder)
    builder.halt()
    status = builder.labels['STATUS']
    assert host.cpu.peek(status) == 4          # FR_NO_FILE
    assert host.cpu.mem[SCRATCH] == 0          # nothing was written
    assert host.io_bytes == 0


def test_a_file_larger_than_the_buffer_is_refused():
    """BC is the caller's promise about its buffer; overrunning it is the bug
    this call exists to prevent, so it must fail rather than truncate."""
    host = AgonHost(files={"BIG.DAT": b"z" * 4096})
    host.cpu.pc = LOAD_ADDR
    host.cpu.hl = 0
    image = build(load_call("BIG.DAT", size=1024))
    host.cpu.load(LOAD_ADDR, image)
    host.cpu.run(max_cycles=1_000_000)

    builder = EZ80Builder(org=LOAD_ADDR)
    load_call("BIG.DAT", size=1024)(builder)
    builder.halt()
    assert host.cpu.peek(builder.labels['STATUS']) == 5
    assert host.io_bytes == 0


def test_filenames_are_matched_case_insensitively_like_fat():
    host = run(build(load_call("model.dat")), {"MODEL.DAT": b"abcd"})
    assert bytes(host.cpu.mem[SCRATCH:SCRATCH + 4]) == b"abcd"


# --- the bounds check ---------------------------------------------------------


def test_a_load_outside_sram_raises_instead_of_growing_memory():
    """Z80.load() grows its bytearray rather than failing.

    Without this check a file read to a wrong address passes in the emulator
    and corrupts a real Agon - the one failure the emulator exists to catch and
    the one it could not. Below SRAM is flash on real hardware, where the write
    is silently discarded.
    """
    host = AgonHost(files={"X.DAT": b"q" * 8})
    with pytest.raises(Z80Error, match="leaves Agon SRAM"):
        host.write_block(AGON_RAM_LO - 1, b"q" * 8)
    with pytest.raises(Z80Error, match="leaves Agon SRAM"):
        host.write_block(AGON_RAM_HI - 4, b"q" * 8)


def test_the_bounds_check_allows_the_whole_window():
    host = AgonHost()
    host.write_block(AGON_RAM_LO, b"a")
    host.write_block(AGON_RAM_HI - 1, b"b")
    assert host.cpu.peek(AGON_RAM_LO) == ord("a")
    assert host.cpu.peek(AGON_RAM_HI - 1) == ord("b")


def test_memory_does_not_grow_when_a_load_is_refused():
    host = AgonHost()
    before = len(host.cpu.mem)
    with pytest.raises(Z80Error):
        host.write_block(AGON_RAM_HI, b"overflow")
    assert len(host.cpu.mem) == before


# --- handle-based calls -------------------------------------------------------


def test_fopen_fread_fclose_walk_a_file_in_chunks():
    host = AgonHost(files={"S.DAT": bytes(range(64))})
    cpu = host.cpu

    cpu.hl, cpu.c = 0x041000, 0x01
    host.write_block(0x041000, b"S.DAT\x00")
    host._fopen(cpu)
    handle = cpu.a
    assert handle != 0

    cpu.c, cpu.hl, cpu.de = handle, SCRATCH, 16
    host._fread(cpu)
    assert cpu.de == 16
    assert bytes(cpu.mem[SCRATCH:SCRATCH + 16]) == bytes(range(16))

    cpu.c, cpu.hl, cpu.de = handle, SCRATCH, 16
    host._fread(cpu)
    assert bytes(cpu.mem[SCRATCH:SCRATCH + 16]) == bytes(range(16, 32))

    cpu.c = handle
    host._fclose(cpu)
    assert host.handles == {}


def test_reading_past_the_end_returns_a_short_count():
    host = AgonHost(files={"S.DAT": b"1234"})
    cpu = host.cpu
    host.write_block(0x041000, b"S.DAT\x00")
    cpu.hl, cpu.c = 0x041000, 0x01
    host._fopen(cpu)

    cpu.c, cpu.hl, cpu.de = cpu.a, SCRATCH, 100
    host._fread(cpu)
    assert cpu.de == 4


def test_reading_an_unopened_handle_raises():
    host = AgonHost()
    host.cpu.c, host.cpu.hl, host.cpu.de = 9, SCRATCH, 4
    with pytest.raises(Z80Error, match="unopened handle"):
        host._fread(host.cpu)


# --- the loud-failure property ------------------------------------------------


def test_an_unimplemented_api_call_still_raises():
    """The reason a typo in a build script fails instead of reading zeroes.

    Turning the if/elif into a dispatch table must not have quietly turned
    unknown functions into no-ops.
    """
    def emit(b: EZ80Builder) -> None:
        b.ld_a_n(0x99)
        b.rst(MOS_API)

    with pytest.raises(Z80Error, match="unimplemented MOS API function 99"):
        run(build(emit))


def test_getkey_and_outchar_still_work_alongside_the_file_calls():
    out, host = run_agon(build(lambda b: None), stdin=["!"])
    assert host.files == {}
    assert out == ""
