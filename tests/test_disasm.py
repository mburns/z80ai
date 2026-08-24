"""Tests for the disassembler.

The decisive one is ``test_boundaries_agree_with_the_emulator``: the
disassembler and ``libz80emu`` are independent implementations of the same
instruction set, so running a real build and checking that they agree about
where every instruction ends is a much stronger claim than any table of
hand-written expectations. A disassembler that is quietly wrong about an
operand width would desynchronise immediately.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import disasm

import buildz80com
from libz80 import Z80Builder

#: Mnemonics that move PC somewhere other than the next instruction, so the
#: emulator's PC delta is not the instruction's length. The repeating block
#: moves are here because they rewind PC onto themselves until BC runs out,
#: which looks like an advance of zero.
BRANCHES = ("JP", "JR", "CALL", "RET", "RST", "DJNZ", "HALT",
            "LDIR", "LDDR", "CPIR", "CPDR", "INIR", "INDR", "OTIR", "OTDR")


def assemble(emit) -> tuple[bytes, Z80Builder]:
    b = Z80Builder(org=0x0100)
    emit(b)
    return b.build(), b


def text_of(emit) -> str:
    image, b = assemble(emit)
    return disasm.decode(image, b.org, b.org).text


# --- round trips against the assembler ---------------------------------------


@pytest.mark.parametrize("emit,expected", [
    (lambda b: b.nop(), "NOP"),
    (lambda b: b.ret(), "RET"),
    (lambda b: b.ret_z(), "RET Z"),
    (lambda b: b.ret_nz(), "RET NZ"),
    (lambda b: b.ld_a_n(0x2A), "LD A,2Ah"),
    (lambda b: b.ld_b_n(0xC0), "LD B,0C0h"),
    (lambda b: b.ld_hl_nn(0x1234), "LD HL,1234h"),
    (lambda b: b.ld_de_nn(0x8000), "LD DE,8000h"),
    (lambda b: b.ld_bc_nn(0x243B), "LD BC,243Bh"),
    (lambda b: b.ld_a_hl(), "LD A,(HL)"),
    (lambda b: b.ld_hl_a(), "LD (HL),A"),
    (lambda b: b.inc_hl(), "INC HL"),
    (lambda b: b.dec_a(), "DEC A"),
    (lambda b: b.or_a(), "OR A"),
    (lambda b: b.xor_a(), "XOR A"),
    (lambda b: b.cp_n(0x7F), "CP 7Fh"),
    (lambda b: b.add_hl_bc(), "ADD HL,BC"),
    (lambda b: b.sbc_hl_de(), "SBC HL,DE"),
    (lambda b: b.ex_de_hl(), "EX DE,HL"),
    (lambda b: b.ldir(), "LDIR"),
    (lambda b: b.di(), "DI"),
    (lambda b: b.ei(), "EI"),
    (lambda b: b.rst(0x10), "RST 10h"),
    (lambda b: b.push_hl(), "PUSH HL"),
    (lambda b: b.pop_iy(), "POP IY"),
    (lambda b: b.out_n_a(0xFE), "OUT (0FEh),A"),
    (lambda b: b.out_c_a(), "OUT (C),A"),
    (lambda b: b.rlca(), "RLCA"),
])
def test_assembled_instructions_read_back(emit, expected):
    assert text_of(emit) == expected


@pytest.mark.parametrize("emit,length", [
    (lambda b: b.nop(), 1),
    (lambda b: b.ld_a_n(1), 2),
    (lambda b: b.ld_hl_nn(0x1234), 3),
    (lambda b: b.ldir(), 2),
    (lambda b: b.out_c_a(), 2),
    (lambda b: b.pop_iy(), 2),
])
def test_lengths_match_what_the_assembler_emitted(emit, length):
    image, b = assemble(emit)
    assert len(image) == length
    assert disasm.decode(image, b.org, b.org).length == length


def test_a_16_bit_store_consumes_exactly_two_operand_bytes():
    """Regression: building the mnemonic from a tuple evaluated every branch,
    so r.word() ran twice and LD (nn),HL claimed to be five bytes long."""
    def emit(b: Z80Builder) -> None:
        b.ld_mem_label_hl("ACC")
        b.nop()
        b.label("ACC")
        b.dw(0)

    image, b = assemble(emit)
    first = disasm.decode(image, b.org, b.org)
    assert first.length == 3
    assert first.text == "LD (0104h),HL"
    assert disasm.decode(image, b.org, b.org + 3).text == "NOP"


# --- relative jumps ----------------------------------------------------------


def test_a_relative_jump_resolves_to_its_target_address():
    def emit(b: Z80Builder) -> None:
        b.label("TOP")
        b.nop()
        b.jr("TOP")

    image, b = assemble(emit)
    ins = disasm.decode(image, b.org, b.org + 1)
    assert ins.text == "JR 0100h"
    assert disasm.annotate(ins.text, disasm.label_map(b)) == "JR TOP"


def test_a_forward_conditional_jump_resolves_too():
    def emit(b: Z80Builder) -> None:
        b.jr_z("AHEAD")
        b.nop()
        b.label("AHEAD")
        b.ret()

    image, b = assemble(emit)
    ins = disasm.decode(image, b.org, b.org)
    assert ins.text == "JR Z,0103h"
    assert disasm.annotate(ins.text, disasm.label_map(b)) == "JR Z,AHEAD"


# --- data --------------------------------------------------------------------


def test_bytes_that_decode_to_nothing_become_db():
    """A weight blob read as code should produce a listing, not an exception."""
    image = bytes([0xED, 0xFF])
    ins = disasm.decode(image, 0x100, 0x100)
    assert ins.is_data
    assert ins.length == 1


def test_a_truncated_instruction_at_the_end_becomes_db():
    image = bytes([0x21])  # LD HL,nn with no operand
    ins = disasm.decode(image, 0x100, 0x100)
    assert ins.is_data
    assert ins.length == 1


def test_disassemble_stops_at_the_end_of_the_image():
    assert len(disasm.disassemble(bytes([0x00, 0x00]), 0x100, 0x100, 10)) == 2


# --- labels ------------------------------------------------------------------


def test_label_map_joins_labels_that_share_an_address():
    """CHAT and CHAT_LOOP are the same address; neither should win silently."""
    def emit(b: Z80Builder) -> None:
        b.label("CHAT")
        b.label("CHAT_LOOP")
        b.nop()

    _image, b = assemble(emit)
    assert disasm.label_map(b)[0x0100] == "CHAT / CHAT_LOOP"


def test_annotate_leaves_addresses_with_no_label_alone():
    assert disasm.annotate("CALL 1234h", {}) == "CALL 1234h"
    assert disasm.annotate("CALL 1234h", {0x1234: "PREQ"}) == "CALL PREQ"


def test_listing_marks_where_labels_land(tiny_model_path):
    builder = buildz80com.build_autoreg(tiny_model_path, max_output_len=4)
    listing = disasm.format_listing(
        disasm.disassemble(builder.build(), builder.org, builder.org, 12),
        disasm.label_map(builder), width=2,
    )
    assert listing.startswith("START:")
    assert "LD HL,0080h" in listing


# --- against the emulator ----------------------------------------------------


def _executed_addresses(image: bytes, org: int, limit: int = 60_000):
    """Step a real build, yielding (pc before, pc after) for each instruction."""
    from libhost import CPMExit, CPMHost

    host = CPMHost(cmdline="HELLO")
    cpu = host.cpu
    cpu.load(org, image)
    cpu.pc = org
    try:
        for _ in range(limit):
            before = cpu.pc
            if before in cpu.hooks:  # BDOS, not an instruction in our image
                break
            cpu.step()
            yield before, cpu.pc
            if cpu.halted:
                break
    except CPMExit:
        return


def test_boundaries_agree_with_the_emulator(tiny_model_path):
    """Two independent decoders, one instruction stream.

    For every instruction a real build actually executes, the disassembler must
    decode it, and - where the instruction does not branch - must agree with the
    emulator about how many bytes it occupies. A wrong operand width shows up
    here on the very next instruction.
    """
    builder = buildz80com.build_autoreg(tiny_model_path, max_output_len=4)
    image, org = builder.build(), builder.org

    checked = branches = 0
    for before, after in _executed_addresses(image, org):
        if not org <= before < org + len(image):
            continue
        ins = disasm.decode(image, org, before)
        assert not ins.is_data, (
            f"the emulator executed {before:04X} but the disassembler could "
            f"not decode it: {image[before - org:before - org + 4].hex()}"
        )
        if any(ins.text.startswith(b) for b in BRANCHES):
            branches += 1
            continue
        assert after == before + ins.length, (
            f"at {before:04X} ({ins.text}): emulator advanced "
            f"{after - before} bytes, disassembler says {ins.length}"
        )
        checked += 1

    assert checked > 2000, f"only {checked} instructions compared"
    assert branches > 100, f"only {branches} branches seen"


def test_a_24_bit_address_is_padded_so_it_can_be_annotated(tiny_model_path):
    """Regression: 040049h rendered as five digits, which annotate() could not
    match, so every eZ80 call target stayed a bare number."""
    import buildez80

    builder = buildez80.build_autoreg(tiny_model_path, max_output_len=4)
    listing = disasm.format_listing(
        disasm.disassemble(builder.build(), builder.org,
                           builder.labels["START"], 4, adl=True),
        disasm.label_map(builder), width=3,
    )
    assert "CHAT_LOOP" in listing.split("\n", 1)[1]


@pytest.mark.parametrize("target", sorted(disasm.TARGETS))
def test_every_backend_decodes_from_its_entry_point(target, tiny_model_path):
    """Whatever a backend emits at START, this should be able to read it."""
    import importlib

    module_name, adl = disasm.TARGETS[target]
    builder = importlib.import_module(module_name).build_autoreg(
        tiny_model_path, max_output_len=4)
    listing = disasm.disassemble(
        builder.build(), builder.org, builder.labels["START"], 12, adl)
    assert len(listing) == 12
    assert not any(ins.is_data for ins in listing), (
        f"{target}: undecodable bytes at its entry point"
    )
