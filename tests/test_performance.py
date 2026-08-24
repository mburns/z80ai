"""Pin what one generated character costs, per target.

test_codegen_stability pins the bytes, so any change to the code generator
already has to be acknowledged. What it cannot tell you is which *direction*
the change went: a kernel that quietly got 20% slower produces a new hash and
nothing else.

These numbers are the record. They are deterministic - same model, same query,
same count every time - so they are pinned exactly rather than with a
tolerance, and a diff here says what happened:

    -    "col": 95_247,
    +    "col": 118_903,

is a regression, and the same line moving down is the speedup a commit claims.
The README and several docstrings quote ratios derived from these; when they
move, those move with them.
"""

from __future__ import annotations

import pytest

import bench

#: One forward pass over the shipped `guess` model, 256->256->192->128->11.
#: Instructions retired, which is the platform-neutral measure - an eZ80 retires
#: most instructions in a fraction of the cycles a Z80 needs, so comparing
#: T-states across the two would flatter the eZ80 for the wrong reason.
FORWARD_PASS = {
    # CP/M, and the same image on three other Z80 machines. The weight layout
    # is what differs; the engine is identical, which is why they tie.
    "com": 2_313_383,      # packed 2-bit weights, unpacked in the inner loop
    "fast": 256_473,       # index list per weight value: 9.0x
    "col": 95_247,         # index list per input column: 24.3x
    "tap": 2_313_383,      # ZX Spectrum, packed
    "next": 2_313_383,     # the same image; the Next's win is its clock
    "cpc": 2_313_383,      # Amstrad CPC, packed
    # eZ80. Unrolling is what the 24-bit address space buys.
    "ez80-compact": 923_194,   # one byte per weight, walked at runtime
    "ez80-row": 90_340,        # unrolled weight-major: 10.2x
    "ez80": 38_012,            # unrolled column-major: 24.3x
}

#: Z80 bus cycles for the Z80 targets. Worth pinning separately because the
#: instruction count can stay put while the cycle count moves - swapping an
#: instruction for a slower one of the same size does exactly that.
TSTATES = {
    "com": 20_698_833,
    "fast": 1_598_895,
    "col": 758_697,
    "tap": 20_698_833,
    "next": 20_698_833,
    "cpc": 20_698_833,
}


@pytest.fixture(scope="module")
def measured(guess_model_path):
    """One forward pass per target, measured once and shared."""
    return {t: bench.measure(t, guess_model_path, "HELLO") for t in FORWARD_PASS}


@pytest.mark.slow
@pytest.mark.parametrize("target", sorted(FORWARD_PASS))
def test_forward_pass_cost_is_unchanged(target, measured):
    got = measured[target]["instructions"]
    want = FORWARD_PASS[target]
    assert got == want, (
        f"{target}: {got:,} instructions, pinned at {want:,} "
        f"({(got - want) / want:+.1%}). If that was intended, update "
        f"FORWARD_PASS in this file and say why in the commit message."
    )


@pytest.mark.slow
@pytest.mark.parametrize("target", sorted(TSTATES))
def test_z80_cycle_cost_is_unchanged(target, measured):
    got = measured[target]["tstates"]
    want = TSTATES[target]
    assert got == want, (
        f"{target}: {got:,} T-states, pinned at {want:,} "
        f"({(got - want) / want:+.1%})"
    )


@pytest.mark.slow
def test_the_layouts_are_ordered_the_way_the_docs_claim(measured):
    """packed > fast > column, and the same on the eZ80. The whole point of
    build.py's fastest-that-fits search is that this ordering holds."""
    counts = {t: measured[t]["instructions"] for t in measured}
    assert counts["com"] > counts["fast"] > counts["col"]
    assert counts["ez80-compact"] > counts["ez80-row"] > counts["ez80"]


@pytest.mark.slow
def test_the_four_packed_z80_targets_retire_the_same_instructions(measured):
    """They share the engine and the weight layout, so a divergence means one
    of them grew a difference in the inner loop rather than at the edges."""
    packed = {t: measured[t]["instructions"] for t in ("com", "tap", "next", "cpc")}
    assert len(set(packed.values())) == 1, packed


@pytest.mark.slow
def test_the_next_is_eight_times_quicker_in_wall_clock_than_the_spectrum(measured):
    """Identical instruction counts; the whole difference is the clock."""
    def seconds(target: str) -> float:
        return measured[target]["tstates"] / bench.TARGETS[target].clock

    assert seconds("tap") / seconds("next") == pytest.approx(8.0, rel=1e-9)


def test_every_pinned_target_is_one_bench_knows():
    assert set(FORWARD_PASS) <= set(bench.TARGETS)
    assert set(TSTATES) <= set(FORWARD_PASS)


def test_no_z80_target_is_left_unpinned():
    """A target added to bench without a number here is one nothing watches."""
    z80 = {t for t, spec in bench.TARGETS.items() if not spec.is_ez80}
    assert z80 <= set(TSTATES)
    assert set(bench.TARGETS) == set(FORWARD_PASS)
