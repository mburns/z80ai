"""End-to-end equivalence: generated machine code vs. the NumPy reference.

These are the tests that actually matter.  Each build script is compiled for a
model, executed instruction-by-instruction in the emulator, and its console
output compared character-for-character against :func:`libinfer.generate`.
Any divergence -- packing, hashing, accumulator overflow, shift direction --
shows up here.
"""

from __future__ import annotations

import pytest

import buildcolz80com
import buildcpc
import buildfastz80com
import buildnext
import buildz80com
import buildz80tap
import libinfer
from libhost import run_cpc, run_cpm, run_next, run_zx

QUERIES = ["HELLO", "ARE YOU A ROBOT", "X", "WHAT IS THIS THING"]
GEN_LEN = 8  # keep emulated runs short; the code path is identical


def cpm_reply(image: bytes, query: str) -> str:
    """Run a .COM in single-query mode and return what it printed."""
    out, host = run_cpm(image, cmdline=query, max_cycles=400_000_000)
    assert host.finished, "program did not return to CP/M"
    return out


@pytest.fixture(scope="module")
def tiny_com(tiny_model_path):
    return buildz80com.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.fixture(scope="module")
def tiny_fast_com(tiny_model_path):
    return buildfastz80com.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.fixture(scope="module")
def tiny_col_com(tiny_model_path):
    return buildcolz80com.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.fixture(scope="module")
def tiny_tap(tiny_model_path):
    return buildz80tap.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.mark.parametrize("query", QUERIES)
def test_com_matches_reference(tiny_com, tiny_model, query):
    assert cpm_reply(tiny_com, query) == libinfer.generate(tiny_model, query, GEN_LEN)


@pytest.mark.parametrize("query", QUERIES)
def test_fast_com_matches_reference(tiny_fast_com, tiny_model, query):
    assert cpm_reply(tiny_fast_com, query) == libinfer.generate(tiny_model, query, GEN_LEN)


@pytest.mark.parametrize("query", QUERIES)
def test_col_com_matches_reference(tiny_col_com, tiny_model, query):
    assert cpm_reply(tiny_col_com, query) == libinfer.generate(tiny_model, query, GEN_LEN)


@pytest.fixture(scope="module")
def tiny_next(tiny_model_path):
    return buildnext.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.fixture(scope="module")
def tiny_cpc(tiny_model_path):
    return buildcpc.build_autoreg(tiny_model_path, max_output_len=GEN_LEN).build()


@pytest.mark.parametrize("query", QUERIES)
def test_tap_matches_reference(tiny_tap, tiny_model, query):
    image = tiny_tap
    out, _host = run_zx(image, stdin=[query, "!"], max_cycles=400_000_000)
    # The ZX build is chat-only; strip the prompt/echo chrome around the reply.
    expected = libinfer.generate(tiny_model, query, GEN_LEN)
    assert expected in out, f"{expected!r} not in {out!r}"


@pytest.mark.parametrize("query", QUERIES)
def test_next_matches_reference(tiny_next, tiny_model, query):
    out, host = run_next(tiny_next, stdin=[query, "!"], max_cycles=400_000_000)
    expected = libinfer.generate(tiny_model, query, GEN_LEN)
    assert expected in out, f"{expected!r} not in {out!r}"
    assert host.cpu_speed == "28"


@pytest.mark.parametrize("query", QUERIES)
def test_cpc_matches_reference(tiny_cpc, tiny_model, query):
    out, _host = run_cpc(tiny_cpc, stdin=[query, "!"], max_cycles=400_000_000)
    expected = libinfer.generate(tiny_model, query, GEN_LEN)
    assert expected in out, f"{expected!r} not in {out!r}"


def test_next_build_still_runs_on_a_plain_spectrum(tiny_next, tiny_tap, tiny_model):
    """Nothing on a 48K machine decodes the clock ports, so it should just run.

    Driven through ZXHost, which has no Next registers at all - the writes go
    nowhere, exactly as on real hardware.
    """
    out, _host = run_zx(tiny_next, stdin=["HELLO", "!"], max_cycles=400_000_000)
    assert libinfer.generate(tiny_model, "HELLO", GEN_LEN) in out
    # And it really is the Spectrum image plus the clock prologue, not a fork.
    assert len(tiny_next) - len(tiny_tap) == 14


def test_every_chat_target_agrees_with_the_cpm_build(
    tiny_com, tiny_tap, tiny_next, tiny_cpc, tiny_model
):
    """Four machines, four I/O paths, one answer.

    The kernels are shared, so what this pins is that no platform's entry code,
    tokenizer wiring or input buffer quietly changes what gets encoded.
    """
    for query in QUERIES:
        expected = cpm_reply(tiny_com, query)
        for image, run in (
            (tiny_tap, run_zx), (tiny_next, run_next), (tiny_cpc, run_cpc)
        ):
            out, _host = run(image, stdin=[query, "!"], max_cycles=400_000_000)
            assert expected in out, f"{run.__name__}: {expected!r} not in {out!r}"


def test_all_cpm_builds_agree(tiny_com, tiny_fast_com, tiny_col_com, tiny_model):
    """Three independently generated programs, one answer.

    This is the strongest signal available and it needs no reference model:
    the packed, row-major and column-major kernels share almost no code, so
    agreement between them is hard to reach by accident.
    """
    for query in QUERIES:
        assert cpm_reply(tiny_com, query) == cpm_reply(tiny_fast_com, query)
        assert cpm_reply(tiny_com, query) == cpm_reply(tiny_col_com, query)


@pytest.mark.parametrize("module", [buildz80com, buildfastz80com, buildcolz80com])
@pytest.mark.parametrize("query", QUERIES[:2])
def test_layer_widths_not_multiple_of_four(module, odd_model_path, odd_model, query):
    """Packed weights must stay aligned when a layer width isn't a multiple of 4."""
    image = module.build_autoreg(odd_model_path, max_output_len=GEN_LEN).build()
    assert cpm_reply(image, query) == libinfer.generate(odd_model, query, GEN_LEN)


def test_chat_mode_roundtrip(tiny_com, tiny_model):
    out, host = run_cpm(tiny_com, cmdline="", stdin=["HELLO", "!"], max_cycles=400_000_000)
    assert host.finished
    assert libinfer.generate(tiny_model, "HELLO", GEN_LEN) in out


def test_empty_query_is_handled(tiny_com, tiny_model):
    """A blank command line must not hang or crash - it enters chat mode."""
    out, host = run_cpm(tiny_com, cmdline="", stdin=["!"], max_cycles=50_000_000)
    assert host.finished
    assert ">" in out


@pytest.mark.slow
@pytest.mark.parametrize("module", [buildz80com, buildfastz80com, buildcolz80com])
@pytest.mark.parametrize("query", ["IS IT AN ANIMAL", "HELLO"])
def test_full_model_matches_reference(module, guess_model_path, query):
    model = libinfer.Model.load(guess_model_path)
    image = module.build_autoreg(guess_model_path, max_output_len=4).build()
    assert cpm_reply(image, query) == libinfer.generate(model, query, 4)
