"""Tests for the position-aware query encoder.

The point of the option is that a reordered query encodes differently. The
risk is that the generated tokenizer and the reference disagree about *how*,
which would produce a model that scores well in training and nonsense on
hardware -- so the emulator comparisons below matter more than the unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import buildcolz80com
import buildez80
import buildfastz80com
import buildz80com
import buildz80tap
import libinfer
from libhost import AgonHost, CPMHost, run_agon, run_cpm
from libz80emu import Z80

REORDERED = [
    ("PUT THE KEY IN THE BOX", "PUT THE BOX IN THE KEY"),
    ("IS IT BIGGER THAN A CAT", "IS A CAT BIGGER THAN IT"),
    ("NORTH THEN EAST", "EAST THEN NORTH"),
]


# --- the reference encoder ---------------------------------------------------


def test_one_band_is_the_original_encoding():
    """Every model built before this option must tokenize unchanged."""
    for text in ("HELLO", "PUT THE KEY IN THE BOX", "", "A"):
        np.testing.assert_array_equal(
            libinfer.trigram_encode(text, position_bands=1),
            libinfer.trigram_encode(text),
        )


@pytest.mark.parametrize("a,b", REORDERED)
def test_bands_separate_reordered_queries(a, b):
    flat_a = libinfer.trigram_encode(a).astype(float)
    flat_b = libinfer.trigram_encode(b).astype(float)
    band_a = libinfer.trigram_encode(a, position_bands=8).astype(float)
    band_b = libinfer.trigram_encode(b, position_bands=8).astype(float)

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    assert cos(flat_a, flat_b) > 0.9, "the flat encoder should barely tell these apart"
    assert cos(band_a, band_b) < cos(flat_a, flat_b) - 0.15


def test_bands_do_not_change_the_number_of_trigrams():
    """Only which bucket each trigram lands in changes, not how many there are."""
    for text in ("HELLO", "A LONGER QUERY THAN THAT ONE"):
        flat = libinfer.trigram_encode(text)
        banded = libinfer.trigram_encode(text, position_bands=8)
        assert flat.sum() == banded.sum() == len(text) * libinfer.BUCKET_WEIGHT


@pytest.mark.parametrize(
    "index,bands,expected",
    [(0, 8, 0), (7, 8, 0), (8, 8, 1), (23, 8, 2), (200, 8, 7), (0, 1, 0), (99, 1, 0)],
)
def test_band_is_a_clamped_shift(index, bands, expected):
    assert libinfer.position_band(index, bands) == expected


def test_queries_shorter_than_one_band_are_unaffected_by_banding():
    """Under 8 characters every trigram is in band 0, so the seed is 0."""
    short = "HELLO"
    np.testing.assert_array_equal(
        libinfer.trigram_encode(short, position_bands=8),
        libinfer.trigram_encode(short),
    )


# --- the band travels with the model -----------------------------------------


def test_bands_round_trip_through_the_model_file(tmp_path, model_factory):
    model = model_factory([256, 8, 4], charset=" AB\x00", position_bands=8)
    path = str(tmp_path / "m.npz")
    model.save_npz(path)
    assert libinfer.Model.load(path).position_bands == 8


def test_a_model_without_the_field_defaults_to_flat(guess_model_path):
    """Shipped models predate the option and must keep tokenizing flat."""
    assert libinfer.Model.load(guess_model_path).position_bands == libinfer.FLAT


def test_encode_query_uses_the_model_s_own_setting(model_factory):
    banded = model_factory([256, 8, 4], charset=" AB\x00", position_bands=8)
    flat = model_factory([256, 8, 4], charset=" AB\x00")
    text = "PUT THE KEY IN THE BOX"
    assert not np.array_equal(banded.encode_query(text), flat.encode_query(text))
    np.testing.assert_array_equal(flat.encode_query(text),
                                  libinfer.trigram_encode(text))


# --- the generated tokenizers agree with the reference -----------------------


def _read_buckets(cpu: Z80, addr: int, count: int, width: int) -> np.ndarray:
    out = []
    for i in range(count):
        v = sum(cpu.peek(addr + width * i + k) << (8 * k) for k in range(width))
        top = 1 << (width * 8 - 1)
        out.append(v - (top << 1) if v & top else v)
    return np.array(out, dtype=np.int64)


@pytest.mark.parametrize("query", ["PUT THE KEY IN THE BOX", "HELLO", "A", "X Y Z W"])
def test_cpm_tokenizer_matches_the_banded_reference(banded_model_path, query):
    builder = buildz80com.build_autoreg(banded_model_path, max_output_len=1)
    host = CPMHost(cmdline=query)
    cpu = host.cpu
    cpu.load(0x0100, builder.build())
    cpu.pc = 0x0100
    cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["ARGMAX"])
    got = _read_buckets(cpu, builder.labels["INBUF"], 128, 2)
    np.testing.assert_array_equal(
        got, libinfer.trigram_encode(query, position_bands=8)
    )


@pytest.mark.parametrize("query", ["PUT THE KEY IN THE BOX", "HELLO"])
def test_ez80_tokenizer_matches_the_banded_reference(banded_model_path, query):
    builder = buildez80.build_autoreg(banded_model_path, max_output_len=1)
    host = AgonHost(stdin=[query, "!"])
    cpu = host.cpu
    cpu.load(buildez80.AGON_LOAD_ADDR, builder.build())
    cpu.pc = buildez80.AGON_LOAD_ADDR
    cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["ARGMAX"])
    got = _read_buckets(cpu, builder.labels["INBUF"], 128, 3)
    np.testing.assert_array_equal(
        got, libinfer.trigram_encode(query, position_bands=8)
    )


@pytest.mark.parametrize("a,b", REORDERED[:1])
def test_a_banded_build_answers_reordered_queries_differently(banded_model_path, a, b):
    """The end the whole option exists for, checked on the emitted binary."""
    image = buildz80com.build_autoreg(banded_model_path, max_output_len=6).build()
    out_a, _ = run_cpm(image, cmdline=a, max_cycles=400_000_000)
    out_b, _ = run_cpm(image, cmdline=b, max_cycles=400_000_000)
    model = libinfer.Model.load(banded_model_path)
    assert out_a == libinfer.generate(model, a, 6)
    assert out_b == libinfer.generate(model, b, 6)


@pytest.mark.parametrize(
    "module",
    [buildz80com, buildz80tap, buildfastz80com, buildcolz80com, buildez80],
)
def test_every_backend_honours_the_band_setting(banded_model_path, module):
    """A backend that ignored it would build a model that tokenizes wrongly."""
    banded = module.build_autoreg(banded_model_path, max_output_len=1)
    assert "TOKPOS" in banded.labels


@pytest.mark.parametrize(
    "module",
    [buildz80com, buildz80tap, buildfastz80com, buildcolz80com, buildez80],
)
def test_flat_models_emit_no_position_machinery(tiny_model_path, module):
    flat = module.build_autoreg(tiny_model_path, max_output_len=1)
    assert "TOKPOS" not in flat.labels


@pytest.mark.slow
def test_ez80_banded_build_matches_the_reference_end_to_end(banded_model_path):
    model = libinfer.Model.load(banded_model_path)
    image = buildez80.build_autoreg(banded_model_path, max_output_len=6).build()
    query = "PUT THE KEY IN THE BOX"
    out, _ = run_agon(image, stdin=[query, "!"], max_cycles=400_000_000)
    assert libinfer.generate(model, query, 6, accum_bits=24) in out
