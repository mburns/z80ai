"""Numeric tests on the column-major CP/M kernel.

Comparing generated text is a weak signal: with few output classes two different
logit vectors often argmax to the same character.  These read the buffers out of
emulator memory and compare every value against the reference, so a packing or
accumulation bug cannot hide.

Two of them check something no correctness test can: that the kernel actually
*skips* the columns it is supposed to skip.  Running a zero column is
numerically harmless - it adds zero everywhere - so without these the whole
optimization could regress into a slower kernel that still passes everything
else.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import read_words, reference_input
from helpers import run_cpm_until as run_until

import buildcolz80com
import libinfer
import libnn
from libz80 import Z80Builder


def read_split(cpu, base: int, count: int, stride: int = 256) -> np.ndarray:
    """Read activations stored low-bytes-then-high-bytes ``stride`` apart."""
    out = []
    for i in range(count):
        value = cpu.peek(base + i) | (cpu.peek(base + stride + i) << 8)
        out.append(value - 0x10000 if value & 0x8000 else value)
    return np.array(out, dtype=np.int64)


def read_collist(cpu, builder, count: int) -> list[int]:
    base = builder.labels["COLLIST"]
    return [cpu.peek(base + i) for i in range(count)]


def read_ncol(cpu, builder) -> int:
    base = builder.labels["NCOL"]
    return cpu.peek(base) | (cpu.peek(base + 1) << 8)


@pytest.fixture(scope="module")
def builder(tiny_model_path):
    return buildcolz80com.build_autoreg(tiny_model_path, max_output_len=1)


# --- the record layout -------------------------------------------------------


def test_records_hold_every_nonzero_weight_exactly_once():
    rng = np.random.default_rng(4)
    w = rng.choice([-2, -1, 0, 1], size=(20, 12), p=[0.05, 0.2, 0.6, 0.15])
    blob, offsets = buildcolz80com.column_records(w)

    rebuilt = np.zeros_like(w)
    for col, start in enumerate(offsets):
        at = start
        for value in buildcolz80com.WEIGHT_VALUES:
            n = blob[at]
            at += 1
            for row in blob[at : at + n]:
                assert rebuilt[row, col] == 0, "a weight was listed twice"
                rebuilt[row, col] = value
            at += n
    np.testing.assert_array_equal(rebuilt, w)


def test_records_are_contiguous_and_cover_the_blob():
    rng = np.random.default_rng(9)
    w = rng.choice([-2, -1, 0, 1], size=(8, 5), p=[0.05, 0.2, 0.6, 0.15])
    blob, offsets = buildcolz80com.column_records(w)
    assert offsets[0] == 0
    assert offsets == sorted(offsets)
    nonzero = int((w != 0).sum())
    assert len(blob) == nonzero + 3 * w.shape[1]


def test_a_column_too_dense_for_a_count_byte_is_refused():
    w = np.ones((256, 1), dtype=np.int8)
    with pytest.raises(ValueError, match="more than a count byte holds"):
        buildcolz80com.column_records(w)


def test_a_layer_too_wide_for_an_eight_bit_index_is_refused():
    with pytest.raises(ValueError, match="exceeds the 8-bit index"):
        buildcolz80com.column_records(np.zeros((257, 4), dtype=np.int8))


def test_page_selection_rejects_an_unaligned_buffer():
    """LD H,<page> only addresses a buffer that starts on a page boundary."""
    b = Z80Builder()
    b.label("START")
    b.ld_h_page("ODD")
    b.ret()
    b.db(0)
    b.label("ODD")
    with pytest.raises(ValueError, match="not page-aligned"):
        b.build()


# --- what the kernel computes ------------------------------------------------


@pytest.mark.parametrize("query", ["HELLO", "IS IT AN ANIMAL", "X"])
def test_output_logits_match_reference(builder, tiny_model, query):
    cpu = run_until(builder, query, "ARGMAX")
    got = read_words(cpu, builder.labels["OUTBUF"], tiny_model.output_size)
    want = libinfer.forward(tiny_model, reference_input(query))
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("query", ["HELLO", "IS IT AN ANIMAL"])
def test_hidden_activations_match_reference(builder, tiny_model, query):
    """Layer by layer, so a failure says which layer rather than just 'wrong'."""
    cpu = run_until(builder, query, "ARGMAX")
    want = libinfer.forward_layers(tiny_model, reference_input(query))
    # Layer 1 writes ACT_B, layer 2 ACT_A, and so on; the last writes OUTBUF.
    for i, activations in enumerate(want[:-1]):
        page = "ACT_B" if i % 2 == 0 else "ACT_A"
        got = read_split(cpu, builder.labels[page], len(activations))
        np.testing.assert_array_equal(got, activations, err_msg=f"layer {i + 1}")


def test_qbase_matches_the_reference_query_partial(builder, tiny_model):
    cpu = run_until(builder, "HELLO", "GENLOOP")
    width = tiny_model.layer_sizes[1]
    got = read_split(cpu, builder.labels["QBASELO"], width, stride=width)
    want = libinfer.query_bias(tiny_model, libinfer.trigram_encode("HELLO"))
    np.testing.assert_array_equal(got, want)


def test_odd_layer_widths(odd_model_path, odd_model):
    """Column records have no alignment to lose, but check rather than assume."""
    builder = buildcolz80com.build_autoreg(odd_model_path, max_output_len=1)
    cpu = run_until(builder, "HELLO", "ARGMAX")
    got = read_words(cpu, builder.labels["OUTBUF"], odd_model.output_size)
    np.testing.assert_array_equal(
        got, libinfer.forward(odd_model, reference_input("HELLO"))
    )


# --- what the kernel skips ---------------------------------------------------


def test_layer_one_lists_exactly_the_nonzero_context_buckets(builder):
    """The query half is PREQ's business; layer 1 only ever sees the context."""
    cpu = run_until(builder, "HELLO", "LAYER1")
    x = reference_input("HELLO")
    want = [j for j in range(libnn.NUM_BUCKETS, len(x)) if x[j] != 0]
    assert want, "the context tokenized to nothing"
    assert read_ncol(cpu, builder) == len(want)
    assert read_collist(cpu, builder, len(want)) == want


def test_preq_lists_exactly_the_nonzero_query_buckets(builder):
    cpu = run_until(builder, "HELLO", "DRIVE1")
    x = reference_input("HELLO")
    want = [j for j in range(libnn.NUM_BUCKETS) if x[j] != 0]
    assert want, "the query tokenized to nothing"
    assert read_ncol(cpu, builder) == len(want)
    assert read_collist(cpu, builder, len(want)) == want


def test_the_column_list_has_no_duplicates(builder):
    """TOK_HASH can hit one bucket twice; listing it twice would double it."""
    cpu = run_until(builder, "ABABABAB", "LAYER1")
    n = read_ncol(cpu, builder)
    entries = read_collist(cpu, builder, n)
    assert len(entries) == len(set(entries))


def test_hidden_layers_list_exactly_their_nonzero_activations(builder, tiny_model):
    """This is the win: a dense list would still be correct, just slower."""
    cpu = run_until(builder, "HELLO", "LAYER2")
    activations = libinfer.forward_layers(tiny_model, reference_input("HELLO"))[0]
    want = [j for j in range(len(activations)) if activations[j] != 0]
    assert want, "the fixture produced an all-zero hidden layer"
    assert len(want) < len(activations), "nothing was skipped, so nothing is tested"
    assert read_ncol(cpu, builder) == len(want)
    assert read_collist(cpu, builder, len(want)) == want


def test_an_activation_that_floors_to_zero_is_not_listed(tmp_path, model_factory):
    """A small *positive* accumulator shifts to zero, and must not be listed.

    ReLU catches the negative case earlier, so a pre-shift accumulator in [0, 3]
    is the only way a neuron reaches the shift and still comes out zero.
    """
    model = model_factory([256, 24, 16], seed=5)
    model.weights[0][7, :] = 0
    model.biases[0][7] = 2
    path = str(tmp_path / "floor.npz")
    model.save_npz(path)

    activations = libinfer.forward_layers(model, reference_input("HELLO"))[0]
    assert activations[7] == 0, "the fixture did not land in the flooring band"

    builder = buildcolz80com.build_autoreg(path, max_output_len=1)
    cpu = run_until(builder, "HELLO", "LAYER2")
    assert 7 not in read_collist(cpu, builder, read_ncol(cpu, builder))


def test_a_full_column_list_is_not_mistaken_for_an_empty_one(tmp_path, model_factory):
    """256 active columns is the count that does not fit in a byte.

    Stored as one byte it reads back as zero, which the driver would take as
    "nothing to do" and skip the layer - the right answer for an empty list and
    catastrophically wrong for a full one. Nothing about the output of a natural
    model makes this likely, which is why it gets a fixture built to hit it.
    """
    model = model_factory([256, 256, 12], seed=13)
    # Enough bias to keep every layer-1 neuron positive whatever the input does,
    # and still far short of the 16-bit accumulator.
    model.biases[0][:] = 6000
    path = str(tmp_path / "full.npz")
    model.save_npz(path)

    activations = libinfer.forward_layers(model, reference_input("HELLO"))[0]
    assert (activations != 0).all(), "the fixture did not fill the list"

    builder = buildcolz80com.build_autoreg(path, max_output_len=1)
    cpu = run_until(builder, "HELLO", "LAYER2")
    assert read_ncol(cpu, builder) == 256

    cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["ARGMAX"])
    assert cpu.pc == builder.labels["ARGMAX"]
    got = read_words(cpu, builder.labels["OUTBUF"], model.output_size)
    np.testing.assert_array_equal(
        got, libinfer.forward(model, reference_input("HELLO"))
    )


def test_an_all_zero_input_still_leaves_the_biases_in_place(tmp_path, model_factory):
    """No active columns at all: every accumulator must still hold its bias."""
    model = model_factory([256, 10], charset=" AB\x00", seed=67)
    path = str(tmp_path / "zeroin.npz")
    model.save_npz(path)
    builder = buildcolz80com.build_autoreg(path, max_output_len=1)

    cpu = run_until(builder, "HELLO", "GENERATE")
    for addr in range(builder.labels["INBUF"], builder.labels["INBUF"] + 256 * 2):
        cpu.poke(addr, 0)
    cpu.run(max_cycles=10_000_000, stop_pc=builder.labels["ARGMAX"])
    assert cpu.pc == builder.labels["ARGMAX"]

    got = read_words(cpu, builder.labels["OUTBUF"], model.output_size)
    want = libinfer.forward(model, np.zeros(256, dtype=np.int64))
    np.testing.assert_array_equal(got, want)
