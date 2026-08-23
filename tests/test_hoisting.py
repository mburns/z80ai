"""The query half of layer 1 is hoisted out of the generation loop.

`generate` encodes the query once and only the context half changes per
character, so layer 1's contribution from the query's 128 buckets is constant
for a whole response.  The backends compute it once, in PREQ, and hand it to
layer 1 as its bias.

The claim that has to hold is *equality*, not closeness: the accumulator is a
sum mod 2**bits and addition mod 2**n is associative, so regrouping the addends
cannot change a single bit.  Every test here is an exact comparison, and a
failure means the argument is wrong rather than that a tolerance needs
loosening.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import read_words
from helpers import run_cpm_until as run_until

import buildfastz80com
import buildz80com
import libinfer
import libnn

QUERIES = ["HELLO", "A", "IS IT AN ANIMAL", "", "   ", "x" * 60]


@pytest.fixture(scope="module")
def tiny_builder(tiny_model_path):
    return buildz80com.build_autoreg(tiny_model_path, max_output_len=1)


@pytest.fixture(scope="module")
def tiny_fast_builder(tiny_model_path):
    return buildfastz80com.build_autoreg(tiny_model_path, max_output_len=1)


def read_network_biases(cpu, builder, count: int) -> np.ndarray:
    """Layer 1's bias slots inside NETWORK, which PREQ rewrites each query.

    Walks the records the way PREQ does: three counted index lists per neuron,
    then the two bias bytes.
    """
    addr = builder.labels["L1REC"]
    out = []
    for _ in range(count):
        for _ in range(3):
            n = cpu.peek(addr)
            addr += 1 + n
        lo, hi = cpu.peek(addr), cpu.peek(addr + 1)
        value = lo | (hi << 8)
        out.append(value - 0x10000 if value & 0x8000 else value)
        addr += 2
    return np.array(out, dtype=np.int64)


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("context", [" " * 8, "hello wo", "yes"])
def test_hoisting_is_bit_identical_to_the_plain_forward(tiny_model, query, context):
    q = libinfer.trigram_encode(query)
    c = libinfer.context_encode(context)
    np.testing.assert_array_equal(
        libinfer.forward_hoisted(tiny_model, q, c),
        libinfer.forward(tiny_model, np.concatenate([q, c])),
    )


def test_hoisting_holds_when_the_accumulator_wraps():
    """The regrouping argument is a mod-2^16 one, so wrapping must not break it.

    A random model averages its weights out and never gets near the ceiling, so
    this one is built to overflow: every layer-1 weight is -2 against saturated
    buckets, which is -65,536 before the wrap.
    """
    rng = np.random.default_rng(3)
    model = libinfer.Model(
        weights=[
            np.full((16, 256), -2, dtype=np.int32),
            rng.integers(-2, 2, size=(12, 16)).astype(np.int32),
        ],
        biases=[np.full(16, 300, dtype=np.int32), np.zeros(12, dtype=np.int32)],
        charset=" ABCDEFGHIJ\x00",
    )
    q = np.full(128, 4 * libinfer.BUCKET_WEIGHT, dtype=np.int32)
    c = np.full(128, 4 * libinfer.BUCKET_WEIGHT, dtype=np.int32)
    raw = model.weights[0].astype(np.int64) @ np.concatenate([q, c]) + model.biases[0]
    assert np.abs(raw).max() > 32767, "this model does not actually overflow"
    np.testing.assert_array_equal(
        libinfer.forward_hoisted(model, q, c),
        libinfer.forward(model, np.concatenate([q, c])),
    )


def test_hoisting_is_exact_across_a_whole_generated_response(tiny_model):
    """Not just the first character: the context half changes, the query does not."""
    for query in QUERIES:
        q = tiny_model.encode_query(query)
        ctx = " " * libinfer.CONTEXT_LEN
        for _ in range(libinfer.MAX_OUTPUT_LEN):
            c = libinfer.context_encode(ctx)
            np.testing.assert_array_equal(
                libinfer.forward_hoisted(tiny_model, q, c),
                libinfer.forward(tiny_model, np.concatenate([q, c])),
            )
            idx = libinfer.argmax(libinfer.forward(tiny_model, np.concatenate([q, c])))
            if idx == tiny_model.eos_idx:
                break
            ctx = (ctx + libinfer._lower(tiny_model.charset[idx]))[-libinfer.CONTEXT_LEN:]


@pytest.mark.parametrize("query", ["HELLO", "IS IT AN ANIMAL"])
def test_preq_fills_qbias_with_the_reference_partial_sum(tiny_builder, tiny_model, query):
    """The value PREQ leaves in QBIAS is layer 1's bias with the query folded in."""
    cpu = run_until(tiny_builder, query, "ARGMAX")
    got = read_words(cpu, tiny_builder.labels["QBIAS"], tiny_model.layer_sizes[1])
    want = libinfer.query_bias(tiny_model, libinfer.trigram_encode(query))
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("query", ["HELLO", "IS IT AN ANIMAL"])
def test_fast_preq_patches_the_network_bias_slots(tiny_fast_builder, tiny_model, query):
    """The index-list build hoists by rewriting NETWORK, leaving INFER untouched."""
    cpu = run_until(tiny_fast_builder, query, "ARGMAX")
    got = read_network_biases(cpu, tiny_fast_builder, tiny_model.layer_sizes[1])
    want = libinfer.query_bias(tiny_model, libinfer.trigram_encode(query))
    np.testing.assert_array_equal(got, want)


def test_fast_layer_one_indexes_the_context_half_of_the_buffer(tiny_fast_builder):
    """Its indices name positions in the full activation buffer, so 128..255."""
    builder = tiny_fast_builder
    image = builder.build()
    addr = builder.labels["L1REC"] - builder.org
    for _ in range(16):  # tiny_model's layer 1 is 16 neurons wide
        for _ in range(3):
            n = image[addr]
            assert all(i >= libnn.NUM_BUCKETS for i in image[addr + 1 : addr + 1 + n])
            addr += 1 + n
        addr += 2
    qaddr = builder.labels["QWTS"] - builder.org
    for _ in range(16):
        for _ in range(3):
            n = image[qaddr]
            assert all(i < libnn.NUM_BUCKETS for i in image[qaddr + 1 : qaddr + 1 + n])
            qaddr += 1 + n
        qaddr += 2


def test_preq_runs_once_per_response_not_once_per_character(tiny_model_path):
    """PREQ is called from GENERATE, outside GENLOOP - that is the whole saving."""
    builder = buildz80com.build_autoreg(tiny_model_path, max_output_len=4)
    assert builder.labels["GENERATE"] < builder.labels["PREQ"]
    assert builder.labels["GENERATE"] < builder.labels["GENLOOP"]
    image = builder.build()
    call_preq = bytes([0xCD]) + (builder.labels["PREQ"]).to_bytes(2, "little")
    assert image.count(call_preq) == 1, "PREQ should be reachable from exactly one call"
    offset = image.index(call_preq) + builder.org
    assert builder.labels["GENERATE"] <= offset < builder.labels["GENLOOP"]


def test_layer_one_reads_only_the_context_half(tiny_model):
    plans = libnn.plan_layers(tiny_model.layer_sizes, "INBUF", hoist_query=True)
    assert plans[0].in_size == libnn.NUM_BUCKETS
    assert plans[0].in_offset == libnn.CONTEXT_OFFSET
    assert plans[0].bias_label == "QBIAS"
    assert plans[1].in_offset == 0
    assert plans[1].bias_label == "BIAS2"


def test_query_plan_walks_the_query_half_unscaled(tiny_model):
    plan = libnn.query_plan(tiny_model.layer_sizes, "INBUF")
    assert (plan.label, plan.weights_label, plan.bias_label) == ("PREQ", "WTS1Q", "BIAS1")
    assert (plan.in_offset, plan.in_size) == (0, libnn.NUM_BUCKETS)
    assert plan.out_buffer == "QBIAS"


def test_a_model_without_query_and_context_halves_is_refused():
    with pytest.raises(ValueError, match="query/context halves"):
        libnn.plan_layers([64, 32, 8], "INBUF", hoist_query=True)
