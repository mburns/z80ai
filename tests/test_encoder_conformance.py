"""One encoder specification, checked against every backend that implements it.

The tokenizer and the context encoder are the part of this project duplicated
in a way that cannot be fixed by sharing code. libnn emits them for the Z80
with 16-bit activations and DJNZ counters; buildez80 emits them for the eZ80
with 24-bit activations and none. The arithmetic in between *is* shared - see
libnn.emit_hash_step, emit_times_seven and emit_band_index - but everything
touching a buffer is written twice and always will be.

What can be deduplicated is the specification check. Before this file each
backend had its own hand-picked queries: the Z80 was checked on three, the eZ80
on a different three, the banded path on two more, and the four newest targets
on none at all. A divergence in a case one list covered and another did not
would have gone unnoticed.

So the corpus lives here once and every backend runs all of it. Adding a case
covers every target; adding a target covers every case. The BACKENDS table
below is where the real per-machine differences are written down - buffer
label, activation width, whether the context has a buffer of its own - rather
than assumed.

These read buffers out of emulator memory rather than comparing generated text,
for the reason test_kernels gives: with few output classes two different logit
vectors often argmax to the same character.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

import buildcolz80com
import buildcpc
import buildez80
import buildfastz80com
import buildnext
import buildz80com
import buildz80tap
import libcpc
import libhost
import libinfer
import libzx
from helpers import MAX_CYCLES, read24, read_words

# --- the corpus --------------------------------------------------------------

#: Queries every backend is checked on. Each earns its place.
QUERIES = [
    "HELLO",                    # the ordinary case
    "A",                        # shorter than one trigram
    "AB",                       # one short of a trigram
    "ABC",                      # exactly one trigram
    "X",                        # from the eZ80's old list
    "ARE YOU A ROBOT",          # likewise
    " ",                        # blank, which is not the same as empty
    "   LEADING",               # leading blanks
    "TRAILING   ",              # and trailing
    "IS IT AN ANIMAL",          # the shipped guess example's shape
    "PUT THE KEY IN THE BOX",   # long enough to cross several position bands
    "AAAAAAAA",                 # every trigram identical: one bucket, hit often
    "ZZZ ZZZ ZZZ",              # repeats separated by spaces
    "MiXeD CaSe",               # only A-Z is lowered - see libinfer._lower
    "DIGITS 0123456789",        # not letters, so not lowered
    "WHY? BECAUSE! OK.",        # punctuation, which still hashes
    "X" * 62,                   # the longest line the input buffers accept
]

# A query may not *begin* with '!': in chat mode that is the exit command, so
# the chat-driven backends would quit rather than answer. CP/M's command-tail
# mode has no such character, which is why the corpus has to respect the
# stricter of the two. '!' elsewhere in a query is fine, and is covered above.
assert not any(q.startswith("!") for q in QUERIES)

#: The banded tests run every backend, so this is a subset of QUERIES chosen
#: to keep the cost down: one short, one crossing several bands, one at the
#: input limit, and the two the old per-backend banded tests used.
BANDED_QUERIES = ["HELLO", "A", "X Y Z W", "PUT THE KEY IN THE BOX", "X" * 62]


# --- the backends ------------------------------------------------------------


class Backend(NamedTuple):
    """A backend, and what a test needs to know to read its buffers back.

    These are the differences that survive sharing: the activation width the
    machine uses, the label its platform gave the buffer, and whether the
    context half follows the query half or lives somewhere else entirely.
    """

    module: Any
    #: Label of the buffer holding the tokenized query.
    buffer: str
    #: Separate context buffer, or None when the context half follows the query
    #: half in the same buffer, as it does on every Z80 target.
    context: str | None
    #: Bytes per activation: 2 on a Z80, 3 in ADL mode.
    width: int
    #: How a query reaches it. CP/M takes a command tail; the rest are
    #: chat-only and have it typed at the prompt.
    host: str


BACKENDS = {
    "cpm": Backend(buildz80com, "INBUF", None, 2, "cpm"),
    "cpm-fast": Backend(buildfastz80com, "INBUF", None, 2, "cpm"),
    "cpm-column": Backend(buildcolz80com, "INBUF", None, 2, "cpm"),
    "zx": Backend(buildz80tap, "TOKBUF", None, 2, "zx"),
    "next": Backend(buildnext, "TOKBUF", None, 2, "zx"),
    "cpc": Backend(buildcpc, "TOKBUF", None, 2, "cpc"),
    "ez80": Backend(buildez80, "INBUF", "CTXBUF", 3, "agon"),
}


@pytest.fixture(scope="module")
def builders(tiny_model_path):
    return {name: spec.module.build_autoreg(tiny_model_path, max_output_len=1)
            for name, spec in BACKENDS.items()}


@pytest.fixture(scope="module")
def banded_builders(banded_model_path):
    return {name: spec.module.build_autoreg(banded_model_path, max_output_len=1)
            for name, spec in BACKENDS.items()}


def _run_until(name: str, builder, query: str, label: str):
    """Drive a build to ``label`` with ``query`` supplied the way it expects."""
    spec = BACKENDS[name]
    if spec.host == "cpm":
        host = libhost.CPMHost(cmdline=query)
    elif spec.host == "zx":
        host = libhost.ZXHost(stdin=[query, "!"], org=builder.org)
    elif spec.host == "cpc":
        host = libhost.CPCHost(stdin=[query, "!"], org=builder.org)
    else:
        host = libhost.AgonHost(stdin=[query, "!"])

    cpu = host.cpu
    cpu.load(builder.org, builder.build())
    cpu.pc = builder.org
    cpu.run(max_cycles=MAX_CYCLES, stop_pc=builder.labels[label])
    assert cpu.pc == builder.labels[label], f"{name} never reached {label}"
    return cpu


def _read(cpu, spec: Backend, addr: int) -> np.ndarray:
    reader = read24 if spec.width == 3 else read_words
    return reader(cpu, addr, libinfer.NUM_BUCKETS)


def tokenize(builders: dict, name: str, query: str) -> np.ndarray:
    """The query half a backend encoded, read out of emulator memory."""
    spec, builder = BACKENDS[name], builders[name]
    cpu = _run_until(name, builder, query, "ARGMAX")
    return _read(cpu, spec, builder.labels[spec.buffer])


def context_of(builders: dict, name: str, query: str) -> np.ndarray:
    """The context half, wherever this backend keeps it."""
    spec, builder = BACKENDS[name], builders[name]
    cpu = _run_until(name, builder, query, "ARGMAX")
    if spec.context is not None:
        return _read(cpu, spec, builder.labels[spec.context])
    offset = libinfer.NUM_BUCKETS * spec.width
    return _read(cpu, spec, builder.labels[spec.buffer] + offset)


# --- the tokenizer -----------------------------------------------------------


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_tokenizer_matches_the_reference(builders, backend, query):
    np.testing.assert_array_equal(
        tokenize(builders, backend, query),
        libinfer.trigram_encode(query),
        err_msg=f"{backend} tokenized {query!r} differently from the reference",
    )


@pytest.mark.parametrize("query", QUERIES)
def test_every_backend_tokenizes_identically(builders, query):
    """No reference involved: the backends must also agree with each other.

    A change to the shared arithmetic that broke every backend the same way
    would still pass the test above if the reference were changed to match it.
    This one would not.
    """
    first = tokenize(builders, "cpm", query)
    for name in sorted(BACKENDS):
        np.testing.assert_array_equal(
            tokenize(builders, name, query), first,
            err_msg=f"{name} disagrees with cpm on {query!r}")


@pytest.mark.parametrize("query", BANDED_QUERIES)
@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_banded_tokenizer_matches_the_reference(banded_builders, backend, query):
    """The position-aware encoder.

    All four shipped models are flat, so nothing in test_codegen_stability pins
    these bytes. This is what stands behind the banded path.
    """
    np.testing.assert_array_equal(
        tokenize(banded_builders, backend, query),
        libinfer.trigram_encode(query, position_bands=8),
        err_msg=f"{backend} banded-tokenized {query!r} differently",
    )


@pytest.fixture(scope="module")
def few_band_builders(tmp_path_factory, model_factory):
    """A four-band model, so the clamp is reachable within one input line.

    With the usual eight bands the clamp is dead code on a Z80: BAND_WIDTH is
    8 and the input buffer holds 62 characters, so the highest band any query
    can reach is 61 // 8 == 7, which is already bands - 1. Only the eZ80, whose
    line is 120 characters, gets past it. Four bands puts the boundary at 32
    characters and brings every backend into range.
    """
    model = model_factory([256, 16, 12], position_bands=4)
    path = str(tmp_path_factory.mktemp("models") / "bands4.npz")
    model.save_npz(path)
    return {name: spec.module.build_autoreg(path, max_output_len=1)
            for name, spec in BACKENDS.items()}


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_a_query_past_the_last_band_is_clamped_not_wrapped(few_band_builders, backend):
    """Everything past the last band shares it.

    Wrapping instead would put the end of a long query in the same band as its
    start, which is exactly the collision bands exist to avoid.
    """
    long_query = "X" * 62
    assert len(long_query) // libinfer.BAND_WIDTH > 4 - 1, "clamp not reached"
    np.testing.assert_array_equal(
        tokenize(few_band_builders, backend, long_query),
        libinfer.trigram_encode(long_query, position_bands=4))


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_word_order_changes_the_banded_encoding(banded_builders, backend):
    """The end the whole option exists for, on every backend that offers it."""
    a = tokenize(banded_builders, backend, "PUT THE KEY IN THE BOX")
    b = tokenize(banded_builders, backend, "PUT THE BOX IN THE KEY")
    assert not np.array_equal(a, b), f"{backend} encodes both orders the same"


# --- the context encoder -----------------------------------------------------


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_initial_context_matches_the_reference(builders, backend):
    """Before a character is generated the window is eight spaces."""
    np.testing.assert_array_equal(
        context_of(builders, backend, "HELLO"),
        libinfer.context_encode(" " * libinfer.CONTEXT_LEN),
        err_msg=f"{backend} seeded its context differently",
    )


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_the_context_half_is_not_the_query_half(builders, backend):
    """A backend that read the wrong buffer would pass the test above by
    accident on a query that happened to encode to the same thing."""
    query = tokenize(builders, backend, "IS IT AN ANIMAL")
    context = context_of(builders, backend, "IS IT AN ANIMAL")
    assert not np.array_equal(query, context)


# --- what the corpus is for --------------------------------------------------


def test_every_target_that_emits_an_encoder_is_covered():
    """A backend added without a line in BACKENDS is one nothing checks here.

    Tied to bench.TARGETS by module rather than by name - the two use different
    labels for the same backends, so comparing names would only ever be a
    coincidence. Every module that emits an encoder has to appear in both.
    """
    import bench

    benched = {spec.module for spec in bench.TARGETS.values()}
    here = {spec.module.__name__ for spec in BACKENDS.values()}
    assert here == benched, f"here {sorted(here)}, bench {sorted(benched)}"


def test_the_declared_widths_match_what_the_backends_actually_emit():
    """The table is only useful if it is true."""
    import libnn

    for name, spec in BACKENDS.items():
        want = 3 if name == "ez80" else libnn.ACTIVATION_SIZE
        assert spec.width == want, name


def test_the_declared_buffer_labels_exist(builders):
    for name, spec in BACKENDS.items():
        labels = builders[name].labels
        assert spec.buffer in labels, f"{name} has no {spec.buffer}"
        if spec.context is not None:
            assert spec.context in labels, f"{name} has no {spec.context}"


def test_the_buffer_labels_are_the_ones_the_platforms_declare():
    """Rather than a second list that can drift from Platform.buffer."""
    import libcpm

    assert BACKENDS["cpm"].buffer == libcpm.CPMPlatform.buffer
    assert BACKENDS["zx"].buffer == libzx.ZXPlatform.buffer
    assert BACKENDS["cpc"].buffer == libcpc.CPCPlatform.buffer


def test_the_corpus_straddles_the_band_boundaries():
    """Bands are BAND_WIDTH apart; the corpus needs cases either side."""
    assert any(len(q) < libinfer.BAND_WIDTH for q in QUERIES)
    assert max(len(q) for q in QUERIES) // libinfer.BAND_WIDTH >= 7


def test_the_corpus_subsumes_the_per_backend_lists_it_replaced():
    """Every query the old hand-picked sets checked, in one place."""
    for query in ("HELLO", "A", "IS IT AN ANIMAL", "PUT THE KEY IN THE BOX"):
        assert query in QUERIES
