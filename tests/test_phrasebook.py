"""A phrasebook model picks an index; the text lives somewhere else.

These cover the reference side only - libinfer.classify and the metadata that
has to survive a round trip through a .npz.  The generated eZ80 code that has to
agree with them comes later; this is what it will be checked against.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

import libinfer


def make_phrasebook(phrases: list[str], seed: int = 3) -> libinfer.Model:
    """A 128 -> 16 -> len(phrases) classifier with arbitrary but fixed weights."""
    rng = np.random.default_rng(seed)
    sizes = [libinfer.NUM_BUCKETS, 16, len(phrases)]
    weights, biases = [], []
    for a, b in itertools.pairwise(sizes):
        weights.append(rng.choice([-2, -1, 0, 1], size=(b, a),
                                  p=[0.05, 0.15, 0.65, 0.15]).astype(np.int32))
        biases.append(rng.integers(-32, 32, size=b).astype(np.int32))
    return libinfer.Model(weights=weights, biases=biases, charset="\x00",
                          phrases=phrases, accum_bits=24)


PHRASES = ["I FROZE YOUR ACCOUNT", "YOUR FLIGHT IS ON TIME", "I DO NOT KNOW THAT ONE"]


def test_classify_returns_the_phrase_its_index_selects():
    model = make_phrasebook(PHRASES)
    for query in ("FREEZE MY ACCOUNT", "IS MY FLIGHT LATE", "WHAT IS A QUARK"):
        idx = libinfer.classify_index(model, query, accum_bits=24)
        assert libinfer.classify(model, query, accum_bits=24) == PHRASES[idx]


def test_classify_is_one_forward_pass_over_the_query_half_only():
    """No context, no autoregression - the input is 128 buckets, not 256."""
    model = make_phrasebook(PHRASES)
    assert model.input_size == libinfer.NUM_BUCKETS

    query = "FREEZE MY ACCOUNT"
    expected = libinfer.argmax(
        libinfer.forward(model, libinfer.trigram_encode(query), accum_bits=24))
    assert libinfer.classify_index(model, query, accum_bits=24) == expected


def test_a_character_model_refuses_to_classify():
    """The two decoders are not interchangeable, and the failure should say so."""
    model = make_phrasebook(PHRASES)
    model.phrases = None
    with pytest.raises(ValueError, match="no phrasebook"):
        libinfer.classify(model, "ANYTHING")


def test_phrases_and_accum_bits_survive_a_round_trip(tmp_path):
    model = make_phrasebook(PHRASES)
    model.split_seed = 7
    path = str(tmp_path / "phrasebook.npz")
    model.save_npz(path)

    loaded = libinfer.Model.load(path)
    assert loaded.phrases == PHRASES
    assert loaded.accum_bits == 24
    assert loaded.split_seed == 7
    for got, want in zip(loaded.weights, model.weights, strict=True):
        np.testing.assert_array_equal(got, want)


def test_a_character_model_round_trips_without_phrases(tmp_path):
    """The new metadata must not appear on models that do not use it."""
    model = make_phrasebook(PHRASES)
    model.phrases = None
    model.charset = " ABC\x00"
    path = str(tmp_path / "chars.npz")
    model.save_npz(path)

    assert "phrases" not in libinfer.Model.load(path).architecture()
    assert libinfer.Model.load(path).phrases is None


def test_shipped_models_predate_the_split_seed_and_say_so(examples_dir):
    """None of them recorded one, and load must not invent a default.

    data/baseline.py distinguishes 'trained on a different split' from 'did not
    record which split', and only the first is a warning worth shouting about.
    """
    model = libinfer.Model.load(
        str(Path(examples_dir) / "smalltalk" / "model.npz"))
    assert model.split_seed is None
    assert model.phrases is None
    assert model.accum_bits == 16
