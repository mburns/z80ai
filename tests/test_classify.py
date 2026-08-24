"""Tests for the phrasebook trainer.

classify.py trains both shipped phrasebook models - clinc150 and smalltalk's -
and until the coverage report was turned on nothing imported it at all. It sat
at 0%, which is how a script that produces released artifacts ends up untested:
it needs torch, CI has none, and nobody noticed the difference between "cannot
run here" and "does not run anywhere".

These need torch and skip without it, like tests/test_model_io.py. They are a
smoke test, not a training benchmark: what is checked is that the trainer still
produces a model the rest of the toolchain accepts.
"""

from __future__ import annotations

import pytest

import libinfer

torch = pytest.importorskip("torch", reason="the trainer needs PyTorch")

import classify  # noqa: E402

#: Four intents with a handful of phrasings each - enough for a split to have
#: something on both sides, small enough to train in a second.
PAIRS = [
    ("HELLO", "HI THERE"), ("HI", "HI THERE"), ("HEY THERE", "HI THERE"),
    ("GOOD MORNING", "HI THERE"), ("GREETINGS", "HI THERE"),
    ("BYE", "SEE YOU LATER"), ("GOODBYE", "SEE YOU LATER"),
    ("SEE YOU", "SEE YOU LATER"), ("FAREWELL", "SEE YOU LATER"),
    ("THANKS", "YOU ARE WELCOME"), ("THANK YOU", "YOU ARE WELCOME"),
    ("CHEERS", "YOU ARE WELCOME"), ("TA VERY MUCH", "YOU ARE WELCOME"),
    ("WHO ARE YOU", "I AM A BOT"), ("WHAT ARE YOU", "I AM A BOT"),
    ("ARE YOU A ROBOT", "I AM A BOT"), ("ARE YOU HUMAN", "I AM A BOT"),
]

TRAIN = {"hidden_sizes": [32], "epochs": 8, "lr": 0.01, "seed": 1,
         "split_seed": 0, "val_frac": 0.25, "accum_bits": 24,
         "position_bands": libinfer.FLAT}


@pytest.fixture(scope="module")
def trained():
    model, overall, macro = classify.train(PAIRS, quiet=True, **TRAIN)
    return model, overall, macro


def test_the_trainer_produces_a_phrasebook_model(trained):
    model, _overall, _macro = trained
    assert model.phrases is not None


def test_the_phrase_list_is_sorted(trained):
    """The index is baked into the weights, into the .DAT and into the
    reference model; a set's iteration order would make the three disagree."""
    model, _overall, _macro = trained
    assert model.phrases == sorted({r for _, r in PAIRS})


def test_the_output_layer_is_one_neuron_per_phrase(trained):
    """buildez80 refuses a model where these disagree, so it is worth knowing
    the trainer is the thing that got it right."""
    model, _overall, _macro = trained
    assert model.output_size == len(model.phrases)


def test_the_input_is_the_query_half_only(trained):
    """A phrasebook answers in one step, so there is no context to condition
    on and layer one is half the size the character decoder's is."""
    model, _overall, _macro = trained
    assert model.input_size == libinfer.NUM_BUCKETS


def test_the_charset_says_it_decodes_through_phrases(trained):
    """Model.charset is not optional and buildez80 sizes CHARTBL from it."""
    model, _overall, _macro = trained
    assert model.charset == "\x00"


def test_the_weights_are_two_bit(trained):
    model, _overall, _macro = trained
    for w in model.weights:
        assert {int(v) for v in w.flatten()} <= {-2, -1, 0, 1}


def test_training_is_reproducible():
    """classify.py is seeded, unlike feedme.py - TRAINING.md says so, and the
    clinc150 README tells people the published numbers reproduce."""
    first, _o1, m1 = classify.train(PAIRS, quiet=True, **TRAIN)
    second, _o2, m2 = classify.train(PAIRS, quiet=True, **TRAIN)
    assert m1 == m2
    for a, b in zip(first.weights, second.weights, strict=True):
        assert (a == b).all()


def test_the_model_round_trips_through_npz(trained, tmp_path):
    """Which is how it reaches buildez80 at all."""
    model, _overall, _macro = trained
    path = str(tmp_path / "phrasebook.npz")
    model.save_npz(path)

    loaded = libinfer.Model.load(path)
    assert loaded.phrases == model.phrases
    for a, b in zip(loaded.weights, model.weights, strict=True):
        assert (a == b).all()


def test_a_trained_phrasebook_builds_and_answers(trained, tmp_path):
    """End to end: the trainer's output through buildez80 and the emulator."""
    import buildez80
    from libhost import run_agon

    model, _overall, _macro = trained
    path = str(tmp_path / "phrasebook.npz")
    model.save_npz(path)

    builder = buildez80.build_autoreg(path, phrases_file="P.DAT")
    out, _host = run_agon(builder.build(), stdin=["HELLO", "!"],
                          files={"P.DAT": builder.phrase_blob})
    assert libinfer.classify(model, "HELLO", accum_bits=24) in out


def test_evaluate_scores_against_the_integer_arithmetic(trained):
    """Not the float model: what ships is the quantized one."""
    model, overall, macro = trained
    again = classify.evaluate(model, PAIRS, accum_bits=24)
    assert 0.0 <= overall <= 1.0
    assert 0.0 <= macro <= 1.0
    assert all(0.0 <= v <= 1.0 for v in again)


def test_encode_produces_one_row_per_query():
    rows = classify.encode(["HELLO", "GOODBYE"])
    assert rows.shape == (2, libinfer.NUM_BUCKETS)


def test_a_split_with_nothing_held_out_is_refused():
    with pytest.raises(SystemExit, match="val-frac"):
        classify.train(PAIRS, quiet=True, **{**TRAIN, "val_frac": 0.001})
