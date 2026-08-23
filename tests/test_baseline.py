"""The accuracy claims, as a test rather than as documentation.

Nothing in CI ran data/baseline.py before this file, so every number in the
README was prose that no build could contradict.  A retrain that halved a
model's accuracy would have shipped green.

Two things are pinned here.  The deterministic rows - the keyword table and the
three retrievers - are pure functions of a checked-in .gz and seed 0, so they
are fixed constants; if one moves, either the encoder changed or the data did.
The model rows move only when a model.npz is retrained, which must happen in the
same commit that updates them, the same discipline test_codegen_stability.py
applies to the built binaries.

The tolerance is loose enough to absorb float noise across NumPy versions and
tight enough that a real regression cannot hide inside it.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

import baseline
import libdata
import libinfer

TOLERANCE = 0.005

#: example -> row -> (overall, macro) on split_pairs(pairs, 0.1, seed=0).
GOLDEN: dict[str, dict[str, tuple[float, float]]] = {
    "smalltalk": {
        "keyword": (0.5936, 0.6209),
        "centroid": (0.7456, 0.7484),
        "nn_trigram": (0.8410, 0.8427),
        "nn_jaccard": (0.8163, 0.8168),
        "model": (0.8057, 0.8075),
    },
    "tinychat": {
        "keyword": (0.2585, 0.1429),
        "centroid": (0.3024, 0.4218),
        "nn_trigram": (0.4000, 0.4212),
        "nn_jaccard": (0.3561, 0.3449),
        "model": (0.4000, 0.4444),
    },
    "guess": {
        "keyword": (0.7661, 0.4839),
        "centroid": (0.5233, 0.5990),
        "nn_trigram": (0.8943, 0.8825),
        "nn_jaccard": (0.7098, 0.6984),
        "model": (0.8131, 0.8654),
    },
}


def _score_all(example: str, repo_root: str | Path) -> dict[str, tuple[float, float]]:
    root = Path(repo_root)
    pairs = libdata.read_files([str(root / "examples" / example
                                    / "training-data.txt.gz")])
    train, val = libdata.split_pairs(pairs, 0.1, 0)

    table = baseline.build_table(train)
    fallback = Counter(r for _, r in train).most_common(1)[0][0]
    scored = {
        "keyword": libdata.score_predictions(
            val, lambda q: baseline.classify(q, table, fallback)),
    }
    for name, factory in (("centroid", baseline.nearest_centroid),
                          ("nn_trigram", baseline.nearest_neighbour),
                          ("nn_jaccard", baseline.nearest_neighbour_words)):
        predict, _ = factory(train)
        scored[name] = libdata.score_predictions(val, predict)

    model = libinfer.Model.load(str(root / "examples" / example / "model.npz"))
    longest = max(len(r) for _, r in pairs) + 1
    scored["model"] = libdata.score_predictions(
        val, lambda q: libinfer.generate(model, q, longest))
    return scored


@pytest.fixture(scope="module")
def scored(repo_root):
    return {name: _score_all(name, repo_root) for name in GOLDEN}


@pytest.mark.slow
@pytest.mark.parametrize("example", sorted(GOLDEN))
def test_every_row_matches_its_pinned_value(scored, example):
    for row, expected in GOLDEN[example].items():
        overall, macro = scored[example][row]
        assert overall == pytest.approx(expected[0], abs=TOLERANCE), \
            f"{example}/{row} overall moved"
        assert macro == pytest.approx(expected[1], abs=TOLERANCE), \
            f"{example}/{row} macro moved"


@pytest.mark.slow
@pytest.mark.parametrize("example", sorted(GOLDEN))
def test_the_model_beats_a_keyword_table(scored, example):
    """The claim the README actually makes, at the budget it makes it for."""
    assert scored[example]["model"][1] > scored[example]["keyword"][1]


@pytest.mark.slow
def test_a_retriever_beats_the_model_once_storage_is_free():
    """Not a regression - a finding, kept honest by being a test.

    On a machine with an SD card the keyword table stops being the floor. If
    this ever fails it means the model finally overtook a plain 1-NN over the
    corpus, which would be worth knowing immediately and is the single result
    that would justify reopening the mixture-of-experts design.
    """
    assert GOLDEN["smalltalk"]["nn_trigram"][1] > GOLDEN["smalltalk"]["model"][1]


def test_a_retriever_states_what_it_would_cost_on_device():
    """Every row carries a byte cost, because accuracy alone stops meaning
    anything once one row is allowed 32GB and another 64KB."""
    pairs = [("HELLO THERE", "HI"), ("GOODBYE NOW", "BYE"),
             ("HI THERE FRIEND", "HI"), ("SEE YOU LATER", "BYE")]
    _, centroid_bytes = baseline.nearest_centroid(pairs)
    _, nn_bytes = baseline.nearest_neighbour(pairs)

    # One dense vector per reply, against one sparse vector per example.
    assert centroid_bytes == 2 * libinfer.NUM_BUCKETS * baseline.BUCKET_BYTES
    assert nn_bytes > 0


def test_the_sparse_cost_is_the_one_an_agon_would_pay():
    """A trigram vector is ~20% nonzero, so the dense int16 form overstates it.

    Costing the retriever at 128 int16 per example would make it look five
    times more expensive than it is, and flatter the model for free.
    """
    pairs = [(q, "R") for q in
             ("WHAT IS THE WEATHER", "HOW ARE YOU TODAY", "TELL ME A JOKE")]
    _, nbytes = baseline.nearest_neighbour(pairs)
    dense = len(pairs) * libinfer.NUM_BUCKETS * baseline.BUCKET_BYTES
    assert nbytes < dense / 2
