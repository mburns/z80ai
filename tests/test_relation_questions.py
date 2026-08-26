"""The question set behind the relation classifier.

Most of these guard the *holdout*, not the data. Generated questions flatter a
classifier by default, and the only defence is that the phrasings used to score
it were never used to train it. If that separation ever silently breaks, the
reported number goes up and nothing fails - which is the worst way for a
measurement to be wrong.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "questions"))

import relations

import libgraph


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "corpus.db"
    conn = sqlite3.connect(path)
    conn.executescript(libgraph.schema(strict=False))
    conn.executemany("INSERT INTO edge VALUES ('simplewiki', ?, ?, ?)", [
        (f"Person {i}", "born_in", "Somewhere") for i in range(50)
    ] + [
        (f"Place {i}", "located_in", "Elsewhere") for i in range(50)
    ] + [
        (f"Book {i}", "created_by", "Someone") for i in range(50)
    ] + [
        (f"Person {i}", "died_in", "Somewhere") for i in range(50)
    ])
    conn.commit()
    conn.close()
    return path


def test_every_chain_names_relations_libgraph_can_walk():
    """A path the graph cannot follow would train the model to answer nothing."""
    for path in relations.CHAINS:
        for relation in path.split():
            assert (relation in libgraph.CANONICAL
                    or relation in libgraph.CLIMB), f"{path}: {relation}"


def test_every_climb_steps_a_relation_that_exists():
    """`in_country` repeating a relation nothing emits would never move."""
    for climb, (step, _kind) in libgraph.CLIMB.items():
        assert step in libgraph.CANONICAL, f"{climb} steps {step}"


def test_held_out_phrasings_never_appear_in_training(db):
    """The whole point. Without this the score would drift up unnoticed.

    It guards the separation and not the number, which is the right division:
    the number moves 13 points across seeds on 320 questions, so no assertion
    on its value could be both true and useful. The separation is exact and
    can be asserted exactly.
    """
    train, unseen = relations.chains(db, "simplewiki", 5, hold_out=2)
    assert train and unseen

    assert not (templates_used(train) & templates_used(unseen))


def templates_used(pairs) -> set[tuple[str, str]]:
    """(path, template) for each question, recovered by matching it back.

    Identifying the exact template matters: a coarser fingerprint - first word,
    length - collides across phrasings and would pass while the sets overlapped.
    """
    found = set()
    for question, path in pairs:
        # A question can match several templates, because a greedy entity can
        # swallow a literal word: "what country is place 3 located in" fits
        # both "...is {s} located in" and "...is {s} in". The one that leaves
        # least to the wildcard is the one that produced it.
        matches = [t for t in relations.CHAINS[path]
                   if re.fullmatch(re.escape(t).replace(r"\{s\}", ".+"), question)]
        if not matches:
            raise AssertionError(f"no template produced {question!r}")
        found.add((path, max(matches, key=len)))
    return found


def test_holding_out_nothing_leaves_nothing_to_score(db):
    train, unseen = relations.chains(db, "simplewiki", 5, hold_out=0)
    assert train and not unseen


def test_entity_names_come_from_the_corpus(db):
    """A question about an entity the graph never heard of teaches nothing."""
    train, _ = relations.chains(db, "simplewiki", 5, hold_out=2)
    born = [q for q, path in train if path == "born_in in_country"]
    assert born
    assert all("person" in q for q in born)


def test_a_climb_draws_entities_from_the_relation_it_steps(db):
    """`in_country` names a type, not an edge. Asking the graph for subjects of
    a relation called `in_country` would quietly yield nothing, and the class
    would vanish from the training set with no error."""
    train, _ = relations.chains(db, "simplewiki", 5, hold_out=2)
    assert [q for q, path in train if path == "in_country"]


def test_a_missing_corpus_yields_no_chains_rather_than_inventing_them(tmp_path):
    train, unseen = relations.chains(tmp_path / "absent.db", "simplewiki", 5)
    assert not train and not unseen


def test_generation_is_reproducible(db):
    assert relations.chains(db, "simplewiki", 5, hold_out=2) == \
        relations.chains(db, "simplewiki", 5, hold_out=2)


#: The three wordings per path that the phrasing curve is scored against. They
#: are the *last* three of each tuple, because `chains` holds out from the end,
#: so appending a new phrasing silently changes what every measurement in
#: `relations.py` was taken against. New ones go before these.
HELD_OUT_TAILS = {
    "born_in in_country": ("in what country was {s} born",
                           "which country did {s} come from",
                           "what country was {s} a native of"),
    "died_in in_country": ("what country was {s} in when they died",
                           "which country did {s} pass away in",
                           "what country did {s} spend their last days in"),
    "in_country": ("what larger country is {s} within",
                   "which country does {s} sit in",
                   "what country is {s} located in"),
    "created_by born_in": ("where was the maker of {s} born",
                           "birthplace of whoever created {s}",
                           "where did the author of {s} come from"),
}


def test_the_held_out_phrasings_are_still_the_ones_the_curve_was_scored_on():
    """`chains` holds out from the end of each tuple, so a phrasing appended
    after these would quietly replace the evaluation set and make the numbers
    in `relations.py` measurements of something else. Add new ones above."""
    for path, tail in HELD_OUT_TAILS.items():
        assert relations.CHAINS[path][-3:] == tail, path


def test_chain_phrasings_trims_the_training_half_and_not_the_other(db):
    """`--chain-phrasings K` is what draws the curve, so it has to leave the
    held-out set **byte for byte** alone - not merely the same phrasings, the
    same questions. It did not at first: both halves drew entity names from one
    random stream, so keeping fewer training phrasings changed which names the
    evaluation half got, and every point on the curve was scored against
    something slightly different.
    """
    paths = len(relations.CHAINS)
    five, held_five = relations.chains(db, "simplewiki", 2, hold_out=3,
                                       phrasings=5)
    all_, held_all = relations.chains(db, "simplewiki", 2, hold_out=3)

    assert held_five == held_all
    assert len(held_all) == 3 * 2 * paths
    # K templates kept, per_template questions each, across every path.
    assert len(five) == 5 * 2 * paths
    assert len(all_) == (len(next(iter(relations.CHAINS.values()))) - 3) * 2 * paths
