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
            assert relation in libgraph.CANONICAL, f"{path}: {relation}"


def test_held_out_phrasings_never_appear_in_training(db):
    """The whole point. Without this the 43.8% would drift up unnoticed."""
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
    born = [q for q, path in train if path == "born_in located_in"]
    assert born
    assert all("person" in q for q in born)


def test_a_missing_corpus_yields_no_chains_rather_than_inventing_them(tmp_path):
    train, unseen = relations.chains(tmp_path / "absent.db", "simplewiki", 5)
    assert not train and not unseen


def test_generation_is_reproducible(db):
    assert relations.chains(db, "simplewiki", 5, hold_out=2) == \
        relations.chains(db, "simplewiki", 5, hold_out=2)
