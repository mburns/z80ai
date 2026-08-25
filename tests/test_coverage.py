"""The coverage harness, on a graph small enough to count by hand.

`coverage.py` is about to be the evidence for every claim about how much of the
corpus the oracle can walk, which makes its arithmetic load-bearing in a way a
one-off script's is not. A harness that is wrong in the flattering direction is
worse than no harness, so these pin the three places it could be:

    the denominator    subjects a path can start from, not all articles
    the hop count      hops taken, not nodes visited
    the type floor     how many independent infoboxes make a country

Each fixture below is small enough that the expected number is written out in
the test rather than computed, because a test that computes its expectation the
same way the code does agrees with the code by construction.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import libgraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))
import coverage
import ingest

#: Three independent infoboxes naming France, which is exactly `TYPE_FLOOR`, so
#: France is a country and nothing else is. Everything below leans on that.
ARTICLES = ["France", "Europe", "Paris", "Lyon", "Marseille", "London",
            "Springfield", "Nowhere", "Alice", "Bob", "Carol"]

FACTS = [
    # `country` is the highest-ranked `located_in` field, so each of these is
    # both a vote for France being a country and a located_in edge.
    ("Paris", "country", "France"),
    ("Lyon", "country", "France"),
    ("Marseille", "country", "France"),
    # France is a country that is itself inside something, so a climb starting
    # here has to stop at zero hops rather than step past what it was asked for.
    ("France", "region", "Europe"),
    # A climb that moves and still never arrives.
    ("Springfield", "state", "Nowhere"),
    # Alice climbs one hop; Bob was born in a country already; Carol was born
    # somewhere the corpus cannot place.
    ("Alice", "birth_place", "Paris"),
    ("Bob", "birth_place", "France"),
    ("Carol", "birth_place", "London"),
]


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(ingest._schema())
    conn.executemany("INSERT INTO article (source, title, lead) "
                     "VALUES ('w', ?, '')", [(a,) for a in ARTICLES])
    conn.executemany(
        "INSERT OR REPLACE INTO fact (source, subject, property, ordinal, "
        "value, kind, num) VALUES ('w', ?, ?, 0, ?, 'text', NULL)", FACTS)
    conn.executemany(
        "INSERT INTO property (source, name, uses, subjects, relation) "
        "VALUES ('w', ?, ?, ?, ?)",
        [("country", 3, 3, "located_in"), ("region", 1, 1, "located_in"),
         ("state", 1, 1, "located_in"), ("birth_place", 3, 3, "born_in"),
         ("name", 900, 900, None), ("clubs", 40, 12, None)])
    libgraph.build(conn, "w")
    return conn


# --- the denominator ----------------------------------------------------------


def test_a_path_is_scored_over_subjects_it_could_start_from(db):
    """Three people have a birthplace, so three is the denominator.

    Scoring over all eleven articles would report 18% and blame the graph for
    the eight articles that are not people.
    """
    subjects = coverage.head_subjects(db, "w", "born_in")
    assert sorted(subjects) == ["Alice", "Bob", "Carol"]

    scored = coverage.score_path(db, "w", ["born_in", "in_country"], subjects)
    assert scored["startable"] == 3
    assert scored["complete"] == 2       # Alice via Paris, Bob directly
    assert scored["rate"] == pytest.approx(2 / 3)


def test_a_climb_starts_from_the_relation_it_steps(db):
    """No edge is ever labelled `in_country`.

    Asking the edge table for one returns nothing, which would report every
    path through a climb as having no subjects at all - a coverage harness
    reporting 0/0 instead of the number it exists to produce.
    """
    assert coverage.head_subjects(db, "w", "in_country") == sorted(
        ["France", "Lyon", "Marseille", "Paris", "Springfield"])


def test_an_incomplete_walk_is_attributed_to_the_hop_that_broke(db):
    """Which hop failed decides what would fix it, so it is worth counting."""
    subjects = coverage.head_subjects(db, "w", "born_in")
    scored = coverage.score_path(db, "w", ["born_in", "in_country"], subjects)
    assert scored["stopped_at"] == {"in_country": 1}      # Carol, via London


# --- the hop count ------------------------------------------------------------


def test_climb_distance_counts_hops_not_nodes(db):
    """`Answer.path` opens with the subject, so a zero-hop climb has length 1.

    The zero-hop case is the one `CLIMB` exists for - a quarter of this
    corpus's birthplaces are already countries - and an off-by-one here would
    report it as one hop and hide that the type test fires first.
    """
    subjects = coverage.head_subjects(db, "w", "in_country")
    hist = coverage.climb_distances(db, "w", "in_country", subjects)
    assert hist == {"0": 1, "1": 3, "never": 1}


def test_the_histogram_puts_never_last(db):
    """It is read as a distance curve, and a failure is not a distance."""
    subjects = coverage.head_subjects(db, "w", "in_country")
    assert list(coverage.climb_distances(db, "w", "in_country",
                                         subjects))[-1] == "never"


# --- the type floor -----------------------------------------------------------


def test_the_floor_curve_shows_where_the_setting_bites(db):
    """France has exactly three votes, so it is a country up to a floor of 3.

    `TYPE_FLOOR` is chosen from two hand-counted points; the curve is what
    makes the choice reviewable.
    """
    assert coverage.type_floors(db, "w") == {
        "1": 1, "2": 1, "3": 1, "4": 0, "5": 0}


# --- the headline numbers -----------------------------------------------------


def test_reach_separates_having_a_fact_from_being_on_the_graph(db):
    """The gap between the two is the whole argument for mapping properties.

    Eight subjects carry a fact and eight reach the graph here only because
    every property in this fixture is mapped; in the real corpus the second
    number is far smaller, and that difference is what the harness is for.
    """
    r = coverage.reach(db, "w")
    assert r["articles"] == 11
    assert r["facts"] == len(FACTS)
    assert r["fact_subjects"] == 8
    assert r["graph_subjects"] == 8
    assert r["properties"] == 6
    assert r["properties_mapped"] == 4


def test_unmapped_properties_are_ranked_by_use(db):
    """The to-do list for `CANONICAL`, biggest first."""
    assert [p["name"] for p in coverage.unmapped(db, "w")] == ["name", "clubs"]


def test_measure_reports_every_chain_the_classifier_can_emit(db):
    """A path the model can emit and the harness cannot score is a blind spot."""
    import relations

    measured = coverage.measure(db, "w", sample=0, seed=0)
    for path in relations.CHAINS:
        assert path in measured["paths"], path
