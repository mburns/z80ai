"""Reading birthplaces out of prose, and refusing to trust it on faith.

This is the only part of the pipeline that reads a sentence rather than a
table, so it is the only part that can be confidently wrong. Two things are
pinned here accordingly:

    the extraction   what the pattern takes, and where it stops taking
    the provenance   that nothing it produces reaches `fact` or `edge`

The scoring is pinned too, because a harness that flatters its extractor is
worse than no harness: `agreement` was the obvious measure and it reads 13.8%
where the extractor is actually 93.4% right about the country, which is what
the oracle is asked.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import libgraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))
import birthplaces
import ingest

ARTICLES = ["France", "Europe", "Paris", "Lyon", "Marseille", "Ottawa",
            "Canada", "Ontario", "Toronto", "Alice", "Bob", "Carol", "Dave",
            "Erin"]

FACTS = [
    # Three infoboxes each naming France and Canada, which is exactly
    # TYPE_FLOOR - so those two are countries and nothing else is. Every climb
    # below depends on it.
    ("Paris", "country", "France"),
    ("Lyon", "country", "France"),
    ("Marseille", "country", "France"),
    ("Ottawa", "country", "Canada"),
    ("Ontario", "country", "Canada"),
    ("Toronto", "country", "Canada"),
    ("France", "region", "Europe"),
    ("Alice", "birth_place", "Paris"),
    ("Bob", "birth_place", "Lyon"),
    ("Carol", "birth_place", "Ottawa"),
]

LEADS = {
    "Alice": "Alice (born 1 May 1931 in Paris) was a French writer.",
    "Bob": "Bob was a chemist born in Lyon. He later moved to Paris.",
    "Carol": "Carol was a singer born in Ottawa, Canada. She began in 1970.",
    # No birthplace in the infobox; the lead has it.
    "Dave": "Dave (born 1940) was a poet born in Paris. He wrote sonnets.",
    # Nothing to find.
    "Erin": "Erin is a musician who plays the cello.",
}


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(ingest._schema())
    conn.executemany(
        "INSERT INTO article (source, title, lead) VALUES ('w', ?, ?)",
        [(a, LEADS.get(a, "")) for a in ARTICLES])
    conn.executemany(
        "INSERT OR REPLACE INTO fact (source, subject, property, ordinal, "
        "value, kind, num) VALUES ('w', ?, ?, 0, ?, 'text', NULL)", FACTS)
    conn.executemany(
        "INSERT INTO property (source, name, uses, subjects, relation) "
        "VALUES ('w', ?, ?, ?, ?)",
        [("country", 6, 6, "located_in"), ("region", 1, 1, "located_in"),
         ("birth_place", 3, 3, "born_in")])
    # Dave is a person by category alone, which is the case the infobox misses.
    conn.execute("INSERT INTO category (source, title, name) "
                 "VALUES ('w', 'Dave', '1940 births')")
    conn.execute("INSERT INTO category (source, title, name) "
                 "VALUES ('w', 'Erin', 'Living people')")
    libgraph.build(conn, "w")
    return conn


# --- the extraction -----------------------------------------------------------


@pytest.mark.parametrize("lead,expected", [
    ("X (born 1 May 1931 in Paris) was a writer.", "Paris"),
    ("X was born in Lyon.", "Lyon"),
    ("X was born in Ottawa, Canada. She began in 1970.", "Ottawa, Canada"),
    ("X was a poet. He was born in Paris and died in Lyon.", "Paris"),
    # The sentence boundary is the guard: a birth year followed by a move is
    # the shape that would otherwise read the second place as the first.
    ("X was born in 1931. He moved to Paris.", None),
    ("X is a musician who plays the cello.", None),
    # A lower-case word is not a place name, so nothing is taken.
    ("X was born in a small village.", None),
])
def test_what_the_pattern_takes(lead, expected):
    assert birthplaces.by_regex("X", lead).place == expected


def test_a_trailing_pronoun_is_not_part_of_the_place():
    """Leads run on, and the pattern spans up to sixty characters to reach
    "born ... in X". Without the trim it takes the next clause with it."""
    got = birthplaces.by_regex("X", "X was born in Ottawa. She sang.")
    assert got.place == "Ottawa"


def test_the_resolver_is_the_graph_s_own(db):
    """Not a titles lookup of its own. A first attempt here split on commas
    tail-first and turned `New York City, New` into `New`, a real article
    about the word."""
    resolve = birthplaces.resolver(db, "w")
    assert resolve("Ottawa, Canada") == "Ottawa"
    assert resolve("Nowhere At All") is None


# --- who gets asked -----------------------------------------------------------


def test_only_people_with_no_birthplace_are_asked(db):
    """Alice, Bob and Carol have one already; Erin is a person without one and
    Dave is a person the infobox never mentions."""
    missing = dict(birthplaces.people_missing_birthplace(db, "w"))
    assert set(missing) == {"Dave", "Erin"}


def test_a_place_is_not_a_person(db):
    assert "Paris" not in dict(birthplaces.people_missing_birthplace(db, "w"))


# --- the scoring --------------------------------------------------------------


def test_agreement_and_country_precision_are_different_numbers(db):
    """The finding this file exists to keep. Carol's infobox says Ottawa and
    her lead says "Ottawa, Canada", which resolves to Ottawa - agreement. Make
    the lead say Ontario instead and agreement fails while the country is
    still right, which is what the oracle is asked."""
    db.execute("UPDATE article SET lead = ? WHERE title = 'Carol'",
               ("Carol was a singer born in Ontario. She began in 1970.",))
    score = birthplaces.evaluate(db, "w", "regex")

    assert score.agreed == 2                     # Alice and Bob, not Carol
    assert score.agreement == pytest.approx(2 / 3)

    # Carol is the one the two measures disagree about: Ontario is not Ottawa,
    # and both are in Canada, which is what was asked.
    assert score.same_country == 3
    assert score.wrong_country == 0
    assert score.country_precision == 1.0


def test_a_genuinely_wrong_country_is_counted_wrong(db):
    """The correction must not excuse everything: Alice's lead saying Ottawa
    when her infobox says Paris is not a difference of granularity."""
    db.execute("UPDATE article SET lead = ? WHERE title = 'Alice'",
               ("Alice was a writer born in Ottawa. She moved later.",))
    score = birthplaces.evaluate(db, "w", "regex")
    assert score.wrong_country == 1
    assert score.country_precision < 1.0


# --- the provenance -----------------------------------------------------------


def test_what_is_extracted_never_reaches_fact_or_edge(db):
    """The constraint the whole table exists for. `fact` is what the
    encyclopedia tabulated and stays that way; anything read out of prose is
    somewhere else, tagged with what read it."""
    facts_before = db.execute("SELECT COUNT(*) FROM fact").fetchone()[0]
    edges_before = db.execute("SELECT COUNT(*) FROM edge").fetchone()[0]

    missing = birthplaces.people_missing_birthplace(db, "w")
    found = birthplaces.extract_all(db, "w", "regex", missing)
    birthplaces.write(db, "w", "regex", found)

    assert db.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == facts_before
    assert db.execute("SELECT COUNT(*) FROM edge").fetchone()[0] == edges_before
    assert db.execute(
        "SELECT subject, object, method FROM derived").fetchall() == [
        ("Dave", "Paris", "regex")]


def test_two_methods_can_disagree_and_both_be_kept(db):
    """`method` is in the primary key so one extractor can be measured against
    another. Overwriting would make that impossible and look like a fix."""
    birthplaces.write(db, "w", "regex",
                      [birthplaces.Extraction("Dave", "Paris", "Paris")])
    birthplaces.write(db, "w", "other",
                      [birthplaces.Extraction("Dave", "Lyon", "Lyon")])
    rows = db.execute("SELECT method, object FROM derived "
                      "WHERE subject = 'Dave' ORDER BY method").fetchall()
    assert rows == [("other", "Lyon"), ("regex", "Paris")]


def test_rewriting_one_method_leaves_the_other_alone(db):
    birthplaces.write(db, "w", "other",
                      [birthplaces.Extraction("Dave", "Lyon", "Lyon")])
    birthplaces.write(db, "w", "regex",
                      [birthplaces.Extraction("Dave", "Paris", "Paris")])
    birthplaces.write(db, "w", "regex",
                      [birthplaces.Extraction("Dave", "Ottawa", "Ottawa")])
    rows = db.execute("SELECT method, object FROM derived "
                      "WHERE subject = 'Dave' ORDER BY method").fetchall()
    assert rows == [("other", "Lyon"), ("regex", "Ottawa")]
