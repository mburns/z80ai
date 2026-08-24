"""Walking the fact graph.

The measurement behind this module: `birth_place -> country` completes for
1.7% of subjects, and the same chain written as `born_in -> located_in`
completes for 40.7%. The graph was not sparse - a city records its country as
`subdivision_name`, so a chain asking for `country` failed on data that plainly
contained the answer.

These pin the collapsing, the traversal, and the part an oracle needs most:
that a broken chain says *where* it broke rather than only that it did.
"""

from __future__ import annotations

import sqlite3

import pytest

import libgraph


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE article (source TEXT, title TEXT, lead TEXT);
        CREATE TABLE redirect (source TEXT, title TEXT, target TEXT);
        CREATE TABLE fact (source TEXT, subject TEXT, property TEXT, value TEXT);
    """)
    conn.executescript(libgraph.schema(strict=False))
    return conn


def load(db, articles, facts, redirects=()):
    db.executemany("INSERT INTO article VALUES ('w', ?, '')",
                   [(a,) for a in articles])
    db.executemany("INSERT INTO redirect VALUES ('w', ?, ?)", redirects)
    db.executemany("INSERT INTO fact VALUES ('w', ?, ?, ?)", facts)
    libgraph.build(db, "w")


# --- the collapsing -----------------------------------------------------------


def test_synonymous_properties_become_one_relation(db):
    """The finding this module exists for: a city says `subdivision_name`
    where a country says `country`, and a chain asking for either fails."""
    load(db,
         ["Alan Turing", "London", "Paris", "England", "France"],
         [("Alan Turing", "birth_place", "London"),
          ("London", "subdivision_name", "England"),   # a city's vocabulary
          ("Paris", "country", "France")])             # a different one

    assert libgraph.follow(db, "w", "London", ["located_in"]).value == "England"
    assert libgraph.follow(db, "w", "Paris", ["located_in"]).value == "France"


def test_the_most_specific_field_wins(db):
    """A page carrying both must contribute one edge, not two that disagree."""
    load(db, ["Berlin", "Germany", "Brandenburg"],
         [("Berlin", "country", "Germany"),
          ("Berlin", "subdivision_name", "Brandenburg")])
    edges = db.execute("SELECT object FROM edge WHERE subject='Berlin' "
                       "AND relation='located_in'").fetchall()
    assert edges == [("Germany",)]      # `country` outranks `subdivision_name`


def test_a_value_naming_no_article_is_dropped(db):
    """A hop can only continue to something the corpus actually holds."""
    load(db, ["Alan Turing"], [("Alan Turing", "birth_place", "Nowhere-at-all")])
    assert db.execute("SELECT COUNT(*) FROM edge").fetchone()[0] == 0


def test_a_value_resolves_through_a_redirect(db):
    load(db, ["Alan Turing", "United Kingdom"],
         [("Alan Turing", "birth_place", "Britain")],
         [("Britain", "United Kingdom")])
    assert libgraph.follow(db, "w", "Alan Turing", ["born_in"]).value == \
        "United Kingdom"


def test_a_trailing_qualifier_is_dropped_to_find_the_article(db):
    """"Edinburgh, Scotland" is a sentence about a place, not its title."""
    load(db, ["Bell", "Edinburgh"],
         [("Bell", "birth_place", "Edinburgh, Scotland")])
    assert libgraph.follow(db, "w", "Bell", ["born_in"]).value == "Edinburgh"


def test_a_middle_segment_is_taken_when_the_narrowest_is_not_an_article(db):
    """Jane Austen's birthplace, and 11% of all edges, hang on this.

    The corpus has no article on Steventon Rectory, so an edge that insists on
    the most specific segment is no edge at all. Falling through to Hampshire
    trades precision for an answer that is still true.
    """
    load(db, ["Jane Austen", "Hampshire"],
         [("Jane Austen", "birth_place",
           "Steventon Rectory, Hampshire, England")])
    assert libgraph.follow(db, "w", "Jane Austen", ["born_in"]).value == "Hampshire"


def test_the_narrowest_segment_still_wins_when_the_corpus_has_it(db):
    """Falling through must be a fallback, not the rule - precision first."""
    load(db, ["Jane Austen", "Steventon", "Hampshire"],
         [("Jane Austen", "birth_place", "Steventon, Hampshire")])
    assert libgraph.follow(db, "w", "Jane Austen", ["born_in"]).value == "Steventon"


# --- traversal ----------------------------------------------------------------


def chain_fixture(db):
    load(db,
         ["Alan Turing", "London", "England", "United Kingdom"],
         [("Alan Turing", "birth_place", "London"),
          ("London", "subdivision_name", "England"),
          ("England", "country", "United Kingdom")])


def test_a_two_hop_chain_completes(db):
    chain_fixture(db)
    answer = libgraph.follow(db, "w", "Alan Turing", ["born_in", "located_in"])
    assert answer.value == "England"
    assert answer.complete


def test_a_three_hop_chain_completes(db):
    chain_fixture(db)
    answer = libgraph.follow(
        db, "w", "Alan Turing", ["born_in", "located_in", "located_in"])
    assert answer.value == "United Kingdom"
    assert answer.path == ["Alan Turing", "London", "England", "United Kingdom"]


def test_a_broken_chain_reports_where_it_broke(db):
    """The difference between "I don't know" and "I know where he was born,
    but not what country that is in" - which is the whole voice of an oracle
    that is wrong half the time."""
    load(db, ["Alan Turing", "London"],
         [("Alan Turing", "birth_place", "London")])

    answer = libgraph.follow(db, "w", "Alan Turing", ["born_in", "located_in"])
    assert not answer.complete
    assert answer.path == ["Alan Turing", "London"]     # got one hop in
    assert answer.missing == "located_in"


def test_an_unknown_subject_stops_at_the_first_hop(db):
    chain_fixture(db)
    answer = libgraph.follow(db, "w", "Nobody", ["born_in"])
    assert answer.path == ["Nobody"]
    assert answer.missing == "born_in"


# --- inverses and counting ----------------------------------------------------


def test_the_graph_walks_backwards(db):
    """"Who was born in London" from the rows saying where people were born."""
    load(db, ["Alan Turing", "Ada Lovelace", "London"],
         [("Alan Turing", "birth_place", "London"),
          ("Ada Lovelace", "birth_place", "London")])
    assert sorted(libgraph.inverse(db, "w", "London", "born_in")) == \
        ["Ada Lovelace", "Alan Turing"]


def test_counting_is_free_once_the_inverse_index_exists(db):
    load(db, ["Alan Turing", "Ada Lovelace", "Charles Babbage", "London"],
         [("Alan Turing", "birth_place", "London"),
          ("Ada Lovelace", "birth_place", "London"),
          ("Charles Babbage", "birth_place", "London")])
    assert libgraph.count(db, "w", "London", "born_in") == 3
    assert libgraph.count(db, "w", "Paris", "born_in") == 0


def test_rebuilding_replaces_rather_than_accumulates(db):
    chain_fixture(db)
    before = db.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    libgraph.build(db, "w")
    assert db.execute("SELECT COUNT(*) FROM edge").fetchone()[0] == before


def test_sources_do_not_leak_into_each_other(db):
    """MetaQA and Wikipedia share the table; a walk must not cross between."""
    chain_fixture(db)
    db.execute("INSERT INTO edge VALUES ('metaqa', 'Alan Turing', 'born_in', "
               "'Somewhere Else')")
    assert libgraph.follow(db, "w", "Alan Turing", ["born_in"]).value == "London"
    assert libgraph.follow(db, "metaqa", "Alan Turing", ["born_in"]).value == \
        "Somewhere Else"
