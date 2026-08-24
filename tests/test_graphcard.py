"""The fact graph as a card file.

Two things are being pinned. That the layout round-trips - a hop is a binary
search over fixed-width records, so an off-by-one in the record size reads
plausible garbage rather than failing. And that `CardGraph` answers what
`libgraph` answers, because the card is meant to be the same graph and the only
way to know is to ask both.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))

import ingest
import libgraph
import libgraphcard

TITLES = ["Jane Austen", "Steventon", "Hampshire", "England",
          "Pride and Prejudice", "Marie Curie", "Warsaw", "Poland"]
DOC = {t: i for i, t in enumerate(TITLES)}


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(ingest._schema())
    conn.executemany("INSERT INTO article (source, title, lead) "
                     "VALUES ('w', ?, '')", [(t,) for t in TITLES])
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('w', ?, ?, 0, ?, 'text', NULL)", [
            ("Jane Austen", "birth_place", "Steventon"),
            ("Steventon", "county", "Hampshire"),
            ("Hampshire", "country", "England"),
            ("Pride and Prejudice", "author", "Jane Austen"),
            ("Marie Curie", "birth_place", "Warsaw"),
            ("Warsaw", "country", "Poland"),
        ])
    # Three independent infoboxes calling a thing a country is what makes it
    # one, so the fixture has to say so three times over.
    fillers = [(f"{name} filler {i}", name)
               for name in ("England", "Poland") for i in range(3)]
    conn.executemany("INSERT INTO article (source, title, lead) "
                     "VALUES ('w', ?, '')", [(t,) for t, _ in fillers])
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('w', ?, 'country', 0, ?, 'text', NULL)", fillers)
    libgraph.build(conn, "w")
    return conn


@pytest.fixture
def card(db, tmp_path):
    relations = sorted({r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = 'w'")})
    rid = {name: i for i, name in enumerate(relations)}
    titles = [t for (t,) in db.execute(
        "SELECT title FROM article WHERE source = 'w' ORDER BY id")]
    doc = {t: i for i, t in enumerate(titles)}

    edges = [(doc[s], rid[r], doc[o]) for s, r, o in db.execute(
        "SELECT subject, relation, object FROM edge WHERE source = 'w'")
        if s in doc and o in doc]
    types: dict[str, list[int]] = {}
    for kind, entity in db.execute(
            "SELECT kind, entity FROM entity_type WHERE source = 'w'"):
        if entity in doc:
            types.setdefault(kind, []).append(doc[entity])

    graph = libgraphcard.build(titles, edges, relations, types, paths=[])
    path = tmp_path / "WIKI.GRF"
    libgraphcard.write(graph, path)
    return libgraphcard.CardGraph(path), doc, rid, titles


# --- the layout ---------------------------------------------------------------


def test_the_header_survives_the_round_trip(card):
    graph, _doc, _rid, titles = card
    assert graph.num_docs == len(titles)
    assert graph.digest == libgraphcard.corpus_digest(titles)
    assert graph.num_edges > 0


def test_a_hop_finds_the_same_object_libgraph_does(card, db):
    graph, doc, rid, _titles = card
    for subject, relation, obj in db.execute(
            "SELECT subject, relation, object FROM edge WHERE source = 'w'"):
        assert doc[obj] in graph.objects(doc[subject], rid[relation])


def test_a_hop_that_has_no_edge_finds_nothing(card):
    graph, doc, rid, _titles = card
    assert graph.objects(doc["England"], rid["born_in"]) == []


def test_the_reverse_table_answers_the_question_backwards(card):
    """"Who was born in Warsaw" is the same row read from the other end."""
    graph, doc, rid, _titles = card
    assert graph.subjects(doc["Warsaw"], rid["born_in"]) == [doc["Marie Curie"]]
    assert graph.count(doc["Warsaw"], rid["born_in"]) == 1
    assert graph.count(doc["England"], rid["born_in"]) == 0


def test_a_type_is_a_membership_test(card):
    graph, doc, _rid, _titles = card
    assert graph.is_a(doc["England"], "country")
    assert graph.is_a(doc["Poland"], "country")
    assert not graph.is_a(doc["Steventon"], "country")


# --- the walk, against libgraph -----------------------------------------------


def test_a_climb_agrees_with_libgraph(card, db):
    """The card and the SQL are meant to be the same graph. Asking both is the
    only way to find out that they are."""
    graph, doc, rid, titles = card
    # Two steps, as libgraph spells it: hop born_in, then climb located_in
    # until the value is a country.
    steps = [(rid["born_in"], libgraphcard.PLAIN),
             (rid["located_in"], graph.type_names.index("country"))]
    answer, _walked, missing = graph.follow(doc["Marie Curie"], steps)
    reference = libgraph.follow(db, "w", "Marie Curie", ["born_in", "in_country"])

    assert missing is None
    assert titles[answer] == reference.value == "Poland"


def test_a_climb_takes_more_than_one_hop_when_it_has_to(card, db):
    graph, doc, rid, titles = card
    steps = [(rid["born_in"], libgraphcard.PLAIN),
             (rid["located_in"], graph.type_names.index("country"))]
    answer, walked, _missing = graph.follow(doc["Jane Austen"], steps)
    reference = libgraph.follow(db, "w", "Jane Austen", ["born_in", "in_country"])

    assert titles[answer] == reference.value == "England"
    # Steventon -> Hampshire -> England: the hop count is the graph's business.
    assert [titles[d] for d in walked] == reference.path


def test_a_walk_that_stops_says_which_step_it_stopped_on(card):
    graph, doc, rid, _titles = card
    steps = [(rid["born_in"], libgraphcard.PLAIN),
             (rid["created_by"], libgraphcard.PLAIN)]
    answer, walked, missing = graph.follow(doc["Marie Curie"], steps)
    assert answer is None
    assert missing == 1                 # the first hop worked, the second did not
    assert len(walked) == 2


def test_a_climb_stops_where_it_starts_when_it_is_already_there(card):
    """A quarter of birthplaces in the real corpus are countries already, and
    a fixed hop count steps straight past the answer for all of them."""
    graph, doc, rid, _titles = card
    country = graph.type_names.index("country")

    answer, walked, missing = graph.follow(doc["Poland"], [(rid["located_in"],
                                                            country)])
    assert missing is None
    assert answer == doc["Poland"]
    assert walked == [doc["Poland"]]       # no wasted step


# --- the mismatch that has no other symptom -----------------------------------


def test_a_graph_built_for_another_corpus_is_refused(card):
    """Every id in a mismatched graph is still a valid article, so this check
    is the only thing between a wrong pair of files and a fluent wrong answer."""
    graph, _doc, _rid, titles = card
    graph.check(len(titles), libgraphcard.corpus_digest(titles))

    with pytest.raises(ValueError, match="different corpus"):
        graph.check(len(titles), libgraphcard.corpus_digest(titles[:-1]))
    with pytest.raises(ValueError, match="different corpus"):
        graph.check(len(titles) + 1, libgraphcard.corpus_digest(titles))


def test_the_digest_notices_a_reordering(card):
    """`--limit` reorders by notability rather than dropping the tail, so a
    digest over the count alone would miss the commonest mismatch."""
    graph, _doc, _rid, titles = card
    swapped = [titles[1], titles[0], *titles[2:]]
    assert libgraphcard.corpus_digest(swapped) != graph.digest
