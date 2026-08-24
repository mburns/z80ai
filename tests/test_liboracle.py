"""The oracle: a question in, a fact out, or an account of why not.

Most of these are about the failures, because the failures are the product.
Two-hop chains complete about 45% of the time over this corpus, so what the
machine says about the other 55% is not an error path - it is most of what
anyone will hear from it.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import libgraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))
import ingest

import liboracle


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    # The real schema, so these cannot pass against a `fact` table the ingest
    # has since changed - which is exactly what a hand-written copy allowed.
    conn.executescript(ingest._schema())
    conn.executemany("INSERT INTO article (source, title, lead) "
                     "VALUES ('simplewiki', ?, '')", [
        ("Jane Austen",), ("Steventon",), ("Hampshire",), ("England",),
        ("Pride and Prejudice",)])
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('simplewiki', ?, ?, 0, ?, 'text', NULL)", [
        ("Jane Austen", "birth_place", "Steventon"),
        ("Steventon", "subdivision_name", "Hampshire"),
        ("Hampshire", "country", "England"),
        ("Pride and Prejudice", "author", "Jane Austen"),
    ])
    libgraph.build(conn, "simplewiki")
    return conn


class FakeRelations:
    """Stands in for the trained classifier, so these test the walk."""

    accum_bits = 24

    def __init__(self, answer):
        self.answer = answer


class FakeSearch:
    """Stands in for the BM25 index: returns whichever title is named."""

    def __init__(self, titles):
        self.titles = titles

    def search(self, question, top=1):
        for i, title in enumerate(self.titles):
            if title.lower() in question.lower():
                return [(i, 1)]
        return []

    def article(self, doc):
        return self.titles[doc], f"{self.titles[doc]} is a thing."


def oracle(db, relation_answer, monkeypatch, titles=None):
    monkeypatch.setattr(
        liboracle.Oracle, "relation",
        lambda self, q: relation_answer.split() if relation_answer else None)
    return liboracle.Oracle(
        db, relations=FakeRelations(relation_answer),
        search=FakeSearch(titles or ["Jane Austen", "Steventon", "Hampshire",
                                     "England", "Pride and Prejudice"]))


# --- answering ----------------------------------------------------------------


def test_a_one_hop_question_is_answered_from_a_fact(db, monkeypatch):
    o = oracle(db, "born_in", monkeypatch)
    r = o.ask("where was jane austen born")
    assert r.kind == liboracle.FACT
    assert r.value == "Steventon"
    assert r.answered


def test_a_two_hop_question_walks(db, monkeypatch):
    o = oracle(db, "born_in located_in", monkeypatch)
    r = o.ask("what county was jane austen born in")
    assert r.value == "Hampshire"
    assert r.path == ["Jane Austen", "Steventon", "Hampshire"]


def test_a_three_hop_question_walks(db, monkeypatch):
    o = oracle(db, "born_in located_in located_in", monkeypatch)
    r = o.ask("what country was jane austen born in")
    assert r.value == "England"


def test_an_inverse_question_walks_backwards(db, monkeypatch):
    """"Who wrote Pride and Prejudice" is the author edge read the other way."""
    o = oracle(db, "created_by_of", monkeypatch)
    r = o.ask("what did jane austen write")
    assert r.kind == liboracle.FACT
    assert "Pride and Prejudice" in r.value


# --- the failures, which are the product --------------------------------------


def test_a_chain_that_breaks_says_what_it_did_learn(db, monkeypatch):
    """The difference between a machine with gaps and one that is unreliable."""
    db.execute("DELETE FROM edge WHERE subject = 'Steventon'")
    o = oracle(db, "born_in located_in", monkeypatch)
    r = o.ask("what county was jane austen born in")

    assert r.kind == liboracle.PARTIAL
    assert r.said == "Steventon"           # got one hop in
    assert r.missing == "located_in"
    assert "Steventon" in liboracle.speak(r)
    assert "does not record" in liboracle.speak(r)


def test_a_climb_that_runs_out_names_the_type_it_wanted(db, monkeypatch):
    """"...does not record what country that is" - not "what contains it".

    The climb asked for a type, so the apology has to name the type, or the
    machine sounds like it does not know what it was looking for.
    """
    o = oracle(db, "born_in in_country", monkeypatch)
    r = o.ask("what country was jane austen born in")
    assert r.kind == liboracle.PARTIAL
    assert "does not record what country that is" in liboracle.speak(r)


def test_no_fact_falls_back_to_the_article_and_says_so(db, monkeypatch):
    o = oracle(db, "capital_is", monkeypatch)     # nothing has a capital
    r = o.ask("what is the capital of jane austen")
    assert r.kind == liboracle.SEARCH
    assert "no record" in liboracle.speak(r)


def test_an_unknown_subject_admits_it(db, monkeypatch):
    o = oracle(db, "born_in", monkeypatch)
    r = o.ask("where was nobody at all born")
    assert r.kind == liboracle.UNKNOWN
    assert liboracle.speak(r) == "The archive holds nothing on that subject."


def test_without_a_relation_model_it_is_only_a_search_engine(db):
    """Which is what it was before any of this - and still a useful fallback."""
    o = liboracle.Oracle(db, relations=None,
                         search=FakeSearch(["Jane Austen"]))
    r = o.ask("where was jane austen born")
    assert r.kind == liboracle.SEARCH


# --- the voice ----------------------------------------------------------------


def test_every_kind_of_response_can_be_spoken():
    """A response with no phrasing would surface as a traceback mid-scene."""
    for kind in (liboracle.FACT, liboracle.PARTIAL, liboracle.SEARCH,
                 liboracle.UNKNOWN):
        said = liboracle.speak(liboracle.Response(
            kind, value="a value", said="somewhere", missing="located_in"))
        assert said and "{" not in said


def test_an_uppercase_phrasebook_still_finds_lowercase_edges(db, monkeypatch):
    """The Z80 charset is uppercase; the graph is not.

    This failed silently before it was fixed: every lookup found no edge, so
    the oracle fell back to search and looked exactly like a corpus gap rather
    than a bug. Worth a test precisely because nothing else would catch it.
    """
    class Uppercase:
        accum_bits = 24
        phrases = ("BORN_IN",)

    import libinfer
    monkeypatch.setattr(libinfer, "classify",
                        lambda *a, **k: "BORN_IN")
    o = liboracle.Oracle(db, relations=Uppercase(),
                         search=FakeSearch(["Jane Austen"]))
    assert o.ask("where was jane austen born").value == "Steventon"


def test_an_unreadable_relation_still_speaks(db, monkeypatch):
    """A relation with no phrasing must degrade, not crash."""
    r = liboracle.Response(liboracle.PARTIAL, said="Steventon",
                           missing="some_new_relation")
    assert "any more than that" in liboracle.speak(r)
