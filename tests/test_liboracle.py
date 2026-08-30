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


def oracle(db, relation_answer, monkeypatch, titles=None, records=True):
    monkeypatch.setattr(
        liboracle.Oracle, "relation",
        lambda self, q: relation_answer.split() if relation_answer else None)
    return liboracle.Oracle(
        db, relations=FakeRelations(relation_answer), records=records,
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
    o = oracle(db, "capital_is", monkeypatch, records=False)
    r = o.ask("what is the capital of jane austen")   # nothing has a capital
    assert r.kind == liboracle.SEARCH
    assert "no record" in liboracle.speak(r)


def test_no_fact_prefers_the_record_to_the_article(db, monkeypatch):
    """The subject resolved and only the relation did not, so say so.

    An article is prose about whatever the index liked for the whole question.
    A record is every edge the graph holds about the person the question named
    - a worse answer than a fact and a better failure than a paragraph, and
    the one fallback that cannot be wrong about anything.
    """
    o = oracle(db, "capital_is", monkeypatch)
    r = o.ask("what is the capital of jane austen")
    assert r.kind == liboracle.RECORD
    assert r.subject == "Jane Austen"
    assert ("born_in", "Steventon") in r.held
    said = liboracle.speak(r)
    assert "Steventon" in said and "Jane Austen" in said


def test_a_record_does_not_say_a_parent_twice(db, monkeypatch):
    """`child_of` duplicates `father_is` in the graph on purpose, and a listing
    that prints both says everyone's parents twice."""
    db.execute("INSERT INTO edge (source, subject, relation, object) "
               "VALUES ('simplewiki', 'Jane Austen', 'child_of', 'Someone')")
    o = oracle(db, "capital_is", monkeypatch)
    assert "Someone" not in liboracle.speak(o.ask("what is the capital of jane austen"))


def test_with_no_relation_model_there_is_nothing_to_fall_back_from(db):
    """A record presupposes the graph was asked something and said no.

    With no classifier nothing was ever asked, so that machine is a search
    engine and should answer like one however many edges the subject has.
    """
    o = liboracle.Oracle(db, relations=None,
                         search=FakeSearch(["Jane Austen"]))
    assert o.ask("where was jane austen born").kind == liboracle.SEARCH


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
    # `rank` rather than `classify`: the oracle reads the ordered choices now,
    # so that it can fall back to the runner-up, and lowercases whichever of
    # them it walks. Patching the function it no longer calls would leave this
    # passing against nothing.
    monkeypatch.setattr(libinfer, "rank",
                        lambda *a, **k: [("BORN_IN", 100)])
    o = liboracle.Oracle(db, relations=Uppercase(),
                         search=FakeSearch(["Jane Austen"]))
    assert o.ask("where was jane austen born").value == "Steventon"


def test_an_unreadable_relation_still_speaks(db, monkeypatch):
    """A relation with no phrasing must degrade, not crash."""
    r = liboracle.Response(liboracle.PARTIAL, said="Steventon",
                           missing="some_new_relation")
    assert "any more than that" in liboracle.speak(r)


# --- the runner-up ------------------------------------------------------------
#
# `backoff` answers more questions and answers more of them wrongly, which is a
# trade rather than an improvement. These pin the trade: that it is off unless
# asked for, that the margin gates it, and that an answer reached this way does
# not sound like a fact.


def _two_choices(monkeypatch, first: str, second: str,
                 gap: int = 10) -> None:
    """A phrasebook that wants `first` and would settle for `second`."""
    import libinfer
    monkeypatch.setattr(libinfer, "rank",
                        lambda *a, **k: [(first, 100), (second, 100 - gap)])


class _Model:
    accum_bits = 24
    phrases = ("A", "B")


def test_the_runner_up_is_not_consulted_unless_asked_for(db, monkeypatch):
    """The default is the shipped behaviour, and the shipped behaviour is that
    a path with no edge goes to search rather than to a second guess."""
    _two_choices(monkeypatch, "NO_SUCH_RELATION", "BORN_IN")
    o = liboracle.Oracle(db, relations=_Model(), records=False,
                         search=FakeSearch(["Jane Austen"]))
    assert o.ask("where was jane austen born").kind == liboracle.SEARCH


def test_the_runner_up_answers_when_the_first_path_has_no_edge(db, monkeypatch):
    _two_choices(monkeypatch, "NO_SUCH_RELATION", "BORN_IN")
    o = liboracle.Oracle(db, relations=_Model(), backoff=25,
                         search=FakeSearch(["Jane Austen"]))
    response = o.ask("where was jane austen born")
    assert response.value == "Steventon"
    assert response.second_choice


def test_a_confident_first_choice_is_left_as_a_gap(db, monkeypatch):
    """The margin is the whole point of gating rather than flagging.

    A first choice that wins by a mile and still finds nothing is more likely a
    hole in the corpus than a misroute, and answering it from the runner-up
    trades a reportable gap for a fluent answer to a different question.
    """
    _two_choices(monkeypatch, "NO_SUCH_RELATION", "BORN_IN", gap=90)
    o = liboracle.Oracle(db, relations=_Model(), backoff=25, records=False,
                         search=FakeSearch(["Jane Austen"]))
    assert o.ask("where was jane austen born").kind == liboracle.SEARCH


def test_a_second_choice_answer_does_not_sound_like_a_fact(db):
    plain = liboracle.Response(liboracle.FACT, value="Steventon")
    hedged = liboracle.Response(liboracle.FACT, value="Steventon",
                                second_choice=True)
    assert liboracle.speak(plain) == "Steventon."
    assert liboracle.speak(hedged) != liboracle.speak(plain)
    assert "Steventon" in liboracle.speak(hedged)


# --- the voice ----------------------------------------------------------------


def test_a_fact_is_said_by_the_path_that_found_it():
    r = liboracle.Response(liboracle.FACT, value="Larry O. Wilson",
                           relations=["father_is"], kind_of="woman")
    assert liboracle.speak(r) == "Her father is Larry O. Wilson."


def test_a_path_with_no_sentence_still_says_the_value():
    """A relation added to the corpus and not to `SAYS` must degrade to what
    the machine said before `SAYS` existed, not to a KeyError."""
    r = liboracle.Response(liboracle.FACT, value="Somewhere",
                           relations=["a_brand_new_relation"])
    assert liboracle.speak(r) == "Somewhere."


def test_an_untyped_subject_is_named_rather_than_pronouned():
    """Most corpora do not say - `simplewiki` types nobody - so this is the
    ordinary case, and "They was born in Steventon" is what makes it worth
    getting right. A name agrees with every verb these templates use.
    """
    r = liboracle.Response(liboracle.FACT, value="Larry O. Wilson",
                           subject="Amanda M. Wilson", relations=["father_is"])
    assert liboracle.speak(r) == "Amanda M. Wilson's father is Larry O. Wilson."
    born = liboracle.Response(liboracle.FACT, value="Steventon",
                              subject="Jane Austen", relations=["born_in"])
    assert liboracle.speak(born) == "Jane Austen was born in Steventon."


def test_a_hedge_outranks_a_sentence():
    """A second-choice answer must not get the fluent register. Dressing up an
    answer to a question nobody asked is the failure, not the fix."""
    r = liboracle.Response(liboracle.FACT, value="Third Shift",
                           relations=["shift_is"], kind_of="man",
                           second_choice=True)
    assert liboracle.speak(r) == "Third Shift, if I have your meaning."


def test_the_pronoun_comes_from_a_type_and_not_from_a_fact(db, monkeypatch):
    """Types are on the card and facts are not, so a pronoun read from `sex`
    would be a register the eZ80 could never speak in."""
    db.execute("INSERT INTO entity_type (source, kind, entity) "
               "VALUES ('simplewiki', 'woman', 'Jane Austen')")
    o = oracle(db, "born_in", monkeypatch)
    assert o.pronoun("Jane Austen") == "woman"
    assert o.pronoun("Steventon") is None
    assert liboracle.speak(o.ask("where was jane austen born")).startswith("She")


# --- two subjects -------------------------------------------------------------
#
# `libgraph.common` could always answer a question about two people and nothing
# could ask it one, because the pipeline resolved a single document. These pin
# the second resolution, which is the part that was missing.


def test_two_names_resolve_to_two_subjects(db, monkeypatch):
    o = oracle(db, "born_in", monkeypatch)
    assert o.subjects("is jane austen related to steventon") == [
        "Jane Austen", "Steventon"]


def test_one_name_does_not_acquire_a_second_subject(db, monkeypatch):
    """BM25 does not decline. Given `where was born` it returns whatever those
    words touch, so a second subject is kept only if its name is still in what
    is left of the question."""
    o = oracle(db, "born_in", monkeypatch)
    assert o.subjects("where was jane austen born") == ["Jane Austen"]


def test_the_second_of_two_names_keeps_the_surname_they_share():
    """The case that decided `residual` against `mask`.

    Masking removes every copy of the subject's words, so a question naming two
    Wongs loses both surnames and the second search goes looking for a man
    called `corey w`. Removing one copy each leaves the second Wong intact -
    and 2,264 of the silo's ten thousand share a first and last name.
    """
    q = "is alexander e wong related to corey w wong"
    assert liboracle.mask(q, "Alexander E. Wong") == "is related to corey w"
    assert liboracle.residual(q, "Alexander E. Wong") == (
        "is related to corey w wong")


def test_a_pair_question_says_yes_or_no_and_not_a_bare_value(db):
    yes = liboracle.Response(liboracle.FACT, value="First Crew 1", pair=True)
    no = liboracle.Response(liboracle.FACT, value=None, pair=True)
    assert liboracle.speak(yes) == "Yes. First Crew 1."
    assert "Not by" in liboracle.speak(no)


def test_a_pair_path_with_one_name_falls_back_rather_than_inventing(db, monkeypatch):
    """A pair question naming one person is a misroute, and the machine should
    fail it the ordinary way rather than pick somebody to be wrong about."""
    o = oracle(db, "shared_born_in", monkeypatch)
    assert o.shared("where was jane austen born", "shared_born_in") is None


# --- masking ------------------------------------------------------------------
#
# `mask` is measured and unused - see its docstring for what it was worth.
# It is tested because the measurement it supports has to stay reproducible,
# and because two of these cases are the reasons it is written the way it is.


def test_masking_takes_the_name_out_and_leaves_the_question():
    assert liboracle.mask("where was jane austen born",
                          "Jane Austen") == "where was born"


def test_masking_strips_the_possessive_that_hides_the_name():
    """Without this rule `wong's` survives, which is most of the name and all
    of the problem the masking was for."""
    assert liboracle.mask("who is alexander e wong's father",
                          "Alexander E. Wong") == "who is father"


def test_masking_removes_a_title_word_even_when_it_is_a_frame_word():
    """The known hazard, pinned rather than papered over: a title made of
    ordinary words takes those words out of the question with it, and what is
    left is not a question any more."""
    assert liboracle.mask("who wrote born in the usa", "Born in the USA") == "who wrote"


def test_masking_a_name_that_is_not_there_changes_nothing():
    assert liboracle.mask("what is a black hole", "Jane Austen") == "what is a black hole"
