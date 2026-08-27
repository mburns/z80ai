"""Importing Wikidata statements into the corpus that already has the articles.

The decisions worth testing here are the ones that decide whether a *wrong*
answer reaches a card: what happens when Wikidata and the encyclopedia disagree,
and what happens when Wikidata says two things at once.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

import conftest


@pytest.fixture(scope="module")
def wikidata(repo_root):
    return conftest.load_script(
        str(Path(repo_root) / "data" / "wikipedia" / "wikidata.py"), "wikidata")


@pytest.fixture(scope="module")
def ingest(repo_root):
    return conftest.load_script(
        str(Path(repo_root) / "data" / "wikipedia" / "ingest.py"), "wd_ingest")


# Carrollton is in Carroll County is in Mississippi is in the United States.
# The county has no article, which is the whole reason the chain is exported
# past the edge of the corpus.
CARROLLTON, CARROLL_COUNTY, MISSISSIPPI, USA = 1, 2, 3, 4
EVEREST, CHINA, NEPAL = 5, 6, 7
PARENTS = {CARROLLTON: {CARROLL_COUNTY}, CARROLL_COUNTY: {MISSISSIPPI},
           MISSISSIPPI: {USA}, EVEREST: {CHINA, NEPAL}}


# --- containment --------------------------------------------------------------


def test_containment_is_followed_through_places_with_no_article(wikidata):
    """The corpus has Carrollton and Mississippi and not the county between."""
    assert wikidata.inside(CARROLLTON, MISSISSIPPI, PARENTS)
    assert wikidata.inside(CARROLLTON, USA, PARENTS)


def test_containment_does_not_run_backwards(wikidata):
    assert not wikidata.inside(MISSISSIPPI, CARROLLTON, PARENTS)


def test_a_thing_is_not_inside_itself(wikidata):
    """Equal is not more specific, and a refinement rule that thought so would
    rewrite every agreeing row for no reason."""
    assert not wikidata.inside(MISSISSIPPI, MISSISSIPPI, PARENTS)


def test_a_cycle_terminates(wikidata):
    """Wikidata has containment cycles in it. A corpus build is not the place
    to find that out by running out of stack."""
    assert not wikidata.inside(1, 99, {1: {2}, 2: {3}, 3: {1}})


def test_containment_is_bounded(wikidata):
    """Past CHAIN_DEPTH the answer is 'not proven', not 'keep looking'."""
    deep = {i: {i + 1} for i in range(40)}
    assert not wikidata.inside(0, 39, deep)


# --- choosing between several values ------------------------------------------


def test_a_single_value_is_the_answer(wikidata):
    assert wikidata.choose({MISSISSIPPI}, PARENTS) == MISSISSIPPI


def test_nested_values_collapse_to_the_innermost(wikidata):
    """`Sialkot`, `Punjab Province` and `British Raj` are three depths of one
    answer, and the innermost is the answer."""
    assert wikidata.choose({CARROLLTON, MISSISSIPPI, USA}, PARENTS) == CARROLLTON


def test_values_that_do_not_nest_are_declined(wikidata):
    """Everest is in China and in Nepal. `derived` holds one object, and
    picking would put a fluent half-truth on a card with nothing to mark it."""
    assert wikidata.choose({CHINA, NEPAL}, PARENTS) is None


def test_unrelated_values_are_declined(wikidata):
    """A band's nine genres do not nest, so there is no non-arbitrary pick."""
    assert wikidata.choose({100, 200, 300}, {}) is None


# --- what the plan decides ----------------------------------------------------


def plan_for(wikidata, facts, existing, placed=None, parents=PARENTS):
    titles = {CARROLLTON: "Carrollton, Mississippi", MISSISSIPPI: "Mississippi",
              USA: "United States", EVEREST: "Mount Everest", CHINA: "China",
              NEPAL: "Nepal", CARROLL_COUNTY: "Carroll County"}
    return wikidata.build_plan(facts, parents, existing,
                               placed if placed is not None else set(titles),
                               titles)


def test_a_gap_is_filled(wikidata):
    plan = plan_for(wikidata, {19: {CARROLLTON: {MISSISSIPPI}}}, {})
    assert plan.rows == [("Carrollton, Mississippi", "born_in", "Mississippi")]
    assert plan.counts["born_in"]["gap"] == 1


def test_agreement_writes_nothing(wikidata):
    plan = plan_for(wikidata, {19: {CARROLLTON: {MISSISSIPPI}}},
                    {"born_in": {"Carrollton, Mississippi": {"Mississippi"}}})
    assert plan.rows == []
    assert plan.counts["born_in"]["agree"] == 1


def test_a_more_specific_value_refines_the_corpus(wikidata):
    """The corpus says Mississippi, Wikidata says Carrollton, Mississippi, and
    the second is provably inside the first. Both are true; one is an answer."""
    plan = plan_for(wikidata, {19: {EVEREST: {CARROLLTON}}},
                    {"born_in": {"Mount Everest": {"Mississippi"}}},
                    parents={**PARENTS, EVEREST: set()})
    assert plan.rows == [("Mount Everest", "born_in", "Carrollton, Mississippi")]
    assert plan.counts["born_in"]["refine"] == 1


def test_a_merely_different_value_leaves_the_corpus_alone(wikidata):
    """Nothing shows China is inside Mississippi, so the encyclopedia keeps its
    answer. This is the rule that stops a fluent wrong one landing."""
    plan = plan_for(wikidata, {19: {EVEREST: {CHINA}}},
                    {"born_in": {"Mount Everest": {"Mississippi"}}})
    assert plan.rows == []
    assert plan.counts["born_in"]["kept"] == 1


def test_a_country_is_not_taken_for_something_that_is_not_a_place(wikidata):
    """P17 on a place is where it is; on a language it is where it is spoken,
    which is how `English language` acquires ninety countries. Only a subject
    Wikidata puts administratively inside something is eligible."""
    facts = {17: {EVEREST: {CHINA}}}
    assert plan_for(wikidata, facts, {}, placed=set()).rows == []
    assert plan_for(wikidata, facts, {}, placed=set()
                    ).counts["located_in"]["untyped"] == 1
    assert plan_for(wikidata, facts, {}, placed={EVEREST}).rows == [
        ("Mount Everest", "located_in", "China")]


def test_country_and_admin_territory_are_one_question_asked_twice(wikidata):
    """Both land on `located_in`, so they are unioned before anything chooses
    between them - otherwise the same subject arrives twice and the second
    write silently replaces the first."""
    plan = plan_for(wikidata, {131: {CARROLLTON: {CARROLL_COUNTY}},
                               17: {CARROLLTON: {USA}}}, {})
    assert plan.rows == [
        ("Carrollton, Mississippi", "located_in", "Carroll County")]


def test_straddling_two_places_declines_rather_than_picking_one(wikidata):
    """Death Valley is in two counties and one country. The counties do not
    nest, so there is no innermost and nothing is written - the corpus keeps
    whatever it had and the card gains no half-truth."""
    inyo, san_bernardino = 20, 21
    parents = {inyo: {USA}, san_bernardino: {USA}}
    plan = plan_for(wikidata, {131: {EVEREST: {inyo, san_bernardino}},
                               17: {EVEREST: {USA}}}, {}, parents=parents)
    assert plan.rows == []
    assert plan.counts["located_in"]["declined"] == 1


def test_a_relation_that_is_not_containment_is_still_typed_freely(wikidata):
    """Only `country` needs the guard; a birthplace is a birthplace."""
    plan = plan_for(wikidata, {19: {EVEREST: {CHINA}}}, {}, placed=set())
    assert plan.rows == [("Mount Everest", "born_in", "China")]


# --- the file format ----------------------------------------------------------


def write_export(path: Path, body: str) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("# format\t1\n# dump\twikidata.lbdb\n" + body)
    return path


def test_chain_rows_are_read_apart_from_statements(wikidata, tmp_path):
    """A chain row is scaffolding for deciding containment, not a fact about
    the corpus, and nothing should mistake it for an edge worth carding."""
    path = write_export(tmp_path / "e.tsv.gz",
                        f"{CARROLLTON}\t19\t{MISSISSIPPI}\n"
                        f"{CARROLL_COUNTY}\t131\t{MISSISSIPPI}\tchain\n")
    rows, parents, header = wikidata.read_export(path)
    assert rows == [(CARROLLTON, 19, MISSISSIPPI)]
    assert parents[CARROLL_COUNTY] == {MISSISSIPPI}
    assert header["dump"] == "wikidata.lbdb"


def test_a_containment_statement_is_also_a_chain_link(wikidata, tmp_path):
    """It is written once and read as both, so an exported P131 about a corpus
    article does not have to be repeated to be walkable."""
    path = write_export(tmp_path / "e.tsv.gz",
                        f"{CARROLLTON}\t131\t{MISSISSIPPI}\n")
    rows, parents, _ = wikidata.read_export(path)
    assert rows == [(CARROLLTON, 131, MISSISSIPPI)]
    assert parents[CARROLLTON] == {MISSISSIPPI}


# --- writing ------------------------------------------------------------------


@pytest.fixture()
def corpus(ingest, tmp_path):
    db = ingest.connect(tmp_path / "wiki.db")
    db.execute("INSERT INTO article (source, title, lead) VALUES "
               "('simplewiki', 'Carrollton, Mississippi', '')")
    db.execute("INSERT INTO sitelink VALUES ('simplewiki', "
               "'Carrollton, Mississippi', 1)")
    db.commit()
    return db


def test_rows_land_in_derived_and_never_in_fact(wikidata, corpus):
    """A reader wanting only what the encyclopedia tabulated reads `fact` and
    never sees any of this."""
    plan = wikidata.Plan()
    plan.rows = [("Carrollton, Mississippi", "born_in", "Mississippi")]
    wikidata.write(corpus, "simplewiki", plan, Path("e.tsv.gz"), {"dump": "d"})

    assert corpus.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 0
    assert corpus.execute(
        "SELECT subject, relation, object, method FROM derived").fetchall() == [
        ("Carrollton, Mississippi", "born_in", "Mississippi", "wikidata")]


def test_a_second_import_replaces_the_first(wikidata, corpus):
    plan = wikidata.Plan()
    plan.rows = [("Carrollton, Mississippi", "born_in", "Mississippi")]
    wikidata.write(corpus, "simplewiki", plan, Path("e.tsv.gz"), {})
    plan.rows = [("Carrollton, Mississippi", "born_in", "United States")]
    wikidata.write(corpus, "simplewiki", plan, Path("e.tsv.gz"), {})
    assert corpus.execute("SELECT object FROM derived").fetchall() == [
        ("United States",)]


def test_another_method_survives_an_import(wikidata, corpus):
    """`method` is in the primary key so two extractors can disagree about the
    same person and both be kept - which is what makes one measurable against
    the other. An import must not clear the regex extractor's work."""
    corpus.execute("INSERT INTO derived VALUES ('simplewiki', "
                   "'Carrollton, Mississippi', 'born_in', 'Georgia', 'regex')")
    corpus.commit()
    plan = wikidata.Plan()
    plan.rows = [("Carrollton, Mississippi", "born_in", "Mississippi")]
    wikidata.write(corpus, "simplewiki", plan, Path("e.tsv.gz"), {})
    assert sorted(corpus.execute(
        "SELECT method, object FROM derived").fetchall()) == [
        ("regex", "Georgia"), ("wikidata", "Mississippi")]


def test_provenance_names_the_dump(wikidata, corpus):
    plan = wikidata.Plan()
    plan.rows = []
    wikidata.write(corpus, "simplewiki", plan, Path("wikidata.tsv.gz"),
                   {"dump": "wikidata-20260401.lbdb"})
    meta = dict(corpus.execute("SELECT key, value FROM meta").fetchall())
    assert meta["simplewiki.wikidata.dump"] == "wikidata-20260401.lbdb"
    assert meta["simplewiki.wikidata.export"] == "wikidata.tsv.gz"


def test_a_database_with_no_sitelinks_is_refused(wikidata, ingest, tmp_path):
    """Without the join there is nothing to import against, and the message
    should say which command supplies it."""
    db_path = tmp_path / "bare.db"
    ingest.connect(db_path).commit()
    with pytest.raises(SystemExit, match="sitelinks"):
        wikidata.main(["--db", str(db_path), "--score", "nonexistent.tsv.gz"])


def test_sqlite_is_not_asked_for_a_thousand_placeholders(wikidata, corpus):
    """A real import is half a million rows; executemany, not a built string."""
    plan = wikidata.Plan()
    plan.rows = [(f"Article {i}", "born_in", "Mississippi") for i in range(5000)]
    wikidata.write(corpus, "simplewiki", plan, Path("e.tsv.gz"), {})
    assert corpus.execute("SELECT COUNT(*) FROM derived").fetchone()[0] == 5000
