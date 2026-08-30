"""Importing Wikidata statements into the corpus that already has the articles.

The decisions worth testing here are the ones that decide whether a *wrong*
answer reaches a card: what happens when Wikidata and the encyclopedia disagree,
and what happens when Wikidata says two things at once.
"""

from __future__ import annotations

import gzip
import shutil
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


def plan_for(wikidata, facts, existing, placed=None, parents=PARENTS, extra=None):
    titles = {CARROLLTON: "Carrollton, Mississippi", MISSISSIPPI: "Mississippi",
              USA: "United States", EVEREST: "Mount Everest", CHINA: "China",
              NEPAL: "Nepal", CARROLL_COUNTY: "Carroll County"}
    titles.update(extra or {})
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


# --- several properties landing on one relation -------------------------------

# Eight properties are `created_by`, and a film has several of them at once.
# Unlike the two that mean containment, they are answers to different questions
# rather than one answer at several depths.
FILM, DIRECTOR, CODIRECTOR, PRODUCER = 30, 31, 32, 33
CREDITS = {FILM: "The Matrix", DIRECTOR: "Lana Wachowski",
           CODIRECTOR: "Lilly Wachowski", PRODUCER: "Joel Silver"}


def test_a_director_outranks_a_producer(wikidata):
    """Unioning the two would reach `choose` as values that do not nest and be
    declined, so the film would get nothing at all. The infobox path never had
    that problem because `libgraph.CANONICAL` ranks its fields, and `PROPERTY`
    is in the same order for the same reason."""
    plan = plan_for(wikidata, {57: {FILM: {DIRECTOR}}, 162: {FILM: {PRODUCER}}},
                    {}, extra=CREDITS)
    assert plan.rows == [("The Matrix", "created_by", "Lana Wachowski")]
    assert plan.counts["created_by"]["outranked"] == 1


def test_an_outranked_property_is_not_a_fallback(wikidata):
    """Two directors is an ambiguous answer to "who directed", and the producer
    is not the repair for it - declining is. Falling through would answer a
    question nobody asked, fluently."""
    plan = plan_for(wikidata, {57: {FILM: {DIRECTOR, CODIRECTOR}},
                               162: {FILM: {PRODUCER}}}, {}, extra=CREDITS)
    assert plan.rows == []
    assert plan.counts["created_by"]["declined"] == 1


def test_an_original_language_outranks_an_official_one(wikidata):
    """Both are `language_is`, and `libgraph.CANONICAL` puts plain `language`
    ahead of `official_language` for the same reason: one is what the thing is
    in, the other is what its country legislates."""
    plan = plan_for(wikidata, {364: {FILM: {DIRECTOR}}, 37: {FILM: {PRODUCER}}},
                    {}, extra=CREDITS)
    assert plan.rows == [("The Matrix", "language_is", "Lana Wachowski")]


def test_the_importer_fetches_every_property_the_questions_ask_about(
        wikidata, repo_root):
    """A question class whose property was never exported can only be answered
    from the 46% of articles carrying an infobox, which is the coverage this
    file exists to get past. That is not hypothetical: the classifier was
    trained on "who directed X" for a release in which the import skipped P57.
    """
    relations = conftest.load_script(
        str(Path(repo_root) / "data" / "questions" / "relations.py"),
        "wd_relations")
    asked = {int(p[1:]): r for p, r in relations.PROPERTY_RELATION.items()}
    assert not set(asked) - set(wikidata.PROPERTY)
    assert {p: wikidata.PROPERTY[p] for p in asked} == asked


# --- what a thing is, rather than where it is ---------------------------------


def test_a_country_class_is_written_and_a_person_class_is_not(wikidata, corpus):
    """The corpus decides what a country is by a vote over infobox fields, and
    Wikidata knows 94 it does not. Personhood it decides in `libgraph` and
    stores nowhere, so a row asserting it would have no reader."""
    import libgraph

    path = write_export(
        Path(str(corpus.execute("PRAGMA database_list").fetchone()[2])).parent
        / "e.tsv.gz",
        f"{CARROLLTON}\t31\t6256\n{MISSISSIPPI}\t31\t5\n")
    plan, _extra = wikidata.load(corpus, "simplewiki", path)
    typed = [r for r in plan.rows if r[1] == libgraph.TYPE_RELATION]
    assert typed == [("Carrollton, Mississippi", libgraph.TYPE_RELATION,
                      "country")]
    assert plan.counts["country"]["typed"] == 1


# --- reading the dump ---------------------------------------------------------


def nt(subj: str, prop: str, obj: str) -> bytes:
    """One N-Triples line, in the shape the truthy dump writes them."""
    e = "http://www.wikidata.org/entity/"
    d = "http://www.wikidata.org/prop/direct/"
    obj = f"<{e}{obj}>" if obj.startswith("Q") else obj
    return f"<{e}{subj}> <{d}{prop}> {obj} .\n".encode()


def test_an_entity_to_entity_statement_is_a_row(wikidata):
    """Returned endpoints-first, because that is the order a rel COPY wants."""
    assert wikidata.triple(nt("Q42", "P19", "Q350")) == (42, 350, 19)


def test_a_literal_valued_statement_is_not_an_edge(wikidata):
    """A birth date is a fact about an article rather than an edge to another
    one, and the corpus already reads dates out of the infobox."""
    assert wikidata.triple(
        nt("Q42", "P569", '"1952-03-11T00:00:00Z"')) is None


def test_a_label_is_not_an_edge(wikidata):
    """Labels, descriptions and aliases are the bulk of the dump."""
    assert wikidata.triple(
        b'<http://www.wikidata.org/entity/Q42> '
        b'<http://www.w3.org/2000/01/rdf-schema#label> "Douglas Adams"@en .\n'
    ) is None


def test_a_non_entity_subject_is_skipped(wikidata):
    """A property can be the subject of a truthy statement - `P31 P31 Q...` -
    and it is not a node in this graph."""
    assert wikidata.triple(
        b'<http://www.wikidata.org/entity/P31> '
        b'<http://www.wikidata.org/prop/direct/P31> '
        b'<http://www.wikidata.org/entity/Q18616576> .\n') is None


def test_a_statement_id_is_not_an_entity(wikidata):
    """`Q1234-deadbeef` parses as far as the digits and then does not."""
    assert wikidata.triple(
        b'<http://www.wikidata.org/entity/Q42-c8f1> '
        b'<http://www.wikidata.org/prop/direct/P19> '
        b'<http://www.wikidata.org/entity/Q350> .\n') is None


def test_a_truncated_line_is_skipped(wikidata):
    """The last line of an interrupted download is half a triple."""
    assert wikidata.triple(b'<http://www.wikidata.org/entity/Q42> <http') is None


def test_the_pure_python_reader_prefilters(wikidata, tmp_path, monkeypatch):
    """The fallback for a machine with no external bzip2 or grep has to drop
    the same lines the shell pipeline drops, or the two paths disagree about
    what the dump contained."""
    import bz2 as bz2_module

    dump = tmp_path / "truthy.nt.bz2"
    dump.write_bytes(bz2_module.compress(
        nt("Q42", "P19", "Q350")
        + b'<http://www.wikidata.org/entity/Q42> '
        b'<http://www.w3.org/2000/01/rdf-schema#label> "Douglas Adams"@en .\n'
        + nt("Q350", "P17", "Q145")))
    monkeypatch.setattr(wikidata.shutil, "which", lambda _name: None)

    rows = [wikidata.triple(line) for line in wikidata.candidates(dump)]
    assert rows == [(42, 350, 19), (350, 145, 17)]


@pytest.mark.skipif(not shutil.which("bzip2") or not shutil.which("grep"),
                    reason="needs an external bzip2 and grep")
def test_the_shell_reader_agrees_with_the_python_one(wikidata, tmp_path,
                                                     monkeypatch):
    """The pipeline is the path that actually runs over the dump, so it is
    worth reading a real one through real processes rather than trusting that
    two implementations of the same filter agree."""
    import bz2 as bz2_module

    body = (nt("Q42", "P19", "Q350")
            + b'<http://www.wikidata.org/entity/Q42> '
            b'<http://www.w3.org/2000/01/rdf-schema#label> "Adams"@en .\n'
            + nt("Q42", "P569", '"1952-03-11T00:00:00Z"')
            + nt("Q350", "P17", "Q145"))
    dump = tmp_path / "truthy.nt.bz2"
    dump.write_bytes(bz2_module.compress(body))

    through_shell = [wikidata.triple(line) for line in wikidata.candidates(dump)]
    monkeypatch.setattr(wikidata.shutil, "which", lambda _name: None)
    through_python = [wikidata.triple(line) for line in wikidata.candidates(dump)]

    assert through_shell == through_python
    assert [r for r in through_shell if r] == [(42, 350, 19), (350, 145, 17)]


# --- the file format ----------------------------------------------------------


def write_export(path: Path, body: str, version: int = 1) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(f"# format\t{version}\n# dump\twikidata.lbdb\n" + body)
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


# P106 is `occupation`, which this corpus does not map to any relation. A
# format 2 export is full of properties like it, on purpose.
UNMAPPED = 106


def test_an_unmapped_property_is_read_past(wikidata, tmp_path):
    """The export carries every property so that mapping a new one costs no
    pass over the dump. Materialising them all would cost memory for rows
    nothing can read, and hand `build_plan` a property with no relation."""
    path = write_export(tmp_path / "e.tsv.gz",
                        f"{CARROLLTON}\t19\t{MISSISSIPPI}\n"
                        f"{CARROLLTON}\t{UNMAPPED}\t{USA}\n", version=2)
    rows, _parents, _ = wikidata.read_export(path)
    assert rows == [(CARROLLTON, 19, MISSISSIPPI)]


def test_the_survey_counts_what_the_import_reads_past(wikidata, tmp_path):
    """The point of keeping unmapped properties is being able to ask which one
    is worth a relation next, so the thing that reads past them at import has
    to be countable at survey."""
    path = write_export(tmp_path / "e.tsv.gz",
                        f"{CARROLLTON}\t19\t{MISSISSIPPI}\n"
                        f"{CARROLLTON}\t{UNMAPPED}\t{USA}\n"
                        f"{EVEREST}\t{UNMAPPED}\t{CHINA}\n"
                        f"{CARROLL_COUNTY}\t131\t{MISSISSIPPI}\tchain\n",
                        version=2)
    counts, header = wikidata.survey(path)
    assert counts == {19: 1, UNMAPPED: 2}
    assert header["format"] == "2"


def test_a_mapped_property_is_not_reported_as_unmapped(wikidata, tmp_path):
    """P37 was unmapped until it was not, and the survey has to follow
    `PROPERTY` rather than a list of its own."""
    path = write_export(tmp_path / "e.tsv.gz",
                        f"{CARROLLTON}\t37\t{USA}\n", version=2)
    counts, _ = wikidata.survey(path)
    assert set(counts) <= set(wikidata.PROPERTY)


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
