"""The oracle as one program: search, classify, walk, answer.

Four stages that were each measured alone, now on one machine and one card.
What these pin is the joins between them - that the classifier's phrase index
means the path the card thinks it means, that the walk starts from the article
the search found, and that a question the graph cannot answer still gets the
article rather than silence.

Needs torch, because the classifier has to be a real trained model: a stub
would test the plumbing and not the thing the plumbing is for.
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
import libsearch
from libhost import AgonHost

torch = pytest.importorskip("torch", reason="the classifier needs PyTorch")

import buildwikibin  # noqa: E402
import buildwikigraph  # noqa: E402

CORPUS = [
    ("Jane Austen", "Jane Austen was an English novelist."),
    ("Steventon", "Steventon is a village in Hampshire."),
    ("Hampshire", "Hampshire is a county in England."),
    ("England", "England is a country in the United Kingdom."),
    ("Marie Curie", "Marie Curie was a physicist and chemist."),
    ("Warsaw", "Warsaw is the capital of Poland."),
    ("Poland", "Poland is a country in central Europe."),
]
FACTS = [
    ("Jane Austen", "birth_place", "Steventon"),
    ("Steventon", "county", "Hampshire"),
    ("Hampshire", "country", "England"),
    ("Marie Curie", "birth_place", "Warsaw"),
    ("Warsaw", "country", "Poland"),
]
#: In the order `classify.train` sorts its labels, which is what makes a phrase
#: index mean the same thing to the model and to the paths table.
PHRASES = ["BORN_IN", "BORN_IN IN_COUNTRY", "COUNT_LOCATED_IN",
           libgraphcard.REFUSE_PATH.upper()]


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    """One card: index, text and graph, all from the same article list."""
    out = tmp_path_factory.mktemp("oracle")
    conn = sqlite3.connect(":memory:")
    conn.executescript(ingest._schema())
    fillers = [(f"{n} filler {i}", n)
               for n in ("England", "Poland") for i in range(3)]
    conn.executemany("INSERT INTO article (source, title, lead) "
                     "VALUES ('w', ?, ?)", CORPUS + [(t, "") for t, _ in fillers])
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('w', ?, ?, 0, ?, 'text', NULL)", FACTS)
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('w', ?, 'country', 0, ?, 'text', NULL)", fillers)
    libgraph.build(conn, "w")

    titles = [t for (t,) in conn.execute(
        "SELECT title FROM article WHERE source='w' ORDER BY id")]
    leads = dict(conn.execute("SELECT title, lead FROM article WHERE source='w'"))
    doc = {t: i for i, t in enumerate(titles)}
    relations = sorted({r for (r,) in conn.execute(
        "SELECT DISTINCT relation FROM edge WHERE source='w'")})
    rid = {n: i for i, n in enumerate(relations)}
    edges = [(doc[s], rid[r], doc[o]) for s, r, o in conn.execute(
        "SELECT subject, relation, object FROM edge WHERE source='w'")]
    types: dict[str, list[int]] = {}
    for kind, entity in conn.execute(
            "SELECT kind, entity FROM entity_type WHERE source='w'"):
        types.setdefault(kind, []).append(doc[entity])

    index = libsearch.build(titles, [leads[t] for t in titles], {})
    idx, dat = out / "O.IDX", out / "O.DAT"
    libsearch.write_index(index, idx)
    libsearch.write_text(index, dat)

    paths = buildwikigraph.paths_for(PHRASES, relations, sorted(types))
    built = libgraphcard.build(titles, edges, relations, types, paths)
    grf = out / "O.GRF"
    libgraphcard.write(built, grf)
    return out, index, libgraphcard.CardGraph(grf), built, titles, paths


@pytest.fixture(scope="module")
def binary(card, tmp_path_factory):
    """The oracle binary, over a classifier trained on the two phrases."""
    import buildez80
    import classify
    import libinfer

    _out, index, graph, built, _titles, paths = card
    pairs = []
    for who in ("jane austen", "marie curie", "ada lovelace", "isaac newton"):
        pairs.extend((f.format(who), "BORN_IN") for f in (
            "where was {} born", "{} was born where",
            "birthplace of {}", "where did {} come from"))
        pairs.extend((f.format(who), "BORN_IN IN_COUNTRY") for f in (
            "what country was {} born in", "which country was {} born in",
            "what nation was {} born in", "{} was born in what country"))
        pairs.extend((f.format(who), libgraphcard.REFUSE_PATH.upper()) for f in (
            "how many cousins does {} have", "count the cousins of {}",
            "is {} related to me", "who is the oldest friend of {}"))
    # A count *is* a path, and shares its wording with the refusals above. The
    # two classes are only told apart by what is being counted, which is the
    # seam worth having a binary exercise.
    for place in ("england", "poland", "hampshire", "france"):
        pairs.extend((f.format(place), "COUNT_LOCATED_IN") for f in (
            "how many places are in {}", "count the places in {}",
            "how many towns are in {}", "number of places in {}"))

    model, _o, _m = classify.train(
        pairs, [32], 200, 0.01, seed=0, split_seed=0, val_frac=0.25,
        accum_bits=24, position_bands=libinfer.FLAT, quiet=True)
    npz = tmp_path_factory.mktemp("model") / "O.npz"
    model.save_npz(str(npz))

    spec = buildwikibin.OracleSpec(
        graph_name="O.GRF", forward_at=graph.forward_at,
        num_edges=graph.num_edges,
        types_at=graph._types_at - 8 * len(graph.type_names),
        num_types=len(graph.type_names), num_docs=built.num_docs,
        digest=built.digest, paths=paths,
        model=buildez80.load_for_build(str(npz), report_io=False))
    return buildwikibin.build(index.num_docs, index_name="O.IDX",
                              text_name="O.DAT", oracle=spec).build()


def ask(card, binary, question: str, files=None) -> str:
    out, *_ = card
    host = AgonHost(stdin=[question, "!"], files=files or {
        "O.IDX": (out / "O.IDX").read_bytes(),
        "O.DAT": (out / "O.DAT").read_bytes(),
        "O.GRF": (out / "O.GRF").read_bytes()})
    return host.run(binary, max_cycles=2_000_000_000)


# --- answering ----------------------------------------------------------------


def test_a_one_hop_question_is_answered_from_the_graph(card, binary):
    assert "Warsaw." in ask(card, binary, "where was marie curie born")


def test_a_climb_of_one_hop_is_answered(card, binary):
    assert "Poland." in ask(card, binary, "what country was marie curie born in")


def test_a_climb_of_two_hops_is_answered(card, binary):
    """Steventon is a village, so this is born_in then located_in twice.

    The classifier never predicted a hop count - it asked for a country, and
    how far away that was is the graph's business. A fixed two-hop path would
    have answered Hampshire.
    """
    assert "England." in ask(card, binary, "what country was jane austen born in")


def test_an_answer_is_the_title_alone(card, binary):
    """A fact is an answer, not a search result. The lead would make it look
    like the machine had found an article rather than known something."""
    out = ask(card, binary, "where was marie curie born")
    assert "Warsaw." in out
    assert "capital of Poland" not in out       # Warsaw's lead


# --- falling back -------------------------------------------------------------


def test_a_question_the_graph_cannot_answer_still_gets_the_article(card, binary):
    """Poland has no birthplace, so the walk finds nothing. Handing over the
    article beats saying nothing, and it is what the search build already is."""
    out = ask(card, binary, "where was poland born")
    assert "central Europe" in out


def test_a_query_matching_no_article_says_so(card, binary):
    """Nothing scored, so there is no entity to ask a question about."""
    out = ask(card, binary, "zzzzqqq")
    assert "Warsaw" not in out and "England" not in out
    assert "nothing" in out.lower() or "no " in out.lower()


# --- refusing, which a corpus with no gaps cannot otherwise do ----------------


def test_a_refused_phrase_says_so_instead_of_walking(card, binary):
    """The whole point of the class. A count over a set is not a path at any
    length, and without somewhere to route it the question lands on a path that
    *does* complete - so the machine answers a different question, fluently."""
    out = ask(card, binary, "how many cousins does jane austen have")
    assert "I do not know that one." in out


def test_a_refusal_offers_no_articles(card, binary):
    """Distinct from the search fallback on purpose. This corpus has no gaps,
    so the article list is never empty, and offering it would be the confident
    wrong answer wearing a different hat."""
    out = ask(card, binary, "how many cousins does jane austen have")
    assert "Steventon" not in out and "Hampshire" not in out


# --- counting, which prints a number rather than a title ----------------------


def test_a_count_is_answered_with_a_number(card, binary):
    """Hampshire and three fillers. The answer is not an article, so the
    program has to print it, and printing a number is the whole of the new
    machinery on this side."""
    out = ask(card, binary, "how many places are in england")
    assert "4." in out
    assert "Hampshire" not in out       # a tally, not the first of them


def test_a_count_of_nothing_prints_zero(card, binary):
    """Nothing is in Jane Austen. Zero is an answer and the article list is
    not - the walk returns carry clear so the program never reaches it."""
    out = ask(card, binary, "how many places are in jane austen")
    assert "0." in out


def test_refusing_is_not_what_an_unwalkable_path_does(card, binary):
    """A step count of zero still hands over the article - `None` and `[]` are
    different things and the card keeps them apart."""
    out = ask(card, binary, "where was poland born")
    assert "I do not know that one." not in out


# --- the pair of files that must agree ----------------------------------------


def test_a_graph_from_another_corpus_is_refused(card, binary, tmp_path):
    """The mismatch with no symptom: every id in the wrong graph is still an
    article, so without this check the machine answers fluently and wrongly."""
    out, _index, graph, _built, titles, paths = card
    other = libgraphcard.build(titles[:-1], [], graph.relations, {}, paths)
    wrong = tmp_path / "W.GRF"
    libgraphcard.write(other, wrong)

    printed = ask(card, binary, "where was marie curie born", files={
        "O.IDX": (out / "O.IDX").read_bytes(),
        "O.DAT": (out / "O.DAT").read_bytes(),
        "O.GRF": wrong.read_bytes()})
    assert "Warsaw." not in printed
