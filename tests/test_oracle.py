"""oracle.py: the search that finds the subject, and the script around it.

:mod:`liboracle` is tested against fakes, which is the right way to test a walk
over a graph - the walk is the thing, and a real index would only make the
failures harder to read. This file is about what those fakes stand in for.

``_DatabaseSearch`` matters more than its own docstring suggests. It calls
itself a stand-in for the card's BM25 index, but building that index needs a
38MB artifact, so until someone does, it is the search every question actually
goes through. Its phrase matching is where a question becomes a subject, and
getting it wrong does not raise - it answers about the wrong article.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import libgraph
import liboracle
import libsearch
import oracle


@pytest.fixture(scope="module")
def ingest(repo_root):
    """data/wikipedia/ingest.py is a script in a subdirectory, not a module."""
    path = Path(repo_root) / "data" / "wikipedia" / "ingest.py"
    spec = importlib.util.spec_from_file_location("wiki_ingest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Titles chosen for what they collide with. "Jane" and "Jane Austen" both
#: exist, so a shorter match is available whenever the longer one is; "Capital"
#: is a newspaper, which is the collision the FRAME list exists to prevent; and
#: "Pride and Prejudice" carries a frame word in the middle of its own name.
ARTICLES = [
    (1, "Jane Austen", "Jane Austen was an English novelist."),
    (2, "Jane", "Jane is a given name."),
    (3, "Steventon", "Steventon is a village in Hampshire."),
    (4, "France", "France is a country in Europe."),
    (5, "Capital", "Capital was a French newspaper."),
    (6, "Pride and Prejudice", "Pride and Prejudice is a novel."),
    (7, "Paris", "Paris is the capital of France."),
]


@pytest.fixture
def db(ingest, tmp_path):
    """The real schema, created the way ingest creates it.

    Written out rather than copied into this file: `article.id` is what
    ``_DatabaseSearch`` selects, and a hand-copied CREATE TABLE that drifted
    from the real one would let these tests pass against a schema that no
    longer exists.
    """
    conn = ingest.connect(tmp_path / "wiki.db", migrate=True)
    conn.executemany("INSERT INTO article (id, source, title, lead) "
                     "VALUES (?, 'simplewiki', ?, ?)", ARTICLES)
    conn.executemany("INSERT INTO redirect VALUES ('simplewiki', ?, ?)", [
        ("Jane Austin", "Jane Austen"),      # the misspelling a reader types
        ("Austen", "Jane Austen"),
    ])
    # Columns named rather than positional: `fact` carries an ordinal, a kind
    # and a number now, and a positional insert is the other half of the
    # drift this fixture's docstring is already worried about.
    conn.executemany(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES ('simplewiki', ?, ?, 0, ?, 'text', NULL)", [
            ("Jane Austen", "birth_place", "Steventon"),
            ("Paris", "country", "France"),
        ])
    libgraph.build(conn, "simplewiki")
    conn.commit()
    return conn


@pytest.fixture
def search(db):
    return oracle._DatabaseSearch(db, "simplewiki")


def ids(hits):
    return [doc for doc, _ in hits]


# --- finding the subject ------------------------------------------------------


def test_a_question_finds_the_article_it_names(search):
    assert ids(search.search("where was jane austen born")) == [1]


def test_the_longest_phrase_wins(search):
    """Both "Jane" and "Jane Austen" are articles, so the shorter one is always
    available. Taking it would answer about a given name."""
    assert ids(search.search("who was jane austen")) == [1]
    assert ids(search.search("who was jane")) == [2]


def test_the_match_says_how_many_words_it_matched(search):
    """The second element is the span length. liboracle ignores it, but it is
    the only signal distinguishing a full-title match from a one-word one."""
    assert search.search("where was jane austen born") == [(1, 2)]
    assert search.search("who was jane") == [(2, 1)]


def test_a_span_of_only_frame_words_is_skipped(search):
    """"What is the capital of France" contains two titles - the newspaper
    `Capital` and `France` - and the first span that matches would take the
    newspaper. Frame words are relation vocabulary, so a span made only of
    them cannot be naming an entity."""
    assert ids(search.search("what is the capital of france")) == [4]


def test_a_title_with_a_frame_word_inside_it_survives(search):
    """"and" is a frame word, but rejecting word by word would break every
    title containing one. The span is judged whole."""
    assert ids(search.search("who wrote pride and prejudice")) == [6]


def test_a_redirect_resolves_to_its_target(search):
    """The article does not exist under this name; the redirect does. This is
    why a misspelling still lands somewhere."""
    assert ids(search.search("where was jane austin born")) == [1]


def test_matching_ignores_case(search):
    assert ids(search.search("WHERE WAS JANE AUSTEN BORN")) == [1]


def test_a_question_mark_is_not_part_of_a_title(search):
    assert ids(search.search("who was jane austen?")) == [1]


def test_a_question_about_nothing_here_finds_nothing(search):
    assert search.search("where was ada lovelace born") == []


def test_a_question_of_only_frame_words_finds_nothing(search):
    """Every span is frame, so every span is skipped, and the loop falls out
    rather than matching something incidental."""
    assert search.search("where was the born") == []


def test_short_words_are_dropped(search):
    """Words of two letters or fewer never form part of a phrase, so a title
    that needs one cannot be found this way. A real limitation of the stand-in,
    recorded rather than fixed: BM25 on the card has no such rule."""
    search.db.execute("INSERT INTO article (id, source, title, lead) "
                      "VALUES (8, 'simplewiki', 'Ho Chi Minh', '')")
    assert search.search("who was ho chi minh") == []


# --- reading the article ------------------------------------------------------


def test_an_article_comes_back_with_its_lead(search):
    assert search.article(1) == ("Jane Austen", "Jane Austen was an English "
                                 "novelist.")


def test_a_missing_document_is_empty_rather_than_an_error(search):
    """liboracle indexes straight into this, so raising here would surface as a
    crash rather than as the absence it is."""
    assert search.article(999) == ("", "")


# --- the search the oracle actually gets --------------------------------------


def test_the_oracle_answers_a_fact_through_the_database_search(db, monkeypatch):
    """The whole path, with only the classifier faked: question -> subject via
    _DatabaseSearch, relation -> fact, fact -> answer."""
    monkeypatch.setattr(liboracle.Oracle, "relation",
                        lambda self, q: ["born_in"])
    o = liboracle.Oracle(db, source="simplewiki", relations=object(),
                         search=oracle._DatabaseSearch(db, "simplewiki"))
    response = o.ask("where was jane austen born")
    assert response.kind == liboracle.FACT
    assert response.value == "Steventon"
    assert response.subject == "Jane Austen"


def test_both_searches_answer_the_same_question_the_same_way(search, tmp_path):
    """`liboracle.Search` has two implementations, and mypy checks they have
    the same shape. It cannot check they behave alike - that a question naming
    an entity finds that entity either way, and that both hand back a document
    the other half of the interface can then read."""
    index = libsearch.build([t for _, t, _ in ARTICLES],
                            [lead for _, _, lead in ARTICLES])
    libsearch.write_index(index, tmp_path / "WIKI.IDX")
    libsearch.write_text(index, tmp_path / "WIKI.DAT")
    card = libsearch.CardSearch(tmp_path / "WIKI.IDX", tmp_path / "WIKI.DAT")

    try:
        question = "where was jane austen born"
        from_db = search.article(search.search(question, top=1)[0][0])
        from_card = card.article(card.search(question, top=1)[0][0])
    finally:
        card.close()

    assert from_db[0] == from_card[0] == "Jane Austen"


# --- starting up --------------------------------------------------------------


def test_a_missing_database_says_how_to_build_one(tmp_path):
    """The database is four minutes of ingest, not something to guess at."""
    with pytest.raises(SystemExit, match=r"ingest\.py"):
        oracle.load(tmp_path / "absent.db", None, None, "simplewiki")


def test_a_model_that_is_not_a_phrasebook_is_refused(db, tmp_path, tiny_model_path):
    """A character decoder loads fine and then classifies into nonsense, so the
    check is on the model rather than on the answers it gives."""
    with pytest.raises(SystemExit, match="not a phrasebook"):
        oracle.load(_db_path(db), Path(tiny_model_path), None, "simplewiki")


def test_without_a_card_it_searches_the_database(db):
    loaded = oracle.load(_db_path(db), None, None, "simplewiki")
    assert isinstance(loaded.search, oracle._DatabaseSearch)


def test_with_a_card_it_uses_the_real_index(db, tmp_path):
    index = libsearch.build([t for _, t, _ in ARTICLES],
                            [lead for _, _, lead in ARTICLES])
    libsearch.write_index(index, tmp_path / "WIKI.IDX")
    libsearch.write_text(index, tmp_path / "WIKI.DAT")

    loaded = oracle.load(_db_path(db), None, tmp_path / "WIKI", "simplewiki")
    assert isinstance(loaded.search, libsearch.CardSearch)


def _db_path(db) -> Path:
    """Where sqlite put the file behind an open connection."""
    return Path(db.execute("PRAGMA database_list").fetchone()[2])


# --- saying it ----------------------------------------------------------------


def test_plain_shows_the_mechanism(db, monkeypatch, capsys):
    monkeypatch.setattr(liboracle.Oracle, "relation",
                        lambda self, q: ["born_in"])
    o = liboracle.Oracle(db, source="simplewiki", relations=object(),
                         search=oracle._DatabaseSearch(db, "simplewiki"))

    oracle.answer(o, "where was jane austen born", plain=True)
    out = capsys.readouterr().out
    assert "subject   Jane Austen" in out
    assert "relations born_in" in out
    assert "kind      fact" in out


def test_the_default_speaks_instead(db, monkeypatch, capsys):
    monkeypatch.setattr(liboracle.Oracle, "relation",
                        lambda self, q: ["born_in"])
    o = liboracle.Oracle(db, source="simplewiki", relations=object(),
                         search=oracle._DatabaseSearch(db, "simplewiki"))

    oracle.answer(o, "where was jane austen born", plain=False)
    out = capsys.readouterr().out
    assert "Steventon" in out
    assert "kind" not in out          # the mechanism stays out of the voice


# --- scoring ------------------------------------------------------------------


def test_evaluate_reports_each_kind_separately(db, monkeypatch, capsys, tmp_path):
    """An oracle that answers 30% with facts and falls back for the rest is a
    different machine from one that answers 30% and guesses, so a single
    accuracy number would hide the distinction the script exists to show."""
    monkeypatch.setattr(
        liboracle.Oracle, "relation",
        lambda self, q: ["born_in"] if "born" in q.lower() else None)
    o = liboracle.Oracle(db, source="simplewiki", relations=object(),
                         search=oracle._DatabaseSearch(db, "simplewiki"))

    path = tmp_path / "questions.txt"
    path.write_text("where was jane austen born|Steventon\n"
                    "tell me about france|country\n")
    oracle.evaluate(o, path)

    out = capsys.readouterr().out
    assert "2 questions" in out
    assert "fact" in out and "search" in out
    assert "overall" in out


def test_evaluate_counts_a_wrong_fact_as_wrong(db, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(liboracle.Oracle, "relation",
                        lambda self, q: ["born_in"])
    o = liboracle.Oracle(db, source="simplewiki", relations=object(),
                         search=oracle._DatabaseSearch(db, "simplewiki"))

    path = tmp_path / "questions.txt"
    path.write_text("where was jane austen born|Bath\n")
    oracle.evaluate(o, path)

    assert "0.0%" in capsys.readouterr().out


# --- the entry point ----------------------------------------------------------
#
# No --relations in any of these: without a model the oracle is a search
# engine, which exercises every branch of main() without needing a trained
# phrasebook on disk.


def run(monkeypatch, db, *argv):
    monkeypatch.setattr("sys.argv", ["oracle.py", "--db", str(_db_path(db)),
                                     *argv])
    oracle.main()


def test_a_question_on_the_command_line_is_asked_once(db, monkeypatch, capsys):
    run(monkeypatch, db, "who", "was", "jane", "austen")
    assert "novelist" in capsys.readouterr().out


def test_a_question_can_be_asked_plainly(db, monkeypatch, capsys):
    run(monkeypatch, db, "--plain", "who was jane austen")
    assert "subject   Jane Austen" in capsys.readouterr().out


def test_evaluate_runs_instead_of_asking(db, monkeypatch, capsys, tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("who was jane austen|novelist\n")
    run(monkeypatch, db, "--evaluate", str(path), "who was jane austen")
    out = capsys.readouterr().out
    assert "1 questions" in out
    assert "subject" not in out          # the question argument was not asked


def test_the_interactive_loop_ends_on_a_blank_line(db, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "")
    run(monkeypatch, db)
    assert "Ask the archive" in capsys.readouterr().out


def test_the_interactive_loop_ends_on_end_of_input(db, monkeypatch, capsys):
    def eof(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    run(monkeypatch, db)
    assert "Ask the archive" in capsys.readouterr().out


def test_the_interactive_loop_answers_until_it_is_dismissed(db, monkeypatch,
                                                            capsys):
    """A REPL that answered once and exited would still pass every test above."""
    asked = iter(["who was jane austen", "what is france", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(asked))
    run(monkeypatch, db)

    out = capsys.readouterr().out
    assert "novelist" in out
    assert "country in Europe" in out


# --- the frame list -----------------------------------------------------------


def test_every_relation_field_is_a_frame_word():
    """FRAME is derived from libgraph.CANONICAL rather than written out, so a
    relation added there cannot leave its own vocabulary out of the list."""
    for fields in libgraph.CANONICAL.values():
        for field in fields:
            for part in field.split("_"):
                assert part in oracle.FRAME, field


def test_the_frame_does_not_swallow_ordinary_nouns():
    """Every frame word is a word a title may not be matched on alone, so a
    list that grew carelessly would start losing entities."""
    assert not {"jane", "austen", "france", "pride", "prejudice"} & oracle.FRAME
