"""Turning a MediaWiki dump into the database the card is built from.

The markup cleaner gets most of the attention here, because every one of these
cases shipped to the screen before it was fixed: an infobox printed as its own
parameters, an image caption cut in half, a `<ref>` tag that a dump had escaped
and the tag pass therefore missed.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ingest(repo_root):
    """data/wikipedia/ingest.py is a script in a subdirectory, not a module."""
    path = Path(repo_root) / "data" / "wikipedia" / "ingest.py"
    spec = importlib.util.spec_from_file_location("wiki_ingest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the cleaner --------------------------------------------------------------


def test_a_nested_infobox_is_removed_whole(ingest):
    """Templates nest, so a regex leaves the outer braces and every parameter."""
    markup = ("{{Infobox person |image = Bell.jpg "
              "|birth_date = {{birth date|1847|3|3}} |place = Edinburgh}} "
              "Alexander Graham Bell was a scientist.")
    assert ingest.clean(markup) == "Alexander Graham Bell was a scientist."


def test_an_image_caption_containing_a_link_is_removed_whole(ingest):
    """Matching the first ']]' cuts the caption and leaves its tail in the prose."""
    markup = ("[[File:x.jpg|thumb|A painting by [[Renoir]] is a work of art.]] "
              "Art is a creative activity.")
    assert ingest.clean(markup) == "Art is a creative activity."


def test_links_keep_the_words_they_display(ingest):
    assert ingest.clean("[[Hamlet]] is a [[tragedy|sad play]].") == \
        "Hamlet is a sad play."


def test_escaped_tags_are_decoded_before_they_are_stripped(ingest):
    """A dump escapes some markup, so entities have to be resolved first - or
    `&lt;ref&gt;` survives the tag pass and becomes a tag afterwards, which is
    how '<ref></ref>' ended up printed on screen."""
    markup = "Everest is high.&lt;ref&gt;{{cite|x}}&lt;/ref&gt; It is tall."
    assert ingest.clean(markup) == "Everest is high. It is tall."


def test_mediawiki_cache_keys_are_dropped(ingest):
    assert "87asy1l2kv58ysrfolvst47vbq3y71p" not in ingest.clean(
        "It is tall. 87asy1l2kv58ysrfolvst47vbq3y71p")


def test_ordinary_long_words_survive(ingest):
    """The cache-key filter needs a digit, so real words are not caught."""
    assert "antidisestablishmentarianism" in ingest.clean(
        "It concerns antidisestablishmentarianism today.")


def test_categories_and_headings_do_not_reach_the_lead(ingest):
    # A heading occupies its own line in wiki markup, which is what the
    # anchored pattern requires and what a dump actually contains.
    markup = "Text here.\n== History ==\n[[Category:Physics]]"
    assert ingest.clean(markup) == "Text here."


def test_a_lead_is_cut_at_a_sentence_boundary(ingest):
    text = ("First sentence here. " * 40)
    lead = ingest.lead_of(text)
    assert len(lead) <= ingest.LEAD_CHARS + 60
    assert lead.endswith(".")


# --- the database -------------------------------------------------------------


DUMP = """<mediawiki>
<page><title>Hamlet</title><ns>0</ns>
<text xml:space="preserve">{{Infobox play
| name = Hamlet
| writer = [[William Shakespeare]]
| genre = [[Tragedy|Tragic play]]
| premiere = {{start date|1602}}
| image = Hamlet.jpg
| image_size = 250px
| caption = A scene from the play
}}
'''Hamlet''' is a play by [[William Shakespeare]].</text>
</page>
<page><title>Bill Shakespeare</title><ns>0</ns><redirect title="Hamlet" />
<text xml:space="preserve">#REDIRECT [[Hamlet]]</text>
</page>
<page><title>Talk:Hamlet</title><ns>1</ns>
<text xml:space="preserve">Not an article.</text>
</page>
</mediawiki>
"""


def build_db(ingest, tmp_path, dump_text=DUMP, name="dump.xml"):
    dump = tmp_path / name
    dump.write_text(dump_text)
    db_path = tmp_path / "wiki.db"
    db = ingest.connect(db_path)
    ingest.ingest(db, dump, "simplewiki")
    db.commit()
    return db, db_path, dump


def test_articles_and_redirects_are_separated(ingest, tmp_path):
    db, _, _ = build_db(ingest, tmp_path)
    articles = db.execute("SELECT title, lead FROM article").fetchall()
    assert articles == [("Hamlet", "Hamlet is a play by William Shakespeare.")]
    assert db.execute("SELECT title, target FROM redirect").fetchall() == \
        [("Bill Shakespeare", "Hamlet")]


def test_pages_outside_namespace_zero_are_ignored(ingest, tmp_path):
    db, _, _ = build_db(ingest, tmp_path)
    assert db.execute(
        "SELECT COUNT(*) FROM article WHERE title LIKE 'Talk:%'").fetchone()[0] == 0


def test_provenance_is_recorded(ingest, tmp_path):
    db, _, dump = build_db(ingest, tmp_path)
    meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
    assert meta["simplewiki.dump"] == dump.name
    assert meta["simplewiki.articles"] == "1"
    assert meta["simplewiki.facts"] == "3"      # name, writer, genre
    assert len(meta["simplewiki.digest"]) == 16
    assert meta["schema_version"] == str(ingest.SCHEMA_VERSION)


def test_a_wikimedia_dump_records_where_it_came_from(ingest, tmp_path):
    """A filename says what a file was called on one machine. The URL says
    which snapshot it is, which is what another contributor needs."""
    db, _, _ = build_db(ingest, tmp_path,
                        name="simplewiki-20260801-pages-articles.xml")
    url = db.execute("SELECT value FROM meta WHERE key = 'simplewiki.url'"
                     ).fetchone()[0]
    assert url == ("https://dumps.wikimedia.org/simplewiki/20260801/"
                   "simplewiki-20260801-pages-articles.xml")


def test_a_dump_named_anything_else_records_no_url(ingest, tmp_path):
    """Guessing one would be worse than admitting there isn't one."""
    assert ingest.dump_url("some-local-export.xml") is None
    db, _, _ = build_db(ingest, tmp_path, name="some-local-export.xml")
    assert db.execute("SELECT COUNT(*) FROM meta WHERE key LIKE '%.url'"
                      ).fetchone()[0] == 0


def test_reingesting_replaces_rather_than_merges(ingest, tmp_path):
    """A dump is a complete snapshot, so an upstream deletion has to propagate."""
    db, _, _ = build_db(ingest, tmp_path)

    smaller = """<mediawiki>
<page><title>Photosynthesis</title><ns>0</ns>
<text xml:space="preserve">Plants make food.</text>
</page>
</mediawiki>
"""
    newer = tmp_path / "newer.xml"
    newer.write_text(smaller)
    ingest.ingest(db, newer, "simplewiki")
    db.commit()

    titles = [r[0] for r in db.execute("SELECT title FROM article")]
    assert titles == ["Photosynthesis"]
    assert db.execute("SELECT COUNT(*) FROM redirect").fetchone()[0] == 0


def test_two_sources_can_share_one_database(ingest, tmp_path):
    """So Wiktionary or Wikibooks could be ingested alongside later."""
    db, _, _ = build_db(ingest, tmp_path)
    other = tmp_path / "other.xml"
    other.write_text("""<mediawiki>
<page><title>Water</title><ns>0</ns>
<text xml:space="preserve">A clear liquid.</text>
</page>
</mediawiki>
""")
    ingest.ingest(db, other, "wiktionary")
    db.commit()

    assert db.execute("SELECT COUNT(*) FROM article WHERE source='simplewiki'"
                      ).fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM article WHERE source='wiktionary'"
                      ).fetchone()[0] == 1


def test_infobox_fields_become_facts(ingest, tmp_path):
    """An infobox is a hand-curated set of facts about its article, which is
    what an oracle answers from - prose is what a search engine answers with."""
    db, _, _ = build_db(ingest, tmp_path)
    facts = dict(db.execute(
        "SELECT property, value FROM fact WHERE subject = 'Hamlet'"))
    assert facts["writer"] == "William Shakespeare"
    assert facts["genre"] == "Tragic play"      # [[target|shown]] shows shown


def test_layout_fields_are_not_facts(ingest, tmp_path):
    """Roughly a third of infobox fields position the page rather than say
    anything about the subject, and none of them answers a question."""
    db, _, _ = build_db(ingest, tmp_path)
    props = {p for (p,) in db.execute(
        "SELECT property FROM fact WHERE subject = 'Hamlet'")}
    assert not props & {"image", "image_size", "caption"}


def test_a_template_valued_field_is_dropped_rather_than_stored_raw(ingest, tmp_path):
    """`{{start date|1602}}` cleans to nothing useful, so it is not a fact.
    Storing the braces would put them on screen."""
    db, _, _ = build_db(ingest, tmp_path)
    value = db.execute("SELECT value FROM fact WHERE subject='Hamlet' "
                       "AND property='premiere'").fetchone()
    assert value is None or "{{" not in value[0]


def test_facts_are_replaced_on_reingest(ingest, tmp_path):
    db, _, _ = build_db(ingest, tmp_path)
    assert db.execute("SELECT COUNT(*) FROM fact").fetchone()[0] > 0

    newer = tmp_path / "newer.xml"
    newer.write_text("""<mediawiki>
<page><title>Photosynthesis</title><ns>0</ns>
<text xml:space="preserve">Plants make food.</text>
</page>
</mediawiki>
""")
    ingest.ingest(db, newer, "simplewiki")
    db.commit()
    assert db.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 0


def test_the_fact_table_supports_both_lookup_directions(ingest, tmp_path):
    """Forward is the primary key; backwards is fact_value, and it is what
    turns "who wrote Hamlet" into "what did Shakespeare write" for free."""
    db, _, _ = build_db(ingest, tmp_path)
    forward = db.execute("SELECT value FROM fact WHERE source=? AND subject=? "
                         "AND property=?", ("simplewiki", "Hamlet", "writer")
                         ).fetchone()
    assert forward[0] == "William Shakespeare"

    backward = db.execute("SELECT subject FROM fact WHERE source=? AND value=?",
                          ("simplewiki", "William Shakespeare")).fetchall()
    assert ("Hamlet",) in backward

    plan = db.execute("EXPLAIN QUERY PLAN SELECT subject FROM fact "
                      "WHERE source=? AND value=?", ("simplewiki", "x")
                      ).fetchall()
    assert any("fact_value" in str(row) for row in plan), plan


def test_the_schema_uses_the_features_it_claims_to(ingest, tmp_path):
    db, _, _ = build_db(ingest, tmp_path)
    sql = dict(db.execute(
        "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))

    # Text primary keys, so the hidden rowid and its btree are dead weight.
    assert "WITHOUT ROWID" in sql["redirect"]
    assert "WITHOUT ROWID" in sql["meta"]
    assert "WITHOUT ROWID" in sql["fact"]
    # article's id IS the rowid, and the card build reads rows in that order.
    assert "WITHOUT ROWID" not in sql["article"]
    # Every index answers a query something actually asks.
    assert {"redirect_target", "fact_value"} <= set(sql)

    if sqlite3.sqlite_version_info >= (3, 37, 0):
        assert all("STRICT" in sql[t]
                   for t in ("meta", "article", "redirect", "fact"))


def test_strict_rejects_a_blob_in_a_text_column(ingest, tmp_path):
    """What STRICT actually buys here. It does *not* stop a stray number -
    TEXT affinity converts that anyway - but an ordinary table would store
    bytes as a BLOB and hand them back as bytes."""
    if sqlite3.sqlite_version_info < (3, 37, 0):
        pytest.skip("STRICT needs SQLite 3.37")
    db, _, _ = build_db(ingest, tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="BLOB"):
        db.execute("INSERT INTO article (source, title) VALUES (?, ?)",
                   ("simplewiki", b"bytes"))


def test_the_schema_version_is_recorded_and_checked(ingest, tmp_path):
    """CREATE TABLE IF NOT EXISTS would leave an old layout in place and
    silently ignore the new one, so the version is checked rather than assumed."""
    _, db_path, _ = build_db(ingest, tmp_path)
    db = sqlite3.connect(db_path)
    assert db.execute("PRAGMA user_version").fetchone()[0] == ingest.SCHEMA_VERSION
    db.execute("PRAGMA user_version = 99")
    db.commit()
    db.close()

    with pytest.raises(SystemExit, match="schema version 99"):
        ingest.connect(db_path)
    # ...but ingest may rebuild, since it replaces every row regardless.
    ingest.connect(db_path, migrate=True)


def test_a_version_bump_removes_tables_the_new_schema_does_not_define(
        ingest, tmp_path):
    """The failure the version check exists to stop, which it once allowed.

    Migration used to drop a hand-written list of tables. A schema that gained
    one left the old definition in place - `CREATE TABLE IF NOT EXISTS` accepts
    it silently - so the database ended up stamped with the new version while
    carrying the old layout, indexes and all.
    """
    _, db_path, _ = build_db(ingest, tmp_path)

    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE leftover (x TEXT)")
    db.execute("CREATE INDEX leftover_x ON leftover (x)")
    db.execute("PRAGMA user_version = 1")
    db.commit()
    db.close()

    db = ingest.connect(db_path, migrate=True)
    names = {n for (n,) in db.execute("SELECT name FROM sqlite_master")}
    assert "leftover" not in names
    assert "leftover_x" not in names
    assert db.execute("PRAGMA user_version").fetchone()[0] == ingest.SCHEMA_VERSION


def test_the_database_is_readable_without_the_ingest_module(ingest, tmp_path):
    """buildwikisearch opens it read-only and knows only the schema."""
    _, db_path, _ = build_db(ingest, tmp_path)
    plain = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    assert plain.execute("SELECT title FROM article").fetchone()[0] == "Hamlet"
