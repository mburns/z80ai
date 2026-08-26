"""Written entries, and the one thing about them that has no symptom.

`data/silo/authored.py` puts human-written documents into the same `article`
table the generator writes, with the same `source`, so that `buildwikisearch`
cannot tell them apart and does not have to. Two things are pinned here.

**That the cap is the device's and not a taste.** An entry too long for
`READ_ARTICLE` to finish is not truncated on the machine - the decoder runs
past what was read - so the ingest refuses it, and the number it refuses at is
derived from `libsearch.MAX_PACKED_ARTICLE` rather than written down again.

**That the parse is the one a person would expect.** A title on the first line
and wrapped lines joined into paragraphs, because the device has no word wrap
and a paragraph break is the only formatting that survives to the screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))
import authored

import libsearch

# --- the parse ----------------------------------------------------------------


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "entry.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_first_line_is_the_title(tmp_path):
    title, body = authored.read_entry(
        write(tmp_path, "Incident Report 214-11\n\nThe pump stopped.\n"))
    assert title == "Incident Report 214-11"
    assert body == "The pump stopped."


def test_wrapped_lines_are_joined_because_the_device_wraps_them_itself(tmp_path):
    """A hard-wrapped file would print with breaks at the author's column and
    again at the screen's, which reads as ragged nonsense on an 80-column VDU."""
    _title, body = authored.read_entry(write(
        tmp_path, "T\n\nthe pump on level\nforty two failed\nat oh six hundred\n"))
    assert body == "the pump on level forty two failed at oh six hundred"


def test_a_blank_line_survives_as_a_paragraph_break(tmp_path):
    _title, body = authored.read_entry(
        write(tmp_path, "T\n\nfirst para\nstill first\n\nsecond para\n"))
    assert body == "first para still first\nsecond para"


def test_several_blank_lines_are_still_one_break(tmp_path):
    _title, body = authored.read_entry(
        write(tmp_path, "T\n\n\nfirst\n\n\n\nsecond\n\n"))
    assert body == "first\nsecond"


def test_an_entry_with_no_title_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no title"):
        authored.read_entry(write(tmp_path, "\n\nbody with no title\n"))


# --- the cap, which belongs to the device -------------------------------------


def test_the_cap_comes_from_what_the_device_reads():
    """Not a character count. Byte-pair packing never makes text longer, so a
    raw cap at the packed limit is one no prose can get past however badly it
    compresses - which is why this needs no compression ratio to be safe."""
    assert authored.MAX_BODY == libsearch.MAX_PACKED_ARTICLE - authored.TITLE_ROOM
    assert authored.MAX_BODY < libsearch.MAX_PACKED_ARTICLE


def test_an_entry_too_long_for_the_device_is_refused(tmp_path):
    body = "the pump failed and the seal was replaced " * 200
    with pytest.raises(SystemExit, match="the device can hold"):
        authored.load(write(tmp_path, f"T\n\n{body}\n").parent)


def test_an_entry_at_the_cap_is_accepted(tmp_path):
    body = "a" * authored.MAX_BODY
    entries = authored.load(write(tmp_path, f"T\n\n{body}\n").parent)
    assert entries == [("T", body)]


# --- writing, and the ghost a rename would leave -------------------------------


@pytest.fixture
def db(tmp_path):
    """A corpus database with one generated article already in it."""
    import schema
    conn = schema.connect(tmp_path / "silo.db", migrate=True)
    conn.execute("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                 (authored.SOURCE, "Amanda M. Wilson", "a cook, first shift"))
    conn.commit()
    yield conn
    conn.close()


def titles(conn) -> list[str]:
    return [t for (t,) in conn.execute(
        "SELECT title FROM article WHERE source = ? ORDER BY title",
        (authored.SOURCE,))]


def test_written_entries_land_beside_the_generated_ones(db):
    written, removed = authored.write(db, [("Incident Report", "the pump")])
    assert (written, removed) == (1, 0)
    assert titles(db) == ["Amanda M. Wilson", "Incident Report"]


def test_running_twice_changes_nothing(db):
    authored.write(db, [("Incident Report", "the pump")])
    authored.write(db, [("Incident Report", "the pump")])
    assert titles(db) == ["Amanda M. Wilson", "Incident Report"]


def test_a_renamed_entry_does_not_haunt_the_card(db):
    """The failure this exists for. Deleting the titles about to be written
    leaves the old one indexed, findable, and attached to no file anybody can
    edit - and because it still answers, nothing reports it."""
    authored.write(db, [("Incident Report 214-11", "the pump")])
    written, removed = authored.write(db, [("Incident Report 214-12", "the pump")])
    assert (written, removed) == (1, 1)
    assert titles(db) == ["Amanda M. Wilson", "Incident Report 214-12"]


def test_withdrawing_an_entry_removes_it(db):
    authored.write(db, [("A", "one"), ("B", "two")])
    authored.write(db, [("A", "one")])
    assert titles(db) == ["A", "Amanda M. Wilson"]


def test_an_entry_named_after_a_generated_article_is_refused(db):
    """Taking the name silently was the first behaviour, and the second run
    would then have deleted the generated article for good - a person the
    archive stops knowing about, restored only by a re-generate nobody would
    know to run."""
    with pytest.raises(SystemExit, match="already an article"):
        authored.write(db, [("Amanda M. Wilson", "written, and colliding")])
    assert titles(db) == ["Amanda M. Wilson"]


def test_a_generated_article_survives_a_withdrawal(db):
    authored.write(db, [("Incident Report", "the pump")])
    authored.write(db, [])
    assert titles(db) == ["Amanda M. Wilson"]


# --- the entries this repository ships -----------------------------------------


def test_every_shipped_entry_fits_and_packs(tmp_path):
    """The ten written entries are the reason the cap exists. If one grows past
    it the build fails at card time, which is late; this fails at test time."""
    entries = authored.load(authored.ENTRIES)
    assert len(entries) >= 10

    titles = [t for t, _ in entries]
    assert len(set(titles)) == len(titles), "two entries share a title"

    index = libsearch.build(titles, [b for _, b in entries], {})
    libsearch.write_index(index, tmp_path / "A.IDX")
    libsearch.write_text(index, tmp_path / "A.DAT")   # raises if one is too long

    reference = libsearch.CardSearch(tmp_path / "A.IDX", tmp_path / "A.DAT")
    try:
        for doc, (title, body) in enumerate(entries):
            assert reference.article(doc) == (title, body)
    finally:
        reference.close()
