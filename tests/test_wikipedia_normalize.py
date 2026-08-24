"""Normalization on the way into the database.

The ingest cleans the *form* of a fact - the property's shape, the index split
off a repeated field, what type the value is. What a property *means* is
libgraph's job, and the split matters: form is mechanical and testable here,
meaning is a judgement about one particular corpus.

Every share quoted is measured over the 1,950,164 facts in the Simple English
snapshot, which is what makes these worth doing rather than tidy-looking.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "wikipedia"))

import ingest

# --- property names -----------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("  Birth Place ", "birth_place"),
    ("BIRTH_PLACE", "birth_place"),
    ("subdivision_name1", "subdivision_name1"),   # the index is split later
    # A hyphen separates words exactly as a space does, and templates use it
    # constantly. Refusing it cost 3,014 fields in 60,000 pages.
    ("honorific-prefix", "honorific_prefix"),
    ("iso-code-region", "iso_code_region"),
    ("b-side", "b_side"),
])
def test_a_key_is_cleaned_into_a_property(raw, expected):
    assert ingest.normalize_property(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    # German templates appear in Simple English Wikipedia, and these mean
    # elevation and municipality key. Not ASCII, not corrupt.
    ("höhe", "höhe"),
    ("gemeindeschlüssel", "gemeindeschlüssel"),
    # A real field that happens to start with a digit.
    ("1-min_winds", "1_min_winds"),
    ("2010pop", "2010pop"),
])
def test_a_key_is_kept_for_having_letters_rather_than_ascii(raw, expected):
    """The first rule here demanded `^[a-z]`, which is a statement about
    English rather than about whether something is a property."""
    assert ingest.normalize_property(raw) == expected


# --- indices, which need the whole vocabulary to spot -------------------------


def test_a_repeated_field_is_folded_into_one_property_and_an_index():
    families = ingest.index_families(
        ["subdivision_name", "subdivision_name1", "subdivision_name2"])
    assert families == {"subdivision_name1": ("subdivision_name", 1),
                        "subdivision_name2": ("subdivision_name", 2)}


def test_a_series_is_recognised_without_a_bare_base():
    families = ingest.index_families(["leader_name1", "leader_name2"])
    assert families == {"leader_name1": ("leader_name", 1),
                        "leader_name2": ("leader_name", 2)}


@pytest.mark.parametrize("units", [
    ["area_km2"], ["population_density_km2"], ["area_total_km2"],
])
def test_a_unit_that_ends_in_a_digit_is_left_alone(units):
    """`area_km2` is square kilometres. Splitting it invents a field called
    `area_km` that nobody wrote, and 42,541 facts hang on getting this right -
    every single one of them ending in km2.

    The name cannot tell you which it is. Only the rest of the vocabulary can,
    by whether it agrees there is a series.
    """
    assert ingest.index_families(units) == {}


def test_a_unit_is_still_left_alone_beside_a_real_series():
    families = ingest.index_families(
        ["area_km2", "subdivision_name1", "subdivision_name2"])
    assert "area_km2" not in families
    assert families["subdivision_name1"] == ("subdivision_name", 1)


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    "123",                       # a positional argument, not a named field
    "&lt;!--_company_slogan",    # a comment that ran into the next key
    "<!--gov_as_of",
    "image_size",                # furniture
    "x" * 80,                    # a paragraph that wandered into a key
])
def test_a_key_that_is_not_a_property_is_refused(raw):
    """3,740 of 13,387 properties were used exactly once, and they were parse
    artifacts rather than vocabulary. Saying what a key may look like drops
    them without a list of special cases."""
    assert ingest.normalize_property(raw) is None


def test_a_comment_between_pipes_does_not_become_a_key():
    """The bug this rule exists for: a comment is not a field, but the splitter
    cannot tell, so its text ran on into the key that followed."""
    fields = ingest.infobox_fields(
        "Infobox person\n| name = Ada\n<!-- scroll down to edit -->\n"
        "| birth_place = London\n")
    assert dict(fields) == {
        "name": "Ada", "birth_place": "London"}


# --- values -------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("* London", "London"),
    ("- London", "London"),
    ("London,", "London"),
    ("  London ;", "London"),
    ("London<ref name=\"", "London"),      # an unterminated tag REF cannot pair
    ("London", "London"),
])
def test_the_edges_of_a_value_are_trimmed(raw, expected):
    assert ingest.normalize_value(raw) == expected


def test_a_doubly_escaped_entity_is_decoded_all_the_way():
    """`&amp;amp;` is `&amp;` after one pass, which is what 19 addresses in the
    corpus were left reading as. Escaping happens twice in a dump often enough
    to be worth going round again."""
    assert ingest.clean("Avenue H &amp;amp; East 16th Street") == \
        "Avenue H & East 16th Street"


def test_decoding_to_a_fixed_point_still_strips_what_it_uncovers():
    """The ordering bug this sits next to: decode first, so a twice-escaped
    tag is a tag by the time the tag pass runs rather than after it. A ref
    goes with its contents, which is why the citation disappears too."""
    assert ingest.clean("Bell &amp;lt;ref&amp;gt;cite&amp;lt;/ref&amp;gt; born") \
        == "Bell born"


def test_trimming_does_not_eat_a_value_that_needs_its_punctuation():
    assert ingest.normalize_value("Yahoo!") == "Yahoo!"
    assert ingest.normalize_value("St. Louis") == "St. Louis"


@pytest.mark.parametrize("value, kind, num", [
    ("1234", "number", 1234.0),
    ("3,679", "number", 3679.0),           # 0.9% of values carry separators
    ("35751.46", "number", 35751.46),
    ("-40", "number", -40.0),
    ("17 March 1328", "date", 1328.0),
    ("March 17, 1328", "date", 1328.0),
    ("1328-03-17", "date", 1328.0),
    ("https://example.org/x", "url", None),
    ("William Shakespeare", "text", None),
    ("9th century", "text", None),         # a parser that guessed would invent
])
def test_a_value_is_classified_and_its_number_pulled_out(value, kind, num):
    """20.5% of values are numbers and 3.8% are dates. Stored as text they sort
    lexically, which puts 9 after 10."""
    assert ingest.value_kind(value) == (kind, num)


# --- what the schema will not accept ------------------------------------------


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(ingest._schema())
    return conn


def insert(db, **over):
    row = {"source": "w", "subject": "S", "property": "p", "ordinal": 0,
           "value": "v", "kind": "text", "num": None, **over}
    db.execute(
        "INSERT INTO fact (source, subject, property, ordinal, value, kind, num)"
        " VALUES (:source, :subject, :property, :ordinal, :value, :kind, :num)",
        row)


@pytest.mark.parametrize("bad", [
    {"value": ""},
    {"ordinal": -1},
    {"kind": "txet"},
    {"kind": "number", "num": None},       # a number with no numeric value
    {"kind": "text", "num": 1.0},          # a number where there is no number
])
def test_the_database_refuses_what_the_cleaner_promised_not_to_write(db, bad):
    """Normalizing on the way in is a promise the writer makes. A CHECK is one
    the database keeps, and it is the difference between a rule and a habit."""
    with pytest.raises(sqlite3.IntegrityError):
        insert(db, **bad)


def test_the_same_field_twice_at_different_indices_is_two_facts(db):
    """`subdivision_name1` and `subdivision_name2` are one property now, so the
    ordinal is the only thing keeping them apart - and 27.6% of facts are
    positional variants."""
    insert(db, property="subdivision_name", ordinal=1, value="England")
    insert(db, property="subdivision_name", ordinal=2, value="Hampshire")
    assert db.execute("SELECT COUNT(*) FROM fact").fetchone()[0] == 2

    with pytest.raises(sqlite3.IntegrityError):
        insert(db, property="subdivision_name", ordinal=1, value="Elsewhere")
