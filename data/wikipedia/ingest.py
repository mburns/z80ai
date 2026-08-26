#!/usr/bin/env python3
"""
Turn a MediaWiki dump into data/simple_english_wikipedia.db.

The database is the source of truth for everything downstream: the search index,
the card files and the binary are all regenerated from it, so refreshing the
corpus means re-running this and rebuilding. Nothing else needs to know which
snapshot it came from.

    python data/wikipedia/ingest.py ~/Downloads/simplewiki-*-pages-articles.xml.bz2
    python data/wikipedia/ingest.py --stats

A dump is a *complete* snapshot, so ingesting one replaces that source's rows
rather than merging into them - which is what makes re-running it a sync, and
what makes deletions propagate. The swap happens in one transaction, so an
interrupted run leaves the previous corpus intact.

Six tables carry the corpus, and a seventh carries what was read out of prose
rather than out of a table:

  article   what the machine can answer with. Title plus the opening sentences,
            because a 300-character lead is about a third of a 40x24 screen and
            a full article is not readable on one anyway.
  redirect  alternate names, pointing at a title. Wikipedia's editors have
            written ~115,000 of these, and they are the reason `jane austin`
            finds Jane Austen - the index does no fuzzy matching of its own.
  sitelink  the Wikidata entity each article is about, as an integer Q-id.
            Not in the XML dump - it comes from `page.sql.gz` and
            `page_props.sql.gz` - and the reason it is worth a second download
            is that it is the *only* exact key between this corpus and
            Wikidata. Matching on the title instead is right 43.5% of the time
            and confidently wrong 2.2% of the time; see the README.
  fact      what an infobox said, as (subject, property, ordinal) -> value,
            with the value's type worked out on the way in.
  property  the vocabulary those facts are written in, counted, and marked
            with the relation libgraph reads it as - where there is one.
  category  what a page files itself under. An infobox says what its author
            chose to tabulate; a category says what they chose to file it
            under, and for a great many pages the second is the only place a
            containment is written down at all - `Infobox U.S. state` has no
            country field, so Michigan says it is in the United States only by
            being filed under `1837 establishments in the United States`.
  derived   facts read out of the lead text, each tagged with the method that
            produced it. Kept apart from `fact` so that nothing inferred can be
            mistaken for something the encyclopedia tabulated, and so that two
            extractors can be measured against each other. Ingest never writes
            it; `birthplaces.py` does.

## Normalizing, and where it stops

An infobox is hand-typed by thousands of people, so what arrives is a
folksonomy: 19,117 distinct property names as typed, 5,266 of them used once and
most of those parse artifacts rather than vocabulary. Cleaning it is worth
doing and worth bounding, so this normalizes **form** and leaves **meaning** to
libgraph:

  form      the shape of a key, the index split off a repeated field, the type
            of a value, the punctuation at its edges. All mechanical, all
            decidable from the data alone, all tested.
  meaning   that `subdivision_name` and `country` are the same question. That
            is a judgement about this corpus and it belongs where the judgement
            is made, not smuggled into the parser.

What cannot be normalized is recorded instead. `property` makes the folksonomy
a table you can query rather than something to be rediscovered by reading a
dump - and *"what is used often and mapped to nothing"* is the query that found
`subdivision_name` and took chaining from 1.7% to 40.7%.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import re
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import libgraph

DB_PATH = Path(__file__).resolve().parent.parent / "simple_english_wikipedia.db"
#: Bumped whenever the table definitions change, or when what is stored in them
#: changes meaning - version 10 decodes the XML escaping in titles, so a
#: database written by 9 holds 724 articles called `AT&amp;T` and every table
#: keyed on a title agrees with it. The database is derived data, so either
#: kind of mismatch is resolved by re-ingesting rather than by migrating.
SCHEMA_VERSION = 10

#: Characters of lead text kept per article. A 40x24 Agon screen holds about
#: 960, so this is a third of one - enough to say what a thing is.
LEAD_CHARS = 300

#: STRICT arrived in SQLite 3.37 (2021). Ubuntu 22.04 ships 3.37.2, so CI has
#: it, but the schema is identical without it and there is no reason to refuse
#: to run on an older library for a type check.
STRICT = "STRICT, " if sqlite3.sqlite_version_info >= (3, 37, 0) else ""

#: Where the dumps come from. A Wikimedia dump filename carries its wiki and
#: its date, so the canonical URL can be reconstructed from the file alone -
#: which means the database can record *where* its contents came from rather
#: than only what the file happened to be called locally.
DUMP_NAME = re.compile(r"^(?P<wiki>[a-z]+)-(?P<date>\d{8})-(?P<kind>.+)$")
DUMP_BASE = "https://dumps.wikimedia.org"


def dump_url(name: str) -> str | None:
    """The canonical download URL for a Wikimedia dump filename."""
    m = DUMP_NAME.match(name)
    if not m:
        return None
    return f"{DUMP_BASE}/{m['wiki']}/{m['date']}/{name}"


def file_digest(path: Path, limit: int = 1 << 26) -> str:
    """A short digest of the dump, so a rebuild can be told apart from a reuse.

    The first 64MB rather than the whole 338MB file: enough that two different
    snapshots cannot collide in practice, and fast enough to run every time
    without anyone deciding to skip it.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()[:16]


def _schema() -> str:
    """The table definitions.

    Three choices worth stating, since none is SQLite's default:

    **WITHOUT ROWID** on ``redirect`` and ``meta`` - both have a real text
    primary key, so the default hidden rowid is an extra integer per row plus
    a second btree, and every lookup goes through it to reach the row. Storing
    the rows in the primary key's own btree is smaller and one indirection
    shorter: measured over these 114,771 redirects, 12.7MB against 15.6MB, a
    19% saving. ``article`` keeps its rowid, because ``id`` *is* the rowid and
    the card build reads rows in that order.

    **The indexes**, each answering a query something actually asks:

    - ``redirect_target`` resolves redirects onto articles, and counts how many
      point at each one, which is how a limited card picks the notable ones. It
      covers both outright, so neither touches the table.
    - ``fact``'s primary key *is* the lookup an oracle performs -
      ``(subject, property, ordinal) -> value`` - so that costs no separate
      index. ``ordinal`` is there because a template spells a list as
      ``subdivision_name1``, ``subdivision_name2``: 4,064 property names and
      25.0% of all facts were positional variants of some other property,
      which made one field look like seven unrelated ones.

    - ``property`` is the vocabulary, counted. 13,268 distinct properties is a
      folksonomy nobody wrote down, and the query worth running against it -
      *what is used often and mapped to nothing* - is the one that found
      ``subdivision_name`` and took chaining from 1.7% to 40.7%. A partial
      index keeps that query cheap. Recording it means the next such find is a
      SELECT rather than an archaeology expedition.
    - ``fact_value`` is the same lookup backwards: "who was born in Edinburgh"
      from the rows that say where people were born. Inverse relations are
      about a third of what a real question set asks - SimpleQuestions labels
      them explicitly - and without this each one scans two million rows.

    An index on ``(source, property)`` was measured and rejected: it cost 86MB
    and took the one query that wanted it, the property histogram in
    ``stats()``, from 0.68s to 0.15s. Half a second on a command run by hand,
    for a fifth of the database. Nothing else groups by property, because the
    card build reads facts in bulk rather than looking them up.

    **STRICT** - a modest guarantee, and worth being accurate about. TEXT
    affinity already converts a stray number to text in an ordinary table, so
    this does not catch that. What it does catch is a BLOB reaching a TEXT
    column, which an ordinary table stores as a BLOB and hands back as bytes -
    and the first sign of that would be ``.replace()`` failing somewhere in
    the index builder. Cheap, and it makes the declared type the type that
    comes back.

    **CHECK** is what actually enforces the cleaning. Normalizing on the way in
    is a promise the writer makes; a constraint is one the database keeps. An
    empty value, a negative ordinal, a ``kind`` invented by a typo in one
    branch of the classifier, or a number with no numeric value are all now
    write errors rather than rows that read strangely six months later.
    """
    return f"""
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) {STRICT}WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS article (
    id     INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    title  TEXT NOT NULL,
    lead   TEXT NOT NULL DEFAULT '',
    UNIQUE (source, title)
) {STRICT.rstrip(', ')};

CREATE TABLE IF NOT EXISTS redirect (
    source TEXT NOT NULL,
    title  TEXT NOT NULL,
    target TEXT NOT NULL,
    PRIMARY KEY (source, title)
) {STRICT}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS redirect_target ON redirect (source, target);

-- Which Wikidata entity an article is about. Stored as the integer rather than
-- the `Q…` string: the prefix is a constant, and the whole point of the column
-- is to be joined against, so it should be as narrow as a key can be.
--
-- `sitelink_qid` is not a nicety. Reading a Wikidata dump means arriving with a
-- Q-id and asking which article it is, which is the opposite direction from
-- everything else here.
CREATE TABLE IF NOT EXISTS sitelink (
    source TEXT NOT NULL,
    title  TEXT NOT NULL,
    qid    INTEGER NOT NULL,
    PRIMARY KEY (source, title),
    CHECK (qid > 0)
) {STRICT}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS sitelink_qid ON sitelink (source, qid);

CREATE TABLE IF NOT EXISTS category (
    source TEXT NOT NULL,
    title  TEXT NOT NULL,
    name   TEXT NOT NULL,
    PRIMARY KEY (source, title, name),
    CHECK (name <> '')
) {STRICT}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS category_name ON category (source, name);

CREATE TABLE IF NOT EXISTS fact (
    source   TEXT NOT NULL,
    subject  TEXT NOT NULL,
    property TEXT NOT NULL,
    ordinal  INTEGER NOT NULL DEFAULT 0,
    value    TEXT NOT NULL,
    kind     TEXT NOT NULL DEFAULT 'text',
    num      REAL,
    PRIMARY KEY (source, subject, property, ordinal),
    CHECK (value <> ''),
    CHECK (ordinal >= 0),
    CHECK (kind IN ('text', 'number', 'date', 'url')),
    CHECK ((num IS NULL) = (kind NOT IN ('number', 'date')))
) {STRICT}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS fact_value ON fact (source, value);

CREATE TABLE IF NOT EXISTS property (
    source   TEXT NOT NULL,
    name     TEXT NOT NULL,
    uses     INTEGER NOT NULL,
    subjects INTEGER NOT NULL,
    relation TEXT,
    PRIMARY KEY (source, name),
    CHECK (uses > 0)
) {STRICT}WITHOUT ROWID;

-- Facts read out of prose rather than out of an infobox, kept apart from
-- `fact` on purpose: every row names the method that produced it, so nothing
-- derived can ever be mistaken for something the encyclopedia tabulated. A
-- reader wanting only what Wikipedia stated reads `fact` and is done.
--
-- `method` is part of the key rather than a column beside it, so two
-- extractors can disagree about the same person and both be kept - which is
-- what makes one measurable against the other. See `birthplaces.py`.
CREATE TABLE IF NOT EXISTS derived (
    source   TEXT NOT NULL,
    subject  TEXT NOT NULL,
    relation TEXT NOT NULL,
    object   TEXT NOT NULL,
    method   TEXT NOT NULL,
    PRIMARY KEY (source, subject, relation, method),
    CHECK (object <> ''),
    CHECK (method <> '')
) {STRICT}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS derived_method ON derived (source, method);

CREATE INDEX IF NOT EXISTS property_unmapped
    ON property (source, uses DESC) WHERE relation IS NULL;
""" + libgraph.schema(strict=bool(STRICT))

# --- dump parsing -------------------------------------------------------------
#
# Line-oriented rather than an XML parse: the dump is one page per <page> block
# and only three fields are wanted, so a real parser costs minutes for nothing.

TITLE = re.compile(r"<title>(.*?)</title>")
NS = re.compile(r"<ns>(\d+)</ns>")
REDIRECT = re.compile(r'<redirect title="(.*?)"')
TEXT_OPEN = re.compile(r"<text[^>]*>")

TAG = re.compile(r"<[^>]+>")
REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL)
HEADING = re.compile(r"^=+.*?=+$", re.MULTILINE)
TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
ENTITY = {"&quot;": '"', "&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
          "&#39;": "'", "&ndash;": "-", "&mdash;": "-"}

#: The five entities an XML escaper produces, and nothing else. `clean()`
#: decodes a much wider set repeatedly, because it is undoing whatever
#: *editors* typed into the markup; this is undoing what the *dump format*
#: did on the way out, which is a smaller and exactly specified job.
XML_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#39);")
XML_ENTITY_TEXT = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                   "&apos;": "'", "&#39;": "'"}


def unescape_title(title: str) -> str:
    """Undo the dump's XML escaping of a title, exactly once.

    `<title>Dungeons &amp; Dragons</title>` is an article called
    `Dungeons & Dragons`, and storing the escaped form means the card prints
    `AT&amp;T` and searches for it under that name. 724 articles were.

    Once, and in a single left-to-right pass, because that is what XML means
    and a second pass would be wrong: a title containing the literal text
    `&lt;` arrives as `&amp;lt;`, and one pass gives `&lt;` where a fixed point
    would give `<`. That is also why this cannot reuse `clean()`, which loops
    to a fixed point on purpose - some markup in this dump is escaped twice.
    """
    return XML_ENTITY.sub(lambda m: XML_ENTITY_TEXT[m.group()], title)

#: Bracketed forms that are furniture rather than prose.
DROPPED_PREFIXES = ("file:", "image:", "category:")


def _strip_braced(text: str) -> str:
    """Remove every ``{{...}}``, however deeply nested.

    A regex cannot do this. `{{Infobox ... {{birth date|1847}} ...}}` needs
    matched pairs, and an infobox that runs past whatever window we captured
    would otherwise leave its opening braces and every parameter in the lead -
    which is exactly what "Alexander Graham Bell" looked like before this.
    """
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif text.startswith("}}", i):
            depth = max(0, depth - 1)
            i += 2
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _strip_links(text: str) -> str:
    """Resolve ``[[...]]``, dropping images and categories entirely.

    Depth-aware for the same reason: an image caption may itself contain a
    link, and matching the first ``]]`` cuts the caption in half and leaves its
    tail in the prose - which is why "Art" began "is a work of art.]]".
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if not text.startswith("[[", i):
            out.append(text[i])
            i += 1
            continue

        depth, j = 1, i + 2
        while j < len(text) and depth:
            if text.startswith("[[", j):
                depth += 1
                j += 2
            elif text.startswith("]]", j):
                depth -= 1
                j += 2
            else:
                j += 1
        if depth:                       # unterminated: drop the rest
            break

        inner = text[i + 2:j - 2]
        i = j
        if inner.lower().startswith(DROPPED_PREFIXES):
            continue                    # furniture, caption and all
        # [[target|shown]] displays the last field; [[target]] displays itself.
        out.append(_strip_links(inner.split("|")[-1]))
    return "".join(out)


#: MediaWiki appends a cache key to some page texts: a long run of letters and
#: digits that is not a word and reads as line noise in a lead.
CACHEKEY = re.compile(r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{20,}\b")


# --- value templates ----------------------------------------------------------
#
# `clean()` strips every `{{...}}`, contents and all, because a lead has to
# survive an infobox that ran past the window we captured. Applied to a *value*
# the same rule is a deletion: `{{birth date|1847|3|3}}` cleans to the empty
# string and the fact is gone.
#
# Measured over the 20260801 snapshot with `templates.py`: 301,306 infobox
# values are written as a template and 232,947 of them are dropped - 8.0% of
# every named field. The losses land on properties that matter. ~41,000 birth
# dates, ~18,400 death dates, 6,199 `spouse` values behind `{{marriage}}`, and
# 4,660 `subdivision_name` values behind `{{flag}}` - the last two are fields
# `libgraph.CANONICAL` maps, so those are `spouse_of` and `located_in` edges
# the graph never saw.
#
# So a value gets one pass of expansion before it is cleaned. Only templates
# whose meaning is unambiguous are listed; everything else still falls to
# `_strip_braced`, because inventing a reading is worse than dropping a field.
#
# Some of the biggest losses cannot be repaired here at all, and it is worth
# writing down which, so the same scan does not send the next reader after them
# again:
#
#   france metadata wikidata  15,471   holds no text. The template fetches a
#   austria population wikidata 2,382  population from Wikidata at render time,
#   official website           1,812   so there is nothing inside to expand.
#                                      These are an argument for ingesting
#                                      Wikidata, not a rule this table can hold.
#
#   cite web                  15,818   a citation is provenance, not a value.
#                                      Expanding it would put a publisher and a
#                                      retrieval date in the fact table.
#
#   medal, medalcompetition   ~7,000   a medal table is a career, and flattening
#   medalcountry, medalsport           one into a single value invents a reading
#                                      of it. `years`/`clubs` are the same shape
#                                      and are already the biggest unmapped
#                                      properties in the corpus.
#
#   coord                      1,536   recoverable, but a coordinate is not a
#                                      relation anything walks, and its argument
#                                      order has enough variants to be its own
#                                      parser. Left until something needs it.


def _positional(args: list[str]) -> list[str]:
    """The unnamed arguments, in order. `abbr=on` is a flag, not a value."""
    return [a.strip() for a in args if "=" not in a]


def _iso_date(args: list[str]) -> str | None:
    """`|1847|3|3` -> `1847-03-03`, the one date shape `value_kind` reads.

    A bare year is returned as itself rather than padded into a January the
    first that nobody wrote. `{{death date and age|1990|5|1|1920|3|2}}` leads
    with the death, so taking the first three is right for both templates.
    """
    parts = [a for a in _positional(args) if a.isdigit()]
    if not parts:
        return None
    year = parts[0]
    if len(year) not in (3, 4):
        return None
    if len(parts) < 3:
        return year
    month, day = parts[1], parts[2]
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return year
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _first(args: list[str]) -> str | None:
    """The first unnamed argument - a wrapper's whole contribution."""
    parts = _positional(args)
    return parts[0] if parts else None


def _second(args: list[str]) -> str | None:
    """`{{lang|fr|Paris}}` says Paris in French; the text is the second."""
    parts = _positional(args)
    return parts[1] if len(parts) > 1 else None


#: A wiki bullet: the marker, and whatever precedes it on its own line.
BULLET = re.compile(r"(?:^|\n)\s*[*#:;]+\s*")


def _joined(args: list[str]) -> str | None:
    """A list template's items, comma-separated.

    Two spellings reach here and both have to work. `{{hlist|Actor|Singer}}`
    puts one item per argument; `{{plainlist|* Actor * Singer}}` puts a wiki
    bullet list inside a single argument. Splitting on the marker as well as
    the pipe reads both, and keeps the asterisks out of the fact either way.
    """
    items: list[str] = []
    for arg in _positional(args):
        items.extend(BULLET.split(arg) if BULLET.search(arg) else [arg])
    kept = [s for s in (i.strip() for i in items) if s]
    return ", ".join(kept) if kept else None


def _measure(args: list[str]) -> str | None:
    """`{{convert|5|km|mi}}` is five kilometres; the rest is display."""
    parts = _positional(args)
    if not parts:
        return None
    return " ".join(parts[:2]) if len(parts) > 1 else parts[0]


#: Template name (folded the way `normalize_property` folds a key) -> what its
#: arguments mean. A name absent here keeps today's behaviour.
VALUE_TEMPLATES: dict[str, Callable[[list[str]], str | None]] = {
    # Dates. The largest single loss in the corpus.
    "birth date": _iso_date,
    "birth date and age": _iso_date,
    "birth-date": _iso_date,
    "death date": _iso_date,
    "death date and age": _iso_date,
    "start date": _iso_date,
    "start date and age": _iso_date,
    "end date": _iso_date,
    "film date": _iso_date,
    # `nowrap` exists to keep a value on one line, so its contents are the
    # value. `{{small}}`, `{{big}}`, `{{nobold}}` and `{{noitalic}}` are not
    # here for the opposite reason, and it cost 686 edges to find out: the wiki
    # idiom is `| successor = Osman Hussein {{small|(Acting)}}`, so reading
    # them turns a title that resolves into one that does not. Deleting an
    # annotation is what the old cleaner did well, and it is kept.
    "nowrap": _first,
    # A flag is a picture of a country and the name of one, and which it is
    # depends on where it sits. `subdivision_name = {{flag|France}}` is the
    # value - 4,660 of these, on the field that made chaining work. But
    # `birth_place = {{flagicon|IRI}} Urmia` is an icon beside the value, and
    # reading it gives "IRI Urmia", which names no article and loses the edge
    # the plain "Urmia" had. So these are read only when the template is the
    # whole value; see WHOLE_VALUE_ONLY.
    "flag": _first,
    "flagcountry": _first,
    "flagu": _first,
    # A marriage names a spouse, which `CANONICAL` maps to `spouse_of`.
    "marriage": _first,
    "url": _first,
    "lang": _second,
    # Lists.
    "plainlist": _joined,
    "flatlist": _joined,
    "hlist": _joined,
    "ubl": _joined,
    "unbulleted list": _joined,
    "collapsible list": _joined,
    # Measurements.
    "convert": _measure,
    "cvt": _measure,
    "val": _measure,
}

#: Templates that are a value when they stand alone and decoration when they do
#: not. A flag beside a place name is an icon; a flag *as* the field is the
#: country. Reading the first kind cost 155 `located_in` edges and a scatter of
#: `born_in` ones, by turning "Urmia" into "IRI Urmia".
#:
#: `flagicon` is absent from `VALUE_TEMPLATES` entirely rather than listed
#: here: its argument is an IOC country code, so even standing alone it yields
#: "IRI" rather than a title anything holds.
WHOLE_VALUE_ONLY = frozenset({"flag", "flagcountry", "flagu"})

#: One value is not a page: a handful of nested templates is the most any of
#: them holds, and a bound means a pathological value cannot spin here.
MAX_EXPANSIONS = 32


def expand_templates(value: str) -> str:
    """Rewrite the templates this module understands into their plain text.

    Innermost first, so `{{nowrap|{{convert|5|km}}}}` becomes `{{nowrap|5 km}}`
    and then `5 km`. A template with no rule is left exactly as it was, for
    `clean()` to strip as before - this only ever adds facts.
    """
    for _ in range(MAX_EXPANSIONS):
        close = value.find("}}")
        if close == -1:
            return value
        open_ = value.rfind("{{", 0, close)
        if open_ == -1:
            return value

        # Innermost, so the body holds no braces and `split_fields` is cutting
        # on the same top-level `|` that separates a template's arguments.
        fields = split_fields(value[open_ + 2:close])
        name = fields[0].strip().lower().replace("_", " ")
        rule = VALUE_TEMPLATES.get(name)
        if name in WHOLE_VALUE_ONLY and value.strip() != value[open_:close + 2]:
            rule = None                 # decoration beside a value, not the value
        replacement = rule(fields[1:]) if rule else None
        if replacement is None:
            # No rule, or the arguments did not fit it. Neutralise the braces
            # so the scan moves on, and leave the text for `_strip_braced`.
            replacement = "\x00" + value[open_ + 2:close] + "\x01"
        value = value[:open_] + replacement + value[close + 2:]
    return value


def unexpanded(value: str) -> str:
    """Put back the braces `expand_templates` set aside, for the cleaner."""
    return value.replace("\x00", "{{").replace("\x01", "}}")


def clean(markup: str) -> str:
    """Wiki markup down to the plain sentences underneath it."""
    # Entities first. A dump escapes some markup, so `&lt;ref&gt;` survives the
    # tag pass and *becomes* a tag afterwards - which is how "<ref></ref>"
    # ended up printed on screen.
    #
    # To a fixed point, because some of the dump is escaped twice: `&amp;amp;`
    # decodes to `&amp;` and one pass would leave that on screen. Bounded
    # rather than `while`, since a value can be adversarial and three rounds is
    # already one more than anything in this corpus needs.
    text = markup
    for _ in range(3):
        before = text
        for entity, char in ENTITY.items():
            text = text.replace(entity, char)
        if text == before:
            break
    text = REF.sub(" ", text)
    text = TABLE.sub(" ", text)
    text = _strip_braced(text)
    text = _strip_links(text)
    text = HEADING.sub(" ", text)
    text = TAG.sub(" ", text)
    text = CACHEKEY.sub(" ", text)
    text = text.replace("'''", "").replace("''", "")
    # Leading list and indent markers survive the above and read as noise.
    text = re.sub(r"^[*#:;|]+", " ", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def lead_of(markup: str) -> str:
    """The opening prose, cut at a sentence boundary where one is in reach."""
    text = clean(markup)
    if len(text) <= LEAD_CHARS:
        return text
    window = text[:LEAD_CHARS + 60]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    return window[:cut + 1] if cut >= LEAD_CHARS // 2 else text[:LEAD_CHARS]


# --- infobox facts ------------------------------------------------------------
#
# An infobox is a hand-curated set of typed key/value pairs about its article,
# which is to say a set of facts. The lead throws them away as furniture; this
# keeps them, because a question like "where was Bell born" is answered by a
# lookup and not by reading prose.

INFOBOX = re.compile(r"\{\{\s*Infobox\b", re.IGNORECASE)

#: Infobox keys that lay the page out rather than saying anything about the
#: subject. About a third of all fields, and none of them answers a question.
FURNITURE = re.compile(
    r"^(image|img|photo|logo|map|flag|seal|banner|cover|picture)(_|$)"
    r"|^(caption|alt|size|width|height|align|float|style|colour|color|border)(_|$)"
    r"|(_(image|img|photo|caption|alt|size|width|height|align|style|colour|color"
    r"|flag|map|link|ref|note|footnote|upright|padding))$"
    r"|^(module|embed|child|nocat|fetchwikidata|onlysourced|suppressfields"
    r"|dateformat|coordinates|latd|latm|lats|longd|longm|longs|pushpin.*)$",
    re.IGNORECASE)

#: Values that survived cleaning but say nothing: markup scraps, bare units,
#: template flags.
JUNK_VALUE = re.compile(r"^[\s|=*#:;{}\[\]<>/-]*$|^\d+\s*px$|^(yes|no|y|n|on|off"
                        r"|none|null|unknown|n/a|tbd|ALL)$", re.IGNORECASE)

#: A value longer than this is a paragraph that wandered into a field.
MAX_VALUE_LEN = 120

#: HTML comments, stripped before fields are split rather than after.
#:
#: A comment sitting between two pipes is not a field, but the splitter has no
#: way to know that, so its text runs on into the next key. That is how the
#: property list came to contain `<!--_company_slogan` and
#: `<!--_scroll_down_to_edit_this_page_-->_<!--_philosopher_category_-->_region`.
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: Word separators inside a key. A template author writes `honorific-prefix`
#: or `iso-code-region` as readily as `birth_place`, and a hyphen is doing the
#: same job as the underscore - so it is normalized, not grounds for refusal.
#:
#: Treating whitespace as a separator but a hyphen as corruption threw away
#: 3,014 fields in 60,000 pages, among them `postal-codes`, `honorific-suffix`
#: and `b-side`. All real, all lost to an inconsistency in this line.
SEPARATORS = re.compile("[\\s\\-\u2010-\u2015]+")

#: A property name worth keeping. Keys were never run through `clean()` - only
#: values were - so anything the markup left behind survived into the schema.
#: Rather than clean a key and hope, this says what a key may look like.
#:
#: Deliberately permissive about *what* the letters are and strict about there
#: being letters at all: `höhe` and `gemeindeschlüssel` come from German
#: templates and mean elevation and municipality key, while `123` is a
#: positional argument and `1-min_winds` is a real field that happens to start
#: with a digit. So the rule is "at least one letter, word characters only,
#: not absurdly long" rather than an ASCII shape that fails honest data.
PROPERTY_OK = re.compile(r"^(?=.*[^\W\d_])\w{1,64}$", re.UNICODE)

#: A trailing index on a repeated field. Templates spell a list as
#: `subdivision_name1`, `subdivision_name2` and so on, which makes seven
#: properties out of one and hides the fact that they are the same field. This
#: is 3,929 of the properties and **25.0% of all facts**, so collapsing them is
#: the single biggest thing that can be done to the shape of this data.
#: `[^\W\d]` rather than `[a-z_]`: the base has to end in a letter, and now
#: that German template keys survive, some of those letters are not ASCII.
INDEXED = re.compile(r"^(.*[^\W\d])(\d{1,2})$", re.UNICODE)

#: Values that are a number, with or without thousands separators. 20.5% of
#: values are one, and stored as text they sort lexically - "9" after "10".
NUMBER = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^[-+]?\d+(?:\.\d+)?$")
URL = re.compile(r"^(?:https?://|www\.)\S+$")
#: Enough date shapes to cover the 3.8% that are dates, and no more. A parser
#: that guesses at "9th century" would be inventing precision.
DATES = (re.compile(r"^(\d{1,2}) (\w+) (\d{3,4})$"),
         re.compile(r"^(\w+) (\d{1,2}),? (\d{3,4})$"),
         re.compile(r"^(\d{3,4})-(\d{2})-(\d{2})$"))
MONTHS = {m: i for i, m in enumerate((
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
), start=1)}

#: What a value turned out to be. A CHECK constraint pins these, so a new kind
#: cannot be introduced by a typo in one branch of the classifier.
KINDS = ("text", "number", "date", "url")


def value_kind(value: str) -> tuple[str, float | None]:
    """Classify a value, and pull out the number in it where there is one.

    ``num`` carries the numeric value for a number and the year for a date,
    which is what makes "before 1900" and "over 50,000" expressible in SQL at
    all. The text is left as written either way: these are facts an oracle
    reads out, and "1,000,000" is better said than 1000000.0.
    """
    if URL.match(value):
        return "url", None
    if NUMBER.match(value):
        return "number", float(value.replace(",", ""))
    for pattern in DATES:
        m = pattern.match(value)
        if not m:
            continue
        parts = m.groups()
        if pattern is DATES[2]:
            return "date", float(parts[0])
        month = parts[1] if pattern is DATES[0] else parts[0]
        if month.lower() in MONTHS:
            return "date", float(parts[2])
    return "text", None


def normalize_property(key: str) -> str | None:
    """A raw infobox key, cleaned, or None if it is not a property at all.

    Only the *form* is settled here - the shape, the case, the whitespace.
    What a property **means** is libgraph's job, because meaning is a
    judgement about this corpus and form is not.
    """
    key = SEPARATORS.sub("_", clean(key).strip().lower()).strip("_")
    if not PROPERTY_OK.match(key) or FURNITURE.search(key):
        return None
    return key


def index_families(names: Iterable[str]) -> dict[str, tuple[str, int]]:
    """raw name -> (property, ordinal), for the names that are a series.

    A trailing digit is not enough to go on. `subdivision_name2` is the second
    subdivision; `area_km2` is square kilometres, and splitting it invents a
    field called `area_km` that no one wrote. The name cannot tell them apart -
    only the rest of the vocabulary can, by whether it agrees there is a
    series. So a base is only indexed when more than one index appears under
    it, or when the bare name is also in use.

    Over Simple English Wikipedia that separates 743 real series (496,000
    facts, re-indexed) from 137 lone names (42,541 facts, left exactly as
    written) - and every one of those 137 turns out to end in `km2`.
    """
    names = list(names)
    present = set(names)
    seen: dict[str, set[int]] = {}
    for name in names:
        m = INDEXED.match(name)
        if m:
            seen.setdefault(m.group(1), set()).add(int(m.group(2)))

    out: dict[str, tuple[str, int]] = {}
    for name in names:
        m = INDEXED.match(name)
        if not m:
            continue
        base, index = m.group(1), int(m.group(2))
        if len(seen[base]) > 1 or base in present:
            out[name] = (base, index)
    return out


#: Leftovers that survive `clean()` at the edges of a value: a list marker, a
#: dangling separator, an unterminated tag the ref pass could not pair up.
EDGES = re.compile("^[\\s*#,;:.|\\-\u2013\u2014]+|[\\s,;:|]+$")
UNTERMINATED = re.compile(r"<[a-zA-Z/][^>]*$")


def normalize_value(value: str) -> str:
    """Trim the edges `clean()` leaves behind.

    0.7% of values began with a bullet or a dash and another 0.7% ended in
    stray punctuation - small shares that are entirely concentrated in the
    fields anyone would actually query, because a list is how an infobox says
    "more than one".
    """
    value = UNTERMINATED.sub("", value)
    return EDGES.sub("", value).strip()


def infobox_body(markup: str) -> str | None:
    """The text between the first Infobox's braces, nesting respected."""
    m = INFOBOX.search(markup)
    if not m:
        return None
    depth, i, start = 0, m.start(), m.start()
    while i < len(markup):
        if markup.startswith("{{", i):
            depth += 1
            i += 2
        elif markup.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                return markup[start + 2:i - 2]
        else:
            i += 1
    return None


def split_fields(body: str) -> list[str]:
    """An infobox body cut into its top-level ``key = value`` fields.

    Split by hand rather than by regex: a value may itself contain ``|`` inside
    a nested template or a piped link, and splitting on those turns one field
    into several fragments, none of which is a fact.

    Separate from ``infobox_fields`` so that anything wanting to look at what a
    value held *before* cleaning - which templates it used, say - reads the
    fields the ingest actually sees rather than its own second copy of this
    rule. ``[0]`` is the template name, not a field.
    """
    body = COMMENT.sub("", body)
    fields: list[str] = []
    brace = brack = 0
    current: list[str] = []
    for i, ch in enumerate(body):
        if body.startswith("{{", i):
            brace += 1
        elif body.startswith("}}", i):
            brace = max(0, brace - 1)
        elif body.startswith("[[", i):
            brack += 1
        elif body.startswith("]]", i):
            brack = max(0, brack - 1)

        if ch == "|" and brace == 0 and brack == 0:
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
    fields.append("".join(current))
    return fields


def infobox_fields(body: str, title: str = "") -> list[tuple[str, str]]:
    """Top-level ``| key = value`` pairs, cleaned but not yet re-indexed.

    A field whose value is the article's own title is dropped: 91,687 of them,
    4.4% of every fact, and not one says anything. `name = Televisa` on the
    page called Televisa is the commonest by far at 69,259, with
    `official_name`, `fullname`, `subject_name` and `municipality_name` behind it.

    It is the same emptiness as `Los Angeles located_in Los Angeles`, which
    libgraph drops at the edge; this drops it a step earlier, where it is a
    fact about nothing rather than an edge to nowhere.

    Splitting an index off a repeated field needs the whole vocabulary - see
    ``index_families`` - so it happens once the dump has been read, not here.
    """
    fields = split_fields(body)

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for field in fields[1:]:                      # [0] is the template name
        key, sep, value = field.partition("=")
        if not sep:
            continue
        named = normalize_property(key)
        if named is None or named in seen:
            continue
        key = named
        # Templates first, because the cleaner deletes them: a value written as
        # `{{birth date|1847|3|3}}` has to become text before a rule that
        # strips every `{{...}}` sees it. Only the templates in
        # `VALUE_TEMPLATES` change; the rest are handed on untouched.
        #
        # Then the same cleaner the lead uses, so entities are resolved before
        # tags are stripped rather than after - the bug that put "<ref></ref>"
        # on screen would otherwise put "&lt;br&gt;" in every multi-part value.
        value = normalize_value(clean(unexpanded(expand_templates(value))))
        if not value or JUNK_VALUE.match(value) or len(value) > MAX_VALUE_LEN:
            continue
        if value == title:
            continue
        seen.add(key)
        out.append((key, value))
    return out


def raw_pages(path: Path) -> Iterator[tuple[str, str | None, str]]:
    """Yield (title, redirect_target_or_None, markup) for every ns0 page.

    Reading a dump and interpreting one are separate jobs, and this is the
    first. Anything that wants the markup before the ingest's opinions are
    applied to it - counting which templates a value used, say - gets it here
    rather than keeping a second copy of a loop whose every line is load
    bearing. A redirect carries no markup worth reading, so it yields "".
    """
    opener = bz2.open if path.suffix == ".bz2" else open
    title = ns = redirect = None
    in_text = False
    body: list[str] = []

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "<page>" in line:
                title = ns = redirect = None
                in_text = False
                body = []
                # Deliberately falling through rather than continuing: <page>
                # and <title> can share a line, and skipping the rest of it
                # loses the title and with it the whole article.

            # Every pattern gets a look at every line. A Wikimedia dump puts
            # one element per line, but nothing in the format promises that,
            # and skipping the rest of a line after the first match makes the
            # parser silently yield nothing on a dump that packs them.
            # Both are unescaped here, at the one place a title enters this
            # program: `article`, `redirect`, `category` and `fact` all key on
            # what this yields, and a redirect target has to be decoded with
            # the title it points at or it stops resolving.
            if title is None:
                m = TITLE.search(line)
                if m:
                    title = unescape_title(m.group(1))
            if ns is None:
                m = NS.search(line)
                if m:
                    ns = int(m.group(1))
            if redirect is None:
                m = REDIRECT.search(line)
                if m:
                    redirect = unescape_title(m.group(1))

            if not in_text:
                m = TEXT_OPEN.search(line)
                if not m:
                    continue
                in_text = True
                line = line[m.end():]
            # Generous, because an infobox alone can run to several kilobytes
            # and a template only strips if its closing braces were captured.
            # Capped so a pathological article cannot be read into memory whole.
            if len(body) < 400:
                body.append(line.split("</text>")[0])

            if "</page>" in line and title is not None:
                if ns == 0:
                    if redirect:
                        yield title, redirect, ""
                    else:
                        yield title, None, "".join(body)
                title = None


#: `[[Category:Cities in France]]`, up to its sort key. Case varies by author
#: and MediaWiki does not care, so neither does this.
CATEGORY = re.compile(r"\[\[\s*category\s*:([^\]|]+)", re.IGNORECASE)


def categories_of(markup: str) -> list[str]:
    """The categories a page files itself under, in the order it names them.

    Read from the raw markup rather than the cleaned text, because `clean()`
    drops them - correctly, since a category is not a sentence and the lead is
    prose. They are kept anyway: an infobox says what a page's author chose to
    tabulate, and a category says what they chose to file it under, and the
    second is often the only place a containment is written down. `Infobox
    U.S. state` has no country field at all, so Michigan records that it is in
    the United States only in `1837 establishments in the United States`.
    """
    seen: dict[str, None] = {}
    for name in CATEGORY.findall(markup):
        cleaned = normalize_value(clean(name))
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


@dataclass
class Page:
    """One ns0 page, as much of it as this ingest reads.

    A tuple until there were five of them, at which point the call site stopped
    saying which was which.
    """

    title: str
    #: The page a redirect points at, or None for an article.
    redirect: str | None
    lead: str
    fields: list[tuple[str, str]]
    categories: list[str]


def pages(path: Path) -> Iterator[Page]:
    """Yield a `Page` per ns0 page.

    A redirect carries no lead, no fields and no categories worth reading, so
    all three are empty for one.
    """
    for title, redirect, markup in raw_pages(path):
        if redirect:
            yield Page(title, redirect, "", [], [])
            continue
        box = infobox_body(markup)
        yield Page(title, None, lead_of(markup),
                   infobox_fields(box, title) if box else [],
                   categories_of(markup))


# --- sitelinks ----------------------------------------------------------------
#
# The XML dump says what an article contains and never says what it is *about*.
# That identity lives in `page_props`, as `wikibase_item`, and it is the only
# exact key between this corpus and Wikidata - the English label is not one,
# because 91.6M Wikidata entities share 83.5M labels and `Paris` is 236 of them.
#
# Two dumps rather than one, because MediaWiki normalizes: `page_props` keys
# `wikibase_item` by page id, and only `page` knows which title a page id is.

#: `CREATE TABLE` in a MediaWiki SQL dump, for reading the column order out of
#: the file rather than hard-coding it. MediaWiki adds and removes columns
#: between releases - `page_restrictions` was dropped from `page` - so a
#: position that is right today is a silent mis-parse two dumps from now.
SQL_CREATE = re.compile(r"CREATE TABLE `\w+` \((.*?)\n\)", re.DOTALL)
SQL_COLUMN = re.compile(r"^\s*`(\w+)`")

#: mysqldump writes these as escapes; everything else after a backslash is the
#: character itself.
SQL_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "Z": "\x1a"}


def _sql_open(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def sql_columns(path: Path) -> list[str]:
    """The dump's column names, in the order its rows are written."""
    with _sql_open(path) as fh:
        head = fh.read(1 << 18)
    body = SQL_CREATE.search(head)
    if not body:
        raise SystemExit(f"{path.name} has no CREATE TABLE - is it a table dump?")
    return [m.group(1) for line in body.group(1).split("\n")
            if (m := SQL_COLUMN.match(line))]


def sql_values(payload: str) -> Iterator[list[str | None]]:
    """Split one ``VALUES`` payload into rows.

    Hand-scanned rather than split on punctuation: the values contain commas,
    parentheses and escaped quotes inside string literals, and every one of
    those is a title somebody has written. `Paris (song)` is a real article.

    A quoted empty string and an unquoted NULL are different things and are
    kept different, which is why the quoting is tracked rather than the text
    compared to ``'NULL'`` afterwards.
    """
    i, n = 0, len(payload)
    while (i := payload.find("(", i)) >= 0:
        i += 1
        row: list[str | None] = []
        field: list[str] = []
        quoted = in_string = False
        while i < n:
            ch = payload[i]
            if in_string:
                if ch == "\\":
                    field.append(SQL_ESCAPES.get(payload[i + 1], payload[i + 1]))
                    i += 2
                elif ch == "'":
                    in_string = False
                    i += 1
                else:
                    field.append(ch)
                    i += 1
            elif ch == "'":
                in_string = quoted = True
                i += 1
            elif ch in ",)":
                text = "".join(field)
                row.append(text if quoted else None if text == "NULL" else text)
                field, quoted = [], False
                i += 1
                if ch == ")":
                    break
            else:
                field.append(ch)
                i += 1
        yield row


def sql_rows(path: Path, *wanted: str) -> Iterator[tuple[str | None, ...]]:
    """Yield the named columns of every row, in the order asked for."""
    names = sql_columns(path)
    if missing := [w for w in wanted if w not in names]:
        raise SystemExit(
            f"{path.name} has no column {', '.join(missing)}. It has: "
            f"{', '.join(names)}")
    picks = [names.index(w) for w in wanted]
    width = len(names)
    with _sql_open(path) as fh:
        for line in fh:
            if not line.startswith("INSERT INTO"):
                continue
            for row in sql_values(line[line.index("VALUES") + 6:]):
                # A short row is this parser being wrong, not the dump being
                # unusual, and skipping it would lose articles quietly.
                if len(row) != width:
                    raise SystemExit(
                        f"{path.name}: parsed {len(row)} values for a row of "
                        f"{width} columns: {row!r}")
                yield tuple(row[p] for p in picks)


def sitelinks(page_dump: Path, props_dump: Path) -> Iterator[tuple[str, int]]:
    """(title, Q-id) for every ns0 article that names a Wikidata entity.

    Redirects are skipped. A redirect is an alternate name for a page that has
    its own row, so where one carries a Q-id at all it is the same entity
    twice, and `redirect` already records the relationship better.
    """
    if (a := DUMP_NAME.match(page_dump.name)) and \
            (b := DUMP_NAME.match(props_dump.name)) and a["date"] != b["date"]:
        # Page ids are stable enough that mixing snapshots mostly works, which
        # is exactly what makes it worth refusing: it fails on the pages that
        # were deleted and recreated between the two, and nowhere else.
        raise SystemExit(
            f"{page_dump.name} is from {a['date']} and {props_dump.name} is "
            f"from {b['date']}; a page id only means the same thing within one "
            f"snapshot")

    titles: dict[str, str] = {}
    for page_id, ns, title, is_redirect in sql_rows(
            page_dump, "page_id", "page_namespace", "page_title",
            "page_is_redirect"):
        if ns != "0" or is_redirect != "0" or page_id is None or title is None:
            continue
        # Stored with underscores in the table and with spaces everywhere else,
        # including in the XML dump's <title>.
        titles[page_id] = title.replace("_", " ")

    for page_id, prop, value in sql_rows(
            props_dump, "pp_page", "pp_propname", "pp_value"):
        if prop != "wikibase_item" or not value or page_id is None:
            continue
        if (title := titles.get(page_id)) is not None:
            yield title, int(value.removeprefix("Q"))


def ingest_sitelinks(db: sqlite3.Connection, page_dump: Path, props_dump: Path,
                     source: str) -> tuple[int, int]:
    """Replace ``source``'s sitelinks. Returns (written, joined to an article).

    The second number is the one to read. A sitelink that names no article in
    this database is a row that can never be used, and a pile of them means the
    two dumps are from different snapshots of the same wiki.
    """
    # Materialized before the transaction opens rather than streamed into it:
    # the parse is the slow half, and there is no reason to hold a write lock
    # across it. 284,000 pairs is about 40MB.
    rows = [(source, title, qid) for title, qid in sitelinks(page_dump, props_dump)]

    with db:
        db.execute("DELETE FROM sitelink WHERE source = ?", (source,))
        db.executemany("INSERT OR REPLACE INTO sitelink VALUES (?, ?, ?)", rows)
        joined, = db.execute(
            "SELECT COUNT(*) FROM sitelink s JOIN article a "
            "ON a.source = s.source AND a.title = s.title "
            "WHERE s.source = ?", (source,)).fetchone()
        provenance = [
            (f"{source}.sitelinks", str(len(rows))),
            (f"{source}.sitelinks.dump", props_dump.name),
            (f"{source}.sitelinks.digest", file_digest(props_dump)),
            (f"{source}.sitelinks.ingested", time.strftime("%Y-%m-%dT%H:%M:%S")),
        ]
        if url := dump_url(props_dump.name):
            provenance.insert(2, (f"{source}.sitelinks.url", url))
        for key, value in provenance:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    return len(rows), joined


def connect(path: Path, migrate: bool = False) -> sqlite3.Connection:
    """Open the database, creating or replacing its tables as needed.

    ``PRAGMA user_version`` carries the schema version, so a database written
    by an older layout is recognised rather than read as if it were current -
    ``CREATE TABLE IF NOT EXISTS`` would otherwise leave the old definitions in
    place and quietly ignore the new ones.

    ``migrate`` rebuilds the tables when the version has moved. Only ingest
    passes it, and only because ingest is about to replace every row anyway;
    nothing is lost that another four minutes cannot rebuild.
    """
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")

    found = db.execute("PRAGMA user_version").fetchone()[0]
    extra, missing = _drift(db)
    if found and (found != SCHEMA_VERSION or extra or missing):
        if not migrate:
            detail = []
            if found != SCHEMA_VERSION:
                detail.append(f"it is version {found}, this is {SCHEMA_VERSION}")
            if extra:
                detail.append(f"it has {', '.join(sorted(extra))}, "
                              f"which this schema does not define")
            if missing:
                detail.append(f"it is missing {', '.join(sorted(missing))}")
            raise SystemExit(
                f"{path} does not match this schema:\n  "
                + "\n  ".join(detail)
                + "\nRe-ingest the dump to rebuild it.")
        # Every user table, read back from the database rather than listed
        # here: a hand-written list has to be updated whenever the schema gains
        # one, and when it is not, the old table survives and
        # `CREATE TABLE IF NOT EXISTS` accepts it silently.
        stale = [name for (name,) in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'")]
        for table in stale:
            db.execute(f"DROP TABLE IF EXISTS {table}")

    db.executescript(_schema())
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return db


def _expected_objects() -> set[str]:
    """Table and index names the current schema defines."""
    return set(re.findall(r"CREATE (?:TABLE|INDEX) IF NOT EXISTS (\w+)",
                          _schema()))


def _drift(db: sqlite3.Connection) -> tuple[set[str], set[str]]:
    """(objects the database has and the schema does not, and the reverse).

    Checked in addition to the version, because a version only catches a change
    somebody remembered to declare. Removing an index and bumping the version
    is not enough on its own: a database that a *previous, buggy* migration
    already stamped with the new version passes the version check and keeps the
    index forever. That happened here - an 86MB index survived its own removal
    twice - and comparing the schema to the database is what notices.
    """
    present = {name for (name,) in db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'index') "
        "AND name NOT LIKE 'sqlite_%'")}
    expected = _expected_objects()
    return present - expected, expected - present


def ingest(db: sqlite3.Connection, dump: Path,
           source: str) -> tuple[int, int, int]:
    """Replace ``source``'s rows with the contents of ``dump``, atomically."""
    db.execute("CREATE TEMP TABLE new_article (title TEXT PRIMARY KEY, lead TEXT)")
    db.execute("CREATE TEMP TABLE new_redirect (title TEXT PRIMARY KEY, target TEXT)")
    db.execute("CREATE TEMP TABLE new_fact "
               "(subject TEXT, property TEXT, ordinal INTEGER, value TEXT, "
               " kind TEXT, num REAL, "
               " PRIMARY KEY (subject, property, ordinal))")
    db.execute("CREATE TEMP TABLE new_category "
               "(title TEXT, name TEXT, PRIMARY KEY (title, name))")

    articles = redirects = facts = filings = 0
    started = time.time()
    for page in pages(dump):
        if page.redirect:
            db.execute("INSERT OR REPLACE INTO new_redirect VALUES (?, ?)",
                       (page.title, page.redirect))
            redirects += 1
        else:
            db.execute("INSERT OR REPLACE INTO new_article VALUES (?, ?)",
                       (page.title, page.lead))
            articles += 1
            if page.fields:
                db.executemany(
                    "INSERT OR REPLACE INTO new_fact VALUES (?, ?, 0, ?, ?, ?)",
                    [(page.title, key, value, *value_kind(value))
                     for key, value in page.fields])
                facts += len(page.fields)
            if page.categories:
                db.executemany(
                    "INSERT OR REPLACE INTO new_category VALUES (?, ?)",
                    [(page.title, name) for name in page.categories])
                filings += len(page.categories)
        total = articles + redirects
        if total % 50_000 == 0:
            rate = total / (time.time() - started)
            print(f"  {total:,} pages, {facts:,} facts ({rate:,.0f}/s)", flush=True)

    # Now that the whole vocabulary is known, split the indices off the
    # repeated fields. UPDATE OR REPLACE rather than UPDATE: `tubeexits3` and
    # `tubeexits03` both land on ordinal 3, and one page carrying each would
    # otherwise abort the run on a primary key it cannot satisfy.
    renames = index_families(
        n for (n,) in db.execute("SELECT DISTINCT property FROM new_fact"))
    for raw, (base, ordinal) in renames.items():
        db.execute("UPDATE OR REPLACE new_fact SET property = ?, ordinal = ? "
                   "WHERE property = ?", (base, ordinal, raw))
    if renames:
        print(f"  {len(renames):,} indexed properties folded into "
              f"{len({b for b, _ in renames.values()}):,} fields", flush=True)
    # Re-counted rather than carried down from the loop: the fold can replace a
    # row, so the number written into `meta` has to be the number of rows the
    # table ends up holding, not the number of fields the parser saw.
    facts, = db.execute("SELECT COUNT(*) FROM new_fact").fetchone()

    # One transaction: an interrupted run leaves the old corpus untouched.
    with db:
        db.execute("DELETE FROM article WHERE source = ?", (source,))
        db.execute("DELETE FROM redirect WHERE source = ?", (source,))
        db.execute("DELETE FROM fact WHERE source = ?", (source,))
        db.execute("DELETE FROM category WHERE source = ?", (source,))
        db.execute("INSERT INTO article (source, title, lead) "
                   "SELECT ?, title, lead FROM new_article", (source,))
        db.execute("INSERT INTO redirect (source, title, target) "
                   "SELECT ?, title, target FROM new_redirect", (source,))
        db.execute("INSERT INTO category (source, title, name) "
                   "SELECT ?, title, name FROM new_category", (source,))
        db.execute("DELETE FROM property WHERE source = ?", (source,))
        db.execute(
            "INSERT INTO fact (source, subject, property, ordinal, value, "
            "                  kind, num) "
            "SELECT ?, subject, property, ordinal, value, kind, num "
            "FROM new_fact", (source,))
        # The vocabulary, recorded rather than inferred. 13,268 properties is a
        # folksonomy nobody wrote down, and the query that matters -
        # "what is used a lot and mapped to nothing" - is the one that found
        # `subdivision_name` and took chaining from 1.7% to 40.7%. Leaving it
        # to be rediscovered by whoever next reads a dump is how it stays lost.
        db.execute(
            "INSERT INTO property (source, name, uses, subjects, relation) "
            "SELECT ?, property, COUNT(*), COUNT(DISTINCT subject), NULL "
            "FROM new_fact GROUP BY property", (source,))
        db.executemany(
            "UPDATE property SET relation = ? WHERE source = ? AND name = ?",
            [(relation, source, name)
             for name, (relation, _rank) in libgraph.FIELD_RELATION.items()])
        # Provenance, so the database says which snapshot it holds and where
        # that snapshot can be fetched again. A filename alone does not: it
        # says what the file was called on one machine.
        provenance = [
            ("schema_version", str(SCHEMA_VERSION)),
            (f"{source}.dump", dump.name),
            (f"{source}.digest", file_digest(dump)),
            (f"{source}.ingested", time.strftime("%Y-%m-%dT%H:%M:%S")),
            (f"{source}.articles", str(articles)),
            (f"{source}.redirects", str(redirects)),
            (f"{source}.facts", str(facts)),
        ]
        url = dump_url(dump.name)
        if url:
            provenance.insert(2, (f"{source}.url", url))
        for key, value in provenance:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    db.execute("DROP TABLE new_article")
    db.execute("DROP TABLE new_redirect")
    db.execute("DROP TABLE new_fact")
    db.execute("DROP TABLE new_category")
    return articles, redirects, facts


def stats(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
    if not rows:
        print("empty database")
        return
    for key, value in rows:
        print(f"  {key:<28} {value}")

    print()
    for (source,) in db.execute("SELECT DISTINCT source FROM article"):
        a = db.execute("SELECT COUNT(*) FROM article WHERE source = ?",
                       (source,)).fetchone()[0]
        r = db.execute("SELECT COUNT(*) FROM redirect WHERE source = ?",
                       (source,)).fetchone()[0]
        resolved = db.execute(
            "SELECT COUNT(*) FROM redirect r JOIN article a "
            "ON a.source = r.source AND a.title = r.target WHERE r.source = ?",
            (source,)).fetchone()[0]
        chars = db.execute("SELECT SUM(LENGTH(lead)) FROM article WHERE source = ?",
                           (source,)).fetchone()[0] or 0
        print(f"  {source}: {a:,} articles, {r:,} redirects "
              f"({resolved / r:.1%} resolve), {chars / 1e6:.0f} MB of lead text")

        linked, joined = db.execute(
            "SELECT COUNT(*), COUNT(a.title) FROM sitelink s "
            "LEFT JOIN article a ON a.source = s.source AND a.title = s.title "
            "WHERE s.source = ?", (source,)).fetchone()
        if linked:
            # Orphans are the staleness alarm. The sitelink dump and the XML
            # dump are separate downloads on the same cadence, and nothing else
            # here would notice one being a snapshot behind the other.
            orphans = linked - joined
            note = f", {orphans:,} name no article" if orphans else ""
            # One decimal place, not none: the gap this number exists to show
            # is a few hundred rows in 284,000, and `.0%` rounds every one of
            # them away and prints 100%.
            print(f"  {' ' * len(source)}  {joined:,} sitelinks "
                  f"({joined / a:.1%} of articles){note}")
            article_dump = db.execute(
                "SELECT value FROM meta WHERE key = ?", (f"{source}.dump",)
            ).fetchone()
            link_dump = db.execute(
                "SELECT value FROM meta WHERE key = ?",
                (f"{source}.sitelinks.dump",)).fetchone()
            dates = [m["date"] for row in (article_dump, link_dump)
                     if row and (m := DUMP_NAME.match(row[0]))]
            if len(dates) == 2 and dates[0] != dates[1]:
                print(f"  {' ' * len(source)}  WARNING: articles are from "
                      f"{dates[0]} and sitelinks from {dates[1]}")

        filings, filed, cats = db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT title), COUNT(DISTINCT name) "
            "FROM category WHERE source = ?", (source,)).fetchone()
        if filings:
            # Printed beside the infobox coverage because the gap between them
            # is the point: a category is the only containment most pages have.
            print(f"  {' ' * len(source)}  {filings:,} category filings over "
                  f"{filed:,} articles ({filed / a:.0%}), {cats:,} categories")

        f, subjects, properties = db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT subject), COUNT(DISTINCT property) "
            "FROM fact WHERE source = ?", (source,)).fetchone()
        if f:
            print(f"  {' ' * len(source)}  {f:,} facts over {subjects:,} "
                  f"subjects ({subjects / a:.0%} of articles), "
                  f"{properties:,} properties")
            pad = " " * len(source)
            # From `property` rather than a GROUP BY over two million rows:
            # counting the vocabulary once at ingest is what that table is for.
            top = db.execute(
                "SELECT name, uses FROM property WHERE source = ? "
                "ORDER BY uses DESC LIMIT 8", (source,)).fetchall()
            print(f"  {pad}  most common: "
                  + ", ".join(f"{p} ({n:,})" for p, n in top))

            kinds = db.execute(
                "SELECT kind, COUNT(*) n FROM fact WHERE source = ? "
                "GROUP BY kind ORDER BY n DESC", (source,)).fetchall()
            print(f"  {pad}  values: "
                  + ", ".join(f"{k} {n / f:.0%}" for k, n in kinds))

            # The question this database exists to keep answerable: what is
            # used a lot and understood by nothing? Every entry is a candidate
            # for libgraph.CANONICAL, and the last one worth adding took
            # chaining from 1.7% to 40.7%.
            unmapped = db.execute(
                "SELECT name, uses FROM property WHERE source = ? "
                "AND relation IS NULL ORDER BY uses DESC LIMIT 8",
                (source,)).fetchall()
            mapped, = db.execute(
                "SELECT COUNT(*) FROM property WHERE source = ? "
                "AND relation IS NOT NULL", (source,)).fetchone()
            print(f"  {pad}  {mapped} properties map to a relation; "
                  f"biggest unmapped: "
                  + ", ".join(f"{p} ({n:,})" for p, n in unmapped))


def _sitelink_pass(db: sqlite3.Connection, dumps: list[Path],
                   source: str) -> None:
    """Ingest the sitelinks and say what joined, because the count is the test.

    A sitelink dump from the wrong snapshot parses perfectly and writes rows
    that match nothing, so the only thing that says it worked is how many of
    them found an article.
    """
    page_dump, props_dump = dumps
    print(f"\nreading sitelinks from {props_dump.name}")
    written, joined = ingest_sitelinks(db, page_dump, props_dump, source)
    articles, = db.execute("SELECT COUNT(*) FROM article WHERE source = ?",
                           (source,)).fetchone()
    print(f"{written:,} sitelinks, {joined:,} of them joined to an article"
          + (f" ({joined / articles:.1%} of the corpus)" if articles else ""))
    if written and joined / max(written, 1) < 0.9:
        print("  that is low enough to mean the two dumps disagree; check "
              "they are the same snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", nargs="?", type=Path,
                        help="pages-articles XML dump (.xml or .xml.bz2)")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--source", default="simplewiki",
                        help="Corpus name, so several can share one database")
    parser.add_argument("--sitelinks", nargs=2, type=Path,
                        metavar=("PAGE.SQL.GZ", "PAGE_PROPS.SQL.GZ"),
                        help="Wikidata Q-ids for each article, from the same "
                             "snapshot as the XML dump. Runs on its own "
                             "against an existing database.")
    parser.add_argument("--stats", action="store_true",
                        help="Report what the database holds and exit")
    args = parser.parse_args()

    db = connect(args.db, migrate=bool(args.dump) and not args.stats)
    if args.stats or not (args.dump or args.sitelinks):
        stats(db)
        return

    for path in [args.dump, *(args.sitelinks or ())]:
        if path and not path.exists():
            raise SystemExit(f"no such dump: {path}")

    if not args.dump:
        _sitelink_pass(db, args.sitelinks, args.source)
        db.commit()
        print()
        stats(db)
        return

    print(f"ingesting {args.dump.name} as '{args.source}' into {args.db}")
    articles, redirects, facts = ingest(db, args.dump, args.source)
    print(f"\n{articles:,} articles, {redirects:,} redirects, {facts:,} facts")

    if args.sitelinks:
        _sitelink_pass(db, args.sitelinks, args.source)

    # The walkable graph, built from the facts: property synonyms collapsed
    # onto canonical relations and values resolved to article titles. Derived
    # rather than stored twice, so it cannot disagree with the facts it came
    # from - and rebuilt here because it is cheap next to the parse.
    print("\nbuilding the graph...")
    edges, _dropped = libgraph.build(db, args.source, report=print)
    db.commit()
    db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
               (f"{args.source}.edges", str(edges)))
    db.commit()
    print()
    stats(db)
    # ANALYZE so the planner has real distributions: the notability ranking in
    # buildwikisearch is a correlated subquery per article, and it should use
    # redirect_target rather than guess. VACUUM afterwards to give the file
    # back the space the replaced rows left behind.
    db.execute("ANALYZE")
    db.commit()
    db.execute("VACUUM")
    print(f"\n{args.db} is {args.db.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
