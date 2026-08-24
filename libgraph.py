"""
Walking the fact graph: chaining, inverses, and the counting that comes free.

``fact`` holds what an infobox said. This is the graph you can traverse: the
same information with its property names collapsed onto canonical relations and
its values resolved to article titles, so a hop can continue.

## Why the collapsing is the whole job

Infoboxes are written by thousands of people reaching for hundreds of
templates, so the property vocabulary is a folksonomy rather than a schema. A
city records what country it is in as ``subdivision_name``, or ``state``, or
``country``, depending on which template the author used. A chain asking for
``country`` therefore fails on a graph that plainly contains the answer.

The ingest normalizes the *form* of a property - splitting the index off
``subdivision_name1``, checking the shape, typing the value. What a property
**means** is decided here instead, because meaning is a judgement about this
particular corpus and form is not.

Measured over Simple English Wikipedia, collapsing those synonyms onto one
``located_in`` relation takes ``birth_place -> country`` from **1.7%** of
subjects completing to **40.7%**, and ``death_place -> country`` from 1.0% to
48.2%. The graph was never sparse. It was inconsistently named.

## Asking for a type, not a number of hops

A question asks for a country, not for two hops. How many hops that takes is a
property of the graph: for the 42,033 people this corpus records a birthplace
for, the birthplace is *already* a country 26.2% of the time, one hop away
35.7%, two hops 1.6%. A fixed two-hop path is right for about a third of them
and destroys a correct answer for the quarter that needed none.

So `in_country` climbs `located_in` until the value is something the corpus
calls a country, and stops immediately if it already is. That takes the share
of people for whom a country can be named at all from **30.6% to 45.6%**.

## What this can and cannot do

Traversal is lookups, so chaining, inverses and counting all fall out of the
same table. Inference does not: there is no rule engine here, and no way to
conclude anything that is not stored or directly derivable from what is.

Even with the collapsing, roughly half of two-hop questions fail because a hop
lands on an article that has no infobox at all - 54% of them do not. That is a
property of the corpus, not of the traversal, and it is why anything built on
this answers confidently about some things and not at all about others.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

#: Canonical relation -> the infobox fields that mean it, most specific first.
#: A subject keeps the best-ranked field it has, so a page carrying both
#: ``country`` and ``subdivision_name`` contributes one edge rather than two
#: that disagree.
CANONICAL: dict[str, tuple[str, ...]] = {
    # `subdivision_name1` and `subdivision_name2` used to be listed here as
    # separate fields. The ingest now splits the index off into `fact.ordinal`,
    # so they arrive as one property and the ordinal decides which is preferred.
    "located_in": (
        "country", "subdivision_name", "state", "province", "region",
        "municipality_name", "arrondissement", "canton", "department",
        "district", "county", "prefecture",
    ),
    "born_in": ("birth_place", "place_of_birth", "birthplace"),
    "died_in": ("death_place", "place_of_death", "deathplace"),
    "capital_is": ("capital", "capital_city"),
    "created_by": (
        "director", "author", "writer", "creator", "composer", "developer",
        "designer", "artist", "producer", "founder", "architect",
    ),
    "spouse_of": ("spouse", "partner"),
    "member_of": ("party", "league", "team", "employer", "organization"),
    "language_is": ("language", "languages", "official_language"),
    "genre_is": ("genre", "genres"),
    "preceded_by": ("predecessor", "before"),
    "followed_by": ("successor", "after"),
}

#: field -> (relation, how specific it is)
FIELD_RELATION: dict[str, tuple[str, int]] = {
    field_name: (relation, rank)
    for relation, fields in CANONICAL.items()
    for rank, field_name in enumerate(fields)
}

#: Pseudo-relations that climb rather than step: repeat a relation until the
#: value is of a given type.
#:
#: "What country was X born in" does not ask for one more hop. It asks for an
#: answer of type country, and how many hops that takes is a property of the
#: graph rather than of the question. Measured over the 42,033 people this
#: corpus records a birthplace for, the distance from that birthplace to a
#: country is 0 hops for 26.2% of them, 1 hop for 35.7% and 2 for 1.6%. A fixed
#: two-hop path is right for about a third and actively destroys the answer for
#: the quarter whose birthplace was already the country.
CLIMB: dict[str, tuple[str, str]] = {
    "in_country": ("located_in", "country"),
}

#: How many times a climb may step before giving up. Containment hierarchies
#: are shallow, and a cycle in the data - two places each inside the other -
#: would otherwise not terminate.
CLIMB_LIMIT = 6

#: An entity has to be named a country by this many independent infoboxes for
#: the corpus to be taken at its word. At 3 it yields 182 countries; at 1 it
#: yields 353 and starts admitting things like "Washington (state)".
TYPE_FLOOR = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS edge (
    source   TEXT NOT NULL,
    subject  TEXT NOT NULL,
    relation TEXT NOT NULL,
    object   TEXT NOT NULL,
    PRIMARY KEY (source, subject, relation, object)
) {strict}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS edge_object ON edge (source, object, relation);

CREATE TABLE IF NOT EXISTS entity_type (
    source TEXT NOT NULL,
    kind   TEXT NOT NULL,
    entity TEXT NOT NULL,
    PRIMARY KEY (source, kind, entity)
) {strict}WITHOUT ROWID;
"""


def schema(strict: bool = True) -> str:
    return SCHEMA.format(strict="STRICT, " if strict else "")


@dataclass
class Resolver:
    """Turns an infobox value into an article title, or admits it cannot.

    Three readings, each more generous: the value as written, the value after a
    redirect, and each comma-separated segment in turn - "Edinburgh, Scotland"
    is a sentence about a place, not the name of the article about it.

    Walking every segment rather than only the first is worth **11% more
    edges** (132,726 to 147,377). Infobox values name a place from the inside
    out and the corpus rarely holds the innermost one, so insisting on it drops
    an edge whose outer segments were perfectly good.
    """

    titles: set[str]
    redirects: dict[str, str]

    def __call__(self, value: str) -> str | None:
        hit = self._exact(value)
        if hit:
            return hit

        # "Steventon Rectory, Hampshire, England" names three places, narrowest
        # first, and the corpus may hold any of them. Trying each in turn keeps
        # the most specific one it actually has - which for Jane Austen is
        # Hampshire, since Simple English Wikipedia has no article on the
        # rectory. That trades precision for an answer, and the trade is worth
        # naming: the oracle will say she was born in Hampshire, which is true
        # and less exact than the infobox was.
        for part in (p.strip() for p in value.split(",")):
            hit = self._exact(part)
            if hit:
                return hit
        return None

    def _exact(self, value: str) -> str | None:
        if value in self.titles:
            return value
        target = self.redirects.get(value)
        return target if target in self.titles else None


def build(db: sqlite3.Connection, source: str,
          report: Callable[[str], None] = lambda _m: None) -> tuple[int, int]:
    """Populate ``edge`` for one source from its facts. Returns (edges, dropped)."""
    titles = {t for (t,) in db.execute(
        "SELECT title FROM article WHERE source = ?", (source,))}
    redirects = dict(db.execute(
        "SELECT title, target FROM redirect WHERE source = ?", (source,)))
    resolve = Resolver(titles, redirects)

    # subject -> relation -> (rank, ordinal, value): the best-ranked field
    # wins, and within one field the lowest ordinal does. A template writes a
    # list as `subdivision_name1`, `subdivision_name2`; the ingest splits that
    # index into `ordinal`, so the tie-break that used to be spelled out in
    # CANONICAL as two separate fields is now just a smaller number.
    best: dict[str, dict[str, tuple[int, int, str]]] = {}
    for subject, prop, ordinal, value in db.execute(
            "SELECT subject, property, ordinal, value FROM fact "
            "WHERE source = ?", (source,)):
        hit = FIELD_RELATION.get(prop)
        if hit is None:
            continue
        relation, rank = hit
        slot = best.setdefault(subject, {})
        if relation not in slot or (rank, ordinal) < slot[relation][:2]:
            slot[relation] = (rank, ordinal, value)

    rows, dropped = [], 0
    for subject, relations in best.items():
        for relation, (_rank, _ordinal, value) in relations.items():
            target = resolve(value)
            if target is None:
                dropped += 1        # a value that names no article we hold
                continue
            rows.append((source, subject, relation, target))

    db.execute("DELETE FROM edge WHERE source = ?", (source,))
    db.executemany("INSERT OR REPLACE INTO edge VALUES (?, ?, ?, ?)", rows)
    report(f"  {len(rows):,} edges, {dropped:,} values naming no article")

    kinds = types(db, source, resolve)
    db.execute("DELETE FROM entity_type WHERE source = ?", (source,))
    db.executemany("INSERT OR REPLACE INTO entity_type VALUES (?, ?, ?)",
                   [(source, kind, entity) for kind, entity in kinds])
    report(f"  {len(kinds):,} typed entities")
    return len(rows), dropped


#: The infobox field whose values name a type, per type. `country = France` is
#: an author asserting that France is a country; enough authors saying so is
#: the corpus defining the word, which beats a hand-written list that would go
#: stale and would not match this corpus's spellings.
TYPE_FIELD = {"country": "country"}


def types(db: sqlite3.Connection, source: str,
          resolve: Resolver) -> list[tuple[str, str]]:
    """(kind, entity) for everything the corpus repeatedly calls a kind.

    Collapsing `country` onto `located_in` is what made chaining work, and it
    threw away exactly what "what COUNTRY was X born in" needs. The signal
    survives in the fact table, which still records the field name.
    """
    counts: dict[tuple[str, str], int] = {}
    for kind, field_name in TYPE_FIELD.items():
        for (value,) in db.execute(
                "SELECT value FROM fact WHERE source = ? AND property = ?",
                (source, field_name)):
            target = resolve(value)
            if target:
                counts[(kind, target)] = counts.get((kind, target), 0) + 1
    return sorted(k for k, n in counts.items() if n >= TYPE_FLOOR)


def is_a(db: sqlite3.Connection, source: str, entity: str, kind: str) -> bool:
    return db.execute(
        "SELECT 1 FROM entity_type WHERE source = ? AND kind = ? AND entity = ?",
        (source, kind, entity)).fetchone() is not None


# --- traversal ----------------------------------------------------------------


@dataclass
class Answer:
    """What a walk found, and how far it got before it stopped."""

    value: str | None
    #: The subjects visited, so a caller can say *where* a chain broke - which
    #: is the difference between "I don't know" and "I know who directed it,
    #: but not where they were born".
    path: list[str] = field(default_factory=list)
    #: The relation that had no edge, when the walk stopped early.
    missing: str | None = None

    @property
    def complete(self) -> bool:
        return self.value is not None


def follow(db: sqlite3.Connection, source: str, subject: str,
           relations: list[str]) -> Answer:
    """Walk ``relations`` from ``subject``, one hop at a time.

    Each hop is one index lookup, so a three-hop chain is three lookups and
    nothing else - the traversal was never the expensive part. What stops a
    walk is an absent edge, and the answer says which one, because a partial
    path is worth reporting rather than discarding.
    """
    here: str = subject
    walked = [subject]
    for relation in relations:
        if relation in CLIMB:
            step, kind = CLIMB[relation]
            reached = _climb(db, source, here, walked, step, kind)
            if reached is None:
                return Answer(None, walked, relation)
            here = reached
            continue
        row = db.execute(
            "SELECT object FROM edge WHERE source = ? AND subject = ? "
            "AND relation = ? LIMIT 1", (source, here, relation)).fetchone()
        if row is None:
            return Answer(None, walked, relation)
        here = row[0]
        walked.append(here)
    return Answer(here, walked)


def _climb(db: sqlite3.Connection, source: str, here: str, walked: list[str],
           step: str, kind: str) -> str | None:
    """Repeat ``step`` until ``here`` is a ``kind``, or the trail runs out.

    The type check comes first, so an entity that is already what was asked for
    is returned rather than stepped past - which is the whole point, since a
    quarter of the birthplaces in this corpus are countries already.

    Intermediate nodes are appended to ``walked`` even when the climb fails, so
    a caller can report "born in Ulm, which is in Baden-Wurttemberg" rather
    than only that it did not find a country.
    """
    for _ in range(CLIMB_LIMIT):
        if is_a(db, source, here, kind):
            return here
        row = db.execute(
            "SELECT object FROM edge WHERE source = ? AND subject = ? "
            "AND relation = ? LIMIT 1", (source, here, step)).fetchone()
        if row is None:
            return None
        here = row[0]
        walked.append(here)
    return None


def inverse(db: sqlite3.Connection, source: str, obj: str,
            relation: str, limit: int = 20) -> list[str]:
    """Subjects pointing at ``obj`` - "who was born in Edinburgh".

    Free, because ``edge_object`` indexes the graph backwards. About a third of
    what a real question set asks is an inverse of something it could already
    answer forwards.
    """
    return [s for (s,) in db.execute(
        "SELECT subject FROM edge WHERE source = ? AND object = ? "
        "AND relation = ? LIMIT ?", (source, obj, relation, limit))]


def count(db: sqlite3.Connection, source: str, obj: str, relation: str) -> int:
    """How many subjects point at ``obj`` - "how many people were born here".

    Aggregation costs nothing once the inverse index exists, which makes it the
    cheapest capability here and the one most likely to be overlooked.
    """
    return int(db.execute(
        "SELECT COUNT(*) FROM edge WHERE source = ? AND object = ? "
        "AND relation = ?", (source, obj, relation)).fetchone()[0])
