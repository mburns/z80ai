"""
Walking the fact graph: chaining, inverses, and the counting that comes free.

``fact`` holds what an infobox said. This is the graph you can traverse: the
same information with its property names collapsed onto canonical relations and
its values resolved to article titles, so a hop can continue.

## Why the collapsing is the whole job

Infoboxes are written by thousands of people reaching for hundreds of
templates, so the property vocabulary is a folksonomy rather than a schema. A
city records what country it is in as ``subdivision_name``, or
``subdivision_name1``, or ``state``, or ``country``, depending on which
template the author used. A chain asking for ``country`` therefore fails on a
graph that plainly contains the answer.

Measured over Simple English Wikipedia, collapsing those synonyms onto one
``located_in`` relation takes ``birth_place -> country`` from **1.7%** of
subjects completing to **40.7%**, and ``death_place -> country`` from 1.0% to
48.2%. The graph was never sparse. It was inconsistently named.

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
    "located_in": (
        "country", "subdivision_name", "subdivision_name1", "state",
        "province", "region", "municipality_name", "arrondissement", "canton",
        "department", "district", "county", "prefecture", "subdivision_name2",
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

SCHEMA = """
CREATE TABLE IF NOT EXISTS edge (
    source   TEXT NOT NULL,
    subject  TEXT NOT NULL,
    relation TEXT NOT NULL,
    object   TEXT NOT NULL,
    PRIMARY KEY (source, subject, relation, object)
) {strict}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS edge_object ON edge (source, object, relation);
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

    # subject -> relation -> (rank, value): the best-ranked field wins.
    best: dict[str, dict[str, tuple[int, str]]] = {}
    for subject, prop, value in db.execute(
            "SELECT subject, property, value FROM fact WHERE source = ?",
            (source,)):
        hit = FIELD_RELATION.get(prop)
        if hit is None:
            continue
        relation, rank = hit
        slot = best.setdefault(subject, {})
        if relation not in slot or rank < slot[relation][0]:
            slot[relation] = (rank, value)

    rows, dropped = [], 0
    for subject, relations in best.items():
        for relation, (_rank, value) in relations.items():
            target = resolve(value)
            if target is None:
                dropped += 1        # a value that names no article we hold
                continue
            rows.append((source, subject, relation, target))

    db.execute("DELETE FROM edge WHERE source = ?", (source,))
    db.executemany("INSERT OR REPLACE INTO edge VALUES (?, ?, ?, ?)", rows)
    report(f"  {len(rows):,} edges, {dropped:,} values naming no article")
    return len(rows), dropped


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
    here = subject
    walked = [subject]
    for relation in relations:
        row = db.execute(
            "SELECT object FROM edge WHERE source = ? AND subject = ? "
            "AND relation = ? LIMIT 1", (source, here, relation)).fetchone()
        if row is None:
            return Answer(None, walked, relation)
        here = row[0]
        walked.append(here)
    return Answer(here, walked)


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
