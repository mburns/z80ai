"""
Ten things to find, and every one of them is a name somebody wrote down.

`libworld.Thing.subject` made the card reachable from inside the world: a thing
that names an entry can be consulted, and the entries a player can reach are
the ones they have physically found a reference to. That was the mechanism.
This is the content, and it is the one place in `data/silo/` that writes prose
about the generated corpus rather than quoting it.

## Why authored rather than generated

The corpus has no objects. It has ten thousand people, their parents, their
flats and their shifts, and nothing you can pick up - so a seed cannot be
derived, only *placed*. What it can be is derived **about**: every item below
is a hand-written sentence with a hole in it, and the corpus fills the hole.

The constraint that shapes the list is smaller than it looks. A thing's name is
one word and every name in a world must be unique, so there is no such thing as
seventy-two ledgers, one per flat. Ten distinct objects is not a small version
of a big idea; it is the only shape this parser has.

## They are one case, not ten props

Nine of the ten hang off a single cleaning, chosen as the alphabetically first
in the corpus so that the same database always seeds the same silo:

    Judicial          a cleaning notice           -> who was sent out
    Sheriff's Office  a key, tagged with a flat   -> where they lived
    the flat itself   a photograph                -> who they married
    Nursery           a school slate              -> what class they were in
    Judicial          committee minutes           -> who reviewed the cleaning

Each one names the next place to stand. The photograph is only there if that
floor was opened with `--floors`, and `buildworld` says so rather than dropping
it, because a chain with a link missing is worse than a chain that says which.

The other three are texture and one of them is the control: a wrench names
nothing, and `CONSULT WRENCH` has to be able to say so.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import schema

#: Where an item is found. A department name puts it in that department's room;
#: `FLAT` puts it in the dwelling the case lived in, which exists only if that
#: floor was opened.
FLAT = "\x00flat"


@dataclass(frozen=True)
class Case:
    """One cleaning, and everything the corpus knows about who it was.

    Alphabetically first rather than random: the same database has to seed the
    same silo every time, or two builds of one card disagree about what is in
    the drawer.
    """

    who: str
    year: str
    flat: str
    schooling: str
    spouse: str


@dataclass(frozen=True)
class Item:
    name: str
    #: `{who}`, `{year}`, `{flat}`, `{schooling}` and `{spouse}` are filled
    #: from the case. A description with no hole in it needs no case.
    description: str
    #: A department name, or `FLAT`.
    where: str
    #: The archive entry this is a reference to, as a format string over the
    #: same fields, or None for a thing that names nothing.
    subject: str | None


#: The list. Names are one word and at most `libworld.MAX_WORD_LEN`, because
#: `SPLIT` takes two words and truncates either at twelve - see `IF.md`.
ITEMS: tuple[Item, ...] = (
    Item("badge",
         "A deputy's badge on a bent pin, tarnished to the colour of the "
         "rail. The number on the back has been filed off.",
         "Sheriff's Office", "Sheriff's Office"),
    Item("notice",
         "A carbon of a cleaning notice, year {year}. Sent out: {who}. There "
         "is no reason given on it, which is the form.",
         "Judicial", "{who}"),
    Item("key",
         "A flat key on a wire loop. The paper tag is soft with handling and "
         "reads {flat}.",
         "Sheriff's Office", "{flat}"),
    Item("minute",
         "Minutes of the Cleaning Oversight Committee, unbound and out of "
         "order. Somebody has folded down the corner of one page.",
         "Judicial", "Cleaning Oversight Committee"),
    Item("slate",
         "A school slate with a class list still chalked on it, half rubbed "
         "out. The heading is legible: {schooling}.",
         "Nursery", "{schooling}"),
    Item("photo",
         "A photograph, curled at one corner. Two people at a rail, and a "
         "name on the back in pencil: {spouse}.",
         FLAT, "{spouse}"),
    Item("ledger",
         "A ration ledger from year 188, Supply's own copy. The back of it "
         "has been drawn on by somebody small.",
         "Supply", "Supply"),
    Item("order",
         "A work order for the number three generator, signed off and "
         "countersigned by nobody.",
         "Mechanical", None),
    Item("wrench",
         "A generator technician's wrench, worn to the shape of a hand that "
         "is not yours.",
         "Mechanical", None),
    Item("chit",
         "A ration chit for one meal, undated. Everybody has a few of these "
         "and nobody counts them until they do.",
         "Cafeteria", None),
)

#: The cleaning this seed is built around. `ORDER BY subject` rather than a
#: sample, so the seed is a function of the database and nothing else.
_CLEANED = """
SELECT subject FROM fact
WHERE source = ? AND property = 'fate' AND value = 'Cleaning'
ORDER BY subject LIMIT 1
"""

_FACT = """
SELECT value FROM fact WHERE source = ? AND subject = ? AND property = ?
ORDER BY ordinal LIMIT 1
"""


def case(db: sqlite3.Connection) -> Case | None:
    """The cleaning the seed hangs off, or None in a corpus with none.

    A corpus can have none: `generate.py` sends 1.5% of deaths out to clean,
    and a small enough `--people` rounds that to zero. The caller drops the
    nine items that need one rather than placing a notice about nobody.
    """
    found = db.execute(_CLEANED, (schema.SOURCE,)).fetchone()
    if found is None:
        return None
    who = found[0]

    def fact(name: str) -> str | None:
        row = db.execute(_FACT, (schema.SOURCE, who, name)).fetchone()
        return None if row is None else row[0]

    year, flat = fact("died"), fact("address")
    schooling, spouse = fact("class"), fact("spouse")
    if not (year and flat and schooling and spouse):
        return None                  # a case with a hole in it is not a case
    return Case(who, year, flat, schooling, spouse)


def seed(db: sqlite3.Connection) -> tuple[list[tuple[Item, str, str | None]],
                                          list[str]]:
    """`[(item, description, subject)]`, and the names of what was dropped.

    Both halves are returned because dropping quietly is the failure this is
    most likely to have: a corpus with no cleanings would seed three objects
    out of ten and look like a world that was always meant to hold three.
    """
    found = case(db)
    fields = {} if found is None else {
        "who": found.who, "year": found.year, "flat": found.flat,
        "schooling": found.schooling, "spouse": found.spouse}

    placed: list[tuple[Item, str, str | None]] = []
    dropped: list[str] = []
    for item in ITEMS:
        try:
            description = item.description.format(**fields)
            subject = None if item.subject is None \
                else item.subject.format(**fields)
        except KeyError:
            dropped.append(item.name)
            continue
        placed.append((item, description, subject))
    return placed, dropped
