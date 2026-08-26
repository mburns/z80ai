"""
Walking the fact graph: chaining, inverses, and the counting that comes free.

``fact`` holds what an infobox said and ``category`` what a page filed itself
under. This is the graph you can traverse: the same information with its
property names collapsed onto canonical relations and its values resolved to
article titles, so a hop can continue.

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

(Those four are the before and after of *that* change alone, kept because the
gap between them is the argument. They are not the current figures - reading
value templates, the rank fallback and the categories below have since taken
the same two chains to 77.7% and 82.3%. `coverage.py` prints today's.)

## Asking for a type, not a number of hops

A question asks for a country, not for two hops. How many hops that takes is a
property of the graph: for the 42,288 people this corpus records a birthplace
for, the birthplace is *already* a country 9.0% of the time, one hop away
57.8%, two hops 10.3%. A fixed two-hop path is right for a tenth of them and
destroys a correct answer for the eleventh that needed none.

So `in_country` climbs `located_in` until the value is something the corpus
calls a country, and stops immediately if it already is. That, the value
templates and the categories below take the share of people for whom a country
can be named at all to **77.7%**.

Those three shares are printed by `data/wikipedia/coverage.py`, against
whichever database is to hand, rather than remembered. They moved once already
without anyone noticing: the figures here read 26.2/35.7/1.6 when they were
counted by hand, and the zero-hop share is the one that drifted, because it
depends on how many entities `TYPE_FLOOR` is willing to call a country.

## What this can and cannot do

Traversal is lookups, so chaining, inverses and counting all fall out of the
same table. Inference does not: there is no rule engine here, and no way to
conclude anything that is not stored or directly derivable from what is.

## Two sources, and which one wins

An infobox is what a page's author chose to tabulate; a category is what they
chose to file it under. 46% of articles carry the first and 95.8% the second,
and for a great many pages the second is the only place a containment appears
at all - ``Infobox U.S. state`` has no country field, so Michigan is in the
United States by virtue of ``1837 establishments in the United States`` and
nothing else. ``from_categories`` reads those, for containment only and behind
three guards, and never overrides an infobox. See its docstring for why each
guard is there; every one of them is a rule the naive version broke.

## What this can and cannot do

Traversal is lookups, so chaining, inverses and counting all fall out of the
same table. Inference does not: there is no rule engine here, and no way to
conclude anything that is not stored or directly derivable from what is. A
category is at the edge of that line and stays on the near side of it: "filed
under `Cities in Denmark`" is a thing the page says, not a thing concluded from
what it says.

Twenty-two percent of birthplace climbs still reach nothing. That used to be
44%, and
the difference was not better traversal - it was noticing, by measuring rather
than assuming, that the climbs were running out of road rather than failing to
recognise a country on arrival. What remains is a property of the corpus, and
it is why anything built on this answers confidently about some things and not
at all about others.
"""

from __future__ import annotations

import re
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
#: graph rather than of the question. Measured over the 42,288 people this
#: corpus records a birthplace for, the distance from that birthplace to a
#: country is 0 hops for 9.0% of them, 1 hop for 57.8%, 2 for 10.3% and 3 for
#: 0.6%. A fixed two-hop path is right for a tenth of them and actively
#: destroys the answer for the eleventh whose birthplace was already the
#: country.
#: `coverage.py --db <db>` prints the curve for whichever corpus is to hand.
CLIMB: dict[str, tuple[str, str]] = {
    "in_country": ("located_in", "country"),
}

#: How many times a climb may step before giving up. Containment hierarchies
#: are shallow, and a cycle in the data - two places each inside the other -
#: would otherwise not terminate.
CLIMB_LIMIT = 6

#: An entity has to be named a country by this many independent infoboxes for
#: the corpus to be taken at its word. At 3 that is 193 claims, which `demote`
#: then cuts to 143 by dropping the ones the containment contradicts; at 1 it
#: is 371 claims and starts admitting things like "Washington (state)".
#: `coverage.py` prints the whole curve, which is what makes the choice
#: reviewable - the floor decides where every `in_country` climb stops.
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


#: `Cities in France`, `Historic counties of England`, `1837 establishments in
#: the United States` - a category name whose tail is where its members are.
CATEGORY_TAIL = re.compile(r"^.+?\s+(?:in|of)\s+(?P<place>.+)$", re.IGNORECASE)

#: A subject with any of these is a person, and a person is not contained by
#: the place their category names. `Presidents of France` parses exactly like
#: `Cities in France` and means something else entirely.
PERSONAL = ("birth_date", "birth_place", "death_date", "death_place")

#: The categories Wikipedia files every person under, and almost nothing else:
#: `1935 births`, `2016 deaths`, `Living people`. An infobox is optional and a
#: birth year is not, which is why this reaches people ``PERSONAL`` misses.
PERSON_CATEGORY = re.compile(r"^(?:\d{1,4}s? (?:births|deaths)|Living people)$",
                             re.IGNORECASE)

#: Relations whose subject must be a person for the question to mean anything.
#: Asking where Microsoft was born is not a gap in the graph; it is a question
#: with no answer, and counting it as a miss understates every chain that ends
#: in one. See ``coverage.py``, which reports those separately.
NEEDS_PERSON = frozenset({"born_in", "died_in", "spouse_of"})

#: `in the United States` names an article called "United States". English puts
#: the article in the category name and Wikipedia leaves it out of the title,
#: so the tail is tried both ways.
LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def from_categories(db: sqlite3.Connection, source: str, resolve: Resolver,
                    placed: dict[str, str]) -> list[tuple[str, str]]:
    """Containment a page records only by what it files itself under.

    ## Why this is here at all

    41.7% of birthplace climbs fail because they reach a place with no
    ``located_in`` edge, and for a great many of those the infobox never had
    one to give: ``Infobox U.S. state`` has no country field, so Michigan is in
    the United States only by being filed under ``1837 establishments in the
    United States``. Measured over this corpus, reading those rescues 2,869 of
    the 4,686 distinct places a birthplace climb dies on.

    ## Why it needs three guards rather than none

    ``<something> in|of <Place>`` is a promising rule and a credulous one. Each
    of these was added because the rule without it produced nonsense, and the
    counts are what the naive version yielded:

        the target must be a place    "established in 2022" and "songs in
                                      Italian" both parse and both resolve,
                                      because this encyclopedia has an article
                                      on 2022 and on Italian
        the subject must be a place   otherwise every band formed in California
                                      is filed inside it - 70,844 edges, of
                                      which 3,763 were about places
        the subject must not be a     `Presidents of France` is not a
        person                        containment, and reads as one

    "A place" is not a judgement here: it is anything the corpus already treats
    as one, which is to say anything on an existing ``located_in``,
    ``born_in``, ``died_in`` or ``capital_is`` edge. That keeps this from
    inventing a category of thing.

    ## What it will not do

    Override an infobox. A page that said where it was only ever says it once,
    and a category is the weaker evidence, so this fills gaps and never
    replaces. It is therefore monotonic: a chain that completed before still
    completes, by the same route.

    ## And what `People from X` is not

    Containment only. ``People from X`` looks like the obvious next win - 16,348
    filings, and 6,029 of them would be a ``born_in`` edge for someone who has
    none, which is a 14% rise on a relation the whole oracle leans on. It was
    measured and rejected.

    The corpus grades it for free: 8,972 of those people already have a
    birthplace from an infobox, so the two can be compared directly.

        names the same place                4,282   47.7%
        names one the graph can relate       842    9.4%
        names something else               3,848   42.9%

    Some of that last group is a coarser containment the graph has not learned
    yet - Doris Leuthard's infobox says Merenschwand and her category says
    Aargau, and Merenschwand is in Aargau. Others are simply different places:
    three of the first five are people the category files under *Aberdeen,
    South Dakota* whose infobox says *Aberdeen*.

    Either way the error rate is tens of percent against a source that is
    essentially exact, and it would land on the one question this machine
    answers confidently. `from` is not `born in`: it takes in where someone grew
    up, worked, or is associated with, and Wikipedia's editors use it that
    loosely. 6,029 more answers are not worth making the existing 42,288 less
    trustworthy.
    """
    try:
        filings = db.execute(
            "SELECT title, name FROM category WHERE source = ?",
            (source,)).fetchall()
    except sqlite3.OperationalError:
        return []              # a database written before schema 7 has none

    people = {s for (s,) in db.execute(
        "SELECT DISTINCT subject FROM fact WHERE source = ? AND property IN "
        f"({','.join('?' * len(PERSONAL))})", (source, *PERSONAL))}
    places = set(placed) | set(placed.values()) | {o for (o,) in db.execute(
        "SELECT DISTINCT object FROM edge WHERE source = ? AND relation IN "
        "('located_in', 'born_in', 'died_in', 'capital_is')", (source,))}

    # Best candidate per subject rather than first-seen: a place filed under
    # both `Cities in Denmark` and a continent should take Denmark, because a
    # climb can carry on from a country and stops dead at Asia. Rows arrive in
    # primary-key order, which is alphabetical and means nothing.
    best: dict[str, tuple[int, str]] = {}
    for title, name in filings:
        if title in people or title in placed or title not in places:
            continue
        m = CATEGORY_TAIL.match(name)
        if m is None:
            continue
        tail = m["place"]
        target = resolve(tail) or resolve(LEADING_ARTICLE.sub("", tail, count=1))
        if target is None or target == title or target not in places:
            continue
        rank = 0 if target in placed or is_a(db, source, target, "country") else 1
        if title not in best or (rank, target) < best[title]:
            best[title] = (rank, target)
    return [(title, target) for title, (_rank, target) in best.items()]


#: An ISO 3166-2 subdivision code names its country in the half before the
#: dash, by definition of the standard. Hampshire records `iso_code = GB-HAM`
#: and records no container at all, and it is not alone: 2,648 climbs - 6.3% of
#: every person with a birthplace - stop at a place carrying one of these.
SUBDIVISION_CODE = re.compile(r"^([A-Z]{2})-[A-Z0-9]{1,3}$")

#: Where a country states its own code. `iso3166code` is the standard itself;
#: `cctld` is the same two letters for all but a handful, and covers four times
#: as many countries here.
COUNTRY_CODE_FIELDS = ("iso3166code", "cctld")

#: The two disagree for a few countries and the United Kingdom is the one that
#: matters: ISO says GB, the internet says .uk.
CCTLD_ISO = {"uk": "GB"}


def _claimed_countries(db: sqlite3.Connection, source: str,
                       resolve: Resolver) -> set[str]:
    """Everything enough infoboxes call a country, before any refinement."""
    counts: dict[str, int] = {}
    for (value,) in db.execute(
            "SELECT value FROM fact WHERE source = ? AND property = ?",
            (source, TYPE_FIELD["country"])):
        target = resolve(value)
        if target:
            counts[target] = counts.get(target, 0) + 1
    return {e for e, n in counts.items() if n >= TYPE_FLOOR}


def country_codes(db: sqlite3.Connection, source: str,
                  countries: set[str]) -> dict[str, str]:
    """ISO code -> the corpus's name for that country.

    Read off the countries' own articles rather than written down here. A
    hardcoded table would be 249 lines that go stale and might not match any
    title in this corpus; this is in the corpus's own spelling by construction,
    and it names 91 of them.
    """
    out: dict[str, str] = {}
    for prop in COUNTRY_CODE_FIELDS:
        for subject, value in db.execute(
                "SELECT subject, value FROM fact WHERE source = ? "
                "AND property = ?", (source, prop)):
            if subject not in countries:
                continue
            # `cctld = ch, .swiss` and `cctld = it d` both occur: take the
            # first token however it was separated, and drop a leading dot.
            first = re.split(r"[,\s]+", value.strip())[0].lstrip(".")
            if not re.fullmatch(r"[A-Za-z]{2}", first):
                continue
            out.setdefault(CCTLD_ISO.get(first.lower(), first.upper()), subject)
    return out


def build(db: sqlite3.Connection, source: str,
          report: Callable[[str], None] = lambda _m: None,
          derived: str | None = None) -> tuple[int, int]:
    """Populate ``edge`` for one source from its facts. Returns (edges, dropped).

    ``derived`` names a method in the `derived` table - "regex" - and admits
    its rows as edges. **Off by default and deliberately so.** Everything else
    here comes from something a Wikipedia author tabulated or filed; those come
    from `birthplaces.py` reading a sentence, and a card built with them
    asserts things no infobox states. The flag is what makes that a decision
    somebody takes rather than one that happens.

    They fill gaps only. `birthplaces.py` asks about people who have no
    birthplace, so a derived row cannot contradict an infobox - but it is
    written last and skips any subject already carrying the relation, so that
    holds even if the table is stale.
    """
    titles = {t for (t,) in db.execute(
        "SELECT title FROM article WHERE source = ?", (source,))}
    redirects = dict(db.execute(
        "SELECT title, target FROM redirect WHERE source = ?", (source,)))
    resolve = Resolver(titles, redirects)

    # subject -> relation -> [(rank, ordinal, value), ...]: every field that
    # means this relation, best-ranked first, and within one field the lowest
    # ordinal first. A template writes a list as `subdivision_name1`,
    # `subdivision_name2`; the ingest splits that index into `ordinal`, so the
    # tie-break that used to be spelled out in CANONICAL as two separate fields
    # is now just a smaller number.
    #
    # All of them rather than only the best, because the best-ranked field is
    # not always one that names an article. A song carrying `artist = Ronnie
    # Milsap` and `writer = Phil Barnhart, Sam Hogin, James House` has a
    # `created_by` answer and a higher-ranked field that is a list of three
    # people, and taking the rank alone throws the answer away for something
    # that resolves to nothing. Ranking still decides; it just no longer
    # decides on behalf of a value that cannot be used.
    candidates: dict[str, dict[str, list[tuple[int, int, str]]]] = {}
    for subject, prop, ordinal, value in db.execute(
            "SELECT subject, property, ordinal, value FROM fact "
            "WHERE source = ?", (source,)):
        hit = FIELD_RELATION.get(prop)
        if hit is None:
            continue
        relation, rank = hit
        candidates.setdefault(subject, {}).setdefault(relation, []).append(
            (rank, ordinal, value))

    rows, dropped = [], 0
    for subject, relations in candidates.items():
        for relation, options in relations.items():
            for _rank, _ordinal, value in sorted(options):
                target = resolve(value)
                # A thing is not inside itself. `Los Angeles located_in Los
                # Angeles` stopped 486 climbs where they stood, because the
                # type test fires before the step - so "what country was X
                # born in" answered Los Angeles. Falling through to the next
                # candidate keeps whatever the page said second.
                if target is not None and target != subject:
                    rows.append((source, subject, relation, target))
                    break
            else:
                dropped += 1        # no field for it named an article we hold

    # A place that named no container it holds, but that states an ISO 3166-2
    # subdivision code, is in the country the half before the dash names. That
    # is what the standard means, and 2,648 climbs - 6.3% of everyone with a
    # birthplace - stop at such a place: Washington (state), Moscow, Maryland
    # and Baden-Wurttemberg record `US-WA`, `RU-MOW`, `US-MD`, `DE-BW` and
    # nothing else at all.
    #
    # A fallback, after every real field has been tried: a container the page
    # states is better evidence than one the standard implies.
    #
    # The country set here is the raw one, counted from `country` fields alone.
    # `types()` refines it by asking which claimed countries the *edges* say sit
    # inside another, so it cannot run until the edges exist - and this has to
    # run before them. The refinement is not needed for this: a place demoted
    # for being inside a country does not also carry a ccTLD.
    have_edge = {subject for _s, subject, relation, _o in rows
                 if relation == "located_in"}
    codes = country_codes(db, source, _claimed_countries(db, source, resolve))
    from_code = 0
    for subject, value in db.execute(
            "SELECT subject, value FROM fact WHERE source = ? "
            "AND property LIKE 'iso%'", (source,)):
        if subject in have_edge or subject not in titles:
            continue
        m = SUBDIVISION_CODE.match(value.strip())
        named = codes.get(m.group(1)) if m else None
        if named and named != subject:
            rows.append((source, subject, "located_in", named))
            have_edge.add(subject)
            from_code += 1
    if from_code:
        report(f"  {from_code:,} places placed by their ISO code")

    db.execute("DELETE FROM edge WHERE source = ?", (source,))
    db.executemany("INSERT OR REPLACE INTO edge VALUES (?, ?, ?, ?)", rows)
    report(f"  {len(rows):,} edges, {dropped:,} values naming no article")

    # Then the containment nobody tabulated. Written after the infobox edges
    # and read against them, so a page that said where it was keeps what it
    # said and only the silent ones are filled in.
    # Types before categories, because `from_categories` asks whether a target
    # is a country in order to prefer Denmark over Europe, and `entity_type` is
    # rebuilt here. Written the other way round it read the *previous* build's
    # answer - which on a fresh database is no rows at all, so every target
    # ranked the same and the preference fell back to alphabetical order.
    #
    # Worth 365 edges and 821 more completed chains, and every one of them an
    # improvement of the same kind: Aberdeen was in the North Sea, Akureyri in
    # Europe, Al-Ahsa Oasis in the Middle East. They are in Scotland, Iceland
    # and Saudi Arabia.
    #
    # The order is easy to get wrong and hard to catch, because it only shows
    # up on a *fresh* ingest. Re-running `build` against a database that
    # already has an `entity_type` reads the last run's rows and comes out
    # right, so measuring a rebuild says the ordering does not matter. It does.
    #
    # Types come from `fact` rather than `edge`, so this does not care that the
    # category edges are not in yet.
    kinds = types(db, source, resolve)
    db.execute("DELETE FROM entity_type WHERE source = ?", (source,))
    db.executemany("INSERT OR REPLACE INTO entity_type VALUES (?, ?, ?)",
                   [(source, kind, entity) for kind, entity in kinds])
    report(f"  {len(kinds):,} typed entities")

    placed = dict(db.execute(
        "SELECT subject, object FROM edge WHERE source = ? "
        "AND relation = 'located_in'", (source,)))
    filed = from_categories(db, source, resolve, placed)
    if filed:
        db.executemany(
            "INSERT OR REPLACE INTO edge VALUES (?, ?, 'located_in', ?)",
            [(source, subject, target) for subject, target in filed])
        report(f"  {len(filed):,} more from categories")

        # And now the types again, because `demote` needs to know what
        # contains what and half of that did not exist when it last ran.
        # California, Chicago and Los Angeles are placed inside the United
        # States by their categories, not by their infoboxes, so the pass above
        # saw them contained by nothing and left them believing they were
        # countries. The documented fix for this landed in #40 and never
        # covered them - "what country was X born in" still answered Chicago
        # 445 times and California 555.
        #
        # Only the demotions can change here, not the claims: a category never
        # says `country = X`. So this re-reads the containment and nothing else.
        kinds = types(db, source, resolve)
        db.execute("DELETE FROM entity_type WHERE source = ?", (source,))
        db.executemany("INSERT OR REPLACE INTO entity_type VALUES (?, ?, ?)",
                       [(source, kind, entity) for kind, entity in kinds])
        report(f"  {len(kinds):,} typed entities after containment")

    # Last, and only when asked. Written after everything an author actually
    # wrote down, and skipping any subject that already has the relation, so a
    # sentence can fill a gap and can never overrule a table.
    read = 0
    if derived:
        read = _from_derived(db, source, derived)
        report(f"  {read:,} more read out of leads, method {derived!r}")

    # All of them, because the caller writes this into `meta` as what the
    # database holds. Returning only the infobox edges recorded 163,807 against
    # a table of 167,868 - a provenance number that disagrees with the thing it
    # is provenance for, which is worse than not having one.
    return len(rows) + len(filed) + read, dropped


def _from_derived(db: sqlite3.Connection, source: str, method: str) -> int:
    """Admit one method's rows from `derived`, for subjects with no such edge."""
    try:
        rows = db.execute(
            "SELECT subject, relation, object FROM derived "
            "WHERE source = ? AND method = ?", (source, method)).fetchall()
    except sqlite3.OperationalError:
        return 0               # a database written before schema 8 has none

    held = {(s, r) for s, r in db.execute(
        "SELECT subject, relation FROM edge WHERE source = ?", (source,))}
    fresh = [(source, s, r, o) for s, r, o in rows if (s, r) not in held]
    db.executemany("INSERT OR REPLACE INTO edge VALUES (?, ?, ?, ?)", fresh)
    return len(fresh)


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
    within = dict(db.execute(
        "SELECT subject, object FROM edge WHERE source = ? AND relation = ?",
        (source, "located_in")))
    demoted = demote({e: n for (kind, e), n in counts.items()
                      if kind == "country" and n >= TYPE_FLOOR}, within)
    return sorted(k for k, n in counts.items()
                  if n >= TYPE_FLOOR and k[1] not in demoted)


def demote(claims: dict[str, int], within: dict[str, str]) -> set[str]:
    """Of two claimed countries in a containment, the one to stop believing.

    Three infoboxes calling a thing a country is a low bar for a corpus this
    size - California, Chicago and Los Angeles all clear it - and a climb that
    reaches one stops and answers it. So a claim is dropped when the same graph
    contradicts it, the contradiction being that it sits *inside another
    country*. Being inside anything at all is not enough: France records that
    it is in Europe and is still a country.

    **Which of the two to drop is decided by how often the corpus calls each a
    country**, not by which is the container. Dropping the contained one always
    is the obvious rule and it is wrong here, because three infoboxes say
    `country = Asia`: that is exactly `TYPE_FLOOR`, so Asia is claimed, and
    every country filed inside it - Japan, China, Iran, forty more - becomes a
    thing "inside another country" and gets demoted. Antarctica and the
    Caribbean clear the floor the same way, and Europe does not, which is why
    the France example above worked and hid this.

    The counts are not close, so this is not a delicate judgement:

        Asia 3            against Japan 257, China 74, Iran 35
        United States 4155 against California 16, Chicago 5, Massachusetts 4
        Canada 283         against Ontario 3

    A tie keeps the container, which is the old behaviour.
    """
    dropped = set()
    for entity, container in within.items():
        if entity not in claims or container not in claims:
            continue
        if claims[container] < claims[entity]:
            dropped.add(container)
        else:
            dropped.add(entity)
    return dropped


def people(db: sqlite3.Connection, source: str) -> set[str]:
    """Everything the corpus says is a person.

    Two sources, because neither alone is enough. An infobox with a birth date
    is decisive but only 45.7% of articles carry any infobox; the birth-year
    categories are near-universal for people and cost nothing, the ingest
    already having read them.

    This exists to tell "we have no birthplace for this person" apart from
    "this is a band". 692 of the 1,558 creators blocking `created_by born_in`
    are companies and groups - Microsoft, ABBA, Capcom - and they account for
    3,318 of the 5,462 works the chain cannot finish.
    """
    found = {s for (s,) in db.execute(
        "SELECT DISTINCT subject FROM fact WHERE source = ? AND property IN "
        f"({','.join('?' * len(PERSONAL))})", (source, *PERSONAL))}
    try:
        rows = db.execute("SELECT title, name FROM category WHERE source = ?",
                          (source,))
    except sqlite3.OperationalError:
        return found           # a database written before schema 7 has none
    found.update(title for title, name in rows if PERSON_CATEGORY.match(name))
    return found


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

    @property
    def at(self) -> str | None:
        """The entity the walk had reached when it stopped.

        Which is what says whether the missing edge is a gap or a category
        error: `missing` alone cannot tell "no birthplace recorded for this
        author" from "asked a software company where it was born".
        """
        return self.path[-1] if self.path else None


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
