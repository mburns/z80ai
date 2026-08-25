"""The silo database: what is stored, and what is a view over it.

Its own file rather than a second `source` inside `data/simple_english_wikipedia.db`,
for one mechanical reason: `ingest.connect` refuses a database holding tables
its schema does not define, and this one holds six. Sharing the file would mean
either weakening that check or keeping the residence out of the database, and
both are worse than another file.

What it does share is the corpus tables. `article`, `fact`, `category`,
`property`, `redirect`, `meta`, `edge` and `entity_type` come from
`ingest._schema()` verbatim - not copied - so `libgraph`, `liboracle`,
`oracle.py` and `buildwikisearch.py` read this corpus with no idea it is not
Wikipedia, and a change to those tables cannot leave this one behind.

## The line this file draws

**Stored:** who someone's parents are, where they sleep, what they do, which
crew and class and committee they belong to. Base facts, one row each.

**A view:** sibling, half-sibling, grandparent, aunt, cousin, ancestor,
housemate, neighbour, coworker, classmate, committee colleague. Every one of
them is a join away, and storing any of them would be storing a conclusion.

That split is the whole experiment. A derived fact that lives in a table is a
fact somebody has to keep true; a derived fact that lives in a view is one the
database works out on the way past. `questions.py` then asks the third
question - whether an eZ80 walking `libgraph`'s edges can reach the same answer
with no arithmetic beyond a comparison - and the answer is not the same for
every view here, which is the interesting part.

## The address

`FLOOR TIME RING`, after Burning Man: `42 600 A` is the forty-second floor,
due south, innermost ring. A silo floor is a circle, so a bearing is a position
on a twelve-hour clock face - stored as minutes past twelve so that arithmetic
on it is arithmetic, and rendered by a **generated column** so that no writer
can spell an address differently from another one.

Adjacency is stored as edges rather than computed from the numbers, because the
machine that has to walk it has no modulo. `next_along` is thirty minutes
clockwise, `next_out` is one ring outward, and "who lives next door" is those
two relations and their inverses - four hops, no arithmetic. The `neighbour`
view computes the same set the other way, from the bearings, which makes the
two independent implementations of one idea and lets the tests disagree.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

SOURCE = "silo"

#: Bumped when anything below changes. A database written by an older layout is
#: refused rather than read as though it were current - the same guard
#: `ingest.py` carries, for the same reason: `CREATE TABLE IF NOT EXISTS` is
#: silent about a table that already exists in the wrong shape.
SCHEMA_VERSION = 5

#: Minutes on a twelve-hour clock face, so a full circle of dwellings is 24
#: bearings at half-hour spacing. Stored as minutes because `(a - b) % 720` is
#: how adjacency is checked, and "6:00" is not a number.
CLOCK = 720
BEARING_STEP = 30
BEARINGS = CLOCK // BEARING_STEP

#: Innermost outward. Three is a choice about density, not about architecture:
#: with 24 bearings it puts 72 dwellings on a residential floor, and 10,000
#: people over seven generations fill 2,088 of them. Fewer rings would open
#: more floors for the same households and thin the neighbours out, which is
#: the one thing an address scheme here exists to avoid.
RINGS = ("A", "B", "C")


def address(floor: int, bearing: int, ring: str) -> str:
    """The Python side of the generated column, and the reason there is a test.

    Two implementations of one format is a smell everywhere except here: the
    writer needs the string before the row exists, and the database needs it to
    be the same string. `tests/test_silo.py` compares them for every occupied
    dwelling rather than trusting that they agree.
    """
    hour = 12 if bearing // 60 == 0 else bearing // 60
    return f"{floor} {hour}{bearing % 60:02d} {ring}"


TABLES = """
CREATE TABLE IF NOT EXISTS apartment (
    source  TEXT NOT NULL,
    floor   INTEGER NOT NULL,
    bearing INTEGER NOT NULL,
    ring    TEXT NOT NULL,
    address TEXT GENERATED ALWAYS AS (
        floor || ' ' ||
        (CASE WHEN bearing / 60 = 0 THEN 12 ELSE bearing / 60 END) ||
        substr('0' || (bearing % 60), -2, 2) || ' ' || ring) STORED,
    PRIMARY KEY (source, floor, bearing, ring),
    CHECK (floor > 0),
    CHECK (bearing >= 0 AND bearing < 720 AND bearing % 30 = 0),
    CHECK (ring IN ('A', 'B', 'C'))
) STRICT, WITHOUT ROWID;

CREATE UNIQUE INDEX IF NOT EXISTS apartment_address
    ON apartment (source, address);

-- `until` is NULL while somebody is still living there. It exists because a
-- dwelling outlives its occupants: 2,088 flats hold 10,000 people over 220
-- years, so "same address" without a date is a claim about the building rather
-- than about the people, and `housemate` without the overlap test below made
-- flatmates of four successive generations.
CREATE TABLE IF NOT EXISTS residence (
    source  TEXT NOT NULL,
    person  TEXT NOT NULL,
    floor   INTEGER NOT NULL,
    bearing INTEGER NOT NULL,
    ring    TEXT NOT NULL,
    since   INTEGER NOT NULL,
    until   INTEGER,
    PRIMARY KEY (source, person),
    CHECK (until IS NULL OR until >= since),
    FOREIGN KEY (source, floor, bearing, ring)
        REFERENCES apartment (source, floor, bearing, ring)
) STRICT, WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS residence_at
    ON residence (source, floor, bearing, ring);

CREATE TABLE IF NOT EXISTS cohort (
    source TEXT NOT NULL,
    name   TEXT NOT NULL,
    kind   TEXT NOT NULL,
    formed INTEGER NOT NULL,
    floor  INTEGER,
    PRIMARY KEY (source, name),
    CHECK (kind IN ('class', 'crew', 'committee'))
) STRICT, WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS cohort_kind ON cohort (source, kind);

CREATE TABLE IF NOT EXISTS membership (
    source TEXT NOT NULL,
    person TEXT NOT NULL,
    cohort TEXT NOT NULL,
    role   TEXT NOT NULL DEFAULT 'member',
    joined INTEGER NOT NULL,
    until  INTEGER,
    PRIMARY KEY (source, person, cohort),
    CHECK (until IS NULL OR until >= joined),
    FOREIGN KEY (source, cohort) REFERENCES cohort (source, name)
) STRICT, WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS membership_cohort
    ON membership (source, cohort, joined);
"""

#: Full-text over the leads, for finding a person by name the way `oracle.py`
#: does before it walks anything. Not what the card uses - that is BM25
#: precomputed at build time, because the device cannot divide - and not an
#: answering baseline either: every fact is in the prose, and getting it back
#: out of the prose is comprehension.
FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS article_fts USING fts5(
    title, lead, content='article', content_rowid='id',
    tokenize='porter unicode61');
"""

VIEWS = """
-- The facts pivoted back into a row, which is what every ad-hoc query wants
-- and what no normalized table gives you. FILTER rather than CASE because it
-- says what it means; the aggregate is over one row either way.
CREATE VIEW IF NOT EXISTS person AS
SELECT source, subject AS name,
       CAST(max(num)   FILTER (WHERE property = 'born')       AS INTEGER) AS born,
       CAST(max(num)   FILTER (WHERE property = 'died')       AS INTEGER) AS died,
       CAST(max(num)   FILTER (WHERE property = 'generation') AS INTEGER) AS generation,
       max(value) FILTER (WHERE property = 'sex')        AS sex,
       max(value) FILTER (WHERE property = 'father')     AS father,
       max(value) FILTER (WHERE property = 'mother')     AS mother,
       max(value) FILTER (WHERE property = 'department') AS department,
       max(value) FILTER (WHERE property = 'occupation') AS occupation,
       max(value) FILTER (WHERE property = 'shift')      AS shift,
       max(value) FILTER (WHERE property = 'address')    AS address,
       max(value) FILTER (WHERE property = 'fate')       AS fate
  FROM fact GROUP BY source, subject;

CREATE VIEW IF NOT EXISTS living AS
SELECT * FROM person WHERE died IS NULL;

-- Two people, one dwelling, at the same time. The tenancies have to overlap:
-- `coalesce(until, 1000000)` is the open-ended one, so anybody still living
-- there overlaps everybody who has not moved out yet.
CREATE VIEW IF NOT EXISTS housemate AS
SELECT a.source, a.person, b.person AS housemate
  FROM residence a JOIN residence b
    ON a.source = b.source AND a.floor = b.floor
   AND a.bearing = b.bearing AND a.ring = b.ring
   AND a.person <> b.person
 WHERE a.since <= coalesce(b.until, 1000000)
   AND b.since <= coalesce(a.until, 1000000);

-- Next door: thirty minutes around the ring, or one ring in or out, while both
-- were living there. The bearing wraps, which is what the `% 720` is for -
-- 12:00 and 11:30 are neighbours and a plain subtraction says they are 690
-- minutes apart.
CREATE VIEW IF NOT EXISTS neighbour AS
SELECT a.source, a.person, b.person AS neighbour,
       CASE WHEN a.ring = b.ring THEN 'along' ELSE 'across' END AS side
  FROM residence a JOIN residence b
    ON a.source = b.source AND a.floor = b.floor AND a.person <> b.person
 WHERE ((a.ring = b.ring
         AND ((a.bearing - b.bearing + 720) % 720) IN (30, 690))
     OR (a.bearing = b.bearing
         AND abs(unicode(a.ring) - unicode(b.ring)) = 1))
   AND a.since <= coalesce(b.until, 1000000)
   AND b.since <= coalesce(a.until, 1000000);

-- Anyone sharing a parent, with how many parents they share: 2 is a full
-- sibling and 1 is a half, and one view says both rather than two views
-- disagreeing about the boundary.
CREATE VIEW IF NOT EXISTS sibling AS
SELECT a.source, a.subject AS person, b.subject AS sibling,
       count(*) AS shared_parents
  FROM edge a JOIN edge b
    ON a.source = b.source AND a.object = b.object
   AND a.relation = 'child_of' AND b.relation = 'child_of'
   AND a.subject <> b.subject
 GROUP BY a.source, a.subject, b.subject;

CREATE VIEW IF NOT EXISTS grandparent AS
SELECT c.source, c.subject AS person, p.object AS grandparent,
       c.object AS via
  FROM edge c JOIN edge p
    ON c.source = p.source AND c.object = p.subject
   AND p.relation = 'child_of'
 WHERE c.relation = 'child_of';

-- A parent's sibling, and then that sibling's children. Both are written as
-- chains of `child_of` rather than in terms of the `sibling` view above, which
-- is not a stylistic choice: `sibling` is an aggregate, SQLite will not push a
-- join condition into a GROUP BY, and building `cousin` on top of it turned a
-- 0.06s query into one that had not finished after three minutes. Aggregates
-- are the end of a view chain here, never the middle.
--
-- `via` names which parent, because "on your mother's side" is half of what
-- the question usually means.
CREATE VIEW IF NOT EXISTS aunt_or_uncle AS
SELECT DISTINCT p.source, p.subject AS person, u.subject AS aunt_or_uncle,
       p.object AS via
  FROM edge p
  JOIN edge g ON g.source = p.source AND g.subject = p.object
             AND g.relation = 'child_of'
  JOIN edge u ON u.source = p.source AND u.object = g.object
             AND u.relation = 'child_of'
 WHERE p.relation = 'child_of' AND u.subject <> p.object;

CREATE VIEW IF NOT EXISTS cousin AS
SELECT DISTINCT p.source, p.subject AS person, k.subject AS cousin
  FROM edge p
  JOIN edge g ON g.source = p.source AND g.subject = p.object
             AND g.relation = 'child_of'
  JOIN edge u ON u.source = p.source AND u.object = g.object
             AND u.relation = 'child_of'
  JOIN edge k ON k.source = p.source AND k.object = u.subject
             AND k.relation = 'child_of'
 WHERE p.relation = 'child_of' AND u.subject <> p.object
   AND k.subject <> p.subject;

-- Every generation above someone, with the distance. A recursive CTE inside a
-- view: the depth is a property of the pedigree and not of the question, which
-- is exactly the shape `libgraph.CLIMB` exists for and exactly what a fixed
-- path cannot express.
CREATE VIEW IF NOT EXISTS ancestor AS
WITH RECURSIVE up(source, person, ancestor, depth) AS (
    SELECT source, subject, object, 1
      FROM edge WHERE relation = 'child_of'
    UNION ALL
    SELECT u.source, u.person, e.object, u.depth + 1
      FROM up u JOIN edge e
        ON e.source = u.source AND e.subject = u.ancestor
       AND e.relation = 'child_of'
)
SELECT * FROM up;

-- Same crew: a department and a shift is 200 people, which is a payroll rather
-- than a set of colleagues, so a crew is the ten or so who actually work
-- together and coworker means that.
CREATE VIEW IF NOT EXISTS coworker AS
SELECT a.source, a.person, b.person AS coworker, a.cohort AS crew
  FROM membership a JOIN membership b
    ON a.source = b.source AND a.cohort = b.cohort AND a.person <> b.person
  JOIN cohort g ON g.source = a.source AND g.name = a.cohort
 WHERE g.kind = 'crew';

CREATE VIEW IF NOT EXISTS classmate AS
SELECT a.source, a.person, b.person AS classmate, a.cohort AS class
  FROM membership a JOIN membership b
    ON a.source = b.source AND a.cohort = b.cohort AND a.person <> b.person
  JOIN cohort g ON g.source = a.source AND g.name = a.cohort
 WHERE g.kind = 'class';

-- A committee outlives its members' terms, so sitting on one together means
-- the terms overlapped. `coalesce(until, 1e9)` is the open-ended term: someone
-- still serving overlaps everyone who has not left yet.
CREATE VIEW IF NOT EXISTS committee_mate AS
SELECT a.source, a.person, b.person AS committee_mate, a.cohort AS committee
  FROM membership a JOIN membership b
    ON a.source = b.source AND a.cohort = b.cohort AND a.person <> b.person
  JOIN cohort g ON g.source = a.source AND g.name = a.cohort
 WHERE g.kind = 'committee'
   AND a.joined <= coalesce(b.until, 1000000)
   AND b.joined <= coalesce(a.until, 1000000);

-- Everything the database can say about how two people are connected, in one
-- place. This is the view to run against a name to see what a corpus of base
-- facts actually implies.
CREATE VIEW IF NOT EXISTS relative AS
    SELECT source, person, sibling AS other,
           CASE shared_parents WHEN 2 THEN 'sibling' ELSE 'half-sibling' END
             AS relation FROM sibling
UNION ALL SELECT source, person, grandparent, 'grandparent' FROM grandparent
UNION ALL SELECT source, person, aunt_or_uncle, 'aunt or uncle'
            FROM aunt_or_uncle
UNION ALL SELECT source, person, cousin, 'cousin' FROM cousin
UNION ALL SELECT source, person, housemate, 'housemate' FROM housemate
UNION ALL SELECT source, person, neighbour, 'neighbour' FROM neighbour
UNION ALL SELECT source, person, coworker, 'coworker' FROM coworker
UNION ALL SELECT source, person, classmate, 'classmate' FROM classmate
UNION ALL SELECT source, person, committee_mate, 'committee'
            FROM committee_mate;
"""


def load_ingest() -> ModuleType:
    """`data/wikipedia/ingest.py` owns the corpus tables; this borrows them.

    Imported by location rather than copied. Two corpora that claim to share a
    schema have to share the definition of it, and a second copy would be
    correct for exactly as long as nobody edited the first.
    """
    path = Path(__file__).resolve().parent.parent / "wikipedia" / "ingest.py"
    spec = importlib.util.spec_from_file_location("wiki_ingest", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wiki_ingest"] = module
    spec.loader.exec_module(module)
    return module


def script() -> str:
    """The whole schema: shared corpus tables, then everything silo-specific."""
    ingest = load_ingest()
    shared: str = ingest._schema()
    return shared + TABLES + FTS + VIEWS


def connect(path: Path, migrate: bool = False) -> sqlite3.Connection:
    """Open the silo database, rebuilding it if the schema has moved.

    `migrate` drops every table first, which only the generator passes, and
    only because it is about to write all of them. There is nothing to preserve
    here that a seed and eight seconds cannot reproduce - which is the one
    genuine advantage a synthetic corpus has over a downloaded one.
    """
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")

    found = db.execute("PRAGMA user_version").fetchone()[0]
    if found and found != SCHEMA_VERSION:
        if not migrate:
            raise SystemExit(
                f"{path} is schema version {found}, this is {SCHEMA_VERSION}\n"
                f"  python data/silo/generate.py --db {path}")
        _drop_everything(db)

    db.executescript(script())
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return db


def _drop_everything(db: sqlite3.Connection) -> None:
    """Read the object list out of the database rather than listing it here.

    A hand-written list has to be updated whenever the schema gains something,
    and when it is not, the stale object survives `CREATE ... IF NOT EXISTS`
    without complaint. `ingest.py` learned that the expensive way; this starts
    from the lesson.

    Foreign keys go off for the duration, because reading the object list gives
    no order and `residence` references `apartment`: dropping the parent first
    is a constraint violation rather than a demolition. They are the reason to
    have the pragma on the rest of the time, so it goes straight back.
    """
    db.execute("PRAGMA foreign_keys=OFF")
    for kind in ("view", "table"):
        for (name,) in db.execute(
                "SELECT name FROM sqlite_master WHERE type = ? "
                "AND name NOT LIKE 'sqlite_%'", (kind,)).fetchall():
            db.execute(f"DROP {kind.upper()} IF EXISTS {name}")
    db.execute("PRAGMA foreign_keys=ON")
