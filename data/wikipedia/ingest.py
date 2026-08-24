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

Two tables carry the corpus:

  article   what the machine can answer with. Title plus the opening sentences,
            because a 300-character lead is about a third of a 40x24 screen and
            a full article is not readable on one anyway.
  redirect  alternate names, pointing at a title. Wikipedia's editors have
            written ~115,000 of these, and they are the reason `jane austin`
            finds Jane Austen - the index does no fuzzy matching of its own.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "simple_english_wikipedia.db"
#: Bumped whenever the table definitions change. The database is derived data,
#: so a mismatch is resolved by re-ingesting rather than by migrating.
SCHEMA_VERSION = 4

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
      ``(subject, property) -> value`` - so that costs no separate index.
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

CREATE TABLE IF NOT EXISTS fact (
    source   TEXT NOT NULL,
    subject  TEXT NOT NULL,
    property TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (source, subject, property)
) {STRICT}WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS fact_value ON fact (source, value);
"""

# --- dump parsing -------------------------------------------------------------
#
# Line-oriented rather than an XML parse: the dump is one page per <page> block
# and only three fields are wanted, so a real parser costs minutes for nothing.

TITLE = re.compile(r"<title>(.*?)</title>")
NS = re.compile(r"<ns>(\d+)</ns>")
REDIRECT = re.compile(r'<redirect title="(.*?)"')
TEXT_OPEN = re.compile(r"<text[^>]*>")

TAG = re.compile(r"<[^>]+>")
REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S)
HEADING = re.compile(r"^=+.*?=+$", re.M)
TABLE = re.compile(r"\{\|.*?\|\}", re.S)
ENTITY = {"&quot;": '"', "&amp;": "&", "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
          "&#39;": "'", "&ndash;": "-", "&mdash;": "-"}

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


def clean(markup: str) -> str:
    """Wiki markup down to the plain sentences underneath it."""
    # Entities first. A dump escapes some markup, so `&lt;ref&gt;` survives the
    # tag pass and *becomes* a tag afterwards - which is how "<ref></ref>"
    # ended up printed on screen.
    text = markup
    for entity, char in ENTITY.items():
        text = text.replace(entity, char)
    text = REF.sub(" ", text)
    text = TABLE.sub(" ", text)
    text = _strip_braced(text)
    text = _strip_links(text)
    text = HEADING.sub(" ", text)
    text = TAG.sub(" ", text)
    text = CACHEKEY.sub(" ", text)
    text = text.replace("'''", "").replace("''", "")
    # Leading list and indent markers survive the above and read as noise.
    text = re.sub(r"^[*#:;|]+", " ", text, flags=re.M)
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

INFOBOX = re.compile(r"\{\{\s*Infobox\b", re.I)

#: Infobox keys that lay the page out rather than saying anything about the
#: subject. About a third of all fields, and none of them answers a question.
FURNITURE = re.compile(
    r"^(image|img|photo|logo|map|flag|seal|banner|cover|picture)(_|$)"
    r"|^(caption|alt|size|width|height|align|float|style|colour|color|border)(_|$)"
    r"|(_(image|img|photo|caption|alt|size|width|height|align|style|colour|color"
    r"|flag|map|link|ref|note|footnote|upright|padding))$"
    r"|^(module|embed|child|nocat|fetchwikidata|onlysourced|suppressfields"
    r"|dateformat|coordinates|latd|latm|lats|longd|longm|longs|pushpin.*)$",
    re.I)

#: Values that survived cleaning but say nothing: markup scraps, bare units,
#: template flags.
JUNK_VALUE = re.compile(r"^[\s|=*#:;{}\[\]<>/-]*$|^\d+\s*px$|^(yes|no|y|n|on|off"
                        r"|none|null|unknown|n/a|tbd|ALL)$", re.I)

#: A value longer than this is a paragraph that wandered into a field.
MAX_VALUE_LEN = 120


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


def infobox_fields(body: str) -> list[tuple[str, str]]:
    """Top-level ``| key = value`` pairs.

    Split by hand rather than by regex: a value may itself contain ``|`` inside
    a nested template or a piped link, and splitting on those turns one field
    into several fragments, none of which is a fact.
    """
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

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for field in fields[1:]:                      # [0] is the template name
        key, sep, value = field.partition("=")
        if not sep:
            continue
        key = re.sub(r"\s+", "_", key.strip().lower())
        if not key or key in seen or FURNITURE.search(key):
            continue
        # The same cleaner the lead uses, so entities are resolved before tags
        # are stripped rather than after - the bug that put "<ref></ref>" on
        # screen would otherwise put "&lt;br&gt;" in every multi-part value.
        value = clean(value)
        if not value or JUNK_VALUE.match(value) or len(value) > MAX_VALUE_LEN:
            continue
        seen.add(key)
        out.append((key, value))
    return out


def pages(path: Path):
    """Yield (title, redirect_target_or_None, lead) for every ns0 page."""
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
            if title is None:
                m = TITLE.search(line)
                if m:
                    title = m.group(1)
            if ns is None:
                m = NS.search(line)
                if m:
                    ns = int(m.group(1))
            if redirect is None:
                m = REDIRECT.search(line)
                if m:
                    redirect = m.group(1)

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
                    markup = "".join(body)
                    if redirect:
                        yield title, redirect, "", []
                    else:
                        box = infobox_body(markup)
                        yield (title, None, lead_of(markup),
                               infobox_fields(box) if box else [])
                title = None


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
               "(subject TEXT, property TEXT, value TEXT, "
               " PRIMARY KEY (subject, property))")

    articles = redirects = facts = 0
    started = time.time()
    for title, target, lead, fields in pages(dump):
        if target:
            db.execute("INSERT OR REPLACE INTO new_redirect VALUES (?, ?)",
                       (title, target))
            redirects += 1
        else:
            db.execute("INSERT OR REPLACE INTO new_article VALUES (?, ?)",
                       (title, lead))
            articles += 1
            if fields:
                db.executemany(
                    "INSERT OR REPLACE INTO new_fact VALUES (?, ?, ?)",
                    [(title, key, value) for key, value in fields])
                facts += len(fields)
        total = articles + redirects
        if total % 50_000 == 0:
            rate = total / (time.time() - started)
            print(f"  {total:,} pages, {facts:,} facts ({rate:,.0f}/s)", flush=True)

    # One transaction: an interrupted run leaves the old corpus untouched.
    with db:
        db.execute("DELETE FROM article WHERE source = ?", (source,))
        db.execute("DELETE FROM redirect WHERE source = ?", (source,))
        db.execute("DELETE FROM fact WHERE source = ?", (source,))
        db.execute("INSERT INTO article (source, title, lead) "
                   "SELECT ?, title, lead FROM new_article", (source,))
        db.execute("INSERT INTO redirect (source, title, target) "
                   "SELECT ?, title, target FROM new_redirect", (source,))
        db.execute("INSERT INTO fact (source, subject, property, value) "
                   "SELECT ?, subject, property, value FROM new_fact", (source,))
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

        f, subjects, properties = db.execute(
            "SELECT COUNT(*), COUNT(DISTINCT subject), COUNT(DISTINCT property) "
            "FROM fact WHERE source = ?", (source,)).fetchone()
        if f:
            print(f"  {' ' * len(source)}  {f:,} facts over {subjects:,} "
                  f"subjects ({subjects / a:.0%} of articles), "
                  f"{properties:,} properties")
            top = db.execute(
                "SELECT property, COUNT(*) n FROM fact WHERE source = ? "
                "GROUP BY property ORDER BY n DESC LIMIT 8", (source,)).fetchall()
            print(f"  {' ' * len(source)}  most common: "
                  + ", ".join(f"{p} ({n:,})" for p, n in top))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", nargs="?", type=Path,
                        help="pages-articles XML dump (.xml or .xml.bz2)")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--source", default="simplewiki",
                        help="Corpus name, so several can share one database")
    parser.add_argument("--stats", action="store_true",
                        help="Report what the database holds and exit")
    args = parser.parse_args()

    db = connect(args.db, migrate=bool(args.dump) and not args.stats)
    if args.stats or not args.dump:
        stats(db)
        return

    if not args.dump.exists():
        raise SystemExit(f"no such dump: {args.dump}")

    print(f"ingesting {args.dump.name} as '{args.source}' into {args.db}")
    articles, redirects, facts = ingest(db, args.dump, args.source)
    print(f"\n{articles:,} articles, {redirects:,} redirects, {facts:,} facts\n")
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
