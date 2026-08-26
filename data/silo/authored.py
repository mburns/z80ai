#!/usr/bin/env python3
"""
Written entries, into the same corpus tables the generator writes.

    python data/silo/authored.py                 # into data/silo.db
    python data/silo/authored.py --report        # what they pack to, and why

`generate.py` invents ten thousand people and a lead apiece, and every one of
those leads is the same sentence with different nouns in it. That is what makes
the corpus useful for measuring a graph walk and useless as something to read.

This puts human-written documents on the card beside them - incident reports,
committee minutes, maintenance logs - as `article` rows with the same `source`,
so `buildwikisearch.py` needs no argument and does not know the difference. A
question that lands on one gets prose somebody wrote; a question that lands on
a person gets the generator's sentence.

## They are prose and nothing else

No `edge`, no `fact`, no `entity_type`. A written entry is findable and
readable, and the graph cannot walk to it or from it. That is the honest shape
for the thing: the oracle answers *about people* from the graph, and these are
the archive it also happens to be sitting on.

## Re-run this after any re-generate

`generate.write` opens with `DELETE FROM article WHERE source = ?`, so
rebuilding the world removes these too. They are files on disk and this is
idempotent, so the repair is to run it again - the same shape as
`data/wikipedia/birthplaces.py` step 2a.

## What fits

An entry is capped at what the *device* can read, which is not a matter of
taste. `READ_ARTICLE` reads one `CHUNK` and `UNPACK` walks it until it has seen
the NULs ending the title and the lead, so an article packing to more than
`libsearch.MAX_PACKED_ARTICLE` does not truncate - the decoder carries on into
whatever the last query left in SRAM.

Byte-pair packing only ever replaces a pair with one byte, so packed text is
never longer than what went in. Capping the *raw* entry at that same number is
therefore a cap that cannot be exceeded however badly the prose compresses,
which is why this does not have to guess at a compression ratio the way a
character count would.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema
from schema import SOURCE

import libsearch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ENTRIES = HERE / "authored"
DB_PATH = REPO / "data" / "silo.db"

#: Room for the title, the two NUL terminators, and a little slack. The device
#: reads title and lead together, so a long title is a shorter entry.
TITLE_ROOM = 96

#: The most body text one entry may hold. Not a preference - see the module
#: docstring. `libsearch.write_text` refuses anything past the device's read,
#: and this is that number with the title's share taken out.
MAX_BODY = libsearch.MAX_PACKED_ARTICLE - TITLE_ROOM


def read_entry(path: Path) -> tuple[str, str]:
    """Title from the first line, body from the rest.

    Wrapped lines are joined and blank lines become one newline, because the
    device has no word wrap: it prints bytes through `RST 10h` and the terminal
    breaks them wherever the column runs out. A paragraph break is the only
    formatting that survives, so it is the only one this keeps.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].strip():
        raise SystemExit(f"{path} has no title on its first line")
    title = lines[0].strip()

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines[1:]:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return title, "\n".join(paragraphs)


def load(directory: Path) -> list[tuple[str, str]]:
    entries = []
    for path in sorted(directory.glob("*.txt")):
        title, body = read_entry(path)
        if len(body.encode()) > MAX_BODY:
            raise SystemExit(
                f"{path.name} is {len(body.encode())} bytes and the device "
                f"can hold {MAX_BODY}. Split it, or shorten it")
        entries.append((title, body))
    if not entries:
        raise SystemExit(f"no .txt entries in {directory}")
    return entries


#: Where the last run's titles are recorded, so that this one can take them
#: away again. The `article` table has no column saying which rows are written
#: and which are generated, and adding one would put a fact about this script
#: into a schema shared with Wikipedia.
TITLES_KEY = "silo.authored"


def write(db: sqlite3.Connection, entries: list[tuple[str, str]]) -> tuple[int, int]:
    """Replace the written entries, leaving the generated ones alone.

    Deleting *last* run's titles rather than this one's is the difference
    between renaming an entry and haunting the card with it. A rename would
    otherwise leave the old article indexed, findable, and attached to no file
    anybody could edit - and since it would still answer, nothing would report
    it. Returns (written, removed).
    """
    previous = db.execute("SELECT value FROM meta WHERE key = ?",
                          (TITLES_KEY,)).fetchone()
    stale = [t for t in (previous[0].split("\n") if previous else []) if t]
    titles = [title for title, _ in entries]

    # Only ever delete what a previous run of this script wrote. Deleting the
    # titles about to be written would be simpler and would quietly swallow a
    # generated article whose name an entry happened to share - and withdrawing
    # that entry later would then take the generated one with it, until the
    # next re-generate put it back.
    if stale:
        marks = ",".join("?" * len(stale))
        db.execute(f"DELETE FROM article WHERE source = ? AND title IN ({marks})",
                   (SOURCE, *stale))

    marks = ",".join("?" * len(titles))
    clash = [t for (t,) in db.execute(
        f"SELECT title FROM article WHERE source = ? AND title IN ({marks})",
        (SOURCE, *titles))]
    if clash:
        raise SystemExit(
            f"{clash[0]!r} is already an article the generator writes. A "
            f"written entry cannot take a generated article's name: rename "
            f"the entry, or the archive loses the thing it is named after")

    db.executemany("INSERT INTO article (source, title, lead) VALUES (?, ?, ?)",
                   [(SOURCE, title, body) for title, body in entries])
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
               (TITLES_KEY, "\n".join(titles)))
    db.execute("INSERT INTO article_fts(article_fts) VALUES('rebuild')")
    db.commit()
    return len(entries), len(set(stale) - set(titles))


def report(entries: list[tuple[str, str]], db: sqlite3.Connection) -> None:
    """What written prose packs to, against what the generator's leads do.

    #51 measured byte-pair packing at 30.6% on Wikipedia leads and the codes
    are learned per corpus, so neither that figure nor the silo's says what
    these will do. Written prose is the case the packer has never been shown:
    it repeats its own vocabulary the way any one author does, and it does not
    repeat the generator's one sentence at all.
    """
    def saving(texts: list[str]) -> tuple[int, int]:
        """Learned over the corpus, packed one entry at a time - which is what
        `write_text` does, and the difference is not small: packing the whole
        blob at once lets a pair straddle two articles and reports a saving no
        seek could ever collect."""
        each = [t.encode() for t in texts]
        raw = b"\x00".join(each)
        merges = libsearch.learn_pairs(raw[:libsearch.LEARN_SAMPLE],
                                       libsearch.free_codes(raw))
        return len(raw), sum(len(libsearch.pack_text(one, merges))
                             for one in each)

    written = [f"{t}\n{b}" for t, b in entries]
    generated = [lead for (lead,) in db.execute(
        "SELECT lead FROM article WHERE source = ? AND title NOT IN "
        f"({','.join('?' * len(entries))}) LIMIT 4000",
        (SOURCE, *[t for t, _ in entries]))]

    print(f"\n{'':<12}{'entries':>9}{'raw':>10}{'packed':>10}{'saving':>9}"
          f"{'bytes each':>12}")
    for name, texts in (("written", written), ("generated", generated)):
        if not texts:
            continue
        raw, packed = saving(texts)
        print(f"{name:<12}{len(texts):>9,}{raw:>10,}{packed:>10,}"
              f"{1 - packed / raw:>8.1%}{raw // len(texts):>12,}")

    longest = max(entries, key=lambda e: len(e[1].encode()))
    print(f"\nlongest entry {len(longest[1].encode()):,} bytes of a possible "
          f"{MAX_BODY:,} ({longest[0]})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--dir", type=Path, default=ENTRIES,
                    help="directory of .txt entries, title on the first line")
    ap.add_argument("--report", action="store_true",
                    help="also report what they pack to")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}; run generate.py first")

    entries = load(args.dir)
    db = schema.connect(args.db)
    count, removed = write(db, entries)
    total = sum(len(b.encode()) for _, b in entries)
    gone = f", {removed} withdrawn" if removed else ""
    print(f"{args.db}: {count} written entries, {total:,} bytes of prose{gone}")
    if args.report:
        report(entries, db)
    db.close()


if __name__ == "__main__":
    main()
