#!/usr/bin/env python3
"""
Build the Agon search card from data/simple_english_wikipedia.db.

    python buildwikisearch.py --out dist/WIKI
    python buildwikisearch.py --out dist/WIKI --limit 20000   # a small card

Reads whatever the database currently holds and writes three files. Refreshing
the corpus is `data/wikipedia/ingest.py` followed by this - nothing here knows
or cares which snapshot it came from, and the binary reads the card by name, so
a rebuilt card needs no rebuilt binary unless the *format* changed.

    WIKI.IDX   hashed dictionary and postings, streamed from the card
    WIKI.DAT   titles and leads behind an offset table
    WIKI.bin   the eZ80 program

The one thing that must agree across all three is the format constants in
libsearch, which the binary bakes in and the files are written with. A card
built by an older version is detected by its magic and refused rather than
misread.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import libsearch

DB_PATH = Path(__file__).resolve().parent / "data" / "simple_english_wikipedia.db"


def load_corpus(db_path: Path, source: str,
                limit: int | None) -> tuple[list[str], list[str], dict[int, list[str]]]:
    """Titles, leads and per-document aliases, straight from the database."""
    if not db_path.exists():
        raise SystemExit(
            f"no database at {db_path}\n"
            f"build one first:\n"
            f"  python data/wikipedia/ingest.py <pages-articles.xml.bz2>")

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # A limited card takes the *notable* articles, ranked by how many redirects
    # point at them. A thing with six alternate names is a thing people look
    # for; a thing with none is usually a village or a fixture list. Ranking by
    # lead length instead - the obvious choice - builds a card full of long
    # stubs that happens to exclude Photosynthesis and Jane Austen, which is
    # how the first attempt at this went.
    if limit:
        rows = db.execute(
            "SELECT a.id, a.title, a.lead, "
            "       (SELECT COUNT(*) FROM redirect r "
            "         WHERE r.source = a.source AND r.target = a.title) AS fame "
            "FROM article a WHERE a.source = ? "
            "ORDER BY fame DESC, LENGTH(a.lead) DESC LIMIT ?",
            (source, limit)).fetchall()
    else:
        rows = db.execute(
            "SELECT id, title, lead FROM article WHERE source = ? ORDER BY id",
            (source,)).fetchall()
    if not rows:
        raise SystemExit(f"no articles for source {source!r} in {db_path}")

    titles = [r[1] for r in rows]
    leads = [r[2] for r in rows]
    slot = {r[1]: i for i, r in enumerate(rows)}

    aliases: dict[int, list[str]] = {}
    kept = dropped = 0
    for title, target in db.execute(
            "SELECT title, target FROM redirect WHERE source = ?", (source,)):
        doc = slot.get(target)
        if doc is None:
            dropped += 1                    # target outside this build
            continue
        aliases.setdefault(doc, []).append(title)
        kept += 1

    print(f"{len(titles):,} articles, {kept:,} redirects "
          f"({dropped:,} pointing outside this build)")
    db.close()
    return titles, leads, aliases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--source", default="simplewiki")
    parser.add_argument("--out", default="dist/WIKI",
                        help="Path stem; .IDX, .DAT and .bin are written")
    parser.add_argument("--limit", type=int, default=None,
                        help="Index only this many articles, longest leads "
                             "first. For a test card, not a shipped one")
    parser.add_argument("--no-binary", action="store_true",
                        help="Write the card files only")
    args = parser.parse_args()

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)

    titles, leads, aliases = load_corpus(args.db, args.source, args.limit)

    print("\nindexing...")
    index = libsearch.build(titles, leads, aliases, report=print)

    idx_path = stem.with_suffix(".IDX")
    dat_path = stem.with_suffix(".DAT")
    idx = libsearch.write_index(index, idx_path)
    dat = libsearch.write_text(index, dat_path)

    print(f"\n{idx_path}  {idx['bytes'] / 1e6:>8.1f} MB   "
          f"{idx['terms']:,} terms, {idx['postings']:,} postings, "
          f"{idx['buckets_used']:,}/{libsearch.NUM_BUCKETS:,} buckets used")
    print(f"{dat_path}  {dat['bytes'] / 1e6:>8.1f} MB")

    accumulator = index.num_docs
    print(f"\naccumulator {accumulator / 1024:.0f} KB resident "
          f"(one byte per article)")
    if accumulator > 380 * 1024:
        print("  WARNING: that leaves under 130KB of Agon SRAM for the "
              "program. Use --limit, or shard the corpus.")

    if args.no_binary:
        return

    import buildwikibin

    bin_path = stem.with_suffix(".bin")
    builder = buildwikibin.build(index.num_docs,
                                 idx_path.name.upper(), dat_path.name.upper())
    builder.save(str(bin_path))
    size = len(builder.code)
    print(f"{bin_path}  {size / 1024:>8.1f} KB   "
          f"(reads {idx_path.name.upper()} and {dat_path.name.upper()} by name)")
    print("\nCopy all three onto the card. The binary carries no corpus, so "
          "rebuilding\nthe database and re-running this replaces the card "
          "without touching it.")


if __name__ == "__main__":
    main()
