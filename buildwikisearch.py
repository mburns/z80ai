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

import buildwikibin

# Only for the climb-limit default. The rest of the graph modules stay inside
# `build_graph`, which is where the ones that want numpy belong.
import libgraphcard
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


def build_graph(args: argparse.Namespace, stem: Path,
                titles: list[str],
                num_docs: int) -> buildwikibin.OracleSpec:
    """Write the .GRF beside the index, and describe it to the binary.

    Built from the same `titles` the index was, which is the only way the ids
    can mean the same articles. The digest in the header lets the program say
    so at startup; this asserts it here as well, because a card that is wrong
    at build time is cheaper to notice than one that is wrong on the machine.
    """
    import buildwikigraph
    import libinfer

    model = libinfer.Model.load(str(args.relations))
    if model.phrases is None:
        raise SystemExit(f"{args.relations} is not a phrasebook model; train "
                         f"one with classify.py on data/questions/relations.py")

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    doc = {t: i for i, t in enumerate(titles)}
    relations = sorted({r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (args.source,))})
    rid = {name: i for i, name in enumerate(relations)}

    edges, outside = [], 0
    for subject, relation, obj in db.execute(
            "SELECT subject, relation, object FROM edge WHERE source = ?",
            (args.source,)):
        left, right = doc.get(subject), doc.get(obj)
        if left is None or right is None:
            outside += 1
            continue
        edges.append((left, rid[relation], right))

    types: dict[str, list[int]] = {}
    for kind, entity in db.execute(
            "SELECT kind, entity FROM entity_type WHERE source = ?",
            (args.source,)):
        if (hit := doc.get(entity)) is not None:
            types.setdefault(kind, []).append(hit)
    db.close()

    paths = buildwikigraph.paths_for(model.phrases, relations, sorted(types))
    graph = libgraphcard.build(titles, edges, relations, types, paths)
    grf_path = stem.with_suffix(".GRF")
    stats = libgraphcard.write(graph, grf_path)

    print(f"\n{grf_path}  {stats['bytes'] / 1e6:>8.1f} MB   "
          f"{stats['edges']:,} edges over {len(relations)} relations "
          f"({outside:,} outside this build)")
    print(f"{' ' * len(str(grf_path))}  {sum(1 for p in paths if p)} of "
          f"{len(paths)} phrases are a path this can walk")

    assert graph.digest == libgraphcard.corpus_digest(titles)
    return buildwikibin.OracleSpec(
        graph_name=grf_path.name.upper(), forward_at=stats["forward_at"],
        num_edges=stats["edges"],
        types_at=libgraphcard.CardGraph(grf_path)._types_at
        - 8 * len(graph.types),
        num_types=len(graph.types), num_docs=num_docs, digest=graph.digest,
        paths=paths,
        climb_limit=args.climb_limit,
        model=libinfer.load_for_build(str(args.relations),
                                      report_io=False))


def report_ceiling(num_docs: int, image_bytes: int, what: str) -> None:
    """Say how much of the machine this corpus has used up.

    The accumulator is one byte an article and the page table one byte per
    256 of them, so a corpus spends the free gap from both ends at once.
    `buildwikibin.max_docs` knows where they meet, and the figure that used
    to be here was a guess at the program's size that came out sixteen times
    too large.
    """
    limit = buildwikibin.max_docs(buildwikibin.fixed_bytes(num_docs,
                                                           image_bytes))
    spare = buildwikibin.headroom(num_docs, image_bytes)
    print(f"  {num_docs:,} of the {limit:,} articles {what} can score in "
          f"SRAM ({num_docs / limit:.0%}), {spare:,} bytes spare")


def main(argv: list[str] | None = None) -> None:
    """`argv` is for callers that are not the command line.

    A corpus with its own climbs has to register them in `libgraph.CLIMB`
    before `paths_for` reads it, which means importing that corpus's module
    first - see `data/silo/buildcard.py`. Passing the arguments in beats
    assigning to `sys.argv` around a call.
    """
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
    parser.add_argument("--relations", type=Path,
                        help="Phrasebook model over relation paths. With it "
                             "the card gains a .GRF and the binary answers "
                             "from the fact graph before it lists articles")
    parser.add_argument("--climb-limit", type=int,
                        default=libgraphcard.CLIMB_LIMIT,
                        help="How many times a climb may step before giving "
                             "up. Counts hops, not nodes, so a pedigree n "
                             "generations deep needs n. Free in card bytes; "
                             f"costs probes only where it is used (default "
                             f"{libgraphcard.CLIMB_LIMIT})")
    args = parser.parse_args(argv)

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
    # Emitting the search program costs milliseconds and no files, and it is
    # the only thing that knows how much room the corpus has left. If it does
    # not fit, its own assertion says so better than a threshold could.
    try:
        report_ceiling(index.num_docs,
                       len(buildwikibin.build(index.num_docs).code),
                       "a search card")
    except AssertionError as exc:
        print(f"  WARNING: {exc}.\n  Use --limit, or shard the corpus.")

    if args.no_binary:
        return

    # The graph is built here rather than by a second command, because its ids
    # are positions in *this* article list. Two commands means two --limit
    # values to keep in step, and a mismatch answers fluently and wrongly.
    spec = None
    if args.relations:
        spec = build_graph(args, stem, titles, index.num_docs)

    bin_path = stem.with_suffix(".bin")
    builder = buildwikibin.build(index.num_docs,
                                 idx_path.name.upper(), dat_path.name.upper(),
                                 oracle=spec)
    builder.save(str(bin_path))
    size = len(builder.code)
    print(f"{bin_path}  {size / 1024:>8.1f} KB   "
          f"(reads {idx_path.name.upper()} and {dat_path.name.upper()} by name)")
    report_ceiling(index.num_docs, size, "this image")
    print(f"\nCopy all {'four' if spec else 'three'} onto the card. The binary "
          f"carries no corpus, so rebuilding\nthe database and re-running this "
          f"replaces the card without touching it.")


if __name__ == "__main__":
    main()
