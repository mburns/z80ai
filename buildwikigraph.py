#!/usr/bin/env python3
"""
Build the fact graph card from data/simple_english_wikipedia.db.

    python buildwikigraph.py --out dist/WIKI --relations relations.npz
    python buildwikigraph.py --out dist/WIKI --limit 20000   # match a small card

Writes `WIKI.GRF` beside the `WIKI.IDX` and `WIKI.DAT` that
`buildwikisearch.py` writes. The three are one card and must be built from the
same corpus: an edge names articles by *document id*, and a document id is a
position in the search card's article list rather than anything stable.

**Pass the same `--limit` to both.** A limited card reorders the corpus by
notability, so the same title takes a different id, and a mismatched pair does
not fail - it answers fluently and wrongly, because every id in the wrong graph
is still some article. The header carries a digest of the title list so the
program can refuse the pair, and this script prints it so a human can compare.

## What ends up on the card

The graph as libgraph resolved it, with titles replaced by ids and relation
names replaced by indices. Anything naming an article outside this build is
dropped, which for a limited card is most of it.

The paths table is the other half of the phrasebook: the classifier answers
with a phrase index, and the card has to know that index 3 means "born_in,
then climb located_in until the value is a country". Built from the model's own
phrase list so the two cannot disagree about what index 3 is.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import libgraph
import libgraphcard

DB_PATH = Path(__file__).resolve().parent / "data" / "simple_english_wikipedia.db"


def corpus(db: sqlite3.Connection, source: str,
           limit: int | None) -> tuple[list[str], dict[str, int]]:
    """The same article list, in the same order, that buildwikisearch takes.

    Imported rather than copied would be better still, but that function also
    loads leads and redirects for the index; this needs only the ordering. The
    queries are identical and the digest catches it if they ever stop being.
    """
    if limit:
        rows = db.execute(
            "SELECT a.title, "
            "       (SELECT COUNT(*) FROM redirect r "
            "         WHERE r.source = a.source AND r.target = a.title) AS fame "
            "FROM article a WHERE a.source = ? "
            "ORDER BY fame DESC, LENGTH(a.lead) DESC LIMIT ?",
            (source, limit)).fetchall()
    else:
        rows = db.execute(
            "SELECT title FROM article WHERE source = ? ORDER BY id",
            (source,)).fetchall()
    titles = [r[0] for r in rows]
    return titles, {t: i for i, t in enumerate(titles)}


def paths_for(phrases: list[str], relations: list[str],
              types: list[str]) -> list[list[tuple[int, int]]]:
    """Turn the model's phrase list into a step list per phrase.

    A phrase is a path written out - "BORN_IN IN_COUNTRY" - and each word is
    either a relation, one of libgraph's climbs, or a relation read backwards.

    `born_in_of` is "who was born here": the same row from the other end, which
    the card already holds sorted the other way. It costs a flag on the step's
    relation byte, and without it a third of this vocabulary is inert.
    """
    out: list[list[tuple[int, int]]] = []
    for phrase in phrases:
        steps: list[tuple[int, int]] = []
        for word in phrase.lower().split():
            if word in libgraph.CLIMB:
                relation, kind = libgraph.CLIMB[word]
                if relation in relations and kind in types:
                    steps.append((relations.index(relation), types.index(kind)))
                    continue
                steps = []
                break
            if word in relations:
                steps.append((relations.index(word), libgraphcard.PLAIN))
                continue
            if word.endswith("_of") and word[:-3] in relations:
                steps.append((relations.index(word[:-3]) | libgraphcard.INVERSE,
                              libgraphcard.PLAIN))
                continue
            steps = []              # a relation this corpus has no edges for
            break
        out.append(steps)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--source", default="simplewiki")
    parser.add_argument("--out", default="dist/WIKI",
                        help="Path stem; .GRF is written beside .IDX and .DAT")
    parser.add_argument("--limit", type=int, default=None,
                        help="Must match the --limit buildwikisearch was given")
    parser.add_argument("--relations", type=Path,
                        help="Phrasebook model, for the paths table")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}\n"
                         f"  python data/wikipedia/ingest.py <dump>")
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    titles, doc = corpus(db, args.source, args.limit)
    print(f"{len(titles):,} articles")

    relations = sorted({r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (args.source,))})
    relation_id = {name: i for i, name in enumerate(relations)}

    edges, outside = [], 0
    for subject, relation, obj in db.execute(
            "SELECT subject, relation, object FROM edge WHERE source = ?",
            (args.source,)):
        left, right = doc.get(subject), doc.get(obj)
        if left is None or right is None:
            outside += 1
            continue
        edges.append((left, relation_id[relation], right))
    print(f"{len(edges):,} edges over {len(relations)} relations "
          f"({outside:,} naming an article outside this build)")

    types: dict[str, list[int]] = {}
    for kind, entity in db.execute(
            "SELECT kind, entity FROM entity_type WHERE source = ?",
            (args.source,)):
        if (hit := doc.get(entity)) is not None:
            types.setdefault(kind, []).append(hit)
    print("  " + ", ".join(f"{k}: {len(v):,}" for k, v in sorted(types.items())))

    phrases: list[str] = []
    if args.relations:
        import libinfer
        model = libinfer.Model.load(str(args.relations))
        if model.phrases is None:
            raise SystemExit(f"{args.relations} is not a phrasebook model")
        phrases = model.phrases
    paths = paths_for(phrases, relations, sorted(types))
    walkable = sum(1 for p in paths if p)
    if phrases:
        print(f"{walkable} of {len(phrases)} phrases are a path this can walk")

    graph = libgraphcard.build(titles, edges, relations, types, paths)
    out = Path(args.out).with_suffix(".GRF")
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = libgraphcard.write(graph, out)

    print(f"\n{out}  {stats['bytes'] / 1e6:>8.1f} MB   "
          f"{stats['edges']:,} edges, forward at {stats['forward_at']:,}, "
          f"reverse at {stats['reverse_at']:,}")
    print(f"corpus digest {graph.digest:08x} over {graph.num_docs:,} documents "
          f"- the card's .IDX must agree")


if __name__ == "__main__":
    main()
