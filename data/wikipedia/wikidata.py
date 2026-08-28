#!/usr/bin/env python3
"""
Wikidata statements for the articles this corpus already has.

    # once, against a Wikidata graph dump (needs `ladybug`, ~22GB on disk)
    python data/wikipedia/wikidata.py --export wikidata.lbdb -o wikidata.tsv.gz

    # thereafter, against the exported file (no exotic dependencies)
    python data/wikipedia/wikidata.py --score wikidata.tsv.gz
    python data/wikipedia/wikidata.py --write wikidata.tsv.gz --rebuild-graph

`ingest.py` reads an encyclopedia written for people, and the ceiling on it is
coverage: 46% of articles carry an infobox and the rest say what they say in
prose, in a category, or not at all. Wikidata has the same facts as a table.

**The join is a table, not a guess.** `sitelink` carries the Q-id of every
article, which is the only exact key between the two - see the README. Matching
on the English label instead is right 43.5% of the time and confidently wrong
2.3% of the time, and nothing about the wrong 2.3% looks wrong.

## Why this is two programs

Reading the graph dump needs `ladybug` and 22GB of disk for a database of 91.6M
nodes and 766.5M edges. Nothing else here needs either, and CI needs neither, so
the export is separated from everything that consumes it: it runs once and
writes a file of a few megabytes, keyed by Q-id, and every later step reads that
with the standard library.

Keyed by Q-id rather than by title on purpose. A title is a fact about one
snapshot of one wiki - it changes when an article is renamed, and 726 of them
changed in this corpus the day the escaping was fixed. A Q-id does not, so the
export outlives the corpus it was cut against and the join is redone from
`sitelink` each time.

## What is exported

Only statements where **both ends are articles in this corpus**, because an edge
whose object has no article is one the card can never name. That is what makes
the file small: 766.5M edges in the dump, and about half a million that this
encyclopedia could use.
"""

from __future__ import annotations

import argparse
import gzip
import sqlite3
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import libgraph

DB_PATH = Path(__file__).resolve().parent.parent / "simple_english_wikipedia.db"

#: Wikidata property -> the relation libgraph already walks, for the properties
#: whose meaning survives the crossing. Two of them land on `located_in`,
#: because that is what libgraph does with the `country` field too - collapsing
#: country onto containment is what made chaining work, and `in_country` is a
#: *question* answered by climbing it, never an edge.
#:
#: P17 is not safe on its own: on a place it is where the place is, and on a
#: *language* it is where the language is spoken, which is how `English
#: language` acquires ninety of them. The importer types the subject before
#: taking one - see `build_plan`.
PROPERTY = {
    19: "born_in",
    20: "died_in",
    131: "located_in",
    17: "located_in",
    36: "capital_is",
    26: "spouse_of",
    170: "created_by",
    50: "created_by",
    136: "genre_is",
}

#: Classes worth knowing about a subject, because the corpus currently infers
#: both by heuristic: who is a person comes from birth-year categories, and what
#: is a country from a vote over infoboxes that once elected California.
CLASS = {5: "human", 6256: "country"}

#: Of those, the ones written into `derived` for `libgraph.types` to read.
#: `human` is measured and not written: personhood is decided inside `libgraph`
#: and stored nowhere, so a row asserting it would have no reader.
TYPED = frozenset({"country"})

#: `instance of`, which is how a class is stated.
P_INSTANCE_OF = 31

#: `located in the administrative territorial entity`, which is also the type
#: test: only a place is administratively inside something, so having one is
#: what makes a subject eligible for `country`.
P_ADMIN_IN = 131

#: `country`, which needs that test.
P_COUNTRY = 17

#: Written into the export so a file can say what it came from.
FORMAT = 1

#: What `derived` records as the producer of these rows. It is part of that
#: table's primary key, so this can disagree with `regex` about the same person
#: and both rows survive - which is what makes one measurable against the other.
METHOD = "wikidata"


def sitelinks(db: sqlite3.Connection, source: str) -> dict[int, str]:
    """qid -> title, for every article that has one."""
    return {q: t for t, q in db.execute(
        "SELECT title, qid FROM sitelink WHERE source = ?", (source,))}


#: Containment, for proving that one place is inside another. The importer
#: needs the *whole* chain and this corpus has articles for only part of it -
#: `Carrollton, Mississippi` reaches `Mississippi` through a county nobody
#: wrote about - so these are collected past the edge of the corpus, unlike
#: everything else here.
CONTAINMENT = (131, 17)

#: How far above the corpus to follow containment. Six is past the deepest real
#: chain (suburb, city, county, state, country) with room to spare; the point of
#: a bound is that a cycle in Wikidata cannot spin here forever.
CHAIN_DEPTH = 6


def export(dump: Path, out: Path, qids: set[int]) -> tuple[int, int, int]:
    """Write every usable statement about ``qids`` to a gzipped TSV.

    Both ends have to be in ``qids``: an edge whose object has no article is one
    the card can never name. Two exceptions, both because they answer a question
    rather than name an article - a class (`instance of human`), and a
    containment link being followed above the corpus to prove ancestry.

    Read through parquet rather than row by row. The obvious loop pulls 78M rows
    across the binding one at a time and does not finish in ten minutes; the
    engine writes the same rows to a file in twenty seconds, and the filtering
    is a vectorised membership test after that.
    """
    import ladybug as lb
    import numpy as np
    import pyarrow.parquet as pq

    con = lb.Connection(lb.Database(str(dump), read_only=True))
    props = ",".join(str(p) for p in PROPERTY)
    classes = ",".join(str(q) for q in CLASS)
    scratch = out.with_suffix(".raw.parquet")

    started = time.time()
    con.execute(
        "COPY (MATCH (a:wikidata_node)-[r:wikidata_rel]->(b:wikidata_node) "
        f"WHERE r.property IN [{props}] "
        f"   OR (r.property = {P_INSTANCE_OF} AND b.qid IN [{classes}]) "
        "RETURN a.qid AS subj, r.property AS prop, b.qid AS obj) "
        f"TO '{scratch}'")
    print(f"  scanned in {time.time() - started:.0f}s", flush=True)

    corpus = np.sort(np.fromiter(qids, dtype=np.int64))

    def among(values: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Vectorised `in`, for tables far too big for a Python loop."""
        i = np.searchsorted(table, values)
        i[i >= len(table)] = 0
        hit: np.ndarray = table[i] == values
        return hit

    def batches() -> Iterator[pa.RecordBatch]:
        yield from pq.ParquetFile(scratch).iter_batches(batch_size=4_000_000)

    # The containment chain, found by walking up from the corpus one hop at a
    # time. Each round is a full pass, which is cheap and much simpler than
    # holding 24M parent links in memory to walk them properly.
    chain: list[tuple[int, int, int]] = []
    seen = corpus
    frontier = corpus
    for _ in range(CHAIN_DEPTH):
        found: list[np.ndarray] = []
        for batch in batches():
            s = batch.column("subj").to_numpy()
            p = batch.column("prop").to_numpy()
            o = batch.column("obj").to_numpy()
            keep = np.isin(p, CONTAINMENT) & among(s, frontier)
            if keep.any():
                chain.extend(zip(s[keep].tolist(), p[keep].tolist(),
                                 o[keep].tolist(), strict=True))
                found.append(o[keep])
        if not found:
            break
        fresh = np.unique(np.concatenate(found))
        frontier = np.sort(fresh[~among(fresh, seen)])
        if not len(frontier):
            break
        seen = np.sort(np.concatenate([seen, frontier]))

    statements = classed = 0
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        fh.write(f"# format\t{FORMAT}\n# dump\t{dump.name}\n")
        for prop, relation in sorted(PROPERTY.items()):
            fh.write(f"# property\t{prop}\t{relation}\n")
        for batch in batches():
            s = batch.column("subj").to_numpy()
            p = batch.column("prop").to_numpy()
            o = batch.column("obj").to_numpy()
            mine = among(s, corpus)
            usable = mine & (among(o, corpus) | (p == P_INSTANCE_OF))
            for subj, prop, obj in zip(s[usable].tolist(), p[usable].tolist(),
                                       o[usable].tolist(), strict=True):
                fh.write(f"{subj}\t{prop}\t{obj}\n")
                if prop == P_INSTANCE_OF:
                    classed += 1
                else:
                    statements += 1
        # Marked, because a chain row is not a fact about the corpus: it is
        # scaffolding for deciding whether one place is inside another, and
        # nothing should mistake it for an edge worth putting on a card.
        for subj, prop, obj in sorted(set(chain)):
            fh.write(f"{subj}\t{prop}\t{obj}\tchain\n")

    scratch.unlink()
    return statements, classed, len(set(chain))


def read_export(path: Path) -> tuple[list[tuple[int, int, int]],
                                     dict[int, set[int]], dict[str, str]]:
    """(statements, containment parents, header) from an export."""
    rows: list[tuple[int, int, int]] = []
    parents: dict[int, set[int]] = defaultdict(set)
    header: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                parts = line[1:].split("\t")
                if len(parts) >= 2:
                    header[parts[0].strip()] = parts[1].strip()
                continue
            fields = line.rstrip("\n").split("\t")
            subj, prop, obj = int(fields[0]), int(fields[1]), int(fields[2])
            if len(fields) > 3 and fields[3] == "chain":
                parents[subj].add(obj)
            else:
                rows.append((subj, prop, obj))
                if prop in CONTAINMENT:
                    # A statement about a corpus article is also a link in the
                    # chain; it is written once and read as both.
                    parents[subj].add(obj)
    return rows, parents, header


def inside(child: int, ancestor: int, parents: dict[int, set[int]]) -> bool:
    """Is ``child`` contained by ``ancestor``, per Wikidata's own chain?

    This is what decides a disagreement. The corpus says a person was born in
    `Mississippi` and Wikidata says `Carrollton, Mississippi`; neither is wrong,
    but one is an answer and the other is most of one. Wikidata is preferred
    only where it can be *shown* to sit inside what the corpus already said -
    which is a fact about the world, not a preference between two sources.

    Breadth-first and bounded, because containment in Wikidata has cycles in it
    and a corpus is not the place to discover that with a stack overflow.
    """
    if child == ancestor:
        return False                          # equal is not more specific
    seen = {child}
    frontier = {child}
    for _ in range(CHAIN_DEPTH):
        nxt: set[int] = set()
        for node in frontier:
            for parent in parents.get(node, ()):
                if parent == ancestor:
                    return True
                if parent not in seen:
                    seen.add(parent)
                    nxt.add(parent)
        if not nxt:
            return False
        frontier = nxt
    return False


def resolve(rows: list[tuple[int, int, int]], titles: dict[int, str]
            ) -> tuple[dict[int, dict[int, set[int]]], dict[str, set[int]],
                       set[int]]:
    """(property -> subject -> objects), classes, and everything with a P131.

    Keyed by property rather than relation because two properties land on
    `located_in` and only one of them needs typing; collapsing them here would
    throw away which is which before the guard could run.

    Kept as Q-ids rather than titles, because containment is decided over them
    and half the chain has no article to be titled with. Anything naming a Q-id
    this corpus has no article for is dropped here rather than at export time:
    the export outlives the corpus, and an article missing today may exist after
    the next dump.

    The P131 set is the type test. `country` on a place is where it is, and
    `country` on a *language* is where it is spoken, which is how
    `English language` acquires ninety of them. Rather than carry a list of what
    counts as a place, this asks Wikidata the question it already answers: only
    a place is administratively inside something.
    """
    facts: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    classes: dict[str, set[int]] = defaultdict(set)
    placed: set[int] = set()
    for subj, prop, obj in rows:
        if subj not in titles:
            continue
        if prop == P_INSTANCE_OF:
            if obj in CLASS:
                classes[CLASS[obj]].add(subj)
            continue
        if prop == P_ADMIN_IN:
            placed.add(subj)
        if obj in titles:
            facts[prop][subj].add(obj)
    return facts, classes, placed


def choose(values: set[int], parents: dict[int, set[int]]) -> int | None:
    """The one value that is inside all the others, or None.

    `derived` holds one object per subject and relation, so a subject with more
    than one has to be resolved or declined. Where the values nest this is not a
    choice at all - `Sialkot`, `Punjab Province` and `British Raj` are three
    depths of the same answer and the innermost is the answer.

    Where they do not nest there is no non-arbitrary pick. Everest is in China
    and in Nepal; a band is nine genres. Declining loses the row, and picking
    would put a fluent half-truth on a card with nothing to mark it as one. A
    missing answer is silent and a wrong one is not.
    """
    if len(values) == 1:
        return next(iter(values))
    innermost = [v for v in values
                 if all(v == other or inside(v, other, parents)
                        for other in values)]
    return innermost[0] if len(innermost) == 1 else None


#: What a subject/relation pair was decided to be, in the order the outcomes
#: matter. `refine` is the only one that changes an answer the card already
#: gives, which is why it is counted apart from `gap`.
OUTCOMES = ("gap", "refine", "agree", "kept", "declined", "untyped",
            "coarser", "typed")


class Plan:
    """What an import would do, decided before anything is written.

    Separated from writing so that `--score` and `--write` cannot disagree
    about what the import does: the report is the plan, printed.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []      # subject, relation, object
        self.counts: dict[str, dict[str, int]] = defaultdict(
            lambda: dict.fromkeys(OUTCOMES, 0))
        self.examples: dict[str, list[str]] = defaultdict(list)
        #: (subject, relation) pairs the plan overrules, which is exactly what
        #: `libgraph.build(replace=...)` will accept and nothing more.
        self.replaces: set[tuple[str, str]] = set()

    def note(self, relation: str, outcome: str) -> None:
        self.counts[relation][outcome] += 1


def build_plan(facts: dict[int, dict[int, set[int]]],
               parents: dict[int, set[int]],
               existing: dict[str, dict[str, set[str]]],
               placed: set[int],
               titles: dict[int, str]) -> Plan:
    """Decide every subject and relation against what the corpus already says.

    Three rules, in order:

    **Type first.** `in_country` is only taken for something Wikidata says is
    administratively inside something else, so a language does not acquire the
    ninety countries it is spoken in.

    **Gaps are free.** Where the corpus has no edge, Wikidata's is taken.

    **A conflict is settled by containment, not by preference.** Where both
    have an answer and they differ, Wikidata's is taken only if it can be shown
    to lie inside the corpus's - `Carrollton, Mississippi` inside `Mississippi`.
    That is a refinement of a true answer rather than a contradiction of it.
    Anything else leaves the encyclopedia's answer alone.
    """
    plan = Plan()
    qid_of = {t: q for q, t in titles.items()}

    # Type first, then collapse onto relations. `country` is only taken for
    # something Wikidata puts administratively inside something else, so a
    # language does not acquire the ninety countries it is spoken in; and the
    # two properties that mean containment are unioned before anything is
    # chosen between, because they are one question asked twice.
    merged: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for prop, subjects in facts.items():
        relation = PROPERTY[prop]
        for subject, values in subjects.items():
            if prop == P_COUNTRY and subject not in placed:
                plan.note(relation, "untyped")
                continue
            merged[relation][subject] |= values

    for relation in sorted(merged):
        current = existing.get(relation, {})
        for subject, values in merged[relation].items():
            pick = choose(values, parents)
            if pick is None:
                plan.note(relation, "declined")
                continue
            title, obj = titles[subject], titles[pick]
            held = current.get(title)
            if not held:
                plan.note(relation, "gap")
                plan.rows.append((title, relation, obj))
            elif obj in held:
                plan.note(relation, "agree")
            elif any((q := qid_of.get(h)) is not None and inside(pick, q, parents)
                     for h in held):
                plan.note(relation, "refine")
                plan.rows.append((title, relation, obj))
                plan.replaces.add((title, relation))
                if len(plan.examples[relation]) < 5:
                    plan.examples[relation].append(
                        f"{title}: {min(held)} -> {obj}")
            else:
                plan.note(relation, "kept")
    return plan


#: Relations whose object is a place, and so whose value has to remain able to
#: climb to a country.
PLACED_RELATIONS = ("born_in", "died_in", "located_in")


def drop_unclimbable(plan: Plan, existing: dict[str, dict[str, set[str]]],
                     countries: set[str]) -> None:
    """Refuse a refinement that would cost the answer to "what country".

    A finer birthplace is only better if it still reaches a country.
    `Carl Wieman: Oregon -> Corvallis, Oregon` is more precise and, on the
    graph as it stands, unreachable - Corvallis is in this encyclopedia and
    nothing in it says Corvallis is in Oregon, because the article has no
    infobox at all.

    **Judged against the graph this import leaves, not the one it found.** The
    first measurement of this used today's edges and reported 6,879 casualties;
    the same import supplies `Corvallis -> Benton County` and most of them were
    never going to happen. Measured after, it is 909, against 1,319 subjects
    that could not climb before and can now.
    """
    up: dict[str, set[str]] = {
        subject: set(objects)
        for subject, objects in existing.get("located_in", {}).items()}
    for subject, relation, obj in plan.rows:
        if relation == "located_in":
            if (subject, relation) in plan.replaces:
                up[subject] = {obj}
            else:
                up.setdefault(subject, set()).add(obj)

    def climbs(place: str) -> bool:
        if place in countries:
            return True
        seen, frontier = {place}, {place}
        for _ in range(CHAIN_DEPTH):
            nxt = set()
            for node in frontier:
                for parent in up.get(node, ()):
                    if parent in countries:
                        return True
                    if parent not in seen:
                        seen.add(parent)
                        nxt.add(parent)
            if not nxt:
                return False
            frontier = nxt
        return False

    kept: list[tuple[str, str, str]] = []
    for subject, relation, obj in plan.rows:
        pair = (subject, relation)
        if (relation not in PLACED_RELATIONS or pair not in plan.replaces
                or climbs(obj)):
            kept.append((subject, relation, obj))
            continue
        # The coarse answer the corpus already holds outlives the fine one.
        held = existing.get(relation, {}).get(subject, set())
        if not any(climbs(o) for o in held):
            kept.append((subject, relation, obj))
            continue
        plan.replaces.discard(pair)
        plan.counts[relation]["refine"] -= 1
        plan.note(relation, "coarser")
    plan.rows = kept


def corpus_edges(db: sqlite3.Connection, source: str
                 ) -> dict[str, dict[str, set[str]]]:
    """What the card can already answer, as relation -> subject -> objects."""
    have: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for subject, relation, obj in db.execute(
            "SELECT subject, relation, object FROM edge WHERE source = ?",
            (source,)):
        have[relation][subject].add(obj)
    return have


#: Column widths for the report, one per entry in OUTCOMES.
WIDTHS = (9, 8, 8, 8, 9, 8, 8, 7)


def _columns(counts: dict[str, int]) -> str:
    return " ".join(f"{counts[o]:>{w},}"
                    for o, w in zip(OUTCOMES, WIDTHS, strict=True))


def report(plan: Plan) -> None:
    print("\n" + f"{'relation':12s} " + " ".join(
        f"{name:>{w}s}" for name, w in zip(OUTCOMES, WIDTHS, strict=True)))
    total = dict.fromkeys(OUTCOMES, 0)
    for relation, counts in sorted(plan.counts.items()):
        for outcome in OUTCOMES:
            total[outcome] += counts[outcome]
        print(f"{relation:12s} {_columns(counts)}")
    print(f"{'':12s} {_columns(total)}")
    print(f"\n{len(plan.rows):,} rows: {total['gap']:,} where the corpus had "
          f"nothing, {total['refine']:,} where Wikidata is inside what it had")
    for relation, shown in sorted(plan.examples.items()):
        for line in shown[:3]:
            print(f"  [{relation}] {line}")


def write(db: sqlite3.Connection, source: str, plan: Plan, path: Path,
          header: dict[str, str]) -> int:
    """Replace this method's rows in `derived`, atomically.

    Nothing reaches `fact`. A reader wanting only what the encyclopedia
    tabulated reads that table and never sees any of this, which is the whole
    reason `derived` keys on the method that produced a row.
    """
    with db:
        db.execute("DELETE FROM derived WHERE source = ? AND method = ?",
                   (source, METHOD))
        db.executemany(
            "INSERT OR REPLACE INTO derived VALUES (?, ?, ?, ?, ?)",
            [(source, subject, relation, obj, METHOD)
             for subject, relation, obj in plan.rows])
        for key, value in (
                (f"{source}.wikidata", str(len(plan.rows))),
                (f"{source}.wikidata.export", path.name),
                (f"{source}.wikidata.dump", header.get("dump", "?")),
                (f"{source}.wikidata.written",
                 time.strftime("%Y-%m-%dT%H:%M:%S"))):
            db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
    return len(plan.rows)


def load(db: sqlite3.Connection, source: str, path: Path
         ) -> tuple[Plan, dict[str, Any]]:
    """Read an export, decide against the corpus, and hand back the plan."""
    rows, parents, header = read_export(path)
    titles = sitelinks(db, source)
    facts, classes, placed = resolve(rows, titles)
    print(f"{len(rows):,} statements and {len(parents):,} containment links "
          f"from {header.get('dump', '?')}, against {len(titles):,} sitelinks")
    existing = corpus_edges(db, source)
    plan = build_plan(facts, parents, existing, placed, titles)
    countries = {e for (e,) in db.execute(
        "SELECT entity FROM entity_type WHERE source = ? AND kind = 'country'",
        (source,))} if _has_entity_type(db) else set()
    drop_unclimbable(plan, existing, countries)

    # What a thing *is*, not where it is. Only `country` crosses: the corpus
    # decides it by a vote over infobox fields that once elected California,
    # and 94 of these it does not know about at all. `human` is exported and
    # not written, because nothing stores personhood - `libgraph` decides it
    # and a row here would be a fact with no reader.
    for kind, members in sorted(classes.items()):
        if kind not in TYPED:
            continue
        for qid in sorted(members):
            if (title := titles.get(qid)) is not None:
                plan.rows.append((title, libgraph.TYPE_RELATION, kind))
                plan.note(kind, "typed")

    return plan, {"header": header, "classes": classes, "titles": titles}


def report_classes(db: sqlite3.Connection, source: str,
                   classes: dict[str, set[int]], titles: dict[int, str]) -> None:
    """What Wikidata says a thing *is*, against what the corpus guessed.

    Both of these the corpus currently infers: what is a country comes from a
    vote over infoboxes that once elected California, and who is a person from
    birth-year categories. Nothing here writes either - `entity_type` is
    libgraph's to build - but the size of the disagreement is worth printing.

    Only `country` is compared. `entity_type` holds that kind and no other, so
    a person would read as "0 in the corpus" and invite somebody to fix a
    disagreement that does not exist: personhood is decided in `libgraph` and
    never stored. Printing a comparison against a table that was never asked
    the question is worse than printing none.
    """
    if not _has_entity_type(db):
        return
    print()
    stored = {kind for (kind,) in db.execute(
        "SELECT DISTINCT kind FROM entity_type WHERE source = ?", (source,))}
    for kind, members in sorted(classes.items()):
        named = {titles[q] for q in members if q in titles}
        if kind not in stored:
            print(f"{kind:8s} wikidata {len(named):>7,}   "
                  f"the corpus does not record this kind")
            continue
        current = {t for (t,) in db.execute(
            "SELECT entity FROM entity_type WHERE source = ? AND kind = ?",
            (source, kind))}
        print(f"{kind:8s} wikidata {len(named):>7,}   corpus {len(current):>7,}"
              f"   only wikidata {len(named - current):>7,}"
              f"   only corpus {len(current - named):>6,}")


def _has_entity_type(db: sqlite3.Connection) -> bool:
    return bool(db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'entity_type'").fetchone())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--source", default="simplewiki")
    parser.add_argument("--export", type=Path, metavar="WIKIDATA.LBDB",
                        help="Cut a fresh export from a Wikidata graph dump")
    parser.add_argument("-o", "--out", type=Path, default=Path("wikidata.tsv.gz"),
                        help="Where --export writes")
    parser.add_argument("--score", type=Path, metavar="WIKIDATA.TSV.GZ",
                        help="Report what importing an export would change, "
                             "and write nothing")
    parser.add_argument("--write", type=Path, metavar="WIKIDATA.TSV.GZ",
                        help="Import an export into `derived`")
    parser.add_argument("--rebuild-graph", action="store_true",
                        help="Put the imported rows on the card. Without this "
                             "they sit in `derived` and nothing walks them.")
    args = parser.parse_args(argv)

    writing = bool(args.write)
    db = sqlite3.connect(args.db) if writing else sqlite3.connect(
        f"file:{args.db}?mode=ro", uri=True)
    if not _has_sitelinks(db, args.source):
        raise SystemExit(
            f"{args.db} has no sitelinks for '{args.source}'. Run:\n"
            "  python data/wikipedia/ingest.py --sitelinks "
            "<page.sql.gz> <page_props.sql.gz>")

    if args.export:
        qids = set(sitelinks(db, args.source))
        print(f"exporting statements about {len(qids):,} articles "
              f"from {args.export.name}")
        statements, classed, chain = export(args.export, args.out, qids)
        print(f"{statements:,} statements, {classed:,} class rows and "
              f"{chain:,} containment links -> {args.out} "
              f"({args.out.stat().st_size / 1e6:.1f} MB)")
        return

    path = args.score or args.write
    if not path:
        parser.error("nothing to do: pass --export, --score or --write")

    plan, extra = load(db, args.source, path)
    report(plan)
    report_classes(db, args.source, extra["classes"], extra["titles"])

    if not writing:
        print("\nnothing written; pass --write to import these rows")
        return

    written = write(db, args.source, plan, path, extra["header"])
    print(f"\n{written:,} rows written to `derived` as method '{METHOD}'")

    # Same bargain as birthplaces.py: what is read out of somewhere other than
    # an infobox reaches the card only when somebody says so.
    if not args.rebuild_graph:
        print("`edge` is unchanged, so no card and no answer moves until you "
              "pass --rebuild-graph")
        return
    # Both methods, because admitting only the last to run would leave the
    # other's rows in `derived` while they silently vanished from `edge`.
    edges, _dropped = libgraph.build(db, args.source, report=print,
                                     derived=("regex", METHOD),
                                     replace=plan.replaces)
    db.commit()
    db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
               (f"{args.source}.edges", str(edges)))
    db.commit()
    print(f"{edges:,} edges")


def _has_sitelinks(db: sqlite3.Connection, source: str) -> bool:
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                      "AND name = 'sitelink'").fetchone():
        return False
    return bool(db.execute("SELECT 1 FROM sitelink WHERE source = ? LIMIT 1",
                           (source,)).fetchone())


if __name__ == "__main__":
    main()
