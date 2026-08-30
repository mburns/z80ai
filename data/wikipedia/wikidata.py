#!/usr/bin/env python3
"""
Wikidata statements for the articles this corpus already has.

    # once per Wikidata snapshot (needs `ladybug`, ~22GB on disk, hours)
    python data/wikipedia/wikidata.py --build latest-truthy.nt.bz2 -o wikidata.lbdb

    # once per corpus (needs `ladybug`, minutes)
    python data/wikipedia/wikidata.py --export wikidata.lbdb -o wikidata.tsv.gz

    # thereafter, against the exported file (no exotic dependencies)
    python data/wikipedia/wikidata.py --survey wikidata.tsv.gz
    python data/wikipedia/wikidata.py --score wikidata.tsv.gz
    python data/wikipedia/wikidata.py --write wikidata.tsv.gz --rebuild-graph

`ingest.py` reads an encyclopedia written for people, and the ceiling on it is
coverage: 46% of articles carry an infobox and the rest say what they say in
prose, in a category, or not at all. Wikidata has the same facts as a table.

**The join is a table, not a guess.** `sitelink` carries the Q-id of every
article, which is the only exact key between the two - see the README. Matching
on the English label instead is right 43.5% of the time and confidently wrong
2.3% of the time, and nothing about the wrong 2.3% looks wrong.

## Why this is three stages behind one dependency

Reading the graph dump needs `ladybug` and 22GB of disk for a database of 91.6M
nodes and 766.5M edges. Nothing else here needs either, and CI needs neither, so
everything that touches it is on one side of a line: `--build` makes the graph
and `--export` cuts it down, and what comes out is a file of a few megabytes,
keyed by Q-id, that every later step reads with the standard library.

`--build` is here rather than in a program of its own because it and `--export`
have to agree about the schema, and the way that agreement was previously kept
was that the graph existed on somebody's disk and nothing wrote down how. That
is a pipeline which works until a machine is wiped.

**The graph `--build` makes is not the graph those figures describe.** 766.5M
edges is what the previous one held; this keeps entity-to-entity truthy
statements and measures out at about 456M on the 2026-08-28 dump. The
difference is statements `export` could never have used - literal values, and
ranks below truthy - so the smaller graph is the same graph for every purpose
here. Where a number below says 766.5M it is describing the old build, and is
left alone rather than quietly rewritten to match.

Keyed by Q-id rather than by title on purpose. A title is a fact about one
snapshot of one wiki - it changes when an article is renamed, and 726 of them
changed in this corpus the day the escaping was fixed. A Q-id does not, so the
export outlives the corpus it was cut against and the join is redone from
`sitelink` each time.

## What is exported

Every statement where **both ends are articles in this corpus**, because an edge
whose object has no article is one the card can never name. That is what makes
the file small: 766.5M edges in the dump, and a few million that this
encyclopedia could use.

**Whatever the property, not only the mapped ones.** The export used to filter
on `PROPERTY` in the query, which quietly made the property map part of the
dump-scanning step: adding P57 meant rebuilding 22GB to answer a question the
previous export had already read past. It does not any more, so the cost of a
new relation is an edit to `PROPERTY` and a re-read of a file that is already on
disk. `--survey` prints what is in the file and unmapped, biggest first, which
is the same question `ingest.py --stats` answers for infobox fields.

That leaves three reasons to go back to the dump, none of which is "we want one
more relation": a newer Wikidata snapshot, a corpus whose article set changed,
or a bug in `export` itself.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import os
import shutil
import sqlite3
import subprocess
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
#:
#: This has to cover every property `data/questions/relations.py` builds a
#: question class from, and for one release it did not: the classifier was
#: trained to answer "who directed X" while the importer never fetched P57, so
#: the only `created_by` edges on a film were the ones its infobox happened to
#: tabulate - which is the 46% this whole file exists to get past. The two maps
#: are kept in step by `test_wikidata.py`, not by hand.
#:
#: **The order is the precedence**, for the eight that land on `created_by`. A
#: film has a director and a producer and a composer, and they are different
#: people rather than one answer at three depths, so `choose` declines all
#: three and the film gets nothing. The infobox path never had that problem
#: because `libgraph.CANONICAL` ranks its fields - director outranks producer -
#: and this is the same ranking in the same order. Containment is the exception
#: and is still unioned: P131 and P17 are one question asked twice.
PROPERTY = {
    19: "born_in",
    20: "died_in",
    131: "located_in",
    17: "located_in",
    36: "capital_is",
    26: "spouse_of",
    57: "created_by",     # director
    50: "created_by",     # author
    170: "created_by",    # creator
    86: "created_by",     # composer
    676: "created_by",    # lyricist
    178: "created_by",    # developer
    162: "created_by",    # producer
    84: "created_by",     # architect
    364: "language_is",   # original language of work
    37: "language_is",    # official language, which is a country's
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
#:
#: 2 carries every property rather than the nine `PROPERTY` held when it was
#: cut. A 1 is still readable - it is a subset - but `--survey` has nothing to
#: report on one and a newly mapped property will silently find no statements,
#: so the reader says so rather than letting that look like an absence of facts.
FORMAT = 2

#: The oldest export this can still read.
FORMAT_MIN = 1

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


#: The dump the builder reads.
#:
#:     https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2
#:
#: Truthy N-Triples rather than the full JSON export, because `export` reads a
#: subject, a property and an object and nothing else - truthy is those same
#: statements with the qualifiers, references and deprecated ranks already
#: dropped. A fraction of the bytes for exactly the columns that get used.
TRUTHY = "latest-truthy.nt.bz2"

#: A statement worth a row, as byte prefixes rather than a regex, because this
#: runs once per line over a file with billions of them.
NT_ENTITY = b"<http://www.wikidata.org/entity/Q"
NT_DIRECT = b"<http://www.wikidata.org/prop/direct/P"

#: Cut before Python sees a line at all. Labels, descriptions, aliases and
#: sitelinks are the bulk of the dump and none of them carry this, so a C loop
#: discards most of the file before the interpreter is involved.
PREFILTER = "/prop/direct/P"

#: Parallel bzip2 if this machine has one. 43.3GB compressed expands to ~762GB,
#: so decompression *is* the runtime rather than a part of it.
#:
#: `lbzip2` first because it is the one that parallelises a bzip2 file it did
#: not write, by finding the block boundaries. `pbzip2` only parallelises what
#: `pbzip2` compressed and falls back to one thread on anything else - measured
#: on this dump at 10.0s against plain `bzip2`'s 10.1s for the same 60MB, which
#: is to say no difference at all. It stays in the list because it costs
#: nothing and helps on a file it did write; it is not the one to install.
DECOMPRESSORS = (("lbzip2", "-dc"), ("pbzip2", "-dc"), ("bzip2", "-dc"))

#: What `export` expects to find, and the only place it is written down.
#: Column order in the edge file is load-bearing: a rel COPY reads the first
#: two columns as the endpoints and everything after them as properties.
NODE_TABLE = "CREATE NODE TABLE wikidata_node(qid INT64, PRIMARY KEY(qid))"
REL_TABLE = ("CREATE REL TABLE wikidata_rel("
             "FROM wikidata_node TO wikidata_node, property INT64)")

#: Triples held before a parquet flush. 4M is ~96MB of int64 columns.
BUILD_CHUNK = 4_000_000


def triple(line: bytes) -> tuple[int, int, int] | None:
    """(subject, object, property) for an entity-to-entity truthy statement.

    `None` for everything else, which is most of the dump: labels,
    descriptions, aliases, sitelinks, and every statement whose value is a
    literal. A birth *date* is a fact about an article rather than an edge to
    another one, and this graph holds only the second kind - the corpus reads
    dates out of the infobox, where they already are.

    Subject and object first, property last, because that is the order a rel
    COPY wants: the two endpoints, and then the columns.
    """
    parts = line.split(b" ", 3)
    if len(parts) < 4:
        return None
    subj, prop, obj = parts[0], parts[1], parts[2]
    if not (subj.startswith(NT_ENTITY) and prop.startswith(NT_DIRECT)
            and obj.startswith(NT_ENTITY)):
        return None
    try:
        return (int(subj[len(NT_ENTITY):-1]), int(obj[len(NT_ENTITY):-1]),
                int(prop[len(NT_DIRECT):-1]))
    except ValueError:
        # `Q1234-deadbeef` and friends: a statement id rather than an entity.
        return None


def candidates(dump: Path) -> Iterator[bytes]:
    """Candidate lines from the dump, decompressed and coarsely filtered.

    Shelling out because both loops that matter here - inflating bzip2 and
    throwing away four lines in five - are C ones, and the pure-Python path
    below is the same work an order of magnitude slower. It is kept because a
    machine without `grep` should still be able to run this overnight rather
    than not at all.
    """
    tool = next(((name, flag) for name, flag in DECOMPRESSORS
                 if shutil.which(name)), None)
    if tool is None or not shutil.which("grep"):
        with bz2.open(dump, "rb") as fh:
            for line in fh:
                if PREFILTER.encode() in line:
                    yield line
        return

    unpack = subprocess.Popen([tool[0], tool[1], str(dump)],
                              stdout=subprocess.PIPE)
    assert unpack.stdout is not None
    sieve = subprocess.Popen(["grep", "-F", PREFILTER],
                             stdin=unpack.stdout, stdout=subprocess.PIPE,
                             env={**os.environ, "LC_ALL": "C"})
    # So that the decompressor is told when grep goes away, rather than filling
    # a pipe nobody is reading.
    unpack.stdout.close()
    assert sieve.stdout is not None
    drained = False
    try:
        yield from sieve.stdout
        drained = True
    finally:
        sieve.stdout.close()
        for proc in (sieve, unpack):
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
    # Only once the stream ended by itself: a caller that stopped early killed
    # these on purpose and their codes say so.
    #
    # A truncated dump is the failure this exists for. Decompressing one ends
    # the pipe cleanly and sets a code nobody was reading, so the build used to
    # finish, report a plausible edge count and write a graph missing however
    # much of Wikidata had not downloaded - which nothing downstream could
    # detect, because a smaller graph is exactly what a smaller corpus makes.
    if drained and unpack.returncode:
        raise RuntimeError(
            f"{tool[0]} exited {unpack.returncode} reading {dump}: the dump is "
            f"truncated or corrupt, and the statements read before it stopped "
            f"are a prefix rather than a graph")


def build(dump: Path, out: Path) -> tuple[int, int]:
    """Turn a truthy N-Triples dump into the graph `export` reads.

    Two parquet files and two COPYs rather than inserts: 766M edges one at a
    time across the binding is not a thing that finishes.

    The node list comes from a bitmap over the qids the edges mention rather
    than from a set of them. A Python set of 91.6M integers is several GB on a
    machine that has already been told to hold a parquet writer, and the ids
    are dense enough that one bit each is 16MB.

    This is the slow one - hours, most of it decompression - and it is what a
    re-export does *not* have to repeat. Only a newer Wikidata snapshot needs
    it run again.
    """
    import ladybug as lb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    edges_path = out.with_suffix(".edges.parquet")
    nodes_path = out.with_suffix(".nodes.parquet")
    schema = pa.schema([("subj", pa.int64()), ("obj", pa.int64()),
                        ("property", pa.int64())])
    subj: list[int] = []
    obj: list[int] = []
    prop: list[int] = []
    kept = high = 0
    started = time.time()

    with pq.ParquetWriter(edges_path, schema) as writer:
        def flush() -> None:
            nonlocal high
            if not subj:
                return
            high = max(high, max(subj), max(obj))
            writer.write_table(pa.table([subj, obj, prop], schema=schema))
            subj.clear()
            obj.clear()
            prop.clear()

        for line in candidates(dump):
            if (got := triple(line)) is None:
                continue
            subj.append(got[0])
            obj.append(got[1])
            prop.append(got[2])
            kept += 1
            if len(subj) >= BUILD_CHUNK:
                flush()
                rate = kept / (time.time() - started)
                print(f"  {kept:,} edges ({rate:,.0f}/s)", flush=True)
        flush()

    seen = np.zeros(high + 1, dtype=bool)
    for batch in pq.ParquetFile(edges_path).iter_batches(
            batch_size=BUILD_CHUNK):
        seen[batch.column("subj").to_numpy()] = True
        seen[batch.column("obj").to_numpy()] = True
    qids = np.flatnonzero(seen)
    pq.write_table(pa.table({"qid": qids}), nodes_path)
    print(f"  {len(qids):,} nodes", flush=True)

    con = lb.Connection(lb.Database(str(out)))
    con.execute(NODE_TABLE)
    con.execute(REL_TABLE)
    con.execute(f"COPY wikidata_node FROM '{nodes_path}'")
    con.execute(f"COPY wikidata_rel FROM '{edges_path}'")
    # Only once both COPYs are in: a failure above should leave the hours of
    # parsing on disk rather than make them be done again.
    edges_path.unlink()
    nodes_path.unlink()
    return len(qids), kept


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
    scratch = out.with_suffix(".raw.parquet")
    chain_scratch = out.with_suffix(".chain.parquet")

    # Unfiltered on purpose. The filter that used to live here - `r.property IN
    # [nine numbers]` - is what made adding a relation cost another pass over
    # 22GB, and the pass is not where the money goes: building the database is.
    # It writes every edge to the scratch file instead, and the membership test
    # below throws away the ones with no article at either end, which is most.
    #
    # Containment gets its own file, filtered, because the ancestry walk below
    # reads its input CHAIN_DEPTH times. Six passes over every edge in Wikidata
    # to find the two properties that mean "inside" is the one place where
    # dropping the filter would have cost real time rather than disk.
    started = time.time()
    con.execute(
        "COPY (MATCH (a:wikidata_node)-[r:wikidata_rel]->(b:wikidata_node) "
        "RETURN a.qid AS subj, r.property AS prop, b.qid AS obj) "
        f"TO '{scratch}'")
    contain = ",".join(str(p) for p in CONTAINMENT)
    con.execute(
        "COPY (MATCH (a:wikidata_node)-[r:wikidata_rel]->(b:wikidata_node) "
        f"WHERE r.property IN [{contain}] "
        "RETURN a.qid AS subj, r.property AS prop, b.qid AS obj) "
        f"TO '{chain_scratch}'")
    print(f"  scanned in {time.time() - started:.0f}s", flush=True)

    corpus = np.sort(np.fromiter(qids, dtype=np.int64))

    def among(values: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Vectorised `in`, for tables far too big for a Python loop."""
        i = np.searchsorted(table, values)
        i[i >= len(table)] = 0
        hit: np.ndarray = table[i] == values
        return hit

    def batches(path: Path = scratch) -> Iterator[pa.RecordBatch]:
        yield from pq.ParquetFile(path).iter_batches(batch_size=4_000_000)

    # The containment chain, found by walking up from the corpus one hop at a
    # time. Each round is a full pass of the containment file, which is cheap
    # and much simpler than holding 24M parent links in memory to walk them
    # properly. It is a full pass of *that* file rather than of every edge in
    # Wikidata, which is why the second COPY above exists.
    chain: list[tuple[int, int, int]] = []
    seen = corpus
    frontier = corpus
    for _ in range(CHAIN_DEPTH):
        found: list[np.ndarray] = []
        for batch in batches(chain_scratch):
            s = batch.column("subj").to_numpy()
            p = batch.column("prop").to_numpy()
            o = batch.column("obj").to_numpy()
            keep = among(s, frontier)
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

    # No `# property` lines any more. They recorded what `PROPERTY` happened to
    # be on the day the file was cut, which is exactly the coupling this change
    # removes - the map belongs to the importer, and a file that carries a stale
    # copy of it is a file someone will eventually believe.
    statements = classed = 0
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        fh.write(f"# format\t{FORMAT}\n# dump\t{dump.name}\n")
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
    chain_scratch.unlink()
    return statements, classed, len(set(chain))


def read_export(path: Path) -> tuple[list[tuple[int, int, int]],
                                     dict[int, set[int]], dict[str, str]]:
    """(statements, containment parents, header) from an export.

    Only the properties `PROPERTY` maps are kept. A format 2 file holds every
    property the dump had for these articles, which is the point of it - but
    materialising all of them costs memory for rows nothing downstream can read,
    and `build_plan` would meet a property it has no relation for. `--survey`
    counts the rest without building any of this.
    """
    wanted = set(PROPERTY) | {P_INSTANCE_OF}
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
                if prop in wanted:
                    rows.append((subj, prop, obj))
                if prop in CONTAINMENT:
                    # A statement about a corpus article is also a link in the
                    # chain; it is written once and read as both.
                    parents[subj].add(obj)
    return rows, parents, header


def survey(path: Path) -> tuple[dict[int, int], dict[str, str]]:
    """How many statements the export holds per property, and its header.

    Streamed rather than collected: the file this reads is the one that holds
    every property, and the whole reason to ask is that nobody knows yet which
    of them is worth a relation. Chain rows are scaffolding and are not counted.
    """
    counts: dict[int, int] = defaultdict(int)
    header: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                parts = line[1:].split("\t")
                if len(parts) >= 2:
                    header[parts[0].strip()] = parts[1].strip()
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 3 and fields[3] == "chain":
                continue
            counts[int(fields[1])] += 1
    return dict(counts), header


def report_survey(counts: dict[int, int], header: dict[str, str],
                  limit: int = 40) -> None:
    """Print the export's properties, unmapped first and biggest first.

    The unmapped list is the same thing `ingest.py --stats` prints for infobox
    fields it does not understand: the corpus saying what it knows that nothing
    yet reads. A property here costs a `PROPERTY` entry and no dump work.
    """
    if header.get("format") == "1":
        print("this export was cut against a fixed property list (format 1), "
              "so an absence here is not an absence of facts. Re-export to "
              "survey properly.")
    mapped = {p: n for p, n in counts.items() if p in PROPERTY}
    unmapped = {p: n for p, n in counts.items() if p not in PROPERTY}
    print(f"\n{sum(counts.values()):,} statements over {len(counts):,} "
          f"properties; {len(mapped)} mapped, {len(unmapped)} not\n")
    print(f"{'mapped':>12s}  {'statements':>12s}  relation")
    for prop, n in sorted(mapped.items(), key=lambda kv: -kv[1]):
        print(f"{'P' + str(prop):>12s}  {n:>12,}  {PROPERTY[prop]}")
    print(f"\n{'unmapped':>12s}  {'statements':>12s}  "
          f"https://www.wikidata.org/wiki/Property:Pn")
    for prop, n in sorted(unmapped.items(), key=lambda kv: -kv[1])[:limit]:
        print(f"{'P' + str(prop):>12s}  {n:>12,}")
    if len(unmapped) > limit:
        print(f"{'':>12s}  and {len(unmapped) - limit:,} more")


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
OUTCOMES = ("gap", "refine", "agree", "kept", "declined", "outranked",
            "untyped", "coarser", "typed")


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
    #
    # Everything else is a precedence rather than a union, in `PROPERTY` order:
    # a director and a producer are two answers, not two depths of one, and
    # unioning them only reaches `choose` to be declined.
    order = list(PROPERTY)
    merged: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for prop in sorted(facts, key=order.index):
        relation = PROPERTY[prop]
        for subject, values in facts[prop].items():
            if prop == P_COUNTRY and subject not in placed:
                plan.note(relation, "untyped")
                continue
            if prop in CONTAINMENT or subject not in merged[relation]:
                merged[relation][subject] |= values
            else:
                plan.note(relation, "outranked")

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
WIDTHS = (9, 8, 8, 8, 9, 10, 8, 8, 7)


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
    version = int(header.get("format", FORMAT))
    if not FORMAT_MIN <= version <= FORMAT:
        raise SystemExit(f"{path} is format {version}; this reads "
                         f"{FORMAT_MIN} to {FORMAT}")
    titles = sitelinks(db, source)
    facts, classes, placed = resolve(rows, titles)
    print(f"{len(rows):,} statements and {len(parents):,} containment links "
          f"from {header.get('dump', '?')}, against {len(titles):,} sitelinks")
    if version < FORMAT:
        # Worth saying out loud, because the symptom of importing an old export
        # after mapping a new property is that the new relation is simply
        # absent - which looks exactly like Wikidata not knowing anything.
        unseen = sorted(set(PROPERTY) - {p for _s, p, _o in rows})
        if unseen:
            print(f"  format {version} export: it was cut before "
                  f"{', '.join('P' + str(p) for p in unseen)} were mapped, so "
                  f"those are absent here rather than absent from Wikidata")
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
    parser.add_argument("--build", type=Path, metavar="LATEST-TRUTHY.NT.BZ2",
                        help=f"Build the graph --export reads, from a Wikidata "
                             f"truthy N-Triples dump ({TRUTHY}). Hours, and "
                             f"only needed for a newer Wikidata snapshot.")
    parser.add_argument("--export", type=Path, metavar="WIKIDATA.LBDB",
                        help="Cut a fresh export from a Wikidata graph dump")
    parser.add_argument("-o", "--out", type=Path,
                        help="Where --build or --export writes "
                             "(default wikidata.lbdb or wikidata.tsv.gz)")
    parser.add_argument("--score", type=Path, metavar="WIKIDATA.TSV.GZ",
                        help="Report what importing an export would change, "
                             "and write nothing")
    parser.add_argument("--write", type=Path, metavar="WIKIDATA.TSV.GZ",
                        help="Import an export into `derived`")
    parser.add_argument("--rebuild-graph", action="store_true",
                        help="Put the imported rows on the card. Without this "
                             "they sit in `derived` and nothing walks them.")
    parser.add_argument("--survey", type=Path, metavar="WIKIDATA.TSV.GZ",
                        help="Print what the export holds per property, "
                             "unmapped first, and write nothing")
    args = parser.parse_args(argv)

    # Both before the database, because neither reads the corpus: one turns a
    # Wikidata dump into a graph and the other says what an export holds. That
    # is what makes them runnable on a machine that has the file and not the
    # 500MB corpus.
    if args.build:
        out = args.out or Path("wikidata.lbdb")
        print(f"building {out} from {args.build.name}")
        nodes, edges = build(args.build, out)
        print(f"{nodes:,} nodes and {edges:,} edges -> {out}")
        return

    if args.survey:
        counts, header = survey(args.survey)
        report_survey(counts, header)
        return

    writing = bool(args.write)
    db = sqlite3.connect(args.db) if writing else sqlite3.connect(
        f"file:{args.db}?mode=ro", uri=True)
    if not _has_sitelinks(db, args.source):
        raise SystemExit(
            f"{args.db} has no sitelinks for '{args.source}'. Run:\n"
            "  python data/wikipedia/ingest.py --sitelinks "
            "<page.sql.gz> <page_props.sql.gz>")

    if args.export:
        out = args.out or Path("wikidata.tsv.gz")
        qids = set(sitelinks(db, args.source))
        print(f"exporting statements about {len(qids):,} articles "
              f"from {args.export.name}")
        kept, classed, chain = export(args.export, out, qids)
        print(f"{kept:,} statements, {classed:,} class rows and "
              f"{chain:,} containment links -> {out} "
              f"({out.stat().st_size / 1e6:.1f} MB)")
        return

    path = args.score or args.write
    if not path:
        parser.error("nothing to do: pass --export, --survey, --score "
                     "or --write")

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
