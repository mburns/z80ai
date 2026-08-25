#!/usr/bin/env python3
"""
What share of the corpus the oracle can actually walk, and where it stops.

    python data/wikipedia/coverage.py
    python data/wikipedia/coverage.py --json > before.json
    python data/wikipedia/coverage.py --baseline before.json

Every coverage claim this project makes is a number in here. They were
previously measured by hand, quoted in a docstring, and never re-run - so
`libgraph.py` says `birth_place -> country` completes for 40.7% of subjects and
nothing checks whether that is still true after an ingest change. This prints
the same numbers from the database in front of it.

## Why the denominator is subjects, not articles

A chain question can only be asked about a subject the chain can start from:
"what country was X born in" is not a question about a river. Scoring over all
283,997 articles would mix "the graph could not finish" with "the question does
not apply", and the first is the one worth fixing. So each path reports:

    startable   subjects holding the path's first relation
    complete    of those, the walks that reached a value
    of corpus   startable/articles, so the conditional share stays honest

`libgraph.py` quotes the middle column, which is why it is the one to compare.

**A falling rate is not always a loss, so read the counts beside it.** Anything
that makes more subjects startable adds the ones that were failing for a reason
- a value naming no article, a field nobody mapped - and they complete at less
than the existing average by construction. Giving `libgraph.build` a fallback
through the ranks took `in_country` from 25,346 completed answers to 26,127 and
its *rate* from 61.7% to 60.1% at the same time. The machine answered 781 more
questions and looks worse. `startable`, `complete` and the delta on each are
printed for that reason; a scoreboard of rates alone would have called that
change a regression.

## Why it walks rather than joins

The obvious implementation is one SQL join per path, and it would be faster and
wrong: it would measure a graph traversal this repo does not ship. `follow()`
is the code the oracle runs, `CLIMB` is a loop with a type test and a hop limit
that no join expresses, and a harness that agrees with the machine only when
both are correct is not a harness. So this calls `libgraph.follow` per subject
and pays for it. `--sample` bounds the cost when that matters; the default is
every subject, because these are shares and a sample reports them with an error
bar this file does not print.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "questions"))

import relations

import libgraph


def paths_to_score() -> list[tuple[str, list[str]]]:
    """(label, relations) for every path this reports, chains before singles.

    The four chains come from `relations.CHAINS` rather than a list here, so
    the harness cannot drift from the class set the classifier is trained on.
    The single relations are added because a one-hop path that quietly loses
    subjects is invisible in a two-hop number.
    """
    chains = [(path, path.split()) for path in relations.CHAINS]
    singles = [(name, [name]) for name in sorted(libgraph.CANONICAL)]
    return chains + singles


def relation_objects(db: sqlite3.Connection, source: str,
                     relation: str) -> list[str]:
    """Where ``relation`` points - the birthplaces, for `born_in`.

    A climb is justified by how far *these* are from a country, not by how far
    its own subjects are. `libgraph.py` measures "the 42,033 people this corpus
    records a birthplace for", and a birthplace that is already a country
    usually holds no `located_in` edge of its own, so measuring from the
    climb's own subjects excludes exactly the cases the climb exists to catch
    and reports the zero-hop share as nil.

    One row per edge rather than per distinct place, for the same reason: the
    quoted figure is per person, and thousands of them were born in London.
    Deduplicating would weight a village the same as a capital and understate
    every share, since the popular birthplaces are the well-connected ones.
    """
    return [o for (o,) in db.execute(
        "SELECT object FROM edge WHERE source = ? AND relation = ?",
        (source, relation))]


def head_subjects(db: sqlite3.Connection, source: str,
                  relation: str) -> list[str]:
    """Subjects a path starting with ``relation`` could be asked about.

    A climb starts from the relation it steps, not from itself: no edge is ever
    labelled `in_country`, so asking for one would report every path through it
    as unstartable.
    """
    if relation in libgraph.CLIMB:
        relation = libgraph.CLIMB[relation][0]
    return [s for (s,) in db.execute(
        "SELECT DISTINCT subject FROM edge WHERE source = ? AND relation = ? "
        "ORDER BY subject", (source, relation))]


def score_path(db: sqlite3.Connection, source: str, steps: list[str],
               subjects: list[str]) -> dict[str, Any]:
    """Walk ``steps`` from every subject and tally where the walks stopped.

    `stopped_at` is the point of the whole file: a path that fails is failing
    at one particular hop, and which one decides what would fix it. A chain
    losing everything at its second hop wants coverage; one losing subjects at
    its first wants a property mapped.
    """
    complete = 0
    stopped_at: dict[str, int] = {}
    for subject in subjects:
        answer = libgraph.follow(db, source, subject, steps)
        if answer.complete:
            complete += 1
        else:
            key = answer.missing or "?"
            stopped_at[key] = stopped_at.get(key, 0) + 1
    return {
        "startable": len(subjects),
        "complete": complete,
        "rate": complete / len(subjects) if subjects else 0.0,
        "stopped_at": dict(sorted(stopped_at.items(),
                                  key=lambda kv: -kv[1])),
    }


def climb_distances(db: sqlite3.Connection, source: str, climb: str,
                    subjects: list[str]) -> dict[str, int]:
    """How many hops each climb needed, or that it never arrived.

    `libgraph.py` justifies `CLIMB` with this distribution - 0 hops for 26.2%
    of birthplaces, 1 for 35.7%, 2 for 1.6% - and it is the evidence for
    climbing rather than walking a fixed two hops. `Answer.path` starts with
    the subject and gains one entry per hop taken, including the hops a failed
    climb took before it ran out, so the count is `len(path) - 1`.
    """
    hist: dict[str, int] = {}
    for subject in subjects:
        answer = libgraph.follow(db, source, subject, [climb])
        key = str(len(answer.path) - 1) if answer.complete else "never"
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(),
                       key=lambda kv: (kv[0] == "never", kv[0])))


def type_floors(db: sqlite3.Connection, source: str,
                highest: int = 6) -> dict[str, int]:
    """Entities named a country by N independent infoboxes, for each N.

    `TYPE_FLOOR` is set to 3 on the strength of two hand-counted numbers (182
    at 3, 353 at 1). The floor decides where every `in_country` climb stops, so
    it is worth seeing the whole curve rather than the two ends of it.
    """
    field = libgraph.TYPE_FIELD["country"]
    titles = {t for (t,) in db.execute(
        "SELECT title FROM article WHERE source = ?", (source,))}
    redirects = dict(db.execute(
        "SELECT title, target FROM redirect WHERE source = ?", (source,)))
    resolve = libgraph.Resolver(titles, redirects)

    counts: dict[str, int] = {}
    for (value,) in db.execute(
            "SELECT value FROM fact WHERE source = ? AND property = ?",
            (source, field)):
        target = resolve(value)
        if target:
            counts[target] = counts.get(target, 0) + 1
    return {str(n): sum(1 for c in counts.values() if c >= n)
            for n in range(1, highest)}


def reach(db: sqlite3.Connection, source: str) -> dict[str, Any]:
    """The headline numbers: how much of the corpus is on the graph at all."""
    def one(sql: str) -> int:
        return int(db.execute(sql, (source,)).fetchone()[0])

    articles = one("SELECT COUNT(*) FROM article WHERE source = ?")
    facts = one("SELECT COUNT(*) FROM fact WHERE source = ?")
    subjects = one("SELECT COUNT(DISTINCT subject) FROM fact WHERE source = ?")
    edges = one("SELECT COUNT(*) FROM edge WHERE source = ?")
    on_graph = one("SELECT COUNT(DISTINCT subject) FROM edge WHERE source = ?")
    props = one("SELECT COUNT(*) FROM property WHERE source = ?")
    mapped = one("SELECT COUNT(*) FROM property "
                 "WHERE source = ? AND relation IS NOT NULL")
    # Categories arrived in schema 7; a database written before that has no
    # such table, and reporting zero for it is more useful than refusing to
    # measure anything else.
    try:
        filed = one("SELECT COUNT(DISTINCT title) FROM category "
                    "WHERE source = ?")
    except sqlite3.OperationalError:
        filed = 0
    return {
        "articles": articles,
        "facts": facts,
        "fact_subjects": subjects,
        "fact_subject_share": subjects / articles if articles else 0.0,
        "edges": edges,
        "graph_subjects": on_graph,
        "graph_subject_share": on_graph / articles if articles else 0.0,
        "facts_per_edge": facts / edges if edges else 0.0,
        "properties": props,
        "properties_mapped": mapped,
        "categorised": filed,
        "categorised_share": filed / articles if articles else 0.0,
    }


def candidates(db: sqlite3.Connection, source: str, limit: int = 15,
               sample: int = 200, floor: int = 500) -> list[dict[str, Any]]:
    """Unmapped properties whose values actually name articles.

    "9,462 properties map to nothing" reads as the size of the opportunity and
    is not: an edge points at an article, and the biggest unmapped properties
    are `name` (the subject's own), `birth_date` (a date), `subdivision_type`
    ("Country", "State") and a footballer's `years`/`clubs`/`goals`. None of
    them can be an edge however they are read.

    So this reports the share of a property's values that resolve to a title
    the corpus holds, which is necessary and *not* sufficient. It is a filter,
    not a ranking, and the difference matters: it removes what could never be
    an edge, and it cannot tell a relation from a label, because a label is
    spelled with words and words have articles. `subdivision_type` resolves at
    90% on values like "Country" and "State"; `unit_pref` at 96% on "metric".
    Both are as unmappable as `birth_date`, for a reason no counting will
    reach.

    Read it as a shortlist to judge, then - it takes 9,462 names down to a few
    hundred worth reading - and not as an answer. Deciding that `nationality`
    is a relation and `postal_code_type` is a label is the judgement
    `libgraph.py` says belongs to a person looking at this corpus, and this
    only spares them the properties that were never in the running.
    """
    titles = {t for (t,) in db.execute(
        "SELECT title FROM article WHERE source = ?", (source,))}
    redirects = dict(db.execute(
        "SELECT title, target FROM redirect WHERE source = ?", (source,)))
    resolve = libgraph.Resolver(titles, redirects)

    wanted = dict(db.execute(
        "SELECT name, uses FROM property WHERE source = ? "
        "AND relation IS NULL AND uses >= ?", (source, floor)))
    if not wanted:
        return []

    # One pass over `fact`, not one per property. The primary key leads with
    # `subject`, so there is no index a per-property filter can use and each
    # one costs a full scan of two million rows.
    #
    # Only text, and never the subject's own title. Both exclusions were put
    # here by a measurement that was wrong without them: `goals` scored 99.5%
    # and `nationalcaps` 100%, because this corpus holds articles called "13"
    # and "100" and a number resolves as happily as a city does. `name` scored
    # 74% by naming the article it is on.
    seen: dict[str, list[str]] = {}
    for prop, value, subject in db.execute(
            "SELECT property, value, subject FROM fact "
            "WHERE source = ? AND kind = 'text'", (source,)):
        if prop not in wanted or value == subject:
            continue
        bucket = seen.setdefault(prop, [])
        if len(bucket) < sample:
            bucket.append(value)

    out: list[dict[str, Any]] = []
    for name, values in seen.items():
        hits = sum(1 for v in values if resolve(v) is not None)
        share = hits / len(values)
        out.append({"name": name, "uses": wanted[name], "resolves": share,
                    "reach": int(wanted[name] * share)})
    out.sort(key=lambda r: -int(r["reach"]))
    return out[:limit]


def unmapped(db: sqlite3.Connection, source: str,
             limit: int = 15) -> list[dict[str, Any]]:
    """What the corpus uses often and nothing understands.

    The partial index `property_unmapped` exists for this query. It is the
    to-do list for `libgraph.CANONICAL`: the last entry it surfaced was
    `subdivision_name`, and mapping that took chaining from 1.7% to 40.7%.
    """
    return [{"name": name, "uses": uses, "subjects": subs}
            for name, uses, subs in db.execute(
                "SELECT name, uses, subjects FROM property "
                "WHERE source = ? AND relation IS NULL "
                "ORDER BY uses DESC LIMIT ?", (source, limit))]


def measure(db: sqlite3.Connection, source: str,
            sample: int, seed: int) -> dict[str, Any]:
    """Every section, as one JSON-shaped dict."""
    rng = random.Random(seed)
    started = time.monotonic()

    result: dict[str, Any] = {
        "source": source,
        "sample": sample,
        "reach": reach(db, source),
        "paths": {},
        "climb": {},
        "type_floors": type_floors(db, source),
        "unmapped": unmapped(db, source),
        "candidates": candidates(db, source),
    }

    for label, steps in paths_to_score():
        subjects = head_subjects(db, source, steps[0])
        if sample and len(subjects) > sample:
            subjects = sorted(rng.sample(subjects, sample))
        scored = score_path(db, source, steps, subjects)
        scored["of_corpus"] = (scored["startable"] / result["reach"]["articles"]
                               if result["reach"]["articles"] else 0.0)
        result["paths"][label] = scored

    # Two populations per climb, because they answer different questions. The
    # climb's own subjects say how far a place that records its container is
    # from a country; the objects of a relation feeding it say how far a
    # *birthplace* is, which is the number `libgraph.py` quotes to justify
    # climbing at all.
    for climb in libgraph.CLIMB:
        populations = {climb: head_subjects(db, source, climb)}
        for path in relations.CHAINS:
            steps = path.split()
            for before, after in itertools.pairwise(steps):
                if after == climb and before not in libgraph.CLIMB:
                    populations[f"{climb} from {before}"] = relation_objects(
                        db, source, before)
        for label, subjects in populations.items():
            if sample and len(subjects) > sample:
                subjects = sorted(rng.sample(subjects, sample))
            result["climb"][label] = climb_distances(
                db, source, climb, subjects)

    result["seconds"] = round(time.monotonic() - started, 1)
    return result


# --- reporting ----------------------------------------------------------------


def pct(x: float) -> str:
    return f"{x:.1%}"


def delta(now: float, was: float | None, as_pct: bool = True) -> str:
    """A signed change, or blank when there is nothing to compare against."""
    if was is None:
        return ""
    diff = now - was
    if abs(diff) < 1e-12:
        return "  ="
    if as_pct:
        return f"  {diff:+.1f}pt" if abs(diff) >= 0.0005 else "  ~0"
    return f"  {diff:+,.0f}"


def report(now: dict[str, Any], was: dict[str, Any] | None = None) -> None:
    """Print the whole scoreboard, with a delta column when given a baseline."""
    old_reach = was["reach"] if was else None
    r = now["reach"]

    def share_delta(key: str) -> str:
        """The change in a share, in points, or blank without a baseline."""
        if old_reach is None:
            return ""
        return delta(r[key] * 100, old_reach.get(key, 0.0) * 100)

    print(f"\n{now['source']}: {r['articles']:,} articles, "
          f"{r['facts']:,} facts, {r['edges']:,} edges "
          f"({r['facts_per_edge']:.1f} facts per edge)")
    print(f"  {r['fact_subjects']:,} subjects have a fact "
          f"({pct(r['fact_subject_share'])}){share_delta('fact_subject_share')}")
    print(f"  {r['graph_subjects']:,} subjects are on the graph "
          f"({pct(r['graph_subject_share'])}){share_delta('graph_subject_share')}")
    print(f"  {r['categorised']:,} articles carry a category "
          f"({pct(r['categorised_share'])}){share_delta('categorised_share')}")
    print(f"  {r['properties_mapped']:,} of {r['properties']:,} properties "
          f"map to a relation")

    # Both deltas, because a one-hop path is complete by construction: its rate
    # is 100% whatever happens, and everything that moved moved in the count.
    # Showing only the rate reports a change of nothing for exactly the rows a
    # newly-mapped property or a recovered value lands on.
    print(f"\n  {'path':<22}{'startable':>10}{'+/-':>8}"
          f"{'answered':>10}{'+/-':>8}{'rate':>7}{'+/-':>9}")
    for label, p in now["paths"].items():
        b = was["paths"].get(label) if was else None
        print(f"  {label:<22}{p['startable']:>10,}"
              f"{delta(p['startable'], b['startable'] if b else None, as_pct=False):>8}"
              f"{p['complete']:>10,}"
              f"{delta(p['complete'], b['complete'] if b else None, as_pct=False):>8}"
              f"{pct(p['rate']):>7}"
              f"{delta(p['rate'] * 100, b['rate'] * 100 if b else None):>9}")

    print("\n  where incomplete walks stopped")
    for label, p in now["paths"].items():
        if p["stopped_at"] and p["rate"] < 1.0:
            worst = ", ".join(f"{k} ({v:,})"
                              for k, v in list(p["stopped_at"].items())[:3])
            print(f"    {label:<24}{worst}")

    for climb, hist in now["climb"].items():
        total = sum(hist.values()) or 1
        parts = "  ".join(f"{k}:{pct(v / total)}" for k, v in hist.items())
        print(f"\n  {climb} distance   {parts}")

    floors = now["type_floors"]
    print("\n  countries at each TYPE_FLOOR   "
          + "  ".join(f"{n}:{c:,}" for n, c in floors.items())
          + f"   (using {libgraph.TYPE_FLOOR})")

    print("\n  biggest unmapped properties")
    for row in now["unmapped"]:
        print(f"    {row['name']:<28}{row['uses']:>9,} uses"
              f"{row['subjects']:>9,} subjects")

    print("\n  unmapped, ranked by values that name an article")
    print(f"    {'property':<28}{'uses':>9}{'resolves':>10}{'reach':>9}")
    for row in now.get("candidates", []):
        print(f"    {row['name']:<28}{row['uses']:>9,}"
              f"{pct(row['resolves']):>10}{row['reach']:>9,}")
    print(f"\n  measured in {now['seconds']}s"
          + (f", sampled to {now['sample']:,} subjects per path"
             if now["sample"] else "")
          + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--db", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "simple_english_wikipedia.db")
    ap.add_argument("--source", default="simplewiki")
    ap.add_argument("--sample", type=int, default=0,
                    help="subjects per path, 0 for every one (the default)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --sample, so a sampled run is repeatable")
    ap.add_argument("--json", action="store_true",
                    help="emit the measurement instead of the table")
    ap.add_argument("--baseline", type=Path,
                    help="a --json file to show this run's change against")
    args = ap.parse_args()

    if not args.db.exists():
        sys.exit(f"{args.db} does not exist - run ingest.py first")

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        now = measure(db, args.source, args.sample, args.seed)
    finally:
        db.close()

    if args.json:
        json.dump(now, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    was = json.loads(args.baseline.read_text()) if args.baseline else None
    report(now, was)


if __name__ == "__main__":
    main()
