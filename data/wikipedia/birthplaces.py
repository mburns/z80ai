#!/usr/bin/env python3
"""
Read birthplaces out of lead text for the people whose infobox has none.

    python data/wikipedia/birthplaces.py            # score the extractors
    python data/wikipedia/birthplaces.py --write    # and fill in the gaps

## Why there is anything to read

`coverage.py` says 78,594 people are in this corpus and 42,288 have a
`born_in` edge. The other **36,191 have a lead and no birthplace**, and a great
many of those leads open with the sentence that would settle it - Wikipedia's
house style puts the birth in the first clause. The infobox is optional; the
opening sentence is very nearly not.

This is the one place in the pipeline that reads prose, so it is also the one
place that can be wrong in a way the rest cannot: an infobox field is a claim
somebody tabulated, and a sentence is a claim somebody wrote. Two things follow.

## Nothing lands in `fact`

Extracted rows go to `derived`, keyed by the method that produced them. A
reader wanting only what the encyclopedia tabulated reads `fact` and never sees
these; a reader wanting both asks for both and can still tell them apart. The
graph build ignores `derived` unless asked, so the card is unchanged by default.

## It is scored before it is trusted

The 42,288 people who *do* have a birthplace also have leads, and their
infoboxes say the answer. So an extractor can be run over them and marked:

    yield       of the people it was asked about, how many it answered
    agreement   of those, how many matched the infobox
    resolved    how many named an article the graph can actually walk to

That is ground truth for free, and it is why the scoring runs first and always,
`--write` being the extra step rather than the default. An extractor that
cannot beat the regex on people whose answers we already know has no business
guessing about the ones we do not.

## Agreement is the wrong number, and it is the obvious one

Scored on exact agreement the regex gets **13.8%**, which reads like a failure.
It is not: the infobox and the lead routinely name *different places that are
both true*. College Park against Georgia, Ontario against Canada, Brooklyn
against New York City, Whitby against Toronto. Marking those wrong measures
granularity, not correctness.

What the oracle is asked is which *country*, so the number that matters is
whether the two climb to the same one. That is **93.4%** - 4,944 right against
352 wrong - and several of the 352 are the climb's own typing rather than the
extraction: `Ontario` climbs to Ontario and `California` to California, both
being called countries by enough infoboxes to clear `TYPE_FLOOR`. So 93.4% is
itself a floor.

## The two populations are not the same, in the useful direction

The eval set is people who *have* an infobox birthplace, and the target set is
people who do not. The regex resolves 18.3% of the first and **37.5% of the
second**, because a person with no birthplace in the infobox frequently has no
infobox at all - and Wikipedia's opening sentence states the birth regardless.

So the yield does not transfer and neither, strictly, does the precision. The
figure to quote is the one `--write` measures on the population it actually
ran on; the eval set is what says the extractor is worth running.

    regex, measured    13,585 rows over 36,191 people    37.5%
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ingest
import libgraph

#: Wikipedia's house style for a biography's first sentence: the name, the
#: dates in parentheses, then "was a ... born in X". `[^.]{0,60}` keeps the
#: match inside one sentence, which is what stops "born in 1931. He moved to
#: Paris" from reading Paris as a birthplace.
BORN_IN = re.compile(
    r"\bborn\b[^.]{0,60}?\bin\s+(?P<place>[A-Z][\w.'\u2019-]*"
    r"(?:[ ,]+[A-Z][\w.'\u2019-]*){0,3})")

#: "(born 12 May 1931 in Vienna)" - the same fact with the preposition first.
BORN_PAREN = re.compile(
    r"\(born[^)]{0,40}?\bin\s+(?P<place>[A-Z][\w.'\u2019-]*"
    r"(?:[ ,]+[A-Z][\w.'\u2019-]*){0,3})\)")

#: Trailing words that are not part of a place name but survive the pattern,
#: because a lead runs on: "born in Ottawa, Canada. She began ...".
TAIL = re.compile(r"\s+(?:He|She|They|It|His|Her|Their|The|A|An|And|But|Who|"
                  r"Which|In|On|At|To|From|Was|Is|Were|Are)\b.*$")


@dataclass
class Extraction:
    """One extractor's answer for one person, before it is resolved."""

    subject: str
    place: str | None
    #: Set once the place has been matched to an article the graph knows.
    target: str | None = None


@dataclass
class Score:
    """How an extractor did over people whose birthplace is already known."""

    method: str
    asked: int = 0
    answered: int = 0
    resolved: int = 0
    agreed: int = 0
    #: Resolved somewhere that climbs to the same country as the infobox did,
    #: including the cases where the two named different places.
    same_country: int = 0
    #: Resolved somewhere that climbs to a *different* country. The only
    #: outcome here that is unambiguously wrong.
    wrong_country: int = 0
    #: (subject, extracted, expected) for the first few of those.
    misses: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def yield_(self) -> float:
        return self.answered / self.asked if self.asked else 0.0

    @property
    def agreement(self) -> float:
        return self.agreed / self.resolved if self.resolved else 0.0

    @property
    def usable(self) -> float:
        """Answers that both resolve and agree, over everyone asked.

        The number that matters: an extractor answering everything and
        resolving nothing has added no edges, and one resolving everything to
        the wrong place has added worse than none.
        """
        return self.agreed / self.asked if self.asked else 0.0

    @property
    def country_precision(self) -> float:
        """Of the answers that climb to a country, how many climb to the right
        one.

        `agreement` badly understates this extractor and would understate any
        other, because the infobox and the lead routinely name *different
        places that are both true*: College Park against Georgia, Ontario
        against Canada, Brooklyn against New York City. Marking those wrong
        measures granularity, not correctness.

        What the oracle is asked is "what country was X born in", so what
        matters is whether the climb lands in the same place. This is that
        number, and it is the one to compare extractors on.
        """
        total = self.same_country + self.wrong_country
        return self.same_country / total if total else 0.0


def clean_place(place: str) -> str:
    """Trim what the pattern over-matched off the end of a place name."""
    place = TAIL.sub("", place)
    return place.strip(" ,.;:").strip()


def by_regex(subject: str, lead: str) -> Extraction:
    """The trivial baseline, and the bar anything cleverer has to clear."""
    m = BORN_PAREN.search(lead) or BORN_IN.search(lead)
    if not m:
        return Extraction(subject, None)
    place = clean_place(m.group("place"))
    return Extraction(subject, place or None)


#: name -> the extractor. A model backend registers itself here; see `--method`.
EXTRACTORS: dict[str, Callable[[str, str], Extraction]] = {"regex": by_regex}


def resolver(db: sqlite3.Connection, source: str) -> libgraph.Resolver:
    """The same resolver the graph build uses, so a place lands the same way.

    Worth using rather than a `titles` lookup of one's own: it reads redirects,
    and it already walks "Edinburgh, Scotland" segment by segment keeping the
    most specific article the corpus actually has. A first attempt here did its
    own comma splitting, tail-first, and turned `New York City, New` into
    `New` - a real article, about the word.
    """
    titles = {t for (t,) in db.execute(
        "SELECT title FROM article WHERE source = ?", (source,))}
    redirects = dict(db.execute(
        "SELECT title, target FROM redirect WHERE source = ?", (source,)))
    return libgraph.Resolver(titles, redirects)


def people_missing_birthplace(db: sqlite3.Connection,
                              source: str) -> list[tuple[str, str]]:
    """(title, lead) for every person the graph has no birthplace for."""
    persons = libgraph.people(db, source)
    known = {s for (s,) in db.execute(
        "SELECT DISTINCT subject FROM edge WHERE source = ? AND relation = ?",
        (source, "born_in"))}
    want = persons - known
    rows = db.execute("SELECT title, lead FROM article WHERE source = ? "
                      "AND lead != '' ORDER BY title", (source,))
    return [(t, lead) for t, lead in rows if t in want]


def people_with_birthplace(db: sqlite3.Connection, source: str,
                           limit: int | None = None) -> list[tuple[str, str, str]]:
    """(title, lead, birthplace) for people the infobox already answers.

    The evaluation set, and the reason this file can report a number rather
    than an impression.
    """
    rows = db.execute(
        "SELECT a.title, a.lead, e.object FROM edge e "
        "JOIN article a ON a.source = e.source AND a.title = e.subject "
        "WHERE e.source = ? AND e.relation = 'born_in' AND a.lead != '' "
        "ORDER BY a.title", (source,))
    out = [(t, lead, place) for t, lead, place in rows]
    return out[:limit] if limit else out


def evaluate(db: sqlite3.Connection, source: str, method: str,
             limit: int | None = None) -> Score:
    """Run an extractor over people whose birthplace is already known."""
    extract = EXTRACTORS[method]
    resolve = resolver(db, source)
    score = Score(method)
    countries: dict[str, str | None] = {}

    def country_of(place: str) -> str | None:
        """Where a climb from ``place`` lands, memoised - the eval set names
        the same few thousand places over and over."""
        if place not in countries:
            countries[place] = libgraph.follow(
                db, source, place, ["in_country"]).value
        return countries[place]

    for subject, lead, expected in people_with_birthplace(db, source, limit):
        score.asked += 1
        found = extract(subject, lead)
        if found.place is None:
            continue
        score.answered += 1
        target = resolve(found.place)
        if target is None:
            continue
        score.resolved += 1
        if target == expected:
            score.agreed += 1

        # Every resolved answer is climbed, not only the disagreements: this is
        # the extractor's precision at the question the oracle is actually
        # asked, and an exact match is a right answer to it too. What the climb
        # adds is the cases that merely named a different granularity of the
        # same truth - College Park for Georgia - which agreement calls wrong.
        got, want = country_of(target), country_of(expected)
        if got is None or want is None:
            continue
        if got == want:
            score.same_country += 1
        else:
            score.wrong_country += 1
            if len(score.misses) < 12:
                score.misses.append((subject, f"{target} ({got})",
                                     f"{expected} ({want})"))
    return score


def extract_all(db: sqlite3.Connection, source: str, method: str,
                subjects: Iterable[tuple[str, str]]) -> list[Extraction]:
    """Every answer an extractor has for people with no birthplace."""
    extract = EXTRACTORS[method]
    resolve = resolver(db, source)
    out = []
    for subject, lead in subjects:
        found = extract(subject, lead)
        if found.place is None:
            continue
        found.target = resolve(found.place)
        if found.target is not None:
            out.append(found)
    return out


def write(db: sqlite3.Connection, source: str, method: str,
          found: list[Extraction]) -> int:
    """Replace this method's rows. Never touches `fact` or `edge`."""
    db.execute("DELETE FROM derived WHERE source = ? AND method = ? "
               "AND relation = 'born_in'", (source, method))
    db.executemany(
        "INSERT OR REPLACE INTO derived VALUES (?, ?, 'born_in', ?, ?)",
        [(source, e.subject, e.target, method) for e in found if e.target])
    db.commit()
    return len(found)


def report(score: Score, missing: int) -> None:
    print(f"\n  {score.method} over {score.asked:,} people whose birthplace "
          f"the infobox already gives")
    print(f"    answered   {score.answered:>7,}  {score.yield_ * 100:5.1f}%  "
          "the lead said something")
    print(f"    resolved   {score.resolved:>7,}  "
          f"{score.resolved / score.asked * 100 if score.asked else 0:5.1f}%  "
          "and it named an article")
    print(f"    agreed     {score.agreed:>7,}  {score.usable * 100:5.1f}%  "
          "and it matched the infobox exactly")
    print(f"\n    agreement where it resolved:  {score.agreement * 100:5.1f}%")
    print(f"    same country when both climb:  {score.country_precision * 100:5.1f}%"
          f"   ({score.same_country:,} right, {score.wrong_country:,} wrong)")
    if score.misses:
        print("\n    where it climbed to a different country")
        for subject, got, want in score.misses[:8]:
            print(f"      {subject:<30} {got} not {want}")
    print(f"\n  {missing:,} people have a lead and no birthplace. What that is "
          "worth is what\n  `--write` reports, not an extrapolation from the "
          "line above: the two\n  populations differ, and in the useful "
          "direction. A person whose infobox\n  gives a birthplace tends to "
          "have an infobox; a person without one often has\n  no infobox at "
          "all, and Wikipedia's opening sentence says it anyway.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--db", type=Path, default=ingest.DB_PATH)
    ap.add_argument("--source", default="simplewiki")
    ap.add_argument("--method", default="regex", choices=sorted(EXTRACTORS))
    ap.add_argument("--limit", type=int, default=None,
                    help="Score against this many known people, not all of them")
    ap.add_argument("--write", action="store_true",
                    help="Also fill in the gaps, into `derived`")
    ap.add_argument("--rebuild-graph", action="store_true",
                    help="Rebuild `edge` with those rows admitted, so a card "
                         "built afterwards carries them. This is the step that "
                         "puts something read out of prose on the device")
    args = ap.parse_args(argv)

    db = ingest.connect(args.db)
    score = evaluate(db, args.source, args.method, args.limit)
    missing = people_missing_birthplace(db, args.source)
    report(score, len(missing))

    if args.write:
        found = extract_all(db, args.source, args.method, missing)
        n = write(db, args.source, args.method, found)
        rate = n / len(missing) * 100 if missing else 0.0
        print(f"\n  wrote {n:,} rows to `derived` as method {args.method!r}, "
              "which nothing reads unless asked")
        print(f"  {rate:.1f}% of the people with no birthplace, against "
              f"{score.resolved / score.asked * 100 if score.asked else 0:.1f}% "
              "of the people who had one")

    if args.rebuild_graph:
        edges, _dropped = libgraph.build(db, args.source, report=print,
                                         derived=args.method)
        db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                   (f"{args.source}.edges", str(edges)))
        db.commit()
        print(f"\n  `edge` rebuilt with {args.method!r} admitted: {edges:,} "
              "edges. A card built now carries them; rebuild without this to\n"
              "  go back to what the encyclopedia tabulated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
