#!/usr/bin/env python3
"""
What the ingest throws away when an infobox value is written as a template.

    python data/wikipedia/templates.py simplewiki-20260801-pages-articles.xml.bz2
    python data/wikipedia/templates.py <dump> --limit 20000

`clean()` strips every `{{...}}`, because a lead has to survive an infobox that
ran past the window we captured. Applied to a *value* the same rule is a
deletion: `{{birth date|1847|3|3}}` cleans to the empty string, `infobox_fields`
drops empty values, and the fact is gone. `tests/test_wikipedia_ingest.py`
records that as intended - "`{{start date|1602}}` cleans to nothing useful, so
it is not a fact" - which is true of the *cleaner* and not of the value.

The tell is in the corpus stats: dates are 4% of values in an encyclopedia
largely made of people. This counts what is behind that number so the fix can
be aimed rather than guessed - which templates, how often, and against which
properties, since a template lost from `birth_date` costs an edge and one lost
from `image_caption` costs nothing.

Reads the dump rather than the database, necessarily: by the time a row exists
the dropped values are already gone.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest

#: The template name at the head of a `{{...}}`, up to its first `|` or `}`.
#: Case and underscores vary by author, so they are folded the way
#: `normalize_property` folds a key.
TEMPLATE = re.compile(r"\{\{\s*([^|{}\n]+)")


def template_names(value: str) -> list[str]:
    """Every template named in ``value``, nested ones included, folded."""
    return [m.group(1).strip().lower().replace("_", " ")
            for m in TEMPLATE.finditer(value)]


def survives(value: str) -> bool:
    """Whether this raw value would reach the `fact` table.

    The same three tests `infobox_fields` applies, in the same order, so a
    value counted as lost here is one the ingest actually loses.
    """
    cleaned = ingest.normalize_value(ingest.clean(value))
    return bool(cleaned) and not ingest.JUNK_VALUE.match(cleaned) \
        and len(cleaned) <= ingest.MAX_VALUE_LEN


@dataclass
class Tally:
    """What a scan found. Split by outcome, because the loss *rate* is the
    interesting column: a template that always survives needs no expansion,
    and one that never does is a rule the ingest is missing.

    **The rate is a lower bound on the damage.** `_strip_braced` takes a
    template's contents away with its braces, so `{{nowrap|Paris}}` standing
    alone loses the value while `Paris, {{nowrap|France}}` keeps a value that
    has quietly lost "France". Only the first is counted lost. Reading a
    template's real cost therefore means reading its loss rate as the floor,
    not the figure - which is enough to rank them, and ranking is what the
    expansion table needs.
    """

    #: template -> values it appeared in that the ingest dropped / kept.
    lost: Counter[str] = field(default_factory=Counter)
    kept: Counter[str] = field(default_factory=Counter)
    #: (property, template) -> dropped values, so the fix can be aimed.
    lost_property: Counter[tuple[str, str]] = field(default_factory=Counter)
    #: Headline counts: infoboxes, fields, templated, templated_lost.
    totals: Counter[str] = field(default_factory=Counter)


def scan(path: Path, limit: int = 0) -> Tally:
    """Count template-bearing infobox values, split by whether they survived."""
    tallies = Tally()

    for n, (_title, redirect, markup) in enumerate(ingest.raw_pages(path)):
        if limit and n >= limit:
            break
        if redirect:
            continue
        body = ingest.infobox_body(markup)
        if body is None:
            continue
        tallies.totals["infoboxes"] += 1

        for raw in ingest.split_fields(body)[1:]:
            key, sep, value = raw.partition("=")
            if not sep:
                continue
            prop = ingest.normalize_property(key)
            if prop is None:
                continue
            tallies.totals["fields"] += 1

            names = set(template_names(value))
            if not names:
                continue
            tallies.totals["templated"] += 1

            if survives(value):
                tallies.kept.update(names)
                continue
            tallies.totals["templated_lost"] += 1
            tallies.lost.update(names)
            for name in names:
                tallies.lost_property[(prop, name)] += 1

    return tallies


def report(t: Tally, top: int) -> None:
    totals = t.totals
    fields = totals["fields"] or 1
    templated = totals["templated"] or 1
    print(f"\n{totals['infoboxes']:,} infoboxes, {totals['fields']:,} named fields")
    print(f"  {totals['templated']:,} values use a template "
          f"({totals['templated'] / fields:.1%} of fields)")
    print(f"  {totals['templated_lost']:,} of those are dropped "
          f"({totals['templated_lost'] / templated:.1%} of templated values, "
          f"{totals['templated_lost'] / fields:.1%} of all fields)")

    print(f"\n  {'template':<28}{'lost':>10}{'kept':>10}   loss rate")
    for name, lost in t.lost.most_common(top):
        kept = t.kept[name]
        print(f"  {name:<28}{lost:>10,}{kept:>10,}{lost / (lost + kept):>11.1%}")

    print(f"\n  {'property':<24}{'template':<24}{'lost':>10}")
    for (prop, name), n in t.lost_property.most_common(top):
        print(f"  {prop:<24}{name:<24}{n:>10,}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("dump", type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many pages, for a quick look")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    if not args.dump.exists():
        sys.exit(f"{args.dump} does not exist")
    report(scan(args.dump, args.limit), args.top)


if __name__ == "__main__":
    main()
