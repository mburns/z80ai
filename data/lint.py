#!/usr/bin/env python3
"""
Report what a training set will actually do to a 2-bit model, before you train.

Line count is the least useful number about a dataset here.  What decides
whether a model can learn it is the number of *distinct responses* - the
character decoder is really a label decoder, so each new response costs
capacity and each new character costs an output neuron - and how often the same
query appears with two different labels, which caps accuracy outright.

    python data/lint.py examples/guess/training-data.txt.gz
    gzip -dc examples/tinychat/training-data.txt.gz | python data/lint.py --strict

`--strict` exits non-zero if anything is flagged, so it can gate a build.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libdata import accuracy_ceiling, build_charset, read_files
from libinfer import CONTEXT_LEN, MAX_OUTPUT_LEN, NUM_BUCKETS, trigram_encode

# Thresholds. These are the ones TRAINING.md quotes; keeping them here means
# the advice and the check cannot drift apart.
MAX_LABEL_SHARE = 0.40      # no response should dominate more than this
MIN_LABEL_SHARE = 0.01      # below this a class is too rare to learn
MAX_RESPONSES = 50          # distinct responses a 2-bit model handles well
# Advisory, and deliberately not named MAX_RESPONSE_LEN: libdata has one of
# those, and it is the hard limit at which a response is truncated (50). This
# is the point past which a response is merely slow.
SLOW_RESPONSE_LEN = 12      # characters; longer is slower and harder
MAX_CONTRADICTION_RATE = 0.05


def phrasing_redundancy(pairs: list[tuple[str, str]], sample: int = 60) -> float:
    """Median cosine to the nearest query wanting the *same* reply.

    How many different ways the data says each thing.  A model generalizes by
    seeing one intent phrased several ways; if every query is a one-off there is
    nothing to generalize from, however many of them there are.  Measured on the
    shipped data:

        smalltalk  0.74   crowdsourced paraphrases, 149 per reply -> 80.6%
        guess      0.58   templated, but 7,180 per reply          -> 81.3%
        tinychat   0.54   hand-written one-offs, 96 per reply     -> 30.3%

    Reported rather than judged: the number only means something next to the
    examples-per-response count above it.  Low on both is the bad case.
    """
    import numpy as np

    by_reply: dict[str, list[str]] = defaultdict(list)
    for query, reply in pairs:
        by_reply[reply].append(query)

    vectors: dict[str, np.ndarray] = {}
    for query, _ in pairs:
        if query not in vectors:
            v = trigram_encode(query).astype(float)
            norm = np.linalg.norm(v)
            vectors[query] = v / norm if norm else v

    rng = np.random.default_rng(0)
    nearest: list[float] = []
    for queries in by_reply.values():
        if len(queries) < 2:
            continue
        chosen = (queries if len(queries) <= sample
                  else list(rng.choice(queries, sample, replace=False)))
        matrix = np.stack([vectors[q] for q in chosen])
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, -1.0)
        nearest.extend(sims.max(axis=1))

    return float(np.median(nearest)) if nearest else 0.0


def report(pairs: list[tuple[str, str]], phrasebook: bool = False) -> list[str]:
    """Print the analysis; return a list of problems found.

    ``phrasebook`` switches off the checks that only bind a *character* decoder.
    When the model emits an index into a table of replies held on an SD card,
    the reply text costs it nothing: response length, response count, the
    charset and rare characters all stop being capacity. What still binds -
    class balance, contradictions, and queries that encode identically - is
    about the input and the labels, and is checked either way.
    """
    problems: list[str] = []
    labels = Counter(r for _, r in pairs)
    by_query: dict[str, set[str]] = defaultdict(set)
    for query, response in pairs:
        by_query[query].add(response)

    charset = build_charset(pairs)
    rows = sum(len(r) + 1 for _, r in pairs)
    duplicates = len(pairs) - len(set(pairs))
    contradictory = {q for q, rs in by_query.items() if len(rs) > 1}
    involved = sum(1 for q, _ in pairs if q in contradictory)

    print(f"{'pairs':<32}{len(pairs):>10,}")
    print(f"{'unique queries':<32}{len(by_query):>10,}")
    print(f"{'unique responses':<32}{len(labels):>10,}")
    print(f"{'character training rows':<32}{rows:>10,}")
    if phrasebook:
        print(f"{'phrases':<32}{len(labels):>10,}  (indices, not spelled)")
        print(f"{'longest phrase':<32}{max(len(r) for _, r in pairs):>10,}  characters, free")
    else:
        print(f"{'charset':<32}{len(charset) + 1:>10}  {charset!r} + EOS")
        print(f"{'output layer weights':<32}{(len(charset) + 1) * 128:>10,}")
    print(f"{'exact duplicate pairs':<32}{duplicates:>10,}  ({_pct(duplicates, len(pairs))})")
    print(f"{'pairs in a contradiction':<32}{involved:>10,}  ({_pct(involved, len(pairs))})")
    print(f"{'accuracy ceiling':<32}{accuracy_ceiling(pairs):>9.1%}")
    print(f"{'examples per response':<32}{len(pairs) / len(labels):>10,.0f}")
    print(f"{'phrasing redundancy':<32}{phrasing_redundancy(pairs):>10.2f}")

    print("\nresponse distribution")
    for label, count in labels.most_common(12):
        share = count / len(pairs)
        bar = '#' * int(share * 40)
        print(f"  {label:<14}{count:>8,}  {share:>6.1%}  {bar}")
    if len(labels) > 12:
        print(f"  ... {len(labels) - 12:,} more")

    # --- checks --------------------------------------------------------------

    if len(labels) > MAX_RESPONSES and not phrasebook:
        problems.append(
            f"{len(labels)} distinct responses; a 2-bit model handles about "
            f"{MAX_RESPONSES}. This is the number that decides whether the "
            f"model can learn the task - not the line count."
        )

    # Judge dominance against an even split, not a flat 40%: with two classes an
    # even split *is* 50%, and flagging that would be nonsense.
    top, top_count = labels.most_common(1)[0]
    even = 1 / len(labels)
    limit = max(MAX_LABEL_SHARE, 1.5 * even)
    if top_count / len(pairs) > limit:
        problems.append(
            f"{top!r} is {top_count / len(pairs):.1%} of the data, against "
            f"{even:.1%} for an even split across {len(labels)} responses. A "
            f"small model falls back to the majority class whenever it is "
            f"unsure, so this becomes its default answer."
        )

    # Judged against an even split for the same reason dominance is: a flat 1%
    # floor calls every class of a perfectly balanced 151-way set too rare,
    # which is exactly backwards. Half of even, or 1%, whichever is smaller -
    # so this is unchanged for every dataset with fewer than fifty responses.
    floor = min(MIN_LABEL_SHARE, even / 2)
    rare = [lab for lab, n in labels.items() if n / len(pairs) < floor]
    if rare:
        shown = ', '.join(repr(r) for r in sorted(rare)[:6])
        problems.append(
            f"{len(rare)} response(s) under {floor:.1%} of the data, against "
            f"{even:.1%} for an even split "
            f"({shown}{', ...' if len(rare) > 6 else ''}). Too rare to learn."
        )

    if involved / len(pairs) > MAX_CONTRADICTION_RATE:
        worst = sorted(contradictory)[:3]
        detail = '; '.join(f"{q!r} -> {sorted(by_query[q])}" for q in worst)
        problems.append(
            f"{involved / len(pairs):.1%} of pairs have a query that appears "
            f"with more than one response. No amount of training fixes this - "
            f"it caps accuracy at {accuracy_ceiling(pairs):.1%}. e.g. {detail}"
        )

    long_responses = {r for _, r in pairs if len(r) > SLOW_RESPONSE_LEN}
    if long_responses and not phrasebook:
        shown = ', '.join(repr(r) for r in sorted(long_responses)[:4])
        problems.append(
            f"{len(long_responses)} response(s) longer than {SLOW_RESPONSE_LEN} "
            f"characters ({shown}). Every character is another forward pass."
        )

    over = {r for _, r in pairs if len(r) > MAX_OUTPUT_LEN}
    if over and not phrasebook:
        problems.append(
            f"{len(over)} response(s) longer than MAX_OUTPUT_LEN={MAX_OUTPUT_LEN}; "
            f"the generated code will truncate them."
        )

    # A character used by only a line or two still costs a whole output neuron.
    char_lines: Counter[str] = Counter()
    for _, response in pairs:
        for c in set(response):
            char_lines[c] += 1
    expensive = sorted(c for c, n in char_lines.items() if n <= 3)
    if expensive and not phrasebook:
        problems.append(
            f"character(s) {''.join(expensive)!r} appear in 3 or fewer pairs but "
            f"each costs 128 output weights and forces a full retrain to add. "
            f"Dropping them shrinks the model."
        )

    # Two queries that encode to the same vector are indistinguishable to the
    # model. That only matters if they disagree about the answer - colliding
    # phrasings of the same command are fine, and common.
    seen: dict[bytes, str] = {}
    collisions: list[tuple[str, str]] = []
    for query in by_query:
        key = trigram_encode(query).tobytes()
        other = seen.setdefault(key, query)
        if other != query and by_query[other] != by_query[query]:
            collisions.append((other, query))
    if collisions:
        shown = '; '.join(
            f"{a!r} {sorted(by_query[a])} == {b!r} {sorted(by_query[b])}"
            for a, b in collisions[:3]
        )
        problems.append(
            f"{len(collisions)} pair(s) of queries with different answers hash "
            f"to the same {NUM_BUCKETS}-bucket vector, so no model can tell "
            f"them apart: {shown}"
        )

    return problems


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('paths', nargs='*', help='Data files (default: stdin)')
    parser.add_argument('--strict', action='store_true',
                        help='Exit non-zero if any problem is reported')
    parser.add_argument('--phrasebook', action='store_true',
                        help='The model will emit an index into a table of '
                             'replies rather than spell them, so response '
                             'length, count and charset stop being capacity')
    args = parser.parse_args()

    pairs = read_files(args.paths)
    if not pairs:
        print("No pairs found.", file=sys.stderr)
        raise SystemExit(2)

    print(f"CONTEXT_LEN={CONTEXT_LEN}, NUM_BUCKETS={NUM_BUCKETS}\n")
    problems = report(pairs, phrasebook=args.phrasebook)

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for i, problem in enumerate(problems, 1):
            print(f"\n  {i}. {problem}")
    else:
        print("\nNo problems found.")

    if args.strict and problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
