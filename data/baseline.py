#!/usr/bin/env python3
"""
Would a keyword table have done this without a neural network?

Worth asking before training anything.  A 2-bit model is ~36KB of weights and an
hour of training; if a word list gets the same accuracy in 2KB, the model is not
earning its size and the task is the wrong one.

    python data/baseline.py examples/smalltalk/training-data.txt.gz

The table is built from the training half only, using the same held-out split
feedme uses, so the two numbers are comparable.  It is deliberately naive - one
vote per word, first known word wins - because the point is to establish the
floor, not to compete.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from libdata import Pair, read_files, split_pairs

MIN_PURITY = 0.9   # a word must predict one reply this consistently to count
MIN_COUNT = 3      # ...and appear at least this often


def build_table(train: list[Pair]) -> dict[str, str]:
    """word -> the reply it predicts, for words that predict one reliably."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for query, reply in train:
        for word in set(query.split()):
            votes[word][reply] += 1

    table = {}
    for word, counter in votes.items():
        reply, n = counter.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(counter.values()) >= MIN_PURITY:
            table[word] = reply
    return table


def classify(query: str, table: dict[str, str], fallback: str) -> str:
    for word in query.split():
        if word in table:
            return table[word]
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('paths', nargs='*', help='Data files (default: stdin)')
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    pairs = read_files(args.paths)
    if not pairs:
        raise SystemExit("no pairs found")

    train, val = split_pairs(pairs, args.val_frac, args.seed)
    table = build_table(train)
    # Guessing the most common reply is what the table does when it knows
    # nothing, and is also the score to beat for "did this learn anything".
    fallback = Counter(r for _, r in train).most_common(1)[0][0]

    print(f"{len(table)} words from {len(train):,} training pairs, "
          f"~{len(table) * 12:,} bytes as a table")
    print(f"  always answering {fallback!r}      "
          f"{sum(r == fallback for _, r in val) / len(val):6.1%}")
    for name, subset in (("keyword table, training", train),
                         ("keyword table, held-out", val)):
        ok = sum(classify(q, table, fallback) == r for q, r in subset)
        print(f"  {name}  {ok / len(subset):6.1%}")


if __name__ == '__main__':
    main()
