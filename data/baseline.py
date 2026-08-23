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

from libdata import Pair, read_files, score_predictions, split_pairs

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
    parser.add_argument('--model', help='Also score a trained .npz on the same split')
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
          f"~{len(table) * 12:,} bytes as a table\n")
    rows: list[tuple[str, list[Pair], object]] = [
        (f"always answering {fallback!r}", val, lambda q: fallback),
        ("keyword table, training", train, lambda q: classify(q, table, fallback)),
        ("keyword table, held-out", val, lambda q: classify(q, table, fallback)),
    ]

    if args.model:
        import libinfer

        model = libinfer.Model.load(args.model)
        longest = max(len(r) for _, r in pairs) + 1
        rows.append((
            f"model {Path(args.model).name}, held-out", val,
            lambda q: libinfer.generate(model, q, longest),
        ))

    print(f"{'':36}{'overall':>9}{'macro':>9}")
    for name, subset, predict in rows:
        overall, macro = score_predictions(subset, predict)
        print(f"  {name:34}{overall:>9.1%}{macro:>9.1%}")

    print("\nOverall weights every pair equally, so a dominant answer inflates it;\n"
          "macro averages over distinct answers.  Compare macro with macro - and\n"
          "never with a model's ValChr, which counts characters, not answers.")


if __name__ == '__main__':
    main()
