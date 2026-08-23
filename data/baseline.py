#!/usr/bin/env python3
"""
Would something simpler have done this without a neural network?

Worth asking before training anything, and worth re-asking whenever the machine
gets bigger.  The answer depends entirely on the storage budget:

  Resident budget - CP/M, ZX Spectrum, a bare Agon.  Everything competes for the
  same 64KB the program lives in.  A 2-bit model is ~36KB of weights; if a word
  list gets the same accuracy in 2KB, the model is not earning its size.

  Storage budget - an Agon with an SD card.  A corpus no longer has to fit in
  RAM, so the honest floor is no longer a word list.  It is a retriever over the
  whole training set, and that is a much harder thing to beat.

    python data/baseline.py examples/smalltalk/training-data.txt.gz

Every row prints what it would cost on-device, because at a 32GB budget the
accuracy column alone stops meaning anything.  The table is built from the
training half only, using the same held-out split feedme uses, so the numbers
are comparable.

The keyword table is deliberately naive - one vote per word, first known word
wins - because the point is to establish the floor, not to compete.  The
retrievers are not naive: they are what a card-equipped machine would actually
run.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import libinfer
from libdata import Pair, read_files, score_predictions, split_pairs

MIN_PURITY = 0.9   # a word must predict one reply this consistently to count
MIN_COUNT = 3      # ...and appear at least this often

#: Bytes per keyword-table entry: a short word plus a reply index.
TABLE_ENTRY_BYTES = 12

#: A trigram bucket vector stored as int16, which is what an Agon would hold.
BUCKET_BYTES = 2


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


def _unit_vectors(queries: list[str]) -> np.ndarray:
    """L2-normalised trigram bucket vectors, one row per query.

    The same encoder the model sees, so a retriever win here cannot be waved
    away as the retriever having better features.
    """
    vecs = np.array([libinfer.trigram_encode(q) for q in queries], dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def nearest_centroid(train: list[Pair]):
    """Mean trigram vector per reply; answer with the nearest one by cosine.

    The sharpest control in the set: it uses the model's own features, holds one
    vector per *reply* rather than per example, and is smaller than the model.
    """
    replies = sorted({r for _, r in train})
    grouped: dict[str, list] = {r: [] for r in replies}
    for query, reply in train:
        grouped[reply].append(libinfer.trigram_encode(query))

    centroids = np.array([np.mean(grouped[r], axis=0) for r in replies], dtype=np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    def predict(query: str) -> str:
        vec = libinfer.trigram_encode(query).astype(np.float32)
        if not vec.any():
            return replies[0]
        return replies[int(np.argmax(centroids @ vec))]

    return predict, len(replies) * libinfer.NUM_BUCKETS * BUCKET_BYTES


def nearest_neighbour(train: list[Pair]):
    """Answer with the reply of the most similar training query, by cosine.

    This is the floor a machine with an SD card actually has to clear: the whole
    corpus fits on the card with room to spare, and a scan of it is roughly one
    forward pass of arithmetic.

    Costed in the sparse form, because that is what the machine would hold.  A
    query lights up ~25 of 128 buckets with counts that never exceed a nibble,
    so one byte of bucket index and one of count per nonzero beats both a dense
    int16 vector (5x larger) and storing the text to re-encode on every lookup
    (cheaper to hold, but thousands of hashes per query to use).
    """
    replies = [r for _, r in train]
    raw = np.array([libinfer.trigram_encode(q) for q, _ in train])
    vectors = raw.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9

    def predict(query: str) -> str:
        vec = libinfer.trigram_encode(query).astype(np.float32)
        if not vec.any():
            return replies[0]
        return replies[int(np.argmax(vectors @ vec))]

    nonzero = int((raw != 0).sum())
    return predict, nonzero * 2 + len(train) * 3


def nearest_neighbour_words(train: list[Pair]):
    """1-NN again, but over word sets by Jaccard rather than trigram buckets.

    A different feature space on purpose: a win here cannot be dismissed as an
    artifact of the trigram encoder the model happens to share.
    """
    word_sets = [set(q.split()) for q, _ in train]
    replies = [r for _, r in train]

    postings: dict[str, list[int]] = defaultdict(list)
    for i, words in enumerate(word_sets):
        for word in words:
            postings[word].append(i)

    def predict(query: str) -> str:
        words = set(query.split())
        shared: Counter = Counter()
        for word in words:
            shared.update(postings.get(word, ()))
        best, best_score = 0, -1.0
        # Ascending index order, strictly-greater: first wins, like ARGMAX.
        for i in sorted(shared):
            union = len(words) + len(word_sets[i]) - shared[i]
            score = shared[i] / union if union else 0.0
            if score > best_score:
                best, best_score = i, score
        return replies[best]

    corpus = sum(len(q) + len(r) + 2 for q, r in train)
    return predict, corpus


def _size(nbytes: int | None) -> str:
    if nbytes is None:
        return ''
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes / (1024 * 1024):.1f} MB"


def _model_split_seed(model) -> int | None:
    """The split seed the model was trained under, if it recorded one."""
    return getattr(model, 'split_seed', None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('paths', nargs='*', help='Data files (default: stdin)')
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', help='Also score a trained .npz on the same split')
    parser.add_argument('--accum-bits', type=int, default=None,
                        help='Accumulator width to score the model at (default: '
                             'whatever the model recorded, else 16 for Z80)')
    parser.add_argument('--no-retrievers', action='store_true',
                        help='Skip the corpus retrievers (they dominate the runtime '
                             'on a large dataset)')
    args = parser.parse_args()

    pairs = read_files(args.paths)
    if not pairs:
        raise SystemExit("no pairs found")

    train, val = split_pairs(pairs, args.val_frac, args.seed)
    table = build_table(train)
    # Guessing the most common reply is what the table does when it knows
    # nothing, and is also the score to beat for "did this learn anything".
    fallback = Counter(r for _, r in train).most_common(1)[0][0]

    print(f"{len(train):,} training pairs, {len(val):,} held out, "
          f"{len({r for _, r in pairs})} distinct replies\n")

    rows: list[tuple[str, list[Pair], object, int | None]] = [
        (f"always answering {fallback!r}", val, lambda q: fallback, 0),
        ("keyword table, training", train, lambda q: classify(q, table, fallback),
         len(table) * TABLE_ENTRY_BYTES),
        ("keyword table, held-out", val, lambda q: classify(q, table, fallback),
         len(table) * TABLE_ENTRY_BYTES),
    ]

    if not args.no_retrievers:
        for name, factory in (("nearest centroid, held-out", nearest_centroid),
                              ("1-NN trigram, held-out", nearest_neighbour),
                              ("1-NN word Jaccard, held-out", nearest_neighbour_words)):
            predict, nbytes = factory(train)
            rows.append((name, val, predict, nbytes))

    if args.model:
        model = libinfer.Model.load(args.model)

        trained_seed = _model_split_seed(model)
        if trained_seed is not None and trained_seed != args.seed:
            print(f"WARNING: {Path(args.model).name} was trained on the seed-"
                  f"{trained_seed} split, but this is seed {args.seed}.  Held-out\n"
                  f"         pairs here were in its training set; the model row is "
                  f"not a held-out\n         score.  Re-run with --seed "
                  f"{trained_seed}.\n", file=sys.stderr)
        elif trained_seed is None:
            print(f"NOTE: {Path(args.model).name} records no split seed, so this "
                  f"cannot check that\n      --seed {args.seed} matches the split it "
                  f"was trained on.\n", file=sys.stderr)

        accum_bits = args.accum_bits or getattr(model, 'accum_bits', None) or 16
        longest = max(len(r) for _, r in pairs) + 1
        nbytes = sum(w.size for w in model.weights) // 4 + sum(b.size * 2 for b in model.biases)
        rows.append((
            f"model {Path(args.model).name}, held-out", val,
            lambda q: libinfer.generate(model, q, longest, accum_bits=accum_bits),
            nbytes,
        ))

    print(f"{'':34}{'on device':>11}{'overall':>9}{'macro':>9}")
    for name, subset, predict, nbytes in rows:
        overall, macro = score_predictions(subset, predict)
        print(f"  {name:32}{_size(nbytes):>11}{overall:>9.1%}{macro:>9.1%}")

    print("\nOverall weights every pair equally, so a dominant answer inflates it;\n"
          "macro averages over distinct answers.  Compare macro with macro - and\n"
          "never with a model's ValChr, which counts characters, not answers.\n"
          "\n'On device' is what each row would occupy on the target: 2-bit weights\n"
          "for the model, int16 bucket vectors for the retrievers, raw text for the\n"
          "word-set index.  A row that beats the model in fewer bytes is the whole\n"
          "reason this script exists.")


if __name__ == '__main__':
    main()
