#!/usr/bin/env python3
"""
Generate an order-dependent corpus: "PUT <A> IN <B>".

Every query is a container command. The answer is OK when A fits inside B and
NO when it does not, so `PUT KEY IN BOX` and `PUT BOX IN KEY` need opposite
answers from almost exactly the same bag of trigrams. That is the case the flat
encoder cannot represent and a position-aware one can - see ENCODING.md.

Object pairs are split, not just examples: the eval set uses pairs the model
never saw in any phrasing, so memorising the training pairs earns nothing.

    ./gendata.py --out-train train.txt --out-eval eval.txt
"""

from __future__ import annotations

import argparse
import random

# Objects by size class. A fits in B exactly when its class is smaller.
SIZES: dict[str, int] = {
    "COIN": 0, "KEY": 0, "RING": 0, "STAMP": 0,
    "CUP": 1, "BOOK": 1, "LAMP": 1, "BOOT": 1,
    "BOX": 2, "BAG": 2, "CRATE": 2, "CHEST": 2,
    "CART": 3, "SHED": 3, "CAVE": 3, "BARN": 3,
}

TEMPLATES = (
    "PUT {a} IN {b}",
    "PUT THE {a} IN THE {b}",
    "PLACE {a} INTO {b}",
    "DROP THE {a} IN THE {b}",
)


def answer(a: str, b: str) -> str:
    """OK when a fits inside b."""
    return "OK" if SIZES[a] < SIZES[b] else "NO"


def pairs() -> list[tuple[str, str]]:
    names = sorted(SIZES)
    return [(a, b) for a in names for b in names if a != b]


def render(pair_list: list[tuple[str, str]]) -> list[str]:
    return [
        f"{template.format(a=a, b=b)}|{answer(a, b)}"
        for a, b in pair_list
        for template in TEMPLATES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-train", default="training-data.txt")
    parser.add_argument("--out-eval", default="eval-data.txt")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="Fraction of object pairs reserved for eval")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_pairs = pairs()
    rng.shuffle(all_pairs)

    # Hold out whole pairs, so eval queries contain object combinations that
    # never appeared in training under any phrasing.
    cut = int(len(all_pairs) * (1 - args.holdout))
    train, evaluate = all_pairs[:cut], all_pairs[cut:]

    for path, subset in ((args.out_train, train), (args.out_eval, evaluate)):
        lines = render(subset)
        rng.shuffle(lines)
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        ok = sum(1 for line in lines if line.endswith("|OK"))
        print(f"{path}: {len(lines):,} lines from {len(subset)} object pairs "
              f"({ok / len(lines):.0%} OK)")

    print(f"\n{len(SIZES)} objects, {len(all_pairs)} ordered pairs, "
          f"{len(TEMPLATES)} phrasings")
    print("Reversing any query flips its answer, so word order carries the label.")


if __name__ == "__main__":
    main()
