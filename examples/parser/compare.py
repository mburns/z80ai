#!/usr/bin/env python3
"""
Train the flat and position-aware query encoders on identical data and compare.

Everything except the encoder is held fixed: same architecture, same seed, same
data, same epochs. The eval set uses object pairs held out entirely, so the
score measures generalisation rather than memorisation.

    ./compare.py --epochs 400
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import feedme


def load(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path) as fh:
        for line in fh:
            if "|" in line:
                q, a = line.strip().split("|", 1)
                pairs.append((q.strip().upper(), a.strip().upper()))
    return pairs


def build_examples(pairs, query_encoder, context_encoder):
    xs, ys = [], []
    for query, response in pairs:
        for x, y in feedme.create_training_examples(
            query, response, query_encoder, context_encoder
        ):
            xs.append(x)
            ys.append(y)
    return (torch.tensor(np.stack(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.long))


def response_accuracy(model, pairs, query_encoder, context_encoder) -> float:
    """Fraction of queries whose whole generated response is correct."""
    correct = 0
    for query, expected in pairs:
        got = feedme.generate_response(model, query, query_encoder,
                                       context_encoder, max_len=8)
        correct += got.strip() == expected
    return correct / len(pairs)


def run(bands: int, train_pairs, eval_pairs, hidden, epochs, seed) -> dict:
    # torch's is the only RNG this draws from: neither feedme nor libdata
    # touches numpy's global one, so the np.random.seed that used to sit here
    # seeded a generator nothing sampled.
    torch.manual_seed(seed)

    query_encoder = feedme.TrigramEncoder(num_buckets=128, position_bands=bands)
    context_encoder = feedme.ContextEncoder(num_buckets=128, context_len=8)

    x, y = build_examples(train_pairs, query_encoder, context_encoder)
    model = feedme.AutoregressiveModel(input_size=256, hidden_sizes=hidden,
                                       num_chars=feedme.NUM_CHARS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        temp = 0.3 + 0.7 * min(1.0, epoch / (epochs * 0.8))
        out = model(x, quant_temp=temp)
        loss = criterion(out, y) + model.compute_quantization_loss() * 0.10
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        int_acc = (model(x, use_int=True).argmax(dim=1) == y).float().mean().item()
    return {
        "bands": bands,
        "char_acc": int_acc,
        "train": response_accuracy(model, train_pairs, query_encoder, context_encoder),
        "eval": response_accuracy(model, eval_pairs, query_encoder, context_encoder),
        "model": model,
        "encoders": (query_encoder, context_encoder),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden", default="96,64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", default="training-data.txt")
    parser.add_argument("--eval", default="eval-data.txt")
    parser.add_argument("--save", default=None,
                        help="Write the position-aware model here (.pt)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    train_pairs = load(os.path.join(here, args.train))
    eval_pairs = load(os.path.join(here, args.eval))
    hidden = [int(v) for v in args.hidden.split(",")]

    # The charset is global state in feedme; set it once from all the data.
    feedme.CHARSET = feedme.build_charset_from_pairs(train_pairs + eval_pairs)
    feedme.CHAR_TO_IDX = {c: i for i, c in enumerate(feedme.CHARSET)}
    feedme.IDX_TO_CHAR = dict(enumerate(feedme.CHARSET))
    feedme.EOS_IDX = len(feedme.CHARSET) - 1
    feedme.NUM_CHARS = len(feedme.CHARSET)

    majority = max(
        sum(1 for _q, a in eval_pairs if a == c) for c in {a for _q, a in eval_pairs}
    ) / len(eval_pairs)

    print(f"train {len(train_pairs)} queries, eval {len(eval_pairs)} queries "
          f"on held-out object pairs")
    print(f"architecture 256 -> {' -> '.join(map(str, hidden))} -> "
          f"{feedme.NUM_CHARS}, {args.epochs} epochs\n")
    print(f"{'encoder':>22} {'char acc':>10} {'train':>8} {'eval':>8}")
    print("-" * 52)

    results = {}
    for bands, label in ((1, "flat (current)"), (8, "8 position bands")):
        r = run(bands, train_pairs, eval_pairs, hidden, args.epochs, args.seed)
        results[bands] = r
        print(f"{label:>22} {r['char_acc']:9.1%} {r['train']:8.1%} {r['eval']:8.1%}")

    print(f"{'always-majority':>22} {'-':>9} {'-':>8} {majority:8.1%}")
    print("\neval is the number that matters: those object pairs never appeared "
          "in training,\nin any phrasing, so a model can only score by "
          "representing word order.")

    if args.save:
        r = results[8]
        torch.save({
            "model_state": r["model"].state_dict(),
            "architecture": {
                "input_size": 256, "hidden_sizes": hidden,
                "num_classes": feedme.NUM_CHARS, "position_bands": 8,
            },
            "charset": feedme.CHARSET,
            "total_epochs": args.epochs,
        }, args.save)
        print(f"\nSaved the position-aware model to {args.save}")


if __name__ == "__main__":
    main()
