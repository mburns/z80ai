#!/usr/bin/env python3
"""
Train a phrasebook classifier: one forward pass, one reply.

feedme.py trains a character decoder - it spells its answer out one output
neuron at a time, so every distinct character costs 128 weights and every
distinct reply costs capacity.  That is the right shape when the whole program
has to fit in 64KB, and it is why the shipped examples answer in words like
`OK` and `MAYBE`.

Given an SD card it is the wrong shape.  The reply text can live on the card, so
the model only has to choose an *index* into it - which makes reply length free,
kills the context window (there is nothing to condition on when the answer is
picked in one step), and halves layer one, because only the 128 query buckets
are input.

    python classify.py --file router.txt -o router.npz
    python classify.py --file banking.txt -o banking.npz --accum-bits 24

Selection is on held-out macro accuracy, and the split is libdata.split_pairs
with the same defaults data/baseline.py uses, so the number this prints and the
number the baseline grid prints are the same number.

Unlike feedme, this is seeded.  With one model per domain an unseeded trainer
means ten models that cannot be reproduced, and a shipped answer that changes
when someone retrains.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import libdata
import libinfer
from libqat import QATCommandClassifier

#: Matches feedme's schedule: float weights early, fully quantized by 80%.
QUANT_LOSS_WEIGHT = 0.10
OVERFLOW_LOSS_WEIGHT = 0.03


def encode(queries: list[str], position_bands: int = libinfer.FLAT,
           num_buckets: int = libinfer.NUM_BUCKETS) -> torch.Tensor:
    """Query buckets, scaled the way the QAT layers expect to see them."""
    vecs = np.array(
        [libinfer.trigram_encode(q, num_buckets, position_bands)
         for q in queries],
        dtype=np.float32,
    )
    return torch.from_numpy(vecs / libinfer.BUCKET_WEIGHT)


def quantized_model(net: QATCommandClassifier, phrases: list[str],
                    split_seed: int, accum_bits: int,
                    position_bands: int,
                    num_buckets: int = libinfer.NUM_BUCKETS) -> libinfer.Model:
    """The integer model the Agon would actually run."""
    params = net.get_quantized_params()
    return libinfer.Model.from_params(
        params,
        # A phrasebook never spells anything, but Model.charset is not optional
        # and buildez80 sizes CHARTBL from it. An empty-but-for-EOS charset says
        # "this model decodes through phrases, not characters" unambiguously.
        charset="\x00",
        position_bands=position_bands,
        num_buckets=num_buckets,
        split_seed=split_seed,
        phrases=phrases,
        accum_bits=accum_bits,
    )


def evaluate(model: libinfer.Model, pairs: list[libdata.Pair],
             accum_bits: int) -> tuple[float, float]:
    """Held-out (overall, macro) under the integer arithmetic that ships."""
    return libdata.score_predictions(
        pairs, lambda q: libinfer.classify(model, q, accum_bits)
    )


def class_weights(pairs: list[libdata.Pair],
                  phrases: list[str]) -> torch.Tensor:
    """N / (K * n_c): every class contributes the same total gradient mass.

    Without this a rare class is answered by the prior. Measured on the oracle's
    relation set, where four multi-hop classes bring 240 examples each against
    1,200 for the one-hop classes: the rare classes go from 40.3% to 64.7% on
    unseen phrasings, and the common ones give up 2.1 points of macro.
    """
    counts = Counter(r for _, r in pairs)
    return torch.tensor(
        [len(pairs) / (len(phrases) * counts[p]) for p in phrases],
        dtype=torch.float32)


def train(pairs: list[libdata.Pair], hidden_sizes: list[int], epochs: int,
          lr: float, seed: int, split_seed: int, val_frac: float,
          accum_bits: int, position_bands: int, quiet: bool = False,
          balance: bool = False,
          num_buckets: int = libinfer.NUM_BUCKETS,
          ) -> tuple[libinfer.Model, float, float]:
    torch.manual_seed(seed)

    train_pairs, val_pairs = libdata.split_pairs(pairs, val_frac, split_seed)
    if not val_pairs:
        raise SystemExit("no held-out pairs; --val-frac is too small")

    # sorted(): the phrase index is baked into the weights, into PHRASES.DAT and
    # into the reference model, and a set's iteration order would make those
    # three disagree between runs.
    phrases = sorted({r for _, r in pairs})
    index = {phrase: i for i, phrase in enumerate(phrases)}

    x = encode([q for q, _ in train_pairs], position_bands, num_buckets)
    y = torch.tensor([index[r] for _, r in train_pairs], dtype=torch.long)

    net = QATCommandClassifier(num_buckets, hidden_sizes, len(phrases))
    # An eZ80 accumulates in 24 bits and cannot wrap for any plausible layer
    # width, so the penalty that keeps a Z80 model inside int16 is pure
    # regularization tax here. Raising the ceiling makes it stop firing on its
    # own rather than needing a branch in the loss.
    net.set_max_accum(2 ** (accum_bits - 1) - 1)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_pairs, phrases) if balance else None)

    best: libinfer.Model | None = None
    best_macro = -1.0
    best_overall = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        net.train()
        net.set_quant_temp(0.3 + 0.7 * min(1.0, epoch / (epochs * 0.8)))

        optimizer.zero_grad()
        logits = net(x)
        loss = (criterion(logits, y)
                + QUANT_LOSS_WEIGHT * net.compute_quantization_loss()
                + OVERFLOW_LOSS_WEIGHT * net.compute_total_overflow_penalty(x))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            net.eval()
            candidate = quantized_model(net, phrases, split_seed, accum_bits,
                                        position_bands, num_buckets)
            overall, macro = evaluate(candidate, val_pairs, accum_bits)
            if macro > best_macro:
                best, best_macro, best_overall, best_epoch = (
                    candidate, macro, overall, epoch)
            if not quiet:
                print(f"  epoch {epoch:4d}  loss {loss.item():7.4f}  "
                      f"val {overall:6.1%} / {macro:6.1%} macro"
                      f"{'  *' if macro == best_macro else ''}")

    assert best is not None
    if not quiet:
        print(f"  best macro {best_macro:.1%} at epoch {best_epoch}")
    return best, best_overall, best_macro


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('paths', nargs='*', help='Data files (default: stdin)')
    parser.add_argument('--file', '-f', help='Training data file (same as a path)')
    parser.add_argument('--output', '-o', default='phrasebook.npz')
    parser.add_argument('--hidden-sizes', default='256,192',
                        help='Comma-separated hidden layer widths')
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0,
                        help='Torch seed. Training IS seeded here, unlike feedme')
    parser.add_argument('--split-seed', type=int, default=0,
                        help='libdata.split_pairs seed; recorded in the model so '
                             'data/baseline.py can check it is scoring the right split')
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--accum-bits', type=int, default=24, choices=[16, 24],
                        help='24 for eZ80 (the default: a phrasebook is an Agon '
                             'shape), 16 to keep the model Z80-compatible')
    parser.add_argument('--position-bands', type=int, default=libinfer.FLAT)
    parser.add_argument('--buckets', type=int, default=libinfer.NUM_BUCKETS,
                        help="Trigram buckets the encoder hashes into. The "
                             "device takes the bucket index from one byte, so "
                             "256 is the most it can address without a wider "
                             "tokenizer - and 256 is where the accuracy stops "
                             "improving anyway. See tools/bucket_sweep.py")
    parser.add_argument('--balance', action='store_true',
                        help='Weight the loss by inverse class frequency. Use '
                             'it when classes differ in size by more than a '
                             'few times; a rare class is otherwise answered by '
                             'the prior. Off by default so the committed '
                             'example models stay reproducible')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    paths = list(args.paths) + ([args.file] if args.file else [])
    pairs = libdata.read_files(paths)
    if not pairs:
        raise SystemExit("no pairs found")

    hidden_sizes = [int(n) for n in args.hidden_sizes.split(',') if n.strip()]
    phrases = sorted({r for _, r in pairs})
    print(f"{len(pairs):,} pairs, {len(phrases)} phrases, "
          f"{args.buckets}->{'->'.join(map(str, hidden_sizes))}"
          f"->{len(phrases)}, {args.accum_bits}-bit accumulator")

    model, overall, macro = train(
        pairs, hidden_sizes, args.epochs, args.lr, args.seed, args.split_seed,
        args.val_frac, args.accum_bits, args.position_bands, args.quiet,
        args.balance, args.buckets,
    )
    model.save_npz(args.output)

    weights = sum(w.size for w in model.weights)
    print(f"\nheld-out {overall:.1%} overall, {macro:.1%} macro")
    print(f"{weights:,} weights, {weights // 4:,} bytes packed 2-bit")
    print(f"wrote {args.output}")

    if not args.quiet:
        json.dump({'overall': overall, 'macro': macro, 'phrases': len(phrases)},
                  sys.stderr)
        print(file=sys.stderr)


if __name__ == '__main__':
    main()
