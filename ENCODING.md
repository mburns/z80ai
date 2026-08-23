# Query encoding: flat vs position-aware

The query encoder hashes trigrams into 128 buckets. By default the same trigram
lands in the same bucket wherever it appears, which makes the encoding
order-insensitive — a feature for paraphrase matching and a wall for anything
that parses commands.

`--position-bands N` seeds each trigram's hash with where in the query it
appeared, making the encoding order-aware. It is off by default (`N=1`), it is
recorded in the model file, and every backend reads it from there, so a model
can never be built with a tokenizer it wasn't trained with.

## Why not wider buckets

The obvious first guess is that queries collide in 128 buckets. Measured on the
shipped data, they do not:

| buckets | `guess` collisions | `tinychat` collisions |
|---:|---:|---:|
| 128 | 0 / 18,087 | 8 / 2,280 (0.4%) |
| 4,096 | 0 | 0 |

Nor does widening improve how far apart the encoder puts queries needing
different answers — on `tinychat` that separation *falls* from 0.074 to 0.066
going 128 → 4,096. And it does not touch word order at all:

| | 128 buckets | 4,096 buckets |
|---|---:|---:|
| `"open the door and turn on the lights"` vs `"turn on the door and open the lights"` | 0.985 | 0.977 |

Cosine 0.98 either way. This is not a hash-capacity problem, so hash capacity
does not fix it.

## What the bands do

A trigram starting at character `i` is hashed with seed
`min(i >> 3, N-1) * 7`. Fixed-width bands, not proportional ones: `i >> 3` is
three `RRCA`s, whereas a proportional band would need a multiply and a divide in
the tokenizer's inner loop. The `* 7` matches the convention the context
encoder already uses for its own position seeding.

Cosine between reordered queries, by scheme:

| | flat | relative bands | `i>>2` | `i>>3` | word index |
|---|---:|---:|---:|---:|---:|
| mean over 5 reordered pairs | 0.959 | 0.555 | 0.552 | **0.521** | 0.490 |
| mean over 3 paraphrase pairs | 0.834 | 0.743 | 0.696 | **0.805** | 0.754 |
| gap (higher is better) | −0.125 | +0.188 | +0.144 | **+0.284** | +0.263 |

`i >> 3` wins and is the cheapest thing on the list.

`TOKENIZE` runs once per query, not once per generated character, so the extra
hash step does not appear in generation time. A model built with `N=1` is
**byte-identical** to one built before the option existed — `TOKPOS` and the
seeding code are not emitted at all.

## What it is worth

`examples/parser/` is a corpus where word order is the only signal: `PUT KEY IN
BOX` is `OK`, `PUT BOX IN KEY` is `NO`. Object *pairs* are held out, not just
examples, so memorising the training pairs earns nothing.

```
256 -> 96 -> 64 -> 4, 400 epochs, identical data and seed

           encoder   char acc    train     eval
    flat (current)     99.3%    97.9%    85.9%
  8 position bands    100.0%   100.0%    98.4%
   always-majority         -        -    56.2%
```

Errors on unseen object pairs fall by 89%. The flat encoder does better than
chance because reversing a query is not quite invisible to it — the word
boundary trigrams differ — but the signal is weak and it does not generalise.

## What it costs

The same comparison on `tinychat`, which is paraphrase-heavy by design:

```
256 -> 192 -> 128 -> 39, 400 epochs, 15% of queries held out

           encoder   char acc    train     eval
    flat (current)     88.0%    54.9%    18.9%
  8 position bands     88.5%    55.9%    12.5%
   always-majority         -        -     9.0%
```

Generalisation gets **worse**, and that is the expected result rather than a
bug. TRAINING.md teaches order-invariance as a virtue — *"same response should
trigger from multiple input styles"* — and a corpus built that way loses value
when the encoder starts distinguishing those styles.

## Choosing

Use position bands when word order carries meaning: command parsers, adventure
games, anything where `X to Y` differs from `Y to X`. Keep the flat encoder for
terse Q&A and paraphrase matching.

The encoder and the training data have to agree. Turning bands on over a
paraphrase corpus makes things worse, and it is not a setting worth sweeping
blindly.

```bash
python feedme.py --position-bands 8 --file training-data.txt
python build.py --model command_model_autoreg.pt --output PARSER.COM
```
