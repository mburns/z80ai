# tinychat

A conversational toy: terse, slightly rude, answers the shape of what you typed.

```
> hello
HI
> why
WHY?
> goodbye
BYE
```

## The vocabulary was the problem

The hand-written corpus (`source-data.txt.gz`, 2,990 queries in a consistent
voice) is good. Its replies were not: **502 distinct ones, 297 used exactly
once**. That is not a vocabulary, it is 502 classes averaging six examples each,
and 27.6% of pairs contradicted another pair.

`canonicalize.py` collapses them onto **11 replies**, in two steps. Almost all
of the tail was style rather than meaning — `YA` / `YUP` / `YE` / `BET` /
`VALID` are one answer wearing five hats, and so are `WAT` / `WUT` / `HUH?` /
`???` — so it maps rather than deletes: every query survives, only its reply
changes.

```bash
./canonicalize.py --unmapped     # audit: which replies matched no rule
./canonicalize.py | python ../../data/lint.py --strict
```

| | before | after |
|---|---:|---:|
| distinct replies | 502 | **11** |
| pairs in a contradiction | 27.6% | **0%** |
| exact duplicate pairs | 10.9% | **0%** |
| accuracy ceiling | 87.0% | **100%** |
| examples per reply | 6 | **187** |
| charset | 40 | **19** |
| held-out overall | 33.8% | **40.0%** |
| held-out macro | 10.7% | **44.4%** |

## Why eleven

The first pass stopped at 21 replies — the distinctions the corpus *makes* —
and that was still too many for the evidence it *has*. Sweeping the merge, with
the same real held-out queries throughout:

| replies | per class | model overall | model macro |
|---:|---:|---:|---:|
| 21 | 87 | 29.9% | 16.8% |
| 16 | 114 | 28.9% | 20.4% |
| 12 | 151 | 31.8% | 27.0% |
| **11** | **165** | **37.8%** | **41.8%** |
| 8 | 227 | 39.3% | 32.7% |

Eleven is the turn. Below it overall keeps creeping up, but only because the
majority class is growing — by eight replies a constant guesser already scores
25.9% — while macro falls away again.

The lesson generalizes: the right vocabulary size is a property of how much
data you have, not of how many distinctions the writer felt like making.

The model also stopped emitting nonsense. With 502 replies it could generate
character sequences that were not answers at all — `thats funny` used to return
`WOTT`. Every output is now a real reply.

## Where it stands

```bash
python ../../data/baseline.py training-data.txt.gz --model model.npz
```

| | overall | macro |
|---|---:|---:|
| always answering `OK` | 23.4% | 9.1% |
| keyword table (~400 bytes) | 25.9% | 14.3% |
| **this model** | **40.0%** | **44.4%** |

Comfortably ahead of a word list now, on both measures, which it was not before.
Still the weakest of the three examples — `smalltalk` answers 80.6% — and the
remaining gap is phrasing redundancy, which `data/lint.py` reports:

| | examples per reply | phrasing redundancy |
|---|---:|---:|
| `examples/smalltalk/` | 149 | **0.74** |
| `examples/guess/` | 7,180 | 0.58 |
| `examples/tinychat/` | 187 | **0.55** |

Redundancy is how similar each query is to the nearest other query wanting the
same reply — how many different ways the data says each thing. `smalltalk` gets
149 crowdsourced paraphrases per reply. tinychat's queries are hand-written
one-offs, so there is less to generalize *from*, and no amount of vocabulary
merging changes that.

## Synthesising phrasings does not help

The obvious next move is to manufacture the missing paraphrases. It was tried
and it does nothing. Chat spelling swaps (`you`/`u`, `are`/`r`, `because`/`cuz`),
filler words, and typos, applied only to the training half and scored on
untouched real queries:

| | pairs | overall | macro |
|---|---:|---:|---:|
| no augmentation | 1,849 | 34.1% | 33.9% |
| +4 variants each | 7,814 | 33.7% | 34.0% |
| +8 variants each | 13,474 | 34.1% | 32.3% |

Flat. Those transformations add *orthographic* variation, and the trigram
encoder is already robust to that by construction — it is the property the
README advertises as typo tolerance. What is missing is *semantic* variation:
different words for the same idea. Generating that needs to know what the query
means, which is exactly what a crowd worker supplies and a regex does not.

It is also not that the answers are ambiguous. Scoring the model against any
reply the source ever gave for a query, rather than the one chosen, moves it by
nothing — only 3% of held-out queries have a second acceptable answer.

## Files

| | |
|---|---|
| `source-data.txt.gz` | the original hand-written corpus, unchanged |
| `canonicalize.py` | the 502 → 21 mapping, and the reasoning |
| `training-data.txt.gz` | generated; do not edit |
| `model.npz` | trained on the generated data |
