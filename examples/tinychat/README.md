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

`canonicalize.py` collapses them onto **21 replies**. Almost all of the tail was
style rather than meaning — `YA` / `YUP` / `YE` / `BET` / `VALID` are one answer
wearing five hats, and so are `WAT` / `WUT` / `HUH?` / `???`. So it maps rather
than deletes: every query survives, only its reply changes.

```bash
./canonicalize.py --unmapped     # audit: which replies matched no rule
./canonicalize.py | python ../../data/lint.py --strict
```

| | before | after |
|---|---:|---:|
| distinct replies | 502 | **21** |
| pairs in a contradiction | 27.6% | **0%** |
| exact duplicate pairs | 10.9% | **0%** |
| accuracy ceiling | 87.0% | **100%** |
| largest class | 9.6% | 14.9% |
| charset | 40 | **23** |
| held-out macro accuracy | 10.7% | **26.6%** |

The model also stopped emitting nonsense. With 502 replies it could generate
character sequences that were not answers at all — `thats funny` used to return
`WOTT`. Every output is now a real reply.

## What this did not fix

Overall held-out accuracy is **28.4%**, against 33.8% before. Cleaning the
vocabulary more than doubled the *macro* score and made the data sound, but the
model still gets most queries wrong, and it is only eight points above a
constant guesser (20.4%).

The reason is measurable, and `data/lint.py` now reports it:

| | examples per reply | phrasing redundancy |
|---|---:|---:|
| `examples/smalltalk/` | 149 | 0.74 |
| `examples/guess/` | 7,180 | 0.58 |
| **`examples/tinychat/`** | **96** | **0.54** |

Phrasing redundancy is how similar each query is to the nearest other query
wanting the same reply — how many different ways the data says each thing. A
model generalizes by seeing one intent phrased several ways. `smalltalk` gets
149 crowdsourced paraphrases per reply; `guess` has only moderate redundancy but
7,180 examples per reply to make up for it. tinychat has neither: its queries are
hand-written one-offs, so there is little to generalize *from*.

It is not that the answers are ambiguous. Checking whether the model's reply is
any reply the source ever gave for that query, rather than the one chosen, moves
the score by nothing at all (28.4% either way) — only 3% of held-out queries have
a second acceptable answer.

So the fix that remains is more phrasings per reply, which means collecting or
borrowing them. [`examples/smalltalk/`](../smalltalk/) is what that looks like
when the phrasings come from crowd workers.

## Files

| | |
|---|---|
| `source-data.txt.gz` | the original hand-written corpus, unchanged |
| `canonicalize.py` | the 502 → 21 mapping, and the reasoning |
| `training-data.txt.gz` | generated; do not edit |
| `model.npz` | trained on the generated data |
