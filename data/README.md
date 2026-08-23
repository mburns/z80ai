# Training data

```bash
python data/lint.py examples/guess/training-data.txt.gz      # is the data sound?
python data/baseline.py examples/guess/training-data.txt.gz  # does it need a model?
python data/lint.py my-data.txt --strict     # exits non-zero if anything is flagged
```

Vendored source data lives in [clinc150/](clinc150/).

## Check the data before you train it

`data/lint.py` reports what a dataset will do to a 2-bit model. It takes a
second; training takes an hour, and most of what goes wrong is visible up front.

```
pairs                               28,718
unique queries                      18,087
unique responses                         4
charset                                 11  'ABEIMNOSWY' + EOS
output layer weights                 1,408
exact duplicate pairs               10,077  (35.1%)
pairs in a contradiction             2,103  (7.3%)
accuracy ceiling                    97.8%
```

The two numbers that matter most are the ones nobody counts:

**Unique responses**, not line count. The character decoder is really a label
decoder — a model with four responses is doing four-way classification however
many lines you feed it. Each additional response costs capacity, and each
additional *character* costs 128 output weights and forces a full retrain to
add. `tinychat` used to have 502 responses and a 40-character charset, which is
how a dataset fights the architecture; it is down to 21 and 23.

**The accuracy ceiling.** If the same query appears with two different
responses, no model can be right about both. `guess` is capped at 97.8% before
training starts, and `tinychat` was capped at 87.0% until its replies were
canonicalized. Without this number a model that has
learned everything learnable looks like it is underperforming, and the fix looks
like "train longer" when it is "fix the data".

## What this architecture is for

A query is hashed into 128 trigram buckets and classified. That is good at
**responding plausibly to open-ended text**, and it tolerates typos and
paraphrase for free. It is *not* a good way to build something with a crisp
right answer.

That distinction is easy to get wrong. A text-adventure command parser looks
like an ideal fit — small label set, keyword-driven, short queries — and a model
trained on generated command phrasings scores 99.8% on a held-out-query split.
But:

- a keyword table built from the same data scores 94.9% in 2.5KB against the
  model's 36KB, and a hand-written verb list gets 100%;
- hold out whole *verbs and prefixes* instead of whole queries, so the
  evaluation uses wording the model never saw, and the model drops to 44.7%.

Both numbers say the same thing: on a task with a correct answer, write the
parser. The model's only edge was fuzzy matching on unseen words, and it was not
good enough to use.

([`examples/parser/`](../examples/parser/) is a parser and is not a
counterexample. It exists to measure what the position-aware encoder can
represent that the flat one cannot — `PUT KEY IN BOX` against `PUT BOX IN KEY` —
which is an experiment about the encoder, not a claim that this is a good use of
the model.)

`data/baseline.py` makes that question routine — it builds a keyword table from
the training half and scores it on the same held-out split, so you can see what
the model is actually adding:

```bash
python data/baseline.py examples/guess/training-data.txt.gz \
                        --model examples/guess/model.npz
```

Held-out, whole responses, on the shipped models:

| | majority answer | keyword table | model |
|---|---:|---:|---:|
| `examples/smalltalk/` | 3.5% / 5.3% | 59.4% / 62.1% | **80.6% / 80.7%** |
| `examples/guess/` | 57.7% / 25.0% | 76.6% / 48.4% | **81.3% / 86.5%** |
| `examples/tinychat/` | 23.4% / 9.1% | 25.9% / 14.3% | **40.0% / 44.4%** |

Each cell is **overall / macro**, and the difference between them is the point.
Overall weights every pair equally, so a dominant answer inflates it: `guess` is
58% `NO`, and a constant guesser scores 57.7% overall for knowing nothing. Macro
averages over distinct answers, where the same guesser gets 25.0%.

Read `guess` with that in mind. On overall the keyword table looks close —
76.6% against the model's 81.3%. On macro it is not close at all: 48.4% against
86.5%, because the table wins by defaulting to `NO` on the majority class and
**never says `WIN` at all**. It cannot end the game. Per class:

| | NO | MAYBE | YES | WIN |
|---|---:|---:|---:|---:|
| keyword table | 99.3% | 46.1% | 48.2% | **0.0%** |
| model | 80.6% | 71.0% | 94.5% | **100%** |

`tinychat` shows what vocabulary size is worth. Collapsing its 502 replies onto
11 took macro from 10.7% to 44.4% — but the intermediate stop at 21 replies only
reached 26.6%, and 21 was already a defensible reading of what the corpus
distinguishes. The right size is a property of how much data you have, not of
how many distinctions the writer felt like making; see
[examples/tinychat/](../examples/tinychat/) for the sweep.

Two numbers explain why, and `lint.py` now prints both:

| | examples per response | phrasing redundancy |
|---|---:|---:|
| `examples/smalltalk/` | 149 | 0.74 |
| `examples/guess/` | 7,180 | 0.58 |
| `examples/tinychat/` | 187 | 0.55 |

**Phrasing redundancy** is how similar each query is to the nearest other query
wanting the same answer — how many ways the data says each thing. A model
generalizes by seeing one intent phrased several times. `smalltalk` gets 149
crowdsourced paraphrases per answer. `guess` has only moderate redundancy but
7,180 examples per answer to compensate. `tinychat` has neither: its queries are
hand-written one-offs, so there is little to generalize *from*.

A clean vocabulary is necessary and not sufficient. Volume without redundancy
does not help either — what a model needs is the *same thing said several ways*.

So look for tasks where **being approximately right is acceptable** — where a
"wrong" answer still reads as a plausible one. `guess` (four answers to
open-ended questions) and `tinychat` (terse conversational replies) are both
that shape, which is why they are the shipped examples.

Within that, aim for:

- **A small closed response set** — 4 to ~30. This is the real complexity
  parameter.
- **Many phrasings per response.** Volume belongs here, not in more responses.
- **Short, keyword-bearing queries.** Long sentences that differ subtly ("bigger
  than a cat" vs "smaller than a cat") share most of their trigrams.
- **Responses that share characters.** One stray `/` in one line costs an output
  neuron.
- **An explicit catch-all.** Out-of-distribution input produces *something*;
  training an `IDK` class decides what.

## A warning about templated data

If your data is generated from templates, the `--val-frac` split will flatter
it. The split holds out unique *queries*, so a held-out `PLEASE TAKE THE LAMP`
still has `TAKE THE LAMP` and `OK TAKE THE LAMP` in the training half, and the
score measures interpolation inside your own grammar rather than generalization.

`guess` is templated this way (`can it cut you` / `can this cut you` / `can that
cut you`), which is part of why its train and held-out numbers are nearly
identical.

To find out what a model has really learned, hold out a whole *dimension* — every
query using a particular verb, or a particular prefix — and evaluate on that.

## Generating data

`examples/guess/gendata.py` (Ollama or an `ANTHROPIC_API_KEY`) and
`examples/tinychat/genpairs.py` (three local Ollama models) both bootstrap data
from an LLM. Neither is seeded and neither one's output is checked in, so the
data behind the shipped models cannot be reproduced. If you use them, commit
what comes out.

Whatever produced it, run `data/lint.py` on the result.
