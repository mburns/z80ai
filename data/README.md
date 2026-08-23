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
add. `tinychat` has 502 responses and a 40-character charset; it is fighting the
architecture.

**The accuracy ceiling.** If the same query appears with two different
responses, no model can be right about both. `tinychat` is capped at 87.0% and
`guess` at 97.8% before training starts. Without this number a model that has
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

`data/baseline.py` makes that question routine — it builds a keyword table from
the training half and scores it on the same held-out split, so you can see what
the model is actually adding:

| | majority answer | keyword table | model |
|---|---:|---:|---:|
| `examples/smalltalk/` | 3.5% | 59.4% | **96.7%** |
| `examples/guess/` | 57.7% | 76.6% | **94.1%** |
| `examples/tinychat/` | 8.5% | 10.5% | 61.0% |

Read those rows carefully. `smalltalk` is the clean case: real phrasings, a word
list gets nowhere near, the model earns its 39KB. `guess` is genuinely learned
but a keyword table already gets three quarters of the way. `tinychat`'s table
scores 10.5% because 502 responses defeat it — but the model only reaches 61%,
so the task is beating both.

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
