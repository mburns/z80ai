# smalltalk

A chatbot trained on **real crowdsourced utterances** rather than invented ones.

```
> hello there
HI
> are you a robot
IM A BOT
> whats your name
IM CHAT
> tell me something funny
HA HA
> what is the point of it all
WHO KNOWS
> whats the stock price of apple
IDK
> thanks a lot
NO PROB
```

**96.7% on held-out queries**, in a 39KB `.COM`.

```bash
./run.sh          # retrain, build and run under CP/M
./run-zx.sh       # ZX Spectrum .TAP
./run-agon.sh     # Agon Light .bin
```

## Where the data comes from

[CLINC150](../../data/clinc150/) — 23,700 utterances collected from crowd
workers for intent-classification research, CC BY 3.0. This example uses the
`smalltalk` recipe: 18 chatbot-register intents (`are_you_a_bot`, `tell_joke`,
`meaning_of_life`, `where_are_you_from`, yes/no/maybe, …) balanced to 149 each,
plus CLINC's own **out-of-scope** class mapped to `IDK`.

The intents are CLINC's; the replies are ours. Regenerate with:

```bash
../../data/clinc150/subset.py --recipe smalltalk | gzip -9n > training-data.txt.gz
```

## Why it is here

`tinychat` does the same job on hand-written data and reaches 61.0% on held-out
queries — its 502 distinct responses are memorized rather than learned. This
example is the control: same task, same architecture, real phrasings and a
disciplined 19-reply vocabulary.

It also settles a question the repo could not previously answer — whether the
model is doing anything a lookup table could not:

| | held-out accuracy |
|---|---:|
| always answering `OLD` | 3.5% |
| keyword table (~1.7KB) | 59.4% |
| **this model** (39KB `.COM`) | **96.7%** |

Check it yourself: `python data/baseline.py examples/smalltalk/training-data.txt.gz`.

On generated data that gap collapses — a keyword table matched the model almost
exactly on a synthetic command set tried earlier. Real utterances vary in ways a
word list cannot cover, which is the condition under which a fuzzy 128-bucket
encoder is worth its size.

## What it cannot do

The 19 replies are the entire vocabulary. Anything outside the 18 intents comes
back `IDK`, which is a deliberate improvement on the other examples — `guess`
has no way to express "I don't know" at all — but it is still a closed world.

CLINC ships 150 utterances per intent, so this trains on ~2,550 pairs against
`guess`'s 28,718. It works, but there is not much headroom; widening the recipe
is the way to grow it, not more epochs.
