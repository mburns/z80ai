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

**80.6% of held-out queries answered correctly**, in a 39KB `.COM`. (96.7% of
individual *characters* are right — that is the number `feedme` prints as
`ValChr`, and it is not the same claim. See [TRAINING.md](../../TRAINING.md).)

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

`tinychat` does the same job on hand-written data and answers 40.0% of held-out
queries correctly. Its vocabulary has since been cleaned up, which was not
enough: what it lacks is *phrasings per answer*, and that is exactly what
crowdsourced data provides. This example is the control.

It also settles a question the repo could not previously answer — whether the
model is doing anything a lookup table could not:

| | overall | macro |
|---|---:|---:|
| always answering `OLD` | 3.5% | 5.3% |
| keyword table (~1.7KB) | 59.4% | 62.1% |
| **this model** (39KB `.COM`) | **80.6%** | **80.7%** |

```bash
python data/baseline.py examples/smalltalk/training-data.txt.gz \
                        --model examples/smalltalk/model.npz
```

Overall and macro agree here because the classes are balanced by construction —
which is itself worth having, since on `guess` they differ by 38 points.

On generated data that gap collapses — a keyword table matched the model almost
exactly on a synthetic command set tried earlier. Real utterances vary in ways a
word list cannot cover, which is the condition under which a fuzzy 128-bucket
encoder is worth its size.

## The same 19 answers, in sentences

On an Agon there is an SD card, so the reply text does not have to be in the
model. `phrasebook.npz` is the same 19 intents and the same utterances, trained
as a classifier that emits an *index* into `TALK-PHR.DAT` instead of spelling a
reply out character by character.

```
> are you a robot
YES I AM A BOT
> what is the point of it all
NOBODY HAS TOLD ME YET
> whats the stock price of apple
I DO NOT KNOW THAT ONE
```

It is also **more accurate**, on the same labels and the same split:

| | on device | overall | macro |
|---|---:|---:|---:|
| keyword table | 1.7 KB | 59.4% | 62.1% |
| nearest centroid | 4.8 KB | 74.6% | 74.8% |
| character decoder (`model.npz`) | 35.9 KB | 80.6% | 80.7% |
| 1-NN over the whole corpus | 130.9 KB | 84.1% | 84.3% |
| **phrasebook (`phrasebook.npz`)** | **38.5 KB** | **86.6%** | **87.2%** |

Six and a half points of macro for two and a half kilobytes, because the decoder
was spending capacity on *spelling* `IM A BOT` rather than on deciding it. Three
training seeds land within 1.1 points of each other, so that is not the seed.

Note the row it passes on the way. Everywhere else in this project a plain
nearest-neighbour retriever over the training corpus beats the model once
storage is free — see [data/README.md](../../data/README.md). This is the one
place it does not: **87.2% against 84.3%, in less than a third of the bytes.**
Giving the model back the capacity it was spending on spelling is what closed a
gap that more weights and more epochs had not.

```bash
../../data/clinc150/subset.py --recipe smalltalk-phrasebook \
    | gzip -9n > phrasebook-data.txt.gz
../../classify.py --file phrasebook-data.txt.gz -o phrasebook.npz \
                  --hidden-sizes 384,256 --epochs 600
../../buildez80.py -m phrasebook.npz -o TALK-PHR.bin --phrases TALK-PHR.DAT
```

**The same swap does nothing for `tinychat`.** Its eleven replies are two and
three letters long, so there is almost no spelling to save: across three seeds
the classifier scores 38.3–46.2% macro against the decoder's 44.4%, a spread
that swallows the difference whole. The head pays in proportion to how much
reply vocabulary the decoder was carrying, and `tinychat` deliberately has none.

Nor does a bigger vocabulary help it. Retrained as a phrasebook at each level:

| replies | pairs per class | macro |
|---:|---:|---:|
| 11 | 186 | **38.3%** |
| 21 | 96 | 25.3% |
| 325 | 5 | 15.5% |

Reply text being free does not make evidence free. The collapse from 502 replies
to 11 was answering the second constraint, and that one has not moved.

## What it cannot do

The 19 replies are the entire vocabulary. Anything outside the 18 intents comes
back `IDK`, which is a deliberate improvement on the other examples — `guess`
has no way to express "I don't know" at all — but it is still a closed world.
The phrasebook makes those 19 answers longer, not more numerous.

CLINC ships 150 utterances per intent, so this trains on ~2,550 pairs against
`guess`'s 28,718. It works, but there is not much headroom; widening the recipe
is the way to grow it, not more epochs.
