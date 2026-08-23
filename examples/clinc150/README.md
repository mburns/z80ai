# clinc150 — every intent, answered in sentences

The other three examples spell their replies one output neuron at a time, so a
reply costs the model capacity and every distinct character costs 128 weights.
That is why `smalltalk` ships 19 of CLINC150's 150 intents and answers
`IM A BOT` rather than a sentence.

This one is for an Agon with an SD card. The model picks an *index*; the text
lives in `PHRASES.DAT` on the card. Reply length becomes free, so all 150
intents ship with full-sentence answers — plus CLINC's own out-of-scope class,
which is what it says when it does not know.

```
> what is my checking balance
YOUR BALANCE IS FOUR HUNDRED DOLLARS
> is my flight going to be on time
YOUR FLIGHT IS ON TIME
> how do i jump start a car
RED TO POSITIVE THEN BLACK TO GROUND
> what is the airspeed velocity of a swallow
I DO NOT KNOW THAT ONE
```

## Running it

```bash
python build.py --model examples/clinc150/model.npz --target ez80 --output CLINC.bin
```

That writes **two** files: `CLINC.bin` and `PHRASES.DAT`. Copy both onto the
card, in the same directory, and run `CLINC` from the MOS prompt. The binary
prints a legible error and stops if the phrase file is missing, rather than
answering out of whatever happened to be in the buffer.

This example is Agon-only. A CP/M or Spectrum build has nowhere to keep the
replies, and `libnn.emit_argmax` counts outputs in a byte, so 151 of them would
not survive the trip.

## Training it

```bash
python data/clinc150/subset.py --recipe clinc150 | gzip -9n > training-data.txt.gz
python classify.py --file training-data.txt.gz -o model.npz \
                   --hidden-sizes 384,256 --epochs 900
```

`classify.py` is seeded, unlike `feedme.py`, so this reproduces.

## What it is worth

Held-out, whole replies, on `libdata.split_pairs(pairs, 0.1, seed=0)` — the
same split `data/baseline.py` scores everything else on:

| | on device | overall | macro |
|---|---:|---:|---:|
| keyword table | 6.4 KB | 36.5% | 36.9% |
| nearest centroid | 37.8 KB | 73.1% | 73.0% |
| **this model** | **47.0 KB** | **81.3%** | **81.9%** |
| 1-NN word Jaccard | 1.3 MB | 82.7% | 83.1% |
| 1-NN trigram | 1.3 MB | 84.4% | 84.9% |

Read the bottom three rows together. Both retrievers over the training set are
**more accurate than the model** — and need twenty-eight times the storage to
be. On a machine with a card that is a real option and it belongs on the page;
the model's claim is accuracy per byte, not accuracy.

Reproduce it with:

```bash
python data/baseline.py examples/clinc150/training-data.txt.gz \
                        --model examples/clinc150/model.npz
```

## The mixture of experts that is not here

The obvious way to cover 150 intents on a machine that can page memory is a
router plus ten domain experts, one paged in per response. It was built and
measured before this example was written, and it lost:

| | macro |
|---|---:|
| router (11-way domain choice) | 81.8% |
| one expert, given the right domain | ~91% |
| **router × expert, end to end** | **75.4%** |
| a perfect router × the same experts | 89.4% |
| **one flat 151-way model (this one)** | **81.9%** |

End-to-end accuracy is a product, and the router is the weaker factor. Getting
the domain right is not an easier problem than getting the intent right — it is
the same problem with the evidence thrown away — and the 18% it gets wrong is
unrecoverable, because the expert it hands off to does not contain the right
answer at all.

The oracle row says the architecture is not wrong in principle: with perfect
routing it beats everything here. There is just no router worth having, and a
flat model needs none.

`data/clinc150/subset.py --recipe clinc-router` and `--domain <name>` still
emit the training files, so the measurement can be repeated.

## Sizing

151 replies, `128 → 384 → 256 → 151`, 186,112 weights.

| kernel | image | fits Agon SRAM |
|---|---:|---|
| `column` | ~552 KB | **no** — over the 508 KB ceiling |
| `row` | 369,484 | yes — and what `auto` picks |
| `compact` | 198,701 | yes |

`column` is refused for a phrasebook anyway — its query hoisting amortizes over
the steps of a response and there is only one — but the sizes are worth knowing:
this is the first shipped model large enough that the fastest kernel would not
have fitted regardless.

Source: [CLINC150](../../data/clinc150/), Larson et al., EMNLP 2019, CC BY 3.0.
The utterances are theirs; the replies are ours.
