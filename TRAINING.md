# Training Z80-μLM

```bash
pip install torch                          # training only; building needs just NumPy
python data/lint.py my-data.txt --strict
cat my-data.txt | ./feedme.py --epochs 150 --chat
./exportmodel.py -m command_model_autoreg.pt -o model.npz
./build.py -m model.npz -o CHAT.COM
```

Requirements: Python 3.10+, PyTorch, NumPy. No emulator needed — `libz80emu.py`
runs the generated code in-process.

`--chat` opens a REPL after training. Ctrl+C stops training gracefully and keeps
the best checkpoint, and a later run continues from it as long as the
architecture and charset match.

**Export to `.npz` when you are done.** `build-examples.sh`, `verify_artifacts.py`,
`bench.py` and the release all read `model.npz`; the `.pt` checkpoint is only an
intermediate. Skipping the export leaves them on the old weights.

## Data format

Pipe-separated, case-insensitive, `#` for comments:

```
# movement
go north|GO N
head north|GO N
n|GO N
take the lamp|TAKE
asdf|IDK
```

Both sides are uppercased. Queries shorter than 2 characters are dropped;
queries over 60 and responses over 50 are **truncated, not rejected**, so data
past those limits silently changes meaning.

## Read this before collecting data

Two numbers decide whether a model can learn your dataset, and neither is the
line count. `data/lint.py` reports both.

### Distinct responses, not lines

The character decoder is really a **label decoder**. A model with four responses
is doing four-way classification no matter how many lines you feed it. Each
additional response costs capacity; each additional *character* costs 128 output
weights and forces a full retrain to add, because the output layer changes shape.

Aim for **4 to ~30 distinct responses**, 1–2 words each. Put your volume into
*phrasings per response*, not into more responses.

```
are you smart|YES        smart bot|YES        youre smart|YES
```

### The accuracy ceiling

If the same query appears with two different responses, no model can be right
about both. That caps accuracy before training starts:

| dataset | responses | ceiling |
|---|---:|---:|
| `examples/guess/` | 4 | 97.8% |
| `examples/tinychat/` | 11 | 100% |

Without this number a model that has learned everything learnable looks like it
is underperforming, and the fix looks like "train longer" when it is "fix the
data".

### Word order

By default the query encoder is order-insensitive: the same trigram hashes to
the same bucket wherever it appears. That is what makes paraphrase matching work
— and a wall for anything that parses commands, where `PUT KEY IN BOX` and `PUT
BOX IN KEY` mean different things and encode nearly identically.

`--position-bands 8` seeds each trigram's hash with where it appeared, making
the encoding order-aware. It is recorded in the model file, so a build can never
use a tokenizer the model was not trained with. It is a real trade rather than a
free upgrade — see [ENCODING.md](ENCODING.md).

**"It's a command parser" is not the condition for turning it on.** Measured on
a generated command-phrasing set, bands *cost* four points — 95.8% against 99.8%
flat — because the signal there was "a keyword appears somewhere" and the
queries carried variable-length prefixes (`take the lamp`, `please take the
lamp`, `i want to take the lamp`). Bands scatter the same verb across several
buckets, so the model has to learn each shift separately.

Bands pay when *competing arguments* must be told apart — `PUT KEY IN BOX` vs
`PUT BOX IN KEY`. Decide it while designing the data: if two queries differ only
by word order and want different answers, the flat encoder cannot separate them,
and `data/lint.py` reports them as colliding.

### Also worth knowing

- **Balance.** No response should be more than ~40% of the data. A small model
  falls back to the majority class whenever it is unsure, so a dominant label
  becomes its default answer. `guess` ships `NO` at 56.2%, and four of six
  nonsense queries (`ASDF QWERTY`, `BANANA PHONE`, …) come back `NO`.
- **Short, distinctive queries.** Long sentences differing subtly — "bigger than
  a cat" vs "smaller than a cat" — share most of their trigrams, and the
  distinguishing signal gets diluted across 128 buckets.
- **Train a catch-all.** Out-of-distribution input produces *something*
  regardless; an explicit `nonsense → IDK` class decides what. Note that this
  only works if your charset can spell it — `guess`'s cannot.
- **Watch the charset.** One line containing a `/` costs an output neuron for
  the whole model. `data/lint.py` flags characters appearing in three or fewer
  pairs.

## Reading the training output

```
Loaded 28718 pairs → 25823 train / 2895 val → 1 chunks of 25823
Ceiling from contradictory labels: train 97.7%, val 98.4%
  Epoch 120: CE=0.1584, Acc=93.6%, IntAcc=93.6%, ValChr=92.9%, ValRsp=72.4%, ValMacro=56.2%
```

| column | measured on | counts |
|---|---|---|
| `Acc` | training data | characters, float weights. Ignore it. |
| `IntAcc` | training data | characters, quantized |
| `ValChr` | **held-out** | characters |
| `ValRsp` | **held-out** | whole responses |
| `ValMacro` | **held-out** | whole responses, averaged per answer |

Three traps, in increasing order of how much they will mislead you.

**`IntAcc` is not `Acc`.** Float accuracy can sit at 99% while integer accuracy
is 60%; only the integer number survives to the Z80.

**`ValChr` is not `ValRsp`.** `ValChr` scores each next character against the
*true* prefix — teacher forcing — so a model that gets the first character wrong
is still credited for the rest. Generation has no true prefix: one wrong
character and the context feeding every later step is wrong too. The gap is
large and it is not a constant:

| dataset | `ValChr` | `ValRsp` |
|---|---:|---:|
| `examples/guess/` | 96.2% | 81.3% |
| `examples/smalltalk/` | 96.7% | 80.6% |
| `examples/tinychat/` | 84.0% | 40.0% |

**`ValRsp` is not `ValMacro`.** Overall response accuracy weights every pair
equally, so a dominant answer inflates it. `guess` is 58% `NO`: answering `NO`
to everything scores 57.7%. Macro averages over distinct answers, where the same
guesser scores 25.0%. If your classes are unbalanced — and they usually are —
`ValMacro` is the one to watch.

`--save-best` selects on `ValMacro` for exactly this reason. The two disagree in
practice: in the run below `ValChr` rises almost monotonically and peaks at the
last epoch, while `ValMacro` peaks at epoch 70 and then drifts down. Selecting on
characters would keep the wrong checkpoint.

A small train/held-out gap is necessary but not sufficient. The split holds out
unique *queries*, so templated data — where a held-out query has near-twins in
the training half — shows a small gap without generalizing. `guess` is templated
that way. To find out what a model has really learned, hold out a whole
dimension (every query using one verb, or one prefix); see
[data/README.md](data/README.md).

`--val-frac 0` restores the old training-set-only behaviour, and then the
summary says so.

## Quantization-aware training

The model trains with progressive quantization and never in pure float:

```python
quant_temp = 0.3 + 0.7 * min(1.0, epoch / (epochs * 0.8))
```

30% quantized at epoch 0, fully quantized at 80% of the run, then refinement.
Starting in float lets the model find solutions that collapse when quantized —
if `IntAcc` suddenly drops, that is what happened.

Weights land on `{-2, -1, 0, +1}` — four values, asymmetric, not `±2`. The scale
is a per-layer 95th percentile of `|w|`, recomputed each forward pass. Gradients
pass through a straight-through estimator; an overflow penalty (`libqat.py`)
discourages accumulator values the Z80's 16 bits could not hold.

**Give it time after `QTemp` reaches 1.0.** The model needs to refine once fully
quantized; stopping the moment it gets there wastes the run.

## Architecture

Default `256 → 256 → 192 → 128 → charset_size`, ~143K weights at 2 bits ≈ 36KB
of information.

| component | size | purpose |
|---|---:|---|
| query buckets | 128 | trigram hash of the user's query |
| context buckets | 128 | hash of the last 8 characters generated |
| hidden | 256, 192, 128 | `--hidden-sizes 256,192,128` |
| output | `len(charset)` | one neuron per character, derived from responses |

Wider layers mean more capacity, a larger binary and slower inference. The Z80
backends count neurons in `B`, so `DJNZ` caps a layer at 256; anything wider
builds only for eZ80 (`buildez80.py`) — see [EZ80.md](EZ80.md). `feedme.py`
warns when you cross that line and the Z80 builders raise rather than
mis-assemble.

Memory, for the default shape: 65,536 + 49,152 + 24,576 weights in the hidden
layers, plus `128 × len(charset)` for the output. About 35KB packed.

## Debugging

| symptom | cause |
|---|---|
| Garbage output | `IntAcc` too low. Check it, not `Acc`. Quantization only reaches full strength 80% of the way through the run, so give it more epochs — or shrink the vocabulary. |
| Always the same answer | Class imbalance — one response above ~40%. Run `data/lint.py`. |
| High `IntAcc`, low `ValRsp` | Memorizing. Too many distinct responses, or too few phrasings each. |
| High `ValChr`, low `ValRsp` | Normal to a degree — but a large gap means the first character of a response is often wrong, and everything after it follows. |
| High `ValRsp`, low `ValMacro` | Riding one dominant answer. Balance the classes; `data/lint.py` flags it. |
| Accuracy plateaus below 100% | Check the ceiling. Contradictory labels may make the rest unreachable. |
| Similar inputs, same output | Trigram collision. Use shorter, more distinctive phrasings. |
| Binary too big | Fewer/narrower layers, or a smaller charset. Check what `data/lint.py` says about rare characters. |

## Example run

```
$ gzip -dc examples/guess/training-data.txt.gz | ./feedme.py --epochs 120 --save-best

============================================================
Loading training data...
Loaded 28718 pairs → 25823 train / 2895 val → 1 chunks of 25823
Ceiling from contradictory labels: train 97.7%, val 98.4%
Epochs per chunk: 120
============================================================
Charset (11 chars): 'ABEIMNOSWY' + EOS
Validation: 2895 pairs → 11209 character examples
No checkpoint found, starting fresh
Model: 256 → 256 → 192 → 128 → 11
Parameters: 141,259

--- Chunk 1/1: 25823 pairs ---
Generated 99762 character examples
  Epoch 10:  CE=1.1247, Acc=61.9%, IntAcc=64.1%, ValChr=64.7%, ValRsp=8.2%,  ValMacro=3.5% *BEST*
  Epoch 40:  CE=0.2233, Acc=91.1%, IntAcc=89.8%, ValChr=89.7%, ValRsp=60.1%, ValMacro=43.2% *BEST*
  Epoch 70:  CE=0.1801, Acc=92.6%, IntAcc=91.7%, ValChr=91.5%, ValRsp=67.1%, ValMacro=60.8% *BEST*
  Epoch 90:  CE=0.1670, Acc=93.3%, IntAcc=93.3%, ValChr=92.8%, ValRsp=72.1%, ValMacro=55.6%
  Epoch 110: CE=0.1868, Acc=92.0%, IntAcc=93.0%, ValChr=92.5%, ValRsp=70.8%, ValMacro=58.2%
  Epoch 120: CE=0.1584, Acc=93.6%, IntAcc=93.6%, ValChr=92.9%, ValRsp=72.4%, ValMacro=56.2%
Saved best (epochs: 120, best: 60.8% @ 70)

============================================================
Finished: 1/1 chunks, 120 total epochs
Best held-out ValMacro: 60.8% at epoch 70
============================================================
```

(Every tenth epoch shown; nothing else altered.)

Four things to notice.

**The three held-out numbers do not move together.** At epoch 120, `ValChr` is
92.9% and `ValRsp` is 72.4% — a fifth of the responses are wrong despite nearly
all the characters being right. Quote `ValChr` at your peril.

**`ValMacro` peaks in the middle.** Best at epoch 70, then down, while `ValChr`
keeps climbing to the last epoch. `--save-best` kept epoch 70; selecting on
characters would have kept epoch 120 instead.

**This run is not as good as the shipped model.** `examples/guess/model.npz`
scores 86.5% macro against this run's 60.8%, so 120 epochs from scratch is not
enough for this dataset — check a new model against the old one before replacing
it (`python data/baseline.py <data> --model <npz>`).

**Training is not seeded** — there is no `torch.manual_seed` anywhere — so the
same command on the same data varies run to run. Do not read a small difference
as a real one.

The ceiling line is worth keeping in view too: contradictory labels cap this
data at 97.7%, so the last 2.3 points are a data problem, not a training one.

See [data/README.md](data/README.md) for what makes a dataset suit this
architecture, and for why a good score on generated data can be misleading.
