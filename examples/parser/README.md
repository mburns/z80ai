# parser — where word order carries the meaning

A container command parser. `PUT KEY IN BOX` is `OK`; `PUT BOX IN KEY` is `NO`.
Both contain nearly the same bag of trigrams, so this is the task the flat query
encoder cannot represent and the position-aware one can.

It exists to measure that difference, not to be clever: 16 objects in four size
classes, four phrasings each, and the answer is whether the first object fits
inside the second.

```bash
./gendata.py            # regenerate the corpus
./compare.py            # train both encoders and score them
./run.sh                # train, build and run a .COM
```

## The measurement

Object *pairs* are held out, not just examples, so the eval set contains
combinations the model never saw in any phrasing. Memorising training pairs
earns nothing there.

```
256 -> 96 -> 64 -> 4, 400 epochs, identical data and seed

           encoder   char acc    train     eval
    flat (current)     99.3%    97.9%    85.9%
  8 position bands    100.0%   100.0%    98.4%
   always-majority         -        -    56.2%
        word table         -   100.0%   100.0%
```

Errors on unseen pairs fall by 89%. See [ENCODING.md](../../ENCODING.md) for why
this does not generalise to the paraphrase-style examples, where the same change
makes things worse.

## The fourth row is not a model

`table.py` is a verb set, a noise-word set, a preposition set, and the sizes
written down. It scores 100% on both splits, and the reason is the whole point:
**held-out pairs cost it nothing, because it never needed to see a pair.** The
model has to infer from examples what the table was told.

That is unfair in exactly the way the comparison is for. An author of an
Interactive Fiction writes the world down — a key is small, a barn is large —
and does not train it. The question the second scope of
[#62](../../../issues/62) turns on is whether `PUT X IN Y` should go through a
model at all, and this is what the authoring step buys.

### What a word it was never given does

The property that decides it, and it is not accuracy:

```
words neither was given, 12 commands:
                flat  'OK' x8, 'NO' x4
    8 position bands  'NO' x7, 'OK' x5
          word table  declined 12/12, naming the word it did not know
```

`PUT ZORKMID IN BOX` gets a confident `OK` or `NO` from both models and nothing
at all from the table. A player types a noun the author never wrote about every
few turns, and *"I don't know the word 'zorkmid'"* is the only useful thing to
say back — which a bare argmax has no way to produce, the same gap
`data/silo/` closed by [teaching a refuse class](../../data/silo/README.md).

The model is not wrong here in a way more training would fix. It is being asked
a question that has no answer and returning its best guess, because returning a
best guess is the only thing it can do.

## Building it

`--position-bands` is recorded in the model file, so the build picks it up
automatically and every backend emits a matching tokenizer:

```bash
./gendata.py
cat training-data.txt | ../../feedme.py --position-bands 8 --epochs 400 \
    --hidden-sizes 96,64 -o parser.pt
../../exportmodel.py -m parser.pt -o parser.npz
../../build.py --model parser.npz --output PARSER.COM
```

Then `iz-cpm PARSER.COM "PUT KEY IN BOX"` answers `OK`, and
`iz-cpm PARSER.COM "PUT BOX IN KEY"` answers `NO`.
