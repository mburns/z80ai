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
```

Errors on unseen pairs fall by 89%. See [ENCODING.md](../../ENCODING.md) for why
this does not generalise to the paraphrase-style examples, where the same change
makes things worse.

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
