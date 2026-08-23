# CLINC150

Real crowdsourced utterances, vendored so the repo has a dataset nobody here
invented. This is the source for the [smalltalk](../../examples/smalltalk/)
example.

```bash
python data/clinc150/subset.py --list                    # all 150 intents
python data/clinc150/subset.py --recipe smalltalk > out.txt
python data/lint.py out.txt --strict
```

## Attribution

> **An Evaluation Dataset for Intent Classification and Out-of-Scope
> Prediction.** Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher
> Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael
> A. Laurenzano, Lingjia Tang, Jason Mars. EMNLP 2019.
>
> <https://github.com/clinc/oos-eval>

Licensed **CC BY 3.0** — see [LICENSE](LICENSE), vendored from the upstream
repository. That licence permits redistribution provided the work is credited
and changes are indicated.

**Changes made:** none to `data_full.json.gz`, which is the upstream
`data/data_full.json` gzipped (`gzip -9n`, so it is byte-reproducible) and
otherwise byte-identical. Everything downstream — uppercasing, the choice of
intents, the balancing, and the reply strings — happens in `subset.py`, and the
replies are ours: CLINC labels intents, not responses.

## What is in it

23,700 utterances: 150 intents × 150, plus 1,200 **out-of-scope** — things a
system genuinely cannot answer, collected the same way. That out-of-scope class
is the reason this dataset was chosen over better-known intent corpora. It gives
the catch-all real data instead of a list of nonsense strings somebody typed.

| | |
|---|---:|
| utterances | 23,700 |
| intents | 150 + out-of-scope |
| utterances per intent | 150 |
| out-of-scope utterances | 1,200 |
| median query length | 38 chars |
| queries over the 60-char truncation | 9.8% |

150 intents is far past what a 2-bit model can learn, so `subset.py` takes a
recipe: a set of intents plus the terse reply each maps to. Recipes are balanced
by construction, capped at the smallest in-scope intent — without that, `oos`
alone would be a third of the data and become the model's default answer.

## Why real data, measured

The point of vendoring rather than generating is that generated data flatters
itself. On a synthetic command-phrasing set built here earlier, a keyword table
matched the neural model almost exactly — the task did not need a model at all.
On CLINC it does not come close:

| | held-out accuracy |
|---|---:|
| keyword table (~1.7KB) | 59.4% |
| the model (39KB `.COM`) | **96.7%** |

Real phrasings vary in ways a word list cannot cover, which is exactly the
condition under which a fuzzy 128-bucket encoder earns its size. Reproduce the
baseline with `python data/baseline.py examples/smalltalk/training-data.txt.gz`.

## Adding a recipe

Add an entry to `RECIPES` in `subset.py`. Two things to watch:

- **Keep the reply set small and the letters shared.** The charset is the output
  layer; every distinct character costs 128 weights and forces a full retrain to
  add. The `smalltalk` recipe's 19 replies come to 22 characters.
- **Run `data/lint.py` on the result.** It will tell you if two intents are
  phrased too similarly to separate, before you spend an hour training.

Existing recipes should be treated as frozen — a shipped example depends on its
recipe producing the same data.
