#!/usr/bin/env python3
"""
Build the Agon card for the silo, classifier and all.

    python data/silo/buildcard.py                    # dist/SILO.{bin,IDX,DAT,GRF}
    python data/silo/buildcard.py --skip-train       # reuse the model already built

Four steps, and the only reason this exists rather than a line in a README is
the first one: `buildwikigraph.paths_for` reads `libgraph.CLIMB` to turn a
phrase like `founding_father` into a step, and the silo's climbs are registered
by importing `generate`. A card built without that import has three inert rows
in its path table and answers those questions with silence - which looks
exactly like a corpus that has no answer.

    1  import generate            registers in_section, in_silo, founding_father
    2  relationpaths.py           templated questions -> paths
    3  classify.py                the phrasebook classifier, 2-bit weights
    4  buildwikisearch.py         .IDX, .DAT, .GRF and the eZ80 binary

Then `python benchwiki.py --card dist/SILO` runs it in the emulator and counts
what a question costs.

## Two models, and why

Step 3 runs twice. The **shipped** classifier is trained on every phrasing,
because a card that has never seen "who is X's father" answers it with the
grandfather and there is no reason to hobble it. The **scored** classifier is
trained without three phrasings per path and measured on exactly those, and it
is thrown away.

Training the shipped card on the reduced set instead is the mistake this
comment exists to stop anyone repeating. It was made here first: the card
answered "who is alexander e wong's father" with his *grandfather*, because
that wording was one of the three held out and the classifier had never met it.

Step 3 also prints a validation score in the high nineties. **Do not quote
it.** It is a split over unique queries, and these questions are templated, so
a held-out "who is X's father" still has "who is Y's father" in the training
half. The number this script prints at the end is the one to quote, and the
gap between them is most of the point - see `data/silo/README.md`.
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate  # noqa: F401  - registers the silo's climbs into libgraph.CLIMB
import relationpaths
from schema import SOURCE

import buildwikisearch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable
    from typing import TextIO

REPO = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO / "data" / "silo.db"

#: Phrasings per path withheld from training, out of the twelve `relationpaths.py`
#: writes. A quarter of the grammar, so the held-out score is a question about
#: generalisation rather than a rounding error.
HELD_OUT = 3


def write_questions(db: Path, out: Path, held_out: int) -> tuple[Path, Path, Path]:
    """(everything, the reduced training half, the phrasings it never saw).

    Three files rather than two, because the shipped card and the measurement
    want different training sets and conflating them is what shipped a card
    that answered "who is X's father" with the grandfather.
    """
    full = out / "silo-relations.txt"
    reduced = out / "silo-relations-reduced.txt"
    unseen = out / "silo-held-out.txt"
    for path, argv in (
            (full, ["--db", str(db)]),
            (reduced, ["--db", str(db), "--held-out-templates", str(held_out)]),
            (unseen, ["--db", str(db), "--held-out-templates", str(held_out),
                      "--emit", "held-out"])):
        with path.open("w") as handle:
            _capture(relationpaths.main, argv, handle)
    return full, reduced, unseen


def _capture(main: Callable[[], None], argv: list[str], handle: TextIO) -> None:
    """Run a `main()` that prints, with stdout pointed at a file.

    `relationpaths.py` is a filter - it writes the dataset to stdout, the way the
    Wikipedia one does - and reimplementing it as a library call so this script
    could import it would leave two ways to produce the questions, which is one
    more than a dataset should have.
    """
    saved = sys.argv
    sys.argv = ["relationpaths.py", *argv]
    try:
        with contextlib.redirect_stdout(handle):
            main()
    finally:
        sys.argv = saved


#: The classifier is the most expensive thing on the card - `benchcard.py`
#: measured it at 78.1% of a query against the graph walk's 1.9% - so its size
#: is a decision worth recording rather than a default worth inheriting.
#: Swept over the silo's twenty paths, trained accuracy against unseen
#: phrasings averaged over three held-out splits:
#:
#:     hidden    weights   trained   unseen
#:     256,192    85,760     96.1%    45.0%     classify.py's default
#:     128,96     30,592     95.3%    45.8%     <- this
#:     64          9,472     90.1%    43.8%
#:     32          4,736     76.1%    39.1%
#:     (none)      2,560     81.4%    37.8%
#:
#: 128,96 gives up 0.8 points of trained accuracy for 2.8x fewer weights, and
#: is no worse on unseen phrasings than the default is - that column is noise
#: at this width either way. Below it the drop is real: 64 costs six points and
#: 32 costs twenty, and a 32-wide bottleneck is *worse* than no hidden layer at
#: all, which is the shape of a layer too narrow to carry 128 buckets rather
#: than a model too small to learn.
HIDDEN_SIZES = "128,96"

#: Trigram buckets the encoder hashes into. 128 was the repository default from
#: its first commit and was never swept; `tools/bucket_sweep.py` did, and it is
#: worth **7.5 points** of held-out accuracy here - 45.0% to 52.5%, with
#: three-seed spreads that do not overlap.
#:
#: The cause is collision rather than capacity. This phrasebook uses 859
#: distinct trigrams, and 128 buckets leaves 85% of them sharing one with
#: something else, so a trigram that distinguishes two paths arrives on top of
#: six that do not.
#:
#: 256 and not more for two reasons that agree: the sweep is flat past it
#: (51.7% at 512, 51.2% at 1,024), and the device takes the bucket index from
#: the hash's low byte, so 256 is the most it can address without a wider
#: tokenizer everywhere.
BUCKETS = 256


def train_model(train: Path, model: Path, hidden: str = HIDDEN_SIZES,
                buckets: int = BUCKETS) -> None:
    subprocess.run(
        [sys.executable, str(REPO / "classify.py"), "--file", str(train),
         "-o", str(model), "--accum-bits", "24", "--balance", "--quiet",
         "--hidden-sizes", hidden, "--buckets", str(buckets)],
        check=True, cwd=REPO)


def score(model: Path, unseen: Path) -> tuple[float, float]:
    """(overall, macro) on phrasings the model never saw, under the integer
    arithmetic that ships - `libinfer.classify` is what the eZ80 imitates."""
    import libdata
    import libinfer

    loaded = libinfer.Model.load(str(model))
    pairs = libdata.read_files([str(unseen)])
    return libdata.score_predictions(
        pairs, lambda q: libinfer.classify(loaded, q, 24))


def _report_audit(db_path: Path, model: str) -> None:
    """What the shipped model does to phrasings it *was* trained on.

    Reported at build time rather than left to be discovered on the machine,
    because the failure has no symptom: the graph answers, the answer is
    fluent, and it is about the wrong hop.
    """
    import sqlite3

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    have = {r for (r,) in db.execute(
        "SELECT DISTINCT relation FROM edge WHERE source = ?", (SOURCE,))}
    result = relationpaths.audit(model, db, have, per_template=40, seed=1)
    db.close()

    print(f"\nthe shipped model, on the {result.phrasings} phrasings it was "
          f"trained on:")
    print(f"  {result.right / result.asked:.1%} of {result.asked:,} questions "
          f"route to the right path")
    print(f"  {result.steady}/{result.phrasings} phrasings answer the same way "
          f"whatever the subject's name")
    print("  the encoder hashes the whole question into 128 trigram buckets, "
          "and a name is\n  most of a short question - so who you ask about "
          "changes what is asked.")
    if result.worst:
        print("\n  the phrasings it is least sure of:")
        for path, template, share, instead in result.worst[:5]:
            print(f"    {share:>6.1%}  {template.format(s='X'):<44} "
                  f"{path} -> mostly {instead}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--out", type=Path, default=REPO / "dist")
    ap.add_argument("--stem", default="SILO")
    ap.add_argument("--held-out", type=int, default=HELD_OUT)
    ap.add_argument("--climb-limit", type=int, default=None,
                    help="Values a climb may examine before giving up, which "
                         "is one more than the hops it may take. Seven "
                         "generations live here, so 7 is what answers "
                         "generation 6 and 6 is what this card shipped with")
    ap.add_argument("--limit", type=int, default=None,
                    help="Index only this many articles. Must match every "
                         "build of one card - a document id is a position in "
                         "the article list, and a wrong one is still an article")
    ap.add_argument("--hidden-sizes", default=HIDDEN_SIZES,
                    help="Classifier width. See HIDDEN_SIZES for the sweep "
                         "this default came from")
    ap.add_argument("--skip-train", action="store_true",
                    help="Reuse the model already in --out")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"no database at {args.db}\n"
                         f"  python data/silo/generate.py")
    args.out.mkdir(parents=True, exist_ok=True)
    # Named after the stem, because they are part of that card. They were not,
    # and building a second card with `--stem SILOBIG` to compare classifier
    # sizes overwrote the first card's model - so a later `--skip-train` built
    # SILO with the wrong classifier and reported it as 94.4 KB without a word.
    model = args.out / f"{args.stem.lower()}-relations.npz"
    probe = args.out / f"{args.stem.lower()}-relations-heldout.npz"

    full, reduced, unseen = write_questions(args.db, args.out, args.held_out)
    if not args.skip_train:
        print(f"training the card's classifier on {full.name} "
              f"(every phrasing)...")
        train_model(full, model, args.hidden_sizes)
        print(f"training a throwaway on {reduced.name} to measure with...")
        train_model(reduced, probe, args.hidden_sizes)
    elif not model.exists() or not probe.exists():
        raise SystemExit(f"--skip-train, but {model} or {probe} is missing")

    overall, macro = score(probe, unseen)
    print(f"\nclassifier, on {args.held_out} phrasings per path it never saw: "
          f"{overall:.1%} overall, {macro:.1%} macro")
    print("  measured on a model trained without them, not the one going on "
          "the card.\n  The score classify.py prints is a split over queries "
          "and this data is\n  templated, so it is much higher and means much "
          "less.")
    _report_audit(args.db, str(model))

    argv = ["--db", str(args.db), "--source", SOURCE,
            "--out", str(args.out / args.stem), "--relations", str(model)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.climb_limit is not None:
        argv += ["--climb-limit", str(args.climb_limit)]
    buildwikisearch.main(argv)

    print(f"\n  python benchwiki.py --card {args.out / args.stem}")


if __name__ == "__main__":
    main()
