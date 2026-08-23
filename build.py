#!/usr/bin/env python3
"""
One front-end for every build target, with automatic selection.

There are three ways to lay out the weights for a Z80:

  packed  two bits per weight, four to a byte. Always fits, but the inner loop
          spends most of its time unpacking - about 21 million T-states per
          generated character for the shipped examples.
  fast    an index list per weight value, one list per neuron. Roughly 75% of
          trained weights are zero and cost nothing at all, so it is around 13x
          quicker, but the index lists are larger and a model with fewer zeros
          may not fit.
  column  the same index lists, but per input column rather than per neuron, so
          a zero *activation* costs nothing either. The activations are sparser
          than the weights, which is worth another 2.9x for about 3KB more.

`--target auto` (the default) takes the fastest layout that fits the transient
program area, falling back through fast to packed. That is the choice you want
almost every time.

The eZ80 target applies the same fastest-that-fits policy to its own three
kernels:

  column   unrolled and accumulated input-major, so zero activations are
           skipped as well as zero weights. 23x fewer instructions, 2.6x the
           size.
  row      unrolled weight-major: the ~73% of weights that are zero cost
           nothing. 10x fewer instructions, 1.7x the size.
  compact  a weight stream walked at runtime. Slow, but its size does not
           depend on the model, so it is the only option for a model too large
           to unroll.

`--target ez80` chooses between them against the 512KB a real Agon has - not
the 16MB ADL can address. `ez80-column`, `ez80-row` and `ez80-compact` force
one.

Usage:
    python build.py --model examples/guess/model.npz --output GUESS.COM
    python build.py --model model.npz --target zx  --output CHAT.TAP
    python build.py --model model.npz --target ez80 --output CHAT.bin
"""

from __future__ import annotations

import argparse
import os

import libcpm
from libinfer import MAX_OUTPUT_LEN
from libz80 import Z80Builder

#: CP/M weight layouts, fastest first. `auto` takes the first one that fits.
CPM_LAYOUTS = ("column", "fast", "packed")


def build_cpm(model: str, max_output_len: int,
              prefer: str = "auto") -> tuple[Z80Builder, str]:
    """Build a CP/M .COM, choosing the fastest weight layout that fits."""
    import buildcolz80com
    import buildfastz80com
    import buildz80com

    modules = {
        "column": buildcolz80com,
        "fast": buildfastz80com,
        "packed": buildz80com,
    }
    if prefer != "auto":
        builder = modules[prefer].build_autoreg(model, max_output_len=max_output_len)
        return builder, prefer

    for layout in CPM_LAYOUTS:
        builder = modules[layout].build_autoreg(model, max_output_len=max_output_len)
        # Packed is the backstop: it always fits, so it is taken unconditionally
        # rather than leaving the loop with nothing to return.
        if layout == CPM_LAYOUTS[-1] or libcpm.fits_in_tpa(builder):
            return builder, layout
        print(
            f"\n{layout} layout needs {len(builder.build()):,} bytes and will "
            f"not fit the TPA; trying the next one down."
        )
    raise AssertionError("unreachable: the packed layout is the backstop")


def build_zx(model: str, max_output_len: int) -> tuple[Z80Builder, str]:
    import buildz80tap

    return buildz80tap.build_autoreg(model, max_output_len=max_output_len), "packed"


def build_ez80(model: str, max_output_len: int,
               kernel: str = "auto") -> tuple[Z80Builder, str]:
    """Build an Agon .bin, choosing the fastest kernel that fits Agon SRAM."""
    import buildez80

    builder = buildez80.build_autoreg(
        model, max_output_len=max_output_len, kernel=kernel
    )
    return builder, builder.kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load (.npz or .pt)")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--target", "-t", default="auto",
                        choices=["auto", "cpm", "cpm-column", "cpm-fast",
                                 "cpm-packed", "zx", "ez80", "ez80-column",
                                 "ez80-row", "ez80-compact"],
                        help="Platform and weight layout (default: auto = cpm)")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    args = parser.parse_args()

    target = args.target
    if target in ("auto", "cpm"):
        builder, layout = build_cpm(args.model, args.max_output_len, "auto")
    elif target.startswith("cpm-"):
        builder, layout = build_cpm(args.model, args.max_output_len,
                                    target.split("-", 1)[1])
    elif target == "zx":
        builder, layout = build_zx(args.model, args.max_output_len)
    else:
        kernel = target.split("-", 1)[1] if "-" in target else "auto"
        builder, layout = build_ez80(args.model, args.max_output_len, kernel)

    if target == "zx":
        import libzx

        image = builder.build()
        name = os.path.basename(args.output).split(".")[0]
        tap = libzx.build_tap(image, builder.org, name)
        with open(args.output, "wb") as fh:
            fh.write(tap)
        print(f"\nWrote {len(tap):,} bytes to {args.output}")
    else:
        builder.save(args.output)

    size = len(builder.code)
    print(f"Target: {target}  layout: {layout}  size: {size:,} bytes "
          f"({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
