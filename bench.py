#!/usr/bin/env python3
"""
Measure inference cost for each build target by running it in the emulator.

Two numbers are reported per target:

  instructions  instructions retired for one forward pass. Platform-neutral,
                and the honest way to compare the eZ80 backend against the Z80
                ones, since the two have completely different per-instruction
                timings.
  T-states      Z80 bus cycles, and the wall-clock estimate that follows from
                them. Only meaningful for the Z80 targets - an eZ80 retires
                most instructions in a fraction of the cycles a Z80 needs, so
                its T-state column is omitted rather than quoted misleadingly.

Usage:
    python bench.py --model examples/guess/model.npz
    python bench.py --model examples/guess/model.npz --target com fast ez80
"""

from __future__ import annotations

import argparse
import importlib
from typing import NamedTuple

from libhost import AgonHost, CPMHost, ZXHost


class Target(NamedTuple):
    """One benchmarkable build: which backend, on what machine, how fast."""

    module: str
    description: str
    #: Machine clock in Hz, for turning T-states into seconds.
    clock: int
    #: eZ80 targets report no T-states: their per-instruction timings differ
    #: from a Z80's, so quoting Z80 cycles for them would mislead.
    is_ez80: bool
    #: Which kernel to ask the backend for, when it offers a choice.
    kernel: str | None = None

    def build_kwargs(self) -> dict[str, str]:
        return {"kernel": self.kernel} if self.kernel else {}


TARGETS = {
    "com": Target("buildz80com", "CP/M, packed 2-bit weights", 4_000_000, False),
    "fast": Target("buildfastz80com", "CP/M, per-value index lists",
                   4_000_000, False),
    "col": Target("buildcolz80com", "CP/M, column-major index lists",
                  4_000_000, False),
    "tap": Target("buildz80tap", "ZX Spectrum, packed weights", 3_500_000, False),
    "ez80-compact": Target("buildez80", "Agon eZ80, one byte per weight",
                           18_432_000, True, "compact"),
    "ez80-row": Target("buildez80", "Agon eZ80, unrolled weight-major",
                       18_432_000, True, "row"),
    "ez80": Target("buildez80", "Agon eZ80, unrolled column-major",
                   18_432_000, True, "column"),
}


def _host(target: str, query: str, org: int) -> AgonHost | CPMHost | ZXHost:
    """Build the host for ``target``, loading at the address the build uses."""
    if target == "tap":
        return ZXHost(stdin=[query, "!"], org=org)
    if TARGETS[target].is_ez80:
        return AgonHost(stdin=[query, "!"])
    return CPMHost(cmdline=query)


def measure(target: str, model_path: str, query: str = "HELLO") -> dict:
    """Cycle and instruction counts for one forward pass of ``target``."""
    spec = TARGETS[target]
    module = importlib.import_module(spec.module)
    builder = module.build_autoreg(model_path, max_output_len=1,
                                   **spec.build_kwargs())
    image = builder.build()

    # Always take the load address from the builder: hardcoding it here meant
    # the ZX numbers were measured on an image loaded at the wrong address once
    # its origin moved.
    org = builder.org
    host = _host(target, query, org)
    cpu = host.cpu
    cpu.load(org, image)
    cpu.pc = org

    cpu.run(max_cycles=2_000_000_000, stop_pc=builder.labels["GENLOOP"])
    if cpu.pc != builder.labels["GENLOOP"]:
        raise RuntimeError(f"{target}: never reached the generation loop")
    start_t, start_i = cpu.tstates, cpu.instructions
    cpu.run(max_cycles=2_000_000_000, stop_pc=builder.labels["ARGMAX"])
    if cpu.pc != builder.labels["ARGMAX"]:
        raise RuntimeError(f"{target}: never completed a forward pass")

    return {
        "target": target,
        "bytes": len(image),
        "tstates": cpu.tstates - start_t,
        "instructions": cpu.instructions - start_i,
        "clock": spec.clock,
        "is_ez80": spec.is_ez80,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", "-m", default="examples/guess/model.npz")
    parser.add_argument("--query", "-q", default="HELLO")
    parser.add_argument("--target", "-t", nargs="+", default=["com", "fast", "ez80"],
                        choices=sorted(TARGETS))
    args = parser.parse_args()

    rows = [measure(t, args.model, args.query) for t in args.target]
    baseline = rows[0]["instructions"]

    width = max(8, *(len(row["target"]) for row in rows))
    print(f"\n{'target':{width}} {'size':>9} {'instructions':>13} {'speedup':>8} "
          f"{'Z80 T-states':>14} {'sec @ clock':>12}")
    print("-" * (width + 62))
    for row in rows:
        speedup = baseline / row["instructions"]
        if row["is_ez80"]:
            cycles, seconds = "-", "-"
        else:
            cycles = f"{row['tstates']:,}"
            seconds = f"{row['tstates'] / row['clock']:.2f}"
        print(
            f"{row['target']:{width}} {row['bytes']:9,} {row['instructions']:13,} "
            f"{speedup:7.2f}x {cycles:>14} {seconds:>12}"
        )
    print("\nspeedup is relative to the first target listed, by instruction count.")
    print("eZ80 T-states are omitted: its per-instruction timings differ from the Z80's.\n")


if __name__ == "__main__":
    main()
