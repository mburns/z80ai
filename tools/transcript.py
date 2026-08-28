#!/usr/bin/env python3
"""
A playthrough as a file the emulator can be held to.

    python tools/transcript.py                         # replay every one
    python tools/transcript.py --update tests/transcripts/silo.txt

An Interactive Fiction is mostly prose, and prose is the part of a program that
changes without anything noticing. A room description gets a comma, a message
loses a word, and every test still passes because no test asserts a paragraph -
`tests/test_if.py` asserts *phrases*, twenty of them, out of a world that says
several thousand words. Six rooms is already past what anybody re-reads by
hand, and the world compiler in `data/silo/buildworld.py` puts two hundred on
the table.

## The format is the game's own output

`READ_INPUT` echoes every character it accepts, so a run already comes back
looking like a session at a terminal:

    > down
    The Cafeteria
    Long tables, and the great screen along the far wall...

That is the whole file. There is no separate script of commands to keep in step
with a separate file of expected output, because the commands are recoverable
from the echo: a line beginning with the prompt is a turn, and the rest of it
is what was typed. Replaying a transcript means reading its own `> ` lines back
and running them.

Two consequences worth knowing rather than discovering:

- **The echo is what was accepted, not what was typed.** `READ_INPUT` drops
  control characters and stops at `MAX_INPUT_LEN`. A command that gets
  truncated is stored truncated, so the file round-trips - but it records the
  truncation as though it were intended, and only the diff on the turn after
  will say otherwise.
- **A transcript must end the game.** `AgonHost` halts the CPU when its input
  runs dry, which is the right thing for a host and the wrong thing here: a
  transcript that stops mid-game comes back *looking finished*, and `--update`
  would write the prefix to disk as though the game had ended there. Nothing
  in the output says which. `host.finished` is the byte that knows, and
  `record` refuses on it.

## What a diff is worth

A failing transcript is not by itself a bug: prose is meant to change. It is a
prompt to read the diff, and `--update` is the answer when the diff is the
change. The value is that the change is *seen* - the same reason
`tests/test_codegen_stability.py` holds a byte count nobody chose.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import buildif
import libworld
from libhost import AgonHost

#: Where the golden files live. One per world, named after it.
TRANSCRIPTS = Path(__file__).resolve().parent.parent / "tests" / "transcripts"

#: What `buildif` prints before reading a line. A transcript's turns are the
#: lines that start with it, so a world that renames the prompt renames this.
PROMPT = "> "

#: Header key naming the world, as `module:function`. Kept in the file so a
#: transcript says what it is a transcript *of* - the alternative is a table
#: somewhere else that goes stale the first time one is renamed.
WORLD_KEY = "world"

#: Generous. A turn is a few thousand instructions, so this is room for
#: thousands of turns and still bounded, which is what makes a hang a failure
#: rather than a wait.
MAX_CYCLES = 200_000_000


class Hung(RuntimeError):
    """The run did not reach `quit`, so the output is a prefix of a game."""


def load_world(spec: str) -> libworld.World:
    """`worlds:silo` -> the world it names. Imports, which is the point."""
    module, _, function = spec.partition(":")
    if not function:
        raise ValueError(f"{spec!r} is not module:function")
    maker: Callable[[], libworld.World] = getattr(
        importlib.import_module(module), function)
    return maker()


def commands(text: str) -> list[str]:
    """The turns out of a transcript, from the game's own echo of them."""
    return [line[len(PROMPT):] for line in text.splitlines()
            if line.startswith(PROMPT)]


def header(text: str) -> dict[str, str]:
    """The `#` lines at the top, which stop at the first line that is not one."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        key, _, value = line[1:].strip().partition(":")
        if value:
            out[key.strip()] = value.strip()
    return out


def record(world: libworld.World, turns: list[str], spec: str) -> str:
    """Play the turns and return the file that should be on disk.

    Raises `Hung` rather than returning a prefix. A transcript that stops in
    the middle of a game is the failure mode this is worth guarding against,
    because it looks exactly like a short game.
    """
    game = buildif.build(world).build()
    host = AgonHost(stdin=list(turns), files={})
    out = host.run(game, max_cycles=MAX_CYCLES).replace("\r\n", "\n")
    if host.finished:
        raise Hung("the game asked for a turn the transcript did not have; it "
                   "has to end with a command that quits")
    if host.stdin:
        raise Hung(f"the game stopped with {len(host.stdin)} commands unread, "
                   f"beginning {host.stdin[0]!r}")
    if not host.cpu.halted:
        raise Hung(f"the game was still running after {MAX_CYCLES:,} cycles")
    return f"# {WORLD_KEY}: {spec}\n{out}"


def replay(path: Path) -> tuple[str, str]:
    """`(what is on disk, what the game says now)` for one transcript."""
    was = path.read_text()
    spec = header(was).get(WORLD_KEY)
    if spec is None:
        raise ValueError(f"{path} has no '# {WORLD_KEY}:' line, so nothing "
                         f"says which world it is a transcript of")
    return was, record(load_world(spec), commands(was), spec)


def diff(path: Path, was: str, now: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        was.splitlines(keepends=True), now.splitlines(keepends=True),
        fromfile=f"{path} (on disk)", tofile=f"{path} (now)"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("paths", nargs="*", type=Path,
                        help="transcripts to run; default is all of them")
    parser.add_argument("--update", action="store_true",
                        help="write what the game says now")
    args = parser.parse_args()

    paths = args.paths or sorted(TRANSCRIPTS.glob("*.txt"))
    if not paths:
        print(f"no transcripts under {TRANSCRIPTS}", file=sys.stderr)
        return 1

    failed = 0
    for path in paths:
        was, now = replay(path)
        if was == now:
            print(f"{path}: {len(commands(was))} turns, unchanged")
        elif args.update:
            path.write_text(now)
            print(f"{path}: updated")
        else:
            failed += 1
            print(f"{path}: changed\n{diff(path, was, now)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
