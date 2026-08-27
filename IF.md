# A world, rather than an oracle

```bash
python buildif.py -o SILO.bin
```

Six rooms down the stair of Silo 18, four things to carry between them, and a
turn loop. It is small on purpose: the question issue #62 asks is whether the
machine can *hold* an Interactive Fiction and what a turn costs, and neither
needs three hundred rooms to answer.

```
Level 1 Landing
The top of the stair. A sealed hatch above you has not been opened in living
memory, and the light through the screen is the colour of dust.

> down
The Cafeteria
Long tables, and the great screen along the far wall showing the hills
outside. Nobody sits at the tables nearest it.
You can see ledger.

> take ledger
Taken.
```

## A turn reads nothing

That is the whole reason this is a separate program from the search card.

| | a question | a turn |
|---|---:|---:|
| card bytes | ~4,600 | **0** |
| instructions | ~370,000 | **~3,400** |

The oracle's figures are fine for a question and hopeless for a step. A player
takes a step every few seconds and most steps are `DOWN`, so a move has to be
free, and the only way it is free is if nothing about it touches the card.

So the world is tables in the image and a small mutable overlay in RAM. Nothing
is read, because there is nothing to read: it is all already there.

## The parser is a word table, and that was measured

`examples/parser/` scores both approaches on the same commands:

```
           encoder   char acc    train     eval
    flat (current)     99.3%    97.9%    85.9%
  8 position bands    100.0%   100.0%    98.4%
        word table         -   100.0%   100.0%
```

Accuracy is not what decides it. Asked about a word neither was given, both
models answer with a confident `OK` or `NO` and the table declines:

```
> xyzzy
I do not know the word 'XYZZY'.

> take zorkmid
I do not know the word 'ZORKMID'.
```

A player types a noun the author never wrote about every few turns. Naming it
back is the only useful reply, and a bare argmax cannot produce it — the same
gap `data/silo/` closed by [teaching a refuse class](data/silo/README.md).

The model is not wrong there in a way more training would fix. It is being
asked a question with no answer and returning its best guess, because that is
the only thing it can do.

## What is mutable, which is what a save file is

| | |
|---|---|
| `rooms` | name, description, six exits — 12 bytes, in the image |
| `things` | name, description, where it starts, portable — 8 bytes, in the image |
| `HERE` | the room the player is in — **1 byte, in RAM** |
| `WHERE[]` | where each thing is now — **1 byte each, in RAM** |
| `FLAGS[]` | one bit a proposition — **in RAM** |

Only the last three change. The image is identical on every copy, so a saved
game is the overlay and nothing else: **13 bytes** for this world, one
contiguous run so that writing it is a single `mos_fwrite` rather than three
and a format.

`mos_fwrite` itself is in `libhost` and tested; the eZ80 side of save and
restore is the next item on #62 and is not here yet.

## The free-SRAM figure, re-measured

Issue #62 budgeted the second scope against **388 KB** of free SRAM. That was
arithmetic over the *oracle's* memory map — 512 KB less the program, the
accumulator, the unpacking scratch and the stack.

A world binary does not share that map. It has no accumulator, no unpacking
buffers and no classifier, so:

```
SILO.bin  3,124 bytes   6 rooms, 4 things
  overlay 13 bytes - the whole saved game
  image ends at 040C34h, 517,068 bytes of SRAM unused
```

**505 KB**, not 388. The two programs were always going to be separate
binaries and the estimate quietly assumed one. What the 388 KB figure is
actually about is a world that lives *inside* the oracle card's program, which
is the "terminal found in the world" item and a different arrangement.

At 12 bytes a room and 8 a thing, 505 KB is more rooms than anybody will write.
Prose dominates and [#67](data/silo/README.md) already measured that a card
holds far more of it than an author will produce.

## What it does not do yet

No rules — `the door opens if you have the key and the power is on` is not a
graph query and not a word table either; it wants a condition→action bytecode,
which is the next item. No daemons, no containers, no save and restore on the
device, and no screen mode or status line. The world is also not wired to the
silo card, so the oracle terminal that ought to be standing in the IT office is
not there.

Those are the rest of #62's second scope.
