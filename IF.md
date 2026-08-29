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

> x ledger
A ration ledger from year 188. The back of it has been drawn on.

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

### What a thing may be called

`SPLIT` takes two words and stops copying either at twelve characters, so a
thing named `ration book` or `identification` is one the parser can never
resolve. Nothing said so, and the failure was quiet twice over — the build
succeeded, and `DESCRIBE` then listed the thing in the room every turn:

```
> take identification
I do not know the word 'IDENTIFICATI'.
```

The truncation in that reply is the only evidence on screen that the *world*
was built wrong rather than the player spelling it wrong. `World.check()` now
refuses both, and `libworld.MAX_WORD_LEN` is where the limit lives because it
bounds what an author may write as much as what the parser may read.

The model is not wrong there in a way more training would fix. It is being
asked a question with no answer and returning its best guess, because that is
the only thing it can do.

## What is mutable, which is what a save file is

| | |
|---|---|
| `rooms` | name, description, six exits — 12 bytes, in the image |
| `things` | name, description, where it starts, portable — 8 bytes, in the image |
| a thing's name | **one word, at most `MAX_WORD_LEN`** — the parser's limit, not a style rule |
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

## Rules, and the three shapes of four they close

`data/silo/README.md` names four things a graph path cannot express, and the
reason is principled rather than accidental: composition works in both
directions and stops at aggregation, ranking and set intersection.

A rule is a flat list of conditions that must all hold, and a list of actions
to take when they do. That is the smallest step past a path, and it is worth
being exact about what the step buys:

| a path cannot | a rule | |
|---|---|---|
| a count over a set | **yes** | `CARRYING 2` — a question about a set, not about any one thing |
| an intersection of two sets | **yes** | two `HAVE` conditions ANDed: not two things, *these* two |
| state that outlives the question | **yes** | `FLAG` set in one room and read in another |
| a maximum over a set | **no** | ranking needs a loop, and a flat condition list has none |

Three of four. The fourth is not an oversight — a condition list is evaluated
once per rule and never iterates, so "the oldest person on X's crew" is exactly
as far out of reach as it was. Closing it wants a loop opcode, which is a
different instrument and a bigger one.

```
> take badge
> take wrench
Your hands are full. Whatever else you find down here is going to have to
wait, or something you already have is going down the stair without you.
The badge and the wrench together look like a story you would rather not
have to tell a deputy.
```

Both of those are rules. The first is a count and the second a conjunction, and
neither is a question the silo card could be asked at any length.

A rule is length-prefixed so that skipping one is an addition rather than a
walk over its parts. Flags and the one-shot markers are a **byte apiece rather
than a bit**: bits are eight times smaller and want a shift and a mask at four
call sites, and there is half a megabyte of SRAM spare — the sixty bytes are
not worth four places to be wrong, and a restore that replayed every event the
player had already seen would be the bug that saved them.

## The card, standing in a room

```bash
python buildwikisearch.py --db data/silo.db --source silo --out dist/SILO
# and a binary that carries the world as well as the card:
#   buildwikibin.build(num_docs, world=worlds.silo())
```

```
> east
IT, Level 34
Racks of machines behind glass, and a bench where a screen sits with its
back off.

> use
The screen wakes. Type a name to look it up, or LEAVE to stand up again.

archive> cistern pump failure
Incident Report 214-11: Cistern Pump Failure, Level 142
At approximately 0340 on the eleventh day of...

archive> leave
IT, Level 34
```

**It goes the other way round from how #62 pictured it.** The issue described a
world with a terminal inside it. The world is 4,050 bytes and the oracle
program is 38,912, so the terminal is not the small thing — the world is, and
`buildwikibin.build(..., world=...)` is what carries it. A world costs the
oracle binary under 5 KB.

### What "the two input paths can coexist" turned out to mean

Not that both fit. They share `INPBUF`: one line, read by whichever parser a
mode byte selects, and neither writes to it. `ATTERM` is that byte, and the
whole of the switch.

What nearly went wrong is narrower and worth recording. `Z80Builder.label`
assigns into a dict:

```python
def label(self, name: str) -> None:
    self.labels[name] = self.addr()
```

A name defined twice resolves to whichever was emitted last, and **nothing says
so**. Both programs had a `PROMPT` and a `BANNER`, so the merged binary printed
the oracle's `?` while the player was walking about — a bug with no error, in a
build that otherwise worked. `test_the_two_programs_define_no_label_twice`
spies on `label` during a merged build and asserts the count, because that is
the only way to see it.

## The map was already in the database

`worlds.py` hand-authors six rooms. `data/silo/buildworld.py` reads a world out
of `data/silo.db` instead, and it is short because the map has been in there
since the schema was written:

```bash
python data/silo/buildworld.py --floors 2 -o SILO.bin
```

| in the corpus | in the world |
|---|---|
| `next_along` — thirty minutes clockwise, and it wraps | `EAST`, and `WEST` back |
| `next_out` — one ring outward, and it does not | `NORTH` out, `SOUTH` in |
| `located_in` — a dwelling's level, a department's level | the stair, and the door off it |
| `article.lead` | every room description |
| `residence.until IS NULL` | the name beside the door |

`data/silo/schema.py` stores those two adjacencies as edges rather than
arithmetic **because the machine that walks them has no modulo**. A card walks
them to answer "who lives next door"; a world walks them to go east. Nothing
here recomputes `(bearing + 30) % 720`, and
`test_the_geometry_is_read_rather_than_recomputed` is the test that can tell
the difference — it deletes one edge and checks the exit went with it.

Nothing here writes prose either. Every description is a lead the corpus
already carries, so editing an article moves a room.

### The wall is the room id, not the memory

The section above measured 505 KB of SRAM free and observed that at 12 bytes a
room, that is more rooms than anybody will write. It is, and **it is not what
stops you**: `libworld.NOWHERE` is `0xFF`, so a room id is one byte.

```
144 landings + 14 departments        158 rooms    12,347 bytes
one residential floor                 72 rooms    24,492 bytes
                                     ---
                                     230 rooms, and 255 is the ceiling
```

One floor fits and two do not — 302 rooms, refused rather than truncated,
because a world quietly missing its bottom forty levels walks perfectly well.
The silo has twenty-nine opened floors. Reaching all of them wants a two-byte
room id, which costs every exit a byte in the image, or a world that streams
floors off the card, which costs a turn the one thing a turn must never cost.

A 230-room world still reads nothing: `io_bytes == 0` holds at two hundred
rooms exactly as it did at six, because a move is a table lookup whatever the
size of the table.

## Two checks the prose needed

**A whole playthrough, kept.** `tests/test_if.py` asserts phrases — fifty-three
tests, out of a world that says several thousand words — and is blind to the
half of an Interactive Fiction that *is* prose. `tools/transcript.py` replays a
session and compares the lot:

```bash
python tools/transcript.py                       # replay every one
python tools/transcript.py --update tests/transcripts/silo.txt
```

The file is the game's own output. `READ_INPUT` echoes what it accepts, so a
run already comes back looking like a session at a terminal, and the commands
are recovered by reading the `> ` lines back — there is no second file of
commands to keep in step. A diff is not by itself a bug; it is the change being
*seen*.

It found one thing on its first run: `DO_INV` prints "You are carrying" once
per item. Consistent with `DESCRIBE`, not what a player expects, and now
pinned.

**And it made a second one visible by omission.** Every thing row has carried a
description pointer since this file was written, and no code path ever read
offset 3 of one — four descriptions sat in the image, indexed and unreachable,
because there was no verb that showed them. `EXAMINE` / `X` / `READ` is the
cheaper of the two resolutions; the other was to stop emitting the text, and
`worlds.py` had already written four descriptions worth reading. It works on
what you are carrying as well as what is in the room: picking a thing up has
not stopped you being able to look at it.

**A rule that can never fire.** `World.check()` refuses every argument that
indexes nothing, and none of those is the bug an author ships. That one is a
key in the room it unlocks: arguments all in range, conditions that can never
all hold, no error. `World.reach()` is a monotone fixpoint whose result is
deliberately a **superset** of what a player can bring about, so a rule outside
it is certainly dead and a rule inside it is only probably live. Erring the
other way would report locked doors that are not locked, and an author who has
seen one false alarm stops reading the report.

Some contradictions are exact rather than searched — `AT 3` with `AT 5`,
`FLAG 2` with `NFLAG 2`, and `HAVE k` with `HERE k`, which reads as a plausible
sentence and is impossible because `where[k]` is `CARRIED` *or* a room.

## What it does not do yet

No daemons, no containers, no ranking, and no save and restore on the device -
`mos_fwrite` is in `libhost` and tested, and the eZ80 side is not written. No
screen mode and no status line: `PRWRAP` decides where a line ends, and nothing
here has ever told the terminal anything.

The compiled world has no things in it, because the corpus has no objects. It
has ten thousand *people*, and they cannot be `Thing`s: `where[]` is a byte
apiece, which would make a saved game 10 KB rather than 13. People stay on the
card and are read; only the world is resident.

Those are what is left of #62's second scope.
