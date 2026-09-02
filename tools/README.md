# tools/

## `wordings.py` — wordings from strangers

```bash
python tools/wordings.py --backend fake                    # offline, for the pipeline
python tools/wordings.py --backend claude --paths father_is shift_is
python tools/wordings.py --backend ollama --model gemma2:9b -n 4
```

The phrasing curve is still climbing at thirty-three wordings a path, and
`data/silo/README.md` has said each time that the next dozen should not come
from the hand that wrote the last three. This asks a model for wordings **in
character** — a child, a Supply clerk, somebody angry, somebody terse — because
a register is the thing one author cannot vary on purpose, and the third dozen
turned out to be written in one.

It writes a review file and nothing else. Each candidate carries the persona
that produced it and its novelty against everything shipped, one minus the
cosine to the nearest existing wording in the encoder's own buckets — the
measure `phrasebook_diversity.py` uses. Most novel first, so the top of a
block is what a stranger brought and the bottom is padding. A candidate
without exactly one `{s}`, or that repeats a shipped wording, never reaches
the file.

Novelty flags repetition and does not predict yield — that was measured on
the third dozen and it did not. Read the sentences. Then the measurement is
`grammar_pilot.py`, five paired seeds, with the accepted dozen training-only
the way `EXTRA` is.

## `grammar_pilot.py` — what another dozen wordings are worth

```bash
python tools/grammar_pilot.py                 # 5 seeds, four arms
python tools/grammar_pilot.py --balance       # weight the loss by class size
```

`data/silo/README.md` drew a phrasing curve that was still climbing at nine
wordings and then stopped, because nine is what `relationpaths.py` happened to
contain. Everything tried against that number since was a change to the encoder
or the architecture, and none of it moved the number much.

This grows the phrasebook in groups — five paths, then five more, then the
remaining ten — and scores every arm on a **byte-identical** held-out set.
That is the whole difficulty: a path given twelve more wordings while still
holding out three has a held-out set with more neighbours to learn from, so it
scores better for a reason that is not grammar. `relationpaths.EXTRA` is
training-only for that reason, and `tests/test_silo.py` asserts the arms nest.

The answer was **55.4% to 65.3%**, and the shape of it matters more than the
number: at five paths grown, 80% of the gain came out of the other classes; at
twenty, 12% did. A measurement stopped at the first group says grammar is
zero-sum, and says it with a straight face.

## `class_cost.py` — what a new phrasebook class costs the old ones

```bash
python tools/class_cost.py --added born_in_year died_in_year fate_is
python tools/class_cost.py --added count_born_on --seeds 10 --watch born_on
```

A new path is nearly free on the card and is not free in the classifier, which
has to find a region in trigram space for it somewhere between the regions it
already has. `data/silo/README.md` measured that twice and got two different
answers — 18.6 points off the refusal class for two count classes, 6.9 for one
of them, which is nothing — and the difference between those conclusions was
entirely the seeds.

So this trains **two arms per seed**, the phrasebook with the new labels and the
same phrasebook without them, paired so the spread that swamps a three-seed
sweep cancels. `shared` scores only the labels present in both arms, because an
arm answering more kinds of question has a different denominator. `refuse` is
scored as *did it refuse at all*, which is the only distinction the eZ80 makes.
`--watch` reports named labels on their own, for a collision you expect.

Ten seeds is roughly ninety minutes. Three seeds is not enough — that is the
whole reason this exists, and it caught its author reporting a four-point loss
that turned out to be −0.9 ± 2.0.

## `optest.py` — validate the eZ80 emulator against a real one

The eZ80 backend's speed comes from instructions that exist only on the eZ80 in
ADL mode. They are implemented in `libz80emu` from a published opcode list, and
nothing in CI can catch an emulator that agrees with itself — every test would
still pass if an encoding were wrong, and only the `.bin` on real hardware would
fail.

This probes each of those instructions and prints the result, so it can be
compared against an independent implementation.

```bash
python tools/optest.py                              # what libz80emu says
python tools/optest.py --firmware /tmp/optest.bin   # build it for a real Agon
```

### Result, 2026-08-23, fab-agon-emulator 1.2.3

All seven agree. Both `LD rr,(IX/IY+d)` directions, negative displacements,
`MLT`, 24-bit `ADD` carry and 3-byte `POP` behave identically:

| probe | libz80emu | fab-agon | |
|---|---|---|---|
| `LD (IY+6),HL` / `LD HL,(IY+6)` | `A1B2C3` | `a1b2c3` | ok |
| `LD (IX+9),HL` / `LD HL,(IX+9)` | `4D5E6F` | `4d5e6f` | ok |
| `LD (IX+3),DE` / `LD DE,(IX+3)` | `778899` | `778899` | ok |
| `LD (IY-4),HL` / `LD HL,(IY-4)` | `0F1E2D` | `0f1e2d` | ok |
| `MLT HL` | `0000F7` | `0000f7` | ok |
| 24-bit `ADD HL,DE` carry | `010001` | `010001` | ok |
| `POP` reads three bytes | `C0FFEE` | `c0ffee` | ok |

fab-agon's own disassembler also decodes the bytes the way we emit them:

```
00000f: LD (IY+$6), HL   | fd 2f 06
000016: LD HL, (IY+$6)   | fd 27 06
```

### How to re-run it

The probe runs **as the firmware**, not as a program under MOS. That removes
MOS, the SD card, autoexec and the VDP from the picture — the eZ80 resets
straight into it. `JP.LIL` at reset switches from Z80 mode into ADL.

```bash
python tools/optest.py --firmware /tmp/optest.bin      # prints the addresses
cd fab-agon-emulator-*/
SDL_VIDEO_DRIVER=dummy ./fab-agon-emulator \
    --mos /tmp/optest.bin -d -u \
    -b 25 -b 48 -b 72 -b 101 -b 111 -b 124 -b 146 </dev/null
```

`HL` at each breakpoint is that probe's result.

### Four things that will waste your afternoon

- **`--breakpoint` takes decimal**, despite the help saying "hex address".
  `-b 6f` is rejected; `-b 111` is the same address and works. `--firmware`
  prints hex addresses, so convert them.
- **Scratch memory must be in RAM.** The firmware loads at `000000h`, which is
  flash on a real Agon, so stores there are silently discarded and every
  store-then-load probe reads back zero — which looks exactly like an emulator
  disagreement. `RAM_SCRATCH` points at `050000h` for this reason.
- **The IO-port state dumps get lost on exit.** `OUT ($20),A` prints CPU state,
  but the output is not flushed when `OUT ($00),A` shuts the emulator down.
  Breakpoints pause and flush, so use those to read results.
- **zsh does not word-split unquoted variables**, so building the `-b` flags in
  a shell variable passes them as one argument and the emulator reports
  "unused arguments left". Write them literally or use an array.

### What this does not cover

Instruction-level agreement, not machine-level. It says nothing about MOS API
behaviour — `mos_getkey` returning what `libhost.AgonHost` assumes, for
instance — nor about timing, interrupts or the VDP. A full `.bin` booting under
MOS would cover those, but MOS did not boot in this emulator on this machine,
which is why the probe bypasses it.

## `mostest.py` — validate `libhost`'s MOS against the real one

```bash
python tools/mostest.py                     # what libhost says
python tools/mostest.py -o MOSTEST.bin --data MOSTEST.DAT
# copy both to a card, run MOSTEST from MOS, compare line for line
```

Three probes, one line each: `mos_load` of a file that exists, of one that
does not, and — since save and restore — a file the probe itself creates
with `mos_fopen`/`mos_fwrite`/`mos_fclose`, appends to with `FA_OPEN_APPEND`,
and loads back. That last line is the whole surface a saved game and the
archive's log depend on.

**The `WRITTEN` line has not been run on hardware yet.** `libhost` says
`00 C0FFEE010203` and zeros; the first Agon to run it should have its answer
recorded here beside the `optest.py` table.
