# ZX Spectrum Next Support

How to build and run Z80-μLM on a ZX Spectrum Next.

## Overview

A Next is Spectrum-compatible, so the existing `.TAP` already runs on one. What
this target adds is the reason to have a separate target at all: the Next
register that sets the CPU clock.

At 28MHz that is **8x the generated characters per second** of the 3.5MHz
build — the difference between watching it type and it answering. It costs
fourteen bytes at startup.

Everything else is the Spectrum build: same engine, same packed 2-bit weights,
same `.TAP` container, same load address at `&6000`. The image is byte-for-byte
`buildz80tap.py`'s plus the clock prologue, and the test suite asserts that so
it stays a thin wrapper rather than drifting into a fork.

## It still runs on a 48K Spectrum

The clock lives in a Next register, reached through a select/value port pair at
`&243B` and `&253B`. Nothing on a 48K machine decodes those ports, so the two
writes go nowhere and the program runs exactly as before, at 3.5MHz.

That is tested: the same image is driven through `ZXHost`, which has no Next
registers at all, and is expected to produce the same text.

## Setting the clock

The registers are above port 255, so `OUT (C),A` addresses them from BC rather
than `OUT (n),A`:

```
LD BC,&243B    ; register select
LD A,&07       ; register 7: CPU speed
OUT (C),A
LD BC,&253B    ; register value
LD A,&03       ; 3 = 28MHz
OUT (C),A
```

| Register 7 | Clock |
|---|---|
| 0 | 3.5 MHz |
| 1 | 7 MHz |
| 2 | 14 MHz |
| 3 | 28 MHz |

`--speed` picks one; the default is 28. The clock is set before the screen is
touched, so even the CLS runs at the new speed.

## Building

```bash
python build.py --model examples/guess/model.npz --target next --output GUESS.TAP
```

or the standalone builder:

```bash
python buildnext.py --model examples/guess/model.npz --output GUESS.TAP
python buildnext.py --model examples/guess/model.npz --speed 14 --output SLOW.TAP
```

## Running it

Copy the `.TAP` to the SD card and load it from the browser, or in BASIC:

```
CLEAR 24575
LOAD "" CODE
RANDOMIZE USR 24576
```

Same as a 48K Spectrum, because it is the same tape.

## What is not done

**The extra RAM is unused.** This is the interesting part, and it is deliberately
left for someone with hardware to test on.

The column-major weight layout — the fastest of the three, ~24x fewer
instructions per character than packed — needs about 48KB. The 48K memory map
has ~41KB above the screen, so it does not fit:

| Layout | Size | Fits in 41KB? |
|---|---:|---|
| packed | ~40KB | yes — what ships |
| index-list (`fast`) | ~45KB | no |
| column-major | ~48KB | no |

Reaching it means paging banks through `&C000` with port `&7FFD`, switching
per layer in the layer dispatch. Two things make that more tractable than it
sounds:

- The weight sizes work out. For `guess` (256→256→192→128→11) layer 1's
  weights are 16KB — exactly one bank — layer 2 is 12KB, and layers 3 and 4
  together are 6.5KB. So a bank per layer group, selected once per layer,
  rather than a bank boundary in the middle of the inner loop.
- **It is independent of the container.** A Next can page banks from a `.TAP`
  as readily as from a `.NEX`, so this does not require the loader work below.

Combined with 28MHz, that would be roughly **190x** the current Spectrum build.

**No `.NEX` output.** `.NEX` is the Next's native format and loads instantly
from SD, so it is worth having. The obstacle is that it loads whole 16KB banks,
and a program at `&6000` occupies bank 5 — which also holds the screen at
`&4000`–`&5AFF` and the ROM's system variables at `&5C00`–`&5CCB`. Shipping
bank 5 would overwrite the system variables the ROM print routine depends on,
and this build uses `RST 10h` for output.

Ways out, for whoever picks this up:

1. Set up the print channel by hand instead of relying on what the loader left,
   and ship bank 5 whole.
2. Move the program above `&8000` so only banks 2 and 0 are shipped, and page
   the weights — which is the banking work above anyway.
3. Write to the screen directly and stop needing the ROM.

All three want a Next to test against, which is why the `.TAP` ships today.

## Testing

```bash
python -m pytest tests/test_libnext.py
python bench.py --model examples/guess/model.npz --target tap next
```

`libhost.NextHost` records writes to the Next registers, which is what lets a
test assert the build really asks for 28MHz — the emulator has one clock and
does not speed up, so the request is otherwise invisible. `verify_artifacts.py`
checks it for every shipped `-NEXT.TAP`: a Next build that forgot to ask for
the faster clock is just the Spectrum build under another name.
