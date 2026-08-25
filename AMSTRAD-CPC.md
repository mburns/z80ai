# Amstrad CPC Support

How to build and run Z80-μLM on an Amstrad CPC 464, 664 or 6128.

## Overview

A CPC is a 4MHz Z80A with a firmware ROM whose entry points are reached through
a jumpblock in RAM. The port is close to the ZX Spectrum one — the engine and
the packed 2-bit weight layout are shared — with `CALL &BB5A` where the
Spectrum has `RST 10h`.

The result is a binary with an AMSDOS header, which loads and runs with:

```
RUN"CHAT.BIN"
```

## Memory layout

This is the whole story of the port, because a CPC has barely enough RAM.

| Region | Use |
|---|---|
| `&0000`–`&003F` | Restart vectors the firmware itself uses: RST 1 (LOW JUMP), RST 3 (FAR CALL), the interrupt entry at `&0038` |
| `&0040`–`&A67A` | **42,555 bytes free.** Where the model goes |
| `&A67B`–`&BFFF` | HIMEM: firmware workspace and the jumpblock |
| `&C000`–`&FFFF` | Screen memory |

The build is assembled at `&0040`, the first byte above the restart vectors.
Starting even one page higher would cost more than the headroom the shipped
models have:

| Model | Image | Headroom below HIMEM |
|---|---:|---:|
| `guess` | 39,625 | 2,930 |
| `tinychat` | 39,921 | 2,634 |
| `smalltalk` | 40,032 | 2,523 |

`&A67B` is HIMEM with the disc ROM active. Building against the higher
tape-only ceiling would produce a file that broke the moment anyone put it on a
disc, which is the only way it ships.

**Only the packed weight layout fits.** The index-list layouts the CP/M target
can choose need 43–48KB, so `build.py`'s fastest-that-fits search does not
arise here — there is one option.

## I/O

| Firmware call | Address | Use |
|---|---|---|
| `TXT_OUTPUT` | `&BB5A` | Print the character in A |
| `KM_WAIT_CHAR` | `&BB06` | Wait for a key, return it in A |
| `SCR_SET_MODE` | `&BC0E` | Set the screen mode, and clear the screen |

Two firmware facts shape the code:

- **Firmware calls corrupt AF, BC, DE and HL.** That is survivable only
  because nothing in `libnn`'s generation loop is live in a register across
  `CALL PRINTCH` — RESULT, GENCNT and CTXCHARS are all in memory — so the I/O
  path may clobber whatever it likes. `READ_INPUT` pushes what it needs.
- **Interrupts must stay enabled**, because `KM_WAIT_CHAR` is fed by the
  interrupt-driven keyboard scan. The packed layout never disables them. (The
  index-list layout does, which is a second reason it is not offered here.)

The build starts in mode 2 — 80 columns, which suits a chat prompt better than
the 40-column mode 1 a CPC boots into. Setting a mode also clears the screen,
so that is one firmware call rather than two.

## Building

```bash
python build.py --model examples/guess/model.npz --target cpc --output GUESS.BIN
```

or the standalone builder, which also reports the headroom:

```bash
python buildcpc.py --model examples/guess/model.npz --output GUESS.BIN
```

`--org` moves the load address if you need the space below it for something
else; the build refuses to assemble an image that would run past HIMEM rather
than emitting one that cannot load.

## Running it

### Emulators

Any CPC emulator will do — [Arnold](http://www.cpctech.org.uk/arnold.html),
[CPCEC](http://cngsoft.no-ip.org/cpcec.htm), WinAPE, or
[RetroVirtualMachine](https://www.retrovirtualmachine.org/).

The binary needs to be on a disc image. With [iDSK](https://github.com/cpcsdk/idsk):

```bash
iDSK GUESS.DSK -n                     # new blank disc
iDSK GUESS.DSK -i GUESS.BIN -t 1      # add the binary
```

Then boot the emulator with the disc in drive A and:

```
RUN"GUESS.BIN"
```

Most emulators can also read a host directory as a virtual disc, which skips
the image-building step.

### Real hardware

Same file. Get it onto a 3" disc (or an M4 card, Gotek, or whatever your
machine has) and `RUN"` it.

## The AMSDOS header

The first 128 bytes of the file are the header AMSDOS reads to know where to
load the image and where to start it. The fields that matter:

| Offset | Bytes | Contents |
|---|---|---|
| 0 | 1 | User number |
| 1 | 8 | Filename, uppercased and space-padded |
| 9 | 3 | Extension, `BIN` |
| 18 | 1 | File type: 2, binary |
| 21 | 2 | Load address |
| 26 | 2 | Entry address |
| 64 | 3 | Image length |
| 67 | 2 | Checksum: the sum of bytes 0–66 |

The checksum is the field worth being careful about. AMSDOS treats a file whose
checksum does not match as *headerless* — so a wrong one does not fail loudly,
it loads the image to the wrong address. `verify_artifacts.py` parses the
header back and checks it, along with the load address and the declared length.

## Testing

`libhost.CPCHost` stubs the three firmware entries and runs a build in
`libz80emu`, so a CPC binary is checked the same way every other target is:
executed instruction by instruction, output compared against the NumPy
reference model. See [TESTING.md](TESTING.md).

```bash
python -m pytest tests/test_libcpc.py
python bench.py --model examples/guess/model.npz --target cpc com
```

## What is not done

- **Only 64K is used.** A 6128 has 128K, and its second bank would hold the
  index-list layout — roughly 5x fewer instructions per character. That needs
  bank switching through `&7Fxx` in the layer loop.
- **No `.DSK` output.** The build writes a headered binary; putting it on a
  disc image is a separate step with an external tool.
- **The container has not been checked on real hardware.** The header is
  written and parsed back by the test suite, and the inference is verified in
  the emulator, but nobody has yet run `RUN"CHAT.BIN"` on a real CPC or in a
  CPC emulator. If you have one, that is the gap worth closing.
