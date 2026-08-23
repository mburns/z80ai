# tools/

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
