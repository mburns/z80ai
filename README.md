# Z80-μLM: A Retrocomputing Micro Language Model

Z80-μLM is a 'conversational AI' that generates short character-by-character sequences, with quantization-aware training (QAT) to run on a Z80 processor with 64kb of ram.

The root behind this project was the question: how small can we go while still having personality, and can it be trained or fine-tuned easily? With easy self-hosted distribution?

The answer is Yes! And a 40kb .com binary (including inference, weights & a chat-style UI) running on a 4MHz processor from 1976.

It won't pass the Turing test, but it might make you smile at the green screen.

For insight on how to best train your own model, see [TRAINING.md](TRAINING.md).

## Examples

Two pre-built examples are included:

### [tinychat](examples/tinychat/)

A conversational chatbot trained on casual Q&A pairs. Responds to greetings, questions about itself, and general banter with terse personality-driven answers.

```
> hello
HI
> are you a robot
YES
> do you dream
MAYBE
```

### [guess](examples/guess/)

A 20 Questions game where the model knows a secret topic and answers YES/NO/MAYBE to your questions. Guess correctly to WIN.

![GUESS.COM example](examples/guess/example.png)

Includes tools for generating training data with LLMs (Ollama or Claude API) and balancing class distributions.

### [smalltalk](examples/smalltalk/)

The same idea as tinychat, but trained on **real crowdsourced utterances** —
[CLINC150](data/clinc150/), CC BY 3.0 — instead of hand-written ones, with an
explicit `IDK` for anything it was not trained on.

```
> are you a robot
IM A BOT
> what is the point of it all
WHO KNOWS
> whats the stock price of apple
IDK
```

80.6% of held-out queries answered correctly, against 59.4% for a keyword table
built from the same data — which is the check that the model is doing something
a word list cannot, **in 64KB of RAM**. Given storage the answer changes; see
below. See it yourself:

```bash
python data/baseline.py examples/smalltalk/training-data.txt.gz \
                        --model examples/smalltalk/model.npz
```

### [clinc150](examples/clinc150/)

For an Agon with an SD card. The model emits an *index* and the reply text
lives on the card, so reply length costs nothing and all 150 CLINC intents ship
with full sentences instead of 19 with two-word ones.

```
> what is my checking balance
YOUR BALANCE IS FOUR HUNDRED DOLLARS
> how do i jump start a car
RED TO POSITIVE THEN BLACK TO GROUND
> what is the airspeed velocity of a swallow
I DO NOT KNOW THAT ONE
```

81.9% macro over 151 classes, in 47KB of weights. Its README also records the
mixture-of-experts design that was built for this and **lost** to it — routing
turned out to be the same problem with the evidence thrown away.

See [data/README.md](data/README.md) for what makes a dataset suit a 2-bit
model, and `data/lint.py` for checking one before you train it.

### [Simple English Wikipedia on a card](data/wikipedia/)

Not a model at all. **283,997 articles searched in plain English on an Agon**,
from an SD card, in a 6KB program:

```
? mount everest

Mount Everest
Mount Everest is the highest mountain on Earth. Mount Everest is in the
Himalayas, a tall mountain range in Asia.
```

An ordinary inverted index, because the trigram encoder the models use scores
2 of 13 on the same queries and returns *Bures Hamlet* for "who wrote hamlet" —
it throws away which words matched, which is the one thing retrieval needs.
BM25 gets 11 of 13. All the arithmetic happens at build time, so the machine
does nothing but add bytes; that is what keeps the score accumulator at one
byte per article — 277KB, resident, for the whole encyclopedia.

`benchwiki.py` runs the card in the emulator and counts what a query costs.
Looking up `zilog z80` retires 66,359 instructions and reads 6,282 bytes;
`mount everest` costs 1.8 M because *mount* is a common word and *everest* is
not. What sets the price is the commonest word in the query, not how many words
it has.

It is a search box, not an oracle: "who wrote hamlet" returns *Hamlet*, and
reading the answer out of the article is comprehension, which is out of reach.

## Which budget is the claim about?

Worth being clear, because the answer differs and only one of them was ever on
the page. Held-out macro on `smalltalk`, with what each would occupy on the
target machine:

| | on device | macro |
|---|---:|---:|
| keyword table | 1.7 KB | 62.1% |
| nearest centroid | 4.8 KB | 74.8% |
| **the model** | **35.9 KB** | **80.7%** |
| 1-NN over the whole corpus | 130.9 KB | 84.3% |

On CP/M or a Spectrum, everything competes for the same 64KB and the model wins
outright. On an Agon with a card, a plain nearest-neighbour retriever is *more
accurate* — and needs about four times the storage to be.

So the honest claim is **accuracy per byte**, not accuracy. That is the axis the
target machines actually have, and it is the one the model wins. `data/baseline.py`
prints both, and `tests/test_baseline.py` fails a build that quietly gives ground
on either.

## Quickstart

Get running in under 5 minutes:

**1. Download** pre-built binaries from [GitHub Releases](../../releases)

**2. Install an emulator:**

| Platform | CP/M (.COM files) | ZX Spectrum (.TAP files) | Amstrad CPC (.BIN files) |
|----------|-------------------|--------------------------|--------------------------|
| **Linux** | [iz-cpm](https://github.com/ivanizag/iz-cpm/releases) | `apt install fuse-emulator-gtk` | [Arnold](http://www.cpctech.org.uk/arnold.html) |
| **Windows** | [iz-cpm](https://github.com/ivanizag/iz-cpm/releases) | [Fuse](https://fuse-emulator.sourceforge.net/) | [WinAPE](http://www.winape.net/) |
| **macOS** | [iz-cpm](https://github.com/ivanizag/iz-cpm/releases) | `brew install fuse-emulator` | [RetroVirtualMachine](https://www.retrovirtualmachine.org/) |

**3. Run:**

- **CP/M**: `iz-cpm CHAT.COM` — or `CHAT-COL.COM`, the same model with the
  fastest weight layout (13x quicker per character, 8KB larger)
- **ZX Spectrum**: `fuse --tape CHAT.TAP`, then `CLEAR 24575`, `LOAD "" CODE` and `RANDOMIZE USR 24576`
- **ZX Spectrum Next**: `CHAT-NEXT.TAP`, loaded the same way — 8x quicker, because
  it clocks the CPU to 28MHz. It also runs on a plain 48K machine
- **Amstrad CPC**: put `CHAT-CPC.BIN` on a disc image and `RUN"CHAT-CPC.BIN"`
- **Agon Light / eZ80**: copy `CHAT.bin` to the SD card and run it by name — or
  `CLINC.bin`, which answers 150 intents in full sentences and needs
  `PHRASES.DAT` copied beside it

For building from source or training your own models, see [TRAINING.md](TRAINING.md).

## Building

`build.py` is one front end for every target. By default it picks the fastest
weight layout that still fits the machine:

```bash
python build.py --model examples/guess/model.npz --output GUESS.COM
python build.py --model examples/guess/model.npz --target zx   --output GUESS.TAP
python build.py --model examples/guess/model.npz --target next --output GUESS.TAP
python build.py --model examples/guess/model.npz --target cpc  --output GUESS.BIN
python build.py --model examples/guess/model.npz --target ez80 --output GUESS.bin
```

The individual builders (`buildz80com.py`, `buildfastz80com.py`,
`buildcolz80com.py`, `buildz80tap.py`, `buildnext.py`, `buildcpc.py`,
`buildez80.py`) still work standalone if
you want a specific layout, as does `--target cpm-column`, `cpm-fast` or
`cpm-packed`.

A model carrying a phrasebook builds an Agon classifier instead of a character
decoder, and writes two files — the binary and the `PHRASES.DAT` it loads from
the card. Both come out of one build, because the offset table and the text it
indexes are two halves of one thing. See [EZ80.md](EZ80.md).

## Features

- **Trigram hash encoding**: Input text is hashed into 128 buckets - typo-tolerant, word-order invariant
- **2-bit weight quantization**: Each weight is {-2, -1, 0, +1}, packed 4 per byte
- **16-bit integer inference**: All math uses Z80-native 16-bit signed arithmetic
- **~40KB .COM file**: Fits in CP/M's Transient Program Area (TPA)
- **Autoregressive generation**: Outputs text character-by-character
- **No floating point**: Everything is integer math with fixed-point scaling
- **Interactive chat mode**: Just run `CHAT` with no arguments
- **Optional order-aware input**: `--position-bands` makes the encoder
  distinguish `PUT KEY IN BOX` from `PUT BOX IN KEY` - see [ENCODING.md](ENCODING.md)
- **Tested against a real CPU emulator**: every build is executed instruction by
  instruction and compared to a NumPy reference model - see [TESTING.md](TESTING.md)

## Platform Support

Z80-μLM runs on multiple Z80-based platforms:

- **CP/M**: Original target platform. Generates `.COM` files three ways, and
  `build.py` picks the fastest that fits the TPA
  - `buildz80com.py` — packed weights, 2 bits each. Always fits
  - `buildfastz80com.py` — an index list per weight value, per neuron. ~9x faster
  - `buildcolz80com.py` — index lists per input *column*, so zero activations
    are skipped as well as zero weights. ~24x faster, about 3KB larger
  - All three share their engine with the ZX build (`libnn.py`) and their
    front end — entry, chat loop, BDOS line buffer — with each other (`libcpm.py`)
- **ZX Spectrum 48K**: Full support via `buildz80tap.py`, over the shared ZX
  target in `libzx.py`. See [ZX-SPECTRUM.md](ZX-SPECTRUM.md) for details
  - Generates `.TAP` files for emulators or real hardware
  - Uses ZX Spectrum ROM routines for I/O
  - Memory optimized for 48K systems
  - Compatible with most ZX Spectrum emulators
- **ZX Spectrum Next**: `buildnext.py`. See [SPECTRUM-NEXT.md](SPECTRUM-NEXT.md)
  - The Spectrum image plus fourteen bytes that set the Next's clock to 28MHz —
    8x the generated characters per second
  - Still loads and runs on a real 48K Spectrum, at 3.5MHz: nothing there
    decodes the clock ports
- **Amstrad CPC 464/664/6128**: `buildcpc.py`. See [AMSTRAD-CPC.md](AMSTRAD-CPC.md)
  - Generates a binary with an AMSDOS header — `RUN"CHAT.BIN"`
  - Uses the firmware jumpblock for I/O, and assembles at `&0040` because a CPC
    leaves only 42,555 bytes below HIMEM for a ~40KB model
- **Agon Light / eZ80 (ADL mode)**: `buildez80.py`. See [EZ80.md](EZ80.md)
  - 24-bit addressing, so the 64KB ceiling on model size is gone
  - 24-bit accumulators, which cannot overflow the way the Z80's 16-bit ones can
  - No 256-neuron layer limit
  - Three kernels, fastest-that-fits: weights unrolled into code, optionally
    accumulated input-major so zero activations are skipped too — 23x fewer
    instructions than walking a weight stream

For ZX Spectrum builds, use `run-zx.sh` in example directories or see the [ZX Spectrum guide](ZX-SPECTRUM.md).

## How fast is it?

`bench.py` runs a target in the emulator and counts what one generated character
costs. For the shipped 256→256→192→128→11 `guess` model:

| target | size | instructions | Z80 T-states | seconds |
|---|---|---|---|---|
| CP/M, packed weights | 39,666 | 1,246,304 | 9,902,388 | 2.48 @ 4 MHz |
| CP/M, index lists | 44,800 | 256,473 | 1,598,895 | 0.40 @ 4 MHz |
| CP/M, column-major | 47,872 | 95,247 | 758,697 | **0.19 @ 4 MHz** |
| ZX Spectrum | 39,695 | 1,246,304 | 9,902,388 | 2.83 @ 3.5 MHz |
| Agon eZ80, byte stream | 146,645 | 923,194 | — | — |
| Agon eZ80, unrolled | 252,016 | 90,340 | — | — |
| Agon eZ80, column-major | 389,319 | 38,012 | — | — |

```bash
python bench.py --model examples/guess/model.npz \
                --target com fast col ez80-compact ez80-row ez80
```

eZ80 T-states are omitted rather than quoted misleadingly: its per-instruction
timings differ substantially from the Z80's, so instruction count is the honest
cross-architecture comparison.

The eZ80 rows are the same network three ways. The byte stream walks every
weight at runtime; unrolling turns each weight into code, so the ~73% that are
zero cost nothing; column-major additionally skips zero *activations*, which
takes the work from 37,865 multiply-accumulates down to 8,192. All three produce
byte-identical output — that equality is the main thing
[the tests](TESTING.md) check. `--target ez80` picks the fastest one that fits
in Agon SRAM. See [EZ80.md](EZ80.md).

### Column-major: skipping zero activations, not just zero weights

The index-list layout walks each neuron's nonzero weights, which skips the ~73%
of the model that is zero and nothing else. The *activations* are sparser than
the weights — measured over the shipped models the input vector is ~17% nonzero
and the hidden layers 14–45% — and a row-major kernel cannot exploit that,
because which activations are zero changes with every character emitted.

Turning the loop inside out fixes that. Each input column owns the list of
neurons its weights reach, so a layer runs only the columns whose activation is
nonzero:

```
for each active column c:              # about a fifth of them
    x = act[c]
    for each nonzero weight w[j][c]:
        acc[j] += w * x
```

That takes one forward pass from 37,940 multiply-accumulates to 8,474. The
accumulator has to move from a register into memory, so each one costs more
instructions — worth it because most of them never happen. Net: **2.9× fewer
instructions, 2.1× fewer T-states**, for about 3KB more.

This is the same transformation the eZ80 `column` kernel makes, but without
16MB to unroll into: at ~10 bytes of code per weight, unrolling this model would
need 378KB. So the weights stay data and the inner loop walks them with `IY`.

`build.py --target auto` now picks it whenever it fits the TPA, falling back
through the index lists to packed weights. Release builds ship all three
(`NAME.COM`, `NAME-FAST.COM`, `NAME-COL.COM`) and the tests check that they
agree character for character — three independently generated programs reaching
one answer is a stronger signal than any of them matching a reference alone.

### The query half is computed once, not once per character

The input vector is 128 query buckets followed by 128 context buckets, and only
the context half changes while a response is being generated. So layer 1's
contribution from the query is a constant for the whole response, and `PREQ`
computes it once and hands it to layer 1 as its bias.

This is exact, not an approximation. The accumulator is a sum modulo 2¹⁶ (2²⁴ on
the eZ80) and addition mod 2ⁿ is associative, so regrouping the addends cannot
change a bit — the same argument [EZ80.md](EZ80.md) makes for reordering the
sum. What may *not* move is the `>>2`, which floors, so `PREQ` stops short of it.

| target | before | after | per character |
|---|---:|---:|---:|
| CP/M, packed weights | 3,004,037 | 2,313,383 | **1.30×** |
| CP/M, index lists | 318,417 | 256,473 | **1.24×** |
| Agon eZ80, column-major | 41,565 | 35,774 | **1.16×** |

These were measured before the packed kernel moved its bookkeeping into
registers, so its absolute counts are higher than the table above. The ratios
are what this section is about, and hoisting removes half of layer 1's inputs
either way.

The packed builds gain most because they walk every weight including the zeros,
and layer 1's query half is 32,768 of the model's 140,672 weights. Measured on a
30-character query; a shorter one activates fewer query buckets and gains less.
`PREQ` itself costs roughly what one character used to, so a one-character
response breaks even and everything longer wins. The `ez80-compact` and
`ez80-row` kernels are unchanged, and still agree with the column kernel
byte for byte.

## Interaction Style

The model doesn't understand you. But somehow, it *gets* you.

Your input is hashed into 128 buckets via trigram encoding - an abstract "tag cloud" representation. The model responds to the *shape* of your input, not the exact words:

```
"hello there"  →  [bucket 23: 64, bucket 87: 32, ...]
"there hello"  →  [bucket 23: 64, bucket 87: 32, ...]  (same!)
"helo ther"    →  [bucket 23: 32, bucket 87: 32, ...]  (similar - typo tolerant)
```

Word order is discarded along with everything else, which is what makes
paraphrases work. When order matters - a command parser, an adventure game -
`--position-bands` seeds each trigram's hash with where it appeared, at no
cost in generation time. See [ENCODING.md](ENCODING.md); it is a real trade,
not a free upgrade.

This is semantically powerful for short inputs, but there's a limit: longer or order-dependent sentences blur together as concepts compete for the same buckets. "Open the door and turn on the lights" will likely be too close to distinguish from "turn on the door and open the lights."

### Small Responses, Big Meaning

A 1-2 word response can convey surprising nuance:

- `OK` - acknowledged, neutral
- `WHY?` - questioning your premise
- `R U?` - casting existential doubt
- `MAYBE` - genuine uncertainty
- `AM I?` - reflecting the question back

This isn't necessarily a limitation - it's a different mode of interaction. The terse responses force you to infer meaning from context or ask probing direct yes/no questions to see if it understands or not (e.g. 'are you a bot', 'are you human', 'am i human' displays logically consistent memorized answers)

### What It's Good At

- Short, varied inputs with consistent categorized outputs
- Fuzzy matching (typos, rephrasing, word order)
- Personality through vocabulary choice
- Running on constrained 8-bit hardware

### What It's Not

- A chatbot that generates novel sentences
- Something that tracks multi-turn context deeply
- A parser that understands grammar
- Anything approaching general intelligence

It's small, but functional. And sometimes that's exactly what you need.

## Architecture

- **Input**: 128 query trigram buckets + 128 context buckets
- **Hidden layers**: Configurable depth/width, e.g., 256 → 192 → 128
- **Output**: One neuron per character in charset
- **Activation**: ReLU between hidden layers

### Quantization Constraints

The Z80 is an 8-bit CPU, but we use its 16-bit register pairs (HL, DE, BC) for activations and accumulators. Weights are packed 4-per-byte (2-bit each) and unpacked into 8-bit signed values for the multiply-accumulate.

The 16-bit accumulator gives us numerical stability (summing 256 inputs without overflow), but the model's expressiveness is still bottlenecked by the 2-bit weights, and naive training may overflow or act 'weirdly' without QAT.

### Z80 Inner Loops

The core of inference is a tight multiply-accumulate loop. Weights are packed 4-per-byte:

```z80
; Unpack 2-bit weight from packed byte
ld a, (PACKED)      ; Get packed weights
and 03h             ; Mask bottom 2 bits
sub 2               ; Map 0,1,2,3 → -2,-1,0,+1
ld (WEIGHT), a

; Rotate for next weight
ld a, (PACKED)
rrca
rrca
ld (PACKED), a
```

The multiply-accumulate handles the 4 possible weight values:

```z80
MULADD:
    or a
    jr z, DONE       ; weight=0: skip entirely
    jp m, NEG        ; weight<0: subtract
    ; weight=+1: add activation
    ld hl, (ACC)
    add hl, de
    ld (ACC), hl
    ret
NEG:
    cp 0FFh
    jr z, NEG1       ; weight=-1
    ; weight=-2: subtract twice, clearing carry before each
    ld hl, (ACC)
    or a
    sbc hl, de
    or a             ; the SBC above may have borrowed
    sbc hl, de
    ld (ACC), hl
    ret
NEG1:
    ; weight=-1: subtract once
    ld hl, (ACC)
    or a
    sbc hl, de
    ld (ACC), hl
    ret
```

After each layer, arithmetic right-shift by 2 to prevent overflow:

```z80
sra h        ; Shift right arithmetic (preserves sign)
rr l
sra h
rr l         ; ACC = ACC / 4
```

That's the entire neural network: unpack weight, multiply-accumulate, shift. Repeat ~100K times per character generated.

## Recent fixes

Building the emulator-backed test suite surfaced four bugs in the generated
code. If you have an older build, rebuild it:

- **`MULADD` borrow** (`buildz80com.py`, `buildz80tap.py`) — a weight of `-2` was
  applied as two consecutive `SBC HL,DE` without clearing carry in between, so
  every `-2` weight subtracted one too many whenever the first subtraction
  borrowed. Affected every inference on both CP/M and ZX builds.
- **ZX Spectrum keyboard** (`buildz80tap.py`) — the buffer-full check did
  `PUSH AF / CP B / POP AF / JR NC`, and `POP AF` restored the flags from
  *before* the compare. `JR NC` therefore tested the preceding `CP 32`, which is
  never carry for a printable character, so every keystroke was discarded. The
  `.TAP` build could not accept input at all.
- **Packed-weight row alignment** — weights were packed as one flat stream while
  the unpack loop reloads a byte at every 4th weight *of each neuron*. Any layer
  whose input width was not a multiple of four desynchronised from row 1 onward.
- **`align()`** — `if overage < boundary` is always true, so aligning an
  already-aligned address inserted a whole extra boundary of padding.

Two more in the Python:

- `feedme.AutoregressiveModel._forward_int` truncated toward zero when shifting
  down; `SRA H / RR L` floors. The reported integer accuracy was optimistic for
  every negative accumulator.
- Layers were discovered with a lexical sort, so a model with ten or more layers
  would have run `fc10` immediately after `fc1`.

---

License: MIT or Apache-2.0 as you see fit.
