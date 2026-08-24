# Testing

```bash
pip install -r requirements-dev.txt
pytest                  # everything
pytest -m "not slow"    # skip the full-size model runs (~20s instead of ~45s)
```

Nothing here needs PyTorch. Six tests that check the training-time encoders
against the reference are skipped if torch is absent. Python 3.10 or newer.

## How it works

The suite does not check that the build scripts emit *some* bytes. It executes
the bytes.

```
  model.npz ──┬──> build script ──> Z80 machine code ──> libz80emu ──> text
              │                                                          ║
              └──> libinfer (NumPy reference) ─────────────────────────> text
                                                                     compare
```

- **`libz80emu.py`** is a pure-Python Z80 / eZ80 CPU. It decodes structurally
  (the `x/y/z/p/q` decomposition), so the base instruction set, CB/ED/DD/FD
  prefixes and ADL mode all fall out of a few hundred lines. Cycle counts come
  from the M-cycle structure — 4T per opcode fetch, 3T per memory access, plus
  explicit internal cycles — and are pinned against the documented Zilog
  timings in `test_emulator.py`.
- **`libhost.py`** supplies the CP/M BDOS, ZX Spectrum ROM, Amstrad CPC
  firmware jumpblock and Agon MOS entry points the platforms call, turning
  console traffic into Python strings. It reads those addresses from
  `libcpm.py`, `libzx.py`, `libcpc.py` and `libez80.py` — the same modules the
  code generator emits calls from, so the emulator and the generated code
  cannot drift apart about where a routine lives. `NextHost` additionally
  records writes to the Next registers, which is the only way to see that a
  build asked for 28MHz: the emulator has one clock and does not speed up.
- **`libinfer.py`** is the golden model: the same tokenizer, context encoder,
  quantized inference and argmax in NumPy, with the hardware's exact integer
  semantics (16-bit accumulator wrap, arithmetic-shift flooring).

## Reading the generated code

`tools/disasm.py` disassembles a build, annotated with the labels that produced
it. Addresses that land on a label are named, so a listing reads as code rather
than as a wall of hex:

```bash
python tools/disasm.py --model examples/guess/model.npz --at MULADD --count 8
python tools/disasm.py --model examples/guess/model.npz --target ez80 --at PREQ
python tools/disasm.py --model examples/guess/model.npz --labels
```

```
MULADD:
  03D3  2A 2A 05      LD HL,(ACC)
  03D6  3D            DEC A
  03D7  28 0C         JR Z,MA_P1
  03D9  ED 52         SBC HL,DE
```

It decodes with the same `x/y/z/p/q` decomposition as `libz80emu`, and the two
are checked against each other: `test_disasm.py` runs a real build and requires
that they agree about where all 49,000 executed instructions end. Bytes that
decode to nothing render as `DB`, so pointing it at a weight blob produces a
listing rather than an exception.

## What each file covers

| File | What it pins down |
|---|---|
| `test_emulator.py` | The emulator itself: flags, block moves, indexed loads, DJNZ, documented cycle counts |
| `test_libz80.py` | Assembler: label resolution, relative-jump range, `align`, instruction encodings |
| `test_packing.py` | Both 2-bit weight layouts round-trip; every neuron starts on a byte boundary |
| `test_encoders.py` | Trigram/context hashing against an independent restatement of the spec |
| `test_kernels.py` | Emulator memory vs. reference, value by value: tokenizer buckets, context buckets, output logits |
| `test_end_to_end.py` | Generated text from every Z80 target vs. the reference, and the three CP/M layouts against each other |
| `test_column_kernel.py` | The column-major CP/M kernel: record layout, per-layer activations, and which columns it skips |
| `test_hoisting.py` | That folding layer 1's query half into its bias is bit-identical, in NumPy and in the emitted code |
| `test_ez80.py` | ADL encoding, the Agon header, and the same numeric comparisons for the eZ80 |
| `test_ez80_kernels.py` | All three eZ80 kernels against the reference *and* against each other, plus the flooring/ReLU boundary, the active-column list, and kernel selection |
| `test_ez80_argmax.py` | That the output layer is scanned in full however wide it is |
| `test_ez80_emulator.py` | eZ80-only opcodes and the two flag facts the kernels branch on |
| `test_builders.py` | TAP container layout, TPA size limits, embedded weight/bias/charset data |
| `test_model_shapes.py` | Layer discovery order, and the widths a Z80 backend can actually assemble |
| `test_build_frontend.py` | Automatic target selection |
| `test_verify_artifacts.py` | The release verifier's own failure paths |
| `test_libnn.py` | Layer planning, the Platform contract, and that the public API stays annotated |
| `test_libcpm.py` | The shared CP/M front end, and that all three `.COM` backends really emit it rather than a copy |
| `test_libzx.py` | The shared ZX target: memory map, entry code, and that `libhost` hooks the ROM addresses the build calls |
| `test_libcpc.py` | The Amstrad CPC target: firmware addresses, the HIMEM ceiling a ~40KB model has to fit under, and the AMSDOS header — where a bad checksum loads the file to the wrong address rather than failing |
| `test_libnext.py` | The Next target: that it asks for 28MHz, that it is still the Spectrum image plus the prologue, and that a 48K machine ignores the clock write rather than crashing |
| `test_build_inputs.py` | The shared build preamble: numeric layer ordering, geometry, and that codegen and the reference share one encoding |
| `test_model_io.py` | That a `.pt` and the `.npz` exported from it build the same image (skipped without PyTorch) |
| `test_classify.py` | The phrasebook trainer: that it is seeded, that the phrase index is stable, and that what it produces builds and answers (skipped without PyTorch) |
| `test_bench.py` | The benchmark target table, and that the faster layouts really do retire fewer instructions |
| `test_disasm.py` | The disassembler, chiefly by requiring it and the emulator to agree about every instruction boundary in a real build |
| `test_release_manifest.py` | That the build script, the manifest, the pins and the release list all name the same artifacts |
| `test_codegen_stability.py` | The exact bytes each backend emits, by hash |
| `test_performance.py` | What one generated character costs on each target, pinned exactly — the hashes say codegen moved, these say which way |
| `test_agon_files.py` | MOS file I/O, and that a load outside Agon SRAM raises instead of growing the emulator's memory |
| `test_baseline.py` | Every row of the accuracy claim, including the retrievers that beat the model once storage is free |
| `test_datasets.py` | The CLINC150 recipes, and that lint's thresholds judge a balanced 151-way set correctly |
| `test_phrasebook.py` | One forward pass, one reply: the reference path a phrasebook model is checked against |

## In CI

`.github/workflows/ci.yml` runs four jobs. Beyond the test suite, the ones that
matter are in `build`:

| Check | What it stops |
|---|---|
| `ruff check` | pycodestyle, pyflakes, bugbear, comprehensions, simplifications, modern syntax, import order and performance rules. Two justified `noqa`s remain, both explained inline - see `ruff.toml` |
| `mypy` | Annotations that contradict the code. `test_libnn.py` already requires the library modules to *be* annotated; this checks they are *right*. See `mypy.ini` for the two exclusions and why |
| Test matrix | Python 3.10 through 3.13, `fail-fast: false` so one version failing doesn't hide the rest |
| Reproducible build | Builds everything twice and compares byte for byte. A build that picks up dict ordering or a hash seed would stop matching what anyone can rebuild |
| Artifact verification | Boots every release binary in the emulator and compares its output to the reference. **A binary that assembles but computes the wrong answer cannot reach a release** |

`test_baseline.py` adds a fifth thing CI stops, in the test job rather than in
`build`: an accuracy regression. Until it existed, every number in the READMEs
was prose no build could contradict, and a retrain that halved a model's macro
score would have shipped green.

That artifact check is the reason any of this exists. Run it yourself:

```bash
./build-examples.sh dist
python verify_artifacts.py --dist dist
```

It found the ZX Spectrum load-address bug — both shipped `.TAP` files were
assembled past the top of RAM and could not load on any Spectrum.

## Why comparing text is not enough

`test_kernels.py` reads the actual 16-bit values out of emulator memory rather
than comparing generated strings. With a small charset two quite different logit
vectors often argmax to the same character, so a text comparison will happily
pass over a broken kernel. Both of the arithmetic bugs found while writing these
tests — the `MULADD` borrow and the packed-weight row desync — produced correct
*text* on the models that first exercised them.

The strongest technique available needs no reference model at all. The eZ80
backend emits the same network three ways — a runtime weight stream, unrolled
weight-major, and unrolled column-major — and
`test_ez80_kernels.py::test_all_kernels_agree_bit_for_bit` requires all three to
produce a byte-identical `OUTBUF`. Three independently generated programs
agreeing exactly is very hard to achieve by accident, and it stays meaningful
for models where writing down the expected answer by hand would not.

The CP/M side now has the same property. `test_all_cpm_builds_agree` runs the
packed, row-major and column-major builds on the same model and requires
identical output; those three share their tokenizer and their argmax and
essentially nothing else.

## When a correct test is not enough

Some optimizations are invisible to correctness tests, because getting them
wrong only makes the program slower. The column-major kernel skips inputs whose
activation is zero; appending one anyway is harmless, since its column adds zero
everywhere. Every numeric test still passes, and the speedup is gone.

`test_hidden_layer_list_holds_exactly_its_nonzero_activations` and
`test_activation_that_floors_to_zero_is_not_listed` check the *list* rather than
the answer. The second exists because the first missed the case: it used a
natural model, and the only way a neuron reaches the shift and still comes out
zero is a pre-shift accumulator in `[0, 3]` — ReLU catches everything more
negative earlier. That band has to be arranged deliberately.
`test_column_kernel.py` repeats both for the CP/M kernel.

The query-half hoisting has the same shape of risk in reverse: running PREQ once
per *character* instead of once per query would still be correct, just as slow as
before. `test_preq_runs_once_per_response_not_once_per_character` checks where
the call is, not what it computes.

If an optimization can regress without failing anything, it needs a test that
looks at the mechanism, not the output.

## Speed

End-to-end tests use a synthetic 256→16→12 model and generate 8 characters, so a
full forward pass is a few hundred thousand emulated instructions rather than a
few million. The shipped examples are covered by the `slow`-marked tests.

To time a target rather than check it:

```bash
python bench.py --model examples/guess/model.npz --target com fast col ez80
```
