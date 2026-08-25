#!/usr/bin/env python3
"""
Build an eZ80 (ADL mode) binary for Agon Light / Console8 and friends.

The Z80 builds spend most of their effort working around 64KB: weights are
squeezed to two bits, activations to 16 bits, and layers are capped at 256
neurons because DJNZ counts in a byte.  An eZ80 in ADL mode has a 24-bit
address space, so this backend drops those compromises: 24-bit accumulators
(the 16-bit overflow the QAT loss trains against cannot happen at all), 24-bit
activations, and no byte counters anywhere, so layers may be any width.

Three kernels, chosen by size (``--kernel``, default ``auto``).  On the shipped
`guess` model, one forward pass costs:

  column   923,194 -> 39,605 instructions, 384KB.  Unrolled and accumulated
           input-major, so it skips zero *activations* as well as zero weights:
           only 8,192 of the model's 37,865 nonzero weights are actually
           reached, because the input vector is ~10% nonzero and hidden layers
           are 16-39%.  Accumulators live in memory rather than a register,
           costing an instruction per multiply-accumulate - a good trade when
           ~78% of columns are skipped outright.
  row      923,194 -> 90,340 instructions, 252KB.  Unrolled weight-major.
           ``LD DE,(nnnnnn)`` reaches any activation in one instruction, so a
           nonzero weight costs a load and an add and a zero weight costs
           nothing.  Cannot skip zero activations: which ones are zero changes
           every step.
  compact  923,194 instructions, 147KB.  One signed byte per weight, walked at
           runtime, neurons and layers closed by sentinels.  Visits every
           weight including the ~73% that are zero, so it is slow - but its
           size does not depend on the model, which makes it the only option
           once a model is too large to unroll.

Unrolling is what the address space actually buys.  An index list, which is
what the CP/M builds use, would still have to turn each index into an address,
and the accumulator sits in HL so ``LD DE,(HL)`` is unavailable.

Reordering and regrouping the sum - splitting it by weight sign, or walking it
input-major - is exact rather than approximate: the reference wraps the
accumulator to 24 bits, and addition mod 2**24 is associative and commutative.
The >>2 is the part that may never move, because it floors.

`auto` takes the first kernel whose image fits in Agon SRAM.  Note that ADL
addresses 16MB but a shipping Agon has 512KB, and that is the real ceiling.

Inference is otherwise identical to the Z80 version: the same trigram hashing,
the same {-2,-1,0,+1} weights, the same >>2 per layer, the same argmax.  A
model built for CP/M runs here unchanged and produces the same text, except
where the Z80 would have overflowed its accumulator - there the eZ80 is right
and the Z80 is not.

Usage:
    python buildez80.py --model examples/guess/model.npz --output CHAT.bin
"""

from __future__ import annotations

import os

import numpy as np

import libagon
import libnn
from libagon import KEY_LABELS, MAX_INPUT_LEN, MOS_OUTCHAR
from libez80 import AGON_LOAD_ADDR, AGON_MAX_IMAGE, EZ80Builder, agon_header
from libinfer import (
    CONTEXT_LEN,
    MAX_OUTPUT_LEN,
    NUM_BUCKETS,
    SHIFT,
    BuildInputs,
    load_for_build,
)
from libz80 import Z80Builder

# Weight stream encoding. Values outside {-2,-1,0,1} act as terminators.
W_END_NEURON = 0x02
W_END_LAYER = 0x03


def encode_weights(weights: np.ndarray) -> bytes:
    """One signed byte per weight, each neuron closed by an end sentinel."""
    w = np.clip(np.asarray(weights), -2, 1).astype(np.int8)
    out = bytearray()
    for row in w:
        out.extend(int(v) & 0xFF for v in row)
        out.append(W_END_NEURON)
    out.append(W_END_LAYER)
    return bytes(out)


def encode_biases(biases: np.ndarray) -> bytes:
    """Sign-extended 24-bit biases, so they add directly to the accumulator."""
    out = bytearray()
    for v in np.asarray(biases).astype(np.int64):
        value = int(v) & 0xFFFFFF
        out.extend((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF))
    return bytes(out)


def neuron_ops(row: np.ndarray) -> list[tuple[int, int]]:
    """``[(column, weight)]`` for each nonzero weight, in ascending column order.

    The unrolled kernels turn this into straight-line code, so a zero weight
    costs nothing at all rather than a load and a branch.  About 73% of a
    trained model's weights are zero, which is where most of the speedup comes
    from.

    Ascending order is not an optimization - it is what keeps the build
    byte-reproducible.  Never iterate a set or a dict to produce this.
    """
    w = np.clip(np.asarray(row), -2, 1).astype(np.int8)
    return [(int(j), int(w[j])) for j in np.nonzero(w)[0]]


def encode_phrases(phrases: list[str]) -> bytes:
    """The on-card phrasebook: a count, an offset table, then the text.

    ``[count:3][offset:3 x count][NUL-terminated text...]``, every field
    24-bit little-endian and every offset relative to the *start of the file*.

    Offsets rather than addresses, deliberately. The rest of the image is
    absolute - libz80.resolve() patches fixups and throws the fixup list away,
    so a .bin carries no relocation information at all - but a file that
    contains no addresses cannot be loaded to the wrong one. The program adds
    the load address once, at runtime.
    """
    header = 3 + 3 * len(phrases)
    text = bytearray()
    offsets = []
    for phrase in phrases:
        offsets.append(header + len(text))
        text += phrase.encode('ascii') + b'\x00'

    out = bytearray()
    for value in (len(phrases), *offsets):
        out += bytes(((value >> (8 * k)) & 0xFF) for k in range(3))
    return bytes(out + text)


def layer_buffers(index: int, num_layers: int) -> tuple[str, str]:
    """(input buffer, output buffer) for layer ``index``, ping-ponging A/B."""
    in_buf = 'INBUF' if index == 0 else ('BUF_A' if index % 2 == 1 else 'BUF_B')
    out_buf = 'OUTBUF' if index == num_layers - 1 else (
        'BUF_A' if (index + 1) % 2 == 1 else 'BUF_B'
    )
    return in_buf, out_buf


def _emit_layer_compact(b: EZ80Builder) -> None:
    """The data-driven kernel: one pass over a sentinel-terminated weight stream.

    Slow - it visits every weight, including the ~73% that are zero - but its
    size is independent of the model, so it is the only option once a model is
    too large to unroll.

        BC  weight stream (one signed byte per weight, sentinel terminated)
        SP  input pointer - POP reads a 24-bit activation and advances in one go
        HL  24-bit accumulator
        IX  output pointer
    """
    b.label('LAYER')
    b.ld_mem_label_sp('SPSAV')
    b.di()

    b.label('LNEUR')
    b.ld_sp_mem_label('INBASE')
    b.ld_hl_nn(0)

    b.label('LWT')
    b.pop_de()          # DE = next activation
    b.ld_a_bc()         # A  = next weight
    b.inc_bc()
    b.or_a()
    b.jr_z('LWT')       # weight 0: nothing to add (the common case)
    b.jp_m('LNEG')
    b.dec_a()
    b.jr_nz('LEND')     # sentinel: end of this neuron
    b.add_hl_de()       # weight +1
    b.jr('LWT')

    b.label('LNEG')
    b.or_a()            # clear carry before each SBC
    b.sbc_hl_de()
    b.inc_a()
    b.jr_z('LWT')       # weight -1
    b.or_a()
    b.sbc_hl_de()       # weight -2
    b.jr('LWT')

    b.label('LEND')
    # Bias, read through SP so all 24 bits land in DE at once.
    b.ld_sp_mem_label('BIASP')
    b.pop_de()
    b.ld_mem_label_sp('BIASP')
    b.add_hl_de()

    # Arithmetic shift right by 2 across all three bytes.
    b.ld_mem_label_hl('TMP0')
    for _ in range(SHIFT):
        b.ld_hl_label('TMP2')
        b.sra_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()

    b.ld_a_mem_label('RELUF')
    b.or_a()
    b.jr_z('NORELU')
    b.ld_a_mem_label('TMP2')
    b.or_a()
    b.jp_p('NORELU')
    b.xor_a()
    b.ld_mem_label_a('TMP0')
    b.ld_mem_label_a('TMP1')
    b.ld_mem_label_a('TMP2')

    b.label('NORELU')
    b.ld_a_mem_label('TMP0')
    b.ld_ixd_a(0)
    b.ld_a_mem_label('TMP1')
    b.ld_ixd_a(1)
    b.ld_a_mem_label('TMP2')
    b.ld_ixd_a(2)
    b.inc_ix()
    b.inc_ix()
    b.inc_ix()

    b.ld_a_bc()
    b.cp_n(W_END_LAYER)
    b.jp_nz('LNEUR')

    b.inc_bc()
    b.ld_sp_mem_label('SPSAV')
    b.ei()
    b.ret()


def _emit_neuron_epilogue(b: EZ80Builder) -> None:
    """Finish one neuron: fold the bias, shift, ReLU, store, advance.

    Called once per neuron by the unrolled kernel, which arrives with

        IX  sum of activations whose weight is +1
        HL  sum of activations whose weight is -1, counted twice for -2
        DE  the neuron's bias, pre-negated at build time
        IY  where this neuron's output goes

    so the value wanted is ``IX - (HL - (-DE))``.  Splitting the sum this way
    is exact, not an approximation: the reference wraps the accumulator to 24
    bits, addition mod 2**24 is associative and commutative, so regrouping the
    addends cannot change the result.  What must *not* move is the >>2 - it
    floors, so it is a nonlinearity rather than a scale factor.
    """
    for label, relu in (('NEUREND', True), ('NEUREND_OUT', False)):
        b.label(label)
        b.add_hl_de()       # HL = sum(neg) - bias
        b.push_ix()
        b.pop_de()          # DE = sum(pos)
        b.ex_de_hl()
        b.ld_ix_nn(0)       # reset the positive accumulator; leaves flags alone
        b.or_a()
        b.sbc_hl_de()       # HL = sum(pos) - sum(neg) + bias, S = bit 23
        if relu:
            # A negative accumulator relus to zero whatever the shift does to
            # it, so the whole shift chain can be skipped.  Roughly 60-75% of
            # hidden neurons take this path.
            b.jp_m('NE_ZERO')
        b.jp('NE_STORE')

    b.label('NE_STORE')
    b.ld_mem_label_hl('TMP0')
    for _ in range(SHIFT):
        b.ld_hl_label('TMP2')
        b.sra_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()
    for k, name in enumerate(('TMP0', 'TMP1', 'TMP2')):
        b.ld_a_mem_label(name)
        b.ld_iyd_a(k)
    b.jp('NE_ADV')

    b.label('NE_ZERO')
    b.xor_a()
    for k in range(3):
        b.ld_iyd_a(k)

    b.label('NE_ADV')
    b.ld_hl_nn(0)           # reset the negative accumulator
    b.ld_de_nn(3)
    b.add_iy_de()
    b.ret()


def _emit_column_epilogue(b: EZ80Builder) -> None:
    """Finish one neuron of a column-major layer.

    Entered by CALL with

        HL  the neuron's accumulator, already holding bias plus contributions
        BC  the address of this neuron's column block in the *next* layer
        IX  where this neuron's output goes
        IY  where to append BC if the output turns out to be nonzero

    The append is what lets the next layer skip zero activations: it only ever
    visits columns this loop put on the list.  The test has to be on the value
    *after* the shift, because a small positive accumulator floors to zero.
    """
    b.label('NEUREND_COL')
    b.ld_de_nn(0)
    b.or_a()
    b.sbc_hl_de()       # HL is unchanged; S and Z now describe it
    b.jp_m('NC_ZERO')   # negative relus to zero whatever the shift does
    b.jp('NC_SHIFT')

    b.label('NEUREND_COL_OUT')
    b.jp('NC_SHIFT_NOLIST')  # output layer: keep negatives, build no list

    b.label('NC_SHIFT')
    _emit_shift(b)
    # Nonzero activations go on the next layer's column list.
    b.ld_a_mem_label('TMP0')
    b.ld_hl_label('TMP1')
    b.or_hl_ind()
    b.inc_hl()
    b.or_hl_ind()
    b.jp_z('NC_STORE')
    b.ld_iyd_bc(0)
    b.inc_iy()
    b.inc_iy()
    b.inc_iy()
    b.jp('NC_STORE')

    b.label('NC_SHIFT_NOLIST')
    _emit_shift(b)

    b.label('NC_STORE')
    for k, name in enumerate(('TMP0', 'TMP1', 'TMP2')):
        b.ld_a_mem_label(name)
        b.ld_ixd_a(k)
    b.label('NC_ADV')
    b.inc_ix()
    b.inc_ix()
    b.inc_ix()
    b.ret()

    b.label('NC_ZERO')
    b.xor_a()
    for k in range(3):
        b.ld_ixd_a(k)
    b.jp('NC_ADV')


def _emit_shift(b: EZ80Builder) -> None:
    """Arithmetic >>2 of HL through TMP0..TMP2, byte at a time."""
    b.ld_mem_label_hl('TMP0')
    for _ in range(SHIFT):
        b.ld_hl_label('TMP2')
        b.sra_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()
        b.dec_hl()
        b.rr_hl_ind()


def _emit_layers_column(b: EZ80Builder, model: BuildInputs,
                        phrasebook: bool = False) -> None:
    """Emit every layer column-major, visiting only nonzero activations.

    Row-major cannot skip a zero activation, because which activations are zero
    changes every step.  Column-major can: each column block is the code for
    "this input is nonzero, add its contribution everywhere it goes", and a
    layer only runs the blocks its predecessor put on the list.  On the shipped
    model that is 8,192 of 37,865 multiply-accumulates.

    Accumulating input-major rather than neuron-major moves the accumulators
    from a register into memory, costing an instruction per multiply-accumulate.
    It is worth it because ~78% of columns are skipped entirely.  Reordering
    the sum is exact for the same reason it is in the row-major kernel: the
    reference wraps to 24 bits and addition mod 2**24 is associative.

    Columns are dispatched as threaded code - the list holds addresses, and IY
    walks it.  Deliberately not SP: keeping the real stack intact means CALL
    and RET stay usable, which is what lets the per-neuron epilogue be shared
    instead of inlined 587 times.

    A phrasebook build (``phrasebook=True``) has one pass over one input, so
    layer 1 starts from its plain biases and there is no PREQ: hoisting the
    query half amortizes across a response's steps, and there is one step.
    """
    num_layers = model.num_layers
    for i in range(num_layers):
        in_buf, out_buf = layer_buffers(i, num_layers)
        weights = np.clip(np.asarray(model.weight(i)), -2, 1)
        biases = np.asarray(model.bias(i)).astype(np.int64)
        num_out, num_in = weights.shape
        last = i == num_layers - 1

        b.label(f'LAYER{i+1}')
        if i == 0 and not phrasebook:
            # QBASE already holds bias1 plus the query half's contribution,
            # which PREQ computed once for the whole response.
            for j in range(num_out):
                b.ld_hl_mem_label('QBASE', 3 * j)
                b.ld_mem_label_hl('ACC', 3 * j)
        else:
            for j, bias in enumerate(biases):
                b.ld_hl_nn(int(bias) & 0xFFFFFF)
                b.ld_mem_label_hl('ACC', 3 * j)
        b.ld_iy_label('COLLIST')

        b.label(f'CDRIVE{i+1}')
        b.ld_hl_iyd(0)      # next active column's block address
        b.ld_de_nn(3)
        b.add_iy_de()
        b.jp_hl()

        # One block per input column, holding that column's nonzero weights.
        for col in range(num_in):
            b.label(f'COL{i+1}_{col}')
            b.ld_de_mem_label(in_buf, 3 * col)   # DE = x_col
            b.ld_hl_nn(0)
            b.or_a()
            b.sbc_hl_de()
            b.ld_mem_label_hl('TMPV')
            b.ld_bc_mem_label('TMPV')            # BC = -x_col
            for out in np.nonzero(weights[:, col])[0]:
                weight = int(weights[out, col])
                b.ld_hl_mem_label('ACC', 3 * int(out))
                if weight == 1:
                    b.add_hl_de()
                else:
                    b.add_hl_bc()
                    if weight == -2:
                        b.add_hl_bc()
                b.ld_mem_label_hl('ACC', 3 * int(out))
            b.jp(f'CDRIVE{i+1}')

        # The list terminator points here, so this is where the walk lands.
        b.label(f'LEPI{i+1}')
        b.ld_ix_label(out_buf)
        b.ld_iy_label('COLLIST')
        for j in range(num_out):
            b.ld_hl_mem_label('ACC', 3 * j)
            if last:
                b.call('NEUREND_COL_OUT')
            else:
                b.ld_bc_label(f'COL{i+2}_{j}')
                b.call('NEUREND_COL')
        if last:
            b.ret()
        else:
            b.ld_hl_label(f'LEPI{i+2}')
            b.ld_iyd_hl(0)                       # terminate the next layer's list
            b.jp(f'LAYER{i+2}')

    if not phrasebook:
        _emit_query_pass(b, model)


def _emit_query_pass(b: EZ80Builder, model: BuildInputs) -> None:
    """Emit PREQ: layer 1's query half, run once per query rather than per step.

    Threaded code makes this nearly free to express. PREQ reuses layer 1's own
    column blocks and its driver; all that differs is which columns the list
    holds and where the terminator sends the walk. The result lands in QBASE,
    which layer 1 then loads instead of the bias.

    Exact rather than approximate: the accumulator is a sum mod 2**24 and
    addition mod 2**n is associative, so splitting it in two cannot move a bit.
    See :func:`libinfer.forward_hoisted`.
    """
    biases = np.asarray(model.bias(0)).astype(np.int64)

    b.label('PREQ')
    for j, bias in enumerate(biases):
        b.ld_hl_nn(int(bias) & 0xFFFFFF)
        b.ld_mem_label_hl('ACC', 3 * j)
    b.call('SCAN_QUERY')
    b.ld_iy_label('COLLIST')
    b.jp('CDRIVE1')  # the walk lands on QEPI, whose RET returns to PREQ's caller

    b.label('QEPI')
    for j in range(len(biases)):
        b.ld_hl_mem_label('ACC', 3 * j)
        b.ld_mem_label_hl('QBASE', 3 * j)
    b.ret()


def _emit_input_scan(b: EZ80Builder, name: str, columns: range,
                     terminator: str) -> None:
    """Build a layer-1 column list over part of the input vector.

    The later layers get their lists for free in the neuron epilogue, but the
    input vector is written by BUCKET_ADD, which can hit the same bucket twice.
    Appending there would put a column on the list twice and double its
    contribution, so the list is built by one pass over the finished vector.

    Two of these get emitted, over the two halves of the input. SCAN_QUERY runs
    once per query and terminates on QEPI; SCAN_CTX runs once per character and
    terminates on LEPI1. Same column blocks either way - only the list and the
    terminator differ, which is the freedom threaded code buys.
    """
    b.label(name)
    b.ld_iy_label('COLLIST')
    b.ld_de_nn(0)
    b.ld_bc_nn(3)
    for j in columns:
        b.ld_hl_mem_label('INBUF', 3 * j)
        b.or_a()
        b.sbc_hl_de()       # DE is 0, so HL survives; Z says whether it is zero
        b.jp_z(f'{name}{j}')
        b.ld_hl_label(f'COL1_{j}')
        b.ld_iyd_hl(0)
        b.add_iy_bc()
        b.label(f'{name}{j}')
    b.ld_hl_label(terminator)
    b.ld_iyd_hl(0)          # terminator: where the walk goes when the list ends
    b.ret()


def _emit_layers_unrolled(b: EZ80Builder, model: BuildInputs) -> None:
    """Emit every layer as straight-line code, one instruction pair per weight.

    With a 24-bit address space ``LD DE,(nnnnnn)`` reaches any activation in
    one instruction, so a nonzero weight costs a load and an add and a zero
    weight costs nothing.  That is what 16MB buys here; an index list would
    still have to turn each index into an address.

    Nothing in this region may use JR: it is a quarter of a megabyte long and
    every relative jump would be out of range.
    """
    num_layers = model.num_layers
    for i in range(num_layers):
        in_buf, out_buf = layer_buffers(i, num_layers)
        weights = np.clip(np.asarray(model.weight(i)), -2, 1)
        biases = np.asarray(model.bias(i)).astype(np.int64)
        epilogue = 'NEUREND_OUT' if i == num_layers - 1 else 'NEUREND'

        b.label(f'LAYER{i+1}')
        b.ld_iy_label(out_buf)
        b.ld_hl_nn(0)
        b.ld_ix_nn(0)

        for row, bias in zip(weights, biases, strict=True):
            for col, weight in neuron_ops(row):
                b.ld_de_mem_label(in_buf, 3 * col)
                if weight == 1:
                    b.add_ix_de()
                else:
                    b.add_hl_de()
                    if weight == -2:
                        b.add_hl_de()
            b.ld_de_nn((-int(bias)) & 0xFFFFFF)
            b.call(epilogue)
    b.ret()


#: Kernels this backend can emit, fastest first.  See the module docstring.
KERNELS = ('column', 'row', 'compact')


def _emit_argmax(b: EZ80Builder) -> None:
    """First-wins argmax over OUTBUF, leaving the index in RESULT.

    Lifted out of build_autoreg so a program that is not a chat loop can
    still classify - the oracle needs the phrase index and none of the
    printing that used to follow it.
    """
    # First-wins argmax over OUTBUF, matching libinfer.argmax.
    #
    # SP walks the buffer so each POP reads a 24-bit logit and advances in one
    # instruction; the loop ends by comparing SP against OUTEND rather than
    # counting in B.  A byte counter is what limited this to 256 outputs while
    # the module advertised no width limit at all, and it failed silently: a
    # 299-entry charset assembled to `LD B,43` and argmaxed over the first 44
    # logits.  Index and result are 24-bit for the same reason.
    b.label('ARGMAX')
    b.ld_mem_label_sp('SPSAV')
    b.di()
    b.ld_sp_label('OUTBUF')
    b.ld_hl_nn(0)
    b.ld_mem_label_hl('MAXI')
    b.ld_mem_label_hl('IDX')
    b.pop_de()  # running maximum = OUTBUF[0]

    b.label('AMLP')
    b.ld_mem_label_sp('SPTMP')
    b.ld_hl_mem_label('SPTMP')
    b.ld_bc_label('OUTEND')
    b.or_a()
    b.sbc_hl_bc()
    b.jp_z('AMDONE')

    b.ld_hl_mem_label('IDX')
    b.inc_hl()
    b.ld_mem_label_hl('IDX')

    b.pop_hl()
    b.ld_mem_label_hl('TMPV')
    b.or_a()
    b.sbc_hl_de()
    b.jp_m('AMLP')  # below the running maximum
    b.jp_z('AMLP')  # equal: the earlier index wins
    b.ld_de_mem_label('TMPV')
    b.ld_hl_mem_label('IDX')
    b.ld_mem_label_hl('MAXI')
    b.jp('AMLP')

    b.label('AMDONE')
    b.ld_sp_mem_label('SPSAV')
    b.ei()
    b.ld_hl_mem_label('MAXI')
    b.ld_mem_label_hl('RESULT')
    b.ret()


def _emit_tokenizer_helpers(b: EZ80Builder, plat: libnn.Platform,
                            position_bands: int) -> None:
    """LOWER, BUCKET_ADD, TOKENIZE, TOK_HASH and the two hash steps.

    The eZ80's own rather than libnn's, because an activation is three
    bytes here and libnn's writes two - which is precisely what
    ``Platform.activation_size`` documents as the boundary of what can be
    shared. Lifted out of build_autoreg so a program that classifies
    without being a chat loop can tokenize.
    """
    # === LOWER: fold A-Z to lower case, everything else untouched ============
    b.label('LOWER')
    libnn.emit_lower_fold(b)
    b.ret()

    # === BUCKET_ADD: (HL + 3*A) += 32 ========================================
    b.label('BUCKET_ADD')
    b.ld_de_nn(0)
    b.ld_e_a()
    b.ex_de_hl()        # HL = index, DE = base
    b.add_hl_hl()       # index * 2
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()       # index * 3
    b.add_hl_de()       # base + index * 3
    b.ld_a_hl()
    b.add_a_n(32)
    b.ld_hl_a()
    b.inc_hl()
    b.ld_a_hl()
    b.adc_a_n(0)
    b.ld_hl_a()
    b.inc_hl()
    b.ld_a_hl()
    b.adc_a_n(0)
    b.ld_hl_a()
    b.ret()

    # === TOKENIZE: query text -> first 128 buckets ===========================
    # The state machine is libnn's, and TOK_HASH below is what it calls. The
    # trigram walk is where a divergence would be silent: two tokenizers that
    # disagree still both produce plausible buckets, and the model would answer
    # confidently and wrongly rather than crash.
    libnn.emit_tokenizer(b, plat, position_bands)

    b.label('TOK_DONE')
    b.ret()

    # === TOK_HASH: h = ((c1*31 + c2)*31 + c3), bucket = h & 127 ==============
    b.label('TOK_HASH')
    b.push_de()
    b.ld_hl_nn(0)
    if position_bands > 1:
        # Seed with the trigram's position band. Both halves come from libnn:
        # they are libinfer.position_band and BAND_SEED rendered as code, and
        # the Z80 backends emit the same two.
        libnn.emit_band_index(b, position_bands)
        b.ld_l_a()
        libnn.emit_times_seven(b, 'bc')
        b.call('HASH_STEP2')
        b.ld_a_mem_label('TOKC1')
        b.call('HASH_ADD')
    else:
        b.ld_a_mem_label('TOKC1')
        b.ld_l_a()
    b.call('HASH_STEP2')
    b.ld_a_mem_label('TOKC2')
    b.call('HASH_ADD')
    b.call('HASH_STEP2')
    b.ld_a_mem_label('TOKC3')
    b.call('HASH_ADD')
    b.ld_a_l()
    b.and_n(NUM_BUCKETS - 1)
    b.ld_hl_label('INBUF')
    b.call('BUCKET_ADD')
    b.pop_de()
    if position_bands > 1:
        b.ld_a_mem_label('TOKPOS')
        b.inc_a()
        b.ld_mem_label_a('TOKPOS')
    b.ret()

    # HASH_STEP2: HL *= 31.  HASH_ADD: HL += A.
    # Both scratch through BC, never DE: CTX_HASH keeps its character pointer
    # in DE across the whole loop.
    b.label('HASH_STEP2')
    libnn.emit_hash_step(b, 'bc')
    b.ret()

    b.label('HASH_ADD')
    b.ld_bc_nn(0)
    b.ld_c_a()
    b.add_hl_bc()
    b.ret()



def build_autoreg(model_path: str = 'command_model_autoreg.pt',
                  max_output_len: int = MAX_OUTPUT_LEN,
                  org: int = AGON_LOAD_ADDR,
                  kernel: str = 'auto',
                  phrases_file: str = 'PHRASES.DAT') -> EZ80Builder:
    """Build the eZ80 inference binary.

    ``kernel`` selects how the layers are emitted; see :data:`KERNELS`.  The
    default, ``'auto'``, takes the fastest kernel whose image still fits in
    Agon SRAM, which is the same fastest-that-fits policy build.py already
    applies to the CP/M target.

    A model carrying a phrasebook (``_architecture['phrases']``) builds a
    classifier instead of a character decoder: one forward pass over the query
    buckets, one argmax, and the reply printed from ``phrases_file`` on the SD
    card.  The builder's ``phrase_blob`` attribute holds the bytes that file
    must contain.
    """
    if kernel == 'auto':
        return _build_fastest_that_fits(model_path, max_output_len, org,
                                        phrases_file)
    if kernel not in KERNELS:
        raise ValueError(
            f"unknown kernel {kernel!r}; choose from {['auto', *KERNELS]}"
        )
    # The input/output split is a Z80 concern - here layers may be any width -
    # so the report leaves it out.
    model = load_for_build(model_path, report_io=False)
    layer_sizes = model.layer_sizes
    num_layers, output_size = model.num_layers, model.output_size
    # Named locally: the emit sequence below reads them dozens of times.
    charset, eos_idx = model.charset, model.eos_idx
    position_bands = model.position_bands
    phrases, phrasebook = model.phrases, model.is_phrasebook
    if output_size < 2:
        raise ValueError("charset must have at least two entries")

    # Hoisting layer 1's query half pays off across the steps of one
    # response, so a one-step phrasebook skips PREQ - but skipping zero
    # activations pays per pass, which is all the column kernel needs.
    # The input scan then covers the whole input vector (SCAN_IN), because
    # there is no context half to split off.
    if phrasebook and model.input_size != NUM_BUCKETS:
        raise ValueError(
            f"a phrasebook takes {NUM_BUCKETS} query buckets, not "
            f"{model.input_size}: there is no context to encode when the "
            f"whole answer is chosen in one pass")

    # The decode table is sized by the charset or the phrase list and ARGMAX by
    # the weight shapes, and nothing ever compared them. Disagreeing, PRINTCH
    # indexes past the table into the scratch bytes that follow it and prints
    # whatever is there - no crash, no warning, just the wrong answer.
    if output_size != model.num_outputs:
        label = 'phrases' if phrasebook else 'charset entries'
        raise ValueError(
            f"the output layer has {output_size} neurons but there are "
            f"{model.num_outputs} {label}; one of them is wrong")

    # Keyed off `phrases` rather than `phrasebook`: the two say the same thing,
    # but only this one shows that encode_phrases cannot be handed None.
    phrase_blob = encode_phrases(phrases) if phrases is not None else b''


    # The unrolled kernels bake the weights into the code, so only the compact
    # one needs a weight stream and a bias table in the data section.
    if kernel == 'compact':
        weight_blobs = [encode_weights(w) for w in model.weights()]
        bias_blob = b''.join(encode_biases(bias) for bias in model.biases())
    else:
        weight_blobs, bias_blob = [], b''

    plat = libagon.AgonPlatform()
    b = EZ80Builder(org=org)
    agon_header(b, 'START')

    # === Entry ===============================================================
    # The program around the layers - entry, prompt loop, line editor - is the
    # machine's, not this backend's, and lives in libagon.
    def answer(bb: EZ80Builder) -> None:
        if phrasebook:
            bb.call('CLASSIFY')
        else:
            bb.call('CLEAR_CTX')
            bb.call('GENERATE')

    libagon.emit_entry(b, answer,
                       phrase_bytes=len(phrase_blob) if phrasebook else None)
    libagon.emit_newline(b)
    libagon.emit_read_input(b)

    # === GENERATE ============================================================
    # The loop itself is the Z80's. Only the EOS test differs: RESULT is 24
    # bits here, so a CP cannot reach it.
    def eos_test(bb: Z80Builder, idx: int) -> None:
        bb.ld_hl_mem_label('RESULT')
        bb.ld_bc_nn(idx)
        bb.or_a()
        bb.sbc_hl_bc()

    libnn.emit_generate(b, eos_idx, max_output_len,
                        lambda bb: bb.call('INFER'),
                        hoist_query=(kernel == 'column' and not phrasebook),
                        emit_eos_test=eos_test)

    # === PRINTCH =============================================================
    b.label('PRINTCH')
    b.ld_hl_label('CHARTBL')
    b.ld_bc_mem_label('RESULT')
    b.add_hl_bc()
    b.ld_a_hl()
    b.rst(MOS_OUTCHAR)
    b.ret()

    if phrasebook:
        # === CLASSIFY ========================================================
        # The whole of a phrasebook's inference. One pass, one argmax, one
        # reply - no GENLOOP, no EOS compare, no context to slide, because
        # there is nothing for a context window to condition on when the
        # entire answer is chosen in a single step.
        b.label('CLASSIFY')
        b.call('INFER')
        b.call('ARGMAX')
        b.call('PRINT_PHRASE')
        b.jp('PRNL')

        # === PRINT_PHRASE ====================================================
        # RESULT indexes the offset table that follows the count, so the entry
        # is at PHRBUF + 3 + 3*RESULT. Its contents are relative to the start
        # of the file, so PHRBUF is added once more to reach the text.
        b.label('PRINT_PHRASE')
        b.ld_hl_mem_label('RESULT')
        b.add_hl_hl()               # RESULT*2
        b.ld_bc_mem_label('RESULT')
        b.add_hl_bc()               # RESULT*3
        b.ld_bc_label('PHRBUF')
        b.add_hl_bc()
        # There is no LD HL,(HL); the eZ80's 24-bit indirect load is through
        # IX, so the pointer goes there. The +3 skips the count.
        b.push_hl()
        b.pop_ix()
        b.ld_hl_ixd(3)              # HL = offset from the start of the file
        b.ld_bc_label('PHRBUF')
        b.add_hl_bc()

        b.label('PP_LOOP')
        b.ld_a_hl()
        b.or_a()
        b.ret_z()
        b.rst(MOS_OUTCHAR)
        b.inc_hl()
        b.jr('PP_LOOP')

    # === INFER: run every layer ==============================================
    # Buffers ping-pong; the assignment is fixed at build time so the layer
    # setup is unrolled rather than table-driven.
    b.label('INFER')
    if kernel == 'compact':
        b.ld_hl_label('BIASES')
        b.ld_mem_label_hl('BIASP')

        for i in range(num_layers):
            in_buf, out_buf = layer_buffers(i, num_layers)
            b.label(f'LAYER{i+1}')
            b.ld_hl_label(in_buf)
            b.ld_mem_label_hl('INBASE')
            b.ld_ix_label(out_buf)
            b.ld_bc_label(f'WTS{i+1}')
            b.ld_a_n(0 if i == num_layers - 1 else 1)
            b.ld_mem_label_a('RELUF')
            b.call('LAYER')
        b.ret()
    elif kernel == 'column':
        # The column list is rebuilt from the input each pass. A character
        # decoder scans only the context half here - the query half cannot
        # change mid-response, so PREQ dealt with it once before the loop. A
        # phrasebook has one pass over one input, so SCAN_IN covers it whole.
        b.call('SCAN_IN' if phrasebook else 'SCAN_CTX')
        b.jp('LAYER1')
    else:
        # The unrolled layers are emitted last, after every JR-using routine,
        # so a quarter of a megabyte of straight-line code cannot put any
        # relative jump out of range.
        b.jp('LAYER1')

    # === LAYER ===============================================================
    if kernel == 'compact':
        _emit_layer_compact(b)


    # === ARGMAX ==============================================================
    _emit_argmax(b)

    _emit_tokenizer_helpers(b, plat, position_bands)

    # === CLEAR_CTX / UPDATE_CTX / ENCODE_CTX =================================
    # Identical on both machines - the context window is eight bytes wide
    # whatever an activation costs, and DJNZ counts it on the eZ80 too.
    libnn.emit_clear_ctx(b, unrolled=False)

    b.label('UPDATE_CTX')
    b.ld_hl_label('CTXCHARS')
    b.inc_hl()
    b.ld_de_label('CTXCHARS')
    b.ld_bc_nn(CONTEXT_LEN - 1)
    b.ldir()

    b.ld_hl_label('CHARTBL')
    b.ld_bc_mem_label('RESULT')
    b.add_hl_bc()
    b.ld_a_hl()
    b.call('LOWER')
    b.ld_hl_label('CTXLAST')
    b.ld_hl_a()
    b.jp('ENCODE_CTX')

    # The loop nest is the Z80's: it walks CTXCHARS and calls CTX_HASH, and
    # never touches an activation. Only clearing the buckets knows how wide one
    # is, and the eZ80 gives them their own buffer rather than an offset.
    def clear_ctx_buckets(bb: Z80Builder) -> None:
        bb.ld_hl_label('CTXBUF')
        bb.ld_de_label('CTXBUF')
        bb.inc_de()
        bb.ld_bc_nn(NUM_BUCKETS * 3 - 1)
        bb.ld_hl_n(0)
        bb.ldir()

    libnn.emit_encode_ctx(b, plat, emit_clear=clear_ctx_buckets)

    # === CTX_HASH: hash CTXN characters from CTXPOS, seeded with CTXPOS*7 ====
    b.label('CTX_HASH')
    b.ld_hl_nn(0)
    b.ld_a_mem_label('CTXPOS')
    b.ld_l_a()
    libnn.emit_times_seven(b, 'de')  # pos * 7

    b.push_hl()
    b.ld_hl_label('CTXCHARS')
    b.ld_bc_nn(0)
    b.ld_a_mem_label('CTXPOS')
    b.ld_c_a()
    b.add_hl_bc()
    b.ex_de_hl()  # DE = &CTXCHARS[pos]
    b.pop_hl()

    b.ld_a_mem_label('CTXN')
    b.ld_b_a()

    b.label('CTX_HLOOP')
    b.push_bc()
    b.call('HASH_STEP2')
    b.ld_a_de()
    b.call('HASH_ADD')
    b.inc_de()
    b.pop_bc()
    b.djnz('CTX_HLOOP')

    b.ld_a_l()
    b.and_n(NUM_BUCKETS - 1)
    b.ld_hl_label('CTXBUF')
    b.jp('BUCKET_ADD')

    # === UNROLLED LAYERS =====================================================
    # Emitted after every routine that uses JR, because from here on the code
    # is far too long for a relative jump to reach across.
    if kernel == 'row':
        _emit_neuron_epilogue(b)
        _emit_layers_unrolled(b, model)
    elif kernel == 'column':
        _emit_column_epilogue(b)
        if phrasebook:
            # One input, one pass: scan the whole vector, terminate on layer
            # 1's epilogue. No query half to hoist, so no SCAN_QUERY or PREQ.
            _emit_input_scan(b, 'SCAN_IN', range(NUM_BUCKETS), 'LEPI1')
        else:
            _emit_input_scan(b, 'SCAN_QUERY', range(NUM_BUCKETS), 'QEPI')
            _emit_input_scan(b, 'SCAN_CTX',
                             range(NUM_BUCKETS, layer_sizes[0]), 'LEPI1')
        _emit_layers_column(b, model, phrasebook=phrasebook)

    # === DATA ================================================================
    libnn.emit_charset_table(b, charset)

    scratch = ['TOKLEN', 'TOKC1', 'TOKC2', 'TOKC3']
    if position_bands > 1:
        scratch.append('TOKPOS')
    scratch += ['CTXPOS', 'CTXN', 'GENCNT', 'RELUF', 'INPLEN']
    for name in scratch:
        b.label(name)
        b.db(0)

    # A three-byte scratch the layer shifts in place, byte by byte.
    for name in ('TMP0', 'TMP1', 'TMP2'):
        b.label(name)
        b.db(0)

    # MAXI/RESULT are 24-bit so the output layer may be any width; the rest are
    # pointers and scratch that were already 24-bit.
    for name in ('SPSAV', 'SPTMP', 'INBASE', 'BIASP', 'TMPV', 'MAXI', 'IDX',
                 'RESULT'):
        b.label(name)
        b.d24(0)

    b.label('CTXCHARS')
    b.ds(CONTEXT_LEN - 1)
    b.label('CTXLAST')
    b.db(0)

    b.label('INPBUF')
    b.ds(MAX_INPUT_LEN + 1)

    # Activation buffers: 24-bit values.  The compact kernel pops one
    # activation past the last weight, so it needs three bytes of slack; the
    # unrolled kernels address exactly the elements they use and need none.
    slack = 3 if kernel == 'compact' else 0
    b.label('INBUF')
    b.ds(NUM_BUCKETS * 3)
    b.label('CTXBUF')
    b.ds(NUM_BUCKETS * 3)
    b.ds(slack)

    hidden = layer_sizes[1:-1] or [layer_sizes[-1]]
    max_hidden = max(hidden)
    b.label('BUF_A')
    b.ds(max_hidden * 3 + slack)
    b.label('BUF_B')
    b.ds(max_hidden * 3 + slack)
    b.label('OUTBUF')
    b.ds(output_size * 3)
    b.label('OUTEND')
    b.ds(3)

    if kernel == 'column':
        # One 24-bit accumulator per neuron of the widest layer, plus the list
        # of active columns - one entry per input of the widest layer, and one
        # more for the terminator.
        widest_out = max(layer_sizes[1:])
        widest_in = max(layer_sizes[:-1])
        b.label('ACC')
        b.ds24(widest_out)
        b.label('COLLIST')
        b.ds24(widest_in + 1)
        if not phrasebook:
            # Layer 1's bias with the query half folded in, once per query.
            b.label('QBASE')
            b.ds24(layer_sizes[1])

    if phrasebook:
        b.label('PHRNAME')
        b.ascii(phrases_file)
        b.db(0)
        b.label('PHRERR')
        b.ascii(f"Could not load {phrases_file} from the SD card.")
        b.db(13)
        b.db(10)
        b.db(0)
        # Sized from the file the build produced, so mos_load's BC argument and
        # the buffer cannot disagree.
        b.label('PHRBUF')
        b.ds(len(phrase_blob))

    if kernel == 'compact':
        b.label('BIASES')
        b.blob(bias_blob)
        for i, blob in enumerate(weight_blobs, start=1):
            b.label(f'WTS{i}')
            b.blob(blob)

    # Layer 0 reads a 256-long input vector through the INBUF label alone, so
    # the context half must sit immediately after the query half.  Today that
    # holds because of the order of the two `ds` calls above; assert it rather
    # than leave the unrolled offsets depending on an accident of layout.
    assert b.labels['CTXBUF'] == b.labels['INBUF'] + NUM_BUCKETS * 3, \
        "INBUF and CTXBUF must be contiguous"

    b.kernel = kernel
    #: The bytes PHRASES.DAT must contain, or b'' for a character decoder.
    #: The build writes it beside the .bin; both come from one encode_phrases
    #: call, so the offsets the program indexes and the text it prints cannot
    #: drift apart.
    b.phrase_blob = phrase_blob
    b.phrases_file = phrases_file if phrasebook else None
    return b


def write_phrase_file(builder: EZ80Builder, binary_path: str) -> str | None:
    """Write the phrasebook beside the .bin, under the name the binary loads.

    Both files come out of one build, so the offsets the program indexes and
    the text it prints cannot drift apart - which they would the moment the
    two were produced by separate commands.
    """
    # The blob and its name are set together, so testing the name covers both
    # and leaves nothing for the join below to be handed None.
    if not builder.phrase_blob or builder.phrases_file is None:
        return None
    path = os.path.join(os.path.dirname(binary_path) or '.', builder.phrases_file)
    with open(path, 'wb') as fh:
        fh.write(builder.phrase_blob)
    print(f"Wrote {len(builder.phrase_blob):,} bytes to {path} - "
          f"copy it onto the card beside the binary")
    return path


def _build_fastest_that_fits(model_path: str, max_output_len: int, org: int,
                             phrases_file: str = 'PHRASES.DAT') -> EZ80Builder:
    """Take the first kernel in :data:`KERNELS` whose image fits Agon SRAM.

    The unrolled kernels trade size for speed, so a large enough model can only
    be built with the compact one.  Preserving that fallback is the whole point
    of keeping it around.
    """
    candidates = list(KERNELS)

    last = len(candidates) - 1
    for i, kernel in enumerate(candidates):
        builder = build_autoreg(model_path, max_output_len, org, kernel=kernel,
                                phrases_file=phrases_file)
        size = len(builder.build())
        fits = org + size <= AGON_LOAD_ADDR + AGON_MAX_IMAGE
        if fits:
            return builder
        if i == last:
            # Nothing smaller left to try. Say so plainly rather than hand back
            # a binary no Agon can load without comment; verify_artifacts will
            # reject it too, but whoever ran this build should hear it first.
            print(f"\nWARNING: even the {kernel} kernel needs {size:,} bytes, "
                  f"more than the {AGON_MAX_IMAGE:,} an Agon can load. "
                  f"The model is too large for this target.")
            return builder
        print(f"\nThe {kernel} kernel needs {size:,} bytes, more than the "
              f"{AGON_MAX_IMAGE:,} an Agon can load; trying the next kernel.")
    raise AssertionError("unreachable: the last kernel is always accepted")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Build eZ80 (Agon) inference binary')
    parser.add_argument('--model', '-m', default='command_model_autoreg.pt',
                        help='Model file to load (.npz or .pt)')
    parser.add_argument('--output', '-o', default='CHAT.bin',
                        help='Output MOS binary')
    parser.add_argument('--max-output-len', type=int, default=MAX_OUTPUT_LEN,
                        help='Maximum characters generated per response')
    parser.add_argument('--kernel', '-k', default='auto', choices=['auto', *KERNELS],
                        help='Layer kernel (default: auto = fastest that fits)')
    parser.add_argument('--phrases', default='PHRASES.DAT',
                        help='Name the phrasebook binary loads its replies '
                             'from, and the file written beside it. Two '
                             'phrasebooks in one directory need two names')
    args = parser.parse_args()

    print("Building eZ80 CHAT.bin...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len,
                      kernel=args.kernel, phrases_file=args.phrases)

    b.report_labels(KEY_LABELS)

    b.save(args.output)
    if b.phrase_blob:
        write_phrase_file(b, args.output)
    size = len(b.code)
    print(f"\nTotal size: {size:,} bytes ({size / 1024:.1f} KB)")
    print(f"Loads at {AGON_LOAD_ADDR:06X}h, runs in ADL mode; "
          f"{AGON_MAX_IMAGE - size:,} bytes of Agon SRAM to spare")


if __name__ == '__main__':
    main()
