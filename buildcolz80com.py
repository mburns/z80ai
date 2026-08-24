#!/usr/bin/env python3
"""
Build a CP/M .COM whose layers run column-major, skipping zero activations.

``buildfastz80com.py`` walks each neuron's nonzero weights, which skips the ~73%
of the model that is zero but nothing else.  The activations are sparser than
the weights: measured over the shipped models the input vector is ~17% nonzero
and the hidden layers 14-45%, and a row-major kernel cannot exploit that,
because which activations are zero changes with every character emitted.

Column-major can.  Each input column owns the list of neurons its weight reaches,
so a layer runs only the columns whose activation is nonzero:

    for each active column c:              # ~22% of them
        x = act[c]
        for each nonzero weight w[j][c]:
            acc[j] += w * x

That takes one forward pass from 37,940 multiply-accumulates to 8,474.  The
price is that the accumulator moves from a register into memory, so a
multiply-accumulate costs more instructions than the row-major kernel's; the
trade is worth it because most of them never happen.

This is the same transformation ``buildez80.py``'s ``column`` kernel makes, but
without 16MB of address space to unroll into: at ~10 bytes of code per weight,
unrolling this model would need 378KB.  So the weights stay data, and the inner
loop walks them with ``IY``.

Reordering the sum is exact, not an approximation: the reference wraps to 16
bits and addition mod 2**16 is associative, so any grouping gives the same
residue.  What may *not* move is the ``>>2``, which floors - see EZ80.md.

Accumulators, activations and column lists all live in 256-byte aligned pages
with the low bytes in one page and the high bytes in the next, so a neuron index
is a whole 8-bit address: ``LD L,j`` reaches its accumulator, and ``INC H``
reaches the other half of it.
"""

from __future__ import annotations

import argparse

import numpy as np

import libcpm
import libinfer
import libnn
from libcpm import CPMPlatform
from libinfer import MAX_OUTPUT_LEN, validate_z80_layers
from libz80 import Z80Builder

#: The three weight values that reach the inner loop; zero is what we skip.
WEIGHT_VALUES = (-2, -1, 1)


class ColumnCPMPlatform(CPMPlatform):
    """Same I/O as the packed CP/M build; the weights are laid out by column."""

    name = "CP/M (column-major)"


def column_records(weights: np.ndarray) -> tuple[bytes, list[int]]:
    """Lay out one layer's weights by input column.

    Each column gets three counted lists of neuron indices, one per weight
    value, in the order :data:`WEIGHT_VALUES`. The driver picks the add or
    subtract for the whole list rather than dispatching per weight.

    Args:
        weights: The layer's ``[out, in]`` weight matrix.

    Returns:
        The record blob, and the byte offset of each column's record.

    Raises:
        ValueError: If a column holds more than 255 weights of one value, which
            the count byte cannot express, or if the layer is too wide for an
            8-bit neuron index.
    """
    w = np.clip(np.asarray(weights), -2, 1).astype(np.int8)
    num_out, num_in = w.shape
    if num_out > 256:
        raise ValueError(f"layer of {num_out} neurons exceeds the 8-bit index")

    blob = bytearray()
    offsets = []
    for col in range(num_in):
        offsets.append(len(blob))
        column = w[:, col]
        for value in WEIGHT_VALUES:
            rows = np.nonzero(column == value)[0]
            if len(rows) > 255:
                raise ValueError(
                    f"column {col} has {len(rows)} weights of {value}, more than "
                    f"a count byte holds"
                )
            blob.append(len(rows))
            blob.extend(int(r) for r in rows)
    return bytes(blob), offsets


def _act_page(index: int) -> str:
    """Which activation page layer ``index`` reads. Layer 1 reads ACT_A."""
    return "ACT_A" if index % 2 == 0 else "ACT_B"


def emit_split_scan(b: Z80Builder) -> None:
    """Emit SPLITSCAN: split interleaved activations and list the nonzero ones.

    ``HL`` is the interleaved source, ``DE`` the split destination, ``IY`` the
    column list to append to and ``B`` the count. Splitting and scanning are one
    pass because they read the same bytes, and because the scan has to happen
    here rather than in the tokenizer: ``TOK_HASH`` can hit the same bucket
    twice, and a column listed twice would contribute twice.
    """
    b.label("SPLITSCAN")
    b.ld_a_hl()
    b.inc_hl()
    b.ld_de_a()
    b.ld_c_a()  # keep the low byte to test the pair for zero
    b.ld_a_hl()
    b.inc_hl()
    b.inc_d()
    b.ld_de_a()
    b.dec_d()
    b.or_c()
    b.jr_z("SS_NEXT")
    b.ld_iyd_e(0)  # E is the index within the page, which is what a column is
    b.inc_iy()

    b.label("SS_NEXT")
    b.inc_e()
    b.djnz("SPLITSCAN")
    b.jp("SET_NCOL")


def emit_set_ncol(b: Z80Builder) -> None:
    """Emit SET_NCOL: record how many columns IY appended, then return.

    Sixteen bits, because a 256-wide layer can list all 256 of its neurons and a
    byte would call that zero - which the driver would read as "nothing to do"
    and skip the layer entirely, silently.
    """
    b.label("SET_NCOL")
    b.push_iy()
    b.pop_hl()
    b.ld_de_label("COLLIST")
    b.or_a()
    b.sbc_hl_de()
    b.ld_mem_label_hl("NCOL")
    b.ret()


def emit_driver(b: Z80Builder, index: int, in_page: str) -> None:
    """Emit DRIVE<n>: run every listed column of one layer into ACC.

    ``IY`` walks the column's weight records, ``HL`` addresses the accumulator
    page with ``L`` as the neuron index, and ``DE`` holds the column's
    activation for as long as the column lasts. Positive weights add ``DE`` and
    negative ones subtract it, so no per-weight sign test is needed and no
    negated copy has to be built.
    """
    tag = index + 1
    b.label(f"DRIVE{tag}")
    b.ld_hl_mem_label("NCOL")
    b.ld_a_h()
    b.or_l()
    b.ret_z()
    # CDCNT counts down after the work, so a stored zero runs 256 times - which
    # is exactly what a full list of 256 columns needs.
    b.ld_a_l()
    b.ld_mem_label_a("CDCNT")
    b.ld_hl_label("COLLIST")
    b.ld_mem_label_hl("CDPTR")

    b.label(f"CD{tag}")
    b.ld_hl_mem_label("CDPTR")
    b.ld_a_hl()
    b.inc_hl()
    b.ld_mem_label_hl("CDPTR")

    # DE = act[c], read out of the split input page.
    b.ld_l_a()
    b.ld_h_page(in_page)
    b.ld_e_hl()
    b.inc_h()
    b.ld_d_hl()

    # IY = this column's weight record.
    b.ld_l_a()
    b.ld_h_n(0)
    b.add_hl_hl()
    b.ld_bc_label(f"COLPTR{tag}")
    b.add_hl_bc()
    b.ld_c_hl()
    b.inc_hl()
    b.ld_b_hl()
    b.push_bc()
    b.pop_iy()

    b.ld_hl_label("ACC")  # H selects the accumulator page, L is set per weight

    for value in WEIGHT_VALUES:
        slot = value + 2
        b.ld_a_iyd(0)
        b.inc_iy()
        b.or_a()
        b.jr_z(f"CS{tag}_{slot}")
        b.ld_b_a()

        b.label(f"CL{tag}_{slot}")
        b.ld_l_iyd(0)
        b.inc_iy()
        for _ in range(2 if value == -2 else 1):
            # INC H and LD (HL),A leave the carry alone, so the low half's
            # borrow or carry is still there for the high half.
            b.ld_a_hl()
            if value > 0:
                b.add_a_e()
            else:
                b.sub_e()
            b.ld_hl_a()
            b.inc_h()
            b.ld_a_hl()
            if value > 0:
                b.adc_a_d()
            else:
                b.sbc_a_d()
            b.ld_hl_a()
            b.dec_h()
        b.djnz(f"CL{tag}_{slot}")

        b.label(f"CS{tag}_{slot}")

    b.ld_a_mem_label("CDCNT")
    b.dec_a()
    b.ld_mem_label_a("CDCNT")
    b.jp_nz(f"CD{tag}")
    b.ret()


def emit_hidden_epilogue(b: Z80Builder, index: int, out_page: str,
                         out_size: int) -> None:
    """Emit EPI<n> for a hidden layer: scale, ReLU, store, and list.

    The ReLU is decided before the scale rather than after. The shift is
    arithmetic and floors, so a negative accumulator can only come out negative;
    testing its sign here skips the whole shift chain for the 60-75% of hidden
    neurons that ReLU to zero anyway.

    The same pass builds the next layer's column list. Each neuron is written
    exactly once, so unlike the input scan this cannot list a column twice.
    """
    tag = index + 1
    b.label(f"EPI{tag}")
    b.ld_iy_label("COLLIST")
    b.ld_hl_label("ACC")
    b.ld_b_n(out_size if out_size <= 255 else 0)

    b.label(f"EL{tag}")
    b.ld_e_hl()
    b.inc_h()
    b.ld_d_hl()
    b.bit_7_d()
    b.jr_z(f"ES{tag}")
    b.ld_de_nn(0)
    b.jr(f"EW{tag}")

    b.label(f"ES{tag}")
    b.sra_d()
    b.rr_e()
    b.sra_d()
    b.rr_e()

    b.label(f"EW{tag}")
    b.ld_h_page(out_page)
    b.ld_hl_e()
    b.inc_h()
    b.ld_hl_d()
    b.ld_a_d()
    b.or_e()
    b.jr_z(f"EN{tag}")
    b.ld_iyd_l(0)
    b.inc_iy()

    b.label(f"EN{tag}")
    b.ld_h_page("ACC")
    b.inc_l()
    b.djnz(f"EL{tag}")
    b.jp("SET_NCOL")


def emit_output_epilogue(b: Z80Builder, index: int, out_size: int) -> None:
    """Emit EPI<n> for the last layer: scale into OUTBUF, no ReLU, no list.

    OUTBUF is interleaved rather than split so ARGMAX can be the shared one.
    """
    tag = index + 1
    b.label(f"EPI{tag}")
    b.ld_iy_label("OUTBUF")
    b.ld_hl_label("ACC")
    b.ld_b_n(out_size if out_size <= 255 else 0)

    b.label(f"EL{tag}")
    b.ld_e_hl()
    b.inc_h()
    b.ld_d_hl()
    b.dec_h()
    b.sra_d()
    b.rr_e()
    b.sra_d()
    b.rr_e()
    b.ld_iyd_e(0)
    b.ld_iyd_d(1)
    b.inc_iy()
    b.inc_iy()
    b.inc_l()
    b.djnz(f"EL{tag}")
    b.ret()


def emit_load_acc(b: Z80Builder, lo: str, hi: str, count: int) -> None:
    """Emit the two block copies that seed ACC for a layer.

    Two copies rather than one: the low and high halves of ACC are a page apart,
    so a layer narrower than 256 neurons would otherwise land its high bytes in
    the middle of the low page.
    """
    for source, offset in ((lo, 0), (hi, 256)):
        b.ld_hl_label(source)
        b.ld_de_label("ACC", offset)
        b.ld_bc_nn(count)
        b.ldir()


def emit_layers(b: Z80Builder, layer_sizes: list[int]) -> None:
    """Emit LAYER<n> for every layer: seed ACC, drive the columns, run the tail."""
    num_layers = len(layer_sizes) - 1
    for i in range(num_layers):
        tag = i + 1
        b.label(f"LAYER{tag}")
        if i == 0:
            # QBASE already holds bias 1 plus the query half's contribution.
            emit_load_acc(b, "QBASELO", "QBASEHI", layer_sizes[1])
        else:
            emit_load_acc(b, f"BIASLO{tag}", f"BIASHI{tag}", layer_sizes[i + 1])
        b.call(f"DRIVE{tag}")
        b.jp(f"EPI{tag}")


def emit_preq(b: Z80Builder, layer_sizes: list[int]) -> None:
    """Emit PREQ: fold layer 1's query columns into its bias, once per query.

    Only the context half of the input changes while a response is generated, so
    layer 1's contribution from the query is a constant for the whole response.
    PREQ reuses layer 1's own driver and its column records; all that differs is
    which columns end up on the list. See :func:`libinfer.forward_hoisted`.
    """
    b.label("PREQ")
    emit_load_acc(b, "BIASLO1", "BIASHI1", layer_sizes[1])
    b.ld_hl_label("INBUF")
    b.ld_de_label("ACT_A")
    b.ld_iy_label("COLLIST")
    b.ld_b_n(libnn.NUM_BUCKETS)
    b.call("SPLITSCAN")
    b.call("DRIVE1")
    for offset, target in ((0, "QBASELO"), (256, "QBASEHI")):
        b.ld_hl_label("ACC", offset)
        b.ld_de_label(target)
        b.ld_bc_nn(layer_sizes[1])
        b.ldir()
    b.ret()


def emit_inference(layer_sizes: list[int]):
    """One forward pass: re-split the context half, then run every layer."""
    def emit(b: Z80Builder) -> None:
        b.ld_hl_label("INBUF", libnn.CONTEXT_OFFSET)
        b.ld_de_label("ACT_A", libnn.NUM_BUCKETS)
        b.ld_iy_label("COLLIST")
        b.ld_b_n(libnn.NUM_BUCKETS)
        b.call("SPLITSCAN")
        for i in range(len(layer_sizes) - 1):
            b.call(f"LAYER{i + 1}")

    return emit


def build_autoreg(
    model_path: str = "command_model_autoreg.pt",
    max_output_len: int = MAX_OUTPUT_LEN,
) -> Z80Builder:
    """Assemble the column-major inference engine and model into a .COM image.

    Args:
        model_path: A ``.npz`` or ``.pt`` model.
        max_output_len: Characters to generate before giving up on an EOS.

    Returns:
        The builder, with all labels resolvable.

    Raises:
        ValueError: If a layer is too wide for an 8-bit neuron index, or the
            model's input is not the usual query/context split.
    """
    model = libinfer.load_for_build(model_path)
    layer_sizes = model.layer_sizes
    num_layers, output_size = model.num_layers, model.output_size

    validate_z80_layers(layer_sizes)
    libnn.validate_hoistable(layer_sizes)

    records = [column_records(w) for w in model.weights()]
    biases = [np.asarray(bias).astype(np.int64) for bias in model.biases()]

    plat = ColumnCPMPlatform()
    b = Z80Builder()

    libcpm.emit_entry(b)

    # === Shared engine =======================================================
    libnn.emit_generate(b, model.eos_idx, max_output_len,
                        emit_inference(layer_sizes), hoist_query=True)
    libnn.emit_printch(b, plat)
    libnn.emit_update_ctx(b)
    libnn.emit_encode_ctx(b, plat)
    libnn.emit_ctx_hash(b, plat)
    libnn.emit_clear_ctx(b)
    libnn.emit_argmax(b, output_size)
    libnn.emit_tokenizer(b, plat, model.position_bands)
    libnn.emit_tok_hash(b, plat, model.position_bands)

    # === Column-major kernel =================================================
    emit_split_scan(b)
    emit_set_ncol(b)
    emit_layers(b, layer_sizes)
    emit_preq(b, layer_sizes)
    for i in range(num_layers):
        emit_driver(b, i, _act_page(i))
        if i == num_layers - 1:
            emit_output_epilogue(b, i, layer_sizes[i + 1])
        else:
            emit_hidden_epilogue(b, i, _act_page(i + 1), layer_sizes[i + 1])

    # === Data ================================================================
    libnn.emit_charset_table(b, model.charset)
    libcpm.emit_crlf(b)

    for name in ("CDPTR", "NCOL"):
        b.label(name)
        b.dw(0)
    b.label("CDCNT")
    b.db(0)
    libnn.emit_engine_variables(b, model.position_bands)

    libcpm.emit_chat_buffer(b)

    b.label("OUTBUF")
    b.ds(output_size * libnn.ACTIVATION_SIZE)

    # Biases split the way ACC wants them, so seeding a layer is two LDIRs.
    for i, bias in enumerate(biases, start=1):
        words = [int(v) & 0xFFFF for v in bias]
        b.label(f"BIASLO{i}")
        b.db(*[v & 0xFF for v in words])
        b.label(f"BIASHI{i}")
        b.db(*[v >> 8 for v in words])
    b.label("QBASELO")
    b.ds(layer_sizes[1])
    b.label("QBASEHI")
    b.ds(layer_sizes[1])

    for i, (blob, offsets) in enumerate(records, start=1):
        b.label(f"COLPTR{i}")
        for offset in offsets:
            b.fixup_word(f"COLREC{i}", offset)
        b.label(f"COLREC{i}")
        b.blob(blob)

    b.label("COLLIST")
    b.ds(libnn.NUM_BUCKETS * 2 + 1)

    # The split pages: low bytes in one, high bytes in the next, so a neuron
    # index is a whole address and INC H reaches its other half.
    b.align(256)
    b.label("INBUF")
    b.ds(layer_sizes[0] * libnn.ACTIVATION_SIZE)
    b.label("ACT_A")
    b.ds(512)
    b.label("ACT_B")
    b.ds(512)
    b.label("ACC")
    b.ds(512)

    return b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a column-major Z80 autoregressive .COM"
    )
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load")
    parser.add_argument("--output", "-o", default="z80/CHAT.COM",
                        help="Output .COM file")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    args = parser.parse_args()

    print("Building column-major CHAT.COM...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len)
    b.save_and_report(args.output, libcpm.KEY_LABELS)


if __name__ == "__main__":
    main()
