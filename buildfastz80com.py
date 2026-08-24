#!/usr/bin/env python3
"""
Build Z80 Autoregressive Character Generation .COM file

This generates Z80 machine code for character-by-character text generation:
1. Tokenize query into first 128 buckets (trigram hashing)
2. Initialize context (next 128 buckets) to zero
3. Loop:
   a. Run neural network inference (256 inputs → 64 character outputs)
   b. Argmax to find best character
   c. If EOS (index 63), stop
   d. Print the character via BDOS
   e. Update context encoding with new character
   f. Repeat until EOS or max length

Like buildz80com.py but uses indices for for each non-zero weight.  That costs
a smidge over 1 byte per weight but since close to 75% of the weights are zero
it tends to be only about 5 more KB.  Also, the input and output values are
stored in split form in 256 byte aligned buffers.  Instead of the low byte of
a value being in the next byte, it is 256 bytes away.  Normally loading a 16
bit value pointed to be HL goes like this:
    ld   e,(hl)
    inc  hl
    ld   d,(hl)
    inc  hl
With split values the sequence is:
    ld   e,(hl)
    inc  h
    ld   d,(hl)
    dec  h
    inc  l
Awkward, but worth it when mapping an index to a value is "ld l,c" instead
of "ld b,0; sla c; rl b; add hl,bc"  And by using the stack pointer step
through the weights forces a very beneficial unrolling by 2 of the summation
loop.

Advantages:
    Close to 20 times faster than the packed weights version and over
    twice as fast as the skip list version.

Disadvantages:
    May produce a .com file too large if zero weights are less common.
    Uses the stack pointer with interrupts disabled.
    Packed weight version could be considerably faster.

Choosing between the layouts is ``build.py --target auto``'s job; this module
is what it calls when the index-list one is the fastest that fits.  The program
around the layers - entry, chat loop, BDOS line buffer - is shared with the
other two CP/M backends in ``libcpm.py``.
"""

from __future__ import annotations

import argparse

import numpy as np

import libcpm
import libinfer
import libnn
from libcpm import CPMPlatform
from libinfer import MAX_OUTPUT_LEN, NUM_BUCKETS, validate_z80_layers
from libz80 import Z80Builder


class FastCPMPlatform(CPMPlatform):
    """Same I/O as the packed CP/M build; the weights are laid out differently.

    Activations still start in INBUF in the ordinary interleaved form; INFER
    copies them into the split high/low buffers its inner loop indexes.
    """

    name = "CP/M (index lists)"


def pack_weights_and_biases(weights: np.ndarray, biases: np.ndarray,
                            first_column: int = 0) -> bytes:
    """Convert weights into index lists by weight and append bias after
    each node.

    ``first_column`` is added to every emitted index, so a matrix that has been
    split by column still names positions in the full activation buffer. Layer
    1's context half is packed with ``first_column=128`` for exactly that
    reason: its inputs sit in the upper half of the buffer INFER indexes.
    """
    wt_bias = []
    for n in range(weights.shape[0]):
        flat = np.clip(weights[n], -2, 1).astype(np.int8)
        for w in (-2, -1, 1):
            indices = [i + first_column for i, v in enumerate(flat) if v == w]
            wt_bias.append(len(indices))
            wt_bias += indices
        bias_val = int(biases[n]) & 0xFFFF
        wt_bias.append(bias_val & 0xFF)
        wt_bias.append((bias_val >> 8) & 0xFF)

    return bytes(wt_bias)


def sum_wt(b: Z80Builder, w: int) -> None:
    """Emit ADD or SUB of (HL) into A, according to the sign of the weight.

    ``w`` is one of {-2, -1, +1}; zero weights never reach the inner loop,
    which is the whole point of the index-list layout.
    """
    if w > 0:
        b.add_a_hl()
    else:
        b.sub_a_hl()


def sum_wt_carry(b: Z80Builder, w: int) -> None:
    """As :func:`sum_wt`, but carrying in from the low half of the accumulator."""
    if w > 0:
        b.adc_a_hl()
    else:
        b.sbc_a_hl()


def emit_split(b: Z80Builder) -> None:
    """Emit SPLIT: copy interleaved 16-bit values into the split hi/lo buffers.

    ``HL`` is the source, ``DE`` the destination; the loop ends when ``E`` wraps
    to zero, so the caller picks the count by where in the page it starts. PREQ
    copies all 256 activations, the per-character path only the 128 that
    changed.
    """
    b.label('SPLIT')
    b.ld_a_hl()
    b.inc_hl()
    b.ld_de_a()
    b.ld_a_hl()
    b.inc_hl()
    b.inc_d()
    b.ld_de_a()
    b.dec_d()
    b.inc_e()
    b.jr_nz('SPLIT')
    b.ret()


def emit_preq(b: Z80Builder) -> None:
    """Emit PREQ: fold layer 1's query-half columns into its per-neuron bias.

    The query does not change while a response is generated, so this runs once
    per query rather than once per character - about a fifth of the whole
    forward pass for the shipped models. Folding a partial sum into the bias is
    exact, not an approximation; see :func:`libinfer.forward_hoisted`.

    QWTS holds the query-half index lists in the same per-neuron record shape
    INFER uses, each ending with the model's *original* bias. PREQ walks it with
    ``IX`` while ``PQB`` walks layer 1's records inside NETWORK, whose bias
    slots it overwrites. Keeping the two streams in lockstep is what lets INFER
    stay byte-for-byte what it was: the hot loop never learns that any of this
    happened.

    ``DE`` accumulates, ``HL`` indexes the split activation buffer, ``B`` counts
    the current list. Cold code - it runs once per query, so it is written for
    clarity rather than for cycles.
    """
    b.label('PREQ')
    b.ld_hl_label('INBUF')
    b.ld_de_label('BUF_A')
    b.call('SPLIT')

    b.ld_ix_label('QWTS')
    b.ld_hl_label('L1REC')
    b.ld_mem_label_hl('PQB')
    b.ld_a_mem_label('L1SIZE')  # zero means 256, which DEC A counts correctly
    b.ld_mem_label_a('PQN')

    b.label('PQ_NEUR')
    b.ld_de_nn(0)
    b.ld_hl_label('BUF_A')  # H selects the low page; L is set per index

    for w in (-2, -1, 1):
        tag = w + 2
        b.ld_a_ixd(0)
        b.inc_ix()
        b.or_a()
        b.jr_z(f'PQ_SKIP{tag}')
        b.ld_b_a()

        b.label(f'PQ_LOOP{tag}')
        b.ld_a_ixd(0)
        b.inc_ix()
        b.ld_l_a()
        for _ in range(2 if w == -2 else 1):
            if w > 0:
                b.ld_a_hl()
                b.add_a_e()
                b.ld_e_a()
                b.inc_h()
                b.ld_a_hl()
                b.adc_a_d()
            else:
                b.ld_a_e()
                b.sub_a_hl()
                b.ld_e_a()
                b.inc_h()
                b.ld_a_d()
                b.sbc_a_hl()
            b.ld_d_a()
            b.dec_h()
        b.djnz(f'PQ_LOOP{tag}')

        b.label(f'PQ_SKIP{tag}')

    # The record's own bias is the model's BIAS1, kept here because the slot it
    # used to live in is now PREQ's output and gets overwritten every query.
    b.ld_a_ixd(0)
    b.inc_ix()
    b.add_a_e()
    b.ld_e_a()
    b.ld_a_ixd(0)
    b.inc_ix()
    b.adc_a_d()
    b.ld_d_a()

    # Step over this neuron's three lists in NETWORK to reach its bias slot.
    b.ld_hl_mem_label('PQB')
    for tag in range(3):
        b.ld_a_hl()
        b.inc_hl()
        b.add_a_l()
        b.ld_l_a()
        b.jr_nc(f'PQ_NC{tag}')
        b.inc_h()
        b.label(f'PQ_NC{tag}')
    b.ld_hl_e()
    b.inc_hl()
    b.ld_hl_d()
    b.inc_hl()
    b.ld_mem_label_hl('PQB')

    b.ld_a_mem_label('PQN')
    b.dec_a()
    b.ld_mem_label_a('PQN')
    b.jp_nz('PQ_NEUR')
    b.ret()


def build_autoreg(model_path: str = 'command_model_autoreg.pt',
                  max_output_len: int = MAX_OUTPUT_LEN) -> Z80Builder:
    """Assemble the index-list inference engine and model into a .COM image.

    Args:
        model_path: A ``.npz`` or ``.pt`` model.
        max_output_len: Characters to generate before giving up on an EOS.

    Returns:
        The builder, with all labels resolvable.

    Raises:
        ValueError: If a layer is wider than a Z80 neuron loop can count.
    """
    model = libinfer.load_for_build(model_path)
    layer_sizes = model.layer_sizes
    num_layers = model.num_layers
    input_size = model.input_size

    validate_z80_layers(layer_sizes)
    libnn.validate_hoistable(layer_sizes)

    # Layer 1 is split by column. The query half goes into QWTS, which PREQ
    # walks once per query; the context half stays in NETWORK with its indices
    # unchanged, since they already name the upper half of the activation
    # buffer INFER indexes. NETWORK's layer-1 bias slots are PREQ's output, so
    # they start at zero and the real BIAS1 travels with QWTS.
    w1q, w1c = libinfer.split_query_half(model.weight(0))
    bias1 = model.bias(0)
    query_weights = pack_weights_and_biases(w1q, bias1)
    weights_biases = [
        pack_weights_and_biases(w1c, np.zeros_like(bias1), NUM_BUCKETS)
    ] + [
        pack_weights_and_biases(model.weight(i), model.bias(i))
        for i in range(1, num_layers)
    ]

    plat = FastCPMPlatform()
    b = Z80Builder()

    libcpm.emit_entry(b)

    # === GENERATE: Main generation loop ===
    b.label('GENERATE')
    b.call('PREQ')  # once per response: the query half cannot change during one
    b.ld_a_n(max_output_len)
    b.ld_mem_label_a('GENCNT')

    b.label('GENLOOP')

    # Copy the context half of INBUF into BUF_A, splitting values as we go.
    # Only the context half: layer 1 no longer reads the query half at all, and
    # BUF_A's lower half is scratch for a later layer by the time it matters.
    b.ld_hl_label('INBUF', libnn.CONTEXT_OFFSET)
    b.ld_de_label('BUF_A', NUM_BUCKETS)
    b.call('SPLIT')

    # Run inference through all layers
    b.ld_hl_label('NETWORK')
    b.call('INFER')

    # Find best character
    b.call('ARGMAX')

    # Check for EOS
    b.ld_a_mem_label('RESULT')
    b.cp_n(model.eos_idx)
    b.ret_z()  # Return if EOS

    # Print character
    b.call('PRINTCH')

    # Update context with new character
    b.call('UPDATE_CTX')

    # Loop if not done
    b.ld_a_mem_label('GENCNT')
    b.dec_a()
    b.ld_mem_label_a('GENCNT')
    b.jr_nz('GENLOOP')
    b.ret()

    # === Query-half hoisting ===
    emit_split(b)
    emit_preq(b)

    # === Shared engine: printing, context encoding, tokenizing ===
    libnn.emit_printch(b, plat)
    libnn.emit_update_ctx(b)
    libnn.emit_encode_ctx(b, plat)
    libnn.emit_ctx_hash(b, plat)
    libnn.emit_clear_ctx(b, unrolled=False)

    # === Inference Evaluation ===
    # HL points to NETWORK:
    #    1 byte   number of layers
    # Followed by the layers which are:
    #    1 byte   number of output values
    #    weight + bias data
    #
    # From this we load:
    #    E   number of layers
    #    D   number of outputs of layer
    #    HL  output buffer (only needs high byte)
    #    HL' input buffer
    # On return:
    #    B   number of outputs of last layer
    #    HL' last output buffer

    b.label('INFER')

    b.ld_e_hl() # number of layers
    b.inc_hl()

    b.ld_mem_label_sp('SPSAV')
    b.di()

    b.ld_sp_hl() # rest of the network data

    b.ld_hl_label('BUF_B') # output buffer
    b.exx()
    b.ld_hl_label('BUF_A') # input buffer
    b.exx()

    b.label('LAYER_LOOP')

    b.dec_sp()
    b.pop_af() # A = number of outputs
    b.ld_d_a() # now in D
    b.ld_b_a() # will be last ouput size for ARGMAX

    # SP=weights + biases, HL'=IN, HL=OUT, D=LEN(OUT), E=# of layers

    b.dec_e() # decrement so easier to test for E=1 in ReLU check

    b.label('LNEUR')

    b.exx()
    b.xor_a()
    b.ld_c_a() # accumulator = 0
    for w in [-2, -1, 1]:
        b.pop_de() # E = number of weight indices, D = first weight
        b.srl_e()
        b.jr_nc(f"even{w+2}") # no carry means even number of weights
        # Calculate the D weight we have
        b.ld_l_d()
        sum_wt(b, w)
        b.ld_d_a()
        b.inc_h()
        b.ld_a_c()
        sum_wt_carry(b, w)
        b.ld_c_a()
        b.ld_a_d()
        b.dec_h()
        b.inc_e()
        b.dec_e()
        b.db(0x16) # ld d,n (to skip adjustment of stack pointer)
        b.label(f"even{w+2}")
        b.dec_sp() # read 1 byte too much, back off
        b.jr_z(f"skip{w+2}")
        b.ld_b_e()
        b.label(f"wt{w+2}")
        b.pop_de()
        b.ld_l_e()
        sum_wt(b, w)
        b.ld_e_a()
        b.inc_h()
        b.ld_a_c()
        sum_wt_carry(b, w)
        b.ld_l_d()
        sum_wt(b, w)
        b.ld_c_a()
        b.dec_h()
        b.ld_a_e()
        sum_wt(b, w)
        b.jr_nc(f"c_ok{w+2}")
        if w > 0:
            b.inc_c()
        else:
            b.dec_c()
        b.label(f"c_ok{w+2}")
        b.djnz(f"wt{w+2}")
        if w == -2:
            b.add_a_a()
            b.rl_c()
        b.label(f"skip{w+2}")

    # Bias follows after weights
    b.pop_de()
    b.add_a_e()
    b.ld_e_a()
    b.ld_a_c()
    b.adc_a_d()
    b.ld_c_a()
    b.ld_a_e()

    # Decide the ReLU before scaling rather than after.  The sign lives in bit
    # 7 of C and the shift is arithmetic, so it cannot change: a hidden neuron
    # that is negative here relus to exactly zero whatever the shift does, and
    # scaling it first is wasted work.
    b.exx()
    b.inc_e()
    b.dec_e()
    b.exx()
    b.jr_z('DO_SCALE')      # last layer has no ReLU, so always scale
    b.bit_7_c()
    b.jr_z('DO_SCALE')      # non-negative, so scale it
    b.xor_a()
    b.ld_c_a()              # negative: the answer is zero, skip the shift
    b.jr('NO_RELU')

    b.label('DO_SCALE')
    b.sra_c()
    b.rra()
    b.sra_c()
    b.rra()
    b.label('NO_RELU')

    # write summation to output
    b.exx()
    b.ld_hl_a()
    b.inc_h()
    b.exx()
    b.ld_a_c()
    b.exx()
    b.ld_hl_a()
    b.dec_h()
    b.inc_l()

    # We're back in the regular registers
    b.dec_d()
    b.jp_nz('LNEUR')

    # Swap input and output buffers.  Could be done with an XOR to each H.
    # Also need to zero L
    # Or, considering only B and and E are live, just EXX and pull them over.
    # And B is just being cute, really.
    b.ld_l_n(0)
    b.ld_a_h() # A=H
    b.exx()
    b.ex_af_af()
    b.ld_l_n(0)
    b.ld_a_h() # A'=H'
    b.ex_af_af()
    b.ld_h_a() # H'=A (=H)
    b.exx()
    b.ex_af_af()
    b.ld_h_a() # H=A' (=H')

    b.inc_e() # was -1 as ReLU flag
    b.dec_e()
    b.jp_nz('LAYER_LOOP')

    b.ld_sp_mem_label('SPSAV')
    b.ei()

    b.ret()

    # === ARGMAX ===
    # HL' = layer values, B = layer size.  Exactly what INFER returns with.
    # Hastily fixed up for split values; code could be much improved.
    # Especially if we work backwards so L is our counter (though beware how
    # that could change things -- we should accept "=" to have same operation)
    b.label('ARGMAX')

    b.ld_a_b()
    b.exx()
    b.ld_b_a()

    b.ld_e_hl()
    b.inc_h()
    b.ld_d_hl()
    b.dec_h()
    b.inc_l()

    b.ld_mem_label_de('MAXV')
    b.xor_a()
    b.ld_mem_label_a('MAXI')
    b.ld_c_n(1)

    b.label('AMLP')
    b.ld_e_hl()
    b.inc_h()
    b.ld_d_hl()
    b.dec_h()
    b.inc_l()

    b.push_hl()
    b.ld_hl_mem_label('MAXV')
    b.push_de()
    b.or_a()
    b.ex_de_hl()
    b.sbc_hl_de()
    b.pop_de()
    b.jp_m('AMSK')
    b.jr_z('AMSK')
    b.ld_mem_label_de('MAXV')
    b.ld_a_c()
    b.ld_mem_label_a('MAXI')

    b.label('AMSK')
    b.pop_hl()
    b.inc_c()
    b.djnz('AMLP')
    b.ld_a_mem_label('MAXI')
    b.ld_mem_label_a('RESULT')
    b.ret()

    libnn.emit_tokenizer(b, plat, model.position_bands)
    libnn.emit_tok_hash(b, plat, model.position_bands)

    # === DATA ===
    # Character table (dynamic size based on charset)
    libnn.emit_charset_table(b, model.charset)

    libcpm.emit_crlf(b)

    # INFER parks the stack pointer here while it walks the weights with POP.
    b.label('SPSAV')
    b.dw(0)
    # PREQ's cursor into NETWORK's layer-1 records, and its neuron countdown.
    b.label('PQB')
    b.dw(0)
    b.label('PQN')
    b.db(0)
    libnn.emit_engine_variables(b, model.position_bands)

    libcpm.emit_chat_buffer(b)

    b.label('NETWORK')
    b.db(num_layers)
    # Weights and biases
    for i in range(num_layers):
        if i == 0:
            b.label('L1SIZE')
        b.db(layer_sizes[i + 1])
        if i == 0:
            b.label('L1REC')  # PREQ walks these records to find the bias slots
        b.blob(weights_biases[i])

    # Layer 1's query-half records, walked once per query by PREQ. Same shape as
    # a NETWORK layer, minus the leading size byte PREQ reads from L1SIZE.
    b.label('QWTS')
    b.blob(query_weights)

    # Buffers
    b.align(256)
    if input_size > 256:
        raise ValueError(f"Input size {input_size} is too big; limit 256.")
    b.label('INBUF')
    b.ds(256 * 2)  # 256 buckets * 2 bytes
    max_hidden = max(layer_sizes[1:-1]) if len(layer_sizes) > 2 else layer_sizes[1]
    if max_hidden > 256:
        raise ValueError(f"Layer size {max_hidden} is too big; limit 256.")
    b.label('BUF_A')
    b.ds(256 * 2)
    b.label('BUF_B')
    b.ds(256 * 2)

    return b


def main() -> None:
    parser = argparse.ArgumentParser(description='Build Z80 autoregressive .COM')
    parser.add_argument('--model', '-m', default='command_model_autoreg.pt',
                        help='Model file to load')
    parser.add_argument('--output', '-o', default='z80/CHAT.COM',
                        help='Output .COM file')
    parser.add_argument('--max-output-len', type=int, default=MAX_OUTPUT_LEN,
                        help='Maximum characters generated per response')
    args = parser.parse_args()

    print("Building autoregressive CHAT.COM...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len)
    b.save_and_report(args.output, libcpm.KEY_LABELS)


if __name__ == '__main__':
    main()
