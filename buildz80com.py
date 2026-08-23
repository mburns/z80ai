#!/usr/bin/env python3
"""
Build a CP/M .COM for character-by-character text generation.

Weights are packed two bits each, four to a byte, which always fits the
transient program area but makes the inner loop spend most of its time
unpacking. ``buildfastz80com.py`` trades size for roughly nine times the speed;
``build.py --target auto`` picks whichever fits.

Run with a query on the command line for a single answer, or with no arguments
for an interactive chat prompt.
"""

from __future__ import annotations

import argparse

import numpy as np

import libcpm
import libinfer
import libnn
from libcpm import CPMPlatform
from libinfer import MAX_OUTPUT_LEN, pack_2bit, validate_z80_layers
from libz80 import Z80Builder


def pack_2bit_weights(weights: np.ndarray) -> bytes:
    """Pack 2-bit weights, four per byte, one output neuron per whole bytes.

    The nibble order is scrambled so MULADD can decide between {-2,-1,0,+1}
    with two DECs, putting the most common weight (zero) on the fastest path.
    See :func:`libinfer.pack_2bit` for the encoding.
    """
    return pack_2bit(weights, layout="rotated")


def build_autoreg(
    model_path: str = "command_model_autoreg.pt",
    max_output_len: int = MAX_OUTPUT_LEN,
) -> Z80Builder:
    """Assemble the inference engine and model into a CP/M .COM image.

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
    validate_z80_layers(layer_sizes)

    # Layer 1's query-half columns are split off into their own stream: the
    # query does not change while a response is being generated, so PREQ walks
    # them once per query and hands layer 1 the result as its bias. Exact, not
    # approximate - see libinfer.forward_hoisted.
    w1q, w1c = libinfer.split_query_half(model.weight(0))
    packed_query = pack_2bit_weights(w1q)
    packed_weights = [pack_2bit_weights(w1c)] + [
        pack_2bit_weights(model.weight(i)) for i in range(1, model.num_layers)
    ]
    biases = model.biases()

    plat = CPMPlatform()
    plans = libnn.plan_layers(layer_sizes, plat.buffer, hoist_query=True)
    qplan = libnn.query_plan(layer_sizes, plat.buffer)
    b = Z80Builder()

    libcpm.emit_entry(b)

    # === Shared engine ===
    libnn.emit_generate(b, plat, model.eos_idx, max_output_len,
                        libnn.emit_layered_inference(plans), hoist_query=True)
    libnn.emit_printch(b, plat)
    libnn.emit_update_ctx(b, plat)
    libnn.emit_encode_ctx(b, plat)
    libnn.emit_ctx_hash(b, plat)
    libnn.emit_clear_ctx(b, plat)
    libnn.emit_layer_dispatch(b, plans)
    libnn.emit_layer(b)
    libnn.emit_query_dispatch(b, qplan)
    libnn.emit_layer(b, name="QLAYER", prefix="Q", scale=False)
    libnn.emit_muladd(b)
    libnn.emit_relu(b, plans)
    libnn.emit_argmax(b, model.output_size)
    libnn.emit_tokenizer(b, plat, model.position_bands)
    libnn.emit_tok_hash(b, plat, model.position_bands)

    # === Data ===
    libnn.emit_charset_table(b, model.charset)
    libcpm.emit_crlf(b)
    libnn.emit_variables(b, model.position_bands)
    libcpm.emit_chat_buffer(b)

    libnn.emit_buffers(b, plat, layer_sizes, hoist_query=True)
    libnn.emit_weights(b, packed_weights, biases)
    libnn.emit_query_weights(b, packed_query)

    return b


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Z80 autoregressive .COM")
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load")
    parser.add_argument("--output", "-o", default="z80/CHAT.COM",
                        help="Output .COM file")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    args = parser.parse_args()

    print("Building autoregressive CHAT.COM...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len)
    b.save_and_report(args.output, libcpm.KEY_LABELS)


if __name__ == "__main__":
    main()
