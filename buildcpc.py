#!/usr/bin/env python3
"""
Build an Amstrad CPC binary for character-by-character text generation.

The engine is shared with every other target (see :mod:`libnn`) and the machine
- firmware jumpblock, memory map, AMSDOS header - with anything else targeting
a CPC (see :mod:`libcpc`).  What is left here is the weight layout: two bits
each, four to a byte, the same as ``buildz80com.py``.

Packed is the only layout that fits.  A CPC has 42,555 bytes between the
restart vectors and HIMEM, and the index-list layouts need 43-48KB, so the
choice ``build.py`` makes for CP/M does not arise here.

The output carries an AMSDOS header, so it loads and runs with::

    RUN"CHAT.BIN"

Usage:
    python buildcpc.py --model examples/guess/model.npz --output CHAT.BIN
"""

from __future__ import annotations

import argparse

import numpy as np

import libcpc
import libinfer
import libnn

# Re-exported: the container and the load address are part of this module's
# published surface, even though the CPC target defines them.
from libcpc import ORG_ADDR, amsdos_header, build_binary
from libinfer import MAX_OUTPUT_LEN, pack_2bit, validate_z80_layers
from libz80 import Z80Builder

__all__ = [
    "ORG_ADDR",
    "amsdos_header",
    "build_autoreg",
    "build_binary",
    "main",
    "pack_2bit_weights",
]


def pack_2bit_weights(weights: np.ndarray) -> bytes:
    """Pack 2-bit weights, four per byte, one output neuron per whole bytes.

    Uses the same scrambled nibble order as the CP/M build so both share one
    inner loop; see :func:`libinfer.pack_2bit`.
    """
    return pack_2bit(weights, layout="rotated")


def build_autoreg(
    model_path: str = "command_model_autoreg.pt",
    max_output_len: int = MAX_OUTPUT_LEN,
    org: int = ORG_ADDR,
) -> Z80Builder:
    """Assemble the inference engine and model into a CPC binary image.

    Args:
        model_path: A ``.npz`` or ``.pt`` model.
        max_output_len: Characters to generate before giving up on an EOS.
        org: Load address. Bounds the model size, since HIMEM is A67Bh.

    Returns:
        The builder, with all labels resolvable.

    Raises:
        ValueError: If a layer is too wide for a Z80 neuron loop, or the image
            would run past HIMEM at ``org``.
    """
    model = libinfer.load_for_build(model_path)
    layer_sizes = model.layer_sizes
    validate_z80_layers(layer_sizes)

    # Layer 1's query-half columns go into their own stream so PREQ can walk
    # them once per query instead of once per generated character; see
    # libinfer.forward_hoisted for why folding them into the bias is exact.
    w1q, w1c = libinfer.split_query_half(model.weight(0))
    packed_query = pack_2bit_weights(w1q)
    packed_weights = [pack_2bit_weights(w1c)] + [
        pack_2bit_weights(model.weight(i)) for i in range(1, model.num_layers)
    ]
    biases = model.biases()

    plat = libcpc.CPCPlatform()
    plans = libnn.plan_layers(layer_sizes, plat.buffer, hoist_query=True)
    qplan = libnn.query_plan(layer_sizes, plat.buffer)
    b = Z80Builder(org=org)

    libcpc.emit_entry(b)
    libcpc.emit_newline(b)
    libcpc.emit_read_input(b)

    # === Shared engine ===
    libnn.emit_generate(b, model.eos_idx, max_output_len,
                        libnn.emit_layered_inference(plans), hoist_query=True)
    libnn.emit_printch(b, plat)
    libnn.emit_update_ctx(b)
    libnn.emit_encode_ctx(b, plat)
    libnn.emit_ctx_hash(b, plat)
    libnn.emit_clear_ctx(b)
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
    libnn.emit_variables(b, model.position_bands)

    libcpc.emit_input_buffer(b)

    libnn.emit_buffers(b, plat, layer_sizes, hoist_query=True)
    libnn.emit_weights(b, packed_weights, biases)
    libnn.emit_query_weights(b, packed_query)

    libcpc.check_fits_in_ram(b.org, len(b.build()))
    return b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an Amstrad CPC autoregressive binary"
    )
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load")
    parser.add_argument("--output", "-o", default="CHAT.BIN",
                        help="Output AMSDOS binary")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    parser.add_argument("--org", type=lambda v: int(v, 0), default=ORG_ADDR,
                        help=f"Load address (default {ORG_ADDR:#06x})")
    args = parser.parse_args()

    print("Building Amstrad CPC CHAT.BIN...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len, org=args.org)

    b.report_labels(libcpc.KEY_LABELS)

    image = b.build()
    name = args.output.rsplit("/", 1)[-1].split(".")[0]
    binary = libcpc.build_binary(image, b.org, name)

    with open(args.output, "wb") as fh:
        fh.write(binary)

    headroom = libcpc.CPC_HIMEM - (b.org + len(image))
    print(f"\nTotal code size: {len(image)} bytes ({len(image) / 1024:.1f} KB)")
    print(f"File size: {len(binary)} bytes (with the {libcpc.AMSDOS_HEADER_LEN}"
          f"-byte AMSDOS header)")
    print(f"Loads at {b.org:#06x}-{b.org + len(image) - 1:#06x}, "
          f"{headroom:,} bytes below HIMEM to spare")
    print(f"Saved to {args.output}")
    print("\nOn a CPC, with the file on a disc:")
    print(f'  RUN"{name}.BIN"')


if __name__ == "__main__":
    main()
