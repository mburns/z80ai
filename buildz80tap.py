#!/usr/bin/env python3
"""
Build a ZX Spectrum 48K .TAP for character-by-character text generation.

The engine is shared with the CP/M build (see :mod:`libnn`) and the machine —
ROM entry points, the keyboard line editor, the .TAP container — with anything
else targeting a Spectrum (see :mod:`libzx`).  What is left here is the weight
layout: two bits each, four to a byte, the same as ``buildz80com.py``.

RAM ends at FFFFh, so the load address bounds how large a model can be. See
``libzx.ORG_ADDR`` and ZX-SPECTRUM.md.
"""

from __future__ import annotations

import argparse

import numpy as np

import libinfer
import libnn
import libzx
from libinfer import MAX_OUTPUT_LEN, pack_2bit, validate_z80_layers
from libz80 import Z80Builder

# Re-exported: the .TAP container and the load address are part of this
# module's published surface, even though the ZX target now defines them.
from libzx import ORG_ADDR, ZX_RAM_TOP, build_tap_data, build_tap_header

__all__ = [
    "ORG_ADDR",
    "ZX_RAM_TOP",
    "build_autoreg",
    "build_tap_data",
    "build_tap_header",
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
    """Assemble the inference engine and model into a Spectrum CODE image.

    Args:
        model_path: A ``.npz`` or ``.pt`` model.
        max_output_len: Characters to generate before giving up on an EOS.
        org: Load address. Bounds the model size, since RAM ends at FFFFh.

    Returns:
        The builder, with all labels resolvable.

    Raises:
        ValueError: If a layer is too wide for a Z80 neuron loop, or the image
            would not fit in RAM at ``org``.
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

    plat = libzx.ZXPlatform()
    plans = libnn.plan_layers(layer_sizes, plat.buffer, hoist_query=True)
    qplan = libnn.query_plan(layer_sizes, plat.buffer)
    b = Z80Builder(org=org)

    libzx.emit_entry(b)
    libzx.emit_read_input(b)

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

    libzx.emit_input_buffer(b)

    libnn.emit_buffers(b, plat, layer_sizes, hoist_query=True)
    libnn.emit_weights(b, packed_weights, biases)
    libnn.emit_query_weights(b, packed_query)

    libzx.check_fits_in_ram(b.org, len(b.build()))
    return b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Z80 autoregressive .TAP for ZX Spectrum"
    )
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load")
    parser.add_argument("--output", "-o", default="CHAT.TAP",
                        help="Output .TAP file")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    parser.add_argument("--org", type=lambda v: int(v, 0), default=ORG_ADDR,
                        help=f"Load address (default {ORG_ADDR:#06x})")
    args = parser.parse_args()

    print("Building ZX Spectrum CHAT.TAP...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len, org=args.org)

    b.report_labels(libzx.KEY_LABELS)

    image = b.build()
    tap_data = libzx.build_tap(image, b.org)

    with open(args.output, "wb") as fh:
        fh.write(tap_data)

    headroom = ZX_RAM_TOP - (b.org + len(image))
    print(f"\nTotal code size: {len(image)} bytes ({len(image) / 1024:.1f} KB)")
    print(f"TAP file size: {len(tap_data)} bytes")
    print(f"Loads at {b.org:#06x}-{b.org + len(image) - 1:#06x}, "
          f"{headroom:,} bytes of RAM to spare")
    print(f"Saved to {args.output}")
    print("\nIn ZX Spectrum BASIC:")
    print(f"  CLEAR {b.org - 1}")
    print('  LOAD "" CODE')
    print(f"  RANDOMIZE USR {b.org}")


if __name__ == "__main__":
    main()
