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

import libcpc
import libnn

# Re-exported: the container and the load address are part of this module's
# published surface, even though the CPC target defines them.
from libcpc import ORG_ADDR, amsdos_header, build_binary
from libinfer import MAX_OUTPUT_LEN
from libz80 import Z80Builder

__all__ = [
    "ORG_ADDR",
    "amsdos_header",
    "build_autoreg",
    "build_binary",
    "main",
]


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
    plat = libcpc.CPCPlatform()
    packed = libnn.prepare_packed(model_path, plat)
    model = packed.model
    b = Z80Builder(org=org)

    libcpc.emit_entry(b)
    libcpc.emit_newline(b)
    libcpc.emit_read_input(b)

    # === Shared engine ===
    libnn.emit_packed_engine(b, plat, packed, max_output_len)

    # === Data ===
    libnn.emit_charset_table(b, model.charset)
    libnn.emit_variables(b, model.position_bands)

    libcpc.emit_input_buffer(b)

    libnn.emit_packed_tail(b, plat, packed)

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
