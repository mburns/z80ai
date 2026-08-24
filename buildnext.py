#!/usr/bin/env python3
"""
Build a ZX Spectrum Next .TAP for character-by-character text generation.

The same image as ``buildz80tap.py`` - same engine, same packed weights, same
container and load address - with the Next's clock register set at startup. At
28MHz that is 8x the generated characters per second of the 3.5MHz build, which
is the difference between watching it type and it answering.

Nothing here is Next-only in a way that breaks a Spectrum: the clock register
lives at a port a 48K machine does not decode, so this .TAP still loads and runs
on original hardware, just at 3.5MHz.

Usage:
    python buildnext.py --model examples/guess/model.npz --output CHAT.TAP
"""

from __future__ import annotations

import argparse

import libnext
import libnn
import libzx
from libinfer import MAX_OUTPUT_LEN
from libnext import DEFAULT_SPEED, ORG_ADDR
from libz80 import Z80Builder

__all__ = ["ORG_ADDR", "build_autoreg", "main"]


def build_autoreg(
    model_path: str = "command_model_autoreg.pt",
    max_output_len: int = MAX_OUTPUT_LEN,
    org: int = ORG_ADDR,
    speed: str = DEFAULT_SPEED,
) -> Z80Builder:
    """Assemble the inference engine and model into a Next CODE image.

    Args:
        model_path: A ``.npz`` or ``.pt`` model.
        max_output_len: Characters to generate before giving up on an EOS.
        org: Load address. Bounds the model size, since RAM ends at FFFFh.
        speed: One of :data:`libnext.SPEEDS`, the clock to ask the Next for.

    Returns:
        The builder, with all labels resolvable.

    Raises:
        ValueError: If a layer is too wide for a Z80 neuron loop, the speed is
            not one the Next offers, or the image would not fit in RAM.
    """
    plat = libnext.NextPlatform()
    packed = libnn.prepare_packed(model_path, plat)
    model = packed.model
    b = Z80Builder(org=org)

    libnext.emit_entry(b, speed=speed)
    libzx.emit_read_input(b)

    # === Shared engine ===
    libnn.emit_packed_engine(b, plat, packed, max_output_len)

    # === Data ===
    libnn.emit_charset_table(b, model.charset)
    libnn.emit_variables(b, model.position_bands)

    libzx.emit_input_buffer(b)

    libnn.emit_packed_tail(b, plat, packed)

    libzx.check_fits_in_ram(b.org, len(b.build()))
    return b


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a ZX Spectrum Next autoregressive .TAP"
    )
    parser.add_argument("--model", "-m", default="command_model_autoreg.pt",
                        help="Model file to load")
    parser.add_argument("--output", "-o", default="CHAT.TAP",
                        help="Output .TAP file")
    parser.add_argument("--max-output-len", type=int, default=MAX_OUTPUT_LEN,
                        help="Maximum characters generated per response")
    parser.add_argument("--org", type=lambda v: int(v, 0), default=ORG_ADDR,
                        help=f"Load address (default {ORG_ADDR:#06x})")
    parser.add_argument("--speed", default=DEFAULT_SPEED,
                        choices=sorted(libnext.SPEEDS, key=float),
                        help=f"CPU clock in MHz (default {DEFAULT_SPEED})")
    args = parser.parse_args()

    print("Building ZX Spectrum Next CHAT.TAP...\n")
    b = build_autoreg(args.model, max_output_len=args.max_output_len,
                      org=args.org, speed=args.speed)

    b.report_labels(libnext.KEY_LABELS)

    image = b.build()
    tap_data = libzx.build_tap(image, b.org)

    with open(args.output, "wb") as fh:
        fh.write(tap_data)

    headroom = libzx.ZX_RAM_TOP - (b.org + len(image))
    print(f"\nTotal code size: {len(image)} bytes ({len(image) / 1024:.1f} KB)")
    print(f"TAP file size: {len(tap_data)} bytes")
    print(f"Loads at {b.org:#06x}-{b.org + len(image) - 1:#06x}, "
          f"{headroom:,} bytes of RAM to spare")
    print(f"Runs at {args.speed}MHz on a Next, 3.5MHz on a 48K Spectrum")
    print(f"Saved to {args.output}")
    print("\nOn a Next, or in BASIC on either machine:")
    print(f"  CLEAR {b.org - 1}")
    print('  LOAD "" CODE')
    print(f"  RANDOMIZE USR {b.org}")


if __name__ == "__main__":
    main()
