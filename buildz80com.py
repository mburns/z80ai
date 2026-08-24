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

import libcpm
import libnn
from libcpm import CPMPlatform
from libinfer import MAX_OUTPUT_LEN
from libz80 import Z80Builder


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
    plat = CPMPlatform()
    packed = libnn.prepare_packed(model_path, plat)
    model = packed.model
    b = Z80Builder()

    libcpm.emit_entry(b)

    # === Shared engine ===
    libnn.emit_packed_engine(b, plat, packed, max_output_len)

    # === Data ===
    libnn.emit_charset_table(b, model.charset)
    libcpm.emit_crlf(b)
    libnn.emit_variables(b, model.position_bands)
    libcpm.emit_chat_buffer(b)

    libnn.emit_packed_tail(b, plat, packed)

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
