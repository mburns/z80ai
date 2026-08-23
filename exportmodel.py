#!/usr/bin/env python3
"""
Export PyTorch model checkpoint to NumPy .npz format.

This allows the build scripts to run without PyTorch installed,
which is useful for CI environments where PyTorch is too heavy.

Usage:
    python exportmodel.py --model model.pt --output model.npz
"""

from __future__ import annotations

import argparse

import torch

from libinfer import FLAT, Model
from loadmodel import quantize_checkpoint


def export_model(model_path: str, output_path: str) -> None:
    """Export a PyTorch checkpoint to the .npz the build scripts read.

    Args:
        model_path: A ``.pt`` checkpoint as saved by ``feedme.py``.
        output_path: Where to write the ``.npz``.
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=True)
    params, arch, charset = quantize_checkpoint(checkpoint)

    print(f"Architecture: input={arch['input_size']}, "
          f"hidden={arch['hidden_sizes']}, output={len(charset)}")
    print(f"Charset ({len(charset)} chars): {charset[:-1]!r} + EOS")

    # Model owns the .npz layout, so the exporter and libinfer.Model.save_npz
    # cannot disagree about where the metadata goes.
    model = Model.from_params(params, charset, arch.get('position_bands', FLAT))
    model.save_npz(output_path)
    print(f"Exported to {output_path}")

    for i, (w, b) in enumerate(zip(model.weights, model.biases, strict=True), start=1):
        print(f"  fc{i}: weight {w.shape}, bias {b.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Export PyTorch model to NumPy format')
    parser.add_argument('--model', '-m', default='command_model_autoreg.pt',
                        help='Input PyTorch model checkpoint (.pt)')
    parser.add_argument('--output', '-o', default='model.npz',
                        help='Output NumPy archive (.npz)')
    args = parser.parse_args()

    export_model(args.model, args.output)


if __name__ == '__main__':
    main()
