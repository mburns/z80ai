"""
Load model parameters from either PyTorch (.pt) or NumPy (.npz) format.

This module allows build scripts to work with either format, enabling
CI environments to run without PyTorch installed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from libinfer import Arch, Params


def load_model_params(model_path: str) -> tuple[Params, Arch, str]:
    """Load model parameters from a .pt or .npz file.

    Args:
        model_path: Path ending in ``.npz`` or ``.pt``.

    Returns:
        ``(params, arch, charset)``: the quantized weights and biases keyed
        ``fc1_weight``/``fc1_bias`` and so on, the architecture dict, and the
        character set.

    Raises:
        ValueError: If the path has neither extension.
    """
    if model_path.endswith('.npz'):
        return _load_npz(model_path)
    if model_path.endswith('.pt'):
        return _load_pt(model_path)
    raise ValueError(f"Unknown model format: {model_path} (expected .pt or .npz)")


def _load_npz(model_path: str) -> tuple[Params, Arch, str]:
    """Load from NumPy npz format."""
    data = np.load(model_path)

    # Metadata rides along as encoded strings under underscore-prefixed keys,
    # which is also how the parameters are told apart from it.
    arch = json.loads(bytes(data['_architecture']).decode('utf-8'))
    charset = bytes(data['_charset']).decode('utf-8')
    params = {k: data[k] for k in data.files if not k.startswith('_')}

    return params, arch, charset


def quantize_checkpoint(checkpoint: dict[str, Any]) -> tuple[Params, Arch, str]:
    """Rebuild the trained model from a checkpoint and quantize it.

    Shared by :func:`_load_pt` and ``exportmodel.py`` so there is one answer to
    "what does a .pt turn into", rather than two that can drift.

    Args:
        checkpoint: A dict as saved by ``feedme.py``: ``model_state``,
            ``architecture`` and ``charset``.

    Returns:
        The same ``(params, arch, charset)`` triple as :func:`load_model_params`.
    """
    from feedme import AutoregressiveModel

    arch = checkpoint['architecture']
    charset = checkpoint['charset']

    model = AutoregressiveModel(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        num_chars=len(charset),
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model.get_quantized_params(), arch, charset


def _load_pt(model_path: str) -> tuple[Params, Arch, str]:
    """Load from PyTorch checkpoint format."""
    import torch

    return quantize_checkpoint(torch.load(model_path, weights_only=True))
