"""Tests for loading and exporting model checkpoints.

The .pt path is what turns a trained model into the integers the build scripts
assemble, and exportmodel.py and loadmodel.py both walk it. They used to have a
copy each; these tests pin that one .pt gives one answer whichever way it is
read, so a build from a .pt and a build from the exported .npz agree.
"""

from __future__ import annotations

import numpy as np
import pytest

import libinfer
import loadmodel

torch = pytest.importorskip("torch", reason="the .pt path needs PyTorch")


CHARSET = " ABCDEFGHIJ\x00"
ARCH = {
    "input_size": 256,
    "hidden_sizes": [16, 12],
    "num_classes": len(CHARSET),
    "position_bands": 4,
}


@pytest.fixture(scope="module")
def checkpoint() -> dict:
    """A checkpoint shaped exactly as feedme.py saves one."""
    from feedme import AutoregressiveModel

    torch.manual_seed(3)
    model = AutoregressiveModel(
        input_size=ARCH["input_size"],
        hidden_sizes=ARCH["hidden_sizes"],
        num_chars=len(CHARSET),
    )
    return {"model_state": model.state_dict(), "architecture": ARCH,
            "charset": CHARSET}


@pytest.fixture(scope="module")
def pt_path(tmp_path_factory, checkpoint) -> str:
    path = str(tmp_path_factory.mktemp("checkpoints") / "model.pt")
    torch.save(checkpoint, path)
    return path


def test_an_unknown_extension_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match=r"expected \.pt or \.npz"):
        loadmodel.load_model_params("model.safetensors")


def test_quantized_weights_are_two_bit_and_biases_are_int16(checkpoint):
    params, _arch, _charset = loadmodel.quantize_checkpoint(checkpoint)
    for name, value in params.items():
        if name.endswith("_weight"):
            assert value.dtype == np.int8
            assert set(np.unique(value)) <= {-2, -1, 0, 1}
        else:
            assert value.dtype == np.int16


def test_quantizing_a_checkpoint_is_deterministic(checkpoint):
    """Two builds from one .pt have to produce the same binary."""
    first, _, _ = loadmodel.quantize_checkpoint(checkpoint)
    second, _, _ = loadmodel.quantize_checkpoint(checkpoint)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_export_produces_an_npz_that_loads_identically(pt_path, tmp_path):
    """The whole point of exporting: CI builds from .npz without torch."""
    import exportmodel

    npz_path = str(tmp_path / "model.npz")
    exportmodel.export_model(pt_path, npz_path)

    from_pt = loadmodel.load_model_params(pt_path)
    from_npz = loadmodel.load_model_params(npz_path)

    pt_params, pt_arch, pt_charset = from_pt
    npz_params, npz_arch, npz_charset = from_npz
    assert pt_arch == npz_arch == ARCH
    assert pt_charset == npz_charset == CHARSET
    assert pt_params.keys() == npz_params.keys()
    for name in pt_params:
        np.testing.assert_array_equal(pt_params[name], npz_params[name])
        assert pt_params[name].dtype == npz_params[name].dtype


def test_export_keeps_the_position_bands_the_model_was_trained_with(pt_path, tmp_path):
    """Losing these would make the build tokenize differently from training."""
    import exportmodel

    npz_path = str(tmp_path / "banded.npz")
    exportmodel.export_model(pt_path, npz_path)
    assert libinfer.Model.load(npz_path).position_bands == ARCH["position_bands"]


def test_the_two_formats_build_the_same_image(pt_path, tmp_path):
    """A .COM built from the .pt and from the exported .npz must be identical."""
    import buildz80com
    import exportmodel

    npz_path = str(tmp_path / "model.npz")
    exportmodel.export_model(pt_path, npz_path)
    from_pt = buildz80com.build_autoreg(pt_path, max_output_len=4).build()
    from_npz = buildz80com.build_autoreg(npz_path, max_output_len=4).build()
    assert from_pt == from_npz
