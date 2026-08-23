"""Tests for the shared build preamble: libinfer.BuildInputs and load_for_build.

Every backend used to open the checkpoint, sort the layers and print the same
five lines itself. These tests cover the one copy that is left, including the
numeric layer ordering that a lexical sort gets wrong past nine layers.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import libinfer


def test_layer_names_sort_numerically_not_lexically():
    """fc10 runs after fc9, not after fc1 - a lexical sort transposes them."""
    params = {f"fc{i}_weight": None for i in range(1, 12)}
    params.update({f"fc{i}_bias": None for i in range(1, 12)})
    assert libinfer.layer_names(params) == [f"fc{i}" for i in range(1, 12)]


def test_discover_layers_and_from_params_agree_on_the_order():
    """Both used to sort independently; only one of them is allowed to be right."""
    sizes = [8, 5, 4, 3]
    params = {}
    for i, (nin, nout) in enumerate(itertools.pairwise(sizes), start=1):
        params[f"fc{i}_weight"] = np.zeros((nout, nin), dtype=np.int8)
        params[f"fc{i}_bias"] = np.zeros(nout, dtype=np.int16)

    names, discovered = libinfer.discover_layers(params)
    model = libinfer.Model.from_params(params, charset="ABC")
    assert names == ["fc1", "fc2", "fc3"]
    assert discovered == sizes == model.layer_sizes


@pytest.fixture(scope="module")
def inputs(tiny_model_path) -> libinfer.BuildInputs:
    return libinfer.load_for_build(tiny_model_path)


def test_geometry_matches_the_model_it_was_loaded_from(inputs, tiny_model):
    assert inputs.layer_sizes == tiny_model.layer_sizes
    assert inputs.num_layers == tiny_model.num_layers
    assert inputs.input_size == tiny_model.input_size
    assert inputs.output_size == tiny_model.output_size
    assert inputs.charset == tiny_model.charset


def test_eos_is_the_last_charset_entry(inputs):
    """GENERATE stops on this index, so an off-by-one prints garbage forever."""
    assert inputs.eos_idx == len(inputs.charset) - 1


def test_weight_and_bias_are_indexed_from_zero(inputs, tiny_model):
    for i in range(inputs.num_layers):
        np.testing.assert_array_equal(inputs.weight(i), tiny_model.weights[i])
        np.testing.assert_array_equal(inputs.bias(i), tiny_model.biases[i])


def test_weights_and_biases_return_every_layer_in_run_order(inputs, tiny_model):
    for got, want in zip(inputs.weights(), tiny_model.weights, strict=True):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(inputs.biases(), tiny_model.biases, strict=True):
        np.testing.assert_array_equal(got, want)


def test_position_bands_travel_with_the_model(banded_model_path, banded_model):
    """A build that tokenizes differently from training answers confidently wrong."""
    assert libinfer.load_for_build(banded_model_path).position_bands == (
        banded_model.position_bands
    )


def test_a_model_without_bands_defaults_to_flat(inputs):
    assert inputs.position_bands == libinfer.FLAT


def test_report_names_the_file_the_charset_and_the_shape(tiny_model_path, capsys):
    libinfer.load_for_build(tiny_model_path)
    out = capsys.readouterr().out
    assert tiny_model_path in out
    assert "+ EOS" in out
    assert "256 → 16 → 12 → 12" in out
    assert f"{libinfer.NUM_BUCKETS} query + {libinfer.NUM_BUCKETS} context" in out


def test_report_io_suppresses_the_bucket_split(tiny_model_path, capsys):
    """The eZ80 backend leaves it out: its layer widths are unconstrained."""
    libinfer.load_for_build(tiny_model_path, report_io=False)
    out = capsys.readouterr().out
    assert "Architecture:" in out
    assert "query + " not in out


def test_the_codegen_and_the_reference_share_one_encoding_geometry():
    """libnn generates code against these; libinfer encodes with them."""
    import libnn

    assert libnn.NUM_BUCKETS is libinfer.NUM_BUCKETS
    assert libnn.CONTEXT_LEN is libinfer.CONTEXT_LEN
    assert libnn.BUCKET_WEIGHT is libinfer.BUCKET_WEIGHT
    assert libnn.CONTEXT_OFFSET == libinfer.NUM_BUCKETS * libnn.ACTIVATION_SIZE
