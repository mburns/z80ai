"""Weight-packing tests.

The Z80 unpack loop reloads a packed byte whenever its per-neuron weight
counter hits a multiple of four, so every output neuron has to start on a byte
boundary.  Getting that wrong silently shifts every weight after the first row.
"""

from __future__ import annotations

import numpy as np
import pytest

import buildfastz80com
import libinfer

LAYOUTS = ["plain", "rotated"]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", [(4, 8), (3, 13), (1, 1), (7, 2), (5, 256)])
def test_pack_unpack_roundtrip(layout, shape):
    rng = np.random.default_rng(shape[0] * 31 + shape[1])
    w = rng.integers(-2, 2, size=shape)
    packed = libinfer.pack_2bit(w, layout=layout)
    np.testing.assert_array_equal(libinfer.unpack_2bit(packed, shape, layout), w)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("in_features", [1, 2, 3, 4, 5, 7, 8, 13, 256])
def test_each_row_starts_on_a_byte_boundary(layout, in_features):
    w = np.zeros((3, in_features), dtype=np.int8)
    packed = libinfer.pack_2bit(w, layout=layout)
    assert len(packed) == 3 * libinfer.row_stride(in_features)


def test_out_of_range_weights_are_clipped():
    w = np.array([[-5, 4, 0, 0]])
    got = libinfer.unpack_2bit(libinfer.pack_2bit(w), (1, 4))
    np.testing.assert_array_equal(got, [[-2, 1, 0, 0]])


def test_rotated_layout_codes():
    """The rotated encoding is what MULADD's two DECs decode."""
    w = np.array([[-2, -1, 0, 1]])
    byte = libinfer.pack_2bit(w, layout="rotated")[0]
    # slot0 -> bits 3:2, slot1 -> 5:4, slot2 -> 7:6, slot3 -> 1:0
    assert (byte >> 2) & 3 == 0  # -2
    assert (byte >> 4) & 3 == 3  # -1
    assert (byte >> 6) & 3 == 1  # 0
    assert byte & 3 == 2  # +1


def test_plain_layout_codes():
    w = np.array([[-2, -1, 0, 1]])
    byte = libinfer.pack_2bit(w, layout="plain")[0]
    assert [(byte >> (2 * i)) & 3 for i in range(4)] == [0, 1, 2, 3]


def test_builders_delegate_to_the_shared_packer(tiny_model_path):
    """Every packed backend uses the rotated layout, so they share one kernel.

    This used to compare four identical per-module pack_2bit_weights wrappers
    against libinfer. They are gone: libnn.prepare_packed does the packing, so
    the property is now structural rather than four coincidences. What is left
    worth checking is that it really is the rotated layout.
    """
    import libcpm
    import libnn

    packed = libnn.prepare_packed(tiny_model_path, libcpm.CPMPlatform())
    model = packed.model
    w1q, w1c = libinfer.split_query_half(model.weight(0))
    assert packed.packed_query == libinfer.pack_2bit(w1q, "rotated")
    assert packed.packed_weights[0] == libinfer.pack_2bit(w1c, "rotated")
    for i in range(1, model.num_layers):
        assert packed.packed_weights[i] == libinfer.pack_2bit(model.weight(i),
                                                              "rotated")


def test_the_plain_layout_is_still_supported():
    """Nothing emits it now, but it is the natural encoding and stays tested."""
    w = np.random.default_rng(4).integers(-2, 2, size=(3, 9))
    np.testing.assert_array_equal(
        libinfer.unpack_2bit(libinfer.pack_2bit(w, "plain"), w.shape, "plain"), w
    )


def test_fast_builder_index_lists_reproduce_the_weights():
    """The fast build stores per-value index lists; decode them back."""
    rng = np.random.default_rng(5)
    w = rng.integers(-2, 2, size=(3, 12)).astype(np.int8)
    biases = rng.integers(-1000, 1000, size=3).astype(np.int16)
    blob = buildfastz80com.pack_weights_and_biases(w, biases)

    pos = 0
    for n in range(w.shape[0]):
        recovered = np.zeros(w.shape[1], dtype=np.int8)
        for value in (-2, -1, 1):
            count = blob[pos]
            pos += 1
            for _ in range(count):
                recovered[blob[pos]] = value
                pos += 1
        np.testing.assert_array_equal(recovered, w[n])
        bias = blob[pos] | (blob[pos + 1] << 8)
        pos += 2
        assert bias == (int(biases[n]) & 0xFFFF)
    assert pos == len(blob)
