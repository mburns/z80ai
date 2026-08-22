"""Tests for the unified front-end and the model-shape validation."""

from __future__ import annotations

import build
import buildfastz80com
import buildz80com
import libinfer
import pytest


def test_auto_picks_the_fast_layout_when_it_fits(tiny_model_path):
    builder, layout = build.build_cpm(tiny_model_path, max_output_len=4)
    assert layout == "fast"
    assert build._fits_in_tpa(builder)


def test_auto_falls_back_to_packed_when_fast_would_not_fit(monkeypatch, tiny_model_path):
    monkeypatch.setattr(build, "_fits_in_tpa", lambda builder: False)
    _builder, layout = build.build_cpm(tiny_model_path, max_output_len=4)
    assert layout == "packed"


def test_explicit_layouts_are_honoured(tiny_model_path):
    _, packed = build.build_cpm(tiny_model_path, 4, prefer="packed")
    _, fast = build.build_cpm(tiny_model_path, 4, prefer="fast")
    assert (packed, fast) == ("packed", "fast")


def test_fits_in_tpa_leaves_room_for_the_stack(tiny_model_path):
    builder = buildz80com.build_autoreg(tiny_model_path, max_output_len=4)
    end = builder.org + len(builder.build())
    assert build._fits_in_tpa(builder) == (
        end + build.CPM_STACK_MARGIN <= build.CPM_TPA_TOP
    )


# --- layer discovery and validation ------------------------------------------


def test_layers_are_discovered_in_numeric_order():
    """Lexical sorting would run fc10 straight after fc1."""
    import numpy as np

    params = {f"fc{i}_weight": np.zeros((1, 1)) for i in range(1, 12)}
    params.update({f"fc{i}_bias": np.zeros(1) for i in range(1, 12)})
    names, _sizes = libinfer.discover_layers(params)
    assert names == [f"fc{i}" for i in range(1, 12)]


def test_discover_layers_reports_sizes():
    import numpy as np

    params = {
        "fc1_weight": np.zeros((8, 256)), "fc1_bias": np.zeros(8),
        "fc2_weight": np.zeros((3, 8)), "fc2_bias": np.zeros(3),
    }
    _names, sizes = libinfer.discover_layers(params)
    assert sizes == [256, 8, 3]


def test_validate_accepts_the_z80_maximum():
    libinfer.validate_z80_layers([256, 256, 11])


def test_validate_rejects_layers_the_z80_cannot_count():
    with pytest.raises(ValueError, match="exceed the Z80 limit"):
        libinfer.validate_z80_layers([256, 300, 11])


@pytest.mark.parametrize(
    "module", [buildz80com, buildfastz80com],
)
def test_z80_builders_reject_oversized_models(module, tmp_path, model_factory):
    model = model_factory([256, 300, 8], charset=" AB\x00", seed=31)
    path = str(tmp_path / "wide.npz")
    model.save_npz(path)
    with pytest.raises(ValueError, match="exceed the Z80 limit"):
        module.build_autoreg(path)
