"""Tests for the benchmark harness's target table.

measure() runs a full forward pass in the emulator, which the slow-marked
end-to-end tests already cover for every backend. What is worth checking
cheaply is the table itself: a target naming a module that does not exist, or
asking for a kernel a backend does not have, fails only when someone runs the
benchmark.
"""

from __future__ import annotations

import importlib

import pytest

import bench


def test_every_target_names_an_importable_backend():
    for name, spec in bench.TARGETS.items():
        module = importlib.import_module(spec.module)
        assert hasattr(module, "build_autoreg"), f"{name} -> {spec.module}"


def test_every_kernel_named_is_one_the_backend_offers():
    import buildez80

    for name, spec in bench.TARGETS.items():
        if spec.kernel is not None:
            assert spec.kernel in buildez80.KERNELS, name


def test_only_ez80_targets_are_flagged_as_ez80():
    for name, spec in bench.TARGETS.items():
        assert spec.is_ez80 == name.startswith("ez80"), name


def test_build_kwargs_are_empty_when_the_backend_has_one_kernel():
    """buildz80com.build_autoreg has no `kernel` parameter to pass."""
    assert bench.TARGETS["com"].build_kwargs() == {}
    assert bench.TARGETS["ez80-row"].build_kwargs() == {"kernel": "row"}


def test_z80_clocks_are_the_real_machines_not_the_ez80s():
    assert bench.TARGETS["com"].clock == 4_000_000  # CP/M on a 4MHz Z80
    assert bench.TARGETS["tap"].clock == 3_500_000  # ZX Spectrum 48K
    assert bench.TARGETS["ez80"].clock > bench.TARGETS["com"].clock


@pytest.mark.parametrize("target", ["com", "tap"])
def test_measure_counts_one_forward_pass(target, tiny_model_path):
    """Both counters must advance: a zero would make every speedup infinite."""
    row = bench.measure(target, tiny_model_path, query="HI")
    assert row["instructions"] > 0
    assert row["tstates"] > row["instructions"]  # every instruction costs cycles
    assert row["bytes"] > 0


def test_the_column_layout_retires_fewer_instructions_than_the_packed_one(
    tiny_model_path,
):
    """The whole claim of the faster layouts, measured rather than asserted."""
    packed = bench.measure("com", tiny_model_path, query="HI")
    column = bench.measure("col", tiny_model_path, query="HI")
    assert column["instructions"] < packed["instructions"]
