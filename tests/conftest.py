"""Shared fixtures.

The end-to-end tests execute generated Z80 code in a pure-Python emulator, so
they use a deliberately tiny synthetic model: same code paths, ~1000x fewer
multiply-accumulates than the shipped examples.  The real examples are covered
by the ``slow``-marked tests.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/helpers.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libinfer import Model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "examples")


def load_script(path: str, name: str):
    """Import a script that lives in a subdirectory and is not on the path.

    Several of these live under ``data/`` - ``ingest.py``, ``subset.py`` - and
    are run by filename rather than imported, so a test that wants their
    internals has to load them by location.

    The `sys.modules` registration is the part that is easy to leave out and
    breaks a long way from here. Every module in this repo starts with ``from
    __future__ import annotations``, which makes annotations strings, and
    ``@dataclass`` resolves those through ``sys.modules[cls.__module__]`` - so
    a script loaded without being registered raises ``'NoneType' object has no
    attribute '__dict__'`` at import, the first time anyone adds a dataclass to
    it. Three copies of this function each had that bug.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Roughly the weight distribution QAT produces: mostly zero, few -2s.
_WEIGHT_VALUES = np.array([-2, -1, 0, 1], dtype=np.int8)
_WEIGHT_PROBS = np.array([0.05, 0.15, 0.65, 0.15])


def make_model(
    layer_sizes: list[int],
    charset: str = " ABCDEFGHIJ\x00",
    seed: int = 7,
    position_bands: int = 1,
) -> Model:
    """Build a random but deterministic quantized model."""
    rng = np.random.default_rng(seed)
    sizes = [*list(layer_sizes), len(charset)]
    weights, biases = [], []
    for nin, nout in itertools.pairwise(sizes):
        weights.append(
            rng.choice(_WEIGHT_VALUES, size=(nout, nin), p=_WEIGHT_PROBS).astype(np.int32)
        )
        biases.append(rng.integers(-400, 400, size=nout).astype(np.int32))
    return Model(weights=weights, biases=biases, charset=charset,
                 position_bands=position_bands)


@pytest.fixture(scope="session")
def model_factory():
    """Expose :func:`make_model` to tests that need a bespoke architecture."""
    return make_model


@pytest.fixture(scope="session")
def tiny_model() -> Model:
    return make_model([256, 16, 12])


@pytest.fixture(scope="session")
def banded_model() -> Model:
    """Same shape as tiny_model, but tokenized with position bands."""
    return make_model([256, 16, 12], position_bands=8)


@pytest.fixture(scope="session")
def banded_model_path(tmp_path_factory, banded_model: Model) -> str:
    path = str(tmp_path_factory.mktemp("models") / "banded.npz")
    banded_model.save_npz(path)
    return path


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory, tiny_model: Model) -> str:
    path = str(tmp_path_factory.mktemp("models") / "tiny.npz")
    tiny_model.save_npz(path)
    return path


@pytest.fixture(scope="session")
def odd_model() -> Model:
    """Layer widths that are not multiples of four, to catch packing desync."""
    return make_model([256, 13, 7], charset=" ABC\x00", seed=11)


@pytest.fixture(scope="session")
def odd_model_path(tmp_path_factory, odd_model: Model) -> str:
    path = str(tmp_path_factory.mktemp("models") / "odd.npz")
    odd_model.save_npz(path)
    return path


@pytest.fixture(scope="session")
def examples_dir() -> str:
    return EXAMPLES


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO


@pytest.fixture(scope="session")
def guess_model_path() -> str:
    path = os.path.join(EXAMPLES, "guess", "model.npz")
    if not os.path.exists(path):
        pytest.skip("guess example model not present")
    return path


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: runs a full-size model in the emulator")
