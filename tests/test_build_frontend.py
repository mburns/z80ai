"""Tests for the unified build front-end and its target selection."""

from __future__ import annotations

import build
import buildz80com
import libcpm


def test_auto_picks_the_fastest_layout_when_it_fits(tiny_model_path):
    builder, layout = build.build_cpm(tiny_model_path, max_output_len=4)
    assert layout == build.CPM_LAYOUTS[0]
    assert libcpm.fits_in_tpa(builder)


def test_auto_falls_back_to_packed_when_nothing_faster_fits(monkeypatch, tiny_model_path):
    """A model with few zero weights makes the index lists overrun the TPA."""
    monkeypatch.setattr(libcpm, "fits_in_tpa", lambda builder: False)
    _builder, layout = build.build_cpm(tiny_model_path, max_output_len=4)
    assert layout == "packed"


def test_auto_steps_down_one_layout_at_a_time(monkeypatch, tiny_model_path):
    """Only the column layout is too big, so the fast one should be next."""
    import buildcolz80com

    real = libcpm.fits_in_tpa
    monkeypatch.setattr(
        libcpm,
        "fits_in_tpa",
        lambda b: False if b.labels.get("SPLITSCAN") else real(b),
    )
    assert buildcolz80com  # the module under test is the one being rejected
    _builder, layout = build.build_cpm(tiny_model_path, max_output_len=4)
    assert layout == "fast"


def test_explicit_layouts_are_honoured(tiny_model_path):
    chosen = [
        build.build_cpm(tiny_model_path, 4, prefer=layout)[1]
        for layout in build.CPM_LAYOUTS
    ]
    assert chosen == list(build.CPM_LAYOUTS)


def test_fits_in_tpa_leaves_room_for_the_stack(tiny_model_path):
    builder = buildz80com.build_autoreg(tiny_model_path, max_output_len=4)
    end = builder.org + len(builder.build())
    assert libcpm.fits_in_tpa(builder) == (
        end + libcpm.STACK_MARGIN <= libcpm.TPA_TOP
    )


def test_ez80_auto_picks_the_fastest_kernel_when_it_fits(tiny_model_path):
    import buildez80

    _builder, layout = build.build_ez80(tiny_model_path, max_output_len=4)
    assert layout == buildez80.KERNELS[0]


def test_ez80_auto_falls_back_to_compact_when_unrolling_would_not_fit(
    monkeypatch, tiny_model_path
):
    """Shrink the ceiling rather than build a model big enough to overrun it."""
    import buildez80

    monkeypatch.setattr(buildez80, "AGON_MAX_IMAGE", 8 * 1024)
    _builder, layout = build.build_ez80(tiny_model_path, max_output_len=4)
    assert layout == "compact"


def test_ez80_explicit_kernels_are_honoured(tiny_model_path):
    import buildez80

    for kernel in buildez80.KERNELS:
        _, chosen = build.build_ez80(tiny_model_path, 4, kernel=kernel)
        assert chosen == kernel


def test_zx_tap_is_named_after_the_output_file(monkeypatch, tmp_path, tiny_model_path):
    """The name in the header is what LOAD "" CODE reports, so it should match."""
    import libzx

    out = tmp_path / "MYCHAT.TAP"
    monkeypatch.setattr(
        "sys.argv",
        ["build.py", "-m", tiny_model_path, "-t", "zx", "-o", str(out),
         "--max-output-len", "4"],
    )
    build.main()

    tap = out.read_bytes()
    assert tap[4 : 4 + libzx.TAP_NAME_LEN] == b"MYCHAT    "


def test_a_long_output_name_is_truncated_to_what_a_tap_header_holds(
    monkeypatch, tmp_path, tiny_model_path
):
    out = tmp_path / "AVERYLONGNAMEINDEED.TAP"
    monkeypatch.setattr(
        "sys.argv",
        ["build.py", "-m", tiny_model_path, "-t", "zx", "-o", str(out),
         "--max-output-len", "4"],
    )
    build.main()

    import libzx

    assert out.read_bytes()[4 : 4 + libzx.TAP_NAME_LEN] == b"AVERYLONGN"
