"""One list of shipped artifacts, checked from every direction.

Four places have to agree about what ships:

    build-examples.sh          builds them
    verify_artifacts.ARTIFACTS boots each one and checks what it prints
    test_codegen_stability     pins their bytes
    ci.yml                     publishes them

The middle two already check each other. These tests close the other two edges.
They exist because the release list had drifted: GUESS-COL.COM, CHAT-COL.COM,
TALK-COL.COM, CLINC.bin and TALK-PHR.bin were built, verified and pinned but
never published - and the README told people to download CHAT-COL.COM from the
releases page.
"""

from __future__ import annotations

import os
import re

import pytest

import verify_artifacts as va


def _read(repo_root: str, name: str) -> str:
    with open(os.path.join(repo_root, name), encoding="utf-8") as fh:
        return fh.read()


def build_script_outputs(script: str) -> set[str]:
    """Every file build-examples.sh writes into $OUT.

    The script builds three examples from one loop and the phrasebooks after
    it, so the loop variable has to be expanded to know what lands on disk.
    """
    loop = re.search(r"for example in ([^;]+); do", script)
    assert loop, "build-examples.sh no longer has the example loop this parses"
    names = [pair.split(":")[1] for pair in loop.group(1).split()]

    outputs = set()
    for raw in re.findall(r'"\$OUT/([^"]+)"', script):
        if "$name" in raw:
            outputs.update(raw.replace("$name", name) for name in names)
        else:
            outputs.add(raw)

    # --phrases names a companion written beside the binary, not an -o target.
    outputs.update(re.findall(r"--phrases (\S+)", script))
    return outputs


def test_the_build_script_builds_exactly_what_the_manifest_names(repo_root):
    script = _read(repo_root, "build-examples.sh")
    assert build_script_outputs(script) == set(va.release_files())


def test_the_release_publishes_everything_the_build_script_builds(repo_root):
    """`dist/*` is the whole point: a hand-written list is what went stale."""
    ci = _read(repo_root, ".github/workflows/ci.yml")
    assert re.search(r"^\s*files:\s*dist/\*\s*$", ci, re.MULTILINE), (
        "the release step should publish dist/* rather than restating the "
        "artifact list, which is how it drifted last time"
    )


def test_the_verify_step_refuses_an_incomplete_dist():
    """Which is what makes publishing dist/* safe rather than optimistic."""
    ci = _read(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
               ".github/workflows/ci.yml")
    verify_step = re.search(r"python verify_artifacts\.py[^\n]*", ci)
    assert verify_step, "CI no longer runs verify_artifacts.py"
    assert "--require-all" in verify_step.group(0)


def test_release_files_lists_every_artifact_and_its_companion():
    files = va.release_files()
    assert len(files) == len(set(files)), "a file is listed twice"
    assert set(files) == set(va.ARTIFACTS) | set(va.COMPANIONS.values())


def test_a_companion_is_listed_next_to_the_binary_that_needs_it():
    """They have to be downloaded together to be any use."""
    files = va.release_files()
    for artifact, companion in va.COMPANIONS.items():
        assert files.index(companion) == files.index(artifact) + 1


def test_missing_from_names_what_is_absent(tmp_path):
    assert set(va.missing_from(str(tmp_path))) == set(va.release_files())


def test_missing_from_is_empty_for_a_complete_dist(tmp_path):
    for name in va.release_files():
        (tmp_path / name).write_bytes(b"")
    assert va.missing_from(str(tmp_path)) == []


def test_require_all_fails_on_a_partial_dist(tmp_path, monkeypatch, capsys):
    """The regression that mattered: CI passing on a dist missing artifacts."""
    only = next(iter(va.ARTIFACTS))
    (tmp_path / only).write_bytes(b"")

    monkeypatch.setattr(
        "sys.argv",
        ["verify_artifacts.py", "--dist", str(tmp_path), "--require-all"],
    )
    assert va.main() == 1
    assert "missing" in capsys.readouterr().err


def test_the_readme_only_promises_files_a_release_publishes(repo_root):
    """The quickstart sends people to the releases page by name."""
    readme = _read(repo_root, "README.md")
    published = set(va.release_files())
    named = set(re.findall(r"`((?:GUESS|CHAT|TALK|CLINC)[A-Z0-9-]*\.[A-Za-z]+)`",
                           readme))
    assert named, "the README no longer names any artifact"
    assert named <= published, f"README promises unpublished files: {named - published}"


@pytest.mark.parametrize("artifact", sorted(va.ARTIFACTS))
def test_every_artifact_names_a_model_file_that_exists(artifact, repo_root):
    model_path, _platform = va.ARTIFACTS[artifact]
    assert os.path.exists(os.path.join(repo_root, model_path))
