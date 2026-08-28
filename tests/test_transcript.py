"""A playthrough held to the file it produced last time.

`tests/test_if.py` asserts phrases - twenty of them, out of a world that says
several thousand words. That is the right shape for "does `TAKE` work", and it
is blind to the half of an Interactive Fiction that is prose: a description
that loses a sentence, a message that gains a comma, a rule that stops firing
because the room it names moved. Every one of those keeps every assertion in
that file green.

So one test here replays a whole session and compares the lot. It fails
whenever anything printed changes, which is not the same as failing whenever
something is wrong - see `tools/transcript.py` on why that is the trade worth
making, and `--update` for what to do when the diff is the change.

The rest of the file is about the harness itself, and mostly about the one way
it could quietly lie: a run that never reaches `quit` returns a prefix of a
game, and a prefix of a game looks exactly like a short game.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import transcript
import worlds
from libworld import Room, World

SILO = transcript.TRANSCRIPTS / "silo.txt"


# --- the golden file ----------------------------------------------------------


def test_the_shipped_transcript_still_plays_the_same():
    """The whole session, not a phrase out of it.

    `python tools/transcript.py --update` is the fix when this fails and the
    diff is the change you meant.
    """
    was, now = transcript.replay(SILO)
    assert was == now, transcript.diff(SILO, was, now)


def test_the_transcript_walks_the_whole_world():
    """A golden file over four rooms would pin four rooms.

    Not a claim that these are the only turns worth recording - a claim that
    the recording is not accidentally a fragment.
    """
    text = SILO.read_text()
    world = worlds.silo()
    for room in world.rooms:
        assert room.name in text
    for message in world.messages:
        assert " ".join(message.split()) in " ".join(text.split())


# --- the format is the game's own output --------------------------------------


def test_the_turns_are_recovered_from_the_games_echo():
    """There is no second file of commands, which is the point.

    `READ_INPUT` echoes what it accepts, so the transcript records the turns
    as a side effect of recording the output. A separate script would be one
    more thing to keep in step.
    """
    turns = transcript.commands(SILO.read_text())
    assert turns[:3] == ["down", "take ledger", "down"]
    assert turns[-1] == "quit"


def test_replaying_what_was_recorded_records_the_same_thing():
    """Idempotence, which is what makes `--update` safe to run twice."""
    was, now = transcript.replay(SILO)
    again = transcript.record(worlds.silo(), transcript.commands(now),
                              "worlds:silo")
    assert now == again == was


def test_a_transcript_says_which_world_it_is_of(tmp_path):
    orphan = tmp_path / "orphan.txt"
    orphan.write_text("> quit\n")
    with pytest.raises(ValueError, match="no '# world:' line"):
        transcript.replay(orphan)


def test_the_header_stops_at_the_first_line_that_is_not_one():
    """Prose is allowed to contain a `#`; a header is not allowed to eat it."""
    assert transcript.header("# world: a:b\nnot a header\n# ignored: yes\n") \
        == {"world": "a:b"}


def test_a_world_is_named_as_module_and_function():
    assert transcript.load_world("worlds:silo").rooms[0].name \
        == worlds.silo().rooms[0].name
    with pytest.raises(ValueError, match="module:function"):
        transcript.load_world("worlds")


# --- the way it could lie -----------------------------------------------------


def test_a_run_that_never_quits_is_a_failure_rather_than_a_short_game():
    """`AgonHost` halts the CPU when its input runs dry, and says nothing.

    That is right for a host and wrong here. The run comes back *looking
    finished* - halted, no error, output that reads as a complete session -
    and `--update` would write the prefix to disk as though the game had ended
    there. Only `host.finished` distinguishes it from a short game, which is
    why this test exists rather than a cycle-count assertion.
    """
    with pytest.raises(transcript.Hung, match="did not have"):
        transcript.record(worlds.silo(), ["down", "look"], "worlds:silo")


def test_commands_after_the_end_are_a_failure_rather_than_ignored():
    """`quit` in the middle silently drops everything after it."""
    with pytest.raises(transcript.Hung, match="'down'"):
        transcript.record(worlds.silo(), ["quit", "down"], "worlds:silo")


def test_a_world_with_nothing_in_it_still_records():
    """The world compiler emits rooms out of a database and no things at all.

    Nothing else exercises that shape, and `DO_INV` branches on it.
    """
    bare = World(rooms=[Room("A", "a room", {"NORTH": 1}),
                        Room("B", "another", {"SOUTH": 0})], things=[])
    out = transcript.record(bare, ["north", "i", "quit"], "x:y")
    assert "empty-handed" in out
    assert "another" in out
