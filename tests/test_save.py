"""Save, restore, and the one file on the card that outlives a game.

Three claims, and the first is the one the others rest on: a game saved and
restored plays on **exactly** as one that was not. Neither verb is a turn, so
the rules do not run and the clock does not tick, and the overlay comes back
byte for byte - `ASKED` and `FIRED` included, which is what stops a restore
re-explaining everything and firing every event a second time.

The second is that a restore that finds the wrong file touches nothing. The
overlay is a run of bytes with no names in it, so a save from a world with
the same shape would load into another without complaint; the header is what
says no.

The third is the archive's log. Every question the card sees goes on the end
of `SILO.LOG`, a game reads the length when it starts, and `C_LOGGED` lets a
rule act on it - so the Voice in the second game knows it has met this
player before, and a restore does not make it forget. Issue #104.
"""

from __future__ import annotations

import pytest

import buildif
import libworld
import worlds
import worlds_mystery
from libhost import AgonHost
from libworld import Room, Rule, World

A, B, C = range(3)


@pytest.fixture(scope="module")
def game():
    return buildif.build(worlds_mystery.mystery()).build()


def play(game: bytes, *commands: str,
         files: dict[str, bytes] | None = None) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=[*commands, "quit"], files=files or {})
    return host.run(game, max_cycles=200_000_000), host


def said(out: str, phrase: str) -> bool:
    return " ".join(phrase.split()) in " ".join(out.split())


def clock_world(deadline: int, once: bool = True) -> World:
    """Three rooms in a line and a message when the clock reaches `deadline`."""
    return World(
        rooms=[Room("A", "a", {"EAST": B}),
               Room("B", "b", {"WEST": A, "EAST": C}),
               Room("C", "c", {"WEST": B})],
        things=[],
        messages=["There are boots on the stair.", "You have been here before."],
        rules=[Rule(when=[(libworld.C_TURN, deadline)],
                    then=[(libworld.A_PRINT, 0, 0)], once=once)])


#: Into the cafeteria, told where Allison worked, and down one more.
BEFORE = ("down", "ask marnes about allison", "down")
#: Into IT, where Walk's second line needs what Marnes said, and back up.
AFTER = ("east", "ask walk about allison", "west", "up")


# --- the file -----------------------------------------------------------------


def test_a_save_is_one_file_of_header_and_overlay(game):
    world = worlds_mystery.mystery()
    _, host = play(game, "save")
    assert set(host.files) == {"SILO1.SAV"}
    data = host.files["SILO1.SAV"]
    assert len(data) == 4 + world.overlay_bytes
    assert data[:4] == b"SV" + bytes([world.stamp & 0xFF, world.stamp >> 8])


def test_a_turn_writes_nothing(game):
    _, host = play(game, *BEFORE, *AFTER, "look", "inventory")
    assert host.files == {}


# --- the round trip -------------------------------------------------------------


def test_a_game_saved_and_restored_plays_on_as_one_that_was_not(game):
    """The whole claim. Flags, `ASKED`, attention and where everybody is come
    back with the overlay, and Walk's gated line is what shows it: she only
    says it once Marnes has said where Allison worked."""
    straight, _ = play(game, *BEFORE, *AFTER)
    _, saved = play(game, *BEFORE, "save")
    restored, _ = play(game, "restore", *AFTER, files=saved.files)

    assert said(restored, "Restored.")
    assert said(restored, "She fitted the landing screen")
    assert straight.split("> east", 1)[1] == restored.split("> east", 1)[1]


def test_the_clock_comes_back_with_the_overlay():
    """A deadline is three looks away at the save. It is one look away after
    the restore, not three - and the restore itself is not one of them."""
    game = buildif.build(clock_world(3)).build()
    _, saved = play(game, "look", "look", "save")
    early, _ = play(game, "restore", files=saved.files)
    late, _ = play(game, "restore", "look", files=saved.files)
    assert "boots on the stair" not in early
    assert "boots on the stair" in late


def test_a_restore_does_not_refire_what_has_already_fired():
    """`FIRED` is in the overlay for exactly this. Without it the boots would
    come up the stair a second time and it would look like the game working."""
    world = clock_world(1)
    world.rules = [Rule(when=[(libworld.C_AT, B)],
                        then=[(libworld.A_PRINT, 1, 0)])]
    game = buildif.build(world).build()
    _, saved = play(game, "east", "save")
    out, _ = play(game, "restore", "look", "look", files=saved.files)
    assert "been here before" not in out


def test_saving_and_restoring_are_not_turns():
    game = buildif.build(clock_world(1)).build()
    quiet, host = play(game, "save", "restore", "restore")
    loud, _ = play(game, "save", "look", files=host.files)
    assert "boots on the stair" not in quiet
    assert "boots on the stair" in loud


def test_an_empty_line_is_not_a_turn_for_the_clock_either():
    game = buildif.build(clock_world(1)).build()
    out, _ = play(game, "", "", "")
    assert "boots on the stair" not in out


# --- slots ------------------------------------------------------------------------


def test_a_slot_is_a_digit_and_the_default_is_one(game):
    _, host = play(game, "save 3")
    assert set(host.files) == {"SILO3.SAV"}
    out, _ = play(game, "restore 2", files=host.files)
    assert said(out, "There is no saved game in that slot.")
    out, _ = play(game, "restore 3", files=host.files)
    assert said(out, "Restored.")


@pytest.mark.parametrize("word", ["x", "0", "12", "ten"])
def test_a_slot_that_is_not_one_to_nine_is_refused(game, word):
    out, host = play(game, f"save {word}")
    assert said(out, "Which slot? 1 to 9.")
    assert host.files == {}


# --- the wrong file -------------------------------------------------------------


def test_another_worlds_save_is_refused(game):
    """`worlds.silo` and the mystery both save to `SILO1.SAV`, and only the
    stamp tells them apart. The player stays where they were."""
    other = buildif.build(worlds.silo()).build()
    _, saved = play(other, "down", "save")
    out, _ = play(game, "restore", "look", files=saved.files)
    assert said(out, "That is not a saved game for this silo.")
    assert "Restored." not in out
    assert out.count("Level 1 Landing") == 2      # the opening, and the look


def test_a_short_file_is_refused(game):
    out, _ = play(game, "restore", files={"SILO1.SAV": b"SV"})
    assert said(out, "That is not a saved game for this silo.")


def test_a_file_of_the_right_length_with_the_wrong_stamp_is_refused(game):
    world = worlds_mystery.mystery()
    junk = b"SV\xff\xff" + bytes(world.overlay_bytes)
    out, _ = play(game, "restore", files={"SILO1.SAV": junk})
    assert said(out, "That is not a saved game for this silo.")


def test_the_stamp_is_the_shape_and_not_the_prose():
    a = worlds_mystery.mystery()
    b = worlds_mystery.mystery()
    b.rooms[0].description = "Somewhere else entirely."
    assert a.stamp == b.stamp
    b.flags += 1
    assert a.stamp != b.stamp


def test_a_save_name_that_would_not_make_an_8_3_filename_is_refused():
    world = clock_world(1)
    world.save_name = "MYSTERY"
    with pytest.raises(ValueError, match=r"8\.3"):
        world.check()


# --- the log ----------------------------------------------------------------------

TITLES = ["Cleaning Record 218-04",
          "Incident Report 214-11: Cistern Pump Failure",
          "Standing Order 11: Screen Fitting",
          "Filler"]
LEADS = ["Allison Becker, IT, Level 34. Sent to clean on the fourth day.",
         "The cistern pump on Level 142 stopped without warning.",
         "A screen is fitted by two people and never by one.",
         "Nothing the world has a name for."]

#: Down to the IT office and sat at the screen.
AT_SCREEN = ("down", "down", "east", "use")
KNEW = "The screen already knew your name."


def remembering() -> World:
    """The mystery, plus a rule that reads the log."""
    world = worlds_mystery.mystery()
    world.messages.append(KNEW)
    world.rules.append(Rule(when=[(libworld.C_LOGGED, 2)],
                            then=[(libworld.A_PRINT, len(world.messages) - 1, 0)]))
    return world


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    import buildwikibin
    import libsearch

    out = tmp_path_factory.mktemp("log")
    index = libsearch.build(TITLES, LEADS, {})
    libsearch.write_index(index, out / "W.IDX")
    libsearch.write_text(index, out / "W.DAT")
    world = remembering()
    libworld.resolve_topics(world, TITLES)
    builder = buildwikibin.build(index.num_docs, index_name="W.IDX",
                                 text_name="W.DAT", world=world)
    return builder.build(), {"W.IDX": (out / "W.IDX").read_bytes(),
                             "W.DAT": (out / "W.DAT").read_bytes()}


def visit(merged, *commands: str,
          files: dict[str, bytes] | None = None) -> tuple[str, AgonHost]:
    game, card = merged
    host = AgonHost(stdin=[*commands, "!"], files={**card, **(files or {})})
    return host.run(game, max_cycles=2_000_000_000), host


def test_walking_and_talking_write_nothing(merged):
    _, host = visit(merged, "down", "ask marnes about allison", "down", "up")
    assert "SILO.LOG" not in host.files


def test_every_question_the_card_sees_is_logged(merged):
    """Two bytes a question: the clock, and the topic - or 0xFF for a record
    the world has no name for, which the archive still saw. A question that
    matched nothing is not a question about anything and is not logged."""
    _, host = visit(merged, *AT_SCREEN, "pump", "allison", "filler", "zzqqxx")
    log = host.files["SILO.LOG"]
    assert len(log) == 6
    assert list(log[1::2]) == [1, 0, 0xFF]           # pump, allison, filler
    assert list(log[0::2]) == [5, 6, 7]              # four turns in, then one each


def test_the_log_outlives_the_game(merged):
    """Two questions in one game, and the next game on the same card opens
    with the Voice already knowing. That is the whole of series memory."""
    first, host = visit(merged, *AT_SCREEN, "pump", "allison")
    assert said(first, KNEW)
    assert first.index(KNEW) > first.index("RECORD SEALED")

    second, _ = visit(merged, files={"SILO.LOG": host.files["SILO.LOG"]})
    assert said(second, KNEW)
    assert second.index(KNEW) < second.index("> ")   # before the first prompt


def test_a_second_game_counts_on_from_the_log(merged):
    _, host = visit(merged, *AT_SCREEN, "pump")
    out, _ = visit(merged, *AT_SCREEN, "pump",
                   files={"SILO.LOG": host.files["SILO.LOG"]})
    assert said(out, KNEW)
    assert out.index(KNEW) > out.index("> pump")


def test_a_restore_does_not_make_the_archive_forget(merged):
    """`LOGGED` is outside the overlay. Save with the log at one, ask a
    second question, restore: the rule fires again, because the count is
    the file's and the file did not go back."""
    _, host = visit(merged, *AT_SCREEN, "pump", "leave", "save")
    out, _ = visit(merged, *AT_SCREEN, "allison", "leave", "restore", "look",
                   files=host.files)
    assert " ".join(out.split()).count(KNEW) == 2


def test_the_standalone_binary_reads_the_same_log():
    """No card to ask, but the same card in the slot: a world binary opens
    knowing what the oracle binary was asked."""
    game = buildif.build(remembering()).build()
    silent, _ = play(game)
    knowing, _ = play(game, files={"SILO.LOG": bytes([1, 1, 2, 0])})
    assert not said(silent, KNEW)
    assert said(knowing, KNEW)


# --- the model ------------------------------------------------------------------


def test_explore_reaches_a_rule_that_reads_the_log():
    world = remembering()
    world.goal = [(libworld.C_LOGGED, 2)]
    search = world.explore()
    assert search.solve() == ["down", "down", "east", "use",
                              "archive allison", "archive allison"]
    assert "rule 5" not in search.unseen()["rules"]


def test_a_world_that_never_reads_the_log_keeps_it_at_zero():
    search = worlds_mystery.mystery().explore()
    assert all(state.logged == 0 for state in search.states)


def test_a_log_count_of_zero_is_refused():
    world = remembering()
    world.rules[-1].when = [(libworld.C_LOGGED, 0)]
    with pytest.raises(ValueError, match="LOGGED 0"):
        world.check()
