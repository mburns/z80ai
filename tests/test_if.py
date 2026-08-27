"""The turn loop: somewhere to be, rather than something to ask.

Issue #62's second scope argued that a world has to be a different machine
from the oracle, and gave a reason that is measurable rather than aesthetic:
**an ordinary move must cost no card reads at all.** The card costs about
4,600 bytes of I/O and 370,000 instructions to answer one question, which is
fine for a question and hopeless for a step, and a player takes a step every
few seconds.

So the first thing here is `io_bytes == 0`. The rest is that the world behaves
- exits lead where the table says, a thing that moves stays moved, and a word
the tables were never given is named back rather than guessed at.

`worlds.silo()` is six rooms because six is enough to walk. Nothing here is a
claim about how large a world could be; `libworld.World.overlay_bytes` is what
says that, and it is one byte a thing.
"""

from __future__ import annotations

import pytest

import buildif
import libworld
import worlds
from libhost import AgonHost
from libworld import CARRIED, NOWHERE, Room, Thing, World


@pytest.fixture(scope="module")
def game():
    return buildif.build(worlds.silo()).build()


def play(game, *commands: str) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=[*commands, "quit"], files={})
    return host.run(game, max_cycles=50_000_000), host


# --- the claim ----------------------------------------------------------------


def test_a_turn_reads_nothing_from_the_card(game):
    """The whole reason this is not the oracle."""
    _out, host = play(game, "down", "look", "take ledger", "i", "down", "up")
    assert host.io_bytes == 0


def test_a_turn_costs_a_fraction_of_a_question(game):
    """The oracle is ~370,000 instructions a question. A turn has to be small
    enough that a player never waits, and the margin is about a hundredfold."""
    _out, host = play(game, "down")
    assert host.cpu.instructions < 20_000


# --- moving -------------------------------------------------------------------


def test_the_game_starts_where_the_world_says(game):
    out, _ = play(game)
    assert "Level 1 Landing" in out


def test_an_exit_leads_where_the_table_says(game):
    out, _ = play(game, "down", "down")
    assert "The Mids Stair" in out


def test_a_direction_that_is_not_an_exit_is_refused(game):
    out, _ = play(game, "north")
    assert "You cannot go that way." in out


def test_a_short_direction_is_the_same_command(game):
    """A player types `D`, not `GO DOWN`. Compared after the echo, which is
    the one part of the transcript that is supposed to differ."""
    long_form, _ = play(game, "down")
    short_form, _ = play(game, "d")
    assert long_form.split("> down", 1)[1] == short_form.split("> d", 1)[1]


def test_moving_back_returns_to_where_it_started(game):
    out, _ = play(game, "down", "up")
    assert out.count("Level 1 Landing") == 2


# --- things, which are the whole of the state ---------------------------------


def test_a_thing_in_the_room_is_listed(game):
    out, _ = play(game, "down")
    assert "You can see ledger." in out


def test_taking_a_thing_removes_it_from_the_room(game):
    out, _ = play(game, "down", "take ledger", "look")
    assert out.count("You can see ledger.") == 1     # only before it was taken
    assert "Taken." in out


def test_a_taken_thing_stays_taken_across_a_move(game):
    """The overlay is the only place this can be remembered, so this is the
    test that it is written and read rather than recomputed."""
    out, _ = play(game, "down", "take ledger", "up", "i")
    assert "You are carrying ledger." in out


def test_dropping_leaves_it_where_you_are(game):
    out, _ = play(game, "down", "take ledger", "up", "drop ledger", "look")
    assert "Dropped." in out
    assert "You can see ledger." in out.split("Dropped.")[1]


def test_an_empty_hand_says_so(game):
    out, _ = play(game, "i")
    assert "You are empty-handed." in out


def test_a_thing_that_is_not_here_cannot_be_taken(game):
    out, _ = play(game, "take wrench")
    assert "That is not here." in out


def test_a_thing_that_is_not_portable_is_refused(game):
    out, _ = play(game, "down", "down", "east", "take screen")
    assert "That is not something you can carry." in out


def test_dropping_what_you_do_not_have_is_refused(game):
    out, _ = play(game, "drop ledger")
    assert "You are not carrying it." in out


# --- words it was never given -------------------------------------------------


def test_an_unknown_verb_is_named_back(game):
    """`examples/parser/` measured why this matters: a model answers a word it
    was never given with a confident guess, and a table can say which word."""
    out, _ = play(game, "xyzzy")
    assert "I do not know the word 'XYZZY'." in out


def test_an_unknown_noun_is_named_back(game):
    out, _ = play(game, "take zorkmid")
    assert "I do not know the word 'ZORKMID'." in out


def test_a_verb_that_needs_a_noun_says_so(game):
    out, _ = play(game, "take")
    assert "What do you want to do that to?" in out


def test_an_empty_line_is_not_a_turn(game):
    out, _ = play(game, "", "", "down")
    assert "The Cafeteria" in out


# --- the world, before anything is emitted ------------------------------------


def one_room_world(**kwargs) -> World:
    return World(rooms=[Room("A", "a")], things=[], **kwargs)


def test_a_world_with_no_rooms_is_refused():
    with pytest.raises(ValueError, match="at least one room"):
        World(rooms=[], things=[]).check()


def test_an_exit_to_a_room_that_does_not_exist_is_refused():
    world = World(rooms=[Room("A", "a", {"NORTH": 7})], things=[])
    with pytest.raises(ValueError, match="does not exist"):
        world.check()


def test_a_direction_that_is_not_a_direction_is_refused():
    world = World(rooms=[Room("A", "a", {"WIDDERSHINS": 0})], things=[])
    with pytest.raises(ValueError, match="WIDDERSHINS"):
        world.check()


def test_a_room_that_leads_to_itself_is_refused():
    """Playable-looking and unplayable: the description reprints and nothing
    moves, which reads as the game being broken rather than the map."""
    world = World(rooms=[Room("A", "a", {"NORTH": 0}), Room("B", "b")],
                  things=[])
    with pytest.raises(ValueError, match="leads NORTH to itself"):
        world.check()


def test_two_things_sharing_a_name_are_refused():
    world = World(rooms=[Room("A", "a")],
                  things=[Thing("key", "one", 0), Thing("KEY", "two", 0)])
    with pytest.raises(ValueError, match="share a name"):
        world.check()


def test_a_thing_starting_nowhere_is_refused():
    world = World(rooms=[Room("A", "a")], things=[Thing("key", "k", 9)])
    with pytest.raises(ValueError, match="does not exist"):
        world.check()


def test_a_carried_thing_may_start_carried():
    World(rooms=[Room("A", "a")], things=[Thing("key", "k", CARRIED)]).check()


# --- what a saved game would be -----------------------------------------------


def test_the_overlay_is_one_byte_a_thing_and_one_bit_a_flag():
    world = World(rooms=[Room("A", "a")],
                  things=[Thing(f"t{i}", "x", 0) for i in range(10)],
                  flags=64)
    assert world.overlay_bytes == 10 + 8 + 1


def test_the_overlay_is_one_contiguous_run():
    """So that saving it is a single `mos_fwrite` rather than three."""
    world = worlds.silo()
    builder = buildif.build(world)
    start, length = buildif.overlay_at(builder, world)
    assert length == world.overlay_bytes
    assert start == builder.labels["HERE"]


def test_the_silo_world_is_walkable():
    world = worlds.silo()
    world.check()
    assert world.reachable() == set(range(len(world.rooms)))


def test_the_room_id_is_one_byte():
    """255 rooms, with 0xFF reserved. A world that wants more wants a
    different overlay rather than a wider table, and should be told."""
    world = World(rooms=[Room(f"r{i}", "x") for i in range(NOWHERE)], things=[])
    with pytest.raises(ValueError, match="one byte"):
        world.check()
    assert libworld.NOWHERE == 0xFF
