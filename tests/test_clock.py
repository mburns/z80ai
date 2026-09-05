"""A clock the world can read, and a deadline the solver can be held to.

`worlds_mystery` has attention, which moves only when the player asks. This is
the other kind of pressure - a rule that fires because turns passed, whether or
not the player was there - and the one property that matters is that the
device and the model count the same way. Both read the number of turns already
taken: the opening pass sees zero and the pass after the N-th command sees N,
so `TURN N` fires on the N-th command and not the one before it.

Issue #101, step one.
"""

from __future__ import annotations

import pytest

import buildif
import libworld
import worlds_mystery
from libhost import AgonHost
from libworld import Room, Rule, Thing, World

A, B, C = range(3)
F_WON, F_LOST = 0, 1


def play(game: bytes, *commands: str) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=[*commands, "quit"], files={})
    return host.run(game, max_cycles=200_000_000), host


def deadline_world(deadline: int, once: bool = True) -> World:
    """Three rooms in a line, and a message when the clock reaches `deadline`."""
    return World(
        rooms=[Room("A", "a", {"EAST": B}),
               Room("B", "b", {"WEST": A, "EAST": C}),
               Room("C", "c", {"WEST": B})],
        things=[],
        messages=["There are boots on the stair."],
        rules=[Rule(when=[(libworld.C_TURN, deadline)],
                    then=[(libworld.A_PRINT, 0, 0)], once=once)])


# --- the count, on the device ---------------------------------------------------


def test_the_deadline_fires_on_the_nth_command_and_not_before():
    game = buildif.build(deadline_world(3)).build()
    early, _ = play(game, "look", "look")
    late, _ = play(game, "look", "look", "look")
    assert "boots on the stair" not in early
    assert late.count("boots on the stair") == 1


def test_a_move_is_a_turn_and_so_is_a_look():
    """The clock counts rule passes, not distance."""
    game = buildif.build(deadline_world(2)).build()
    out, _ = play(game, "east", "look")
    assert "boots on the stair" in out


def test_the_clock_still_reads_nothing_from_the_card():
    game = buildif.build(deadline_world(2)).build()
    _, host = play(game, "look", "look", "look")
    assert host.io_bytes == 0


def test_the_clock_saturates_rather_than_wrapping():
    """A clock that rolled over would hand back every deadline that had
    passed. A standing rule past 254 keeps firing through the wrap point."""
    game = buildif.build(deadline_world(254, once=False)).build()
    out, _ = play(game, *(["look"] * 300))
    # Passes after commands 254..300 inclusive, and none after.
    assert out.count("boots on the stair") == 300 - 254 + 1


# --- the count, in the model -----------------------------------------------------


def test_the_model_counts_the_way_the_device_does():
    """The solver's own answer, walked through the emulator, for a goal that
    is nothing but the clock."""
    world = deadline_world(3)
    world.goal = [(libworld.C_TURN, 3)]
    walkthrough = world.explore().solve()
    assert walkthrough is not None and len(walkthrough) == 3
    out, _ = play(buildif.build(world).build(), *walkthrough)
    assert "boots on the stair" in out


def test_a_deadline_is_a_question_about_time_not_reachability():
    """WON is set in room C, two moves away. LOST is set on the deadline.
    Whether the game can be won is the same question either way; whether it
    can be won *in time* is what the clock lets `solve` ask."""
    def world(deadline: int) -> World:
        w = deadline_world(deadline)
        w.rules.append(Rule(when=[(libworld.C_AT, C)],
                            then=[(libworld.A_SET, F_WON, 0)]))
        w.rules[0].then.append((libworld.A_SET, F_LOST, 0))
        w.goal = [(libworld.C_FLAG, F_WON), (libworld.C_NFLAG, F_LOST)]
        return w

    assert world(5).explore().solve() == ["east", "east"]
    assert world(1).explore().solve() is None


def test_a_world_that_never_reads_the_clock_has_no_bigger_a_state_space():
    """The cap is zero when nothing tests the clock, so it never leaves zero
    and `worlds_mystery` is exactly the size it is without one: 30,688
    before the accusation, and three times that since - not accused, won,
    lost - which is the accusation's doing and not the clock's."""
    search = worlds_mystery.mystery().explore()
    assert all(state.turn == 0 for state in search.states)
    assert len(search.states) == 92_064


def test_the_clock_stops_at_the_latest_deadline():
    """Above it nothing can tell one turn from another, which is what keeps
    a world with a deadline finite."""
    search = deadline_world(2).explore()
    assert max(state.turn for state in search.states) == 2


# --- what an author may write ----------------------------------------------------


def test_a_deadline_of_zero_is_refused():
    world = deadline_world(1)
    world.rules[0].when = [(libworld.C_TURN, 0)]
    with pytest.raises(ValueError, match="TURN 0"):
        world.check()


def test_a_deadline_past_the_clock_is_refused():
    world = deadline_world(1)
    world.rules[0].when = [(libworld.C_TURN, 256)]
    with pytest.raises(ValueError, match="TURN 256"):
        world.check()


def test_a_deadline_rule_is_never_dead_on_the_clocks_account():
    """`reach` is a ceiling and any deadline is reachable by waiting."""
    assert deadline_world(200).dead_rules() == []


# --- the overlay -----------------------------------------------------------------


def test_the_clock_is_in_the_saved_game():
    world = World(rooms=[Room("A", "a")], things=[Thing("t", "x", 0)])
    builder = buildif.build(world)
    start, length = buildif.overlay_at(builder, world)
    assert start <= builder.labels["CLOCK"] < start + length
    assert length == world.overlay_bytes
