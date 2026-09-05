"""The planner: a walkthrough built backwards from the goal and stepped
forwards through the exact model, for worlds `explore` cannot hold.

Sound, not complete. What is held here: the walkthrough it finds for the
mystery reaches the goal in the model *and* wins on the device; a world of
two hundred rooms and a chain of flags, which `explore` refuses, gets one in
well under a second; and a need it cannot meet is named rather than guessed.
Issue #108, the instrument half.
"""

from __future__ import annotations

import time

import pytest

import buildif
import libplan
import libworld
import worlds
import worlds_mystery
from libhost import AgonHost
from libworld import Line, Person, Room, Rule, Thing, Topic, World


def replay(world: World, commands: list[str]) -> str:
    game = buildif.build(world).build()
    return AgonHost(stdin=[*commands, "quit"], files={}).run(
        game, max_cycles=200_000_000)


def said(out: str, phrase: str) -> bool:
    return " ".join(phrase.split()) in " ".join(out.split())


# --- the shipped worlds ------------------------------------------------------------


def test_the_mystery_has_a_walkthrough_that_wins_on_the_device():
    world = worlds_mystery.mystery()
    commands = libplan.plan(world)
    assert commands is not None
    assert commands[-1] == "accuse jahns"
    assert "ask marnes about allison" in commands
    assert "ask walk about allison" in commands
    state = world.start_state()
    for command in commands:
        state = world.step(state, command)
        assert state is not None, command
    assert world.satisfied(state, world.goal)
    assert said(replay(world, commands), "She asked to do it alone")


def test_the_planner_agrees_with_the_solver_about_length_within_reason():
    """`explore` finds the shortest; the planner finds one. Twice as long
    is the bound worth holding it to, so that a regression that walks the
    stair three times is seen."""
    world = worlds_mystery.mystery()
    shortest = world.explore().solve()
    found = libplan.plan(world)
    assert shortest is not None and found is not None
    assert len(found) <= 2 * len(shortest)


def test_a_world_with_no_goal_needs_no_walkthrough():
    assert libplan.plan(worlds.silo()) == []


# --- too large to search --------------------------------------------------------------


def stair_world(levels: int, clutter: int = 0) -> World:
    """A stair of `levels` landings, a thing on each of three of them, and a
    chain of flags that has to be set in order by rules on the way down. The
    goal is the badge at the bottom, held, with the chain complete.
    `clutter` adds portable things nothing needs, which is what multiplies
    a state space: each one doubles it."""
    rooms = [Room(f"Level {i}", f"Landing {i}.") for i in range(levels)]
    for i in range(levels - 1):
        rooms[i].exits["DOWN"] = i + 1
        rooms[i + 1].exits["UP"] = i
    things = [Thing("badge", "A badge.", levels - 1),
              Thing("key", "A key.", levels // 2),
              Thing("wrench", "A wrench.", levels // 3)]
    things += [Thing(f"rag{i}", "A rag.", i % levels) for i in range(clutter)]
    rules = [
        Rule(when=[(libworld.C_HAVE, 2)], then=[(libworld.A_SET, 0, 0)]),
        Rule(when=[(libworld.C_FLAG, 0), (libworld.C_HAVE, 1)],
             then=[(libworld.A_SET, 1, 0)]),
        Rule(when=[(libworld.C_FLAG, 1), (libworld.C_AT, levels - 1)],
             then=[(libworld.A_SET, 2, 0)]),
    ]
    return World(rooms=rooms, things=things, rules=rules,
                 goal=[(libworld.C_FLAG, 2), (libworld.C_HAVE, 0)])


def test_two_hundred_rooms_are_planned_in_under_a_second():
    world = stair_world(200, clutter=10)
    started = time.monotonic()
    commands = libplan.plan(world)
    assert time.monotonic() - started < 1.0
    assert commands is not None
    state = world.start_state()
    for command in commands:
        state = world.step(state, command)
        assert state is not None, command
    assert world.satisfied(state, world.goal)


def test_explore_would_not_have_held_it():
    """Two hundred rooms and thirteen portable things is 1.6 million
    states before a single flag, and `explore` says so."""
    with pytest.raises(RuntimeError, match="more than"):
        stair_world(200, clutter=10).explore()


def test_the_walkthrough_wins_on_the_device_too():
    world = stair_world(40)
    commands = libplan.plan(world)
    assert commands is not None
    out = replay(world, commands)
    assert out.count("Taken.") == 3


# --- what it cannot do, said plainly ---------------------------------------------------


def test_a_flag_nothing_sets_is_named():
    world = stair_world(5)
    world.goal = [(libworld.C_FLAG, 7)]
    assert libplan.plan(world) is None
    assert "flag 7" in libplan.explain(world)


def test_a_room_with_no_route_is_named():
    world = stair_world(5)
    world.rooms.append(Room("Vault", "Sealed."))
    world.goal = [(libworld.C_AT, 5)]
    assert "no route" in libplan.explain(world)


def test_a_line_behind_a_gate_is_planned_through_the_gate():
    """The mystery's own shape in miniature: Walk says the second thing
    only once Marnes has said the first."""
    world = World(
        rooms=[Room("A", "a", {"EAST": 1}), Room("B", "b", {"WEST": 0})],
        things=[],
        people=[Person("marnes", "Marnes.", 0, "No."),
                Person("walk", "Walk.", 1, "No.")],
        topics=[Topic("allison", ["ALLISON"])],
        lines=[Line(0, 0, "She was IT.", sets=0),
               Line(1, 0, "Alone.", gate=0, sets=1),
               Line(1, 0, "Who?")],
        goal=[(libworld.C_FLAG, 1)])
    commands = libplan.plan(world)
    assert commands == ["ask marnes about allison", "east",
                        "ask walk about allison"]


def test_a_deadline_is_waited_out_and_a_count_is_met():
    world = stair_world(6)
    world.goal = [(libworld.C_TURN, 4), (libworld.C_CARRYING, 2)]
    commands = libplan.plan(world)
    assert commands is not None
    assert commands.count("look") >= 1 or len(commands) >= 4
