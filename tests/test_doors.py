"""Doors: a dwelling as a word on a corridor, knocked on and never entered.

The silo has 2,088 opened dwellings and a room id is one byte. A door is
what a dwelling becomes when it stops being a room: a name and a sentence
in the image, no overlay, looked up only among the doors of the room the
player stands in. Issue #103.
"""

from __future__ import annotations

import pytest

import buildif
from libhost import AgonHost
from libworld import Door, Person, Room, Thing, World

LANDING, RING_A, RING_B = range(3)


def corridors() -> World:
    """Two rings off one landing, with the same numbers painted on both."""
    return World(
        rooms=[Room("Landing", "The stair.", {"WEST": RING_A, "DOWN": RING_B}),
               Room("Ring 1", "Doors all the way round.", {"EAST": LANDING}),
               Room("Ring 2", "More doors.", {"UP": LANDING})],
        things=[Thing("ledger", "A ledger.", RING_A)],
        doors=[Door(RING_A, "600A", "The name beside the door is Wong."),
               Door(RING_A, "630A", "Nobody has the key to this one."),
               Door(RING_B, "600A", "The name beside the door is Torres.")])


def play(world: World, *commands: str) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=[*commands, "quit"], files={})
    return host.run(buildif.build(world).build(), max_cycles=50_000_000), host


def said(out: str, phrase: str) -> bool:
    return " ".join(phrase.split()) in " ".join(out.split())


# --- knocking ----------------------------------------------------------------------


def test_knocking_says_what_the_door_says():
    out, _ = play(corridors(), "west", "knock 600A")
    assert said(out, "The name beside the door is Wong.")


def test_the_same_number_on_another_floor_is_another_door():
    out, _ = play(corridors(), "down", "knock 600A")
    assert said(out, "The name beside the door is Torres.")
    assert "Wong" not in out


def test_a_door_that_is_not_here_is_named_back():
    """Not "I do not know the word": the word may be a door somewhere
    else, and the useful reply is that it is not one here."""
    out, _ = play(corridors(), "knock 600A")
    assert said(out, "There is no door marked '600A' here.")
    out, _ = play(corridors(), "west", "knock 900C")
    assert said(out, "There is no door marked '900C' here.")


def test_knocking_on_nothing_asks_which():
    out, _ = play(corridors(), "west", "knock")
    assert said(out, "Knock on which door?")


def test_examine_reaches_a_door_after_the_things():
    out, _ = play(corridors(), "west", "x 630A", "x ledger")
    assert said(out, "Nobody has the key to this one.")
    assert said(out, "A ledger.")


def test_a_door_costs_no_card_bytes_and_no_overlay():
    world = corridors()
    _, host = play(world, "west", "knock 600A", "knock 630A", "x 600A")
    assert host.io_bytes == 0
    bare = corridors()
    bare.doors = []
    assert world.overlay_bytes == bare.overlay_bytes


def test_seventy_two_doors_cost_a_knock_a_few_thousand_instructions():
    """A silo floor. The scan is the room's own word table and nothing
    else, so it is bounded by one floor however many floors there are."""
    world = corridors()
    world.doors = [Door(RING_A, f"{h}{m:02d}{r}", f"Door {h}{m:02d}{r}.")
                   for h in range(1, 13) for m in (0, 30) for r in "ABC"]
    assert len(world.doors) == 72
    _, quiet = play(world, "west")
    _, knocked = play(world, "west", "knock 1130C")      # the last one
    assert knocked.cpu.instructions - quiet.cpu.instructions < 40_000


# --- what an author may write -----------------------------------------------------


def test_a_door_on_a_room_that_does_not_exist_is_refused():
    world = corridors()
    world.doors.append(Door(9, "700A", "x"))
    with pytest.raises(ValueError, match="does not exist"):
        world.check()


def test_two_doors_with_one_number_on_one_ring_are_refused():
    world = corridors()
    world.doors.append(Door(RING_A, "600a", "x"))
    with pytest.raises(ValueError, match="two doors"):
        world.check()


def test_a_door_that_is_also_a_thing_or_a_person_is_refused():
    world = corridors()
    world.doors.append(Door(RING_A, "ledger", "x"))
    with pytest.raises(ValueError, match="also a thing or a person"):
        world.check()
    world = corridors()
    world.people.append(Person("marnes", "Marnes is here.", LANDING, "No."))
    world.doors.append(Door(RING_A, "marnes", "x"))
    with pytest.raises(ValueError, match="also a thing or a person"):
        world.check()


def test_a_door_word_the_parser_cannot_hold_is_refused():
    world = corridors()
    world.doors.append(Door(RING_A, "apartment 600 A", "x"))
    with pytest.raises(ValueError, match="one word"):
        world.check()


def test_a_silent_door_is_refused():
    world = corridors()
    world.doors.append(Door(RING_A, "700A", "  "))
    with pytest.raises(ValueError, match="says nothing"):
        world.check()


def test_more_doors_than_a_byte_on_one_ring_are_refused():
    world = corridors()
    world.doors = [Door(RING_A, f"d{i}", "x") for i in range(256)]
    with pytest.raises(ValueError, match="one byte"):
        world.check()


def test_a_world_with_no_doors_still_answers_a_knock():
    world = corridors()
    world.doors = []
    out, _ = play(world, "west", "knock 600A")
    assert said(out, "There is no door marked '600A' here.")
