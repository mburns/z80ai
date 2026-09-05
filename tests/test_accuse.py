"""The accusation: one shot, a person or a door, and a flag either way.

The engine does not gate an accusation on evidence. An author who wants the
clues found first puts them in the goal beside `won`, which is what the
mystery does, and the generator will prove they are enough. What the engine
holds to is narrower and exact: the right name wins, any other name loses,
there is only the one, and a restore does not give it back. Issue #112, PR A.
"""

from __future__ import annotations

import pytest

import buildif
import libworld
import worlds_mystery
from libhost import AgonHost
from libworld import Door, Person, Room, Rule, World

RING, LANDING = 0, 1
F_WON, F_LOST = 0, 1


def doors_world() -> World:
    return World(
        rooms=[Room("Ring", "Doors.", {"EAST": LANDING}),
               Room("Landing", "The stair.", {"WEST": RING})],
        things=[],
        people=[Person("marnes", "Marnes is here.", LANDING, "No comment.")],
        doors=[Door(RING, "600A", "Wong lives here.", subject="Alexander E. Wong"),
               Door(RING, "630A", "Becker lives here.", subject="Holston Becker"),
               Door(RING, "700A", "Nobody has the key.")],
        messages=["The deputy nods.", "The deputy sighs."],
        rules=[Rule(when=[(libworld.C_FLAG, F_WON)], then=[(libworld.A_PRINT, 0, 0)]),
               Rule(when=[(libworld.C_FLAG, F_LOST)], then=[(libworld.A_PRINT, 1, 0)])],
        culprit="alexander e. wong", won=F_WON, lost=F_LOST,
        win_text="Wong does not deny it.", lose_text="Wrong.",
        goal=[(libworld.C_FLAG, F_WON)])


def play(world: World, *commands: str,
         files: dict[str, bytes] | None = None) -> tuple[str, AgonHost]:
    host = AgonHost(stdin=[*commands, "quit"], files=files or {})
    return host.run(buildif.build(world).build(), max_cycles=50_000_000), host


def said(out: str, phrase: str) -> bool:
    return " ".join(phrase.split()) in " ".join(out.split())


# --- on the device ----------------------------------------------------------------


def test_the_right_door_wins_and_the_rules_see_it():
    out, _ = play(doors_world(), "accuse 600A")
    assert said(out, "Wong does not deny it.")
    assert said(out, "The deputy nods.")


def test_the_wrong_door_loses():
    out, _ = play(doors_world(), "accuse 630A")
    assert said(out, "Wrong.")
    assert said(out, "The deputy sighs.")


def test_a_person_can_be_accused_from_anywhere_and_loses_here():
    out, _ = play(doors_world(), "accuse marnes")
    assert said(out, "Wrong.")


def test_a_door_has_to_be_in_the_room():
    out, _ = play(doors_world(), "east", "accuse 600A")
    assert "I do not know the word '600A'." in out
    assert "Wong does not deny" not in out


def test_an_empty_flat_cannot_be_accused():
    out, _ = play(doors_world(), "accuse 700A")
    assert "Nobody answers." in out


def test_there_is_only_the_one_accusation():
    out, _ = play(doors_world(), "accuse 630A", "accuse 600A")
    assert said(out, "Wrong.")
    assert said(out, "You have made your accusation.")
    assert "Wong does not deny" not in out


def test_a_restore_does_not_give_the_accusation_back():
    _, saved = play(doors_world(), "save")
    out, _ = play(doors_world(), "accuse 630A", "restore", "accuse 600A",
                  files=saved.files)
    # The save was taken before the accusation, so restoring it does give
    # it back - which is the overlay working. Saving *after* does not.
    assert said(out, "Wong does not deny it.")
    _, saved = play(doors_world(), "accuse 630A", "save")
    out, _ = play(doors_world(), "restore", "accuse 600A", files=saved.files)
    assert said(out, "You have made your accusation.")


def test_accuse_alone_asks_whom_and_is_not_a_turn():
    world = doors_world()
    world.rules.append(Rule(when=[(libworld.C_TURN, 1)],
                            then=[(libworld.A_PRINT, 1, 0)]))
    out, _ = play(world, "accuse")
    assert "Accuse whom?" in out
    assert "The deputy sighs" not in out


def test_a_world_with_no_culprit_says_so():
    world = doors_world()
    world.culprit = world.won = world.lost = None
    world.rules, world.goal = [], []              # nothing sets the flags now
    out, _ = play(world, "accuse 600A")
    assert "nobody here to accuse" in out


def test_the_mystery_is_won_by_accusing_the_mayor():
    world = worlds_mystery.mystery()
    game = buildif.build(world).build()
    walkthrough = world.explore().solve()
    assert walkthrough is not None and walkthrough[-1] == "accuse jahns"
    out = AgonHost(stdin=[*walkthrough, "quit"], files={}).run(
        game, max_cycles=50_000_000)
    assert said(out, "She asked to do it alone")


def test_the_mystery_cannot_be_won_by_guessing():
    """An accusation on turn one is legal, and it is not the goal: the goal
    names the clues, so `solve` never reports a walkthrough that guessed."""
    world = worlds_mystery.mystery()
    out = AgonHost(stdin=["accuse jahns", "quit"], files={}).run(
        buildif.build(world).build(), max_cycles=50_000_000)
    assert said(out, "She asked to do it alone")
    walkthrough = world.explore().solve()
    assert walkthrough is not None and "ask walk about allison" in walkthrough


# --- what an author may write --------------------------------------------------------


def test_a_culprit_nobody_could_accuse_is_refused():
    world = doors_world()
    world.culprit = "Nobody Real"
    with pytest.raises(ValueError, match="neither a person"):
        world.check()


def test_a_culprit_needs_both_flags_and_they_differ():
    world = doors_world()
    world.lost = None
    with pytest.raises(ValueError, match="won flag and a lost flag"):
        world.check()
    world = doors_world()
    world.lost = world.won
    with pytest.raises(ValueError, match="same flag"):
        world.check()


def test_flags_without_a_culprit_are_refused():
    world = doors_world()
    world.culprit = None
    with pytest.raises(ValueError, match="without a culprit"):
        world.check()


def test_the_overlay_carries_the_accusation():
    world = doors_world()
    builder = buildif.build(world)
    start, length = buildif.overlay_at(builder, world)
    assert length == world.overlay_bytes
    assert start <= builder.labels["ACCUSED"] < start + length


def test_explore_models_the_one_shot():
    world = doors_world()
    search = world.explore()
    assert search.solve() == ["accuse 600a"]
    assert max(s.accused for s in search.states) == 1
    assert not any(s.flags[F_WON] and s.flags[F_LOST] for s in search.states)
