"""The Voice: an archive that seals and rewrites records because of what the
player did, and rules that can read what it has done.

Two overlay bytes a topic, four actions and two conditions. What they are
held to here is the fair-play contract `IF.md` states for the archive: it may
decline, and when it states a fact that fact holds, *except where it has
been sealed and says so* - or, now, altered in a fixed way the player can
catch. A seal wins over an alteration, an unsealed record reads as written,
and a truthed record reads as written; `data/silo/plant.py` set the rule
that a stable lie is a clue and a random one is noise, and every one of
these is a rule firing, not a coin. Issue #102, less the margin hedge.
"""

from __future__ import annotations

import pytest

import buildif
import libworld
import worlds_mystery
from libhost import AgonHost
from libworld import Room, Rule, Topic, World

CLEANING = "Cleaning Record 218-04"
INCIDENT = "Incident Report 214-11: Cistern Pump Failure"
ORDER = "Standing Order 11: Screen Fitting"
FILLER = "Hatch Log"
TITLES = [CLEANING, INCIDENT, ORDER, FILLER]
LEADS = ["Allison Becker, IT, Level 34. Sent to clean on the fourth day.",
         "The cistern pump on Level 142 stopped without warning.",
         "A screen is fitted by two people and never by one.",
         "The hatch has not been opened in living memory."]

T_PUMP, T_ORDER, T_RECORD, T_HATCH = range(4)
AMENDED = "AMENDED: ONE PERSON MAY FIT A SCREEN WHERE A SECOND IS NOT AVAILABLE."
CLOSED = "The pump report is closed."
CHANGED = "The order has been changed."


def voice_world() -> World:
    """One room with the terminal in it, and a clock that drives the Voice.

    The rules fire on the clock so that the sequence is fixed: the pump
    report starts sealed and opens on turn 2, the order is rewritten on turn
    3 and put right on turn 5, the cleaning record is sealed on turn 4, and
    on turn 6 the order is both sealed and altered at once.
    """
    return World(
        rooms=[Room("IT", "Racks of machines, and the screen.")],
        things=[], terminal=0,
        topics=[
            Topic("pump", ["PUMP"], titles=[INCIDENT], censor="SEALED BY JUDICIAL."),
            Topic("order", ["ORDER"], titles=[ORDER], alter=AMENDED),
            Topic("record", ["RECORD"], titles=[CLEANING]),
            Topic("hatch", ["HATCH"], titles=[FILLER], censor="NOT FOR YOU.",
                  sealed=False),
        ],
        messages=[CLOSED, CHANGED],
        rules=[
            Rule(when=[(libworld.C_TURN, 2)], then=[(libworld.A_UNSEAL, T_PUMP, 0)]),
            Rule(when=[(libworld.C_TURN, 3)], then=[(libworld.A_ALTER, T_ORDER, 0)]),
            Rule(when=[(libworld.C_TURN, 4)], then=[(libworld.A_SEAL, T_RECORD, 0)]),
            Rule(when=[(libworld.C_TURN, 5)], then=[(libworld.A_TRUTH, T_ORDER, 0)]),
            Rule(when=[(libworld.C_TURN, 6)], then=[(libworld.A_ALTER, T_ORDER, 0),
                                                    (libworld.A_SEAL, T_ORDER, 0)]),
            Rule(when=[(libworld.C_SEALED, T_PUMP)], then=[(libworld.A_PRINT, 0, 0)]),
            Rule(when=[(libworld.C_ALTERED, T_ORDER)], then=[(libworld.A_PRINT, 1, 0)]),
        ])


def said(out: str, phrase: str) -> bool:
    return " ".join(phrase.split()) in " ".join(out.split())


def order(out: str, *phrases: str) -> bool:
    """Whether the phrases appear in this order, each after the last."""
    flat = " ".join(out.split())
    position = 0
    for phrase in phrases:
        found = flat.find(" ".join(phrase.split()), position)
        if found < 0:
            return False
        position = found + 1
    return True


def card(tmp_path_factory, world: World) -> tuple[bytes, dict[str, bytes]]:
    import buildwikibin
    import libsearch

    out = tmp_path_factory.mktemp("voice")
    index = libsearch.build(TITLES, LEADS, {})
    libsearch.write_index(index, out / "W.IDX")
    libsearch.write_text(index, out / "W.DAT")
    libworld.resolve_topics(world, TITLES)
    builder = buildwikibin.build(index.num_docs, index_name="W.IDX",
                                 text_name="W.DAT", world=world)
    return builder.build(), {"W.IDX": (out / "W.IDX").read_bytes(),
                             "W.DAT": (out / "W.DAT").read_bytes()}


@pytest.fixture(scope="module")
def voice(tmp_path_factory):
    return card(tmp_path_factory, voice_world())


@pytest.fixture(scope="module")
def mystery(tmp_path_factory):
    return card(tmp_path_factory, worlds_mystery.mystery())


def visit(game_and_card, *commands: str,
          files: dict[str, bytes] | None = None) -> tuple[str, AgonHost]:
    game, files_on_card = game_and_card
    host = AgonHost(stdin=[*commands, "!"], files={**files_on_card, **(files or {})})
    return host.run(game, max_cycles=2_000_000_000), host


# --- the archive, doing as it is told ---------------------------------------------

#: One question a turn from the second turn on. `use` is turn 1.
SCRIPT = ("use", "pump", "pump", "order", "record", "order", "order", "hatch")


def test_a_seal_the_rules_open_reads_as_written_afterwards(voice):
    out, _ = visit(voice, *SCRIPT)
    assert order(out, "SEALED BY JUDICIAL.", "> pump",
                 "The cistern pump on Level 142 stopped")


def test_an_altered_record_reads_as_the_voice_has_it_and_then_as_written(voice):
    out, _ = visit(voice, *SCRIPT)
    assert order(out, "> order", AMENDED, "> order",
                 "A screen is fitted by two people")


def test_a_topic_with_no_seal_text_of_its_own_uses_the_worlds(voice):
    out, _ = visit(voice, *SCRIPT)
    assert order(out, "> record", "RECORD SEALED. THIS ACCESS HAS BEEN LOGGED.")


def test_a_seal_wins_over_an_alteration(voice):
    """Turn 6 does both to the order. A record that is closed cannot also
    be read wrong, so the last `order` gets the seal and not the amendment."""
    out, _ = visit(voice, *SCRIPT)
    flat = " ".join(out.split())
    assert flat.count(AMENDED) == 1
    assert flat.count("RECORD SEALED. THIS ACCESS HAS BEEN LOGGED.") == 2


def test_censor_text_with_sealed_false_starts_open(voice):
    out, _ = visit(voice, *SCRIPT)
    assert said(out, "The hatch has not been opened")
    assert "NOT FOR YOU" not in out


def test_a_sealed_or_altered_question_is_still_a_question(voice):
    """Marked asked, charged, and logged - the seal or the lie is what the
    player learned. Eight lines at the terminal, eight entries in the log."""
    _, host = visit(voice, *SCRIPT)
    assert len(host.files["SILO.LOG"]) == 2 * (len(SCRIPT) - 1)


def test_the_rules_can_read_what_the_archive_is_doing(voice):
    """`C_SEALED` holds on the opening pass, because the pump starts sealed;
    `C_ALTERED` holds on the pass after the alteration, before the player
    has asked for the order at all - a person could react to it first."""
    out, _ = visit(voice, *SCRIPT)
    assert out.index(CLOSED) < out.index("> ")
    assert order(out, "> pump", "> pump", CHANGED, "> order")


# --- the mystery's Voice ---------------------------------------------------------

#: Down to the IT office and sat at the screen.
AT_SCREEN = ("down", "down", "east", "use")


def test_the_pump_report_is_open_until_the_deputy_comes(mystery):
    before, _ = visit(mystery, *AT_SCREEN, "pump")
    assert said(before, "The cistern pump on Level 142 stopped")

    # Two for raising her in front of Jahns, three for the sealed record.
    after, _ = visit(mystery, "ask jahns about allison", *AT_SCREEN,
                     "allison", "pump")
    assert said(after, "boots on the stair")
    assert order(after, "> pump", "RECORD SEALED. THIS ACCESS HAS BEEN LOGGED.")
    assert not said(after, "The cistern pump on Level 142 stopped")


def test_the_standing_order_is_rewritten_once_the_clues_meet(mystery):
    honest, _ = visit(mystery, *AT_SCREEN, "order")
    assert said(honest, "A screen is fitted by two people")

    out, _ = visit(mystery, "down", "ask marnes about allison", "down", "east",
                   "ask walk about allison", "use", "order")
    assert said(out, "Somebody signed that off")
    assert said(out, "as amended, year 218")
    assert not said(out, "A screen is fitted by two people and never by one")


def test_what_the_voice_has_done_survives_a_save(mystery):
    _, host = visit(mystery, "down", "ask marnes about allison", "down", "east",
                    "ask walk about allison", "save")
    out, _ = visit(mystery, "restore", "use", "order", files=host.files)
    assert said(out, "as amended, year 218")


def test_the_mysterys_voice_costs_it_no_states(mystery):
    """Both actions ride on rules that already fired in distinct states, so
    the search is the size it was before the Voice could act."""
    assert len(worlds_mystery.mystery().explore().states) == 30_688


# --- the standalone world, and the model --------------------------------------------


def test_the_standalone_binary_moves_the_same_bytes():
    """No terminal to show a seal, but the rules run and read the arrays."""
    world = voice_world()
    world.terminal = None
    out = AgonHost(stdin=["look", "look", "look", "quit"], files={}).run(
        buildif.build(world).build(), max_cycles=50_000_000)
    assert order(out, CLOSED, "> look", "> look", CHANGED)


def test_explore_starts_from_what_is_authored_and_reaches_what_is_ruled():
    world = voice_world()
    search = world.explore()
    assert search.states[0].sealed == (1, 0, 0, 0)
    assert search.states[0].altered == (0, 0, 0, 0)
    world.goal = [(libworld.C_SEALED, T_RECORD), (libworld.C_ALTERED, T_ORDER)]
    assert world.explore().solve() == ["look", "look", "look", "look"]


def test_altering_a_topic_with_nothing_to_serve_is_refused():
    world = voice_world()
    world.rules.append(Rule(when=[(libworld.C_TURN, 7)],
                            then=[(libworld.A_ALTER, T_PUMP, 0)]))
    with pytest.raises(ValueError, match="no altered text"):
        world.check()


def test_an_empty_seal_or_alteration_is_refused():
    world = voice_world()
    world.topics[T_ORDER].alter = "  "
    with pytest.raises(ValueError, match="empty alter"):
        world.check()
    world = voice_world()
    world.seal = ""
    with pytest.raises(ValueError, match="seal text is empty"):
        world.check()


def test_the_overlay_carries_two_bytes_a_topic():
    world = voice_world()
    builder = buildif.build(world)
    start, length = buildif.overlay_at(builder, world)
    assert length == world.overlay_bytes
    assert start <= builder.labels["SEALED"] < builder.labels["ALTERED"] < start + length
