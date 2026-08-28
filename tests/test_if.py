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


def said(out: str, phrase: str) -> bool:
    """Whether the game said this, ignoring where `PRWRAP` broke the lines.

    A sentence longer than 76 columns arrives with a newline somewhere in the
    middle of it, and which word that lands on is the wrapper's business
    rather than the assertion's.
    """
    return " ".join(phrase.split()) in " ".join(out.split())


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


# --- rules, which are the step past a path ------------------------------------
#
# `data/silo/README.md` names four shapes a graph path cannot express:
# aggregation, ranking, set intersection, and a count around a ring. A flat
# list of ANDed conditions closes three of them and not the fourth, and that
# is the finding rather than a gap to be apologised for - see IF.md.


def test_a_count_over_a_set_fires(game):
    """Aggregation. `CARRYING 2` is a question about a set rather than about
    any one thing, which is the first of the four."""
    out, _ = play(game, "down", "take ledger", "down", "take badge")
    assert said(out, "Your hands are full.")


def test_a_count_does_not_fire_below_its_threshold(game):
    out, _ = play(game, "down", "take ledger")
    assert not said(out, "Your hands are full.")


def test_a_conjunction_of_two_particular_things_fires(game):
    """Intersection. Not two things - *these* two, which no single path
    reaches and no single condition states."""
    out, _ = play(game, "down", "down", "take badge", "down", "down",
                  "take wrench")
    assert said(out, "look like a story you would rather not have to tell")


def test_a_conjunction_needs_both(game):
    out, _ = play(game, "down", "down", "take badge")
    assert not said(out, "tell a deputy")


def test_a_flag_set_in_one_room_is_read_in_another(game):
    """State that outlives the turn that made it, which is what a flag is for
    and what a stateless card cannot have at all."""
    out, _ = play(game, "down", "down", "down", "down", "up", "up", "up", "up")
    assert said(out, "you know what makes it")


def test_the_flag_rule_does_not_fire_before_its_flag(game):
    out, _ = play(game, "down", "up")
    assert not said(out, "you know what makes it")


def test_a_once_rule_fires_once(game):
    """Most rules are events. One that printed every turn would be a bug that
    reads as a design decision."""
    out, _ = play(game, "down", "take ledger", "down", "take badge", "look",
                  "look")
    assert out.count("Your hands are full.") == 1


def test_rules_cost_no_card_reads_either(game):
    _out, host = play(game, "down", "take ledger", "down", "take badge")
    assert host.io_bytes == 0


# --- rules, before anything is emitted ----------------------------------------


def rules_world(**kwargs) -> World:
    return World(rooms=[Room("A", "a", {"NORTH": 1}), Room("B", "b")],
                 things=[Thing("key", "k", 0)], messages=["hello"], **kwargs)


def test_a_rule_with_no_conditions_is_refused():
    """It fires on the first turn and every turn, which is never intended."""
    world = rules_world(rules=[libworld.Rule(when=[], then=[])])
    with pytest.raises(ValueError, match="no conditions"):
        world.check()


def test_a_rule_naming_a_room_that_does_not_exist_is_refused():
    world = rules_world(rules=[libworld.Rule(when=[(libworld.C_AT, 9)],
                                             then=[])])
    with pytest.raises(ValueError, match="AT 9"):
        world.check()


def test_a_rule_naming_a_thing_that_does_not_exist_is_refused():
    world = rules_world(rules=[libworld.Rule(when=[(libworld.C_HAVE, 4)],
                                             then=[])])
    with pytest.raises(ValueError, match="HAVE 4"):
        world.check()


def test_a_rule_printing_a_message_that_does_not_exist_is_refused():
    world = rules_world(rules=[
        libworld.Rule(when=[(libworld.C_AT, 0)],
                      then=[(libworld.A_PRINT, 7, 0)])])
    with pytest.raises(ValueError, match="PRINT 7"):
        world.check()


def test_a_rule_counting_past_what_exists_is_refused():
    """`CARRYING 5` in a world of one thing never fires, and a rule that can
    never fire is a typo rather than a decision."""
    world = rules_world(rules=[
        libworld.Rule(when=[(libworld.C_CARRYING, 5)], then=[])])
    with pytest.raises(ValueError, match="only 1 things"):
        world.check()


def test_a_rule_setting_a_flag_the_world_does_not_reserve_is_refused():
    world = rules_world(flags=4, rules=[
        libworld.Rule(when=[(libworld.C_AT, 0)],
                      then=[(libworld.A_SET, 9, 0)])])
    with pytest.raises(ValueError, match="SET 9"):
        world.check()


def test_an_unknown_opcode_is_refused():
    world = rules_world(rules=[libworld.Rule(when=[(99, 0)], then=[])])
    with pytest.raises(ValueError, match="no condition 99"):
        world.check()


# --- the key behind the door it opens -----------------------------------------
#
# Every check above is about an argument that indexes nothing. These are about
# arguments that all index something and still cannot happen, which is the bug
# an author finds ten minutes in rather than at build time - the game runs, and
# the ending is simply never reached.


def test_every_rule_in_the_shipped_world_can_fire():
    """Dead code with a story attached is still dead code."""
    assert worlds.silo().dead_rules() == []


def test_a_room_behind_no_exit_makes_its_rule_dead():
    world = World(rooms=[Room("A", "a"), Room("B", "b")],
                  things=[Thing("key", "k", 0)], messages=["hello"],
                  rules=[libworld.Rule(when=[(libworld.C_AT, 1)],
                                       then=[(libworld.A_PRINT, 0, 0)])])
    assert world.dead_rules() == [(0, "room 1 ('B') cannot be reached")]
    with pytest.raises(ValueError, match="can never fire"):
        world.check()


def test_a_thing_in_a_room_nobody_reaches_cannot_be_carried():
    """The locked-key bug in the only shape it can be seen in.

    Room B is where the key is and room B is behind the rule the key opens, so
    the rule is unreachable *through itself*. Nothing about the emitted binary
    says so: the tables are consistent and every index is in range.
    """
    world = World(rooms=[Room("A", "a"), Room("B", "b", {"SOUTH": 0})],
                  things=[Thing("key", "k", 1)], messages=["hello"],
                  rules=[libworld.Rule(when=[(libworld.C_HAVE, 0)],
                                       then=[(libworld.A_PRINT, 0, 0)])])
    assert world.dead_rules() == [(0, "thing 0 ('key') cannot be picked up")]


def test_scenery_can_be_stood_beside_but_never_held():
    """A door is a thing and `HAVE` a door is a mistake, not a puzzle."""
    world = World(rooms=[Room("A", "a")], messages=["hello"],
                  things=[Thing("door", "d", 0, portable=False)],
                  rules=[libworld.Rule(when=[(libworld.C_HAVE, 0)],
                                       then=[(libworld.A_PRINT, 0, 0)])])
    assert world.dead_rules() == [(0, "thing 0 ('door') is not portable")]
    assert 0 in world.reach().present


def test_a_flag_nothing_sets_makes_its_reader_dead():
    world = rules_world(rules=[
        libworld.Rule(when=[(libworld.C_AT, 0), (libworld.C_FLAG, 3)],
                      then=[(libworld.A_PRINT, 0, 0)])])
    assert world.dead_rules() == [(0, "flag 3 is never set")]


def test_a_rule_reached_only_through_another_rule_is_live():
    """The fixpoint has to iterate, and this is what it is iterating for.

    Rule 1 needs a flag only rule 0 sets. A single pass over the rules in
    order would find it live and the reverse order would find it dead, so the
    answer must not depend on the order they are written in.
    """
    chained = [
        libworld.Rule(when=[(libworld.C_FLAG, 0)],
                      then=[(libworld.A_PRINT, 0, 0)]),
        libworld.Rule(when=[(libworld.C_AT, 0)],
                      then=[(libworld.A_SET, 0, 0)]),
    ]
    assert rules_world(rules=chained).dead_rules() == []
    assert rules_world(rules=chained[::-1]).dead_rules() == []


def test_a_rule_that_teleports_opens_the_rooms_behind_it():
    """`A_GOTO` is an exit that no room's table lists, and it counts as one."""
    rooms = [Room("A", "a"), Room("B", "b"), Room("C", "c", {"NORTH": 1})]
    world = World(rooms=rooms, things=[Thing("key", "k", 0)],
                  messages=["hello"], rules=[
        libworld.Rule(when=[(libworld.C_AT, 0)],
                      then=[(libworld.A_GOTO, 2, 0)]),
        libworld.Rule(when=[(libworld.C_AT, 1)],
                      then=[(libworld.A_PRINT, 0, 0)])])
    assert world.reach().rooms == frozenset({0, 1, 2})
    assert world.dead_rules() == []


def test_a_rule_that_moves_a_thing_puts_it_within_reach():
    """`A_MOVE` is the other way a thing arrives somewhere a player can be."""
    world = World(rooms=[Room("A", "a"), Room("B", "b")],
                  things=[Thing("key", "k", 1)], messages=["hello"], rules=[
        libworld.Rule(when=[(libworld.C_AT, 0)],
                      then=[(libworld.A_MOVE, 0, 0)]),
        libworld.Rule(when=[(libworld.C_HAVE, 0)],
                      then=[(libworld.A_PRINT, 0, 0)])])
    assert world.dead_rules() == []


def test_a_clear_never_makes_a_rule_dead():
    """The analysis errs upward on purpose, and this is where it shows.

    Rule 1 clears the flag rule 2 reads, and in a real play the order they
    fire in decides whether rule 2 ever sees it set. `reach` ignores `A_CLEAR`
    rather than guessing, so rule 2 is reported live. A report of *dead* here
    would be a false alarm, and an author who has seen one false alarm stops
    reading the report.
    """
    world = rules_world(flags=8, rules=[
        libworld.Rule(when=[(libworld.C_AT, 0)],
                      then=[(libworld.A_SET, 0, 0)]),
        libworld.Rule(when=[(libworld.C_AT, 1)],
                      then=[(libworld.A_CLEAR, 0, 0)]),
        libworld.Rule(when=[(libworld.C_FLAG, 0)],
                      then=[(libworld.A_PRINT, 0, 0)])])
    assert world.dead_rules() == []


def test_a_rule_needing_two_rooms_at_once_is_refused():
    """Exact rather than approximate: one byte cannot equal two values."""
    world = rules_world(rules=[
        libworld.Rule(when=[(libworld.C_AT, 0), (libworld.C_AT, 1)],
                      then=[])])
    with pytest.raises(ValueError, match=r"rooms \[0, 1\] at once"):
        world.check()


def test_a_rule_needing_a_flag_set_and_clear_is_refused():
    world = rules_world(flags=8, rules=[
        libworld.Rule(when=[(libworld.C_FLAG, 2), (libworld.C_NFLAG, 2)],
                      then=[])])
    with pytest.raises(ValueError, match="flag 2 both set and clear"):
        world.check()


def test_a_rule_needing_a_thing_carried_and_in_the_room_is_refused():
    """`where[thing]` is `CARRIED` or a room, and the two tests are exclusive.

    This one reads as a plausible sentence - "you are holding the key and the
    key is here" - which is exactly why it wants naming rather than a search.
    """
    world = rules_world(rules=[
        libworld.Rule(when=[(libworld.C_HAVE, 0), (libworld.C_HERE, 0)],
                      then=[])])
    with pytest.raises(ValueError, match="carried and in the room"):
        world.check()


def test_a_rule_counting_past_what_can_be_picked_up_is_refused():
    """Tighter than the count against `len(things)`: scenery does not count."""
    world = World(rooms=[Room("A", "a")], messages=["hello"], things=[
        Thing("key", "k", 0), Thing("door", "d", 0, portable=False)],
        rules=[libworld.Rule(when=[(libworld.C_CARRYING, 2)], then=[])])
    with pytest.raises(ValueError, match="only 1 of 2 things"):
        world.check()


def test_an_nflag_never_makes_a_rule_dead():
    """Every flag is clear on turn one, so `NFLAG` holds for any argument."""
    world = rules_world(flags=8, rules=[
        libworld.Rule(when=[(libworld.C_NFLAG, 7)], then=[])])
    assert world.dead_rules() == []


# --- one binary, two parsers --------------------------------------------------
#
# The last item of #62's second scope: wire the card in as a terminal found in
# the world, and check the two input paths can coexist. They do, and what that
# turned out to mean is narrower and more interesting than "both fit" - see
# `test_the_two_programs_define_no_label_twice`.


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """The oracle program carrying a world, over a tiny two-article card."""
    import buildwikibin
    import libsearch

    out = tmp_path_factory.mktemp("merged")
    titles = ["Pump Failure", "Filler"]
    leads = ["The cistern pump on Level 142 stopped without warning.", "x"]
    index = libsearch.build(titles, leads, {})
    libsearch.write_index(index, out / "W.IDX")
    libsearch.write_text(index, out / "W.DAT")

    world = worlds.silo()
    world.terminal = 3                       # the IT office
    builder = buildwikibin.build(index.num_docs, index_name="W.IDX",
                                 text_name="W.DAT", world=world)
    return builder.build(), {
        "W.IDX": (out / "W.IDX").read_bytes(),
        "W.DAT": (out / "W.DAT").read_bytes()}


def visit(merged, *commands: str) -> str:
    game, files = merged
    host = AgonHost(stdin=[*commands, "!"], files=files)
    return host.run(game, max_cycles=2_000_000_000)


def test_the_two_programs_define_no_label_twice():
    """The hazard, and it is silent. `Z80Builder.label` assigns into a dict,
    so a name defined twice resolves to whichever was emitted last and nothing
    says so. Both programs had a `PROMPT` and a `BANNER`, and the merged build
    printed the oracle's prompt while the player was walking about."""
    import collections

    import buildwikibin
    import libez80

    seen: collections.Counter[str] = collections.Counter()
    original = libez80.EZ80Builder.label

    def spy(self, name: str) -> None:
        seen[name] += 1
        original(self, name)

    libez80.EZ80Builder.label = spy
    try:
        world = worlds.silo()
        world.terminal = 3
        buildwikibin.build(600, world=world)
    finally:
        libez80.EZ80Builder.label = original

    assert [name for name, n in seen.items() if n > 1] == []


def test_walking_uses_the_word_table(merged):
    out = visit(merged, "down", "down", "east")
    assert "IT, Level 34" in out


def test_the_terminal_is_only_in_the_room_it_stands_in(merged):
    out = visit(merged, "use")
    assert said(out, "There is no terminal here.")


def test_sitting_down_switches_which_parser_listens(merged):
    out = visit(merged, "down", "down", "east", "use")
    assert said(out, "The screen wakes.")
    assert "archive>" in out


def test_the_card_answers_at_the_terminal(merged):
    """The other input path, on the same `INPBUF` the word table just read."""
    out = visit(merged, "down", "down", "east", "use", "pump")
    assert "Pump Failure" in out


def test_leaving_gives_the_world_back(merged):
    out = visit(merged, "down", "down", "east", "use", "leave", "take screen")
    assert said(out, "That is not something you can carry.")


def test_walking_still_reads_nothing_from_the_card(merged):
    """The card is open the whole time. A move must still not touch it."""
    game, files = merged
    host = AgonHost(stdin=["down", "down", "up", "up", "!"], files=files)
    host.run(game, max_cycles=2_000_000_000)
    before = host.io_bytes

    host2 = AgonHost(stdin=["!"], files=files)
    host2.run(game, max_cycles=2_000_000_000)
    assert before == host2.io_bytes          # four moves cost nothing extra


def test_a_world_costs_the_oracle_what_it_says(merged):
    import buildwikibin

    world = worlds.silo()
    world.terminal = 3
    plain = len(buildwikibin.build(600).code)
    carried = len(buildwikibin.build(600, world=world).code)
    assert carried - plain < 5_000           # the world is the small half


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


def test_the_overlay_is_a_byte_apiece():
    """Flags were bits first. Bits are eight times smaller and want a shift
    and a mask at four call sites, and a world binary has half a megabyte of
    SRAM spare - so the sixty bytes are not worth the four places to be
    wrong."""
    world = World(rooms=[Room("A", "a")],
                  things=[Thing(f"t{i}", "x", 0) for i in range(10)],
                  flags=64)
    assert world.overlay_bytes == 1 + 10 + 64 + 1


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
