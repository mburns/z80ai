"""The eZ80 phrasebook build, against libinfer and against itself.

Same discipline as test_ez80_kernels.py: the strongest signal is two
independently generated programs agreeing byte for byte, and that needs no
reference model at all. Here it is two kernels producing the same RESULT for
the same query, plus the printed text matching libinfer.classify.

The reply text is never in the binary - it comes off the SD card - so these
also check the one thing that can go wrong with an offset table: printing the
right index out of the wrong file.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import buildez80
import libinfer
from libhost import AgonHost, run_agon

KERNELS = ["compact", "row", "column"]

PHRASES = [
    "I FROZE YOUR ACCOUNT",
    "IT IS SIXTY DEGREES AND CLEAR",
    "WHY DID THE CHICKEN CROSS THE ROAD",
    "I DO NOT KNOW THAT ONE",
]


@pytest.fixture(scope="module")
def phrasebook_path(tmp_path_factory):
    """A small phrasebook model: 128 query buckets in, one phrase out."""
    rng = np.random.default_rng(11)
    sizes = [libinfer.NUM_BUCKETS, 24, len(PHRASES)]
    weights, biases = [], []
    for a, b in itertools.pairwise(sizes):
        weights.append(rng.choice([-2, -1, 0, 1], size=(b, a),
                                  p=[0.05, 0.15, 0.65, 0.15]).astype(np.int32))
        biases.append(rng.integers(-40, 40, size=b).astype(np.int32))
    model = libinfer.Model(weights=weights, biases=biases, charset="\x00",
                           phrases=PHRASES, accum_bits=24)
    path = tmp_path_factory.mktemp("phrasebook") / "model.npz"
    model.save_npz(str(path))
    return str(path)


@pytest.fixture(scope="module")
def reference(phrasebook_path):
    return libinfer.Model.load(phrasebook_path)


def build(path: str, kernel: str) -> buildez80.EZ80Builder:
    return buildez80.build_autoreg(path, kernel=kernel)


QUERIES = ["FREEZE MY ACCOUNT", "WHAT IS THE WEATHER", "TELL ME A JOKE",
           "WHAT IS A QUARK", "HELLO"]


@pytest.mark.parametrize("kernel", KERNELS)
def test_the_printed_reply_is_the_one_the_reference_picks(kernel, phrasebook_path,
                                                          reference):
    builder = build(phrasebook_path, kernel)
    out, _ = run_agon(builder.build(), stdin=[*QUERIES, "!"],
                      files={"PHRASES.DAT": builder.phrase_blob})
    for query in QUERIES:
        assert libinfer.classify(reference, query, accum_bits=24) in out


@pytest.mark.parametrize("kernel", KERNELS)
def test_result_holds_the_index_the_reference_argmaxes_to(kernel, phrasebook_path,
                                                          reference):
    """Checked as an index, not as text: with four phrases two different logit
    vectors often argmax to the same one, and comparing strings would pass
    over a broken kernel."""
    builder = build(phrasebook_path, kernel)
    for query in QUERIES:
        host = AgonHost(stdin=[query, "!"],
                        files={"PHRASES.DAT": builder.phrase_blob})
        cpu = host.cpu
        cpu.load(host.LOAD_ADDR, builder.build())
        cpu.pc = host.LOAD_ADDR
        cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["PRINT_PHRASE"])
        assert cpu.pc == builder.labels["PRINT_PHRASE"]
        got = cpu.peek_word(builder.labels["RESULT"], 3)
        assert got == libinfer.classify_index(reference, query, accum_bits=24)


def test_all_kernels_agree_without_a_reference_model(phrasebook_path):
    builders = {k: build(phrasebook_path, k) for k in KERNELS}
    outs = {}
    for kernel, builder in builders.items():
        outs[kernel], _ = run_agon(builder.build(), stdin=[*QUERIES, "!"],
                                   files={"PHRASES.DAT": builder.phrase_blob})
    assert outs["compact"] == outs["row"] == outs["column"]


def test_every_kernel_encodes_the_same_phrase_file(phrasebook_path):
    blobs = {k: build(phrasebook_path, k).phrase_blob for k in KERNELS}
    assert blobs["compact"] == blobs["row"] == blobs["column"]


# --- the phrase file ----------------------------------------------------------


def test_the_offset_table_addresses_the_right_text():
    blob = buildez80.encode_phrases(PHRASES)
    count = int.from_bytes(blob[0:3], "little")
    assert count == len(PHRASES)
    for i, phrase in enumerate(PHRASES):
        offset = int.from_bytes(blob[3 + 3 * i:6 + 3 * i], "little")
        end = blob.index(b"\x00", offset)
        assert blob[offset:end].decode() == phrase


def test_the_file_holds_no_addresses_only_offsets():
    """A blob with no addresses in it cannot be loaded to the wrong one.

    Everything else in the image is absolute - libz80.resolve() patches the
    fixups and discards them, so a .bin carries no relocation information -
    which is exactly why the one thing that moves is offset-based.
    """
    blob = buildez80.encode_phrases(PHRASES)
    for i in range(len(PHRASES)):
        offset = int.from_bytes(blob[3 + 3 * i:6 + 3 * i], "little")
        assert offset < len(blob)


def test_a_missing_phrase_file_says_so_rather_than_printing_rubbish(phrasebook_path):
    """Without the guard the program would run on whatever the buffer held."""
    builder = build(phrasebook_path, "compact")
    out, _ = run_agon(builder.build(), stdin=["HELLO", "!"], files={})
    assert "Could not load PHRASES.DAT" in out
    assert not any(p in out for p in PHRASES)


def test_the_phrase_file_name_is_configurable(phrasebook_path):
    builder = buildez80.build_autoreg(phrasebook_path, kernel="compact",
                                      phrases_file="TALK.PHR")
    out, _ = run_agon(builder.build(), stdin=["HELLO", "!"],
                      files={"TALK.PHR": builder.phrase_blob})
    assert "Could not load" not in out
    assert builder.phrases_file == "TALK.PHR"


# --- the guards ---------------------------------------------------------------


def test_the_column_kernel_runs_a_phrasebook(phrasebook_path, reference):
    """One pass over one input: no PREQ to amortize, but skipping zero
    activations pays per pass. SCAN_IN lists the whole input vector, and the
    result must match the reference index for index."""
    builder = build(phrasebook_path, "column")
    assert "PREQ" not in builder.labels
    assert "QBASE" not in builder.labels
    for query in QUERIES:
        host = AgonHost(stdin=[query, "!"],
                        files={"PHRASES.DAT": builder.phrase_blob})
        cpu = host.cpu
        cpu.load(host.LOAD_ADDR, builder.build())
        cpu.pc = host.LOAD_ADDR
        cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["PRINT_PHRASE"])
        assert cpu.pc == builder.labels["PRINT_PHRASE"]
        got = cpu.peek_word(builder.labels["RESULT"], 3)
        assert got == libinfer.classify_index(reference, query, accum_bits=24)


def test_auto_considers_the_column_kernel_for_a_phrasebook(phrasebook_path):
    """This fixture is small enough that the fastest kernel also fits."""
    builder = buildez80.build_autoreg(phrasebook_path, kernel="auto")
    assert builder.kernel == "column"


def test_a_phrasebook_must_take_the_query_buckets_alone(tmp_path):
    rng = np.random.default_rng(5)
    model = libinfer.Model(
        weights=[rng.choice([-1, 0, 1], size=(4, 256)).astype(np.int32)],
        biases=[np.zeros(4, dtype=np.int32)],
        charset="\x00", phrases=PHRASES, accum_bits=24)
    path = str(tmp_path / "wide.npz")
    model.save_npz(path)
    with pytest.raises(ValueError, match="no context to encode"):
        buildez80.build_autoreg(path, kernel="compact")


def test_a_mismatched_output_layer_is_caught_at_build_time(tmp_path):
    """CHARTBL is sized by the charset and ARGMAX by the weight shapes, and
    nothing compared them: PRINTCH would index past the table and print
    whatever followed it."""
    rng = np.random.default_rng(5)
    model = libinfer.Model(
        weights=[rng.choice([-1, 0, 1], size=(9, 128)).astype(np.int32)],
        biases=[np.zeros(9, dtype=np.int32)],
        charset="\x00", phrases=PHRASES, accum_bits=24)
    path = str(tmp_path / "mismatch.npz")
    model.save_npz(path)
    with pytest.raises(ValueError, match="9 neurons but there are 4 phrases"):
        buildez80.build_autoreg(path, kernel="compact")


def test_the_character_builds_are_untouched(guess_model_path):
    """Every guard added for the phrasebook must be inert on a char decoder."""
    builder = buildez80.build_autoreg(guess_model_path, kernel="compact",
                                      max_output_len=1)
    assert builder.phrase_blob == b""
    assert builder.phrases_file is None


# --- how many buckets, which was 128 and never asked --------------------------
#
# `tools/bucket_sweep.py` swept it: on `data/silo/`'s phrasebook, 256 buckets
# is worth 7.5 points of held-out accuracy over 128, because 859 distinct
# trigrams into 128 buckets leaves 85% of them sharing one. It is flat past
# 256, and 256 is also the most the device can address - the tokenizer masks
# the hash's low byte and the index has to fit in one register.
#
# The count is a property of the model now, so what has to be pinned is that
# the card tokenizes at the width the model was trained at rather than at the
# width the module happens to default to.


def wide_phrasebook(tmp_path, buckets: int, hidden: int = 24) -> str:
    rng = np.random.default_rng(3)
    sizes = [buckets, hidden, len(PHRASES)]
    weights, biases = [], []
    for a, b in itertools.pairwise(sizes):
        weights.append(rng.choice([-2, -1, 0, 1], size=(b, a),
                                  p=[0.05, 0.15, 0.65, 0.15]).astype(np.int32))
        biases.append(rng.integers(-40, 40, size=b).astype(np.int32))
    model = libinfer.Model(weights=weights, biases=biases, charset="\x00",
                           phrases=PHRASES, accum_bits=24, num_buckets=buckets)
    path = str(tmp_path / f"wide{buckets}.npz")
    model.save_npz(path)
    return path


def test_the_bucket_count_survives_the_model_file(tmp_path):
    """A card built at one width and tokenized at another scores something
    meaningless rather than failing, so the width travels with the weights."""
    path = wide_phrasebook(tmp_path, 256)
    assert libinfer.Model.load(path).num_buckets == 256
    assert libinfer.load_for_build(path, report_io=False).num_buckets == 256


def test_a_model_at_the_default_width_writes_no_key(tmp_path):
    """Every model trained before this field existed was trained at 128, and
    its `.npz` should not grow a key saying so - which is also why none of the
    shipped artifacts moved when the field was added."""
    path = wide_phrasebook(tmp_path, libinfer.NUM_BUCKETS)
    assert "num_buckets" not in libinfer.Model.load(path).architecture()


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_wide_model_tokenizes_at_its_own_width(tmp_path, kernel):
    """The two implementations, at 256. `INBUF` is twice as long and every
    trigram has to land where the reference put it - a card that masked to 127
    would still run, and would answer from half the vector."""
    path = wide_phrasebook(tmp_path, 256)
    builder = buildez80.build_autoreg(path, kernel=kernel)
    query = "WHAT IS THE WEATHER"

    host = AgonHost(stdin=[query, "!"],
                    files={"PHRASES.DAT": builder.phrase_blob})
    cpu = host.cpu
    cpu.load(host.LOAD_ADDR, builder.build())
    cpu.pc = host.LOAD_ADDR
    cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["PRINT_PHRASE"])

    at = builder.labels["INBUF"]
    got = np.array([cpu.peek_word(at + 3 * i, 3) for i in range(256)])
    np.testing.assert_array_equal(got, libinfer.trigram_encode(query, 256))


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_wide_model_answers_what_the_reference_answers(tmp_path, kernel):
    path = wide_phrasebook(tmp_path, 256)
    builder = buildez80.build_autoreg(path, kernel=kernel)
    reference = libinfer.Model.load(path)
    for query in QUERIES:
        host = AgonHost(stdin=[query, "!"],
                        files={"PHRASES.DAT": builder.phrase_blob})
        cpu = host.cpu
        cpu.load(host.LOAD_ADDR, builder.build())
        cpu.pc = host.LOAD_ADDR
        cpu.run(max_cycles=400_000_000, stop_pc=builder.labels["PRINT_PHRASE"])
        assert cpu.peek_word(builder.labels["RESULT"], 3) == \
            libinfer.classify_index(reference, query, accum_bits=24)


def test_more_buckets_than_a_byte_can_index_is_refused(tmp_path):
    """512 would need nine bits of index, in the tokenizer and in the scan.
    The sweep says there is nothing there to want - 51.7% at 512 against
    52.5% at 256 - so the limit is stated rather than engineered around."""
    path = wide_phrasebook(tmp_path, 512)
    with pytest.raises(ValueError, match="256 is the most"):
        buildez80.build_autoreg(path, kernel="compact")
