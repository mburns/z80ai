#!/usr/bin/env python3
"""
Wordings from strangers, for a phrasebook written by one hand.

    python tools/wordings.py --backend fake                     # no network
    python tools/wordings.py --backend claude --paths father_is shift_is
    python tools/wordings.py --backend ollama --model gemma2:9b -n 4

The phrasing curve is the largest lever in the repository: 9, 21 and 33
wordings per path bought 55.4%, 65.3% and 68.5% held out, and it is still
climbing. `data/silo/README.md` records the caveat every time - 240 sentences
by one hand over two afternoons resemble each other more than a stranger's
would - and `tools/phrasebook_diversity.py` found the third dozen written in
one register, so that the frame carried no class information at all.

So the fourth dozen should not come from the hand that wrote the first three.
This asks a model for wordings *in character* - a child, a clerk, somebody
angry, somebody terse - because a register is exactly what one author cannot
vary on purpose, and writes a **review file** rather than editing
`relationpaths.py`. Nothing here ships without a person reading it.

## What the review file says

For every candidate, the path it was asked for, the persona that produced it,
and its novelty against everything already shipped - one minus the cosine to
the nearest existing wording in the encoder's own 256 buckets, which is the
measure `phrasebook_diversity.py` uses and the one the model can actually see.
Candidates are sorted with the most novel first, so the top of each block is
what a stranger brought and the bottom is padding.

Novelty is a filter for repetition and **not a prediction of yield**:
`data/silo/README.md` measured the diversity check against the third dozen
and it did not predict the gain. Read the sentences.

## What is refused

A candidate that does not name the subject exactly once as `{s}`, or that
duplicates a shipped wording or another candidate once lowercased, is dropped
before it reaches the file - a duplicate is worth nothing to the curve, and
`relationpaths.build` fills `{s}` from the corpus so a wording without it has
no subject at all.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "silo"))

import phrasebook_diversity

#: Who asks. Each is a register, which is the thing one author cannot vary on
#: purpose and the thing the third dozen turned out to be short of.
PERSONAS: tuple[str, ...] = (
    "a child of eight who has just learned the archive answers questions",
    "a Supply clerk at a filing terminal, terse and procedural",
    "somebody angry who has been told the record is sealed",
    "an old porter who speaks in short plain words and never uses a title",
    "a gossip in the cafeteria who asks sideways",
    "a Judicial officer dictating a formal request",
    "a nervous person asking as if they might be overheard",
    "a Mechanical hand who talks about people the way they talk about pumps",
)

#: The name `{s}` stands in for when measuring novelty, as in
#: `phrasebook_diversity.report`. Any name would do; it has to be the same one
#: on both sides of the comparison.
SUBJECT = "amanda m wilson"

#: A backend takes a prompt and returns the model's text, or an empty string.
Backend = Callable[[str], str]


@dataclass(frozen=True)
class Candidate:
    path: str
    wording: str
    persona: str
    #: 1 - cosine to the nearest shipped wording of *any* path, in the
    #: encoder's buckets. Near zero is a wording the model has already seen.
    novelty: float
    #: The shipped wording it is nearest to, so a reviewer sees the twin.
    nearest: str


def prompt_for(path: str, shipped: tuple[str, ...], persona: str, n: int) -> str:
    """One request: the question shape, what exists, who is asking."""
    examples = "\n".join(f"  {w}" for w in shipped[:6])
    return (
        "These are ways of asking one question about a person who lives in a "
        "sealed underground silo. `{s}` stands for the person's name.\n\n"
        f"{examples}\n\n"
        f"You are {persona}. Write {n} more ways of asking exactly the same "
        "question, in your own voice. Use `{s}` exactly once in each. One per "
        "line. No numbering, no quotation marks, no commentary. Do not repeat "
        "any wording above and do not change what is being asked."
    )


def parse(text: str) -> list[str]:
    """Lines back into wordings, stripping what models add anyway."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        line = line.strip("\"'`").strip()
        if line:
            out.append(line.lower())
    return out


def _fill(wording: str) -> str:
    return wording.replace("{s}", SUBJECT)


def score(path: str, persona: str, wordings: list[str],
          shipped: dict[str, tuple[str, ...]]) -> list[Candidate]:
    """Novelty against every shipped wording of every path.

    Every path rather than this one, because a candidate that lands on a
    neighbouring class's region is the misrouting `data/silo/README.md`
    measures, and a reviewer should see which class it is near.
    """
    pool = [(p, w) for p, ws in shipped.items() for w in ws]
    filled = [_fill(w) for _, w in pool]
    out = []
    for wording in wordings:
        target = _fill(wording)
        sims = [phrasebook_diversity.similarity(target, other) for other in filled]
        best = max(range(len(sims)), key=sims.__getitem__) if sims else -1
        nearest = f"{pool[best][0]}: {pool[best][1]}" if best >= 0 else ""
        out.append(Candidate(path, wording, persona,
                             1.0 - (sims[best] if best >= 0 else 0.0), nearest))
    return out


def keep(wording: str, seen: set[str]) -> bool:
    """Exactly one subject, and not a wording anybody has already written."""
    if wording.count("{s}") != 1:
        return False
    if wording in seen:
        return False
    seen.add(wording)
    return True


def generate(paths: dict[str, tuple[str, ...]], backend: Backend,
             per_persona: int, personas: tuple[str, ...] = PERSONAS,
             shipped: dict[str, tuple[str, ...]] | None = None
             ) -> dict[str, list[Candidate]]:
    """Ask every persona about every path; return candidates, most novel first.

    `shipped` is what novelty is measured against and what duplicates are
    refused against; it defaults to `paths`. The two differ when the caller
    wants candidates for a few paths measured against the whole phrasebook.
    """
    shipped = paths if shipped is None else shipped
    seen: set[str] = {w.lower() for ws in shipped.values() for w in ws}
    result: dict[str, list[Candidate]] = {}
    for path, existing in paths.items():
        found: list[Candidate] = []
        for persona in personas:
            text = backend(prompt_for(path, existing, persona, per_persona))
            fresh = [w for w in parse(text) if keep(w, seen)]
            found.extend(score(path, persona, fresh, shipped))
        result[path] = sorted(found, key=lambda c: -c.novelty)
    return result


def write_review(candidates: dict[str, list[Candidate]], out: Path) -> None:
    """The file a person reads. `{s}` is kept so a line can be pasted."""
    lines = ["# Candidate wordings, most novel first. novelty = 1 - cosine to",
             "# the nearest shipped wording of any path, in the encoder's",
             "# buckets. Read the sentences; the number only flags padding.",
             ""]
    for path, found in candidates.items():
        lines.append(f"# {path}  ({len(found)} candidates)")
        for c in found:
            lines.append(f"{c.novelty:.3f}  {c.wording}")
            lines.append(f"        [{c.persona.split(',')[0]}]  near {c.nearest}")
        lines.append("")
    out.write_text("\n".join(lines))


# --- backends --------------------------------------------------------------------


def fake_backend(prompt: str) -> str:
    """Deterministic and offline, so the pipeline can be tested without a
    model. It rewrites the examples in the prompt rather than inventing, so
    what it returns is mostly *not* novel - which is what the filter and the
    sort are for."""
    examples = [line.strip() for line in prompt.splitlines()
                if line.startswith("  ")]
    persona = prompt.split("You are ", 1)[1].split(".", 1)[0]
    word = persona.split()[1] if len(persona.split()) > 1 else "one"
    lines = [f"{word} asks {ex}" for ex in examples[:3]]
    lines.append(examples[0] if examples else "")           # a duplicate
    lines.append("who is the father of nobody")             # no subject
    lines.append("3. " + (examples[1] if len(examples) > 1 else "{s}"))
    return "\n".join(lines)


def ollama_backend(model: str) -> Backend:
    """Local Ollama, the way `examples/guess/gendata.py` calls it."""
    def call(prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.9}}
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return str(json.loads(resp.read().decode()).get("response", ""))
        except OSError as e:
            print(f"# ollama: {e}", file=sys.stderr)
            return ""
    return call


def claude_backend(model: str) -> Backend:
    """The Claude API through the official SDK. Reads `ANTHROPIC_API_KEY`
    or an `ant auth login` profile; a refusal is treated as no wordings."""
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit("pip install anthropic") from e
    client = anthropic.Anthropic()

    def call(prompt: str) -> str:
        try:
            with client.messages.stream(
                    model=model, max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]) as stream:
                message = stream.get_final_message()
        except anthropic.APIError as e:
            print(f"# claude: {e}", file=sys.stderr)
            return ""
        if message.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in message.content if b.type == "text")
    return call


def shipped_phrasebook() -> dict[str, tuple[str, ...]]:
    """`PATHS` with `EXTRA` and `EXTRA_THIRD` folded in - everything the
    classifier has been trained on, which is what novelty is against."""
    import relationpaths

    merged = {path: tuple(ws) for path, ws in relationpaths.PATHS.items()}
    for table in (relationpaths.EXTRA, relationpaths.EXTRA_THIRD):
        for path, more in table.items():
            merged[path] = (*merged.get(path, ()), *more)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("fake", "ollama", "claude"),
                    default="fake")
    ap.add_argument("--model", default=None,
                    help="claude-opus-5 for claude, gemma2:9b for ollama")
    ap.add_argument("--paths", nargs="*", default=None,
                    help="which paths to ask about; default all but refuse")
    ap.add_argument("-n", "--per-persona", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("wordings-review.txt"))
    args = ap.parse_args()

    shipped = shipped_phrasebook()
    wanted = args.paths or [p for p in shipped if p != "refuse"]
    missing = [p for p in wanted if p not in shipped]
    if missing:
        raise SystemExit(f"not a path in the phrasebook: {', '.join(missing)}")
    paths = {p: shipped[p] for p in wanted}

    backend: Backend
    if args.backend == "claude":
        backend = claude_backend(args.model or "claude-opus-5")
    elif args.backend == "ollama":
        backend = ollama_backend(args.model or "gemma2:9b")
    else:
        backend = fake_backend

    found = generate(paths, backend, args.per_persona, shipped=shipped)
    write_review(found, args.out)
    total = sum(len(v) for v in found.values())
    print(f"{total} candidates over {len(found)} paths -> {args.out}")


if __name__ == "__main__":
    main()
