"""The multiple-choice prompt the re-ranker was trained on.

Training and inference must build the prompt the same way, so everything that
shapes it lives here.
"""

from __future__ import annotations

import re
import string

LETTERS = string.ascii_uppercase
MAX_OPTIONS = 10  # candidates shown per prompt, not counting "None of the above."
NONE_OPTION = "None of the above."


def candidate_id(candidate) -> str:
    """Read the concept id from a candidate given as a dict ("id" or "cui") or an (id, name) pair."""
    if isinstance(candidate, dict):
        return candidate["id"] if "id" in candidate else candidate["cui"]
    return candidate[0]


def candidate_name(candidate) -> str:
    return candidate["name"] if isinstance(candidate, dict) else candidate[1]


def augment_name(name: str, preferred_name: str | None) -> str:
    """Attach the ontology's preferred name to a candidate name.

    "short toe"                  + "brachydactyly" -> "short toe (aka brachydactyly)"
    "short toe (brachydactyly)"  + "brachydactyly" -> "short toe (aka brachydactyly)"
    "brachydactyly"              + "brachydactyly" -> "brachydactyly"
    """
    pref = (preferred_name or "").lower()
    if not pref:
        return name
    target = f"({pref})"
    if target in name.lower():
        return re.sub(re.escape(target), lambda _: f"(aka {pref})", name, flags=re.I)
    if name.lower().strip() != pref:
        return f"{name} (aka {pref})"
    return name


def select_options(candidates, preferred_names=None, max_options: int = MAX_OPTIONS):
    """Turn ranked candidates into the options shown at inference time.

    Keeps the first retrieved name of each concept, in retrieval order, and caps the
    list at ``max_options``. Returns a list of (concept_id, display_name).
    """
    options, seen_ids, seen_names = [], set(), set()
    for cand in candidates:
        cid = candidate_id(cand)
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        name = candidate_name(cand)
        if preferred_names is not None:
            name = augment_name(name, preferred_names.get(cid))
        if name not in seen_names:
            seen_names.add(name)
            options.append((cid, name))
    return options[:max_options]


def format_options(option_names) -> tuple[str, str]:
    """Return the lettered option block and the letter used for "None of the above."."""
    if len(option_names) >= len(LETTERS):
        raise ValueError(f"At most {len(LETTERS) - 1} options are supported, got {len(option_names)}.")
    none_letter = LETTERS[len(option_names)]
    lines = [f"{LETTERS[i]}: {name}" for i, name in enumerate(option_names)]
    lines.append(f"{none_letter}: {NONE_OPTION}")
    return "\n".join(lines), none_letter


def build_user_message(context: str, mention: str, option_names) -> str:
    """The user turn: instruction, a blank line, then the options."""
    options, _ = format_options(option_names)
    instruction = (
        f"<Instruct>: Given the context '{context}', "
        f"select the correct biomedical concept corresponding "
        f"to '{mention}'. Answer using one of the provided options."
    ).strip()
    return f"{instruction}\n\n" + f"<Options>: {options}".strip()


def make_row(context: str, mention: str, option_names, correct_letter: str) -> dict:
    """One training/evaluation row in chat format."""
    return {
        "messages": [
            {"role": "user", "content": build_user_message(context, mention, option_names)},
            {"role": "assistant", "content": correct_letter},
        ]
    }
