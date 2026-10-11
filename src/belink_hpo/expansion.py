"""Generate a standard HPO-style name for a mention with an LLM.

The standard name is used as relevance feedback during candidate retrieval (see
``belink_hpo.retrieval``). Requires: pip install "belink-hpo[expand]" and the
OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import json
import time

DEFAULT_MODEL = "gpt-5.4"


def build_prompt(label: str, sentence: str) -> str:
    return f"""You are an expert biomedical terminologist.

Given the context "{sentence}", specify the standard concept name from the Human Phenotype Ontology (HPO) for the concept {label}.
Return the output as the following JSON object: {{ "name": "<a standard community-recognized concept name>" }}.
"""


def generate_standard_name(label, sentence, client=None, model=DEFAULT_MODEL, max_retries=3, retry_delay=2.0):
    """Return the standard name for one mention, or None if every attempt failed."""
    from openai import APIError, OpenAI, RateLimitError

    client = client or OpenAI()  # reads OPENAI_API_KEY from the environment
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(label, sentence)}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content.strip()).get("name")
        except RateLimitError:
            time.sleep(retry_delay * attempt)
        except APIError as err:
            print(f"API error on attempt {attempt}/{max_retries}: {err}")
            time.sleep(retry_delay)
    print(f"Failed to generate a standard name for {label!r} after {max_retries} attempts.")
    return None


def add_standard_names(mentions: dict, model=DEFAULT_MODEL) -> dict:
    """Fill ``standard_name`` for every record that lacks one. Records need "label" and "sentence"."""
    from openai import OpenAI

    client = OpenAI()
    todo = [info for info in mentions.values() if "standard_name" not in info]
    for i, info in enumerate(todo, start=1):
        info["standard_name"] = generate_standard_name(info["label"], info["sentence"], client, model)
        print(f"[{i}/{len(todo)}] {info['label']!r} -> {info['standard_name']!r}")
    return mentions
