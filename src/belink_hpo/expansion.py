"""Generate a standard HPO name for a mention with an LLM (query expansion).

The standard name is mixed into the retrieval query as relevance feedback (see
``belink_hpo.retrieval``), which is how the reported retrieval and re-ranking numbers
were obtained. Requires: pip install "belink-hpo[expand]" and the OPENAI_API_KEY
environment variable. Any OpenAI-compatible server works: set OPENAI_BASE_URL, or pass
``base_url``.

Python:

    namer = StandardNameGenerator()
    namer("mild up-slant", "EYES: mild epicanthus, mild up-slant.")   # "Upslanted palpebral fissure"

Command line, for a JSON file of mention records (the format `belink-hpo retrieve` reads):

    belink-hpo expand --input mentions.json --output_file mentions.json
"""

from __future__ import annotations

import argparse
import json
import time

DEFAULT_MODEL = "gpt-5.4"


def build_prompt(label: str, sentence: str) -> str:
    return f"""You are an expert biomedical terminologist.

Given the context "{sentence}", specify the standard concept name from the Human Phenotype Ontology (HPO) for the concept {label}.
Return the output as the following JSON object: {{ "name": "<a standard community-recognized concept name>" }}.
"""


class StandardNameGenerator:
    """Callable that returns the standard HPO name of a mention in its context, or None.

    Results are cached, so the same (mention, context) pair is only sent once.
    """

    def __init__(self, model=DEFAULT_MODEL, client=None, base_url=None, max_retries=3, retry_delay=2.0):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url) if base_url else OpenAI()  # reads OPENAI_API_KEY
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache = {}

    def _ask(self, mention, context):
        try:
            from openai import RateLimitError
        except ImportError:  # a custom client without the openai package
            RateLimitError = ()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": build_prompt(mention, context)}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                name = json.loads(response.choices[0].message.content.strip()).get("name")
                return name.strip() if isinstance(name, str) and name.strip() else None
            except RateLimitError:
                time.sleep(self.retry_delay * attempt)
            except Exception as err:  # API errors and unparsable answers
                print(f"Attempt {attempt}/{self.max_retries} for {mention!r} failed: {err}")
                time.sleep(self.retry_delay)
        print(f"Failed to generate a standard name for {mention!r} after {self.max_retries} attempts.")
        return None

    def __call__(self, mention, context):
        key = (mention, context)
        if key not in self._cache:
            self._cache[key] = self._ask(mention, context)
        return self._cache[key]

    def generate_batch(self, items, progress=False):
        """Standard names for a list of dicts with "mention" and "context", in order."""
        names = []
        for i, it in enumerate(items, start=1):
            names.append(self(it["mention"], it["context"]))
            if progress:
                print(f"[{i}/{len(items)}] {it['mention']!r} -> {names[-1]!r}")
        return names


def generate_standard_name(label, sentence, client=None, model=DEFAULT_MODEL, max_retries=3, retry_delay=2.0):
    """Return the standard name for one mention, or None if every attempt failed."""
    return StandardNameGenerator(model, client, max_retries=max_retries, retry_delay=retry_delay)(label, sentence)


def add_standard_names(mentions: dict, model=DEFAULT_MODEL, generator=None) -> dict:
    """Fill ``standard_name`` for every record that lacks one. Records need "label" and "sentence"."""
    generator = generator or StandardNameGenerator(model)
    todo = [info for info in mentions.values() if not info.get("standard_name")]
    for i, info in enumerate(todo, start=1):
        info["standard_name"] = generator(info["label"], info["sentence"])
        print(f"[{i}/{len(todo)}] {info['label']!r} -> {info['standard_name']!r}")
    return mentions


def expand_main(argv=None):
    p = argparse.ArgumentParser(
        prog="belink-hpo expand",
        description="Add an LLM-generated standard name to every record of a JSON file of mentions",
    )
    p.add_argument("--input", required=True,
                   help='JSON object of records, each with "label" (the mention) and "sentence" (its context)')
    p.add_argument("--output_file", required=True, help="Output JSON (may be the same file as --input)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="LLM used to generate the names")
    p.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint, e.g. a local vLLM server")
    args = p.parse_args(argv)

    with open(args.input, encoding="utf-8") as f:
        records = json.load(f)
    add_standard_names(records, generator=StandardNameGenerator(args.model, base_url=args.base_url))
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    missing = sum(not r.get("standard_name") for r in records.values())
    print(f"Wrote {args.output_file} ({missing} of {len(records)} records without a standard name)")
