"""``belink-hpo`` command line entry point."""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from . import DEFAULT_MODEL, __version__

COMMANDS = {
    "link": (None, "Link mentions in a JSONL file end to end: expansion, retrieval, re-ranking"),
    "rerank": (None, "Re-rank the candidates of mentions in a JSONL file and write the chosen HPO ids"),
    "predict": ("belink_hpo.predict:main", "Predict option letters for a JSONL file in the messages format"),
    "train": ("belink_hpo.train:main", "Fine-tune a LoRA adapter on a messages-format dataset"),
    "build-dataset": ("belink_hpo.dataset:main", "Turn examples with candidates into train/test JSONL"),
    "prepare-obo": ("belink_hpo.prepare_obo:main", "Turn an OBO ontology into the list of names to index"),
    "embed-terms": ("belink_hpo.retrieval:embed_terms_main", "Embed ontology names into a retrieval index"),
    "retrieve": ("belink_hpo.retrieval:retrieve_main", "Retrieve candidates for a JSON file of mentions"),
    "expand": ("belink_hpo.expansion:expand_main", "Generate standard names for a JSON file of mentions with an LLM"),
}


def rerank_main(argv=None):
    p = argparse.ArgumentParser(prog="belink-hpo rerank", description=COMMANDS["rerank"][1])
    p.add_argument("--input", required=True, help="JSONL with context, mention and candidates per line")
    p.add_argument("--out", default="linked.jsonl")
    p.add_argument("--adapter", default=DEFAULT_MODEL, help="Hub id or local adapter directory")
    p.add_argument("--preferred-names", default=None, help="hp.obo, or a JSON file mapping id -> preferred name")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--load-4bit", action="store_true", help="Load in 4-bit to save VRAM (slower)")
    args = p.parse_args(argv)

    from .dataset import load_jsonl
    from .reranker import HPOReranker

    preferred = None
    if args.preferred_names:
        from .ontology import load_preferred_names

        preferred = load_preferred_names(args.preferred_names)

    items = load_jsonl(args.input)
    reranker = HPOReranker.from_pretrained(args.adapter, load_in_4bit=args.load_4bit, preferred_names=preferred)
    results = reranker.rerank_batch(items, batch_size=args.batch_size, progress=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for item, res in zip(items, results):
            f.write(json.dumps({
                "context": item["context"],
                "mention": item["mention"],
                "id": res.id,
                "name": res.name,
                "confidence": round(res.confidence, 4),
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} rows to {args.out}")

    gold = [(res.id, item["gold_id"]) for item, res in zip(items, results) if item.get("gold_id")]
    if gold:
        print(f"Accuracy: {sum(p == g for p, g in gold) / len(gold):.2%} on {len(gold)} rows with gold_id")


def link_main(argv=None):
    p = argparse.ArgumentParser(prog="belink-hpo link", description=COMMANDS["link"][1])
    p.add_argument("--input", required=True,
                   help='JSONL with "context" and "mention" per line; "standard_name" and "gold_id" are optional')
    p.add_argument("--out", default="linked.jsonl")
    p.add_argument("--kb-vectors", required=True, help="Vectors written by embed-terms, e.g. index/hpo_vec.npy")
    p.add_argument("--kb-mapping", default=None, help="Default: ontology_mapping.txt next to --kb-vectors")
    p.add_argument("--preferred-names", default=None, help="index/preferred_names.json, or an .obo file")
    p.add_argument("--expand", action="store_true",
                   help="Generate a standard name with an LLM for lines that have none (needs OPENAI_API_KEY)")
    p.add_argument("--llm", default=None, help="LLM for --expand (default: gpt-5.4)")
    p.add_argument("--llm-base-url", default=None, help="OpenAI-compatible endpoint for --expand")
    p.add_argument("--top-k", type=int, default=30, help="Names retrieved per mention")
    p.add_argument("--alpha", type=float, default=0.6, help="Weight of the mention vector against the standard name")
    p.add_argument("--encoder", default=None, help="Retrieval encoder; must match the index (default: SapBERT)")
    p.add_argument("--adapter", default=DEFAULT_MODEL, help="Hub id or local adapter directory")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--load-4bit", action="store_true", help="Load in 4-bit to save VRAM (slower)")
    args = p.parse_args(argv)

    from pathlib import Path

    from .dataset import load_jsonl
    from .linker import HPOLinker
    from .reranker import HPOReranker
    from .retrieval import DEFAULT_ENCODER, MAPPING_FILE, CandidateRetriever

    items = load_jsonl(args.input)

    expander = None
    if args.expand:
        from .expansion import DEFAULT_MODEL as DEFAULT_LLM, StandardNameGenerator

        expander = StandardNameGenerator(args.llm or DEFAULT_LLM, base_url=args.llm_base_url)
    elif not any(it.get("standard_name") for it in items):
        print("No standard names given and --expand not set: retrieval uses the mention text alone.")

    preferred = None
    if args.preferred_names:
        from .ontology import load_preferred_names

        preferred = load_preferred_names(args.preferred_names)

    mapping = args.kb_mapping or Path(args.kb_vectors).parent / MAPPING_FILE
    linker = HPOLinker(
        CandidateRetriever.from_files(args.kb_vectors, mapping, args.encoder or DEFAULT_ENCODER),
        HPOReranker.from_pretrained(args.adapter, load_in_4bit=args.load_4bit, preferred_names=preferred),
        expander=expander, top_k=args.top_k, alpha=args.alpha,
    )
    results, standard_names = linker.link_batch(items, batch_size=args.batch_size, return_standard_names=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for item, res, std in zip(items, results, standard_names):
            f.write(json.dumps({
                "context": item["context"],
                "mention": item["mention"],
                "standard_name": std,
                "id": res.id,
                "name": res.name,
                "confidence": round(res.confidence, 4),
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} rows to {args.out}")

    gold = [(res.id, item["gold_id"]) for item, res in zip(items, results) if item.get("gold_id")]
    if gold:
        print(f"Accuracy: {sum(p == g for p, g in gold) / len(gold):.2%} on {len(gold)} rows with gold_id")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(f"belink-hpo {__version__}\n\nusage: belink-hpo <command> [options]\n\ncommands:")
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:<14} {help_text}")
        print("\nRun 'belink-hpo <command> --help' for the options of a command.")
        return
    if argv[0] == "--version":
        print(__version__)
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}. Run 'belink-hpo --help' for the list.")
    if command == "link":
        return link_main(rest)
    if command == "rerank":
        return rerank_main(rest)
    module, func = COMMANDS[command][0].split(":")
    return getattr(importlib.import_module(module), func)(rest)


if __name__ == "__main__":
    main()
