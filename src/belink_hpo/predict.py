"""Batched label prediction over a JSONL file in the messages format.

    belink-hpo predict --file data/test.jsonl
    belink-hpo predict --file data/test.jsonl --adapter qwen3-mcq-lora/checkpoint-1500 --load-4bit

Input:  JSONL in the messages format (an assistant message, if present, is used as the gold label)
Output: JSONL with one {"index", "pred", "confidence", "gold"} object per input row
"""

from __future__ import annotations

import argparse
import json
import time

from . import DEFAULT_MODEL
from .prompt import LETTERS


def main(argv=None):
    p = argparse.ArgumentParser(prog="belink-hpo predict", description=__doc__.split("\n")[0])
    p.add_argument("--adapter", default=DEFAULT_MODEL, help="Hub id, adapter directory, or a checkpoint inside it")
    p.add_argument("--file", default="data/test.jsonl")
    p.add_argument("--out", default="predictions.jsonl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--labels", default=LETTERS, help="Candidate labels, e.g. ABCDEFGHIJK")
    p.add_argument("--n", type=int, default=0, help="Only run the first N rows (0 = all)")
    p.add_argument("--load-4bit", action="store_true", help="Load in 4-bit to save VRAM (slower, adapter not merged)")
    args = p.parse_args(argv)

    from .reranker import HPOReranker

    with open(args.file, encoding="utf-8") as f:
        rows = [json.loads(line)["messages"] for line in f if line.strip()]
    if args.n:
        rows = rows[: args.n]

    users = [next(m["content"] for m in msgs if m["role"] == "user") for msgs in rows]
    gold = [msgs[-1]["content"].strip() if msgs[-1]["role"] == "assistant" else None for msgs in rows]

    reranker = HPOReranker.from_pretrained(args.adapter, load_in_4bit=args.load_4bit)

    start = time.time()
    results = reranker.predict_labels(users, args.labels, args.batch_size, progress=True)
    elapsed = time.time() - start

    with open(args.out, "w", encoding="utf-8") as f:
        for i, ((pred, conf), g) in enumerate(zip(results, gold)):
            f.write(json.dumps({"index": i, "pred": pred, "confidence": round(conf, 4), "gold": g}) + "\n")

    print(f"\n{len(rows)} rows in {elapsed:.1f}s ({len(rows) / max(elapsed, 1e-9):.1f} rows/s) -> {args.out}")
    labelled = [(r[0], g) for r, g in zip(results, gold) if g]
    if labelled:
        acc = sum(pred == g for pred, g in labelled) / len(labelled)
        print(f"Accuracy: {acc:.2%} on {len(labelled)} labelled rows")


if __name__ == "__main__":
    main()
