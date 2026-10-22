#!/usr/bin/env python3
"""Copy quote positions from a re-collected predictions.tsv into relinked files.

After `run_ontogpt_mentions.py --collect-only --fuzzy-locate`, predictions.tsv has positions
for quotes the LLM did not copy verbatim. The relinked files (relink_mentions.py output) list
the same mentions in the same order, so their positions can be updated without running BeLink
again: BeLink's ids stay exactly as they were, only start/end change.

  python update_positions.py --pred results/ontogpt_gptoss/predictions.tsv \
      results/ontogpt_gptoss/predictions_relinked_fill.tsv \
      results/ontogpt_gptoss/predictions_relinked_all.tsv \
      results/ontogpt_gptoss/predictions_relinked_all_term.tsv

Each file is rewritten in place after a backup copy (<file>.bak). The script stops without
writing if a file does not list the same mentions (docid, quote) in the same order.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys


def read(path: str) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        return list(r.fieldnames or []), list(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="re-collected predictions.tsv")
    ap.add_argument("relinked", nargs="+", help="relinked files to update")
    args = ap.parse_args()
    _, new = read(args.pred)
    # relink_mentions.py may fill an empty OntoGPT term from the id, so only docid and quote are compared
    key = lambda r: (r.get("docid"), (r.get("predicted_span") or "").strip())
    for path in args.relinked:
        cols, rows = read(path)
        if len(rows) != len(new) or any(key(a) != key(b) for a, b in zip(rows, new)):
            print(f"{path}: mentions differ from {args.pred}; not changed", file=sys.stderr)
            return 1
        moved = located = 0
        for r, n in zip(rows, new):
            if (r.get("start"), r.get("end")) != (n.get("start"), n.get("end")):
                moved += 1
                located += not r.get("start") and bool(n.get("start"))
                r["start"], r["end"] = n.get("start", ""), n.get("end", "")
        shutil.copyfile(path, path + ".bak")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"{path}: {moved} positions changed ({located} quotes newly located); backup {path}.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
