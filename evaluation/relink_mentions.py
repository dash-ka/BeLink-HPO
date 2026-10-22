#!/usr/bin/env python3
"""Ground OntoGPT's phenotype mentions with your own entity linker.

Mention-level counterpart of 03b_relink.py. It reads predictions.tsv written by
run_ontogpt_mentions.py (one row per extracted mention, with its quotation and
character offsets), sends the mentions to the `link()` function of your linker
module, and writes a predictions table in the same format, which
eval_baselines.py --ontogpt accepts unchanged.

Modes
  --mode ungrounded   (default) keep OntoGPT's HP ids; ask your linker only for the
                      mentions OntoGPT could not ground (the AUTO: ones)
  --mode all          ask your linker for every mention; OntoGPT's id is kept only
                      in the ontogpt_id column for comparison

Output columns
  docid hpo_id predicted_name predicted_span start end grounded   (as the input;
          hpo_id is the id used for evaluation)
  ontogpt_id    OntoGPT's own id, always
  linker_id     the linker's id ("" when it declined or was not asked)
  linker_label  the label the linker returned
  linker_score  the linker's score
  source        linker = hpo_id came from the linker; ontogpt = from OntoGPT

Usage
  python relink_mentions.py --corpus Clinical-Genetics-Physical-Exam-Data-Dev.tsv \
      --pred results/ontogpt_gptoss/predictions.tsv --linker my_linker.py \
      --out results/ontogpt_gptoss/predictions_relinked.tsv
  python eval_baselines.py ... --ontogpt results/ontogpt_gptoss/predictions_relinked.tsv

The linker module must define (same interface as for 03b_relink.py)

    def link(mentions: list[dict]) -> list[dict | None]

Each mention is
    {"text": the exact quotation (or the model's term if there is none),
     "llm_term": the model's interpreted term, "context": text around it,
     "start": int, "end": int, "docid": str, "pmid": str (= docid),
     "section_id": "", "status": "observed",
     "ontogpt_id": OntoGPT's HP id, or "" if OntoGPT did not ground it}
where context[start:end] == text when the quotation was located in the document,
and context == text (start 0) otherwise. Return, in the same order, None (no link)
or {"id": "HP:0001250", "score": 0.93, "label": "Seizure"}; score and label are optional.

Standard library only (your linker brings its own dependencies).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from urllib.parse import unquote

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def load_texts(path: str) -> dict:
    texts = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            d = (row.get("StringID") or "").strip()
            if d:
                texts.setdefault(d, row.get("Text") or "")
    return texts


def context_for(text: str, mention: str, start: int, end: int, width: int) -> tuple[str, int, int]:
    """Window of the document around the mention; falls back to the bare mention."""
    if not (0 <= start < end <= len(text)):
        i = text.lower().find(mention.lower()) if mention else -1
        if i < 0:
            return mention, 0, len(mention)
        start, end = i, i + len(mention)
    lo, hi = max(0, start - width), min(len(text), end + width)
    return text[lo:hi], start - lo, end - lo


def load_linker(path: str):
    spec = importlib.util.spec_from_file_location("user_linker", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "link", None)):
        raise SystemExit(f"{path} must define a function link(mentions) -> list")
    return mod.link


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus TSV (StringID, Text, ...), for the context")
    ap.add_argument("--pred", required=True, help="predictions.tsv from run_ontogpt_mentions.py")
    ap.add_argument("--linker", required=True, help="Python file defining link(mentions)")
    ap.add_argument("--out", required=True, help="output predictions TSV")
    ap.add_argument("--mode", choices=["ungrounded", "all"], default="ungrounded")
    ap.add_argument("--min-score", type=float, default=None, help="discard links scored below this")
    ap.add_argument("--context-chars", type=int, default=200, help="characters of context each side (default 200)")
    args = ap.parse_args()

    texts = load_texts(args.corpus)
    with open(args.pred, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    mentions, todo, seen, missing = [], [], {}, 0
    for ri, r in enumerate(rows):
        tid = r["hpo_id"]
        is_hp = tid.startswith("HP:")
        label = (r.get("predicted_name") or unquote(tid.split(":", 1)[-1])).strip()
        r.update(predicted_name=label, ontogpt_id=tid, linker_id="", linker_label="", linker_score="",
                 source="ontogpt", grounded="yes" if is_hp else "no")
        if not (args.mode == "all" or not is_hp):
            continue
        text = texts.get(r["docid"])
        if text is None:
            missing += 1
            text = ""
        quote = (r.get("predicted_span") or "").strip()
        mention = quote or label
        start = int(r["start"]) if r.get("start") else -1
        end = int(r["end"]) if r.get("end") else -1
        ctx, a, b = context_for(text, mention, start, end, args.context_chars)
        key = (r["docid"], start, end, mention.lower(), label.lower())
        if key not in seen:
            seen[key] = len(mentions)
            mentions.append({"text": mention, "llm_term": label, "context": ctx, "start": a, "end": b,
                             "docid": r["docid"], "pmid": r["docid"], "section_id": "", "status": "observed",
                             "ontogpt_id": tid if is_hp else ""})
        todo.append((ri, seen[key]))
    if missing:
        print(f"warning: {missing} predictions refer to documents not in {args.corpus}")

    print(f"{len(rows)} predictions; sending {len(mentions)} distinct mentions to the linker ({args.mode})")
    results = load_linker(args.linker)(mentions) if mentions else []
    if len(results) != len(mentions):
        raise SystemExit(f"link() returned {len(results)} results for {len(mentions)} mentions")

    linked = changed = 0
    for ri, mi in todo:
        res, row = results[mi], rows[ri]
        ok = isinstance(res, dict) and str(res.get("id", "")).startswith("HP:")
        if ok and args.min_score is not None and res.get("score") is not None and float(res["score"]) < args.min_score:
            ok = False
        if ok:
            linked += 1
            changed += row["hpo_id"] != res["id"]
            row.update(hpo_id=res["id"], grounded="yes", source="linker", linker_id=res["id"],
                       linker_label=str(res.get("label") or ""),
                       linker_score="" if res.get("score") is None else f"{float(res['score']):.4f}")
        elif args.mode == "all":
            # the linker declined: no fall-back to OntoGPT's id, so the comparison stays clean
            row.update(hpo_id="AUTO:" + row["predicted_name"].replace(" ", "%20"), grounded="no", source="linker")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cols = ["docid", "hpo_id", "predicted_name", "predicted_span", "start", "end", "grounded",
            "ontogpt_id", "linker_id", "linker_label", "linker_score", "source"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    grounded = sum(1 for r in rows if r["grounded"] == "yes")
    print(f"Linker grounded {linked} of {len(todo)} mentions it was asked about"
          + (f" ({changed} differ from OntoGPT's id)" if args.mode == "all" else ""))
    print(f"Overall {grounded} of {len(rows)} predictions now have an HPO id")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
