#!/usr/bin/env python3
"""Error analysis for an OntoGPT + BeLink run on the Clinical Genetics Physical Exam corpus.

Every document-level error of the system (the same false positives and false negatives
eval_baselines.py counts) is put in one category, so you can say how much each stage of
the pipeline costs. Document level only: where a quote starts and ends plays no part.

False negatives (a gold concept the system did not predict in that observation)
  not extracted          no OntoGPT quote overlaps any gold span of the concept
  retrieval miss         a quote overlaps, but the gold concept was not among the options
                         shown to the re-ranker (from the .candidates.jsonl log)
  re-ranker: other       the gold concept was among the options; BeLink chose another
  re-ranker: declined    the gold concept was among the options; BeLink said "None of the above"
  no candidate log       a quote overlaps, but the log has no entry for it (e.g. fill mode,
                         where OntoGPT's own id was kept)

False positives (a predicted concept not in that observation's gold)
  ungrounded             no HPO id (AUTO:...): BeLink declined or was not asked
                         (absent with --drop-ungrounded, where these are abstentions, not predictions)
  negated finding        the quote overlaps a gold span marked Negated (dropped from the gold
                         with --negated exclude, the default)
  wrong concept          the quote overlaps a gold mention of another concept; split by how the
                         two are related in HPO: too general (ancestor of the gold concept),
                         too specific (descendant), sibling (same parent), unrelated
  quote not located      the quote could not be found in the text (no position)
  no gold nearby         nothing in the gold overlaps the quote: an annotation gap or an
                         over-extraction; review by hand (manual_tag column)

Near misses: a false positive and a false negative in the same observation whose concepts
are ancestor/descendant within --near steps, where a quote of the predicted concept overlaps
a gold span of the missed one (the same finding in the text). A lenient document-level score
counts each such pair (one-to-one, closest first) as a match.

Outputs in --out
  error_summary.tsv     counts and shares per category
  errors_fn.tsv         every false negative with its category and evidence
      columns: gold_id/gold_label/gold_text = the missed gold concept and its annotated text;
      ontogpt_quotes_at_gold / ontogpt_terms_at_gold = what OntoGPT extracted at that position;
      final_ids_at_gold = the ids those quotes ended up with (BeLink's choice, AUTO: = declined);
      belink_vs_gold / hpo_steps = how BeLink's choice relates to the gold concept in HPO:
      "n up" = n is_a steps more general (ancestor), "n down" = more specific (descendant),
      sibling, or unrelated; filled whenever BeLink chose a concept (re-ranker: other, and
      retrieval misses where it picked something). Rows are sorted by category, then distance.
      gold_in_options = was the gold id among the options shown to the re-ranker;
      gold_option_rank = its position there (1 = first); options_shown = per quote, the options
      (gold marked with *) and the re-ranker's choice.
  errors_fp.tsv         every false positive with its category and evidence (+ manual_tag)
and a printed summary.

Usage
  python error_analysis.py --corpus Clinical-Genetics-Physical-Exam-Data-Dev.tsv --hpo hp.obo \
      --pred results/ontogpt_gptoss/predictions_relinked_all.tsv \
      --out results/errors_all [--drop-ungrounded]
  (--candidates defaults to the --pred file with .tsv replaced by .candidates.jsonl)

Keep eval_baselines.py and run_ontogpt_mentions.py next to this script.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict, deque
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_baselines import finalize, load_hpo_full, load_ontogpt, norm_id  # noqa: E402
from run_ontogpt_mentions import consolidate, evaluate, load_corpus  # noqa: E402

# --------------------------------------------------------------------------- HPO hierarchy

def load_parents(path: str) -> dict:
    parents, cur = defaultdict(list), None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                cur = None
            elif line.startswith("id: "):
                cur = line[4:].strip()
            elif line.startswith("is_a: ") and cur:
                parents[cur].append(line[6:].split("!")[0].strip())
    return parents


def ancestors(term: str, parents: dict) -> dict:
    """Every ancestor of term with its shortest distance (term itself excluded)."""
    dist, q = {}, deque([(term, 0)])
    while q:
        t, d = q.popleft()
        for p in parents.get(t, ()):
            if p not in dist:
                dist[p] = d + 1
                q.append((p, d + 1))
    return dist


def relation(pred: str, gold: str, parents: dict, cache: dict) -> tuple[str, int]:
    """How pred relates to gold: too general / too specific / sibling / unrelated, and the distance."""
    for t in (pred, gold):
        if t not in cache:
            cache[t] = ancestors(t, parents)
    if pred in cache[gold]:
        return "too general", cache[gold][pred]
    if gold in cache[pred]:
        return "too specific", cache[pred][gold]
    if set(parents.get(pred, ())) & set(parents.get(gold, ())):
        return "sibling", 2
    return "unrelated", -1


# --------------------------------------------------------------------------- helpers

def overlaps(a_start, a_end, ranges) -> bool:
    return a_start >= 0 and any(min(a_end, e) > max(a_start, s) for s, e in ranges)


def load_candidates(path: str) -> dict:
    """(docid, quote, OntoGPT term) -> list of log entries (options, chosen, ...)."""
    out = defaultdict(list)
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(str(r.get("pmid") or r.get("docid") or ""), (r.get("text") or "").strip(),
                     (r.get("llm_term") or "").strip())].append(r)
    return out


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--hpo", required=True)
    ap.add_argument("--pred", required=True, help="relinked predictions (relink_mentions.py output)")
    ap.add_argument("--candidates", help="BeLink log (default: --pred with .tsv -> .candidates.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--negated", choices=("exclude", "include"), default="exclude")
    ap.add_argument("--near", type=int, default=2, help="max HPO steps for a near miss (default 2)")
    ap.add_argument("--all-hpo-branches", action="store_true")
    ap.add_argument("--drop-ungrounded", action="store_true",
                    help="ignore predictions without an HPO id (BeLink abstentions), as eval_baselines.py --drop-ungrounded")
    args = ap.parse_args()
    cand_path = args.candidates or re.sub(r"\.tsv$", "", args.pred) + ".candidates.jsonl"
    os.makedirs(args.out, exist_ok=True)

    hpo = load_hpo_full(args.hpo)
    parents = load_parents(args.hpo)
    texts, gold = load_corpus(args.corpus, hpo)
    docs = sorted(d for d, t in texts.items() if t.strip())
    docset = set(docs)
    include = args.negated == "include"
    preds, _ = finalize(load_ontogpt(args.pred, docs, texts, hpo), hpo, args)
    preds_all = preds   # abstentions are kept to explain false negatives ("re-ranker: declined")
    if args.drop_ungrounded:
        preds = [p for p in preds if p["grounded"] != "no"]
    cands = load_candidates(cand_path)
    if not cands:
        print(f"note: no candidate log at {cand_path}; retrieval and re-ranker errors cannot be told apart")

    # raw rows, for OntoGPT's term and BeLink's own fields
    with open(args.pred, encoding="utf-8", newline="") as fh:
        raw = [r for r in csv.DictReader(fh, delimiter="\t") if r["docid"] in docset]
    # keyed by position and OntoGPT term: the scorer rebuilds the quote from the position, which
    # differs from the LLM's own quote when it was located with --fuzzy-locate
    extra = defaultdict(list)
    for r in raw:
        k = (r["docid"], int(r["start"]) if r.get("start") else -1, int(r["end"]) if r.get("end") else -1,
             (r.get("predicted_name") or "").strip())
        extra[k].append(r)

    G, NEG, P = defaultdict(list), defaultdict(list), defaultdict(list)
    for g in gold:
        if g["docid"] in docset:
            (G if include or not g["negated"] else NEG)[g["docid"]].append(g)
    for p in preds:
        P[p["docid"]].append(p)
    PALL = defaultdict(list)
    for p in preds_all:
        PALL[p["docid"]].append(p)

    # the scorer's own totals, to check against
    with contextlib.redirect_stdout(io.StringIO()):
        m = evaluate(gold, preds, docset, SimpleNamespace(negated=args.negated, out=tempfile.mkdtemp(),
                                                          mention_match="exact", drop_ungrounded=False))   # preds already filtered
    want_fp, want_fn = m["document_level"]["fp"], m["document_level"]["fn"]

    fn_rows, fp_rows = [], []
    rel_cache: dict = {}
    near_pairs = 0

    def log_entries(p):
        # the same quote can carry several OntoGPT terms ("X of A and B" -> two terms), so the term is part of the key
        span, term = (p.get("predicted_span") or "").strip(), (p.get("predicted_name") or "").strip()
        rows = extra.get((p["docid"], p["start"], p["end"], term), [])
        r = next((x for x in rows if (x.get("predicted_span") or "").strip() == span), rows[0] if rows else {})
        quote = (r.get("predicted_span") or span).strip()          # the LLM's own quote, as in BeLink's log
        return cands.get((p["docid"], quote, term or (r.get("predicted_name") or "").strip()), []), r

    for d in docs:
        text, g, p = texts[d], G.get(d, []), P.get(d, [])
        gs, ps = {x["hpo_id"] for x in g}, {x["hpo_id"] for x in p}
        fn_ids, fp_ids = sorted(gs - ps), sorted(ps - gs)

        # ---------------- false negatives
        for gid in fn_ids:
            gm = [x for x in g if x["hpo_id"] == gid]
            ranges = [rg for x in gm for rg in (x["ranges"] or [(x["start"], x["end"])])]
            over = [x for x in PALL.get(d, []) if overlaps(x["start"], x["end"], ranges)]
            options_seen, chosen, logged = False, [], False
            ranks, shown, seen_q, picked = [], [], set(), set()
            for x in over:
                entries, _ = log_entries(x)
                for e in entries:
                    logged = True
                    opts = [o for o in e.get("options", []) if o[1]]
                    ids = [o[1] for o in opts]
                    if gid in ids:
                        options_seen = True
                        ranks.append(ids.index(gid) + 1)
                    chosen.append(e.get("chosen") or "None")
                    if e.get("chosen") and e["chosen"] != gid:
                        picked.add(consolidate(norm_id(e["chosen"]), hpo) or norm_id(e["chosen"]))
                    q = (x["predicted_span"], e.get("llm_term", ""))
                    if q in seen_q:
                        continue
                    seen_q.add(q)
                    listed = " | ".join(f'{"*" if o[1] == gid else ""}{o[0]} {o[2]} ({o[1]})' for o in opts)
                    pick = (" ".join(str(v) for v in (e.get("letter"), e.get("chosen_name")) if v)
                            + f' ({e["chosen"]})').strip() if e.get("chosen") else "None of the above"
                    shown.append(f'{x["predicted_span"]} [term: {e.get("llm_term", "")}] -> [{listed}] -> chose: {pick}')
            if not over:
                cat = "not extracted"
            elif not logged:
                cat = "no candidate log"
            elif not options_seen:
                cat = "retrieval miss"
            elif all(c == "None" for c in chosen):
                cat = "re-ranker: declined"
            else:
                cat = "re-ranker: other"
            # how BeLink's choice relates to the gold concept in HPO (closest choice if several)
            rel, steps, srt = "", "", (9, 0)
            for c in picked:
                if not c.startswith("HP:") or c == gid:
                    continue
                r_, dd = relation(c, gid, parents, rel_cache)
                key = ({"too general": 0, "too specific": 0, "sibling": 1, "unrelated": 2}[r_], abs(dd))
                if key < srt:
                    srt = key
                    rel = r_
                    steps = (f"{dd} up" if r_ == "too general" else f"{dd} down" if r_ == "too specific"
                             else r_)
            fn_rows.append([d, gid, hpo["names"].get(gid, ""), cat, "; ".join(x["text"] for x in gm),
                            "; ".join(sorted({x["predicted_span"] for x in over})),
                            "; ".join(sorted({x.get("predicted_name", "") for x in over} - {""})),
                            "; ".join(sorted({f'{x["hpo_id"]} {hpo["names"].get(x["hpo_id"], "")}'.strip()
                                              for x in over})),
                            rel, steps,
                            ("yes" if options_seen else "no") if logged else "",
                            min(ranks) if ranks else "",
                            " || ".join(shown),
                            text, srt])

        # ---------------- false positives
        for pid in fp_ids:
            pm = [x for x in p if x["hpo_id"] == pid]
            located = [x for x in pm if x["start"] >= 0]
            rel, dist, near_gold = "", "", ""
            if not pid.startswith("HP:"):
                cat = "ungrounded"
            elif not include and any(overlaps(x["start"], x["end"], rg)
                                     for x in located for n in NEG.get(d, [])
                                     for rg in [n["ranges"] or [(n["start"], n["end"])]]):
                cat = "negated finding"
            else:
                over_gold = [gg for gg in g for x in located
                             if overlaps(x["start"], x["end"], gg["ranges"] or [(gg["start"], gg["end"])])]
                if over_gold:
                    best = None
                    for gg in over_gold:
                        r_, dd = relation(pid, gg["hpo_id"], parents, rel_cache)
                        rank = {"too general": 0, "too specific": 0, "sibling": 1, "unrelated": 2}[r_]
                        if best is None or (rank, abs(dd)) < best[0]:
                            best = ((rank, abs(dd)), r_, dd, gg["hpo_id"])
                    _, rel, dist, near_gold = best
                    cat = f"wrong concept: {rel}"
                elif not located:
                    cat = "quote not located"
                else:
                    cat = "no gold nearby"
            belink_score = ""
            for x in pm:
                _, r = log_entries(x)
                if r.get("linker_score"):
                    belink_score = r["linker_score"]
                    break
            fp_rows.append([d, pid, hpo["names"].get(pid, ""), cat, "; ".join(sorted({x["predicted_span"] for x in pm})),
                            "; ".join(sorted({x.get("predicted_name", "") for x in pm})),
                            near_gold, hpo["names"].get(near_gold, "") if near_gold else "", dist, belink_score,
                            "", text])

        # ---------------- near misses (one-to-one, closest first)
        # a pair needs a quote of the predicted concept that overlaps a gold span of the missed
        # concept, so both errors are about the same finding in the text
        pairs = []
        for pid in fp_ids:
            if not pid.startswith("HP:"):
                continue
            pspans = [(x["start"], x["end"]) for x in p if x["hpo_id"] == pid and x["start"] >= 0]
            for gid in fn_ids:
                granges = [rg for x in g if x["hpo_id"] == gid for rg in (x["ranges"] or [(x["start"], x["end"])])]
                if not any(overlaps(a, b, granges) for a, b in pspans):
                    continue
                r_, dd = relation(pid, gid, parents, rel_cache)
                if r_ in ("too general", "too specific") and dd <= args.near:
                    pairs.append((dd, pid, gid))
        used_p, used_g = set(), set()
        for dd, pid, gid in sorted(pairs):
            if pid not in used_p and gid not in used_g:
                used_p.add(pid)
                used_g.add(gid)
                near_pairs += 1

    assert len(fp_rows) == want_fp and len(fn_rows) == want_fn, \
        f"analysis found {len(fp_rows)} FP / {len(fn_rows)} FN, scorer {want_fp} / {want_fn}"

    # ---------------- outputs
    fn_order = ["not extracted", "no candidate log", "retrieval miss", "re-ranker: declined", "re-ranker: other"]
    fn_rows.sort(key=lambda r: (fn_order.index(r[3]) if r[3] in fn_order else 99, r[-1], r[0], r[1]))
    fn_rows = [r[:-1] for r in fn_rows]
    fp_rows.sort(key=lambda r: (r[3], r[0], r[1]))
    def write(name, header, rows):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)

    write("errors_fn.tsv", ["docid", "gold_id", "gold_label", "category", "gold_text",
                            "ontogpt_quotes_at_gold", "ontogpt_terms_at_gold", "final_ids_at_gold",
                            "belink_vs_gold", "hpo_steps",
                            "gold_in_options", "gold_option_rank", "options_shown", "observation"], fn_rows)
    write("errors_fp.tsv", ["docid", "pred_id", "pred_label", "category", "quotes", "ontogpt_terms",
                            "overlapping_gold_id", "overlapping_gold_label", "hpo_distance", "belink_score",
                            "manual_tag", "observation"], fp_rows)

    tp = m["document_level"]["tp"]
    lt, lfp, lfn = tp + near_pairs, want_fp - near_pairs, want_fn - near_pairs
    prf = lambda a, b, c: (a / (a + b) if a + b else 0, a / (a + c) if a + c else 0,
                           2 * a / (2 * a + b + c) if a else 0)
    summary = []
    for kind, rows in (("FN", fn_rows), ("FP", fp_rows)):
        cnt = Counter(r[3] for r in rows)
        cats = [c for c in fn_order if c in cnt] if kind == "FN" else [c for c, _ in cnt.most_common()]
        for cat in cats:
            n = cnt[cat]
            summary.append([kind, cat, n, f"{100 * n / len(rows):.1f}%"])
            if kind == "FN" and cat == "re-ranker: other":
                # BeLink's choice vs the gold concept: n steps up (more general) / down (more specific)
                sub = Counter(r[9] or "no HPO choice" for r in rows if r[3] == cat)
                def k(x):
                    v, _, d = x.partition(" ")
                    return ({"up": 0, "down": 1}.get(d, 2), int(v) if v.isdigit() else 0, x)
                for st in sorted(sub, key=k):
                    summary.append([kind, f"  {cat} | BeLink {st}" if st[0].isdigit() else f"  {cat} | {st}",
                                    sub[st], f"{100 * sub[st] / n:.1f}% of {cat}"])
    s_p, s_r, s_f = prf(tp, want_fp, want_fn)
    l_p, l_r, l_f = prf(lt, lfp, lfn)
    summary += [["score", "document strict P/R/F1", tp, f"{s_p:.3f} / {s_r:.3f} / {s_f:.3f}"],
                ["score", f"document lenient P/R/F1 (near misses within {args.near} steps count)", lt,
                 f"{l_p:.3f} / {l_r:.3f} / {l_f:.3f}"],
                ["score", "near-miss pairs (FP ancestor/descendant of an FN, quote overlapping its gold span)", near_pairs, ""]]
    write("error_summary.tsv", ["errors", "category", "count", "share"], summary)

    print(f"Errors of {os.path.basename(args.pred)} (negated gold terms: {args.negated}; ungrounded predictions "
          f"{'dropped' if args.drop_ungrounded else 'counted as FP'}); "
          f"{want_fn} false negatives, {want_fp} false positives (document level)")
    for kind, title in (("FN", "False negatives"), ("FP", "False positives")):
        print(f"\n{title}")
        for _, cat, n, share in [r for r in summary if r[0] == kind]:
            print(f"  {cat:52}{n:>6}  {share}")
    print(f"\nNear misses within {args.near} HPO steps, quote overlapping the gold span: {near_pairs} pairs")
    print(f"Document level strict  P/R/F1: {s_p:.3f} / {s_r:.3f} / {s_f:.3f}")
    print(f"Document level lenient P/R/F1: {l_p:.3f} / {l_r:.3f} / {l_f:.3f}")
    print(f"\nWrote error_summary.tsv, errors_fn.tsv, errors_fp.tsv in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
