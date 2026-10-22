#!/usr/bin/env python3
"""Run OntoGPT mention-level phenotype extraction on the Clinical Genetics Physical
Exam corpus (one TSV: StringID, Text, Term, Negated, Spans, ...) and score it
against the gold annotations.

Steps
  1. Read the corpus TSV. Each StringID is one document; its text is written to
     <out>/texts/<file>.txt and the gold annotations to <out>/gold.tsv.
  2. For each document call
         ontogpt extract -t phenotype_mentions.yaml -i <file>.txt -O json -o <file>.json -m <model>
     (outputs in <out>/ontogpt/). Documents already processed are skipped, so
     the run can be interrupted and resumed.
  3. Flatten the outputs into <out>/predictions.tsv:
         docid  hpo_id  predicted_name  predicted_span  start  end  grounded  span_pairing
     `start`/`end` locate the quotation in the
     text (empty if it could not be found verbatim).
  4. Score predictions against gold:
       document level  presence/absence of each HPO id in the text; positions ignored
       mention level   each mention with its position: same HPO id and the same
                       character boundaries (--mention-match overlap: overlapping)

Negated gold terms (Negated == X, e.g. "Normal lips" -> HP:0000159, "no clinodactyly")
The evaluation measures recognition only, never present/negated classification:
  --negated exclude   (default) negated gold terms are dropped from the gold
                      standard; all predictions are still scored
  --negated include   negated gold terms are scored like any other term
  The template extracts every phenotype mention, negated or normal ones included,
  so switching --negated never needs a new model run: use --collect-only.

Usage
  python run_ontogpt_mentions.py --corpus Clinical-Genetics-Physical-Exam-Data-Dev.tsv \
      --out work/biocgs --template phenotype_mentions.yaml --model gpt-4o \
      --hpo hp.obo --workers 4
  # trial on 20 documents
  python run_ontogpt_mentions.py ... --limit 20
  # rescore including negated gold terms, no model calls
  python run_ontogpt_mentions.py ... --collect-only --negated include

Requires `pip install ontogpt` and an API key for the chosen model. Keep
ontogpt_fixed.py next to this script (or pass --plain-ontogpt). --hpo is
optional: with it, alternative/secondary HPO ids in gold and predictions are
mapped to their primary id, obsolete ids (gold and predictions) to their replaced_by term in the
current hp.obo, and gold ids not in hp.obo, or obsolete without a replacement, are dropped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

FIXED_ONTOGPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontogpt_fixed.py")
# (term field, quotation field). The last two are from earlier template versions.
FIELDS = (("phenotypes", "phenotype_spans"), ("present_phenotypes", "present_spans"),
          ("negated_phenotypes", "negated_spans"))
NEGATED_FLAGS = {"X", "Y", "YES", "TRUE", "1"}


# --------------------------------------------------------------------------- HPO

def load_hpo(path: str) -> dict:
    """Minimal OBO reader: current term ids, alt_id -> current id, and obsolete ids with their
    replaced_by (see consolidate). Same as eval_baselines.py."""
    terms, alt, cur, in_term = set(), {}, None, False
    obsolete, replaced = set(), {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                in_term, cur = line == "[Term]", None
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                cur = line[4:].strip()
                terms.add(cur)
            elif line.startswith("alt_id: ") and cur:
                alt[line[8:].strip()] = cur
            elif line.startswith("is_obsolete: true") and cur:
                obsolete.add(cur)
            elif line.startswith("replaced_by: ") and cur:
                replaced[cur] = line[13:].strip()
    terms -= obsolete
    return {"terms": terms, "alt": alt, "obsolete": obsolete,
            "replaced": {o: n for o, n in replaced.items() if o in obsolete}}


def consolidate(hpo_id: str, hpo: dict | None, gold: bool = False) -> str | None:
    """Current id for hpo_id, the same for gold and predictions: itself, the term that lists it as
    alt_id, or, for an obsolete id, its replaced_by term (else the term listing it as alt_id).
    None for an obsolete id without a replacement, or an id not in hp.obo. (gold: unused, kept
    for callers.)"""
    if hpo is None:
        return hpo_id
    if hpo_id in hpo["terms"]:
        return hpo_id
    if hpo_id in hpo.get("obsolete", ()):
        return hpo.get("replaced", {}).get(hpo_id) or hpo["alt"].get(hpo_id)
    return hpo["alt"].get(hpo_id)


# --------------------------------------------------------------------------- corpus

def parse_spans(s: str) -> list[tuple[int, int]]:
    """'12-20' or discontiguous '12-20,30-35' -> [(12, 20), (30, 35)]."""
    ranges = []
    for part in re.split(r"[,;]", s or ""):
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", part)
        if m:
            ranges.append((int(m.group(1)), int(m.group(2))))
    if not ranges:                       # unknown format: first and last number
        nums = [int(x) for x in re.findall(r"\d+", s or "")]
        if len(nums) >= 2:
            ranges.append((nums[0], nums[-1]))
    return ranges


def doc_filename(docid: str) -> str:
    """File-safe, unique (also on case-insensitive file systems) name for a document."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", docid).strip("._")[:60] or "doc"
    return f"{base}_{hashlib.md5(docid.encode('utf-8')).hexdigest()[:10]}"


def load_corpus(path: str, hpo: dict | None) -> tuple[dict, list[dict]]:
    texts: dict[str, str] = {}
    gold: list[dict] = []
    dropped, no_span, remapped = Counter(), 0, Counter()
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            docid = (row.get("StringID") or "").strip()
            if not docid:
                continue
            texts.setdefault(docid, row.get("Text") or "")
            term = (row.get("Term") or "").strip()
            if not term or term.upper() == "NA":
                continue
            hid = consolidate(term, hpo, gold=True)
            if not hid:
                dropped[term] += 1
                continue
            if hpo and term in hpo.get("obsolete", ()):
                remapped[f"{term}->{hid}"] += 1
            ranges = parse_spans(row.get("Spans", ""))
            start, end = (min(r[0] for r in ranges), max(r[1] for r in ranges)) if ranges else (-1, -1)
            no_span += not ranges
            gold.append({"docid": docid, "hpo_id": hid, "start": start, "end": end, "ranges": ranges,
                         "text": texts[docid][start:end] if ranges else "",
                         "negated": (row.get("Negated") or "").strip().upper() in NEGATED_FLAGS,
                         "spans": row.get("Spans", "")})
    if remapped:
        print(f"note: {sum(remapped.values())} gold annotations with {len(remapped)} obsolete ids remapped to the "
              f"current term ({', '.join(list(remapped)[:5])}{', ...' if len(remapped) > 5 else ''})")
    if dropped:
        print(f"note: {sum(dropped.values())} gold annotations with {len(dropped)} ids not in hp.obo dropped "
              f"(e.g. {', '.join(list(dropped)[:5])})")
    if no_span:
        print(f"note: {no_span} gold annotations without a readable span (scored at document level only)")
    return texts, gold


# --------------------------------------------------------------------------- OntoGPT

def run_one(doc: dict, args) -> tuple[dict, str]:
    """Run OntoGPT on one document. Returns (doc, 'done' | 'skipped' | error text)."""
    src = os.path.abspath(os.path.join(args.out, "texts", doc["file"] + ".txt"))
    dst = os.path.abspath(os.path.join(args.out, "ontogpt", doc["file"] + ".json"))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        try:
            with open(dst, encoding="utf-8") as fh:
                if json.load(fh).get("raw_completion_output"):
                    return doc, "skipped"
        except (ValueError, OSError):
            pass
        os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # ontogpt_fixed.py fixes replies with an empty field,
    # which otherwise make OntoGPT drop the next field.
    launcher = ["ontogpt"] if args.plain_ontogpt else [sys.executable, FIXED_ONTOGPT]
    part = dst + ".part"
    cmd = launcher + ["extract", "-t", args.template, "-i", src, "-O", "json", "-o", part, "-m", args.model]
    if args.extra:
        cmd += args.extra.split()
    # OntoGPT 1.2.0's ./.litellm_cache cannot be read back (missing optional modules),
    # so each call starts in its own empty folder.
    with tempfile.TemporaryDirectory(prefix="ontogpt_") as workdir:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, cwd=workdir)
        except subprocess.TimeoutExpired:
            return doc, f"timeout after {args.timeout}s"
    lines = [ln.strip() for ln in p.stderr.splitlines() if ln.strip()]
    if p.returncode != 0:
        if os.path.exists(part):
            os.remove(part)
        cause = [ln for ln in lines if any(k in ln.lower() for k in ("encountered", "error", "exception"))]
        return doc, (cause[-1] if cause else (lines or ["unknown error"])[-1])[:400]
    # A failed model call still exits 0 but leaves no raw_completion_output.
    raw = None
    if os.path.exists(part) and os.path.getsize(part) > 0:
        try:
            with open(part, encoding="utf-8") as fh:
                raw = json.load(fh).get("raw_completion_output")
        except (json.JSONDecodeError, OSError):
            raw = None
    if not raw:
        if os.path.exists(part):
            os.remove(part)
        cause = [ln for ln in lines if "encountered" in ln.lower() or "exception" in ln.lower()]
        errors = [ln for ln in lines if "error" in ln.lower()]
        msg = cause[-1] if cause else (errors[-1] if errors else "no error message; rerun with -vv")
        return doc, ("model returned no answer: " + msg)[:400]
    os.replace(part, dst)
    return doc, "done"


# --------------------------------------------------------------------------- collect

PLACEHOLDERS = {"", "none", "unknown", "na", "n/a", "nil", "null", "not reported", "none reported", "not stated",
                "not specified", "not mentioned", "not available", "not applicable", "none stated", "none mentioned",
                "none explicitly reported", "none explicitly stated", "no abnormalities", "not provided", "unspecified"}


def is_placeholder(term_id: str, label: str) -> bool:
    text = (label or term_id.split(":", 1)[-1].replace("%20", " ")).strip().lower().rstrip(".")
    return text in PLACEHOLDERS


def as_list(x) -> list:
    return x if isinstance(x, list) else ([] if x in (None, "") else [x])


def _words(s: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def _acronym(s: str) -> str:
    w = [x for x in re.findall(r"[a-z]+", s.lower().replace("-", " ")) if x not in {"of", "the", "and", "in", "with"}]
    return "".join(x[0] for x in w) if len(w) >= 3 else ""


def pair_spans(spans: list[str], labels: list[str]) -> tuple[list[str], str]:
    """Quotation for each term: by position when the lists have the same length,
    otherwise the quotation sharing most words with the term."""
    if not spans:
        return [""] * len(labels), ""
    if len(spans) == len(labels):
        return list(spans), "position"
    out, used = [], set()
    for i, lab in enumerate(labels):
        lw, acr = _words(lab), _acronym(lab)

        def score(j):
            hit = len(lw & _words(spans[j])) + (2 if acr and acr in re.findall(r"[a-z]+", spans[j].lower()) else 0)
            return hit, j not in used, -abs(j - i)
        best = max(range(len(spans)), key=score)
        if score(best)[0]:
            used.add(best)
            out.append(spans[best])
        else:
            out.append("")
    return out, "overlap"


FUZZY_MAX_GAP = 6          # words the text may have between two words of the quote
_BREAK = re.compile(r"[.;\n]")   # a fuzzy match never crosses a sentence or line break


def _close(a: str, b: str) -> bool:
    """Same word, or one letter added, removed or changed (both words 4+ letters): alae/allae."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)):] == b[i + 1:]


def fuzzy_hits(text: str, quote: str, max_gap: int = FUZZY_MAX_GAP) -> list[tuple[int, int]]:
    """Spans where the quote's words occur in order, each word equal or one letter off, with at
    most max_gap other words between two of them and no sentence break: the LLM's
    'Flat nasal bridge' for 'Flat, short nasal bridge', 'cleft palate' for 'cleft lip and palate'.
    Best (fewest skipped words) first. Quotes of one word only get the typo tolerance."""
    qw = [w.lower() for w in re.findall(r"\w+", quote)]
    toks = [(m.start(), m.end(), m.group().lower()) for m in re.finditer(r"\w+", text)]
    if not qw or not toks:
        return []
    found = []
    for i, (s0, e0, w0) in enumerate(toks):
        if not _close(qw[0], w0):
            continue
        j, skipped, end, ok = i, 0, e0, True
        for w in qw[1:]:
            k = j + 1
            while k < len(toks) and k - j - 1 <= max_gap and not _BREAK.search(text[toks[k - 1][1]:toks[k][0]]):
                if _close(w, toks[k][2]):
                    break
                k += 1
            else:
                ok = False
            if not ok or k >= len(toks) or k - j - 1 > max_gap or not _close(w, toks[k][2]):
                ok = False
                break
            skipped += k - j - 1
            j, end = k, toks[k][1]
        if ok and (len(qw) > 1 or skipped == 0):
            found.append((skipped, s0, end))
    return [(s, e) for _, s, e in sorted(found)]


def locate(text: str, quote: str, taken: set, fuzzy: bool = False) -> tuple[int, int] | None:
    """Character offsets of a quotation in the text; the first occurrence not yet
    used by another quotation (a repeated quotation may reuse one). With fuzzy, a quotation
    that is not verbatim is matched word by word (see fuzzy_hits); exact matches come first."""
    q = quote.strip().strip("\"'")
    if not q:
        return None
    hits = [(m.start(), m.end()) for m in re.finditer(re.escape(q), text, flags=re.I)]
    if not hits:                                   # tolerate differences in spacing/punctuation
        words = re.findall(r"\w+", q)
        if words:
            pat = r"\W+".join(re.escape(w) for w in words)
            hits = [(m.start(), m.end()) for m in re.finditer(pat, text, flags=re.I)]
    if not hits and fuzzy:
        hits = fuzzy_hits(text, q)
    if not hits:
        return None
    for h in hits:
        if h not in taken:
            taken.add(h)
            return h
    return hits[0]


def collect(docs: list[dict], texts: dict, hpo: dict | None, args) -> tuple[list[dict], set]:
    preds, scored_docs, misaligned, unlocated, fuzzy_located, obsolete_converted = [], set(), 0, 0, 0, 0
    for d in docs:
        path = os.path.join(args.out, "ontogpt", d["file"] + ".json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                res = json.load(fh)
        except (json.JSONDecodeError, OSError):
            print(f"  warning: unreadable output {path}", file=sys.stderr)
            continue
        scored_docs.add(d["docid"])
        text = texts[d["docid"]]
        labels = {e.get("id"): e.get("label", "") for e in as_list(res.get("named_entities")) if isinstance(e, dict)}
        obj = res.get("extracted_object") or {}
        taken: set = set()
        for slot, span_slot in FIELDS:
            terms = as_list(obj.get(slot))
            spans = [str(x).strip() for x in as_list(obj.get(span_slot)) if str(x).strip()]
            labs = []
            for term in terms:
                tid = term.get("id") if isinstance(term, dict) else str(term)
                labs.append(labels.get(tid) or (term.get("label", "") if isinstance(term, dict) else "")
                            or unquote(str(tid).split(":", 1)[-1]))
            paired, how = pair_spans(spans, labs)
            misaligned += how == "overlap"
            for i, term in enumerate(terms):
                tid = term.get("id") if isinstance(term, dict) else str(term)
                if not tid:
                    continue
                grounded = tid.startswith("HP:")
                if not grounded and is_placeholder(tid, labs[i]):
                    continue
                if grounded:
                    obsolete_converted += bool(hpo) and tid in hpo.get("obsolete", ()) \
                        and consolidate(tid, hpo) not in (None, tid)
                    tid = consolidate(tid, hpo) or tid
                loc = locate(text, paired[i], taken) if paired[i] else None
                if loc is None and paired[i] and getattr(args, "fuzzy_locate", False):
                    loc = locate(text, paired[i], taken, fuzzy=True)
                    fuzzy_located += loc is not None
                unlocated += bool(paired[i]) and loc is None
                preds.append({"docid": d["docid"], "hpo_id": tid, "predicted_name": labs[i],
                              "predicted_span": paired[i], "start": loc[0] if loc else -1,
                              "end": loc[1] if loc else -1,
                              "grounded": "yes" if grounded else "no",
                              "span_pairing": how if paired[i] else ""})
    out = os.path.join(args.out, "predictions.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        cols = ["docid", "hpo_id", "predicted_name", "predicted_span", "start", "end", "grounded",
                "span_pairing"]
        w.writerow(cols)
        w.writerows([["" if p[c] == -1 else p[c] for c in cols] for p in preds])
    n_ground = sum(p["grounded"] == "yes" for p in preds)
    print(f"Collected {len(preds)} predictions from {len(scored_docs)} documents "
          f"({n_ground} grounded to HPO, {len(preds) - n_ground} ungrounded); "
          f"{misaligned} span/term lists of different lengths, {unlocated} quotations not found in the text"
          + (f" ({fuzzy_located} more located by --fuzzy-locate)" if getattr(args, "fuzzy_locate", False) else ""))
    if obsolete_converted:
        print(f"{obsolete_converted} OntoGPT ids were obsolete and converted to the current HPO term")
    print(f"Wrote {out}")
    return preds, scored_docs


# --------------------------------------------------------------------------- evaluate

def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def evaluate(gold: list[dict], preds: list[dict], docs: set, args) -> dict:
    """--negated exclude: negated gold terms are dropped from the gold standard.
    --negated include: all gold terms count. All predictions are always scored, and a
    system's own negation status is never used."""
    include = args.negated == "include"
    G, P = defaultdict(list), defaultdict(list)
    for g in gold:
        if g["docid"] in docs and (include or not g["negated"]):
            G[g["docid"]].append(g)
    for p in preds:
        if p["docid"] in docs and not (args.drop_ungrounded and p.get("grounded") == "no"):
            P[p["docid"]].append(p)

    exact = getattr(args, "mention_match", "exact") == "exact"
    doc_c, men_c = [0, 0, 0], [0, 0, 0]
    errors = []
    for d in sorted(docs):
        g, p = G.get(d, []), P.get(d, [])
        # Document level: presence/absence of each concept id in the text, positions ignored.
        gs, ps = {x["hpo_id"] for x in g}, {x["hpo_id"] for x in p}
        doc_c[0] += len(gs & ps); doc_c[1] += len(ps - gs); doc_c[2] += len(gs - ps)
        for hid in sorted(gs - ps):
            errors.append([d, "document", "FN", hid, "", "", "; ".join(x["text"] for x in g if x["hpo_id"] == hid)])
        for hid in sorted(ps - gs):
            errors.append([d, "document", "FP", hid, "", "", "; ".join(
                f'{x["predicted_name"]} | {x["predicted_span"]}' for x in p if x["hpo_id"] == hid)])
        # Mention level: each mention = (concept id, start, end). A prediction is correct only
        # if it has the same id and the same character boundaries as a gold mention
        # (--mention-match overlap: overlapping boundaries, matched one-to-one).
        gm = sorted({(x["hpo_id"], x["start"], x["end"]) for x in g})
        pm = sorted({(x["hpo_id"], x["start"], x["end"]) for x in p})
        if exact:
            hit = set(gm) & {m for m in pm if m[1] >= 0}
            tp_p, tp_g = hit, hit
        else:
            tp_p, tp_g = set(), set()
            for m in sorted(pm, key=lambda m: m[1]):
                if m[1] < 0:
                    continue
                cands = [x for x in gm if x not in tp_g and x[0] == m[0] and x[1] >= 0
                         and min(x[2], m[2]) - max(x[1], m[1]) > 0]
                if cands:
                    tp_g.add(max(cands, key=lambda x: min(x[2], m[2]) - max(x[1], m[1])))
                    tp_p.add(m)
        men_c[0] += len(tp_p); men_c[1] += len(pm) - len(tp_p); men_c[2] += len(gm) - len(tp_g)
        text_of = lambda m: next((x.get("text") or x.get("predicted_span", "") for x in g + p
                                  if (x["hpo_id"], x["start"], x["end"]) == m), "")
        errors += [[d, "mention", "FN", m[0], m[1], m[2], text_of(m)] for m in gm if m not in tp_g]
        errors += [[d, "mention", "FP", m[0], "" if m[1] < 0 else m[1], "" if m[1] < 0 else m[2], text_of(m)]
                   for m in pm if m not in tp_p]

    metrics = {"negated": args.negated, "mention_match": "exact" if exact else "overlap", "documents": len(docs),
               "gold_mentions": sum(len(v) for v in G.values()),
               "predicted_mentions": sum(len(v) for v in P.values()),
               "document_level": prf(*doc_c), "mention_level": prf(*men_c)}
    tag = args.negated + ("_grounded" if args.drop_ungrounded else "")
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(args.out, f"errors_{tag}.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["docid", "level", "type", "hpo_id", "start", "end", "text"])
        w.writerows(errors)

    print(f"\nScored {len(docs)} documents (negated gold terms: {args.negated}, "
          f"mention boundaries: {metrics['mention_match']}); {metrics['gold_mentions']} gold mentions")
    print(f"{'':16}{'P':>8}{'R':>8}{'F1':>8}{'TP':>7}{'FP':>7}{'FN':>7}")
    for name, key in (("document level", "document_level"), ("mention level", "mention_level")):
        m = metrics[key]
        print(f"{name:16}{m['precision']:8.3f}{m['recall']:8.3f}{m['f1']:8.3f}{m['tp']:7}{m['fp']:7}{m['fn']:7}")
    print(f"Wrote metrics_{tag}.json and errors_{tag}.tsv in {args.out}")
    return metrics


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus TSV (StringID, Text, Term, Negated, Spans)")
    ap.add_argument("--out", required=True, help="folder for texts, OntoGPT outputs, predictions and scores")
    ap.add_argument("--template", default="phenotype_mentions.yaml", help="OntoGPT template (.yaml)")
    ap.add_argument("--model", default="gpt-4o", help="model name as OntoGPT expects it")
    ap.add_argument("--hpo", help="hp.obo, to map alternative ids to primary ids (recommended)")
    ap.add_argument("--negated", choices=("exclude", "include"), default="exclude",
                    help="negated gold terms: exclude (default; dropped from the gold standard) "
                         "or include them in the evaluation")
    ap.add_argument("--mention-match", choices=("exact", "overlap"), default="exact",
                    help="mention level: exact boundaries (default) or overlapping boundaries")
    ap.add_argument("--drop-ungrounded", action="store_true",
                    help="do not count predictions OntoGPT could not ground to HPO as false positives")
    ap.add_argument("--limit", type=int, help="only the first N documents (trial run)")
    ap.add_argument("--ids", help="file with one StringID per line: only these documents")
    ap.add_argument("--workers", type=int, default=1, help="parallel OntoGPT processes (default 1)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds allowed per document (default 600)")
    ap.add_argument("--plain-ontogpt", action="store_true", help="call `ontogpt` instead of ontogpt_fixed.py")
    ap.add_argument("--extra", default="", help='extra options for `ontogpt extract`, e.g. "--temperature 0"')
    ap.add_argument("--collect-only", action="store_true", help="no model calls; rebuild predictions and scores")
    ap.add_argument("--fuzzy-locate", action="store_true",
                    help="locate quotations the LLM did not copy verbatim: words in order with up to "
                         f"{FUZZY_MAX_GAP} words skipped, one-letter typos (exact matches still come first)")
    args = ap.parse_args()
    if not args.plain_ontogpt and not args.collect_only and not os.path.exists(FIXED_ONTOGPT):
        raise SystemExit(f"{FIXED_ONTOGPT} not found: keep ontogpt_fixed.py next to this script")
    args.template = os.path.abspath(args.template)

    hpo = load_hpo(args.hpo) if args.hpo else None
    texts, gold = load_corpus(args.corpus, hpo)
    docs = [{"docid": k, "file": doc_filename(k)} for k, t in texts.items() if t.strip()]
    if args.ids:
        with open(args.ids, encoding="utf-8") as fh:
            wanted = {ln.strip() for ln in fh if ln.strip()}
        docs = [d for d in docs if d["docid"] in wanted]
    if args.limit:
        docs = docs[:args.limit]
    print(f"{len(docs)} documents, {sum(g['docid'] in {d['docid'] for d in docs} for g in gold)} gold "
          f"annotations ({sum(g['negated'] for g in gold)} negated in the whole corpus)")

    os.makedirs(os.path.join(args.out, "texts"), exist_ok=True)
    for d in docs:
        path = os.path.join(args.out, "texts", d["file"] + ".txt")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(texts[d["docid"]])
    with open(os.path.join(args.out, "index.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["docid", "file"])
        w.writerows([[d["docid"], d["file"]] for d in docs])
    with open(os.path.join(args.out, "gold.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["docid", "hpo_id", "start", "end", "text", "negated", "spans"])
        w.writerows([[g["docid"], g["hpo_id"], g["start"], g["end"], g["text"], "X" if g["negated"] else "",
                      g["spans"]] for g in gold])

    if not args.collect_only:
        failures, todo = [], docs
        if args.workers > 1 and todo:          # first call installs the template; do it alone
            d, st = run_one(todo[0], args)
            if st not in ("done", "skipped"):
                failures.append((d, st))
            todo = todo[1:]
        done, start = 0, time.time()
        print(f"Running OntoGPT on {len(todo)} documents", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(run_one, d, args) for d in todo]
            for fut in as_completed(futures):
                d, st = fut.result()
                done += 1
                if st not in ("done", "skipped"):
                    failures.append((d, st))
                shown = st if st in ("done", "skipped") else "FAILED: " + st[:120]
                print(f"  [{done}/{len(futures)}, {time.time() - start:.0f}s] {d['docid']}: {shown}", flush=True)
        if failures and args.workers > 1:
            print(f"Retrying {len(failures)} failed documents ...")
            failures = [(d, st) for d, st in (run_one(d, args) for d, _ in failures) if st not in ("done", "skipped")]
        if failures:
            with open(os.path.join(args.out, "failures.tsv"), "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, delimiter="\t")
                w.writerow(["docid", "error"])
                w.writerows([[d["docid"], st] for d, st in failures])
            print(f"{len(failures)} documents failed; see failures.tsv. First error: {failures[0][1]}")

    preds, scored = collect(docs, texts, hpo, args)
    if not scored:
        print("No OntoGPT outputs yet; nothing to score.")
        return 0
    if len(scored) < len(docs):
        print(f"note: {len(docs) - len(scored)} documents have no output and are left out of the scores")
    evaluate(gold, preds, scored, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
