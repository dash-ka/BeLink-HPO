#!/usr/bin/env python3
"""Evaluate HPO concept recognisers on the Clinical Genetics Physical Exam corpus
(one TSV: StringID, Text, Term, Negated, Spans) with the same gold standard and
the same scorer as run_ontogpt_mentions.py, at two levels:
  document level  presence/absence of each HPO id in the text; positions ignored
  mention level   each mention with its position: same HPO id and the same
                  character boundaries as a gold mention (boundary detection counts;
                  --mention-match overlap accepts overlapping boundaries)

Systems
  fasthpocr    FastHPOCR (github.com/tudorgroza/fast_hpo_cr), run in-process with its
               default configuration. The HPO index is built from --hpo on first use
               (restricted to Phenotypic abnormality, as FastHPOCR does by default).
               Variants are separate systems in --systems, e.g.
                 fasthpocr,fasthpocr:longest,fasthpocr:longest+acronyms
               options: longest   annotate(longestMatch=True): drop matches nested in a longer one
                        acronyms  index 3-letter acronyms (allow3LetterAcronyms)
                        dup       keep labels shared by several concepts (allowDuplicateEntries)
                        syn       add synonyms from --fasthpocr-synonyms (externalSynFile;
                                  lines "HP:0001250=fits")
               Each variant is a row "fasthpocr (longest, acronyms)" with folder
               fasthpocr_longest_acronyms; indexes are shared in <out>/fasthpocr_index/.
  phenotagger  PhenoTagger (github.com/ncbi-nlp/PhenoTagger), run as a subprocess
               inside its own Python environment (TensorFlow 2.12 etc.). Its NegBio
               negation step is not used (recognition only), so NegBio, Java and
               their packages need not be installed.
  ncbo         NCBO BioPortal Annotator REST API, ontology HP. Needs an API key
               (--ncbo-apikey or the NCBO_API_KEY environment variable).
  ontogpt      Not run here: --ontogpt <predictions.tsv> from run_ontogpt_mentions.py
               is rescored against the same gold so it appears in the summary.
               Given the output of relink_mentions.py, it is scored twice:
               "ontogpt" with OntoGPT's own grounding (ontogpt_id column) and
               "ontogpt+belink (fill)" with BeLink filling the mentions OntoGPT could
               not ground (hpo_id column), with a line on how many mentions each grounded.
               --ontogpt-all <file> adds the output of relink_mentions.py --mode all
               (BeLink grounds every mention) as a third row, "ontogpt+belink (all)".
               Their folders are <out>/ontogpt, <out>/ontogpt_belink_fill, <out>/ontogpt_belink_all.
               With --ontogpt-all, the script also compares, mention by mention, the id
               OntoGPT gave with the id BeLink gave, and checks both against the gold
               standard: belink_changes_<negated>.tsv (counts) and
               belink_changes_<negated>_detail.tsv (every mention where they differ).

Paired comparison (--compare A.tsv B.tsv)
  Two OntoGPT prediction files over the same mentions, e.g. relink_mentions.py outputs
  of two BeLink settings (hpo_id column is scored). Reports, for document level and
  mention level (--mention-match), F1 of each, the difference B - A, a 95% bootstrap
  confidence interval and two-sided p-value (documents resampled, --bootstrap times,
  --seed), and McNemar's exact test on mention-level ids (an id is right when its
  concept is in the observation's gold). Writes <out>/compare_<A>_vs_<B>_<mode>.tsv.
  Can run alone: --systems "" with no --ontogpt.

Outputs, per system, in <out>/<system>/
  raw outputs (cached: rerunning reuses them; --rerun forces a fresh run),
  predictions_<mode>.tsv, metrics_<mode>.json, errors_<mode>.tsv
and <out>/summary_<mode>.tsv comparing all systems.

Negated gold terms (Negated == X, e.g. "Normal lips" -> HP:0000159, "no clinodactyly")
The evaluation measures recognition only, never present/negated classification:
  --negated exclude  (default) negated gold terms are dropped from the gold
                     standard; all predictions are still scored.
  --negated include  negated gold terms are scored like any other: a system must
                     recognise them too.
Whether a system itself marks a mention as negated is never used.

Predictions outside Phenotypic abnormality (HP:0000118), e.g. modes of
inheritance, onset or modifiers such as "mild", are dropped for every system
unless --all-hpo-branches is given.

Usage
  python eval_baselines.py --corpus Clinical-Genetics-Physical-Exam-Data-Dev.tsv \
      --hpo hp.obo --out work/baselines --systems fasthpocr,phenotagger,ncbo \
      --phenotagger-dir ~/PhenoTagger --phenotagger-python ~/envs/pt/bin/python \
      --ncbo-apikey XXXX --ontogpt work/biocgs/predictions.tsv
  # rescore including negated gold terms, no system calls
  python eval_baselines.py ... --negated include

Keep run_ontogpt_mentions.py next to this script. FastHPOCR needs `pip install
FastHPOCR` (or --fasthpocr-src pointing at the repo's pypi/FastHPOCR folder) and
pronto.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ontogpt_mentions import consolidate, evaluate, load_corpus  # noqa: E402

PHENOTYPIC_ABNORMALITY = "HP:0000118"
SYSTEMS = ("fasthpocr", "phenotagger", "ncbo")
FILL, ALL = "ontogpt+belink (fill)", "ontogpt+belink (all)"      # row labels
FOLDERS = {FILL: "ontogpt_belink_fill", ALL: "ontogpt_belink_all"}  # their output folders


# --------------------------------------------------------------------------- HPO

def load_hpo_full(path: str) -> dict:
    """OBO reader: ids, names, is_a parents, alt_id -> current id, obsolete ids and their
    replaced_by (applied to gold and predictions). Works with run_ontogpt_mentions.consolidate()."""
    terms, names, parents, alt = set(), {}, {}, {}
    obsolete_ids, replaced_map = set(), {}
    cur, in_term, obsolete, replaced = None, False, False, None

    def close():
        if cur and obsolete:
            terms.discard(cur)
            obsolete_ids.add(cur)
            if replaced:
                replaced_map[cur] = replaced
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                close()
                in_term, cur, obsolete, replaced = line == "[Term]", None, False, None
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                cur = line[4:].strip()
                terms.add(cur)
                parents[cur] = []
            elif not cur:
                continue
            elif line.startswith("name: "):
                names[cur] = line[6:].strip()
            elif line.startswith("is_a: "):
                parents[cur].append(line[6:].split("!")[0].strip())
            elif line.startswith("alt_id: "):
                alt[line[8:].strip()] = cur
            elif line == "is_obsolete: true":
                obsolete = True
            elif line.startswith("replaced_by: "):
                replaced = line[13:].strip()
    close()
    # descendants of Phenotypic abnormality
    children: dict[str, list] = {}
    for c, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(c)
    abn, stack = set(), [PHENOTYPIC_ABNORMALITY]
    while stack:
        t = stack.pop()
        if t not in abn:
            abn.add(t)
            stack.extend(children.get(t, []))
    return {"terms": terms, "alt": alt, "names": names, "abnormality": abn,
            "obsolete": obsolete_ids, "replaced": replaced_map}


def norm_id(raw: str) -> str:
    """'http://purl.obolibrary.org/obo/HP_0001250', 'HP_0001250', 'hp:0001250' -> 'HP:0001250'."""
    s = str(raw).strip().rsplit("/", 1)[-1]
    m = re.fullmatch(r"(?i)hp[_:](\d{7})", s)
    return f"HP:{m.group(1)}" if m else s


# --------------------------------------------------------------------------- systems

FASTHPOCR_OPTS = ("longest", "acronyms", "dup", "syn")


def parse_fasthpocr(spec: str) -> list[str]:
    """'fasthpocr' -> [], 'fasthpocr:longest+acronyms' -> ['longest', 'acronyms'] (canonical order)."""
    opts = [o.strip().lower() for o in spec.split(":", 1)[1].split("+")] if ":" in spec else []
    bad = [o for o in opts if o not in FASTHPOCR_OPTS]
    if bad:
        raise SystemExit(f"unknown FastHPOCR option(s) {bad} in {spec!r}; choose from {', '.join(FASTHPOCR_OPTS)}")
    return [o for o in FASTHPOCR_OPTS if o in opts]


def run_fasthpocr(docs, texts, hpo, sysdir, args, opts=()) -> list[dict]:
    """opts: subset of FASTHPOCR_OPTS; [] is FastHPOCR's default configuration."""
    index_opts = [o for o in opts if o != "longest"]           # options that change the index
    cfg = "-".join(opts) or "default"
    raw_path = os.path.join(sysdir, f"raw_{cfg}.jsonl")         # cache is specific to the configuration
    if args.rerun or not os.path.exists(raw_path):
        if args.fasthpocr_src:
            sys.path.insert(0, os.path.abspath(args.fasthpocr_src))
        try:
            from FastHPOCR.HPOAnnotator import HPOAnnotator
        except ImportError as e:
            raise SystemExit(f"FastHPOCR not importable ({e}): pip install FastHPOCR pronto, "
                             f"or pass --fasthpocr-src <repo>/pypi/FastHPOCR")
        if "syn" in opts and not (args.fasthpocr_synonyms and os.path.exists(args.fasthpocr_synonyms)):
            raise SystemExit("fasthpocr:syn needs --fasthpocr-synonyms <file> (lines 'HP:0001250=fits')")
        index = args.fasthpocr_index if (args.fasthpocr_index and not index_opts) else None
        if not index:
            idx_dir = os.path.join(args.out, "fasthpocr_index", "-".join(index_opts) or "default")
            found = os.listdir(idx_dir) if os.path.isdir(idx_dir) else []
            if not found:
                from FastHPOCR.IndexHPO import IndexHPO
                os.makedirs(idx_dir, exist_ok=True)
                config = {"rootConcepts": [PHENOTYPIC_ABNORMALITY],
                          "allow3LetterAcronyms": "acronyms" in opts,
                          "includeTopLevelCategory": False,
                          "allowDuplicateEntries": "dup" in opts,
                          "compressIndex": False}
                if "syn" in opts:
                    config["externalSynFile"] = os.path.abspath(args.fasthpocr_synonyms)
                print(f"FastHPOCR: building HPO index ({'-'.join(index_opts) or 'default'}) from {args.hpo} "
                      f"(15-30 min, once) ...", flush=True)
                IndexHPO(os.path.abspath(args.hpo), idx_dir, indexConfig=config).index()
                found = os.listdir(idx_dir)
            index = os.path.join(idx_dir, sorted(found)[0])
        annotator = HPOAnnotator(index)
        t0 = time.time()
        with open(raw_path + ".part", "w", encoding="utf-8") as fh:
            for d in docs:
                anns = annotator.annotate(texts[d], longestMatch="longest" in opts)
                fh.write(json.dumps({"docid": d, "annotations": [
                    {"id": a.getHPOUri(), "label": a.getHPOLabel(), "text": a.getTextSpan(),
                     "start": a.getStartOffset(), "end": a.getEndOffset()} for a in anns]}) + "\n")
        os.replace(raw_path + ".part", raw_path)
        print(f"FastHPOCR ({cfg}): {len(docs)} documents in {time.time() - t0:.1f}s")
    preds = []
    with open(raw_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            for a in r["annotations"]:
                preds.append(mk_pred(r["docid"], a["id"], a.get("label"), a["start"], a["end"], texts, hpo))
    return preds


def run_phenotagger(docs, texts, hpo, sysdir, args) -> list[dict]:
    if not args.phenotagger_dir:
        raise SystemExit("phenotagger: pass --phenotagger-dir (the PhenoTagger repository folder)")
    src = os.path.join(os.path.abspath(args.phenotagger_dir), "src")
    if not os.path.exists(os.path.join(src, "PhenoTagger_tagging.py")):
        raise SystemExit(f"phenotagger: {src}/PhenoTagger_tagging.py not found")
    # absolute paths: PhenoTagger runs from inside its own src/ folder
    in_dir = os.path.join(os.path.abspath(sysdir), "input") + "/"
    out_dir = os.path.join(os.path.abspath(sysdir), "output") + "/"
    result = out_dir + "corpus.PubTator"
    if args.rerun or not os.path.exists(result):
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(in_dir):
            os.remove(in_dir + f)
        # PubTator needs numeric ids; the tagger reads title + " " + abstract, so the whole
        # text goes in the title and offsets are those of the original text. Line breaks
        # and tabs become spaces (same length).
        with open(in_dir + "corpus.PubTator", "w", encoding="utf-8") as fh:
            for i, d in enumerate(docs, 1):
                t = re.sub(r"[\r\n\t|]", " ", texts[d])
                fh.write(f"{i}|t|{t}\n{i}|a|\n\n")
        with open(os.path.join(sysdir, "id_map.tsv"), "w", encoding="utf-8") as fh:
            fh.writelines(f"{i}\t{d}\n" for i, d in enumerate(docs, 1))
        para = {"model_type": args.pt_model, "onlyLongest": args.pt_only_longest, "abbrRecog": not args.pt_no_abbr,
                "negation": False, "ML_Threshold": args.pt_threshold}
        # PhenoTagger_tagging.py imports NegBio (Java, bllipparser, JPype) on startup even
        # though negation is off; a stand-in module is put in its place so PhenoTagger
        # runs without any of NegBio's dependencies installed.
        stub = ("import types\n_m=types.ModuleType('negbio2.negbio_run')\n"
                "_m.negbio_load=_m.negbio_main=lambda *a,**k: None\n"
                "sys.modules.setdefault('negbio2', types.ModuleType('negbio2'))\n"
                "sys.modules['negbio2.negbio_run']=_m\n")
        code = ("import sys\nsys.argv=['PhenoTagger_tagging.py']\n" + stub +
                "from PhenoTagger_tagging import phenotagger_tag\n"
                f"phenotagger_tag({in_dir!r}, {para!r}, {out_dir!r})\n")
        log = os.path.join(sysdir, "phenotagger.log")
        print(f"PhenoTagger: tagging {len(docs)} documents with {args.pt_model} (log: {log}) ...", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as lf:
            # PhenoTagger sets TensorFlow memory growth on the first GPU only, which fails
            # ("Memory growth cannot differ between GPU devices") when several GPUs are
            # visible, so it is shown a single GPU (or none, for CPU).
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if args.pt_gpu == "cpu" else args.pt_gpu)
            p = subprocess.run([args.phenotagger_python, "-c", code], cwd=src, stdout=lf, stderr=subprocess.STDOUT,
                               env=env)
        if p.returncode != 0 or not os.path.exists(result):
            with open(log, encoding="utf-8", errors="replace") as lf:
                tail = lf.read()[-1500:]
            raise SystemExit(f"PhenoTagger failed (exit {p.returncode}); end of {log}:\n{tail}")
        print(f"PhenoTagger: done in {time.time() - t0:.0f}s")
    id_map = {}
    with open(os.path.join(sysdir, "id_map.tsv"), encoding="utf-8") as fh:
        for line in fh:
            i, d = line.rstrip("\n").split("\t", 1)
            id_map[i] = d
    preds = []
    with open(result, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or f[0] not in id_map:
                continue
            d, s, e = id_map[f[0]], int(f[1]), int(f[2])
            for hid in re.split(r"[;,]", f[5]):
                if hid.strip():
                    p = mk_pred(d, hid, None, s, e, texts, hpo)
                    p["score"] = f[6] if len(f) > 6 else ""
                    preds.append(p)
    return preds


def ncbo_call(text: str, args) -> list:
    data = {"text": text, "ontologies": "HP", "whole_word_only": "true",
            "longest_only": str(args.ncbo_longest_only).lower(),
            "exclude_synonyms": str(args.ncbo_exclude_synonyms).lower(),
            "exclude_numbers": "false", "expand_mappings": "false", "display_links": "false",
            "display_context": "false"}
    req = urllib.request.Request(args.ncbo_url, data=urllib.parse.urlencode(data).encode(),
                                 headers={"Authorization": f"apikey token={args.ncbo_apikey}",
                                          "Accept": "application/json"})
    delay = 2.0
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(f"NCBO: HTTP {e.code} - check the API key")
            if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(delay)
        delay *= 2
    return []


def run_ncbo(docs, texts, hpo, sysdir, args) -> list[dict]:
    cache = os.path.join(sysdir, "raw")
    os.makedirs(cache, exist_ok=True)
    from run_ontogpt_mentions import doc_filename
    files = {d: os.path.join(cache, doc_filename(d) + ".json") for d in docs}
    todo = [d for d in docs if args.rerun or not os.path.exists(files[d])]
    if todo:
        if not args.ncbo_apikey:
            raise SystemExit("ncbo: pass --ncbo-apikey or set NCBO_API_KEY (free account at bioportal.bioontology.org)")
        print(f"NCBO: annotating {len(todo)} documents ({len(docs) - len(todo)} cached) ...", flush=True)
        failures, t0 = [], time.time()

        def one(d):
            res = ncbo_call(texts[d], args)
            with open(files[d] + ".part", "w", encoding="utf-8") as fh:
                json.dump(res, fh)
            os.replace(files[d] + ".part", files[d])
        with ThreadPoolExecutor(max_workers=args.ncbo_workers) as pool:
            futs = {pool.submit(one, d): d for d in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    fut.result()
                except SystemExit:
                    raise
                except Exception as e:  # keep going, report at the end
                    failures.append((futs[fut], repr(e)[:200]))
                if n % 50 == 0 or n == len(todo):
                    print(f"  [{n}/{len(todo)}, {time.time() - t0:.0f}s]", flush=True)
        if failures:
            print(f"NCBO: {len(failures)} documents failed (rerun to retry); first: {failures[0]}")
    preds, shifted = [], 0
    for d in docs:
        if not os.path.exists(files[d]):
            continue
        text = texts[d]
        for item in json.load(open(files[d], encoding="utf-8")):
            hid = (item.get("annotatedClass") or {}).get("@id", "")
            for a in item.get("annotations") or []:
                # BioPortal offsets are 1-based and inclusive; verify against the text.
                s, e, want = int(a["from"]) - 1, int(a["to"]), str(a.get("text", "")).lower()
                if want and text[s:e].lower() != want:
                    shifted += 1
                    hit = text.lower().find(want, max(0, s - 5))
                    s, e = (hit, hit + len(want)) if hit >= 0 else (s, e)
                preds.append(mk_pred(d, hid, None, s, e, texts, hpo))
    if shifted:
        print(f"NCBO: {shifted} offsets did not match the text and were realigned by search")
    return preds


def load_ontogpt(path, docs, texts, hpo, idcol="hpo_id") -> list[dict]:
    keep, preds = set(docs), []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["docid"] not in keep:
                continue
            p = mk_pred(r["docid"], r.get(idcol) or r["hpo_id"], r.get("predicted_name"),
                        int(r["start"]) if r.get("start") else -1, int(r["end"]) if r.get("end") else -1,
                        texts, hpo)
            p["predicted_span"] = r.get("predicted_span") or p["predicted_span"]
            preds.append(p)
    return preds


def mk_pred(docid, raw_id, label, start, end, texts, hpo) -> dict:
    hid = norm_id(raw_id)
    grounded = hid.startswith("HP:")
    from_obsolete = grounded and hid in hpo.get("obsolete", ())
    if grounded:
        hid = consolidate(hid, hpo) or hid
    span = texts[docid][start:end] if 0 <= start < end else ""
    return {"docid": docid, "hpo_id": hid, "predicted_name": label or hpo["names"].get(hid, ""),
            "predicted_span": span, "start": start, "end": end, "grounded": "yes" if grounded else "no",
            "from_obsolete": from_obsolete and hid not in hpo.get("obsolete", ())}


# --------------------------------------------------------------------------- OntoGPT vs BeLink ids

def compare_ids(path, docs, gold, hpo, args) -> None:
    """Mention by mention, compare the id OntoGPT gave (ontogpt_id) with the id BeLink gave
    in --mode all (hpo_id), and check each against the gold standard.

    An id counts as correct when the gold standard has that HPO concept in the same
    observation (as in document-level scoring), so boundary mismatches play no part.
    Ids outside Phenotypic abnormality count as wrong unless --all-hpo-branches.
    Gold follows --negated."""
    include = args.negated == "include"
    gold_ids: dict[str, set] = {}
    for g in gold:
        if g["docid"] in docs and (include or not g["negated"]):
            gold_ids.setdefault(g["docid"], set()).add(g["hpo_id"])

    def norm(raw):
        hid = norm_id(raw or "")
        if not hid.startswith("HP:"):
            return None
        hid = consolidate(hid, hpo) or hid
        if not args.all_hpo_branches and hid not in hpo["abnormality"]:
            return hid, False
        return hid, True

    def correct(docid, n):
        return bool(n and n[1] and n[0] in gold_ids.get(docid, set()))

    c = Counter()
    detail = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            d = r["docid"]
            if d not in docs:
                continue
            o, b = norm(r.get("ontogpt_id")), norm(r.get("hpo_id"))
            oc, bc = correct(d, o), correct(d, b)
            if o is None and b is None:
                c["neither grounded"] += 1
                continue
            if o is None:                                   # BeLink grounded what OntoGPT could not
                c["new: right" if bc else "new: wrong"] += 1
                continue
            if b is None:                                   # BeLink declined an OntoGPT id
                c["declined: ontogpt right" if oc else "declined: ontogpt wrong"] += 1
                kind = "declined"
            elif o[0] == b[0]:
                c["same: right" if oc else "same: wrong"] += 1
                continue
            else:
                c[f"changed: ontogpt {'right' if oc else 'wrong'}, belink {'right' if bc else 'wrong'}"] += 1
                kind = "changed"
            detail.append([d, kind, r.get("predicted_span", ""), r.get("start", ""), r.get("end", ""),
                           o[0], hpo["names"].get(o[0], r.get("predicted_name", "")), "yes" if oc else "no",
                           b[0] if b else "", (hpo["names"].get(b[0], r.get("linker_label", "")) if b else ""),
                           ("yes" if bc else "no") if b else "",
                           r.get("linker_score", "")])

    o_grounded = sum(v for k, v in c.items() if not k.startswith(("new", "neither")))
    changed = sum(v for k, v in c.items() if k.startswith("changed"))
    declined = c["declined: ontogpt right"] + c["declined: ontogpt wrong"]
    same = c["same: right"] + c["same: wrong"]
    new = c["new: right"] + c["new: wrong"]
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"
    rows = [
        ["mentions OntoGPT grounded", o_grounded, ""],
        ["  BeLink gave the same id", same, pct(same, o_grounded)],
        ["    right", c["same: right"], pct(c["same: right"], same)],
        ["    wrong", c["same: wrong"], pct(c["same: wrong"], same)],
        ["  BeLink changed the id", changed, pct(changed, o_grounded)],
        ["    OntoGPT wrong, BeLink right (fixed)", c["changed: ontogpt wrong, belink right"],
         pct(c["changed: ontogpt wrong, belink right"], changed)],
        ["    OntoGPT right, BeLink wrong (broken)", c["changed: ontogpt right, belink wrong"],
         pct(c["changed: ontogpt right, belink wrong"], changed)],
        ["    both wrong", c["changed: ontogpt wrong, belink wrong"],
         pct(c["changed: ontogpt wrong, belink wrong"], changed)],
        ["    both right", c["changed: ontogpt right, belink right"],
         pct(c["changed: ontogpt right, belink right"], changed)],
        ["  BeLink declined (no id)", declined, pct(declined, o_grounded)],
        ["    OntoGPT's id was right (lost)", c["declined: ontogpt right"], pct(c["declined: ontogpt right"], declined)],
        ["    OntoGPT's id was wrong (removed)", c["declined: ontogpt wrong"], pct(c["declined: ontogpt wrong"], declined)],
        ["mentions OntoGPT could not ground, BeLink grounded", new, ""],
        ["    right", c["new: right"], pct(c["new: right"], new)],
        ["    wrong", c["new: wrong"], pct(c["new: wrong"], new)],
        ["mentions neither grounded", c["neither grounded"], ""],
    ]
    net = (c["changed: ontogpt wrong, belink right"] - c["changed: ontogpt right, belink wrong"]
           - c["declined: ontogpt right"])
    print(f"\n=== OntoGPT's ids vs BeLink (all), checked against gold (negated gold terms: {args.negated}; "
          f"an id is right when the concept is in the observation's gold)")
    for name, n, share in rows:
        print(f"{name:52}{n:>7}{share:>9}")
    print(f"{'net change (fixed - broken - lost)':52}{net:>+7}")
    base = os.path.join(args.out, f"belink_changes_{args.negated}")
    with open(base + ".tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["category", "mentions", "share"])
        w.writerows([[n.strip(), k, sh] for n, k, sh in rows])
        w.writerow(["net on OntoGPT-grounded mentions (fixed - broken - lost)", net, ""])
    with open(base + "_detail.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["docid", "kind", "span", "start", "end", "ontogpt_id", "ontogpt_label", "ontogpt_right",
                    "belink_id", "belink_label", "belink_right", "belink_score"])
        w.writerows(detail)
    print(f"Wrote {base}.tsv and {base}_detail.tsv")


# --------------------------------------------------------------------------- paired comparison

def doc_counts(gold, preds, docs, args) -> dict:
    """Per document [doc tp, fp, fn, mention tp, fp, fn], with exactly the rules of
    run_ontogpt_mentions.evaluate() (its totals are checked against these sums)."""
    include = args.negated == "include"
    exact = args.mention_match == "exact"
    G, P = {}, {}
    for g in gold:
        if g["docid"] in docs and (include or not g["negated"]):
            G.setdefault(g["docid"], []).append(g)
    for p in preds:
        if p["docid"] in docs:
            P.setdefault(p["docid"], []).append(p)
    out = {}
    for d in docs:
        g, p = G.get(d, []), P.get(d, [])
        gs, ps = {x["hpo_id"] for x in g}, {x["hpo_id"] for x in p}
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
        out[d] = [len(gs & ps), len(ps - gs), len(gs - ps), len(tp_p), len(pm) - len(tp_p), len(gm) - len(tp_g)]
    return out


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: discordant pairs b, c ~ Binomial(b + c, 0.5)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare_files(path_a, path_b, docs, texts, gold, hpo, args) -> None:
    import random
    name = lambda p: re.sub(r"\.tsv$", "", os.path.basename(p))
    la, lb = name(path_a), name(path_b)
    docset = set(docs)
    pa, _ = finalize(load_ontogpt(path_a, docs, texts, hpo), hpo, args)
    pb, _ = finalize(load_ontogpt(path_b, docs, texts, hpo), hpo, args)
    if args.drop_ungrounded:
        # same rule as the main scoring: an AUTO: id (no HPO concept) is an abstention, not a prediction
        pa = [p for p in pa if p["grounded"] != "no"]
        pb = [p for p in pb if p["grounded"] != "no"]
    ca, cb = doc_counts(gold, pa, docset, args), doc_counts(gold, pb, docset, args)
    order = sorted(docset)

    # check against the scorer's own totals
    for preds, counts in ((pa, ca), (pb, cb)):
        quiet = SimpleNamespace(negated=args.negated, out=tempfile.mkdtemp(), mention_match=args.mention_match,
                                drop_ungrounded=False)
        with contextlib.redirect_stdout(io.StringIO()):
            m = evaluate(gold, preds, docset, quiet)
        tot = [sum(counts[d][i] for d in order) for i in range(6)]
        want = [m["document_level"][k] for k in ("tp", "fp", "fn")] + [m["mention_level"][k] for k in ("tp", "fp", "fn")]
        assert tot == want, f"per-document counts {tot} differ from the scorer's {want}"

    def f1s(counts, idx):
        t = [0] * 6
        for d in idx:
            for i, v in enumerate(counts[d]):
                t[i] += v
        return _f1(*t[:3]), _f1(*t[3:])

    obs_a, obs_b = f1s(ca, order), f1s(cb, order)
    rng = random.Random(args.seed)
    diffs = ([], [])
    n = len(order)
    for _ in range(args.bootstrap):
        idx = [order[rng.randrange(n)] for _ in range(n)]
        fa, fb = f1s(ca, idx), f1s(cb, idx)
        diffs[0].append(fb[0] - fa[0])
        diffs[1].append(fb[1] - fa[1])

    # McNemar on mention-level ids: rows paired by position in the two files (same mentions)
    gold_ids: dict[str, set] = {}
    include = args.negated == "include"
    for g in gold:
        if g["docid"] in docset and (include or not g["negated"]):
            gold_ids.setdefault(g["docid"], set()).add(g["hpo_id"])

    def rows(path):
        with open(path, encoding="utf-8", newline="") as fh:
            return [r for r in csv.DictReader(fh, delimiter="\t") if r["docid"] in docset]

    def right(r):
        hid = norm_id(r.get("hpo_id") or "")
        if not hid.startswith("HP:"):
            return False
        hid = consolidate(hid, hpo) or hid
        return (args.all_hpo_branches or hid in hpo["abnormality"]) and hid in gold_ids.get(r["docid"], set())

    ra, rb = rows(path_a), rows(path_b)
    key = lambda r: (r["docid"], r.get("start", ""), r.get("end", ""), r.get("predicted_span", ""))
    if len(ra) != len(rb) or any(key(x) != key(y) for x, y in zip(ra, rb)):
        raise SystemExit("--compare: the two files do not list the same mentions in the same order; "
                         "relink both from the same predictions.tsv")
    both = only_a = only_b = neither = 0
    for x, y in zip(ra, rb):
        a, b = right(x), right(y)
        both += a and b
        only_a += a and not b
        only_b += b and not a
        neither += not a and not b
    p_mcnemar = mcnemar_exact(only_a, only_b)

    def ci(v):
        s_ = sorted(v)
        return s_[int(0.025 * (len(s_) - 1))], s_[int(0.975 * (len(s_) - 1))]

    def pval(v):
        lo = sum(x <= 0 for x in v) / len(v)
        hi = sum(x >= 0 for x in v) / len(v)
        return min(1.0, 2 * min(lo, hi))

    table = []
    for i, lev in enumerate(("document", f"mention ({args.mention_match})")):
        d = obs_b[i] - obs_a[i]
        lo, hi = ci(diffs[i])
        table.append([lev, f"{obs_a[i]:.3f}", f"{obs_b[i]:.3f}", f"{d:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]",
                      (f"<{1 / args.bootstrap:.4f}" if pval(diffs[i]) == 0 else f"{pval(diffs[i]):.3f}")])
    print(f"\n=== Paired comparison: A = {la}, B = {lb} (negated gold terms: {args.negated}; "
          f"ungrounded predictions {'dropped' if args.drop_ungrounded else 'counted as FP'}; "
          f"{args.bootstrap} bootstrap samples over {n} documents, seed {args.seed})")
    print(f"{'level':22}{'F1 A':>8}{'F1 B':>8}{'B - A':>9}{'95% CI':>20}{'p':>8}")
    for r in table:
        print(f"{r[0]:22}{r[1]:>8}{r[2]:>8}{r[3]:>9}{r[4]:>20}{r[5]:>8}")
    print(f"mention ids (right = concept in the observation's gold), {len(ra)} mentions:")
    print(f"  both right {both}, only A right {only_a}, only B right {only_b}, both wrong {neither}; "
          f"McNemar exact p = {p_mcnemar:.3f}")
    out = os.path.join(args.out, f"compare_{la}_vs_{lb}_{args.negated}"
                                 f"{'_overlap' if args.mention_match == 'overlap' else ''}"
                                 f"{'_grounded' if args.drop_ungrounded else ''}.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["level", f"f1_{la}", f"f1_{lb}", "difference", "ci95", "p_bootstrap"])
        w.writerows(table)
        w.writerow([])
        w.writerow(["mention ids", "both right", f"only {la} right", f"only {lb} right", "both wrong", "p_mcnemar"])
        w.writerow(["", both, only_a, only_b, neither, f"{p_mcnemar:.4f}"])
    print(f"Wrote {out}")


# --------------------------------------------------------------------------- main

def finalize(preds, hpo, args) -> tuple[list[dict], int]:
    """Drop predictions outside Phenotypic abnormality (unless --all-hpo-branches). Obsolete ids
    were already converted to their current term (mk_pred); one without a replacement is kept as
    it is: it never matches the gold, so it counts as an error."""
    out, off_branch, n_obsolete = [], 0, 0
    n_converted = sum(1 for p in preds if p.get("from_obsolete"))
    for p in preds:
        if p["hpo_id"] in hpo.get("obsolete", ()):
            out.append(p)
            n_obsolete += 1
            continue
        if not args.all_hpo_branches and p["grounded"] == "yes" and p["hpo_id"] not in hpo["abnormality"]:
            off_branch += 1
            continue
        out.append(p)
    if n_converted:
        print(f"{n_converted} predictions with an obsolete HPO id converted to the current term")
    if n_obsolete:
        print(f"{n_obsolete} predictions with an obsolete HPO id without replacement (counted as errors)")
    return out, off_branch


def write_preds(preds, path):
    cols = ["docid", "hpo_id", "predicted_name", "predicted_span", "start", "end", "grounded", "score"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        w.writerows([["" if p.get(c, "") == -1 else p.get(c, "") for c in cols] for p in preds])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="corpus TSV (StringID, Text, Term, Negated, Spans)")
    ap.add_argument("--hpo", required=True, help="hp.obo (id mapping, labels, branch filter, FastHPOCR index)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--systems", default="fasthpocr,phenotagger,ncbo",
                    help=f"comma-separated: {','.join(SYSTEMS)}; FastHPOCR variants as fasthpocr:opt+opt "
                         f"with opt in {'/'.join(FASTHPOCR_OPTS)}")
    ap.add_argument("--ontogpt", help="OntoGPT predictions.tsv (from run_ontogpt_mentions.py or "
                                      "relink_mentions.py), rescored for the summary")
    ap.add_argument("--ontogpt-all", help="relink_mentions.py --mode all output (linker grounds every mention), "
                                          "scored as an extra row 'ontogpt+belink (all)'")
    ap.add_argument("--compare", nargs=2, metavar=("A.tsv", "B.tsv"),
                    help="paired comparison of two OntoGPT prediction files over the same mentions "
                         "(bootstrap CI and p-value for F1, McNemar on mention ids)")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap samples for --compare (default 10000)")
    ap.add_argument("--seed", type=int, default=13, help="random seed for --compare (default 13)")
    ap.add_argument("--negated", choices=("exclude", "include"), default="exclude",
                    help="negated gold terms: exclude (default; dropped from the gold standard) "
                         "or include them in the evaluation")
    ap.add_argument("--mention-match", choices=("exact", "overlap"), default="exact",
                    help="mention level: exact boundaries (default) or overlapping boundaries")
    ap.add_argument("--all-hpo-branches", action="store_true",
                    help="keep predictions outside Phenotypic abnormality (HP:0000118)")
    ap.add_argument("--drop-ungrounded", action="store_true", help="ignore OntoGPT predictions without an HPO id")
    ap.add_argument("--limit", type=int, help="only the first N documents")
    ap.add_argument("--ids", help="file with one StringID per line: only these documents")
    ap.add_argument("--rerun", action="store_true", help="ignore cached system outputs")
    g = ap.add_argument_group("FastHPOCR")
    g.add_argument("--fasthpocr-index", help="existing default-configuration index file "
                                             "(default: built in <out>/fasthpocr_index/default)")
    g.add_argument("--fasthpocr-src", help="folder containing the FastHPOCR package, if not pip-installed")
    g.add_argument("--fasthpocr-synonyms", help="extra synonyms for fasthpocr:syn, one 'HP:0001250=fits' per line")
    g = ap.add_argument_group("PhenoTagger")
    g.add_argument("--phenotagger-dir", help="PhenoTagger repository folder (with src/, dict/, models/)")
    g.add_argument("--phenotagger-python", default=sys.executable, help="python of PhenoTagger's environment")
    g.add_argument("--pt-model", default="bioformer", choices=("bioformer", "pubmedbert", "biobert", "cnn"))
    g.add_argument("--pt-threshold", type=float, default=0.95, help="ML threshold (default 0.95)")
    g.add_argument("--pt-only-longest", action="store_true", help="only the longest of overlapping concepts")
    g.add_argument("--pt-gpu", default="0",
                   help='GPU number PhenoTagger may use (default 0), or "cpu"')
    g.add_argument("--pt-no-abbr", action="store_true", help="switch off abbreviation recognition")
    g = ap.add_argument_group("NCBO Annotator")
    g.add_argument("--ncbo-apikey", default=os.environ.get("NCBO_API_KEY"), help="BioPortal API key")
    g.add_argument("--ncbo-url", default="https://data.bioontology.org/annotator")
    g.add_argument("--ncbo-longest-only", action="store_true")
    g.add_argument("--ncbo-exclude-synonyms", action="store_true")
    g.add_argument("--ncbo-workers", type=int, default=4, help="parallel requests (default 4)")
    args = ap.parse_args()

    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    bad = [s for s in systems if s.split(":", 1)[0] not in SYSTEMS or (":" in s and not s.startswith("fasthpocr:"))]
    if bad:
        raise SystemExit(f"unknown system(s): {', '.join(bad)} (choose from {', '.join(SYSTEMS)})")

    hpo = load_hpo_full(args.hpo)
    texts, gold = load_corpus(args.corpus, hpo)
    docs = [d for d, t in texts.items() if t.strip()]
    if args.ids:
        with open(args.ids, encoding="utf-8") as fh:
            wanted = {ln.strip() for ln in fh if ln.strip()}
        docs = [d for d in docs if d in wanted]
    if args.limit:
        docs = docs[:args.limit]
    docset = set(docs)
    g_sel = [x for x in gold if x["docid"] in docset]
    off = Counter(x["hpo_id"] for x in g_sel if x["hpo_id"] not in hpo["abnormality"])
    print(f"{len(docs)} documents, {len(g_sel)} gold annotations ({sum(x['negated'] for x in g_sel)} negated)")
    if off and not args.all_hpo_branches:
        print(f"note: {sum(off.values())} gold annotations lie outside Phenotypic abnormality "
              f"(e.g. {', '.join(list(off)[:5])}); predictions there are filtered, so these count as misses. "
              f"Use --all-hpo-branches to keep them.")
    os.makedirs(args.out, exist_ok=True)

    if args.compare:
        for pth in args.compare:
            if not os.path.exists(pth):
                raise SystemExit(f"--compare: {pth} not found")
        compare_files(args.compare[0], args.compare[1], docs, texts, gold, hpo, args)
        if not systems and not args.ontogpt and not args.ontogpt_all:
            return 0

    runners = {"fasthpocr": run_fasthpocr, "phenotagger": run_phenotagger, "ncbo": run_ncbo}
    results, grounding = {}, {}
    tag = args.negated + ("_overlap" if args.mention_match == "overlap" else "")
    ontogpt_runs = []          # (row name, file, id column)
    if args.ontogpt:
        if not os.path.exists(args.ontogpt):
            raise SystemExit(f"--ontogpt: {args.ontogpt} not found")
        with open(args.ontogpt, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
        if "ontogpt_id" in header:     # relinked: score OntoGPT's own grounding and the linker's
            ontogpt_runs = [("ontogpt", args.ontogpt, "ontogpt_id"), (FILL, args.ontogpt, "hpo_id")]
        else:
            ontogpt_runs = [("ontogpt", args.ontogpt, "hpo_id")]
    if args.ontogpt_all:
        if not os.path.exists(args.ontogpt_all):
            raise SystemExit(f"--ontogpt-all: {args.ontogpt_all} not found")
        with open(args.ontogpt_all, encoding="utf-8") as fh:
            if "ontogpt_id" not in fh.readline().split("\t"):
                raise SystemExit("--ontogpt-all: expected the output of relink_mentions.py --mode all")
        if not ontogpt_runs:
            ontogpt_runs = [("ontogpt", args.ontogpt_all, "ontogpt_id")]
        ontogpt_runs.append((ALL, args.ontogpt_all, "hpo_id"))
    todo = []
    for s in systems:
        if s.startswith("fasthpocr"):
            opts = parse_fasthpocr(s)
            label = f"fasthpocr ({', '.join(opts)})" if opts else "fasthpocr"
            FOLDERS[label] = "_".join(["fasthpocr"] + opts)
            runners[label] = (lambda o: lambda *a: run_fasthpocr(*a, opts=o))(opts)
            todo.append((label, None, None))
        else:
            todo.append((s, None, None))
    todo += ontogpt_runs
    for name, path, idcol in todo:
        sysdir = os.path.join(args.out, FOLDERS.get(name, name))
        os.makedirs(sysdir, exist_ok=True)
        print(f"\n=== {name}")
        try:
            raw = load_ontogpt(path, docs, texts, hpo, idcol) if path else runners[name](docs, texts, hpo, sysdir, args)
        except SystemExit as e:
            print(f"{name}: skipped - {e}")
            continue
        if path:
            n_hp = sum(p["grounded"] == "yes" for p in raw)
            print(f"grounding ({idcol}): {n_hp} of {len(raw)} mentions have an HPO id "
                  f"({100 * n_hp / max(1, len(raw)):.1f}%)")
            grounding[name] = (n_hp, len(raw))
        preds, off_branch = finalize(raw, hpo, args)
        write_preds(preds, os.path.join(sysdir, f"predictions_{tag}.tsv"))
        print(f"{len(preds)} predictions"
              + (f" ({off_branch} outside Phenotypic abnormality dropped)" if off_branch else ""))
        ev = SimpleNamespace(negated=args.negated, out=sysdir, mention_match=args.mention_match,
                             drop_ungrounded=args.drop_ungrounded and bool(path))
        # every selected document is scored, also those where a system found nothing
        # (for OntoGPT: run it on the same documents, or its missing ones count as misses)
        results[name] = evaluate(gold, preds, docset, ev)
        # evaluate() names its files by --negated only; add the boundary mode to the name
        for stem in ("metrics", "errors"):
            ext = "json" if stem == "metrics" else "tsv"
            src = os.path.join(sysdir, f"{stem}_{args.negated}{'_grounded' if ev.drop_ungrounded else ''}.{ext}")
            if os.path.exists(src):
                os.replace(src, os.path.join(sysdir, f"{stem}_{tag}.{ext}"))

    if not results:
        return 1
    levels = (("document", "document_level"), ("mention", "mention_level"))
    summ = os.path.join(args.out, f"summary_{tag}.tsv")
    with open(summ, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["system", "level", "precision", "recall", "f1", "tp", "fp", "fn"])
        for name, m in results.items():
            for lab, key in levels:
                x = m[key]
                w.writerow([name, lab, f"{x['precision']:.4f}", f"{x['recall']:.4f}", f"{x['f1']:.4f}",
                            x["tp"], x["fp"], x["fn"]])
    print(f"\n=== Summary (negated gold terms: {args.negated}, mention boundaries: {args.mention_match})"
          f"  F1 / P / R")
    width = max(22, max(len(n) for n in results) + 2)
    print(f"{'system':{width}}" + "".join(f"{lab:>24}" for lab, _ in levels))
    for name, m in results.items():
        cells = [f"{m[k]['f1']:.3f} / {m[k]['precision']:.3f} / {m[k]['recall']:.3f}" for _, k in levels]
        print(f"{name:{width}}" + "".join(f"{c:>24}" for c in cells))
    print(f"Wrote {summ}")

    if args.ontogpt_all:
        compare_ids(args.ontogpt_all, docset, gold, hpo, args)

    variants = [n for n in (FILL, ALL) if n in results]
    if "ontogpt" in results and variants:
        cols = ["ontogpt"] + variants
        rows = []
        r = ["grounded mentions"]
        for n in cols:
            g, t = grounding[n]
            r.append(f"{g}/{t} ({100 * g / max(1, t):.1f}%)")
        for n in variants:
            r.append(f"{grounding[n][0] - grounding['ontogpt'][0]:+d}")
        rows.append(r)
        for lab, key in levels:
            for meas in ("precision", "recall", "f1"):
                r = [f"{lab} {meas}"] + [f"{results[n][key][meas]:.3f}" for n in cols]
                r += [f"{results[n][key][meas] - results['ontogpt'][key][meas]:+.3f}" for n in variants]
                rows.append(r)
        header = ["measure"] + cols + [f"change {n.split()[-1]}" for n in variants]
        print("\n=== Grounding: OntoGPT's own vs BeLink (change relative to ontogpt)")
        print(f"{'':20}" + "".join(f"{h:>23}" for h in header[1:]))
        for r in rows:
            print(f"{r[0]:20}" + "".join(f"{c:>23}" for c in r[1:]))
        gpath = os.path.join(args.out, f"grounding_{tag}.tsv")
        with open(gpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(header)
            w.writerows(rows)
        print(f"Wrote {gpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
