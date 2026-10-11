"""Loaders for the two corpora the released model was trained on.

Both return ``{docid: [sentence, ...]}`` where a sentence is
``{"text", "start", "end", "annotations": [{"start", "end", "hpoId", "text"}]}``
with annotation offsets relative to the sentence.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from os.path import join

OFFSET_PATTERN = re.compile(r"(\d+):(\d+)")


def _skip(name):
    """macOS AppleDouble / Finder files that ship inside the GSC+ tarball."""
    base = name.rsplit("/", 1)[-1]
    return base.startswith(".") or not base


def load_gsc(text_dir, annot_dir, reader, spacy_model="en_core_web_sm"):
    """HPO-GSC+ 2024: one text file and one annotation file per abstract.

    Annotation lines are ``start:end<TAB>HP_0001156|...``. Abstracts are split into
    sentences with spaCy, and each annotation is assigned to the sentence that fully
    contains it. Annotations whose id is unknown to the ontology are dropped.
    """
    import spacy

    nlp = spacy.load(spacy_model)

    spans = {}
    for file in os.listdir(annot_dir):
        if _skip(file):
            continue
        data = []
        with open(join(annot_dir, file), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                segs = line.split("\t")
                hpo_id = reader.consolidate(segs[1].strip().split("|")[0].strip().replace("_", ":"))
                offsets = OFFSET_PATTERN.match(segs[0].strip())
                if hpo_id and offsets:
                    data.append((hpo_id, int(offsets.group(1)), int(offsets.group(2))))
        spans[file] = data

    dataset = {}
    for file in os.listdir(text_dir):
        if _skip(file):
            continue
        with open(join(text_dir, file), encoding="utf-8") as fh:
            passage = fh.read().strip()

        sentences = []
        for sent in nlp(passage).sents:
            s, e = int(sent.start_char), int(sent.end_char)
            sentences.append({
                "text": sent.text,
                "start": s,
                "end": e,
                "annotations": [
                    {"start": a - s, "end": b - s, "hpoId": hpo_id, "text": sent.text[a - s : b - s]}
                    for hpo_id, a, b in spans.get(file, [])
                    if a >= s and b <= e
                ],
            })
        dataset[file] = sentences
    return dataset


def load_biocreative(tsv_file, reader, skip_negated=True):
    """BioCreative VIII Track 3 dev set: one TSV with text and annotations, keyed by StringID.

    Each observation is used whole as the context (no sentence splitting). Rows without
    a term, negated findings (unless ``skip_negated=False``) and ids unknown to the
    ontology are dropped. For discontiguous spans the mention runs from the first
    start to the last end.
    """
    texts = {}
    spans = defaultdict(list)

    with open(tsv_file, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            docid = row["StringID"]
            texts.setdefault(docid, row["Text"])

            hpo_id = row["Term"]
            if not hpo_id or hpo_id == "NA":
                continue
            if skip_negated and row["Negated"] == "X":
                continue
            hpo_id = reader.consolidate(hpo_id)
            if not hpo_id:
                continue

            parts = [p for p in row["Spans"].split("-") if p.strip()]
            if not parts:
                continue
            spans[docid].append((hpo_id, int(parts[0]), int(parts[-1])))

    return {
        docid: [{
            "text": passage,
            "start": 0,
            "end": len(passage),
            "annotations": [
                {"start": a, "end": b, "hpoId": hpo_id, "text": passage[a:b]}
                for hpo_id, a, b in spans.get(docid, [])
            ],
        }]
        for docid, passage in texts.items()
    }


def unique_mentions(dataset_sentences):
    """First occurrence of every distinct mention text, keyed by the mention text."""
    mentions = {}
    for docid, sentences in dataset_sentences.items():
        for sent in sentences:
            for ann in sent["annotations"]:
                mentions.setdefault(ann["text"], {
                    "concept_id": ann["hpoId"],
                    "document_id": docid,
                    "sentence": sent["text"],
                    "label": ann["text"],
                })
    return mentions


def attach_candidates(dataset_sentences, mentions_with_candidates):
    """Copy retrieved candidates from the unique-mention records onto every annotation.

    Returns the number of annotations that had no candidates.
    """
    from .retrieval import parse_candidates

    by_text = {e["label"]: parse_candidates(e.get("candidates", "")) for e in mentions_with_candidates.values()}
    missing = 0
    for sentences in dataset_sentences.values():
        for sent in sentences:
            for ann in sent["annotations"]:
                ann["candidates"] = by_text.get(ann["text"], [])
                missing += not ann["candidates"]
    return missing
