"""Rebuild the dataset the released model was trained on (HPO-GSC+ 2024 + BioCreative dev).

Three stages; stage 2 is a separate command because it needs the retrieval index.

1. Collect the unique mentions of both corpora and generate a standard name for each:

    python scripts/build_hpo_training_data.py mentions --obo hp.obo \
        --gsc-text GSC_2024/Text --gsc-annotations GSC_2024/Annotations \
        --biocreative Clinical-Genetics-Physical-Exam-Data-Dev.tsv --expand

2. Retrieve 30 candidates per mention (once per file):

    belink-hpo retrieve --input data_gsc_standard_names.json --output_file data_gsc_standard_names.json \
        --kb_vectors index/hpo_vec.npy --top_k 30 --apply_grf --use_rocchio --alpha 0.6

3. Build the multiple-choice rows:

    python scripts/build_hpo_training_data.py dataset --obo hp.obo \
        --gsc-text GSC_2024/Text --gsc-annotations GSC_2024/Annotations \
        --biocreative Clinical-Genetics-Physical-Exam-Data-Dev.tsv --out-dir data

Requires: pip install "belink-hpo[data,expand]" and python -m spacy download en_core_web_sm
"""

import argparse
import json
import random
from pathlib import Path

from belink_hpo.corpora import attach_candidates, load_biocreative, load_gsc, unique_mentions
from belink_hpo.dataset import (
    accuracy_at_1,
    examples_from_sentences,
    prepare_test,
    prepare_train,
    push_to_hub,
    save_jsonl,
)
from belink_hpo.ontology import OntoReader

GSC_MENTIONS = "data_gsc_standard_names.json"
BC_MENTIONS = "data_bc_standard_names.json"


def load_corpora(args, reader):
    gsc = load_gsc(args.gsc_text, args.gsc_annotations, reader)
    bc = load_biocreative(args.biocreative, reader)
    return gsc, bc


def stage_mentions(args):
    reader = OntoReader(args.obo)
    for corpus, name in zip(load_corpora(args, reader), (GSC_MENTIONS, BC_MENTIONS)):
        mentions = unique_mentions(corpus)
        print(f"{name}: {len(mentions)} unique mentions")
        if args.expand:
            from belink_hpo.expansion import add_standard_names

            add_standard_names(mentions, model=args.llm)
        with open(Path(args.work_dir) / name, "w", encoding="utf-8") as f:
            json.dump(mentions, f, indent=4, ensure_ascii=False)


def stage_dataset(args):
    reader = OntoReader(args.obo)
    gsc, bc = load_corpora(args, reader)

    examples = {}
    for label, corpus, name in (("HPO-GSC+", gsc, GSC_MENTIONS), ("BioCreative", bc, BC_MENTIONS)):
        with open(Path(args.work_dir) / name, encoding="utf-8") as f:
            missing = attach_candidates(corpus, json.load(f))
        examples[label] = examples_from_sentences(corpus, reader.normalize)
        print(f"{label}: {len(examples[label])} annotations, {missing} without candidates, "
              f"retrieval accuracy@1 = {accuracy_at_1(examples[label]):.4f}")

    train = []
    for label in examples:
        train += prepare_train(examples[label], reader.terms, epochs=args.epochs)
    random.Random(42).shuffle(train)
    test = prepare_test(examples["BioCreative"], reader.terms)

    save_jsonl(train, Path(args.out_dir) / "train.jsonl")
    save_jsonl(test, Path(args.out_dir) / "test.jsonl")
    print(f"Wrote {len(train)} training rows and {len(test)} test rows to {args.out_dir}")
    if args.push_to:
        push_to_hub(train, args.push_to, args.out_dir, test_data=test)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["mentions", "dataset"])
    p.add_argument("--obo", required=True, help="Path to hp.obo")
    p.add_argument("--gsc-text", required=True, help="GSC+ 2024 Text folder")
    p.add_argument("--gsc-annotations", required=True, help="GSC+ 2024 Annotations folder")
    p.add_argument("--biocreative", required=True, help="Clinical-Genetics-Physical-Exam-Data-Dev.tsv")
    p.add_argument("--work-dir", default=".", help="Where the unique-mention JSON files live")
    p.add_argument("--expand", action="store_true", help="[mentions] generate standard names (needs OPENAI_API_KEY)")
    p.add_argument("--llm", default="gpt-5.4", help="[mentions] model used for the standard names")
    p.add_argument("--epochs", type=int, default=5, help="[dataset] passes over each corpus")
    p.add_argument("--out-dir", default="data", help="[dataset] where train.jsonl and test.jsonl are written")
    p.add_argument("--push-to", default=None, help="[dataset] Hub dataset id to upload to")
    args = p.parse_args()
    {"mentions": stage_mentions, "dataset": stage_dataset}[args.stage](args)


if __name__ == "__main__":
    main()
