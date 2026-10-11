"""Build multiple-choice training and test sets for the re-ranker.

Input is a list of *examples*, one per annotated mention:

    {
        "context": "A syndrome of brachydactyly is reported in five members of a family.",
        "mention": "brachydactyly",
        "gold_id": "HP:0001156",
        "candidates": [{"id": "HP:0001156", "name": "short fingers or toes (brachydactyly)"}, ...]
    }

``candidates`` are the retrieved names in rank order; one concept may appear several
times under different names. Output rows are in chat format:

    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": "C"}]}

Command line:

    belink-hpo build-dataset --input examples.jsonl --out-dir data --test-fraction 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .prompt import (
    LETTERS,
    MAX_OPTIONS,
    augment_name,
    candidate_id,
    candidate_name,
    make_row,
    select_options,
)


def prepare_train(examples, preferred_names=None, epochs=5, nota_rate=0.0, max_options=MAX_OPTIONS, seed=42):
    """Build shuffled training rows.

    Every pass over the data (``epochs``) re-samples the options of each mention: one
    name of the gold concept plus up to ``max_options - 1`` negatives, each from a
    different concept, in random order. If the gold concept was not retrieved, the row
    holds negatives only and the answer is "None of the above.".

    ``nota_rate`` hides the gold concept in that fraction of rows to produce extra
    "None of the above." examples. The released model was trained with 0.
    """
    rng = random.Random(seed)
    answers = Counter()
    data = []

    for _ in range(epochs):
        for ex in examples:
            negatives_by_id = defaultdict(list)
            seen_names, positives = [], []

            for cand in ex.get("candidates", []):
                cid = candidate_id(cand)
                name = candidate_name(cand)
                if preferred_names is not None:
                    name = augment_name(name, preferred_names.get(cid))

                # collect all positive synonyms
                if cid == ex["gold_id"]:
                    positives.append(name)
                elif name not in seen_names:
                    negatives_by_id[cid].append(name)
                seen_names.append(name)

            hide_positive = rng.random() < nota_rate
            options = []
            selected_positive = None

            if positives and not hide_positive:
                selected_positive = rng.choice(positives)
                options.append(selected_positive)
                num_neg = min(max_options - 1, len(negatives_by_id))
            else:
                num_neg = min(max_options, len(negatives_by_id))

            for key in rng.sample(list(negatives_by_id), k=num_neg):
                options.append(rng.choice(negatives_by_id[key]))

            options = list(dict.fromkeys(options))  # drop duplicate names, keep order
            rng.shuffle(options)

            if selected_positive is not None:
                correct_letter = LETTERS[options.index(selected_positive)]
            else:
                correct_letter = LETTERS[len(options)]  # "None of the above."

            answers[correct_letter] += 1
            data.append(make_row(ex["context"], ex["mention"], options, correct_letter))

    rng.shuffle(data)
    print(f"Training rows: {len(data)}")
    print("Answer distribution:", dict(sorted(answers.items())))
    return data


def prepare_test(examples, preferred_names=None, unique_mentions=False, max_options=MAX_OPTIONS):
    """Build test rows exactly as the model sees candidates at inference time.

    Options keep the retrieval order: the first retrieved name of each concept, capped
    at ``max_options``. The answer is "None of the above." when the gold concept is not
    among them. Mentions without candidates are skipped. ``unique_mentions`` keeps only
    the first occurrence of each mention text (case-insensitive).
    """
    data, seen_mentions = [], set()

    for ex in examples:
        mention_key = ex["mention"].lower()
        if unique_mentions and mention_key in seen_mentions:
            continue
        if not ex.get("candidates"):
            continue

        options = select_options(ex["candidates"], preferred_names, max_options)
        correct_letter = LETTERS[len(options)]
        for i, (cid, _) in enumerate(options):
            if cid == ex["gold_id"]:
                correct_letter = LETTERS[i]
                break

        data.append(make_row(ex["context"], ex["mention"], [name for _, name in options], correct_letter))
        seen_mentions.add(mention_key)

    print(f"Test rows: {len(data)}")
    return data


def examples_from_sentences(dataset_sentences, normalize_id=None):
    """Flatten ``{docid: [{"text", "annotations": [{"text", "hpoId", "candidates"}]}]}`` into examples.

    ``normalize_id`` is applied to every candidate id, e.g. to upper-case it and map
    alternate HPO ids to the primary id (see ``OntoReader.normalize``).
    """
    normalize_id = normalize_id or (lambda x: x)
    examples = []
    for docid, sentences in dataset_sentences.items():
        for sentence in sentences:
            for anno in sentence["annotations"]:
                examples.append({
                    "doc_id": docid,
                    "context": sentence["text"],
                    "mention": anno["text"],
                    "gold_id": anno["hpoId"],
                    "candidates": [
                        {"id": normalize_id(candidate_id(c)), "name": candidate_name(c)}
                        for c in anno.get("candidates", [])
                    ],
                })
    return examples


def split_examples(examples, test_fraction=0.1, seed=42):
    """Split into (train, test), keeping all examples with the same ``doc_id`` together.

    Examples without a ``doc_id`` are treated as one document each.
    """
    groups = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[ex.get("doc_id", f"__row_{i}")].append(ex)
    keys = sorted(groups, key=str)
    random.Random(seed).shuffle(keys)
    n_test = max(1, round(len(keys) * test_fraction))
    test = [ex for k in keys[:n_test] for ex in groups[k]]
    train = [ex for k in keys[n_test:] for ex in groups[k]]
    return train, test


def accuracy_at_1(examples) -> float:
    """Share of mentions whose top retrieved candidate is the gold concept (the retrieval baseline)."""
    hits = sum(bool(ex.get("candidates")) and candidate_id(ex["candidates"][0]) == ex["gold_id"] for ex in examples)
    return hits / len(examples)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def push_to_hub(train_data, repo_id, local_dir, test_data=None, token=None, private=False):
    """Save the splits as JSONL in ``local_dir`` and upload them as a Hugging Face dataset."""
    from datasets import Dataset, DatasetDict
    from huggingface_hub import create_repo, upload_file

    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token, private=private)

    splits = {"train": Dataset.from_list(train_data)}
    if test_data is not None:
        splits["test"] = Dataset.from_list(test_data)
    ds = DatasetDict(splits)

    for split in ds:  # raw JSONL files for manual download
        local_path = Path(local_dir) / f"{split}.jsonl"
        save_jsonl(ds[split], local_path)
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{split}.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    ds.push_to_hub(repo_id, token=token)  # parquet version, loadable with load_dataset()
    print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="belink-hpo build-dataset", description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="JSONL with one example per line (context, mention, gold_id, candidates)")
    p.add_argument("--out-dir", default="data", help="Where train.jsonl and test.jsonl are written")
    p.add_argument("--test-input", default=None, help="Separate JSONL of test examples; overrides --test-fraction")
    p.add_argument("--test-fraction", type=float, default=0.1, help="Share of documents held out for testing")
    p.add_argument("--epochs", type=int, default=5, help="Passes over the data, each with freshly sampled options")
    p.add_argument("--nota-rate", type=float, default=0.0, help="Share of rows where the gold concept is hidden")
    p.add_argument("--preferred-names", default=None, help="JSON file mapping concept id -> preferred name")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push-to", default=None, help="Hub dataset id to upload to, e.g. username/my-dataset")
    args = p.parse_args(argv)

    preferred = None
    if args.preferred_names:
        with open(args.preferred_names, encoding="utf-8") as f:
            preferred = json.load(f)

    examples = load_jsonl(args.input)
    if args.test_input:
        train_ex, test_ex = examples, load_jsonl(args.test_input)
    else:
        train_ex, test_ex = split_examples(examples, args.test_fraction, args.seed)

    print(f"Retrieval accuracy@1 on the test examples: {accuracy_at_1(test_ex):.2%}")
    train = prepare_train(train_ex, preferred, args.epochs, args.nota_rate, seed=args.seed)
    test = prepare_test(test_ex, preferred)

    save_jsonl(train, Path(args.out_dir) / "train.jsonl")
    save_jsonl(test, Path(args.out_dir) / "test.jsonl")
    print(f"Wrote {args.out_dir}/train.jsonl and {args.out_dir}/test.jsonl")

    if args.push_to:
        push_to_hub(train, args.push_to, args.out_dir, test_data=test)


if __name__ == "__main__":
    main()
