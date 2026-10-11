# BeLink-HPO

An LLM re-ranker for linking phenotype mentions to [Human Phenotype Ontology](https://hpo.jax.org/) (HPO) terms.

Given a mention in its sentence and a list of candidate HPO terms from a retriever, the model picks the correct term or answers "None of the above". It is [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) with a LoRA adapter, fine-tuned on HPO-GSC+ 2024 and the BioCreative VIII Track 3 development set, and published as [`Dash00/qwen3-4b-lora-hpo-reranker`](https://huggingface.co/Dash00/qwen3-4b-lora-hpo-reranker).

The candidates are presented as a multiple-choice question. The model is not asked to generate text: each mention takes one forward pass, and the scores of the option letters are compared.

```
<Instruct>: Given the context 'A syndrome of brachydactyly is reported in five members
of an Italian family.', select the correct biomedical concept corresponding to
'brachydactyly'. Answer using one of the provided options.

<Options>: A: short toe
B: short fingers or toes (aka brachydactyly)
C: type a brachydactyly
D: None of the above.
```

## Results

Linking accuracy on the BioCreative development set (607 mentions), with a re-ranker trained on HPO-GSC+ only, so the test set was unseen:

| | Accuracy |
|---|---|
| Top-1 retrieved candidate (SapBERT with standard-name feedback) | 78.58% |
| After re-ranking the top 10 concepts | 83.86% |

The released adapter was then re-trained on both corpora, so it has no held-out score of its own.

## Install

```bash
pip install git+https://github.com/dash-ka/BeLink-HPO.git
```

Optional extras:

```bash
# fine-tuning
pip install "belink-hpo[train] @ git+https://github.com/dash-ka/BeLink-HPO.git"
# rebuilding the HPO training data from the raw corpora
pip install "belink-hpo[data,expand] @ git+https://github.com/dash-ka/BeLink-HPO.git"
```

Python 3.9 or newer. A GPU is strongly recommended: the base model has about 8 GB of weights in bf16. Pass `load_in_4bit=True` (needs `bitsandbytes`) to fit a smaller card at lower speed.

## Use the re-ranker

```python
from belink_hpo import HPOReranker

reranker = HPOReranker.from_pretrained()   # downloads Dash00/qwen3-4b-lora-hpo-reranker

result = reranker.rerank(
    context="A syndrome of brachydactyly is reported in five members of an Italian family.",
    mention="brachydactyly",
    candidates=[                             # in retrieval order, best first
        {"id": "HP:0001831", "name": "short toe"},
        {"id": "HP:0001156", "name": "short fingers or toes"},
        {"id": "HP:0009370", "name": "type a brachydactyly"},
    ],
)

result.id          # chosen HPO id, or None if the model answers "None of the above."
result.name        # the option text that was chosen
result.confidence  # probability of the chosen option among those offered
result.options     # [(letter, id, name), ...] exactly as shown to the model
```

How candidates are handled:

- **Order matters.** Candidates are shown in the order given.
- **One name per concept.** If an id appears several times, only its first name is used.
- **At most 10 concepts** are shown; the rest are ignored.
- **Many mentions at once**: `reranker.rerank_batch([{"context": ..., "mention": ..., "candidates": [...]}, ...])`.

### Show preferred names, as in training

In the training data each candidate name was followed by the term's preferred HPO name, as in `short fingers or toes (aka brachydactyly)`. To get the same prompts at inference, give the re-ranker the preferred names:

```python
from belink_hpo import HPOReranker
from belink_hpo.ontology import load_preferred_names

preferred = load_preferred_names("index/preferred_names.json")   # {"HP:0001156": "Brachydactyly", ...}
reranker = HPOReranker.from_pretrained(preferred_names=preferred)
```

`index/preferred_names.json` is written by `belink-hpo prepare-obo --ontology_path hp.obo --output_dir index`. An `.obo` file can be passed in its place, which needs `pronto` (the `[data]` extra). Candidate ids must match the keys of the mapping (`HP:0001156`, upper case).

Without `preferred_names`, candidate names are shown exactly as you give them. The model still works, but it was trained on the `(aka ...)` form, so prompts without it differ from what it saw in training.

Which commands use preferred names depends on who writes the prompt:

| Command or function | Who writes the prompt | Preferred names |
|---|---|---|
| `rerank`, `rerank_batch`, `belink-hpo rerank`, `HPOLinker` | The package, from your candidates | Only if you pass `preferred_names` / `--preferred-names` |
| `belink-hpo build-dataset`, `prepare_train`, `prepare_test` | The package, from your examples | Only if you pass `--preferred-names` / `preferred_names` |
| `predict_labels`, `belink-hpo predict` | You, beforehand | Not applied: prompts are scored exactly as written |

`predict` is meant for files of ready-made prompts, such as the `test.jsonl` written by `build-dataset`. If that file was built with preferred names, they are already in the option text.

### From the command line

```bash
belink-hpo rerank --input mentions.jsonl --out linked.jsonl --preferred-names index/preferred_names.json
```

`mentions.jsonl` has one JSON object per line with `context`, `mention` and `candidates` (see [`examples/examples.jsonl`](examples/examples.jsonl)). If a line also has `gold_id`, accuracy is printed.

## Full pipeline: retrieval + re-ranking

The re-ranker needs candidates. This package includes the dense retriever used to build the training data: [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) embeddings of every HPO name and synonym, searched by inner product.

Build the index once (details in [docs/TRAINING.md](docs/TRAINING.md#1-hpo-name-index)):

```bash
belink-hpo prepare-obo --ontology_path hp.obo --output_dir index
belink-hpo embed-terms --ontology index/terminology.json.gz --out_vectors index/hpo_vec.npy
```

Then link mentions:

```python
from belink_hpo import CandidateRetriever, HPOLinker, HPOReranker
from belink_hpo.ontology import load_preferred_names

linker = HPOLinker(
    CandidateRetriever.from_files("index/hpo_vec.npy", "index/ontology_mapping.txt"),
    HPOReranker.from_pretrained(preferred_names=load_preferred_names("index/preferred_names.json")),
)

result = linker.link(
    context="EYES: mild epicanthus, mild up-slant.",
    mention="mild up-slant",
    standard_name="Upslanted palpebral fissure",   # optional
)
```

`standard_name` is a canonical name for the mention, which is mixed into the retrieval query. The reported numbers use standard names generated by an LLM (`belink_hpo.expansion`). Without it, retrieval uses the mention text alone and will find the right concept less often.

## Fine-tuning

[docs/TRAINING.md](docs/TRAINING.md) describes how the released model was trained and how to train your own re-ranker on a custom dataset or ontology. In short:

```bash
belink-hpo build-dataset --input my_examples.jsonl --out-dir data
belink-hpo train --data-dir data --epochs 1 --no-4bit --output-dir my-reranker
belink-hpo predict --file data/test.jsonl --adapter my-reranker
```

## Commands

| Command | Purpose |
|---|---|
| `belink-hpo rerank` | Choose an HPO id for each mention in a JSONL file |
| `belink-hpo predict` | Predict option letters for a file of ready-made prompts, and report accuracy |
| `belink-hpo train` | Fine-tune a LoRA adapter |
| `belink-hpo build-dataset` | Turn examples with candidates into training and test prompts |
| `belink-hpo prepare-obo` | Turn an OBO ontology into the list of names to index |
| `belink-hpo embed-terms` | Embed ontology names into a retrieval index |
| `belink-hpo retrieve` | Retrieve candidates for a JSON file of mentions |

Run any command with `--help` for its options.

### `predict` or `rerank`?

Both run the same model with the same scoring. They differ in what you give them and what you get back.

| | `predict` | `rerank` |
|---|---|---|
| Input | Ready-made prompts in the messages format | Context, mention and a candidate list |
| Builds the prompt | No, scores the text as written | Yes: one name per concept, at most 10, optional `(aka ...)` |
| Output | An option letter, e.g. `B` | An HPO id, e.g. `HP:0001156`, or none |
| Letters compared | All 26 | Only the ones offered in the prompt |
| Accuracy measured against | The gold letter in the file | `gold_id`, if present |
| Use it for | Evaluating a model on a test set | Linking mentions |

**`predict` is the evaluation tool.** It reads a file such as the `test.jsonl` written by `build-dataset` and reports how often the model picks the right letter. Because the prompts are fixed in the file, two models scored on it are compared on exactly the same inputs.

**`rerank` is the tool for actual use.** It writes the prompt from your candidates, runs the model, and translates the chosen letter back into an HPO id. The id is empty when the model answers "None of the above".

In Python the same split is `reranker.predict_labels(prompts)`, which returns letters, and `reranker.rerank(context, mention, candidates)`, which returns a result with `.id`.

**Which letters can win.** The model scores every letter from A to Z on each prompt, even when the prompt offers only a few. Take a prompt with three candidates:

```
A: short toe
B: short fingers or toes (aka brachydactyly)
C: type a brachydactyly
D: None of the above.
```

Only A to D mean anything here. `predict` receives just the prompt text and has no record of how many options it holds, so it takes the highest score among all 26 letters. If that were `E`, the answer would be meaningless and counted as wrong. `rerank` built the prompt itself, so it compares A to D only and reports its confidence over those four. A fine-tuned model has learned to answer with a listed letter, so the two rarely differ.

On the same mentions, `predict` and `rerank` therefore give the same answers, provided `rerank` is given the preferred names that were used to build the prompts in the file.

## Development

```bash
git clone https://github.com/dash-ka/BeLink-HPO.git && cd BeLink-HPO
pip install -e ".[dev,train]"
pytest
```

The tests build tiny random models on the fly, so they need no downloads and run on CPU.
