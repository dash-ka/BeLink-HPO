# Training

- [How the released model was trained](#how-the-released-model-was-trained)
- [Fine-tune your own re-ranker](#fine-tune-your-own-re-ranker)

## How the released model was trained

[`Dash00/qwen3-4b-lora-hpo-reranker`](https://huggingface.co/Dash00/qwen3-4b-lora-hpo-reranker) is a LoRA adapter for `Qwen/Qwen3-4B`. The steps below rebuild its training data and repeat the training run.

```bash
pip install "belink-hpo[train,data,expand] @ git+https://github.com/dash-ka/BeLink-HPO.git"
python -m spacy download en_core_web_sm
```

### 1. HPO name index

Every HPO name and synonym is embedded with SapBERT.

First, `hp.obo` is turned into a flat list of names:

```bash
python scripts/prepare_obo.py --ontology_path hp.obo --output_dir index
```

This writes `index/terminology.json.gz`, a gzipped JSON list with one `{"id": "hp:0001156", "name": "short fingers or toes (brachydactyly)"}` record per name. Each synonym is written as `synonym (preferred name)` and each preferred name as `preferred name (longest synonym)`, so that no two terms share a name. Obsolete terms and the root term `All` are skipped. The file must be called `hp.obo`, because the script takes the ontology prefix from the file name.

From the HPO release used here this gave 19,894 terms and 45,176 names, 2.27 per term.

The index behind the released model was built before the root term was excluded, so it also contained `All` (HP:0000001). An index rebuilt with the current script differs from it only by that term.

Then the names are embedded:

```bash
belink-hpo embed-terms --ontology index/terminology.json.gz \
    --model_name cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --out_vectors index/hpo_vec.npy
```

This writes `index/hpo_vec.npy` (45,176 × 768) and `index/ontology_mapping.txt`, which lists the `id||name` of each vector.

### 2. Corpora

| Corpus | Texts | Annotations | Unique mentions |
|---|---|---|---|
| [HPO-GSC+ 2024](https://github.com/tudorgroza/code-for-papers/tree/main/gsc-2024) | 228 abstracts | 2,909 | 1,139 |
| BioCreative VIII Track 3, development set (`Clinical-Genetics-Physical-Exam-Data-Dev.tsv`) | 454 observations | 607 | 522 |

- **HPO-GSC+** abstracts are split into sentences with spaCy (`en_core_web_sm`); the sentence is the context of each mention.
- **BioCreative** observations are short and are used whole as the context. Negated findings are skipped. For a discontiguous mention, the text from its first to its last character is used.
- Annotated ids are mapped to current primary HPO ids through `hp.obo`; annotations with unknown ids are dropped.

### 3. Standard names for the mentions

For each unique mention, an LLM (`gpt-5.4`, temperature 0) is asked for the standard HPO name of the concept, given the sentence. The prompt is in [`belink_hpo/expansion.py`](../src/belink_hpo/expansion.py).

```bash
export OPENAI_API_KEY=...
python scripts/build_hpo_training_data.py mentions --obo hp.obo \
    --gsc-text GSC_2024/Text --gsc-annotations GSC_2024/Annotations \
    --biocreative Clinical-Genetics-Physical-Exam-Data-Dev.tsv --expand
```

This writes `data_gsc_standard_names.json` and `data_bc_standard_names.json`, one record per unique mention.

### 4. Candidate retrieval

30 names are retrieved per mention. The query vector is `0.6 × mention + 0.4 × standard name`.

```bash
for f in data_gsc_standard_names.json data_bc_standard_names.json; do
  belink-hpo retrieve --input $f --output_file $f \
      --kb_vectors index/hpo_vec.npy --kb_mapping ontology_mapping.txt --top_k 30 \
      --model_name cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
      --apply_grf --use_rocchio --alpha 0.6
done
```

Accuracy of the top-1 retrieved candidate, which is the baseline the re-ranker has to beat:

| Corpus | Accuracy@1 |
|---|---|
| HPO-GSC+ | 2407 / 2909 = 82.74% |
| BioCreative dev | 477 / 607 = 78.58% |

### 5. Multiple-choice dataset

```bash
python scripts/build_hpo_training_data.py dataset --obo hp.obo \
    --gsc-text GSC_2024/Text --gsc-annotations GSC_2024/Annotations \
    --biocreative Clinical-Genetics-Physical-Exam-Data-Dev.tsv \
    --out-dir data --push-to Dash00/data-hpo-belink
```

**Candidate names.** Each name is followed by the preferred name of its term: `short toe (aka brachydactyly)`. A name that already ends in the preferred name in brackets gets `aka` inserted, and the preferred name itself is left as it is.

**Training rows.** Every annotation produces one row per pass, and 5 passes are made over each corpus, so each mention is seen with 5 different option sets:

- one randomly chosen name of the gold concept;
- up to 9 negatives, each from a different concept, with a randomly chosen name;
- options in random order, then `None of the above.` as the last option.

If the gold concept is not among the 30 retrieved names, the row has up to 10 negatives and the answer is `None of the above.`. No other "none" rows are created (`--nota-rate` in `belink-hpo build-dataset` can add them).

This gives 14,545 rows from HPO-GSC+ and 3,035 from BioCreative, 17,580 in total, shuffled together.

Options are sampled at random, so a rebuilt dataset has the same size and structure as [`Dash00/data-hpo-belink`](https://huggingface.co/datasets/Dash00/data-hpo-belink) but not the same option sets. To repeat the training run exactly, use the uploaded dataset.

**Test rows** are built as the model sees candidates in use: the first retrieved name of each concept, in retrieval order, capped at 10, with `None of the above.` as the answer when the gold concept is not among them. The uploaded test split is the 607 BioCreative annotations.

Each row is stored in chat format:

```json
{"messages": [
  {"role": "user", "content": "<Instruct>: Given the context '...', select the correct biomedical concept corresponding to 'brachydactyly'. Answer using one of the provided options.\n\n<Options>: A: ...\nB: ...\nC: None of the above."},
  {"role": "assistant", "content": "B"}
]}
```

### 6. Fine-tuning

```bash
belink-hpo train --dataset Dash00/data-hpo-belink --epochs 1 --no-4bit
```

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B`, loaded in bf16 (`--no-4bit`) |
| LoRA | r = 16, alpha = 32, dropout = 0.05, on all attention and MLP projections |
| Epochs | 1 over the 17,580 rows |
| Batch size | 8, with gradient accumulation 2 (effective 16) |
| Learning rate | 2e-4, cosine schedule, 5% warm-up |
| Maximum length | 1,024 tokens; longer rows are dropped |

The prompt is rendered with Qwen's chat template with thinking disabled, and the loss is computed only on the answer letter and the end-of-turn token. The adapter is saved to `qwen3-mcq-lora/`.

### 7. Evaluation and upload

```bash
belink-hpo predict --file data/test.jsonl --adapter qwen3-mcq-lora --batch-size 64
```

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

AutoPeftModelForCausalLM.from_pretrained("qwen3-mcq-lora").push_to_hub("Dash00/qwen3-4b-lora-hpo-reranker")
AutoTokenizer.from_pretrained("Qwen/Qwen3-4B").push_to_hub("Dash00/qwen3-4b-lora-hpo-reranker")
```

**Held-out result.** A model trained with the same recipe on HPO-GSC+ only reached 83.86% on the BioCreative development set, against 78.58% for the top-1 retrieved candidate. The released model also saw the BioCreative development set during training, so its accuracy on that test split is not a held-out measure.

## Fine-tune your own re-ranker

You need annotated mentions and, for each one, a ranked list of candidate concepts. Any ontology works: nothing in the training code is specific to HPO.

### 1. Write your examples

One JSON object per line ([`examples/examples.jsonl`](../examples/examples.jsonl)):

```json
{"doc_id": "doc1", "context": "A syndrome of brachydactyly is reported in five members of a family.", "mention": "brachydactyly", "gold_id": "HP:0001156", "candidates": [{"id": "HP:0001156", "name": "short fingers or toes"}, {"id": "HP:0001831", "name": "short toe"}]}
```

| Field | Meaning |
|---|---|
| `context` | The sentence or passage containing the mention |
| `mention` | The mention text |
| `gold_id` | The correct concept id |
| `candidates` | Retrieved names, best first. A concept may appear several times under different names. Ids must be written the same way as `gold_id` |
| `doc_id` | Optional. Examples that share it stay on the same side of the train/test split |

Use the retriever you will use in production to produce the candidates, so that the model trains on the mistakes it will have to correct. To use the one in this package, build an index of your ontology's names (`prepare_obo.py` and `embed-terms`, as in step 1 above; any OBO file works) and call `CandidateRetriever.retrieve`. Retrieve more names than you will show (30 here), which gives the sampler a pool of negatives.

### 2. Build the prompts

```bash
belink-hpo build-dataset --input my_examples.jsonl --out-dir data \
    --test-fraction 0.1 --epochs 5 --preferred-names preferred_names.json
```

- `--test-fraction` holds out that share of documents. Use `--test-input other.jsonl` instead if you have a separate test set.
- `--epochs` is the number of option sets sampled per mention.
- `--preferred-names` (optional) is a JSON file mapping concept id to preferred name, used to add `(aka ...)` to the candidate names. `OntoReader("my.obo").save_preferred_names("preferred_names.json")` writes one from an OBO file.
- `--nota-rate 0.1` hides the gold concept in 10% of rows, to teach the model to answer `None of the above.` more readily.
- `--push-to username/my-dataset` uploads the result to the Hugging Face Hub.

The command prints the top-1 accuracy of your retriever on the test examples. That is the number to beat.

If your data is already in the chat format shown above, skip this step and put `train.jsonl` and `test.jsonl` in one folder.

### 3. Train

```bash
belink-hpo train --data-dir data --epochs 1 --no-4bit --output-dir my-reranker
```

| Option | Use |
|---|---|
| `--model` | Base model. Default `Qwen/Qwen3-4B`; other Qwen3 sizes work, e.g. `Qwen/Qwen3-8B` |
| `--dataset` | Hub dataset id, instead of `--data-dir` |
| `--epochs` | Passes over the rows; fractions are allowed. The rows already repeat each mention `--epochs` times from step 2, so 1 is usually enough |
| `--no-4bit` | Load the base model in bf16. Faster, needs more VRAM. Without it the model is loaded in 4-bit |
| `--grad-checkpointing` | Saves memory at the cost of speed |
| `--batch-size`, `--grad-accum`, `--lr`, `--max-length` | As usual |
| `--lora-r`, `--lora-alpha`, `--lora-dropout` | LoRA size |
| `--eval-steps`, `--eval-samples` | How often, and on how many test rows, to evaluate during training |

Accuracy on the test split is printed before and after training. Without a `test.jsonl` the run trains without evaluation.

Another model family can be used if its tokenizer encodes each option letter as a single token and its attention and MLP layers use the module names `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`. The end-of-turn token `<|im_end|>` in `train.py` would need changing to that model's.

### 4. Evaluate and use

```bash
belink-hpo predict --file data/test.jsonl --adapter my-reranker
```

```python
from belink_hpo import HPOReranker

reranker = HPOReranker.from_pretrained("my-reranker")   # local folder or Hub id
```

`--adapter` also accepts a checkpoint folder such as `my-reranker/checkpoint-500`.
