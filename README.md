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

## Evaluation: phenotype recognition with OntoGPT and BeLink

BeLink can ground the phenotype mentions that an LLM extracts from clinical text. We extracted every phenotype mention with [OntoGPT](https://github.com/monarch-initiative/ontogpt) and grounded them with BeLink in two ways: only the mentions OntoGPT could not ground itself (*fill*), or every mention (*all*). Both pipelines were compared with three established HPO concept recognisers on the same corpus.

Everything needed to reproduce it (scripts, pinned environments, setup and run scripts) is in [`evaluation/`](evaluation/README.md).

### Setup

- **Corpus:** BioCreative VIII Track 3 development set (`Clinical-Genetics-Physical-Exam-Data-Dev.tsv`): 454 physical-examination observations with 685 HPO annotations, 78 of them negated.
- **Negated annotations** (e.g. *"Normal lips"* annotated with *Abnormal lip morphology*) are removed from the gold standard, leaving 607 mentions. Systems are not asked to classify mentions as present or negated; every prediction is scored.
- **Scope:** all systems are restricted to the *Phenotypic abnormality* branch of HPO (HP:0000118), the scope of the gold standard. FastHPOCR and PhenoTagger are restricted by construction. For NCBO and OntoGPT, predictions outside the branch (mostly clinical modifiers such as *mild* or *bilateral*) are removed: 197 for NCBO and 2 for each BeLink pipeline.
- **BeLink model:** the re-ranker trained on HPO-GSC+ only, so the evaluation corpus was not seen in training.

### Systems

| System | Approach |
|---|---|
| [FastHPOCR](https://github.com/tudorgroza/fast_hpo_cr) | Dictionary-based concept recognition over HPO labels and synonyms, using clusters of related tokens. Default configuration (v0.1.4), index built from the same `hp.obo`. Also shown with longest-match output, which keeps only the longest of nested matches. |
| [PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger) v1.2 | Hybrid dictionary and deep-learning tagger (Bioformer model, threshold 0.95, longest-match off). Its NegBio negation step is not used. |
| [NCBO Annotator](https://bioportal.bioontology.org/annotator) | BioPortal's dictionary-based annotation web service, run with the HP ontology. |
| [OntoGPT](https://github.com/monarch-initiative/ontogpt) | LLM extraction with SPIRES (`gpt-oss-120b`, temperature 0) using a template that quotes every phenotype mention and names it as a short HPO-style term. Grounded by OntoGPT's own lookup of HPO labels and synonyms. |
| ontogpt+belink (fill) | The same OntoGPT extractions. Mentions OntoGPT could not ground are linked by BeLink (SapBERT retrieval + re-ranking); OntoGPT's own HPO ids are kept for the rest. |
| **ontogpt+belink (all)** | The same OntoGPT extractions, all linked by BeLink; OntoGPT's own ids are not used. A mention BeLink declines (*None of the above*) is left ungrounded, i.e. BeLink abstains. |

### Metrics

- **Document level:** whether each HPO concept is present in an observation, regardless of where or how often it is mentioned. This measures concept recognition and normalisation, not boundary detection. There are 604 gold concepts at this level, since a few concepts occur twice in one observation.
- **Mention level:** each mention with its position in the text. A prediction is a true positive if it has the same HPO id as a gold mention and the same character boundaries (*exact*) or overlapping ones (*overlap*). Each gold mention can be matched only once.
- **Locating OntoGPT's quotes.** OntoGPT returns quotes, not positions, so each quote is located in the text: verbatim first (ignoring case, spacing and punctuation), then, for quotes the LLM did not copy exactly, word by word in the same order with up to six extra words in between and one-letter typos allowed, within one sentence (*"Flat nasal bridge"* is found in *"Flat, short nasal bridge"*, *"alae"* in *"allae"*). This places 18 more of the 662 quotes; 2 remain unlocated and cannot match at mention level.
- **Ungrounded mentions are abstentions.** A mention left without an HPO concept (OntoGPT's lookup failed and, in the BeLink pipelines, BeLink answered *None of the above*) is not counted as a prediction. BeLink is designed to abstain when no candidate fits; the gold concept it missed still counts as a false negative, so recall is unaffected. The baselines output only concepts they matched, so the rule only affects the OntoGPT rows. Scores with ungrounded mentions counted as false positives are given [below](#secondary-scoring-ungrounded-mentions-as-false-positives).

**What the totals count.** Recall and precision have different denominators. Recall = TP / (TP + FN) is the share of the **gold** that was found, so TP + FN is always the gold total, the same for every system: **607 gold mentions** (685 annotations minus the 78 negated ones) at mention level, and **604 gold concepts** at document level, where each concept counts once per observation (three concepts are annotated twice in one observation). Precision = TP / (TP + FP) is the share of the **predictions** that were right, so TP + FP is the system's own prediction total. For ontogpt+belink (all) with overlapping boundaries:

| Level | TP + FN (gold) | TP + FP (predictions) |
|---|---|---|
| Mention | 460 + 147 = **607** gold mentions | 460 + 116 = 576 predicted mentions |
| Document | 461 + 143 = **604** gold concepts | 461 + 110 = 571 predicted concepts |

The LLM extracted 662 mentions; they reach 576 scored predictions in three steps: 74 that BeLink left ungrounded are dropped as abstentions (662 → 588), 2 outside *Phenotypic abnormality* are removed (588 → 586), and identical predictions are merged (586 → 576), since a mention is scored as (concept, start, end) and repeated quotes, or quotes that could not be located, can give the same triple twice. At document level the predictions merge the same way, to 571 distinct concepts per observation.

### Results

Best values in bold. † FastHPOCR with longest-match output (`longestMatch=True`), a setting chosen on this corpus and shown for reference; the FastHPOCR row is its default configuration.

**Document level**

| System | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| FastHPOCR | 369 | 278 | 235 | 0.570 | 0.611 | 0.590 |
| FastHPOCR, longest match † | 362 | 179 | 242 | 0.669 | 0.599 | 0.632 |
| PhenoTagger | 424 | 300 | 180 | 0.586 | 0.702 | 0.639 |
| NCBO Annotator | 277 | 123 | 327 | 0.693 | 0.459 | 0.552 |
| OntoGPT | 323 | **71** | 281 | **0.820** | 0.535 | 0.647 |
| ontogpt+belink (fill) | 450 | 132 | 154 | 0.773 | 0.745 | 0.759 |
| **ontogpt+belink (all)** | **461** | 110 | **143** | 0.807 | **0.763** | **0.785** |

**Mention level, overlapping boundaries**

| System | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| FastHPOCR | 369 | 353 | 238 | 0.511 | 0.608 | 0.555 |
| FastHPOCR, longest match † | 362 | 212 | 245 | 0.631 | 0.596 | 0.613 |
| PhenoTagger | 424 | 318 | 183 | 0.571 | 0.699 | 0.629 |
| NCBO Annotator | 278 | 132 | 329 | 0.678 | 0.458 | 0.547 |
| OntoGPT | 322 | **75** | 285 | **0.811** | 0.530 | 0.641 |
| ontogpt+belink (fill) | 449 | 139 | 158 | 0.764 | 0.740 | 0.751 |
| **ontogpt+belink (all)** | **460** | 116 | **147** | 0.799 | **0.758** | **0.778** |

**Mention level, exact boundaries**

| System | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| FastHPOCR | 337 | 385 | 270 | 0.467 | 0.555 | 0.507 |
| FastHPOCR, longest match † | 325 | 249 | 282 | 0.566 | 0.535 | 0.550 |
| PhenoTagger | **380** | 362 | **227** | 0.512 | **0.626** | **0.563** |
| NCBO Annotator | 260 | **150** | 347 | **0.634** | 0.428 | 0.511 |
| OntoGPT | 223 | 174 | 384 | 0.562 | 0.367 | 0.444 |
| ontogpt+belink (fill) | 304 | 284 | 303 | 0.517 | 0.501 | 0.509 |
| ontogpt+belink (all) | 321 | 255 | 286 | 0.557 | 0.529 | 0.543 |

**Effect of grounding with BeLink** (same extracted mentions)

| | OntoGPT | ontogpt+belink (fill) | ontogpt+belink (all) | Change (fill) | Change (all) |
|---|---:|---:|---:|---:|---:|
| Mentions with an HPO id | 398 / 662 (60.1%) | 595 / 662 (89.9%) | 588 / 662 (88.8%) | +197 | +190 |
| Document-level precision | 0.820 | 0.773 | 0.807 | −0.047 | −0.013 |
| Document-level recall | 0.535 | 0.745 | 0.763 | +0.210 | +0.228 |
| Document-level F1 | 0.647 | 0.759 | 0.785 | +0.112 | +0.138 |
| Mention-level F1, overlap | 0.641 | 0.751 | 0.778 | +0.110 | +0.137 |
| Mention-level F1, exact | 0.444 | 0.509 | 0.543 | +0.065 | +0.099 |

**Mentions with an HPO id** is grounding coverage: how many of OntoGPT's extracted mentions ended up with a real HPO concept rather than an `AUTO:` placeholder.

**662** is the number of phenotype mentions the LLM extracted across the 454 observations, i.e. quote-and-term pairs. All three pipelines use exactly these mentions. They differ only in who assigns the HPO ID.

| Column | Count | What it is |
|---|---|---|
| OntoGPT | 398 / 662 (60.1%) | OntoGPT's own lookup found an exact HPO label or synonym for 398 terms; the other 264 stayed `AUTO:` |
| ontogpt+belink (fill) | 595 / 662 (89.9%) | OntoGPT's 398, plus 197 of the 264 that BeLink grounded. BeLink declined the other 67 |
| ontogpt+belink (all) | 588 / 662 (88.8%) | BeLink re-linked all 662 and declined 74 |
| Change (fill), Change (all) | +197, +190 | Difference from OntoGPT's 398 |

The 588 in "all" breaks down as:
- **From OntoGPT's 398:** BeLink gave an ID to 393 and declined 5.
- **From the 264 OntoGPT couldn't ground:** BeLink grounded 195. That leaves the 69 that neither grounded.
- **Total:** 393 + 195 = 588.

So "all" grounds 7 fewer mentions than "fill" but still scores better. It's more selective, and it replaced some of OntoGPT's wrong IDs (26 fixed, 6 broken).

Three things to keep in mind:
- **Coverage, not correctness.** It counts mentions that got some HPO concept, right or wrong. Of BeLink's IDs, 85.7% are right on OntoGPT's grounded mentions and 67.2% on the ones only BeLink grounded.
- **Counted before scoring filters.** It counts mentions before the branch filter (2 off-branch IDs dropped) and before duplicate concepts within an observation are merged. That's why it doesn't equal TP + FP in the tables.
- **197 vs 195.** The same 264 ungrounded mentions were sent to BeLink in both runs: 197 got an ID in "fill" and 195 in "all". That gap is slight run-to-run variation in BeLink inference, not a difference in method.

**Ablation: OntoGPT's term in the re-ranker prompt** (ontogpt+belink (all))

By default the re-ranker sees the quote alone (*mild up-slant*). In this ablation it sees the quote followed by OntoGPT's term, *mild up-slant (upslanted palpebral fissure)*. Retrieval is the same in both: the query combines the quote (weight 0.6) and OntoGPT's term (0.4).

| Re-ranker sees | Doc TP | Doc FP | Doc FN | Doc P / R / F1 | Mention F1, overlap | Mention F1, exact | Grounded mentions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quote only (main setting) | 461 | **110** | 143 | **0.807** / 0.763 / 0.785 | **0.778** | **0.543** | 588 |
| Quote (OntoGPT's term) | **468** | 119 | **136** | 0.797 / **0.775** / **0.786** | 0.775 | 0.527 | 601 |

| On the 398 mentions OntoGPT grounded itself | Quote only | Quote (OntoGPT's term) |
|---|---:|---:|
| BeLink keeps OntoGPT's id | 351 | 374 |
| BeLink changes it: fixed / broken | 26 / 6 | 20 / 1 |
| BeLink declines | 5 | 0 |
| Net change in correct ids | +18 | +19 |

With the term, BeLink follows OntoGPT more closely and abstains less: it grounds 13 more mentions, which raises recall and lowers precision by about the same amount. Paired bootstrap over the 454 observations (10,000 samples): document F1 +0.001 (95% CI −0.014 to +0.017, p = 0.89), overlap mention F1 −0.003 (−0.018 to +0.013, p = 0.69), exact mention F1 −0.016 (−0.029 to −0.004, p = 0.008). The mention ids differ on only 30 of 662 mentions (12 vs 18 correct; McNemar p = 0.36). The quote-only setting is kept as the main configuration: it is how the re-ranker was trained, and it is not worse on any measure.

**Error analysis** (ontogpt+belink (all), quote only, document level)

Each missed gold concept is assigned to the first step that failed, using BeLink's log of the options it was shown; each false positive by what the quote overlaps in the gold.

| False negatives (143) | | False positives (110) | |
|---|---:|---|---:|
| not extracted by the LLM | 17 | wrong concept: too general (ancestor of the gold concept) | 27 |
| retrieval miss (gold concept not among the options) | 45 | wrong concept: too specific (descendant) | 14 |
| re-ranker declined (gold concept was an option) | 21 | wrong concept: sibling / unrelated | 8 / 25 |
| re-ranker chose another option | 60 | no gold mention at that position | 32 |
| of which one step too general / too specific | 21 / 7 | finding negated in the text | 4 |

A third of the misses are retrieval misses, which no change to the re-ranker can fix, and the most common re-ranker error is choosing the parent of the gold concept (*Cryptorchidism* for *Bilateral cryptorchidism*). Counting a predicted concept as correct when it is one or two HPO steps from a missed gold concept and its quote overlaps that concept's gold span (38 pairs) gives a lenient document F1 of 0.849.

**Lenient scoring.** A near miss pairs a wrong predicted concept with a missed gold concept in the same observation when one is the ancestor or descendant of the other within two HPO steps and the predicted concept's quote overlaps the missed concept's gold span. Pairs are matched one-to-one, closest first, and each pair counts as a true positive instead of a false positive and a false negative.

| Document level | Quote only (main setting) | Quote (OntoGPT's term) |
|---|---:|---:|
| Strict P / R / F1 | 0.807 / 0.763 / 0.785 | 0.797 / 0.775 / 0.786 |
| Near-miss pairs | 38 | 38 |
| Lenient P / R / F1 | 0.874 / 0.826 / 0.849 | 0.862 / 0.838 / 0.850 |

#### Secondary scoring: ungrounded mentions as false positives

The same runs, with every ungrounded mention counted as a false positive (as plain OntoGPT output would be used). Baseline rows are unchanged. Mention-level columns are from before the quotes were located word by word.

| System | Doc TP / FP / FN | Doc P / R / F1 | Mention F1, overlap | Mention F1, exact |
|---|---:|---:|---:|---:|
| OntoGPT | 323 / 329 / 281 | 0.495 / 0.535 / 0.514 | 0.502 | 0.348 |
| ontogpt+belink (fill) | 450 / 196 / 154 | 0.697 / 0.745 / 0.720 | 0.702 | 0.475 |
| ontogpt+belink (all) | 461 / 180 / 143 | 0.719 / 0.763 / 0.741 | 0.722 | 0.504 |
| ontogpt+belink (all), quote (OntoGPT's term) | 468 / 181 / 136 | 0.721 / 0.775 / 0.747 | 0.727 | 0.494 |

### Findings

- **ontogpt+belink (all) recognises the most phenotypes.** It has the highest F1 at document level (0.785) and at mention level with overlapping boundaries (0.778), +0.146 and +0.149 over the strongest baseline, PhenoTagger, with the highest recall at both levels.
- **OntoGPT's own lookup is precise but grounds too little.** It needs HPO's exact wording ("posteriorly rotated ear" fails where "posteriorly rotated ears" succeeds), so only 60% of its mentions get a concept. Those it grounds are mostly right: OntoGPT alone has the highest precision of all systems (0.820) but recall 0.535. BeLink grounds the rest: +0.228 recall for −0.013 precision, +0.138 document F1.
- **BeLink also improves on OntoGPT's lookup where the lookup succeeds.** Re-linking every mention (*all*) beats keeping OntoGPT's ids and filling the gaps (*fill*) on every measure: +0.026 document F1, +0.034 precision. Of the 398 ids OntoGPT found, BeLink changes 42, fixing 26 and breaking 6.
- **Abstaining works as a filter.** BeLink leaves 74 of 662 mentions ungrounded. Counted as false positives instead of abstentions, they would lower document F1 from 0.785 to 0.741; recall is the same either way.
- **Most remaining errors are retrieval misses or too-general choices.** A third of the missed concepts never reach the re-ranker (45 of 143), and the most common re-ranker error is choosing the gold concept's parent.
- **Exact boundaries favour the dictionary tools.** With exact boundaries, PhenoTagger leads (0.563), ahead of ontogpt+belink (all) (0.543). About a third of OntoGPT's correctly located mentions do not match the gold boundaries exactly, against 6–10% for the baselines: the LLM quotes the clinical phrase as written, including degree, laterality and location words (*"very light capillary malformation over left extremity and trunk"*), whereas the annotators marked the core term.
- **FastHPOCR benefits from longest-match output.** Dropping matches nested in a longer one (e.g. *Abnormal thumb* inside *Abnormal thumb morphology*) removes 99 document-level false positives for 7 lost true positives, raising document F1 from 0.590 to 0.632, close to PhenoTagger (0.639).
- **Adding OntoGPT's term to the re-ranker prompt does not help.** BeLink abstains less and follows OntoGPT more closely, with no change in document or overlap F1 and a small loss with exact boundaries (−0.016, p = 0.008).
- **NCBO is precise but misses many findings.** Exact dictionary matching rarely produces wrong concepts, but it finds less than half of the gold mentions.

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
| `rerank`, `rerank_batch`, `belink-hpo rerank`, `belink-hpo link`, `HPOLinker` | The package, from your candidates | Only if you pass `preferred_names` / `--preferred-names` |
| `belink-hpo build-dataset`, `prepare_train`, `prepare_test` | The package, from your examples | Only if you pass `--preferred-names` / `preferred_names` |
| `predict_labels`, `belink-hpo predict` | You, beforehand | Not applied: prompts are scored exactly as written |

`predict` is meant for files of ready-made prompts, such as the `test.jsonl` written by `build-dataset`. If that file was built with preferred names, they are already in the option text.

### From the command line

```bash
belink-hpo rerank --input mentions.jsonl --out linked.jsonl --preferred-names index/preferred_names.json
```

`mentions.jsonl` has one JSON object per line with `context`, `mention` and `candidates` (see [`examples/examples.jsonl`](examples/examples.jsonl)). If a line also has `gold_id`, accuracy is printed.

## Full pipeline: expansion, retrieval, re-ranking

The re-ranker needs candidates. The package includes the pipeline that produced the reported numbers, in three steps:

1. **Query expansion.** An LLM is given the mention and its sentence and asked for the standard HPO name of the concept, e.g. `mild up-slant` → `Upslanted palpebral fissure`.
2. **Retrieval.** [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) embeds the mention and the standard name; the query is `0.6 × mention + 0.4 × standard name`, searched against every HPO name and synonym. The top 30 names are kept.
3. **Re-ranking.** The fine-tuned model picks one of the first 10 concepts, or "None of the above".

Expansion is optional. Without it, retrieval uses the mention text alone and finds the right concept less often, so the top-1 retrieval figure in [Results](#results) only holds with expansion.

### Build the index once

Details in [docs/TRAINING.md](docs/TRAINING.md#1-hpo-name-index).

```bash
belink-hpo prepare-obo --ontology_path hp.obo --output_dir index
belink-hpo embed-terms --ontology index/terminology.json.gz --out_vectors index/hpo_vec.npy
```

### Link mentions from the command line

`mentions.jsonl` has one JSON object per line with `context` and `mention`:

```bash
pip install "belink-hpo[expand] @ git+https://github.com/dash-ka/BeLink-HPO.git"
export OPENAI_API_KEY=...

belink-hpo link --input mentions.jsonl --out linked.jsonl \
    --kb-vectors index/hpo_vec.npy --preferred-names index/preferred_names.json --expand
```

Each output line has the mention, the generated `standard_name`, and the chosen `id`, `name` and `confidence`. If the input lines have `gold_id`, accuracy is printed.

- **Without `--expand`**, no LLM is called. A `standard_name` already present in an input line is still used.
- **`--llm`** chooses the model (default `gpt-5.4`, as for the reported numbers). **`--llm-base-url`** points at any OpenAI-compatible server, such as a local vLLM; the `OPENAI_BASE_URL` environment variable works too.
- **`--top-k`** and **`--alpha`** change how many names are retrieved and the weight of the mention against the standard name.

Expansion sends each mention and its sentence to the LLM provider. With clinical text, check that this is allowed, or use a model you host yourself through `--llm-base-url`.

### Link mentions in Python

```python
from belink_hpo import CandidateRetriever, HPOLinker, HPOReranker, StandardNameGenerator
from belink_hpo.ontology import load_preferred_names

linker = HPOLinker(
    CandidateRetriever.from_files("index/hpo_vec.npy", "index/ontology_mapping.txt"),
    HPOReranker.from_pretrained(preferred_names=load_preferred_names("index/preferred_names.json")),
    expander=StandardNameGenerator(),        # leave out to skip expansion
)

result = linker.link(context="EYES: mild epicanthus, mild up-slant.", mention="mild up-slant")
result.id      # e.g. "HP:0000582"
```

`linker.link(..., standard_name="Upslanted palpebral fissure")` uses your own standard name instead of calling the LLM. The expander can also be any function `(mention, context) -> name`, for example one calling another LLM API. Generated names are cached, so a mention repeated in the same context is sent only once.

### Generate standard names separately

To run each step on its own, as was done to build the training data, `belink-hpo expand` adds a `standard_name` to every record of a JSON file of mentions, and `belink-hpo retrieve --apply_grf` then uses it. See [docs/TRAINING.md](docs/TRAINING.md#3-standard-names-for-the-mentions).

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
| `belink-hpo link` | Expansion, retrieval and re-ranking for a JSONL file of mentions |
| `belink-hpo rerank` | Choose an HPO id for each mention in a JSONL file whose candidates you already have |
| `belink-hpo predict` | Predict option letters for a file of ready-made prompts, and report accuracy |
| `belink-hpo train` | Fine-tune a LoRA adapter |
| `belink-hpo build-dataset` | Turn examples with candidates into training and test prompts |
| `belink-hpo prepare-obo` | Turn an OBO ontology into the list of names to index |
| `belink-hpo embed-terms` | Embed ontology names into a retrieval index |
| `belink-hpo retrieve` | Retrieve candidates for a JSON file of mentions |
| `belink-hpo expand` | Generate standard names for a JSON file of mentions with an LLM |

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
