# Evaluation: phenotype recognition with OntoGPT and BeLink

Scripts and pinned environments to reproduce the evaluation in the main README: OntoGPT
(gpt-oss-120b) extracts phenotype mentions from the BioCreative VIII Track 3 dev set, BeLink
grounds them to HPO, and the result is compared with FastHPOCR, PhenoTagger and the NCBO
Annotator at document and mention level.

## Environments

Four virtual environments, all Python 3.10, because their requirements conflict
(PhenoTagger needs transformers 4.30 and TensorFlow 2.12; BeLink needs transformers ≥ 4.51;
vLLM pins its own torch build).

| venv | Requirements | Used by | GPU |
|---|---|---|---|
| `eval` | `requirements/eval.txt` (OntoGPT 1.2.0, FastHPOCR 0.1.4) | extraction, FastHPOCR, NCBO, scoring, error analysis | no |
| `belink` | `requirements/belink.txt` (belink-hpo @ af084c7, torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.2) | relinking | yes |
| `phenotagger` | `requirements/phenotagger.txt` (TensorFlow 2.12.0, transformers 4.30.1) | PhenoTagger baseline | optional |
| `vllm` | `requirements/vllm.txt` (vLLM 0.10.2) | serving gpt-oss-120b | yes |

Each requirements file lists the packages the step needs and pins everything else with a
constraints file (`requirements/constraints/`), made from `pip freeze` of the environments
that produced the reported results (kept unedited in `requirements/frozen/`). The reported
runs used one environment, `belink-test`, for both extraction and relinking; OntoGPT's stack in
`eval` is the same as there (only pydantic differs by a patch release).

Only `run_ontogpt_mentions.py` (through OntoGPT) and `relink_mentions.py` (through
`belink_linker.py`) need anything beyond the Python standard library; `eval_baselines.py`
imports FastHPOCR only for the FastHPOCR runs, and the other scripts use the standard
library only.

## Fixed inputs

| Input | Version |
|---|---|
| Corpus | `Clinical-Genetics-Physical-Exam-Data-Dev.tsv` (not redistributed; copy it to `data/`) |
| HPO | `hp.obo`, release 2026-09-01 (downloaded by `setup_envs.sh`) |
| LLM | `openai/gpt-oss-120b`, temperature 0, served by vLLM 0.10.2 |
| OntoGPT template | `phenotype_mentions.yaml` |
| BeLink code | belink-hpo commit `af084c7a232f02ecc3f29531d120af0b183327cf` |
| BeLink retrieval index | Hugging Face dataset `Dash00/belink-hpo-index` @ `6b9ae0dbdcdf3cbbd5a9a8b9cd8ce18b646b6b8f`, built from the same `hp.obo` with belink-hpo `af084c7`; downloaded to `data/belink_index/` by `setup_envs.sh` (an index already there is used as is) |
| BeLink re-ranker | `Dash00/qwen3-4b-lora-hpo-gsc-reranker` @ `91fda211eba5f5eccfb731b93cc40b88c00f9391`: Qwen3-4B LoRA adapter trained on HPO-GSC+ only, so this corpus was unseen. Downloaded to a local folder by `setup_envs.sh` (the base Qwen3-4B model is fetched from the Hub by the adapter's configuration) |
| PhenoTagger | v1.2 (repository commit `10f8ee3e1ea4c26fb44c47df28a555082748ebf2`), Bioformer model `bioformer_PT_v1.2.h5`, threshold 0.95, NegBio off |
| FastHPOCR | 0.1.4, default configuration, index built from the same `hp.obo` |
| NCBO Annotator | BioPortal REST API, ontology HP (live service) |

## Run

```bash
cd evaluation
# 1. check config.sh (GPUs, paths); copy the corpus to data/
bash setup_envs.sh                  # venvs, hp.obo, PhenoTagger models, BeLink index
export NCBO_API_KEY=<your key>      # BioPortal account; not stored in any file
bash serve_llm.sh                   # in a separate terminal (tmux), only for step 1
bash run_all.sh                     # all steps; or e.g. bash run_all.sh 4 7
```

`run_all.sh` steps, in the order the reported results were made:

1. **Extract** mentions with OntoGPT (`run_ontogpt_mentions.py`); quotes located verbatim.
2. **Relink** with BeLink: `fill` (only mentions OntoGPT could not ground), `all` (every
   mention), and `all` with OntoGPT's term shown to the re-ranker (`BELINK_RERANK_TERM=all`).
3. **Locate** the quotes the LLM did not copy verbatim (`--collect-only --fuzzy-locate`) and copy
   the positions into the relinked files (`update_positions.py`; BeLink's ids unchanged).
4. **Score** all systems, ungrounded mentions as abstentions (`--drop-ungrounded`), exact and
   overlapping mention boundaries.
5. **Ablation**: the term run scored the same way.
6. **Significance**: paired bootstrap and McNemar, quote only vs quote (term).
7. **Error analysis** of both BeLink (all) runs.

Outputs are in `results/`: `full/summary_*.tsv` and `full/log_*.txt`, `term/` (ablation), `full/compare_*.tsv`, `errors_all*_grounded/`.

## Expected results

Document level, negated gold terms excluded, ungrounded mentions as abstentions:

| System | P | R | F1 |
|---|---:|---:|---:|
| FastHPOCR | 0.570 | 0.611 | 0.590 |
| PhenoTagger | 0.586 | 0.702 | 0.639 |
| NCBO Annotator | 0.693 | 0.459 | 0.552 |
| OntoGPT | 0.820 | 0.535 | 0.647 |
| ontogpt+belink (fill) | 0.771 | 0.745 | 0.758 |
| ontogpt+belink (all) | 0.806 | 0.763 | 0.784 |

Mention level, ontogpt+belink (all): F1 0.777 (overlap), 0.542 (exact).

## Publish the index

The retrieval index of the reported runs can be shared as a Hugging Face dataset, so a rerun
uses exactly the same one instead of rebuilding it. Once, from `evaluation/`, with write access
to the Hub (`hf auth login`):

```bash
hf repo create Dash00/belink-hpo-index --repo-type dataset          # once; choose public or private
hf upload Dash00/belink-hpo-index data/belink_index . --repo-type dataset \
  --commit-message "BeLink retrieval index, hp.obo 2026-09-01, belink-hpo af084c7"
```

Then put the repository id in `BELINK_INDEX_REPO` in `config.sh`, and the commit hash of the
upload (shown on the dataset's "Files and versions" page, or by
`python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('Dash00/belink-hpo-index').sha)"`)
in `BELINK_INDEX_REVISION`. `setup_envs.sh` then downloads that revision into `data/belink_index/`.
Without them, it builds the index from `hp.obo`.

## What can differ in a rerun

- **OntoGPT's own grounding** uses the HPO database that OntoGPT (oaklib) downloads on first
  use (`sqlite:obo:hp`), which is not version-pinned. Check its date in `~/.data/oaklib/`.
- **The LLM output**: temperature 0 is close to deterministic but not guaranteed across vLLM
  versions or GPUs. Keep the OntoGPT outputs (`results/ontogpt_gptoss/ontogpt/`) to rescore
  without new model calls.
- **BeLink** inference varies slightly between runs (195 vs 197 mentions grounded from the same
  input in two runs).
- **NCBO** is a live service whose HPO version changes; its raw outputs are cached in
  `results/full/ncbo/` after the first run.

Keeping `results/` (or at least the cached system outputs and the relinked files) makes every
scoring step exactly reproducible.

## Files

| File | Role |
|---|---|
| `run_ontogpt_mentions.py` | runs OntoGPT per observation, locates quotes, writes `predictions.tsv`; shared corpus loader and scorer |
| `ontogpt_fixed.py` | OntoGPT launcher that fixes replies with an empty field or one item per line |
| `phenotype_mentions.yaml` | OntoGPT template: every phenotype mention as a quote and a short HPO-style term |
| `relink_mentions.py` | sends OntoGPT's mentions to a linker (`--mode ungrounded` or `all`) |
| `belink_linker.py` | BeLink as that linker; settings by `BELINK_*` environment variables |
| `update_positions.py` | copies re-located quote positions into relinked files |
| `eval_baselines.py` | runs the baselines, scores every system, `--compare` significance test |
| `error_analysis.py` | document-level error categories from BeLink's candidate log |
| `config.sh`, `setup_envs.sh`, `serve_llm.sh`, `run_all.sh` | settings, setup, LLM server, pipeline |
