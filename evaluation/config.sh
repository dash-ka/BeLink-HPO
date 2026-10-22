# Settings for setup_envs.sh and run_all.sh. Edit, then: source config.sh is done by the scripts.
# Paths are relative to this folder (evaluation/) unless absolute.

# ---- inputs
CORPUS=data/Clinical-Genetics-Physical-Exam-Data-Dev.tsv     # BioCreative VIII Track 3, dev set
HPO_RELEASE=2026-09-01                                         # hp.obo used for every reported run
HPO=data/hp.obo
OUT=results                                                    # all outputs go here

# ---- environments (created by setup_envs.sh)
VENVS=venvs
PY310=python3.10                                               # all four venvs used Python 3.10.12

# ---- LLM served by vLLM
LLM_MODEL=openai/gpt-oss-120b
LLM_NAME=gpt-oss-120b
LLM_PORT=8000
LLM_GPU=0

# ---- BeLink
BELINK_GPU=1
BELINK_INDEX=data/belink_index   # retrieval index; setup_envs.sh fills it only if it is missing
# The exact index of the reported runs, published as a Hugging Face dataset (see README, "Publish the
# index"). When set, setup_envs.sh downloads it at this revision; when empty, it builds the index
# from $HPO instead (equivalent if built from the same hp.obo with the same belink-hpo version).
BELINK_INDEX_REPO=Dash00/belink-hpo-index
BELINK_INDEX_REVISION=6b9ae0dbdcdf3cbbd5a9a8b9cd8ce18b646b6b8f
# The re-ranker adapter trained on HPO-GSC+ only (the evaluation corpus unseen), pinned to the
# revision used for the reported runs. setup_envs.sh downloads it to $BELINK_ADAPTER, a local
# folder, so later changes on the Hub cannot change the results.
BELINK_ADAPTER_ID=Dash00/qwen3-4b-lora-hpo-gsc-reranker
BELINK_ADAPTER_REVISION=91fda211eba5f5eccfb731b93cc40b88c00f9391
BELINK_ADAPTER=adapters/qwen3-4b-lora-hpo-gsc-reranker

# ---- baselines
PHENOTAGGER_DIR=tools/PhenoTagger
PHENOTAGGER_COMMIT=10f8ee3e1ea4c26fb44c47df28a555082748ebf2   # commit used for the reported runs
PT_GPU=1
# NCBO: export NCBO_API_KEY=<your key> in your shell (never write it in this file)
