#!/usr/bin/env bash
# Create the four environments and download the fixed inputs. Run once, from evaluation/:
#   bash setup_envs.sh
# Re-running skips what already exists.
set -euo pipefail
cd "$(dirname "$0")"
source config.sh

mkvenv() {   # mkvenv <name> <requirements file>
  local v="$VENVS/$1"
  if [ ! -x "$v/bin/python" ]; then
    echo "== creating $v"
    "$PY310" -m venv "$v"
    "$v/bin/pip" install -U pip wheel
  fi
  (cd requirements && "../$v/bin/pip" install -r "$2")
}

mkvenv eval eval.txt              # OntoGPT, FastHPOCR, scoring
mkvenv belink belink.txt          # BeLink relinking (GPU)
mkvenv phenotagger phenotagger.txt
mkvenv vllm vllm.txt              # LLM server (GPU)

# ---- HPO release used for gold mapping, FastHPOCR, the branch filter and the BeLink index
mkdir -p data
if [ ! -s "$HPO" ]; then
  curl -L -o "$HPO" "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v${HPO_RELEASE}/hp.obo"
fi
grep -m1 "^data-version:" "$HPO"

# ---- corpus: not redistributed here; copy it in
[ -s "$CORPUS" ] || echo "!! copy the BioCreative VIII Track 3 dev file to $CORPUS"

# ---- PhenoTagger code and models
if [ ! -d "$PHENOTAGGER_DIR/.git" ]; then
  git clone https://github.com/ncbi-nlp/PhenoTagger.git "$PHENOTAGGER_DIR"
  [ -n "$PHENOTAGGER_COMMIT" ] && git -C "$PHENOTAGGER_DIR" checkout "$PHENOTAGGER_COMMIT"
fi
"$VENVS/phenotagger/bin/python" -c "import nltk; [nltk.download(x, quiet=True) for x in ['punkt','averaged_perceptron_tagger','wordnet','omw-1.4']]"
if [ ! -s "$PHENOTAGGER_DIR/models/bioformer_PT_v1.2.h5" ]; then
  "$VENVS/eval/bin/hf" download lingbionlp/PhenoTagger_v1.2 bioformer_PT_v1.2.h5 --local-dir "$PHENOTAGGER_DIR/models"
  "$VENVS/eval/bin/hf" download bioformers/bioformer-8L --local-dir "$PHENOTAGGER_DIR/models/bioformer-cased-v1.0"
fi

# ---- BeLink retrieval index: the published one if configured, else built from the same hp.obo
if [ ! -s "$BELINK_INDEX/hpo_vec.npy" ] && [ -n "$BELINK_INDEX_REPO" ]; then
  "$VENVS/eval/bin/hf" download "$BELINK_INDEX_REPO" --repo-type dataset \
    ${BELINK_INDEX_REVISION:+--revision "$BELINK_INDEX_REVISION"} --local-dir "$BELINK_INDEX"
fi
if [ ! -s "$BELINK_INDEX/hpo_vec.npy" ]; then
  "$VENVS/belink/bin/belink-hpo" prepare-obo --ontology_path "$HPO" --output_dir "$BELINK_INDEX"
  CUDA_VISIBLE_DEVICES=$BELINK_GPU "$VENVS/belink/bin/belink-hpo" embed-terms \
    --ontology "$BELINK_INDEX/terminology.json.gz" --out_vectors "$BELINK_INDEX/hpo_vec.npy"
fi

# ---- BeLink re-ranker adapter, at the pinned revision
if [ ! -s "$BELINK_ADAPTER/adapter_config.json" ]; then
  "$VENVS/eval/bin/hf" download "$BELINK_ADAPTER_ID" --revision "$BELINK_ADAPTER_REVISION" --local-dir "$BELINK_ADAPTER"
fi
echo "Setup done."
