#!/usr/bin/env bash
# The whole evaluation, in the order the reported results were made. From evaluation/:
#   bash serve_llm.sh            (in another terminal; only needed for step 1)
#   bash run_all.sh              (all steps)
#   bash run_all.sh 4 7          (steps 4 to 7 only)
# Steps: 1 extract  2 relink  3 locate quotes word by word  4 score  5 ablation
#        6 significance  7 error analysis
# System outputs are cached under $OUT, so rescoring steps take seconds.
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
FROM=${1:-1}; TO=${2:-7}
step() { [ "$1" -ge "$FROM" ] && [ "$1" -le "$TO" ]; }
mkdir -p "$OUT/full" "$OUT/term"

EVAL="$VENVS/eval/bin/python"
BEL="$VENVS/belink/bin/python"
R="$OUT/ontogpt_gptoss"
F="$R/predictions_relinked_fill.tsv"
A="$R/predictions_relinked_all.tsv"
T="$R/predictions_relinked_all_term.tsv"
COMMON=(--corpus "$CORPUS" --hpo "$HPO" --drop-ungrounded)
BASELINES=(--systems fasthpocr,fasthpocr:longest,phenotagger,ncbo
           --phenotagger-dir "$PHENOTAGGER_DIR" --phenotagger-python "$PWD/$VENVS/phenotagger/bin/python"
           --pt-gpu "$PT_GPU")

# 1. OntoGPT extraction (quotes located verbatim, as in the reported run; step 3 adds the rest)
if step 1; then
  curl -sf "http://localhost:$LLM_PORT/v1/models" >/dev/null || { echo "vLLM is not running: bash serve_llm.sh"; exit 1; }
  OPENAI_API_KEY=dummy "$EVAL" run_ontogpt_mentions.py --corpus "$CORPUS" --hpo "$HPO" --out "$R" \
    --template phenotype_mentions.yaml --model "$LLM_NAME" \
    --extra "--model-provider openai --api-base http://localhost:$LLM_PORT/v1 --temperature 0" --workers 4
fi

# 2. BeLink relinking: fill, all, and all with OntoGPT's term shown to the re-ranker (ablation)
if step 2; then
  [ -s "$BELINK_ADAPTER/adapter_config.json" ] || { echo "re-ranker adapter missing: bash setup_envs.sh"; exit 1; }
  echo "BeLink re-ranker: $BELINK_ADAPTER_ID @ $BELINK_ADAPTER_REVISION ($BELINK_ADAPTER)"
  export BELINK_INDEX BELINK_ADAPTER CUDA_VISIBLE_DEVICES=$BELINK_GPU
  for mode_out in "ungrounded $F" "all $A"; do
    set -- $mode_out
    "$BEL" relink_mentions.py --corpus "$CORPUS" --pred "$R/predictions.tsv" --linker belink_linker.py \
      --mode "$1" --out "$2"
  done
  BELINK_RERANK_TERM=all "$BEL" relink_mentions.py --corpus "$CORPUS" --pred "$R/predictions.tsv" \
    --linker belink_linker.py --mode all --out "$T"
  unset CUDA_VISIBLE_DEVICES
fi

# 3. Locate the quotes the LLM did not copy verbatim, and copy the positions into the relinked files
#    (BeLink's ids unchanged, as in the report)
if step 3; then
  cp "$R/predictions.tsv" "$R/predictions.exact_locate.tsv"
  "$EVAL" run_ontogpt_mentions.py --corpus "$CORPUS" --hpo "$HPO" --out "$R" --collect-only --fuzzy-locate
  "$EVAL" update_positions.py --pred "$R/predictions.tsv" "$F" "$A" "$T"
fi

# 4. Score all systems: ungrounded mentions as abstentions, exact and overlap boundaries
if step 4; then
  for mm in exact overlap; do
    "$EVAL" eval_baselines.py "${COMMON[@]}" --out "$OUT/full" "${BASELINES[@]}" \
      --ontogpt "$F" --ontogpt-all "$A" --mention-match "$mm" | tee "$OUT/full/log_$mm.txt"
  done
fi

# 5. Ablation: OntoGPT's term in the re-ranker prompt
if step 5; then
  for mm in exact overlap; do
    "$EVAL" eval_baselines.py "${COMMON[@]}" --out "$OUT/term" --systems "" \
      --ontogpt "$F" --ontogpt-all "$T" --mention-match "$mm" | tee "$OUT/term/log_term_$mm.txt"
  done
fi

# 6. Paired significance test, quote only vs quote (term)
if step 6; then
  for mm in exact overlap; do
    "$EVAL" eval_baselines.py "${COMMON[@]}" --out "$OUT/full" --systems "" --compare "$A" "$T" \
      --mention-match "$mm" | tee "$OUT/full/log_compare_$mm.txt"
  done
fi

# 7. Error analysis (document level)
if step 7; then
  "$EVAL" error_analysis.py "${COMMON[@]}" --pred "$A" --out "$OUT/errors_all_grounded"
  "$EVAL" error_analysis.py "${COMMON[@]}" --pred "$T" --out "$OUT/errors_all_term_grounded"
fi
echo "Done (steps $FROM-$TO). Logs in $OUT/full and $OUT/term."
