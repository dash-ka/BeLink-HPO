#!/usr/bin/env bash
# Serve the LLM for OntoGPT with vLLM (keep it running, e.g. in tmux):  bash serve_llm.sh
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
CUDA_VISIBLE_DEVICES=$LLM_GPU "$VENVS/vllm/bin/vllm" serve "$LLM_MODEL" \
  --served-model-name "$LLM_NAME" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --port "$LLM_PORT" \
  --override-generation-config '{"max_new_tokens": 16384}'
