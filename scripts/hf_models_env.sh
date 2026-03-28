#!/usr/bin/env bash
# Local HF weight roots (flat snapshot dirs). Override HF_MODELS before sourcing if needed.
: "${HF_MODELS:=/sg-pvc/hfmodels}"
export HF_MODELS
export MATH_STUDENT_15="$HF_MODELS/Qwen_Qwen2.5-Math-1.5B"
export QWEN3_17="$HF_MODELS/Qwen_Qwen3-1.7B"
export QWEN3_4="$HF_MODELS/Qwen_Qwen3-4B"
export QWEN3_8="$HF_MODELS/Qwen_Qwen3-8B"
