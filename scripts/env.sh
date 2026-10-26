#!/usr/bin/env bash
# Shared runtime environment for all project services and pipeline jobs.
#
# Model inputs use the explicitly configured CHECKPOINTS_DIR.
# Writable caches are kept in the external artifact directory.

if [ -z "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PROJECT_ROOT

if [ -n "${CHECKPOINTS_DIR:-}" ]; then
    export CHECKPOINTS_DIR
    HF_HOME="${HF_HOME:-$CHECKPOINTS_DIR}"
    HF_HUB_CACHE="${HF_HUB_CACHE:-$CHECKPOINTS_DIR}"
    TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CHECKPOINTS_DIR}"
    TORCH_HOME="${TORCH_HOME:-$CHECKPOINTS_DIR/torch}"
    export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME
fi

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

if [ -n "${ARTIFACT_DIR:-}" ]; then
    artifact_cache="$(python3 "$PROJECT_ROOT/speech_edit/runtime/paths.py" \
        --project-root "$PROJECT_ROOT" --scope shared --path cache)" || return 1
    export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$artifact_cache/hf_modules}"
    unset artifact_cache
fi
