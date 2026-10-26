#!/bin/bash
# Distributed-pipeline worker entrypoint; service startup follows the resolved config.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$PROJECT_ROOT"

if [[ " ${*:-} " == *" --help "* || " ${*:-} " == *" -h "* ]]; then
    exec "${PYTHON_BIN:?PYTHON_BIN is required}" -m speech_edit.runtime.entry "$@"
fi

RUN_CONFIG=${RUN_CONFIG:?RUN_CONFIG is required}
PIPELINE=${PIPELINE:?PIPELINE is required}
STAGE=${STAGE:?STAGE is required}
WORKER_ID=${WORKER_ID:-"${STAGE}-$(date +%s)-$$"}
PYTHON_BIN=${PYTHON_BIN:?PYTHON_BIN is required}

# Effective configs pin the submitting process's artifact root. Resolve it before
# sourcing the common environment so restarted workers do not depend on the
# caller's shell environment.
ARTIFACT_DIR=$("$PYTHON_BIN" -c 'import sys; from speech_edit.runtime.config import load_run_config; print(load_run_config(sys.argv[1])["artifact_dir"])' "$RUN_CONFIG")
export ARTIFACT_DIR
source "$PROJECT_ROOT/scripts/env.sh"


exec "$PYTHON_BIN" -m speech_edit.runtime.entry \
    --run-config "$RUN_CONFIG" \
    --pipeline "$PIPELINE" \
    --stage "$STAGE" \
    --worker-id "$WORKER_ID" \
    "$@"
