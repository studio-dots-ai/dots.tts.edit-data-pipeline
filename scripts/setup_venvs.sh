#!/usr/bin/env bash
# Explicitly install selected project environments using the checked-in locks.
# Existing environments can instead be reused through absolute interpreter paths.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"
DRY_RUN=false
PYTHON_BIN=""
SELECTED_ENVS=()
usage() {
    echo "Usage: bash scripts/setup_venvs.sh [--env NAME] [--python PATH] [--dry-run]"
    echo "Without --env, sync all environments. Install uv and system build dependencies first."
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --env|--python)
            [[ -n "${2:-}" && "${2:-}" != --* ]] || { usage >&2; exit 2; }
            if [[ "$1" == --env ]]; then SELECTED_ENVS+=("$2"); else PYTHON_BIN="$2"; fi
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ENV_DIRS=()
if [[ ${#SELECTED_ENVS[@]} -eq 0 ]]; then
    for project in "$PROJECT_ROOT"/envs/*/pyproject.toml; do
        [[ -f "$project" ]] && ENV_DIRS+=("$(dirname "$project")")
    done
else
    for name in "${SELECTED_ENVS[@]}"; do
        [[ "$name" != */* && -f "$PROJECT_ROOT/envs/$name/pyproject.toml" ]] || {
            echo "Unknown environment: $name" >&2; exit 2;
        }
        ENV_DIRS+=("$PROJECT_ROOT/envs/$name")
    done
fi
[[ ${#ENV_DIRS[@]} -gt 0 ]] || { echo "No environment definitions found" >&2; exit 1; }
for env_dir in "${ENV_DIRS[@]}"; do
    args=(sync --frozen --project "$env_dir")
    [[ -z "$PYTHON_BIN" ]] || args+=(--python "$PYTHON_BIN")
    if $DRY_RUN; then
        printf 'uv'; printf ' %q' "${args[@]}"; printf '\n'
    else
        command -v uv >/dev/null || { echo "Install uv before running setup." >&2; exit 1; }
        uv "${args[@]}"
    fi
done
