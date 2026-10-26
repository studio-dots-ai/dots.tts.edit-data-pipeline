#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

GPUS=""
HOST="0.0.0.0"
PORT="18112"
MODEL_PATH=""
DRAFT_MODEL_PATH=""
SERVED_MODEL_NAME="gemma-4-31b-it-mtp-tp2-32k"
VLLM_PYTHON="$REPO_ROOT/envs/pipe-llm/.venv/bin/python"
PYTHONPATH_OVERLAY="${PYTHONPATH:-}"
GPU_MEMORY_UTILIZATION="0.92"
MAX_MODEL_LEN=""
MAX_NUM_BATCHED_TOKENS=""
MAX_NUM_SEQS=""
TP_SIZE=""
DP_SIZE=""
DP_LOCAL_SIZE=""
TIMEOUT_SECONDS="900"
LOG_DIR=""

usage() {
  cat <<'USAGE'
Usage:
  launch_gemma4_mtp_vllm.sh [--gpus GPU_IDS] [options]

Common options:
  --gpus IDS                     Comma-separated visible GPU ids, e.g. 0,1.
                                 Omit or set to auto to use all visible GPUs.
  --port PORT                    vLLM OpenAI server port. Default: 18112.
  --host HOST                    vLLM bind host. Default: 0.0.0.0.
  --served-model-name NAME       Served model name. Default: gemma-4-31b-it-mtp-tp2-32k.
  --timeout SECONDS              Wait time for /v1/models readiness. Default: 900.
  --log-dir DIR                  Directory for logs and pid files.

Advanced options:
  --model-path PATH              Main Gemma 4 checkpoint path.
  --draft-model-path PATH        MTP assistant checkpoint path.
  --vllm-python PATH             Python binary for the custom vLLM env.
  --gpu-memory-utilization N     vLLM GPU memory fraction. Default: 0.92.
  --max-model-len N              Default: 8192 for 1 GPU, otherwise 32768.
  --max-num-batched-tokens N     Default: 8192 for 1 GPU, otherwise 32768.
  --max-num-seqs N               Default: 4 for 1 GPU, otherwise 8.
  --tensor-parallel-size N       Defaults to GPU count, except 8 GPUs -> 4.
  --data-parallel-size N         Defaults to GPU count / TP size.
  --data-parallel-size-local N   Defaults to data parallel size when DP > 1.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        GPUS="auto"
        shift
      else
        GPUS="$2"
        shift 2
      fi
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --draft-model-path)
      DRAFT_MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --vllm-python)
      VLLM_PYTHON="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TP_SIZE="$2"
      shift 2
      ;;
    --data-parallel-size)
      DP_SIZE="$2"
      shift 2
      ;;
    --data-parallel-size-local)
      DP_LOCAL_SIZE="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LOG_DIR" ]]; then
  echo "Pass --log-dir with a directory under ARTIFACT_DIR." >&2
  exit 2
fi

detect_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -
  else
    echo "0"
  fi
}

if [[ -z "$GPUS" || "$GPUS" == "auto" ]]; then
  GPUS="$(detect_visible_gpus)"
fi

if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "vLLM python not found or not executable: $VLLM_PYTHON" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$DRAFT_MODEL_PATH" ]]; then
  echo "Draft model path not found: $DRAFT_MODEL_PATH" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
GPU_COUNT="${#GPU_ARRAY[@]}"
if [[ "$GPU_COUNT" -lt 1 || -z "${GPU_ARRAY[0]}" ]]; then
  echo "No visible GPUs found" >&2
  exit 2
fi
if [[ -z "$MAX_MODEL_LEN" ]]; then
  if [[ "$GPU_COUNT" -eq 1 ]]; then
    MAX_MODEL_LEN="8192"
  else
    MAX_MODEL_LEN="32768"
  fi
fi
if [[ -z "$MAX_NUM_BATCHED_TOKENS" ]]; then
  if [[ "$GPU_COUNT" -eq 1 ]]; then
    MAX_NUM_BATCHED_TOKENS="8192"
  else
    MAX_NUM_BATCHED_TOKENS="32768"
  fi
fi
if [[ -z "$MAX_NUM_SEQS" ]]; then
  if [[ "$GPU_COUNT" -eq 1 ]]; then
    MAX_NUM_SEQS="4"
  else
    MAX_NUM_SEQS="8"
  fi
fi
if [[ -z "$TP_SIZE" ]]; then
  if [[ "$GPU_COUNT" -ge 8 ]]; then
    TP_SIZE="4"
  else
    TP_SIZE="$GPU_COUNT"
  fi
fi
if [[ -z "$DP_SIZE" ]]; then
  if (( GPU_COUNT % TP_SIZE != 0 )); then
    echo "GPU count ($GPU_COUNT) must be divisible by tensor parallel size ($TP_SIZE)" >&2
    exit 2
  fi
  DP_SIZE="$(( GPU_COUNT / TP_SIZE ))"
fi
if [[ -z "$DP_LOCAL_SIZE" ]]; then
  DP_LOCAL_SIZE="$DP_SIZE"
fi

BASE_URL="http://127.0.0.1:${PORT}/v1"
MODELS_URL="${BASE_URL}/models"
METADATA_FILE="$LOG_DIR/gemma4_mtp_vllm_${PORT}.meta"
STARTED_PID=""

is_ready() {
  local body
  body="$(curl -sf -m 5 "$MODELS_URL" 2>/dev/null || true)"
  [[ "$body" == *"\"$SERVED_MODEL_NAME\""* ]]
}

metadata_value() {
  local key="$1"
  if [[ -f "$METADATA_FILE" ]]; then
    awk -F= -v k="$key" '$1 == k {sub(/^[^=]*=/, ""); print; exit}' "$METADATA_FILE"
  fi
}

check_metadata_field() {
  local key="$1"
  local expected="$2"
  local actual
  actual="$(metadata_value "$key")"
  [[ "$actual" == "$expected" ]]
}

metadata_matches_request() {
  [[ -f "$METADATA_FILE" ]] || {
    echo "Existing Gemma 4 MTP service is ready on $BASE_URL, but launcher metadata is missing: $METADATA_FILE" >&2
    return 1
  }

  local mismatches=()
  check_metadata_field "served_model_name" "$SERVED_MODEL_NAME" || mismatches+=("served_model_name")
  check_metadata_field "gpus" "$GPUS" || mismatches+=("gpus")
  check_metadata_field "gpu_count" "$GPU_COUNT" || mismatches+=("gpu_count")
  check_metadata_field "tensor_parallel_size" "$TP_SIZE" || mismatches+=("tensor_parallel_size")
  check_metadata_field "data_parallel_size" "$DP_SIZE" || mismatches+=("data_parallel_size")
  check_metadata_field "data_parallel_size_local" "$DP_LOCAL_SIZE" || mismatches+=("data_parallel_size_local")
  check_metadata_field "max_model_len" "$MAX_MODEL_LEN" || mismatches+=("max_model_len")
  check_metadata_field "max_num_batched_tokens" "$MAX_NUM_BATCHED_TOKENS" || mismatches+=("max_num_batched_tokens")
  check_metadata_field "max_num_seqs" "$MAX_NUM_SEQS" || mismatches+=("max_num_seqs")
  check_metadata_field "gpu_memory_utilization" "$GPU_MEMORY_UTILIZATION" || mismatches+=("gpu_memory_utilization")
  check_metadata_field "model_path" "$MODEL_PATH" || mismatches+=("model_path")
  check_metadata_field "draft_model_path" "$DRAFT_MODEL_PATH" || mismatches+=("draft_model_path")

  local metadata_pid
  metadata_pid="$(metadata_value "pid")"
  if [[ -n "$metadata_pid" && "$metadata_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$metadata_pid" 2>/dev/null; then
    mismatches+=("pid_not_running")
  fi

  if [[ "${#mismatches[@]}" -gt 0 ]]; then
    echo "Existing Gemma 4 MTP service is ready on $BASE_URL, but metadata does not match requested launch: ${mismatches[*]}" >&2
    echo "Metadata file: $METADATA_FILE" >&2
    return 1
  fi
  return 0
}

wait_ready() {
  local pid="${1:-}"
  local elapsed=0
  while [[ "$elapsed" -lt "$TIMEOUT_SECONDS" ]]; do
    if is_ready; then
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM process exited before readiness: pid=$pid" >&2
      return 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "Timed out waiting for $MODELS_URL after ${TIMEOUT_SECONDS}s" >&2
  return 1
}

if is_ready; then
  if ! metadata_matches_request; then
    echo "Refusing to reuse an unmanaged or mismatched Gemma 4 MTP service." >&2
    exit 3
  fi
  echo "BASE_URL=$BASE_URL"
  echo "STATUS=reused"
  echo "PID=$(metadata_value "pid")"
  echo "LOG_FILE=$(metadata_value "log_file")"
  echo "METADATA_FILE=$METADATA_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$RANDOM"
LOG_FILE="$LOG_DIR/gemma4_mtp_vllm_${RUN_ID}.log"
PID_FILE="$LOG_DIR/gemma4_mtp_vllm_${RUN_ID}.pid"

cleanup_started_vllm() {
  local pid="${1:-}"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 5
  fi
  if kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
}

handle_signal() {
  cleanup_started_vllm "$STARTED_PID"
  rm -f "$METADATA_FILE"
  exit 130
}

trap handle_signal INT TERM HUP

cmd=(
  "$VLLM_PYTHON" -u -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --no-enable-flashinfer-autotune
  --reasoning-parser gemma4
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --enforce-eager
  --tensor-parallel-size "$TP_SIZE"
  --speculative-config "{\"method\":\"mtp\",\"model\":\"$DRAFT_MODEL_PATH\",\"num_speculative_tokens\":1}"
)

if [[ "$DP_SIZE" -gt 1 ]]; then
  cmd+=(--data-parallel-size "$DP_SIZE")
  cmd+=(--data-parallel-size-local "$DP_LOCAL_SIZE")
  cmd+=(--data-parallel-backend mp)
fi

(
  # Absolute interpreter paths do not activate the environment's build tools.
  export PATH="$(dirname "$VLLM_PYTHON"):$PATH"
  export CUDA_VISIBLE_DEVICES="$GPUS"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export PYTHONPATH="$PYTHONPATH_OVERLAY"
  echo "launcher_started $(date -Is)"
  echo "gpus=$GPUS tp=$TP_SIZE dp=$DP_SIZE dp_local=$DP_LOCAL_SIZE"
  echo "base_url=$BASE_URL"
  echo "log_file=$LOG_FILE"
  if command -v setsid >/dev/null 2>&1; then
    exec setsid "${cmd[@]}"
  fi
  exec "${cmd[@]}"
) > "$LOG_FILE" 2>&1 < /dev/null &

PID="$!"
STARTED_PID="$PID"
echo "$PID" > "$PID_FILE"
cat > "$METADATA_FILE" <<EOF
pid=$PID
log_file=$LOG_FILE
pid_file=$PID_FILE
base_url=$BASE_URL
port=$PORT
served_model_name=$SERVED_MODEL_NAME
gpus=$GPUS
gpu_count=$GPU_COUNT
tensor_parallel_size=$TP_SIZE
data_parallel_size=$DP_SIZE
data_parallel_size_local=$DP_LOCAL_SIZE
max_model_len=$MAX_MODEL_LEN
max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS
max_num_seqs=$MAX_NUM_SEQS
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
model_path=$MODEL_PATH
draft_model_path=$DRAFT_MODEL_PATH
started_at=$(date -Is)
EOF

if ! wait_ready "$PID"; then
  echo "LOG_FILE=$LOG_FILE" >&2
  echo "PID_FILE=$PID_FILE" >&2
  echo "METADATA_FILE=$METADATA_FILE" >&2
  cleanup_started_vllm "$PID"
  rm -f "$METADATA_FILE"
  exit 1
fi
STARTED_PID=""
trap - INT TERM HUP

echo "BASE_URL=$BASE_URL"
echo "STATUS=started"
echo "PID=$PID"
echo "LOG_FILE=$LOG_FILE"
echo "PID_FILE=$PID_FILE"
echo "METADATA_FILE=$METADATA_FILE"
