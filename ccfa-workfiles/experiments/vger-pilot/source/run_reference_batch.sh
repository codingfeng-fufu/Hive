#!/usr/bin/env bash
set -Eeuo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$project"
: "${VGER_GPU:?Set VGER_GPU to the available GPU index}"
port="${VGER_PORT:-8147}"
run="${VGER_RUN_DIR:-runs/vger-pilot/reference-batch}"
source_dir=ccfa-workfiles/experiments/vger-pilot/source
pdfs=data/upstream/mmlongbench-doc/documents
model=Qwen/Qwen2.5-VL-7B-Instruct
python_bin="${VGER_PYTHON:-python}"
vllm_bin="${VGER_VLLM:-data/serve-mirror-env/bin/vllm}"
limit_style="${VGER_VLLM_LIMIT_STYLE:-json}"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$VGER_GPU"
export HF_HUB_OFFLINE=1

mkdir -p "$run"
exec 9>"$run/lock"
flock -n 9 || { printf '%s\n' "Batch already running: $run" >&2; exit 1; }
server_pid=""
cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" >"$run/exit"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "$$" >"$run/pid"
rm -f "$run/exit"
printf 'gpu=%s\nport=%s\npython=%s\nmodel=%s\nvllm=%s\nlimit_style=%s\n' \
    "$VGER_GPU" "$port" "$python_bin" "$model" "$vllm_bin" "$limit_style" >"$run/config.txt"

case "$limit_style" in
    json) image_limit=(--limit-mm-per-prompt '{"image": 2}') ;;
    dotted) image_limit=(--limit-mm-per-prompt.image 2) ;;
    *) printf 'Unknown VGER_VLLM_LIMIT_STYLE: %s\n' "$limit_style" >&2; exit 2 ;;
esac
command -v "$vllm_bin" >/dev/null || {
    printf 'vLLM executable not found: %s\n' "$vllm_bin" >&2
    exit 127
}
"$python_bin" scripts/reference_assets.py check

"$python_bin" - "$port" <<'PY'
import socket
import sys
with socket.socket() as probe:
    probe.bind(("127.0.0.1", int(sys.argv[1])))
PY

input="$run/expansion.json"
if [[ ! -f "$input" ]]; then
    timeout --kill-after=15s 5m "$python_bin" "$source_dir/scope_pair_probe.py" \
        --scores runs/vger-pilot/raw_scores.json \
        --pdfs "$pdfs" --model data/models/bge-small-en-v1.5 \
        --reference-pairs "$source_dir/reference-pair-expansion.yaml" \
        --vision-model data/models/siglip-so400m --device cuda \
        --output "$run/expansion.tmp"
    mv "$run/expansion.tmp" "$input"
fi
"$python_bin" - "$input" "$source_dir/reference-pair-expansion.yaml" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
cached, manifest = map(Path, sys.argv[1:])
assert json.loads(cached.read_text())["reference_sha256"] == hashlib.sha256(
    manifest.read_bytes()
).hexdigest(), "Expansion cache is from a different annotation manifest"
PY

setsid "$vllm_bin" serve data/models/qwen2.5-vl-7b \
    --served-model-name "$model" --host 127.0.0.1 --port "$port" \
    --gpu-memory-utilization 0.85 --max-model-len 4096 \
    "${image_limit[@]}" \
    --mm-processor-kwargs '{"max_pixels": 337120}' \
    --max-num-seqs 1 --enforce-eager >"$run/server.log" 2>&1 9>&- &
server_pid=$!
base_url="http://127.0.0.1:$port/v1"
ready=0
for ((attempt = 0; attempt < 60; attempt++)); do
    kill -0 "$server_pid" 2>/dev/null || { tail -40 "$run/server.log" >&2; exit 1; }
    if curl -fsS --max-time 2 "$base_url/models" 2>/dev/null | grep -Fq "\"$model\""; then
        ready=1
        break
    fi
    sleep 3
done
if [[ "$ready" != 1 ]]; then
    tail -40 "$run/server.log" >&2
    exit 1
fi

timeout --kill-after=15s 40m "$python_bin" "$source_dir/reference_batch.py" \
    --legacy runs/vger-pilot/scope-pair/results.json \
    --seed runs/vger-pilot/scope-pair/reference-seed.json \
    --expansion "$input" --pdfs "$pdfs" \
    --out "$run" --url "$base_url" --model "$model"
