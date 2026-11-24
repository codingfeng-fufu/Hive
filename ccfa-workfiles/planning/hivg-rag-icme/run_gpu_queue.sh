#!/usr/bin/env bash
set -u

out=runs/vger-pilot/full-document-raganything-bounded
layouts=runs/full-corpus-layouts
mode="${1:-all}"
if [[ "$mode" == repair-crop ]]; then out=runs/vger-pilot/raganything-crop-repair; fi
mkdir -p "$out" "$layouts"

run_raganything() (
    set -euo pipefail
    gpu="${HIVG_LAYOUT_GPU:-1}"
    port=8131
    free_memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free_memory < 21500 )); then
        echo "GPU $gpu has ${free_memory} MiB free, below the configured Qwen allocation"
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
    data/serve-mirror-env/bin/vllm serve data/models/qwen2.5-vl-7b \
        --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port "$port" \
        --gpu-memory-utilization 0.85 --max-model-len 32768 --limit-mm-per-prompt '{"image":24}' \
        --mm-processor-kwargs '{"max_pixels":200704}' --max-num-seqs 4 --enforce-eager \
        --disable-log-requests > "$out/server.log" 2>&1 &
    service=$!
    trap 'kill "$service" 2>/dev/null || true; wait "$service" 2>/dev/null || true' EXIT
    for attempt in $(seq 1 90); do
        kill -0 "$service"
        if curl -fsS -m 2 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null; then break; fi
        sleep 5
    done
    curl -fsS -m 2 "http://127.0.0.1:$port/v1/models" -o /dev/null
    export PATH="$PWD/data/mineru-env/bin:$PATH"
    export PYTHONPATH=.:data/tools
    export HIVG_RAGANYTHING_LAYOUT_ROOT=data HIVG_GRAPH_EMBED_URL=http://127.0.0.1:8133/v1
    export HIVG_GRAPH_EMBED_MODEL=bge-small-en-v1.5 HIVG_GRAPH_EMBED_DIM=384
    export HIVG_GRAPH_EMBED_MAX_TOKENS=512 HIVG_RAGANYTHING_COMPLETION_LIMIT=2048
    manifests=(--manifest data/mmlongbench-full-document-pilot.json)
    if [[ "$mode" == repair-crop ]]; then
        while IFS= read -r case_id; do
            manifests+=(--case-id "$case_id")
        done < "$out/case-ids.txt"
    else
        manifests+=(--manifest data/docbench-full-document-pilot.json)
    fi
    data/baseline-env/bin/python -m eval.run_benchmark --experiment main \
        --config data/vger-screen/encoder.yaml --servers data/vger-screen/full-servers.yaml \
        "${manifests[@]}" \
        --method rag-anything --scorer mmlongbench-doc-full-document-pilot=eval.official:mmlongbench \
        --backbone 7b --resume --output "$out/results"
)

run_raganything > "$out/benchmark.log" 2>&1
rag_status=$?
printf '%s\n' "$rag_status" > "$out/exit"
if [[ "$mode" == repair-crop ]]; then exit "$rag_status"; fi
bash ccfa-workfiles/planning/hivg-rag-icme/parse_corpora.sh > "$layouts/launch.log" 2>&1
layout_status=$?
printf '%s\n' "$layout_status" > "$layouts/exit"
(( rag_status == 0 && layout_status == 0 ))
