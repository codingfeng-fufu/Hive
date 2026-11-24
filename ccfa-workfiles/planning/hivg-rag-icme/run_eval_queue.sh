#!/usr/bin/env bash
set -euo pipefail

phase="${1:-default}"
out=runs/hivg-rag-7b
if [[ "$phase" == page-baselines ]]; then out=runs/full-page-baselines-7b; fi
mkdir -p "$out"
gpu="${HIVG_LAYOUT_GPU:-1}"
port=8131
free_memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free_memory < 22500 )); then
    echo "GPU $gpu has ${free_memory} MiB free, below the Qwen and SigLIP allocation"
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export PYTHONPATH=.:data/tools:ccfa-workfiles/experiments/vger-pilot/source

data/serve-mirror-env/bin/vllm serve data/models/qwen2.5-vl-7b \
    --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port "$port" \
    --gpu-memory-utilization 0.74 --max-model-len 12288 --limit-mm-per-prompt '{"image":24}' \
    --mm-processor-kwargs '{"max_pixels":200704}' --max-num-seqs 1 --enforce-eager \
    --disable-log-requests > "$out/server.log" 2>&1 &
service=$!
trap 'kill "$service" 2>/dev/null || true; wait "$service" 2>/dev/null || true' EXIT
for attempt in $(seq 1 90); do
    kill -0 "$service"
    if curl -fsS -m 2 "http://127.0.0.1:$port/v1/models" -o /dev/null 2>/dev/null; then break; fi
    sleep 5
done
curl -fsS -m 2 "http://127.0.0.1:$port/v1/models" -o /dev/null

failures=0
run() {
    local label="$1" status=0
    shift
    python -u -m eval.run_benchmark \
        --config data/vger-screen/encoder.yaml --servers data/vger-screen/full-servers.yaml \
        --backbone 7b --resume --output "$out/$label" "$@" >> "$out/$label.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$out/$label.exit"
    if (( status != 0 )); then failures=$((failures + 1)); fi
}

if [[ "$phase" == page-baselines ]]; then
    export HIVG_PAGE_BASELINE_CACHE="$out"
    baseline_inputs=(
        --manifest data/mmlongbench-full-layouts-provisional-granularity.json
        --manifest data/docbench-full-layouts-provisional-granularity.json
        --scorer mmlongbench-doc=eval.official:mmlongbench
        --adapters ccfa-workfiles/experiments/vger-pilot/source/full-baselines.yaml
    )
    for method in visrag colpali; do
        if [[ "$(cat "$out/$method-index.exit" 2>/dev/null || true)" == 0 ]]; then
            run "$method" --experiment main "${baseline_inputs[@]}" --method "$method"
        else
            failures=$((failures + 1))
        fi
    done
    exit "$(( failures != 0 ))"
fi

pilot=(
    --manifest data/mmlongbench-full-document-pilot.json
    --manifest data/docbench-full-document-pilot.json
    --scorer mmlongbench-doc-full-document-pilot=eval.official:mmlongbench
)
run cases --experiment case "${pilot[@]}" \
    --method hivg-rag --method macro-only --method micro-only \
    --case-id 2303.05039v2.pdf/289 --case-id BRO-GL-MMONEY.pdf/795 \
    --case-id BRO-GL-MMONEY.pdf/800 --case-id 179/4 --case-id 103/1
if [[ "$(cat "$out/cases.exit")" == 0 ]]; then
    python -m eval.compare_traces "$out/cases" "$out/cases.html"
fi

if [[ -s runs/vger-pilot/routing-score-components.json ]]; then
    python - "$out" <<'PY'
import json
import sys
from pathlib import Path

import yaml

directory = Path(sys.argv[1])
config = yaml.safe_load(Path("data/vger-screen/encoder.yaml").read_text())
config["retrieval"] = {"macro_k": 6, "micro_k": 6}
(directory / "matched-default.yaml").write_text(yaml.safe_dump(config))
probe = json.loads(Path("runs/vger-pilot/routing-score-components.json").read_text())
cutoff = probe["summary"]["entropy_focus"]["development_cutoff"]
config["retrieval"].update(lambda_lex=0.0, tau_zoom=cutoff)
(directory / "matched-entropy-only.yaml").write_text(yaml.safe_dump(config))
PY
    run matched-default --experiment main "${pilot[@]}" --method hivg-rag \
        --method micro-only --config "$out/matched-default.yaml"
    run matched-entropy-only --experiment main "${pilot[@]}" --method hivg-rag \
        --config "$out/matched-entropy-only.yaml"
fi

if [[ -s data/mmlongbench-full-layouts.json && -s data/docbench-full-layouts.json ]]; then
    for corpus in mmlongbench docbench; do
        python -m eval.prepare_granularity "data/$corpus-full-layouts.json" \
            "data/$corpus-full-layouts-provisional-granularity.json" \
            --reviewed "data/$corpus-full-document-pilot.json" > "$out/$corpus-granularity.log"
    done
    run full-defaults --experiment main --method hivg-rag --method direct --direct-max-pages 24 \
        --manifest data/mmlongbench-full-layouts-provisional-granularity.json \
        --manifest data/docbench-full-layouts-provisional-granularity.json \
        --scorer mmlongbench-doc=eval.official:mmlongbench
else
    echo "Full-document QA awaits the two complete layout manifests" > "$out/full-defaults.log"
    failures=$((failures + 1))
fi

run pilot-sensitivity --experiment sensitivity "${pilot[@]}"
if [[ "$(cat "$out/pilot-sensitivity.exit")" == 0 ]]; then
    python -m eval.plot_results "$out/pilot-sensitivity/summary.json" sensitivity \
        "$out/pilot-sensitivity.pdf" --metric anls
fi

if [[ "$(cat runs/data-acquisition/mmgraphrag-deps.exit 2>/dev/null || true)" == 0 ]]; then
    export PATH="$PWD/data/mineru-env/bin:$PATH"
    export HIVG_MMGRAPH_LAYOUT_ROOT="$PWD/data"
    cat > "$out/mmgraphrag-adapter.yaml" <<'YAML'
mmgraphrag:
  command:
    - data/mmgraph-env/bin/python
    - eval/upstream_worker.py
    - mmgraphrag
    - data/upstream-code/pinned/mmgraphrag-api
  persistent: true
  timeout: 3600
YAML
    run pilot-mmgraphrag --experiment main "${pilot[@]}" --method mmgraphrag \
        --adapters "$out/mmgraphrag-adapter.yaml"
else
    echo "MMGraphRAG awaits the verified model environment" > "$out/pilot-mmgraphrag.log"
    failures=$((failures + 1))
fi
(( failures == 0 ))
