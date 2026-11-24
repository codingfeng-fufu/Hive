#!/usr/bin/env bash
set -euo pipefail
out=runs/full-page-baselines-7b
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES="${HIVG_LAYOUT_GPU:-1}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export HIVG_VISRAG_MODEL=data/models/visrag-ret
export HIVG_COLPALI_MODEL=data/models/colpali-v1.3-hf
export HIVG_COLPALI_FORMAT=transformers

manifests=(
    --manifest data/mmlongbench-full-layouts-provisional-granularity.json
    --manifest data/docbench-full-layouts-provisional-granularity.json
)
for method in visrag colpali; do
    status=0
    python -u -m ccfa-workfiles.experiments.vger-pilot.source.visrag_cache \
        "${manifests[@]}" --method "$method" --resume --output "$out/$method-rankings.json" \
        > "$out/$method-index.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$out/$method-index.exit"
done
bash ccfa-workfiles/planning/hivg-rag-icme/run_eval_queue.sh page-baselines
