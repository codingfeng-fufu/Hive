#!/usr/bin/env bash
set -euo pipefail

out=${1:-runs/colpali-granularity}
mkdir -p "$out"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TRANSFORMERS_OFFLINE=1
export HIVG_BASELINE_DEVICE=cpu
export HIVG_COLPALI_FORMAT=transformers
export HIVG_COLPALI_MODEL=data/models/colpali-v1.3-hf
export HIVG_COLPALI_CPU_BF16=${HIVG_COLPALI_CPU_BF16:-1}
export PYTHONPATH="data/tools${PYTHONPATH:+:$PYTHONPATH}"

trap 'status=$?; printf "%s\n" "$status" > "$out/run.exit"' EXIT
python -u -m eval.colpali_granularity index --output "$out"

methods=(colpali relevance scope page-relevance page-scope colpali-top6)
heading_methods=(heading-relevance heading-scope)
python -u -m eval.colpali_granularity qa --cohort public --output "$out" \
    --methods "${methods[@]}" "${heading_methods[@]}"
python -u -m eval.colpali_granularity qa --cohort target --output "$out" \
    --methods "${methods[@]}" text-layout "${heading_methods[@]}" heading-text
python -u -m eval.colpali_granularity index --cohort heldout --output "$out"
python -u -m eval.colpali_granularity qa --cohort heldout --output "$out" \
    --methods colpali page-relevance page-scope "${heading_methods[@]}" colpali-top6
for cohort in target heldout; do
    python -u -m eval.colpali_granularity qa --cohort "$cohort" --output "$out" \
        --methods verification-control verification
done
