#!/usr/bin/env bash
set -uo pipefail

plan=ccfa-workfiles/planning/hivg-rag-icme
gpu="${HIVG_LAYOUT_GPU:-1}"
mkdir -p runs/full-corpus-layouts runs/hivg-rag-7b runs/full-page-baselines-7b runs/polysemy-stress-prep

exec 9>"${TMPDIR:-/tmp}/hivg-rag-$(id -u)-gpu-$gpu.lock"
flock 9

stage() {
    local out="$1" status=0
    shift
    if [[ -f "$out/exit" && "$(cat "$out/exit")" == 0 ]]; then
        return
    fi
    "$@" >> "$out/launch.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$out/exit"
}

stage runs/vger-pilot/raganything-crop-repair bash "$plan/run_gpu_queue.sh" repair-crop
stage runs/full-corpus-layouts bash "$plan/parse_corpora.sh"

if [[ -s data/mmlongbench-full-layouts.json ]]; then
    (
        stage runs/polysemy-stress-prep python -u -m eval.prepare_stress \
            --manifest data/mmlongbench-full-layouts.json --size 100 --evidence-questions-only \
            --output runs/polysemy-stress-prep/candidate-manifest.json
    ) 9>&- &
fi
stage runs/hivg-rag-7b bash "$plan/run_eval_queue.sh"
if [[ -s data/mmlongbench-full-layouts-provisional-granularity.json && \
      -s data/docbench-full-layouts-provisional-granularity.json ]]; then
    stage runs/full-page-baselines-7b bash "$plan/run_page_baselines.sh"
fi
wait
