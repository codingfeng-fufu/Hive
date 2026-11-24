#!/usr/bin/env bash
set -euo pipefail

out=runs/full-corpus-layouts
mkdir -p "$out"
gpu="${HIVG_LAYOUT_GPU:-1}"
port=8130
free_memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free_memory < 16500 )); then
    echo "GPU $gpu has ${free_memory} MiB free, below the configured MinerU allocation"
    exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4

CUDA_VISIBLE_DEVICES="$gpu" data/serve-mirror-env/bin/vllm serve data/models/mineru-2.5 \
    --host 127.0.0.1 --port "$port" --served-model-name MinerU2.5-2509-1.2B \
    --gpu-memory-utilization 0.65 --max-model-len 16384 --max-num-seqs 16 \
    --limit-mm-per-prompt '{"image":1}' --enforce-eager > "$out/server.log" 2>&1 &
service=$!
cleanup() {
    kill "$service" 2>/dev/null || true
    wait "$service" 2>/dev/null || true
}
trap cleanup EXIT

health="http://127.0.0.1:$port/v1/models"
for attempt in $(seq 1 100); do
    kill -0 "$service"
    if curl -fsS --max-time 2 "$health" -o /dev/null 2>/dev/null; then break; fi
    sleep 4
done
curl -fsS --max-time 2 "$health" -o /dev/null

for corpus in mmlongbench docbench; do
    python - "data/$corpus-full.json" > "$out/$corpus.paths" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
for document in json.loads(manifest.read_text())["documents"]:
    print(manifest.parent / document["pdf"])
PY
done

valid_layout() {
    [[ -s "$1" ]] && python - "$1" <<'PY'
import json
import sys
from pathlib import Path

try:
    layout = json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(layout, list) and layout else 1)
PY
}

failures=0
printf 'corpus\tpdf\tstatus\tseconds\n' > "$out/progress.tsv"
for corpus in mmlongbench docbench; do
    layout_root=data/mineru-docbench
    if [[ "$corpus" == mmlongbench ]]; then layout_root=data/mineru-full; fi
    mkdir -p "$out/$corpus"
    while IFS= read -r pdf; do
        stem=$(basename "$pdf" .pdf)
        layout="$layout_root/$stem/vlm/${stem}_content_list.json"
        start=$SECONDS
        if valid_layout "$layout"; then
            status=cached
        else
            status=failed
            for attempt in 1 2; do
                if CUDA_VISIBLE_DEVICES='' data/mineru-env/bin/python -u \
                    ccfa-workfiles/planning/hivg-rag-icme/parse_mineru_document.py \
                    "$pdf" "$layout_root" "http://127.0.0.1:$port" \
                    > "$out/$corpus/$stem-$attempt.log" 2>&1 && valid_layout "$layout"; then
                    status=parsed
                    break
                fi
                kill -0 "$service"
            done
            if [[ "$status" == failed ]]; then failures=$((failures + 1)); fi
        fi
        printf '%s\t%s\t%s\t%s\n' "$corpus" "$pdf" "$status" "$((SECONDS - start))" \
            >> "$out/progress.tsv"
        kill -0 "$service"
    done < "$out/$corpus.paths"
done
(( failures == 0 ))
python - <<'PY'
import json
from pathlib import Path

for dataset, root in (("mmlongbench", "mineru-full"), ("docbench", "mineru-docbench")):
    source = Path("data") / f"{dataset}-full.json"
    manifest = json.loads(source.read_text())
    for document in manifest["documents"]:
        stem = Path(document["pdf"]).stem
        layout = Path(root) / stem / "vlm" / f"{stem}_content_list.json"
        if not (source.parent / layout).is_file():
            raise FileNotFoundError(layout)
        document["layout"] = layout.as_posix()
        rebuilt_pdf = layout.parent / "source" / Path(document["pdf"]).name
        if (source.parent / rebuilt_pdf).is_file():
            document["original_pdf"] = document["pdf"]
            document["pdf"] = rebuilt_pdf.as_posix()
    target = source.with_name(f"{dataset}-full-layouts.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)
PY
