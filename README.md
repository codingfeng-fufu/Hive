# HiVG-RAG

HiVG-RAG is a training-free multimodal document QA system. It indexes section summaries and full figures or tables in a macro graph. A micro graph stores localized evidence and entity occurrences. Directed `PART_OF` edges connect each resolved entity to its occurrence parents.

The next experiment is a frozen VGER pair-resolution batch with six evidence conditions. See the [research path map](ccfa-workfiles/planning/hivg-rag-icme/research-path-map.md) and [server migration guide](MIGRATION.md) for its protocol, assets, and single-GPU launch command.

```text
src/
  document.py       Documents, node types and native-pixel regions
  layout.py         MinerU input, PDF rendering and localized regions
  embeddings.py     Unified SigLIP text/image embeddings
  graph_builder.py  Two graph layers, VGER, and parent-scoped micro projection
  retriever.py      QAHT routing, PageRank, and evidence selection
  generation.py     vLLM client and multimodal prompt accounting
configs/
  default.yaml      Encoder, VGER, QAHT, and rasterization parameters
  servers.yaml      Qwen2.5-VL endpoint and model identifiers
eval/
  prepare.py        Upstream dataset conversion and annotation attachment
  prepare_stress.py Builds matched distractor pages from alternate figures
  run_benchmark.py  Experiment presets, baseline interfaces, JSON/CSV logging
  baselines.py     VisRAG and ColPali retrieval with common Qwen generation
  graph_baseline.py RAG-Anything retrieval with common Qwen generation
  baseline_command.py  Adapter for upstream projects in separate environments
  upstream_worker.py  MMGraphRAG and MG²-RAG retrieval workers
  metrics.py        Accuracy, ANLS, and entity/graph diagnostics
  stress.py         Matched visual polysemy perturbations
  official.py       Adapter for the upstream MMLongBench-Doc scorer
  plot_results.py   Stress curves, sensitivity heatmaps, and token Pareto plots
tests/              Graph invariants, native coordinates, and HTTP smoke tests
```

## Setup

```bash
python -m pip install -e '.[encoder,evaluation,test]'
python -m pytest -q
```

Run MinerU 2.x in its own environment. The importer accepts the 2.x Content List V1 `*_content_list.json` format with zero-based `page_idx` and bounding boxes on a 0–1000 grid. The PDF raster defaults to 250 DPI. Raster decks use released slide pixels. An enriched content list may declare `coordinate_space: "pixels"` and supply explicit localized `regions` with `bbox`, `text`, and an optional `entity: true` flag. For a visual block without an attached caption, the loader uses nearby, horizontally aligned short text above it when available. When explicit regions are unavailable, the parser uses detected PDF table rows or embedded subimages and falls back to overlapping tiles. Section summaries use the first two available sentences. Entity names come from layout text, captions, and labeled regions. A body-text entity inherits a figure or table region when its sentence contains one extracted name and one unambiguous reference.

Serve one Qwen backbone at a time. The vLLM instance needs an image-per-prompt limit large enough for the selected documents and a matching context limit. Configure both limits and the image processor consistently across methods within a backbone. The client sends PNG crops at their original raster dimensions. vLLM may resize images during model preprocessing. Context overflow appears in the per-question logs as a request error. The 32B command below illustrates the original dual-GPU configuration.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 --max-model-len 32768 --limit-mm-per-prompt.image 128
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --port 8001 --tensor-parallel-size 2 --max-model-len 32768 --limit-mm-per-prompt.image 128
```

These commands use the dotted multimodal limit syntax in modern vLLM. For a one-backbone run, use `--backbone 7b` or `--backbone 32b`. Keep the SigLIP encoder on CPU or allocate it compatible GPU memory through `encoder.device`. Record the vLLM version, exact model revisions, MinerU version, and server flags with any result release.

## Data and annotations

`eval.prepare` converts the public MMLongBench-Doc Parquet or JSONL and its PDF directory, DocBench's per-document `*_qa.jsonl` and PDFs, or SlideVQA's released Parquet shards and deck images. Use `--layouts` to attach precomputed MinerU output. `--parse` with `--layouts` runs the MinerU executable for the selected documents. For E-VQA supply a fixed candidate corpus with the exported questions, references, and original query images. The converter accepts these preassembled bundles and resolves the paths. Record the corpus, candidate construction, and benchmark split used for E-VQA. [MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc) · [DocBench](https://github.com/Anni-Zou/DocBench) · [SlideVQA](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA) · [E-VQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa)

SlideVQA also accepts the official `annotations/qa/test.jsonl` and a local slide directory such as `images/test` from the upstream downloader. The converter preserves the order of `image_urls` and uses the original slide files for native crops. [SlideVQA repository](https://github.com/nttmdlab-nlp/SlideVQA)

```bash
python -m eval.prepare mmlongbench-doc \
  --source data/mmlongbench/data/train-00000-of-00001.parquet \
  --assets data/mmlongbench/documents --layouts data/mineru/mmlongbench \
  --annotations data/annotations/mmlongbench.json --split train \
  --output data/mmlongbench.json
```

A manifest groups questions under each document and resolves asset paths relative to the manifest. Questions can carry an optional query `image` path. A document may provide a PDF and MinerU layout, page raster paths and a layout, or prebuilt `macro` and `mentions` lists using the fields in `src/document.py` with each `region` represented by its native `bbox` on the node's page.

```json
{
  "dataset": "example",
  "split": "test",
  "documents": [
    {
      "id": "book",
      "pdf": "pdfs/book.pdf",
      "layout": "mineru/book_content_list.json",
      "questions": [
        {"id": "q1", "question": "What value is in row 3?", "answers": ["72%"], "subset": "micro"}
      ],
      "entity_gold": {"p0000:b00001:e0": "unit_a"},
      "distractors": [
        {
          "target": "p0000:b00001:e0",
          "image": "distractors/book_1.png",
          "bbox": [40, 55, 800, 600],
          "caption": "Figure 1. Module A",
          "gold_entity": "distractor_a"
        }
      ]
    }
  ]
}
```

Annotation files key each document by its ID and each question by its prepared ID. Add the `subsets` map, `entity_gold` and, for the stress test, `distractors`. Paths in annotations are relative to the annotations file. Assign **micro** to localized evidence such as a chart mark or a table cell. Assign **macro** to summaries, global comparisons, and cross-region synthesis. Leave ambiguous items **unlabeled**. Use the same frozen labels for every method. The runner computes strata from these labels independently of the QAHT route. The stress preset samples 100 documents by default from ones with at least four distinct, annotated distractor assets. The 0, 1, 2, and 4 conditions use the same documents and nested page sets. Each distractor has a distinct annotated identity and uses the target mention's surface form. Supply pages whose appearance differs from the target and whose insertion preserves the reference answer. To construct a deterministic pool without pre-existing distractor assets, run `python -m eval.prepare_stress --manifest data/mmlongbench.json --output data/mmlongbench-stress.json`. It relabels visual content from another document and uses a pixel-distance filter to exclude near copies. Each generated page records the donor and target. Review target relevance and answer preservation before treating the stress scores as publication evidence.

## Retrieval scores

QAHT combines the lexical detail prior with normalized entropy over the macro similarity softmax. It uses lexical evidence alone for a one-node macro graph. Per query and search scope, visual similarities are aligned to the candidate text-score mean and variance before they can raise a grounded node's score. Micro retrieval keeps one grounded occurrence in the evidence budget when the selected parents contain a crop. The native-crop ablation uses the same ranking and sends the accompanying text.

## Experiments

| Preset | Evaluation |
| --- | --- |
| `main` | HiVG-RAG, Direct, VisRAG, ColPali, RAG-Anything, MMGraphRAG, and MG²-RAG on the supplied MMLongBench-Doc and DocBench manifests |
| `generalization` | HiVG-RAG on supplied SlideVQA and E-VQA manifests, with both Qwen sizes and one frozen YAML configuration |
| `ablation` | HiVG-RAG and the five requested component ablations |
| `stress` | HiVG-RAG and the split ablation on matched distractor counts |
| `sensitivity` | Split grid 0.4–0.8 by 0.1 and zoom grid 0.35–0.75 by 0.1, configurable through `--zoom-grid` |
| `efficiency` | HiVG-RAG and Direct by default. Use `--method` to include additional baselines |
| `case` | Full traces for the ablation configurations. Use `--case-id` to select question IDs |

```bash
python -m eval.run_benchmark --experiment ablation \
  --manifest data/mmlongbench.json --backbone 7b --output runs/ablation-7b
python -m eval.run_benchmark --experiment generalization \
  --manifest data/slidevqa.json --manifest data/e-vqa.json --output runs/generalization
python -m eval.run_benchmark --experiment stress \
  --manifest data/mmlongbench.json --output runs/visual-polysemy
python -m eval.plot_results runs/visual-polysemy/summary.json stress runs/polysemy.pdf
python -m eval.plot_results runs/efficiency/summary.json pareto runs/accuracy-tokens.pdf
python -m eval.plot_results runs/sensitivity/summary.json sensitivity runs/sensitivity.pdf
```

Select `--method hivg-rag --method direct` for a comparison using the core dependencies. VisRAG needs its published weights and model code. Set `HIVG_VISRAG_REVISION` to the tested upstream commit. ColPali uses the `colpali` dependency extra. Both page retrieval baselines use top six pages. RAG-Anything uses the `rag-anything` dependency extra, MinerU, and a text embedding endpoint declared by `HIVG_GRAPH_EMBED_URL`, `HIVG_GRAPH_EMBED_MODEL`, and `HIVG_GRAPH_EMBED_DIM`. It uses the configured Qwen endpoint during indexing and the same final answer prompt as other methods. [VisRAG](https://github.com/OpenBMB/VisRAG) · [ColPali](https://github.com/illuin-tech/colpali) · [RAG-Anything](https://github.com/HKUDS/RAG-Anything)

Use `--direct-max-pages N` to evaluate a page-budgeted Direct baseline when a document exceeds the model's image limit. Direct samples document pages by uniformly spaced positions, independent of the question. The run records the cap, each question's page counts, and the number of capped questions with successful responses. Report this setting as page-budgeted Direct. Choose the cap to fit the server's token and image limits, including any query image.

Long benchmark runs can use `--resume` with the original command and output directory. The runner restores completed documents from checkpoints and restarts any document that was interrupted. Each checkpoint preserves the indexing diagnostics and all question statuses for that document. The final JSONL, CSV, and summary are rebuilt in manifest order.

MMGraphRAG and MG²-RAG use upstream code and model checkpoints from separate environments. Fill in the checkout and interpreter paths in `configs/adapters.example.yaml` and pass the file with `--adapters`. MMGraphRAG uses its upstream text and image selection. For MG²-RAG, this adapter indexes each PDF page as its extracted text and rendered image, then passes the ranked pages to Qwen. A custom adapter can use a `module:Factory` entry or a command. [MMGraphRAG](https://github.com/wanxueyao/MMGraphRAG) · [MG²-RAG](https://github.com/Daboolu/MG2-RAG)

```yaml
mmgraphrag:
  command: [/path/to/mmgraph-env/bin/python, /path/to/mmgraph_adapter.py]
mg2-rag: project_adapters:MG2Adapter
```

A Python adapter exposes `index(document)` and `retrieve(index, question)`, returning a `Retrieval` with ordered `Evidence` items. Command adapters receive a JSON request on stdin and return a JSON response on stdout. On `index`, the request carries `document` with `id`, `pdf`, and `pages`, plus `workdir`. Return `{"index": ...}` with a durable JSON-serializable index handle. On `retrieve`, the request carries `index`, `question`, `workdir`, and an optional `query_image`. Set `persistent: true` for workers that consume one JSON request and emit one JSON response per line. Requests also carry the answer server configuration. Return `{"evidence": [...], "trace": {...}}` using the fields in `Evidence`. Return text plus a document `region`, a whole page, or an `image_path` relative to `workdir`. The benchmark harness calls the same answer backend for all methods. For published comparisons, pin each upstream checkout and record its settings and retrieved evidence limits.

## Measurement

The aggregate `summary.json` and `summary.csv` report normalized exact-match accuracy and ANLS for overall, macro, micro, and unlabeled questions. For an official score, use `--scorer module:function`. Scope a scorer with `--scorer DATASET=module:function` in a multidataset run. To run the MMLongBench-Doc paper's typed scoring, save its upstream `eval/eval_score.py` as `mmlongbench_eval_score.py` on the Python path and pass `--scorer eval.official:mmlongbench`. DocBench's official evaluator uses a model judge and its evidence prompt. The runner's exact match and ANLS remain available for DocBench as separate common metrics. [MMLongBench-Doc evaluator](https://github.com/mayubo2333/MMLongBench-Doc/blob/main/eval/eval_score.py) · [DocBench evaluator](https://github.com/Anni-Zou/DocBench/blob/main/evaluate.py)

Each run records input manifest hash, effective configuration, seed, dependency versions, and model settings. Query latency covers routing, retrieval, client image serialization, and answer generation. Retrieval time and combined generation/serialization time are also recorded. Prompt tokens are the actual vLLM `usage.prompt_tokens` including image tokens, system prompt, and question. Missing usage stays empty. Index times cover embeddings and graph construction for HiVG-RAG. Loading/rasterization and topology diagnostics have separate boundaries. MinerU preprocessing is offline when a precomputed layout is supplied. Report parsing cost separately if measuring complete document ingestion.

Topology compares fine-grained graphs built with and without both VGER operations on the same input. A super-hub has entity degree at or above `superhub_degree` in the micro graph. Pairwise entity precision uses annotated endpoints and is reported for merges, aliases, and their union. Aggregate precision pools pair counts across documents. The value is empty when there are zero evaluated predicted links. Accuracy averages successfully scored questions and reports both error and scored counts. The CLI exits with failure when a group has errors or no scored questions.

Traces record the canonical IDs, rejected pair scores, alias scores, gamma, parent probabilities, selected parents, evidence occurrence IDs, native bounding boxes, and final answers. PNG files record the exact image payloads. Case directories share stable names across methods for side-by-side review.
