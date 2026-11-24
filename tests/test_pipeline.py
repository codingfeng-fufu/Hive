import json
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import fitz
import numpy as np
from PIL import Image

from eval.metrics import summarize, topology
from eval.run_benchmark import _direct_pages, main
from src.config import GraphConfig, RetrievalConfig
from src.document import Document, MacroNode, Mention, Region
from src.graph_builder import GraphBuilder
from src.layout import load_document
from src.retriever import Retriever


class TinyEncoder:
    def text(self, texts):
        vectors = {
            "Module A": [1, 0, 0, 0],
            "Cooling Unit": [0, 1, 0, 0],
            "Chapter one": [1, 0, 0, 0],
            "Chapter two": [0.8, 0, 0.6, 0],
            "Chapter three": [0, 0, 0, 1],
            "What is in row 3?": [1, 0, 0, 0],
        }
        return np.asarray([vectors.get(text, [0, 0, 1, 0]) for text in texts], dtype=np.float32)

    def image(self, images):
        return np.asarray(
            [(*np.asarray(image)[0, 0] / 255, 0) for image in images], dtype=np.float32
        )


def test_direct_page_cap_and_coverage():
    pages = [1, 3, 7, 10, 11, 18, 20]
    assert _direct_pages(pages, None) == pages
    assert _direct_pages(pages, 1) == [1]
    assert _direct_pages(pages, 3) == [1, 10, 20]
    assert _direct_pages(pages, 7) == pages
    stats = summarize(
        [
            {"subset": "macro", "status": "ok", "document_pages": 7, "input_document_pages": 3},
            {"subset": "micro", "status": "ok", "document_pages": 2, "input_document_pages": 2},
        ],
        [],
    )
    assert stats["overall"]["page_capped"] == 1
    assert stats["overall"]["mean_input_document_pages"] == 2.5


def test_resolution_and_local_projection(tmp_path):
    pages = {i: tmp_path / f"{i}.png" for i in range(4)}
    for page, color in enumerate(("red", "blue", "red", "red")):
        Image.new("RGB", (20, 20), color).save(pages[page])
    regions = [Region(i, (0, 0, 15, 17)) for i in pages]
    doc = Document(
        "fixture",
        pages,
        (
            MacroNode("a", "figure", "Chapter one", 0, references=("b",)),
            MacroNode("b", "figure", "Chapter two", 1),
            MacroNode("c", "figure", "Chapter three", 3),
        ),
        (
            Mention("a1", "a", "Module A", 0, regions[0]),
            Mention("b1", "b", "Module A", 1, regions[1]),
            Mention("b2", "b", "Cooling Unit", 2, regions[2]),
            Mention("c1", "c", "Module A", 3, regions[3]),
            Mention("c2", "c", "Module A", 3),
        ),
    )
    builder = GraphBuilder(TinyEncoder(), GraphConfig())
    index = builder.build(doc)
    assert index.entity_of["a1"] == index.entity_of["c1"]
    assert index.entity_of["a1"] != index.entity_of["b1"]
    assert ("a1", "b2", 1.0) in [(link.left, link.right, link.weight) for link in index.aliases]
    assert index.hierarchy.has_edge(index.entity_of["c1"], "c")
    assert index.project_micro({"a", "b"}).nodes[index.entity_of["a1"]]["members"] == ("a1",)
    assert index.project_micro({"a"}).number_of_edges() == 0
    labels = {"a1": "red", "b1": "blue", "b2": "red", "c1": "red", "c2": "red"}
    before = GraphBuilder(
        TinyEncoder(), replace(builder.config, enable_split=False, enable_bridge=False)
    ).build(doc)
    assert topology(index, labels, 2)["entity_precision"] == 1
    assert topology(before, labels, 2)["entity_precision"] == 0.5
    route = Retriever(TinyEncoder(), RetrievalConfig())
    zoom = route.retrieve(index, "What is in row 3?")
    assert zoom.trace["selected_parents"] == ["a", "b"]
    assert all(item.occurrence_id in {"a1", "b1", "b2"} for item in zoom.evidence)
    assert any(item.include_image for item in zoom.evidence)
    assert all(
        item.occurrence_id in {"a", "b", "c"}
        for item in route.retrieve(index, "Summarize the overall document").evidence
    )
    assert any(
        item.occurrence_id.startswith("c") or item.node_id == "a1"
        for item in route.retrieve(index, "What is in row 3?", "micro").evidence
    )


class FakeChat(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(request)
        body = json.dumps(
            {
                "choices": [{"message": {"content": "red"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 55, "completion_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_pdf_crop_and_benchmark_cli(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    with fitz.open() as fixture:
        page = fixture.new_page(width=100, height=100)
        page.draw_rect(fitz.Rect(10, 20, 30, 60), color=(1, 0, 0), fill=(1, 0, 0))
        fixture.save(pdf)
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            [
                {
                    "type": "image",
                    "page_idx": 0,
                    "bbox": [100, 200, 300, 600],
                    "image_caption": ["Figure 1. Module A"],
                    "regions": [{"bbox": [120, 220, 280, 580], "text": "Module A", "entity": True}],
                }
            ]
        )
    )
    raw = {
        "id": "doc",
        "pdf": pdf.name,
        "layout": layout.name,
        "questions": [
            {"id": "1", "question": "What is in row 3?", "answers": ["red"], "subset": "micro"},
        ],
    }
    docs = load_document(raw, tmp_path, tmp_path / "pages", 144)
    child = next(mention for mention in docs.mentions if mention.id.endswith(":r0"))
    assert child.region.bbox == (24, 44, 56, 116)
    image = docs.crop(child.region)
    assert image.size == (32, 72)
    assert image.getpixel((16, 36)) == (255, 0, 0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset": "fixture", "documents": [raw]}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        servers = tmp_path / "servers.yaml"
        servers.write_text(
            f"7b:\n  base_url: http://127.0.0.1:{server.server_port}/v1\n  model: fake\n"
        )
        output = tmp_path / "results"
        config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        monkeypatch.setattr("eval.run_benchmark.SiglipEncoder", lambda config: TinyEncoder())
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_benchmark",
                "--manifest",
                str(manifest),
                "--servers",
                str(servers),
                "--config",
                str(config),
                "--experiment",
                "ablation",
                "--output",
                str(output),
                "--render-cache",
                str(tmp_path / "cache"),
                "--trace",
            ],
        )
        assert main() == 0
        first_summary = (output / "summary.json").read_text()
        run = output / "fixture" / "7b" / "hivg-rag" / "default"
        (run / "predictions.jsonl").write_text('{"interrupted":')
        monkeypatch.setattr("sys.argv", [*sys.argv, "--resume"])
        assert main() == 0
        assert (output / "summary.json").read_text() == first_summary
        assert json.loads((run / "predictions.jsonl").read_text())["question_id"] == "1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert len(FakeChat.requests) == 6
    summary = json.loads((output / "summary.json").read_text())
    assert sum(row["subset"] == "overall" and row["accuracy"] == 1 for row in summary) == 6
    for method, expected in [("hivg-rag", True), ("without-native-crop", False)]:
        run = output / "fixture" / "7b" / method / "default"
        trace = next(run.glob("traces/*/trace.json"))
        obj = json.loads(trace.read_text())
        assert obj["mode"] == "micro"
        assert obj["prompt_tokens"] == 55
        assert bool(list(trace.parent.glob("*.png"))) == expected


def test_explicit_reference_grounds_a_cross_modal_alias(tmp_path):
    from src.layout import read_mineru

    pages = {index: tmp_path / f"{index}.png" for index in range(2)}
    for path in pages.values():
        Image.new("RGB", (40, 40), "red").save(path)
    layout = tmp_path / "content_list.json"
    layout.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "page_idx": 0,
                    "text": "The core cooling unit is shown in Fig. 1.",
                },
                {
                    "type": "image",
                    "page_idx": 1,
                    "bbox": [100, 100, 900, 900],
                    "image_caption": ["Figure 1. Part #402"],
                },
            ]
        )
    )
    doc = read_mineru("reference", pages, layout)
    body = next(node for node in doc.mentions if node.text == "core cooling unit")
    caption = next(node for node in doc.mentions if node.text == "Part #402")
    assert body.page == 0 and caption.page == 1
    assert body.parent == caption.parent
    assert body.region == caption.region
    assert doc.macro[0].references == (caption.parent,)

    class AliasEncoder(TinyEncoder):
        def text(self, texts):
            return super().text(
                [
                    {"core cooling unit": "Cooling Unit", "Part #402": "Module A"}.get(text, text)
                    for text in texts
                ]
            )

    graph = GraphBuilder(AliasEncoder(), GraphConfig()).build(doc)
    assert any({edge.left, edge.right} == {body.id, caption.id} for edge in graph.aliases)


def test_matched_polysemy_conditions(tmp_path):
    from eval.stress import inject, select_documents

    original = tmp_path / "source.png"
    Image.new("RGB", (32, 24), "red").save(original)
    doc = Document(
        "paper",
        {0: original},
        (MacroNode("figure", "figure", "Module A", 0),),
        (Mention("target", "figure", "Module A", 0, Region(0, (0, 0, 16, 16))),),
    )
    specs = []
    for i, color in enumerate(("green", "blue", "yellow", "purple")):
        path = tmp_path / f"distractor-{i}.png"
        Image.new("RGB", (32, 24), color).save(path)
        specs.append(
            {
                "target": "target",
                "image": path.name,
                "bbox": [0, 0, 16, 16],
                "caption": "Module A",
                "gold_entity": f"d{i}",
            }
        )
    selected = select_documents([{"id": "paper", "distractors": specs}], 1, 42)[0]["distractors"]
    labels = {"target": "original"}
    pair, pair_labels = inject(doc, labels, selected, tmp_path, 2)
    full, full_labels = inject(doc, labels, selected, tmp_path, 4)
    assert list(pair.pages.values()) == list(full.pages.values())[:3]
    assert pair.mentions == full.mentions[:3]
    assert all(node.text == "Module A" for node in full.mentions)
    assert pair_labels.items() <= full_labels.items()
    assert len(set(full_labels.values())) == 5
    assert len(doc.pages) == 1 and len(labels) == 1


def test_command_baseline_persists_its_index(tmp_path):
    import sys

    from eval.baseline_command import CommandBaseline

    worker = tmp_path / "worker.py"
    worker.write_text(
        """import json
import sys
active = None
for line in sys.stdin:
    req = json.loads(line)
    if req["operation"] == "index":
        active = req["document"]["id"]
        print(json.dumps({"index": active}), flush=True)
    else:
        assert active == req["index"]
        print(json.dumps({"evidence": [{"node_id": "res", "occurrence_id": "result",
                                       "page": 0, "text": active}]}), flush=True)
"""
    )
    adapter = CommandBaseline([sys.executable, str(worker)], tmp_path / "state", persistent=True)
    path = tmp_path / "page.png"
    try:
        handle = adapter.index(Document("page", {0: path}))
        result = adapter.retrieve(handle, "How many?")
    finally:
        adapter.close()
    assert result.evidence[0].text == "page"


def test_lexical_prior_uses_whole_terms_and_document_scope():
    from src.retriever import lexical_prior

    assert lexical_prior("Who was interviewed by cellphone?") == 0
    assert lexical_prior("Which cell reports 5%?") == 1
    assert lexical_prior("Among all 12 charts, how many show a decline?") == 0.30


def test_unattached_caption_uses_adjacent_aligned_text(tmp_path):
    from src.layout import read_mineru

    page = tmp_path / "chart.png"
    Image.new("RGB", (100, 100), "red").save(page)
    layout = tmp_path / "content_list.json"
    layout.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "page_idx": 0,
                    "bbox": [600, 100, 900, 170],
                    "text": "Monthly units",
                },
                {
                    "type": "text",
                    "page_idx": 0,
                    "bbox": [100, 100, 550, 180],
                    "text": "Separate column",
                },
                {"type": "image", "page_idx": 0, "bbox": [600, 200, 850, 650], "image_caption": []},
            ]
        )
    )
    document = read_mineru("caption", {0: page}, layout)
    figure = next(node for node in document.macro if node.kind == "figure")
    assert figure.text == "Monthly units"
    assert figure.region.bbox == (60, 20, 85, 65)


def test_generation_pads_subpatch_crops_without_resampling(tmp_path):
    from src.generation import _image

    crop = Image.new("RGB", (27, 26), (12, 34, 56))
    _image(crop, tmp_path, "small.png")

    with Image.open(tmp_path / "small.png") as encoded:
        assert encoded.size == (28, 28)
        assert encoded.getpixel((0, 1)) == (12, 34, 56)
        assert encoded.getpixel((27, 0)) == (255, 255, 255)
    assert crop.size == (27, 26)
